//! Language Server Protocol implementation for Tridentix.

use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

struct Backend {
    client: Client,
}

const KEYWORDS: &[(&str, &str)] = &[
    ("fn", "Defines a function."),
    ("let", "Declares an immutable (by default) variable binding."),
    ("mut", "Marks a `let` binding as mutable."),
    ("if", "Conditional branch."),
    ("elif", "Else-if branch."),
    ("else", "Else branch."),
    ("loop", "`loop var in range(a, b):` — bounded iteration."),
    ("while", "Condition-checked loop."),
    ("match", "Pattern matching over literals and enum variants."),
    ("return", "Returns a value from the current function."),
    ("try", "Begins a try/catch error-handling block."),
    ("catch", "Catches any runtime error raised in the paired `try` block."),
    ("raise", "Raises a runtime error with the given value."),
    ("import", "Imports top-level definitions from another .trix file."),
    ("struct", "Defines a struct type."),
    ("enum", "Defines an enum (sum) type."),
    ("actor", "Defines an actor: an isolated, message-driven concurrent unit."),
    ("supervisor", "Defines a supervisor that restarts crashed child actors."),
    ("spawn", "Spawns a new actor, returning its handle."),
    ("send", "Sends a message to an actor's mailbox."),
    ("async", "Marks a function as async (see README for the current simplified model)."),
    ("await", "Awaits an async function's result."),
    ("not", "Logical negation."),
];

const BUILTINS: &[(&str, &str)] = &[
    ("print", "print(value) — writes value + newline to stdout."),
    ("len", "len(list|string) -> int"),
    ("map_new", "map_new() -> Map — creates a new empty HashMap."),
    ("map_set", "map_set(map, key, value) -> Map"),
    ("map_get", "map_get(map, key) -> value"),
    ("file_read", "file_read(path) -> string"),
    ("file_write", "file_write(path, content) -> bool"),
    ("http_get", "http_get(host, port, path) -> HttpResponse{status, body}"),
    ("http_serve", "http_serve(port, handler_closure, max_requests) -> int"),
    ("json_stringify", "json_stringify(value) -> string"),
    ("json_parse", "json_parse(string) -> value"),
    ("regex_match", "regex_match(pattern, text) -> bool"),
    ("assert", "assert(condition, message?) — raises if condition is false."),
];

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> LspResult<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions::default()),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "tridentix-lsp".to_string(),
                version: Some("0.1.0".to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client.log_message(MessageType::INFO, "tridentix-lsp initialized").await;
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.check_and_publish(&params.text_document.uri, &params.text_document.text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.check_and_publish(&params.text_document.uri, &change.text).await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        if let Some(text) = params.text {
            self.check_and_publish(&params.text_document.uri, &text).await;
        }
    }

    async fn hover(&self, params: HoverParams) -> LspResult<Option<Hover>> {
        let pos = params.text_document_position_params.position;
        let uri = params.text_document_position_params.text_document.uri;
        let word = match self.word_at(&uri, pos).await {
            Some(w) => w,
            None => return Ok(None),
        };
        let doc = KEYWORDS
            .iter()
            .chain(BUILTINS.iter())
            .find(|(name, _)| *name == word)
            .map(|(_, desc)| desc.to_string());
        Ok(doc.map(|d| Hover {
            contents: HoverContents::Scalar(MarkedString::String(d)),
            range: None,
        }))
    }

    async fn completion(&self, _: CompletionParams) -> LspResult<Option<CompletionResponse>> {
        let items: Vec<CompletionItem> = KEYWORDS
            .iter()
            .map(|(name, desc)| CompletionItem {
                label: name.to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                detail: Some(desc.to_string()),
                ..Default::default()
            })
            .chain(BUILTINS.iter().map(|(name, desc)| CompletionItem {
                label: name.to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(desc.to_string()),
                ..Default::default()
            }))
            .collect();
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn shutdown(&self) -> LspResult<()> {
        Ok(())
    }
}

impl Backend {
    /// Lex, parse, type-check, and publish LSP diagnostics.
    async fn check_and_publish(&self, uri: &Url, text: &str) {
        let mut diagnostics = Vec::new();

        match tridentix::lexer::tokenize(text) {
            Ok(tokens) => {
                let mut p = tridentix::parser::Parser::new(tokens);
                match p.parse_program() {
                    Ok(program) => {
                        let errors = tridentix::typechecker::check(&program);
                        for e in errors {
                            diagnostics.push(Diagnostic {
                                range: Range::new(Position::new(0, 0), Position::new(0, 1)),
                                severity: Some(DiagnosticSeverity::ERROR),
                                source: Some("tridentix-typecheck".to_string()),
                                message: e.to_string(),
                                ..Default::default()
                            });
                        }
                    }
                    Err(e) => {
                        diagnostics.push(Diagnostic {
                            range: Range::new(Position::new(0, 0), Position::new(0, 1)),
                            severity: Some(DiagnosticSeverity::ERROR),
                            source: Some("tridentix-parse".to_string()),
                            message: e.message,
                            ..Default::default()
                        });
                    }
                }
            }
            Err(e) => {
                let line = (e.line.saturating_sub(1)) as u32;
                diagnostics.push(Diagnostic {
                    range: Range::new(Position::new(line, 0), Position::new(line, 200)),
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("tridentix-lex".to_string()),
                    message: e.message,
                    ..Default::default()
                });
            }
        }

        self.client.publish_diagnostics(uri.clone(), diagnostics, None).await;
    }

    /// Hover word-at-cursor; document buffer is not tracked yet.
    async fn word_at(&self, _uri: &Url, _pos: Position) -> Option<String> {
        None
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend { client });
    Server::new(stdin, stdout, socket).serve(service).await;
}
