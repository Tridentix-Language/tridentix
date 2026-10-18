//! Indentation-aware Tridentix lexer (Indent/Dedent).

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    Fn,
    Let,
    Mut,
    Const,
    If,
    Elif,
    Else,
    Loop,
    While,
    In,
    Return,
    True,
    False,
    Actor,
    On,
    Spawn,
    Send,
    Supervisor,
    As,
    Not,
    Try,
    Catch,
    Raise,
    Import,
    Match,
    Struct,
    Enum,
    Async,
    Await,
    Plus,
    Minus,
    Star,
    Slash,
    Assign,
    EqEq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Dot,
    Arrow,
    FatArrow,
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Debug)]
pub struct LexError {
    pub message: String,
    pub line: usize,
}

pub fn tokenize(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let mut indent_stack: Vec<usize> = vec![0];

    for (line_no, raw_line) in source.lines().enumerate() {
        let line_no_comment = match raw_line.find('#') {
            Some(idx) => &raw_line[..idx],
            None => raw_line,
        };

        let trimmed = line_no_comment.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }

        let indent_width = trimmed.len() - trimmed.trim_start().len();
        let code = trimmed.trim_start();

        let current_indent = *indent_stack.last().unwrap();
        if indent_width > current_indent {
            indent_stack.push(indent_width);
            tokens.push(Token::Indent);
        } else {
            while indent_width < *indent_stack.last().unwrap() {
                indent_stack.pop();
                tokens.push(Token::Dedent);
            }
            if indent_width != *indent_stack.last().unwrap() {
                return Err(LexError {
                    message: format!("inconsistent indentation ({} spaces)", indent_width),
                    line: line_no + 1,
                });
            }
        }

        tokenize_line(code, line_no + 1, &mut tokens)?;
        tokens.push(Token::Newline);
    }

    while indent_stack.len() > 1 {
        indent_stack.pop();
        tokens.push(Token::Dedent);
    }
    tokens.push(Token::Eof);
    Ok(tokens)
}

fn tokenize_line(code: &str, line_no: usize, tokens: &mut Vec<Token>) -> Result<(), LexError> {
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;
    let n = chars.len();

    while i < n {
        let c = chars[i];

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c == '"' {
            let mut s = String::new();
            i += 1;
            while i < n && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < n {
                    let escaped = match chars[i + 1] {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '"' => '"',
                        '0' => '\0',
                        other => other,
                    };
                    s.push(escaped);
                    i += 2;
                } else {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            if i >= n {
                return Err(LexError {
                    message: "unterminated string literal".into(),
                    line: line_no,
                });
            }
            i += 1;
            tokens.push(Token::Str(s));
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            let mut is_float = false;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < n && chars[i] == '.' && i + 1 < n && chars[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let text: String = chars[start..i].iter().collect();
            if is_float {
                tokens.push(Token::Float(text.parse().map_err(|_| LexError {
                    message: format!("invalid float literal '{}'", text),
                    line: line_no,
                })?));
            } else {
                tokens.push(Token::Int(text.parse().map_err(|_| LexError {
                    message: format!("invalid int literal '{}'", text),
                    line: line_no,
                })?));
            }
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            tokens.push(match text.as_str() {
                "fn" => Token::Fn,
                "let" => Token::Let,
                "mut" => Token::Mut,
                "const" => Token::Const,
                "if" => Token::If,
                "elif" => Token::Elif,
                "else" => Token::Else,
                "loop" => Token::Loop,
                "while" => Token::While,
                "in" => Token::In,
                "return" => Token::Return,
                "true" => Token::True,
                "false" => Token::False,
                "actor" => Token::Actor,
                "on" => Token::On,
                "spawn" => Token::Spawn,
                "send" => Token::Send,
                "supervisor" => Token::Supervisor,
                "as" => Token::As,
                "not" => Token::Not,
                "try" => Token::Try,
                "catch" => Token::Catch,
                "raise" => Token::Raise,
                "import" => Token::Import,
                "match" => Token::Match,
                "struct" => Token::Struct,
                "enum" => Token::Enum,
                "async" => Token::Async,
                "await" => Token::Await,
                _ => Token::Ident(text),
            });
            continue;
        }

        let two_char: String = chars[i..(i + 2).min(n)].iter().collect();
        match two_char.as_str() {
            "->" => {
                tokens.push(Token::Arrow);
                i += 2;
                continue;
            }
            "=>" => {
                tokens.push(Token::FatArrow);
                i += 2;
                continue;
            }
            "==" => {
                tokens.push(Token::EqEq);
                i += 2;
                continue;
            }
            "!=" => {
                tokens.push(Token::NotEq);
                i += 2;
                continue;
            }
            "<=" => {
                tokens.push(Token::LtEq);
                i += 2;
                continue;
            }
            ">=" => {
                tokens.push(Token::GtEq);
                i += 2;
                continue;
            }
            _ => {}
        }

        let tok = match c {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '=' => Token::Assign,
            '<' => Token::Lt,
            '>' => Token::Gt,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '.' => Token::Dot,
            ':' => Token::Colon,
            ',' => Token::Comma,
            other => {
                return Err(LexError {
                    message: format!("unexpected character '{}'", other),
                    line: line_no,
                })
            }
        };
        tokens.push(tok);
        i += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_and_indent() {
        let src = "fn main():\n    let x = 1\n";
        let toks = tokenize(src).unwrap();
        assert!(matches!(toks[0], Token::Fn));
        assert!(toks.iter().any(|t| matches!(t, Token::Indent)));
        assert!(toks.iter().any(|t| matches!(t, Token::Dedent)));
    }
}
