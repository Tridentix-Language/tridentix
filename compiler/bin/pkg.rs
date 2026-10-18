//! `pkg` CLI: init, build, run, test, deps, doc.

use tridentix::{ast, interpreter, lexer, parser, typechecker};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

#[derive(Debug, Deserialize)]
struct Manifest {
    package: PackageInfo,
    #[serde(default)]
    dependencies: HashMap<String, DependencySpec>,
}

#[derive(Debug, Deserialize)]
struct PackageInfo {
    name: String,
    version: String,
    #[serde(default = "default_entry")]
    entry: String,
}

fn default_entry() -> String {
    "src/main.trix".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DependencySpec {
    /// Version requirement string; registry resolution is not implemented.
    Version(String),
    /// Local path dependency, resolved by `pkg build` / `pkg run`.
    Path { path: String },
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    let result = match cmd {
        "init" => cmd_init(),
        "build" => cmd_build(),
        "run" => cmd_run(&args[2..]),
        "test" => cmd_test(),
        "deps" => cmd_deps(),
        "doc" => cmd_doc(),
        "publish" => cmd_publish(),
        _ => {
            print_help();
            return;
        }
    };

    if let Err(e) = result {
        eprintln!("pkg: error: {}", e);
        process::exit(1);
    }
}

fn print_help() {
    println!("pkg — Tridentix package manager / build tool");
    println!();
    println!("USAGE:");
    println!("    pkg init      Scaffold a new project.toml + src/main.trix");
    println!("    pkg build     Type-check the project (entry point + path dependencies)");
    println!("    pkg run       Build, then interpret the entry point");
    println!("    pkg test      Run every .trix file in tests/, report pass/fail");
    println!("    pkg deps      Print the resolved dependency graph");
    println!("    pkg doc       Generate Markdown docs from ## doc-comments");
}

fn load_manifest() -> Result<Manifest, String> {
    let text = fs::read_to_string("project.toml")
        .map_err(|e| format!("couldn't read project.toml (are you in a project directory? try `pkg init`): {}", e))?;
    toml::from_str(&text).map_err(|e| format!("invalid project.toml: {}", e))
}

fn cmd_init() -> Result<(), String> {
    if Path::new("project.toml").exists() {
        return Err("project.toml already exists in this directory".into());
    }
    let dir_name = std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "my_project".to_string());

    fs::write(
        "project.toml",
        format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nentry = \"src/main.trix\"\n\n[dependencies]\n# example_lib = {{ path = \"../example_lib\" }}\n",
            dir_name
        ),
    )
    .map_err(|e| e.to_string())?;

    fs::create_dir_all("src").map_err(|e| e.to_string())?;
    fs::create_dir_all("tests").map_err(|e| e.to_string())?;

    fs::write(
        "src/main.trix",
        "## Entry point for this project.\nfn main():\n    print(\"Hello from Tridentix!\")\n",
    )
    .map_err(|e| e.to_string())?;

    fs::write(
        "tests/example_test.trix",
        "## A test file: `pkg test` runs every .trix file under tests/.\n## A file PASSES if it runs to completion without a runtime error;\n## use assert() to make failures explicit.\nfn main():\n    assert(1 + 1 == 2, \"basic arithmetic\")\n    print(\"example_test passed\")\n",
    )
    .map_err(|e| e.to_string())?;

    println!("Initialized new Tridentix project:");
    println!("  project.toml");
    println!("  src/main.trix");
    println!("  tests/example_test.trix");
    Ok(())
}

/// Parses + type-checks one file, returning the AST on success.
fn build_file(path: &Path) -> Result<ast::Program, String> {
    let source = fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let tokens = lexer::tokenize(&source).map_err(|e| format!("{}: lex error (line {}): {}", path.display(), e.line, e.message))?;
    let mut p = parser::Parser::new(tokens);
    let program = p.parse_program().map_err(|e| format!("{}: parse error: {}", path.display(), e.message))?;
    let errors = typechecker::check(&program);
    if !errors.is_empty() {
        let mut msg = format!("{}: {} type error(s):\n", path.display(), errors.len());
        for e in &errors {
            msg.push_str(&format!("  - {}\n", e));
        }
        return Err(msg);
    }
    Ok(program)
}

fn cmd_build() -> Result<(), String> {
    let manifest = load_manifest()?;
    println!("Building {} v{}", manifest.package.name, manifest.package.version);

    for (dep_name, spec) in &manifest.dependencies {
        if let DependencySpec::Path { path } = spec {
            let dep_manifest_path = Path::new(path).join("project.toml");
            if dep_manifest_path.exists() {
                let dep_text = fs::read_to_string(&dep_manifest_path).map_err(|e| e.to_string())?;
                let dep_manifest: Manifest = toml::from_str(&dep_text).map_err(|e| e.to_string())?;
                let dep_entry = Path::new(path).join(&dep_manifest.package.entry);
                println!("  checking dependency '{}' ({})", dep_name, dep_entry.display());
                build_file(&dep_entry)?;
            } else {
                println!("  warning: path dependency '{}' has no project.toml at {}", dep_name, path);
            }
        }
    }

    println!("  checking {}", manifest.package.entry);
    build_file(Path::new(&manifest.package.entry))?;
    println!("Build OK");
    Ok(())
}

fn cmd_run(extra_args: &[String]) -> Result<(), String> {
    let manifest = load_manifest()?;
    let entry = PathBuf::from(&manifest.package.entry);
    let program = build_file(&entry)?;

    // Resolve `import` the same way as the `tridentix run` CLI.
    let base_dir = entry.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut visited = std::collections::HashSet::new();
    let program = resolve_imports(program, &base_dir, &mut visited)?;

    std::env::set_var("TRIDENTIX_ARGS_MARKER", "1"); // no-op; program_args() reads real argv
    let _ = extra_args; // reserved for future arg-passthrough to program_args()

    let mut interp = interpreter::Interpreter::new(&program).map_err(|e| e.to_string())?;
    interp.run(&program).map_err(|e| e.to_string())?;
    Ok(())
}

fn resolve_imports(program: ast::Program, base_dir: &Path, visited: &mut std::collections::HashSet<PathBuf>) -> Result<ast::Program, String> {
    let mut resolved = Vec::new();
    for stmt in program {
        match stmt {
            ast::Stmt::Import(rel_path) => {
                let full_path = base_dir.join(&rel_path);
                let canonical = full_path.canonicalize().map_err(|e| format!("import \"{}\": {}", rel_path, e))?;
                if visited.contains(&canonical) {
                    continue;
                }
                visited.insert(canonical);
                let source = fs::read_to_string(&full_path).map_err(|e| format!("import \"{}\": {}", rel_path, e))?;
                let tokens = lexer::tokenize(&source).map_err(|e| format!("import \"{}\": lex error: {}", rel_path, e.message))?;
                let mut p = parser::Parser::new(tokens);
                let imported = p.parse_program().map_err(|e| format!("import \"{}\": {}", rel_path, e.message))?;
                let imported_base = full_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                resolved.extend(resolve_imports(imported, &imported_base, visited)?);
            }
            other => resolved.push(other),
        }
    }
    Ok(resolved)
}

/// Run every `.trix` file under `tests/`; pass = no type or runtime error.
fn cmd_test() -> Result<(), String> {
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        return Err("no tests/ directory found (try `pkg init`, or create tests/ yourself)".into());
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(tests_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("trix"))
        .collect();
    entries.sort();

    if entries.is_empty() {
        println!("no test files found in tests/");
        return Ok(());
    }

    let mut passed = 0;
    let mut failed = 0;
    for path in &entries {
        print!("test {} ... ", path.display());
        match run_test_file(path) {
            Ok(()) => {
                println!("ok");
                passed += 1;
            }
            Err(e) => {
                println!("FAILED");
                println!("  {}", e.replace('\n', "\n  "));
                failed += 1;
            }
        }
    }

    println!();
    println!("test result: {} passed, {} failed", passed, failed);
    if failed > 0 {
        return Err(format!("{} test(s) failed", failed));
    }
    Ok(())
}

fn run_test_file(path: &Path) -> Result<(), String> {
    let program = build_file(path)?;
    let mut interp = interpreter::Interpreter::new(&program).map_err(|e| e.to_string())?;
    interp.run(&program).map_err(|e| e.to_string())
}

fn cmd_deps() -> Result<(), String> {
    let manifest = load_manifest()?;
    println!("{} v{}", manifest.package.name, manifest.package.version);
    if manifest.dependencies.is_empty() {
        println!("(no dependencies)");
        return Ok(());
    }
    for (name, spec) in &manifest.dependencies {
        match spec {
            DependencySpec::Version(v) => println!("├── {} {} (registry — NOT resolvable, no registry exists yet)", name, v),
            DependencySpec::Path { path } => println!("├── {} (path: {})", name, path),
        }
    }
    Ok(())
}

/// Not implemented: there is no package registry.
fn cmd_publish() -> Result<(), String> {
    Err("pkg publish isn't implemented — there is no Tridentix package registry to publish to yet (this needs a real hosted service, not just more CLI code; see this file's `cmd_publish` doc comment)".into())
}

/// Extract `##` docs above `fn`/`struct`/`enum`/`actor` into Markdown.
fn cmd_doc() -> Result<(), String> {
    let manifest = load_manifest()?;
    let entry = PathBuf::from(&manifest.package.entry);
    let source = fs::read_to_string(&entry).map_err(|e| e.to_string())?;

    let mut out = String::new();
    out.push_str(&format!("# {} — API Documentation\n\n", manifest.package.name));

    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("fn ") || line.starts_with("struct ") || line.starts_with("enum ") || line.starts_with("actor ") {
            // Walk backward collecting contiguous `##` doc-comment lines directly above.
            let mut doc_lines = Vec::new();
            let mut j = i;
            while j > 0 && lines[j - 1].trim().starts_with("##") {
                j -= 1;
                doc_lines.insert(0, lines[j].trim().trim_start_matches('#').trim().to_string());
            }
            let signature = line.trim_end_matches(':');
            out.push_str(&format!("## `{}`\n\n", signature));
            if doc_lines.is_empty() {
                out.push_str("_(undocumented)_\n\n");
            } else {
                out.push_str(&doc_lines.join("\n"));
                out.push_str("\n\n");
            }
        }
        i += 1;
    }

    let out_path = "docs.md";
    fs::write(out_path, &out).map_err(|e| e.to_string())?;
    println!("Wrote {}", out_path);
    Ok(())
}
