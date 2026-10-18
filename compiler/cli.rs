//! Tridentix CLI implementation: `parse | check | run | build | aot`.

use crate::{ast, codegen, interpreter, lexer, parser, typechecker};
use ast::{Program, Stmt};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn resolve_imports(program: Program, base_dir: &Path, visited: &mut HashSet<PathBuf>) -> Result<Program, String> {
    let mut resolved = Vec::new();
    for stmt in program {
        match stmt {
            Stmt::Import(rel_path) => {
                let full_path = base_dir.join(&rel_path);
                let canonical = full_path
                    .canonicalize()
                    .map_err(|e| format!("import \"{}\": couldn't resolve path: {}", rel_path, e))?;
                if visited.contains(&canonical) {
                    continue; // already imported elsewhere in the tree — skip silently
                }
                visited.insert(canonical.clone());

                let source = fs::read_to_string(&full_path)
                    .map_err(|e| format!("import \"{}\": couldn't read file: {}", rel_path, e))?;
                let tokens = lexer::tokenize(&source)
                    .map_err(|e| format!("import \"{}\": lex error (line {}): {}", rel_path, e.line, e.message))?;
                let mut p = parser::Parser::new(tokens);
                let imported_program = p
                    .parse_program()
                    .map_err(|e| format!("import \"{}\": parse error: {}", rel_path, e.message))?;

                let imported_base_dir = full_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                let nested_resolved = resolve_imports(imported_program, &imported_base_dir, visited)?;
                resolved.extend(nested_resolved);
            }
            other => resolved.push(other),
        }
    }
    Ok(resolved)
}

pub fn run() {
    let args: Vec<String> = env::args().collect();

    let (mode, path) = match args.len() {
        1 => ("run".to_string(), "examples/hello.trix".to_string()),
        2 => ("run".to_string(), args[1].clone()),
        _ => (args[1].clone(), args[2].clone()),
    };

    let source = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: couldn't read '{}': {}", path, e);
            process::exit(1);
        }
    };

    let tokens = match lexer::tokenize(&source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lex error (line {}): {}", e.line, e.message);
            process::exit(1);
        }
    };

    let mut p = parser::Parser::new(tokens);
    let program = match p.parse_program() {
        Ok(prog) => prog,
        Err(e) => {
            eprintln!("Parse error: {}", e.message);
            process::exit(1);
        }
    };

    let base_dir = Path::new(&path).parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut visited = HashSet::new();
    let program = match resolve_imports(program, &base_dir, &mut visited) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Import resolution error: {}", e);
            process::exit(1);
        }
    };

    match mode.as_str() {
        "parse" => {
            println!("--- AST ({} top-level statements) ---", program.len());
            ast::print_program(&program);
        }
        "check" => {
            let errors = typechecker::check(&program);
            if errors.is_empty() {
                println!("No type errors found.");
            } else {
                println!("{} type error(s):", errors.len());
                for e in &errors {
                    println!("  - {}", e);
                }
                process::exit(1);
            }
        }
        "run" => {
            let errors = typechecker::check(&program);
            if !errors.is_empty() {
                eprintln!("Refusing to run: {} type error(s) found:", errors.len());
                for e in &errors {
                    eprintln!("  - {}", e);
                }
                process::exit(1);
            }
            let mut interp = match interpreter::Interpreter::new(&program) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Interpreter init error: {}", e);
                    process::exit(1);
                }
            };
            if let Err(e) = interp.run(&program) {
                eprintln!("Runtime error: {}", e);
                process::exit(1);
            }
        }
        "build" => {
            if let Err(e) = codegen::compile_and_run(&program) {
                eprintln!("Codegen error: {}", e);
                process::exit(1);
            }
        }
        "aot" => {
            let output_path = args.get(3).cloned().unwrap_or_else(|| "a.out".to_string());
            let opt_level = args.get(4).cloned().unwrap_or_else(|| "O2".to_string());
            let opt_passes = format!("default<{}>", opt_level);
            let target = codegen::CompileTarget::AheadOfTime { output_path, opt_passes };
            if let Err(e) = codegen::compile_and_execute(&program, target) {
                eprintln!("AOT compilation error: {}", e);
                process::exit(1);
            }
        }
        other => {
            eprintln!("unknown mode '{}'. Use: parse | check | run | build | aot", other);
            process::exit(1);
        }
    }
}
