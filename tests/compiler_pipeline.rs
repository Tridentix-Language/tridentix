use tridentix::{lexer, parser, typechecker};

fn parse_file(path: impl AsRef<std::path::Path>) {
    let path = path.as_ref();
    let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    parse_ok(&src);
}

fn parse_ok(src: &str) {
    let tokens = lexer::tokenize(src).expect("lex");
    let mut p = parser::Parser::new(tokens);
    let program = p.parse_program().expect("parse");
    let errors = typechecker::check(&program);
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn lex_parse_check_hello() {
    parse_ok(
        r#"
fn classify(n: int) -> string:
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"

fn main():
    let greeting: string = "Hello, Tridentix!"
    print(greeting)
    let label = classify(1)
    print(label)
"#,
    );
}

#[test]
fn borrow_checker_rejects_move_while_borrowed() {
    let src = r#"
fn main():
    let a = tensor([1.0, 2.0])
    let r = borrow(a)
    let b = a
    print(r)
    print(b)
"#;
    let tokens = lexer::tokenize(src).unwrap();
    let mut p = parser::Parser::new(tokens);
    let program = p.parse_program().unwrap();
    let errors = typechecker::check(&program);
    assert!(!errors.is_empty());
    assert!(errors.iter().any(|e| e.message.contains("borrowed") || e.message.contains("moved")));
}

#[test]
fn actor_supervisor_parses() {
    parse_ok(
        r#"
actor Worker:
    on receive(msg):
        print(msg)

fn main():
    supervisor Root:
        strategy = one_for_one
        max_restarts = 3
        time_window = 60
        children:
            spawn Worker() as w1
    send(w1, 1)
"#,
    );
}

#[test]
fn broker_service_example_typechecks() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    parse_file(root.join("examples/broker_service/main.trix"));
}

#[test]
fn actor_supervisor_example_typechecks() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    parse_file(root.join("examples/actor_supervisor/main.trix"));
}
