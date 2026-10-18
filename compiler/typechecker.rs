//! Static type checker plus tensor ownership / borrow exclusivity.

use crate::ast::{Expr, Program, Stmt};
use crate::borrow_checker::BorrowChecker;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

const BUILTINS: &[(&str, usize)] = &[
    ("print", usize::MAX), // variadic
    ("tensor", usize::MAX),
    ("tensor.zeros", 1),
    ("tensor.sum", 1),
    ("len", 1),
    ("str_upper", 1),
    ("str_lower", 1),
    ("str_split", 2),
    ("str_trim", 1),
    ("str_contains", 2),
    ("int_to_str", 1),
    ("str_to_int", 1),
    ("list_push", 2),
    ("list_get", 2),
    ("range_list", 2),
    ("file_read", 1),
    ("file_write", 2),
    ("file_append", 2),
    ("file_exists", 1),
    ("json_stringify", 1),
    ("json_parse", 1),
    ("regex_match", 2),
    ("regex_find", 2),
    ("regex_replace", 3),
    ("net_tcp_send", 3),
    ("map_new", 0),
    ("map_set", 3),
    ("map_get", 2),
    ("map_has", 2),
    ("map_delete", 2),
    ("map_keys", 1),
    ("map_len", 1),
    ("env_get", 1),
    ("env_set", 2),
    ("program_args", 0),
    ("process_exit", 1),
    ("process_run", usize::MAX), // (cmd) or (cmd, args_list)
    ("udp_send", 3),
    ("http_get", 3),
    ("http_post", 4),
    ("http_serve", 3),
    ("net_tcp_send_to_actor", 4),
    ("assert", usize::MAX), // (cond) or (cond, message)
];

struct Checker {
    functions: HashMap<String, usize>, // name -> arity
    actors: HashSet<String>,
    structs: HashMap<String, HashSet<String>>, // struct name -> field names
    enums: HashMap<String, HashSet<String>>,   // enum name -> variant names
    /// Enclosing function's declared return type, if any.
    current_fn_return_type: Option<String>,
    scopes: Vec<HashMap<String, Option<String>>>,
    borrows: BorrowChecker,
    errors: Vec<TypeError>,
}

pub fn check(program: &Program) -> Vec<TypeError> {
    let mut functions = HashMap::new();
    let mut actors = HashSet::new();
    let mut structs: HashMap<String, HashSet<String>> = HashMap::new(); // struct name -> field names
    let mut enums: HashMap<String, HashSet<String>> = HashMap::new(); // enum name -> variant names
    enums.insert("Option".to_string(), ["Some", "None"].iter().map(|s| s.to_string()).collect());
    enums.insert("Result".to_string(), ["Ok", "Err"].iter().map(|s| s.to_string()).collect());

    for stmt in program {
        match stmt {
            Stmt::FnDef { name, params, .. } => {
                functions.insert(name.clone(), params.len());
            }
            Stmt::Actor { name, .. } => {
                actors.insert(name.clone());
            }
            Stmt::StructDef { name, fields } => {
                structs.insert(name.clone(), fields.iter().map(|f| f.name.clone()).collect());
            }
            Stmt::EnumDef { name, variants } => {
                enums.insert(name.clone(), variants.iter().map(|v| v.name.clone()).collect());
            }
            _ => {}
        }
    }

    let mut checker = Checker {
        functions,
        actors,
        structs,
        enums,
        current_fn_return_type: None,
        scopes: vec![HashMap::new()],
        borrows: BorrowChecker::new(),
        errors: Vec::new(),
    };

    for stmt in program {
        checker.check_stmt(stmt);
    }

    checker.errors
}

impl Checker {
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.borrows.push_scope();
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.borrows.pop_scope();
    }
    fn declare(&mut self, name: &str, ty: Option<String>) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), ty);
    }
    fn is_declared(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|s| s.contains_key(name))
    }
    fn err(&mut self, message: impl Into<String>) {
        self.errors.push(TypeError { message: message.into() });
    }

    fn declare_tensor_owned(&mut self, name: &str) {
        self.borrows.declare_owned(name);
    }
    fn is_tensor(&self, name: &str) -> bool {
        self.borrows.is_tensor(name)
    }
    fn check_tensor_move(&mut self, expr: &Expr, context: &str) {
        if let Some(msg) = self.borrows.check_tensor_move(expr, context) {
            self.err(msg);
        }
    }
    fn is_tensor_ident(&self, expr: &Expr) -> bool {
        self.borrows.is_tensor_ident(expr)
    }
    fn check_borrow_conflict(&mut self, source: &str, want_mut: bool) -> bool {
        if let Some(msg) = self.borrows.check_borrow_conflict(source, want_mut) {
            self.err(msg);
            false
        } else {
            true
        }
    }
    fn register_persistent_borrow(&mut self, source: &str, is_mut: bool) {
        self.borrows.register_persistent_borrow(source, is_mut);
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::FnDef { params, ret_type, body, .. } => {
                self.push_scope();
                for p in params {
                    self.declare(&p.name, p.type_ann.clone());
                    if p.type_ann.as_deref() == Some("tensor") {
                        self.declare_tensor_owned(&p.name);
                    }
                }
                let outer_ret_type = self.current_fn_return_type.take();
                self.current_fn_return_type = ret_type.clone();
                for s in body {
                    self.check_stmt(s);
                }
                self.current_fn_return_type = outer_ret_type;
                self.pop_scope();
            }
            Stmt::Actor { state_vars, param_name, param_type, body, .. } => {
                self.push_scope();
                for (sname, stype, sinit) in state_vars {
                    self.check_expr(sinit);
                    self.declare(sname, stype.clone());
                }
                self.declare(param_name, param_type.clone());
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            Stmt::Let { name, type_ann, value, .. } => {
                // Tensor-constructing calls declare a fresh Owned tensor.
                let is_tensor_ctor = matches!(
                    value,
                    Expr::Call(fname, _) if fname == "tensor" || fname == "tensor.zeros"
                );
                // Persistent borrow: tracked until the binding's scope ends.
                let borrow_call = match value {
                    Expr::Call(fname, args) if (fname == "borrow" || fname == "borrow_mut") && args.len() == 1 => {
                        if let Expr::Ident(src) = &args[0] {
                            Some((src.clone(), fname == "borrow_mut"))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if is_tensor_ctor {
                    self.check_expr(value); // still validate args (e.g. undefined vars in shape list)
                    self.declare_tensor_owned(name);
                } else if let Some((src, is_mut)) = borrow_call {
                    if self.is_tensor(&src) {
                        if self.check_borrow_conflict(&src, is_mut) {
                            self.register_persistent_borrow(&src, is_mut);
                        }
                    } else {
                        self.err(format!("borrow()/borrow_mut() target '{}' is not a tensor", src));
                    }
                } else if let Expr::Ident(src) = value {
                    // Tensor let-binding is a move, not a copy.
                    if self.is_tensor(src) {
                        self.check_tensor_move(value, "let-binding value");
                        self.declare_tensor_owned(name);
                    } else {
                        self.check_expr(value);
                    }
                } else {
                    self.check_expr(value);
                }
                if let Some(declared) = type_ann {
                    if let Some(inferred) = literal_type(value) {
                        if &inferred != declared {
                            self.err(format!(
                                "type mismatch: 'let {}: {}' but value is {}",
                                name, declared, inferred
                            ));
                        }
                    }
                }
                self.declare(name, type_ann.clone());
            }
            Stmt::Assign { name, value } => {
                if !self.is_declared(name) {
                    self.err(format!("assignment to undeclared variable '{}'", name));
                }
                self.check_expr(value);
            }
            Stmt::IndexAssign { base, index, value } => {
                self.check_expr(base);
                self.check_expr(index);
                self.check_expr(value);
            }
            Stmt::FieldAssign { base, value, .. } => {
                self.check_expr(base);
                self.check_expr(value);
            }
            Stmt::Send { target, message } => {
                self.check_expr(target);
                // Sending a tensor moves ownership to the receiving actor.
                if self.is_tensor_ident(message) {
                    self.check_tensor_move(message, "send() message");
                } else {
                    self.check_expr(message);
                }
            }
            Stmt::Raise(e) => {
                self.check_expr(e);
            }
            Stmt::Try { body, catch_var, catch_body } => {
                self.push_scope();
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
                self.push_scope();
                self.declare(catch_var, None);
                for s in catch_body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            Stmt::Import(_) => {
                // Resolved/spliced by main.rs before type-checking runs.
            }
            Stmt::Supervisor { children, .. } => {
                for c in children {
                    if !self.actors.contains(&c.actor_name) {
                        self.err(format!(
                            "supervisor child references undefined actor '{}'",
                            c.actor_name
                        ));
                    }
                    self.declare(&c.bind_name, Some("actor_handle".to_string()));
                }
            }
            Stmt::StructDef { .. } | Stmt::EnumDef { .. } => {
                // Already registered in the pre-scan pass (see `check()`).
            }
            Stmt::ExprStmt(e) => self.check_expr(e),
            Stmt::If { cond, then_block, elif_blocks, else_block } => {
                self.check_expr(cond);
                self.push_scope();
                for s in then_block {
                    self.check_stmt(s);
                }
                self.pop_scope();
                for (c, block) in elif_blocks {
                    self.check_expr(c);
                    self.push_scope();
                    for s in block {
                        self.check_stmt(s);
                    }
                    self.pop_scope();
                }
                if let Some(block) = else_block {
                    self.push_scope();
                    for s in block {
                        self.check_stmt(s);
                    }
                    self.pop_scope();
                }
            }
            Stmt::Loop { var, start, end, body } => {
                self.check_expr(start);
                self.check_expr(end);
                self.push_scope();
                self.declare(var, Some("int".to_string()));
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            Stmt::While { cond, body } => {
                self.check_expr(cond);
                self.push_scope();
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            Stmt::Return(val) => {
                if let Some(e) = val {
                    if self.is_tensor_ident(e) {
                        self.check_tensor_move(e, "return value");
                    } else {
                        self.check_expr(e);
                    }
                    if let Some(declared) = self.current_fn_return_type.clone() {
                        if let Some(inferred) = literal_type(e) {
                            if inferred != declared {
                                self.err(format!(
                                    "return type mismatch: function declared to return '{}' but this `return` gives {}",
                                    declared, inferred
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::IntLit(_) | Expr::FloatLit(_) | Expr::StrLit(_) | Expr::BoolLit(_) => {}
            Expr::ListLit(items) => {
                for e in items {
                    self.check_expr(e);
                }
            }
            Expr::Ident(name) => {
                if !self.is_declared(name) {
                    self.err(format!("undefined variable '{}'", name));
                }
            }
            Expr::Binary(lhs, _, rhs) => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            Expr::Unary(_, e) => {
                self.check_expr(e);
            }
            Expr::StructLit(type_name, fields) => {
                if let Some(known_fields) = self.structs.get(type_name).cloned() {
                    for (fname, _) in fields {
                        if !known_fields.contains(fname) {
                            self.err(format!("struct '{}' has no field '{}'", type_name, fname));
                        }
                    }
                    for kf in &known_fields {
                        if !fields.iter().any(|(fname, _)| fname == kf) {
                            self.err(format!("struct literal for '{}' is missing field '{}'", type_name, kf));
                        }
                    }
                } else {
                    self.err(format!("use of undefined struct '{}'", type_name));
                }
                for (_, fvalue) in fields {
                    self.check_expr(fvalue);
                }
            }
            Expr::FieldAccess(base, _field) => {
                // Field names are not checked without inferring `base`'s type.
                self.check_expr(base);
            }
            Expr::Index(base, index) => {
                self.check_expr(base);
                self.check_expr(index);
            }
            Expr::EnumLit(type_name, variant, args) => {
                match self.enums.get(type_name) {
                    Some(known_variants) => {
                        if !known_variants.contains(variant) {
                            self.err(format!("enum '{}' has no variant '{}'", type_name, variant));
                        }
                    }
                    None => self.err(format!("use of undefined enum '{}'", type_name)),
                }
                for a in args {
                    self.check_expr(a);
                }
            }
            Expr::Closure(params, body) => {
                self.push_scope();
                for p in params {
                    self.declare(&p.name, p.type_ann.clone());
                }
                self.check_expr(body);
                self.pop_scope();
            }
            Expr::Await(e) => {
                self.check_expr(e);
            }
            Expr::EnumIsVariant(subject, type_name, variant) => {
                self.check_expr(subject);
                match self.enums.get(type_name) {
                    Some(known_variants) => {
                        if !known_variants.contains(variant) {
                            self.err(format!("enum '{}' has no variant '{}'", type_name, variant));
                        }
                    }
                    None => self.err(format!("use of undefined enum '{}' in match pattern", type_name)),
                }
            }
            Expr::EnumPayload(subject, type_name, variant, _idx) => {
                self.check_expr(subject);
                match self.enums.get(type_name) {
                    Some(known_variants) => {
                        if !known_variants.contains(variant) {
                            self.err(format!("enum '{}' has no variant '{}'", type_name, variant));
                        }
                        // Payload arity against declared variants is not checked yet.
                    }
                    None => self.err(format!("use of undefined enum '{}' in match pattern", type_name)),
                }
            }
            Expr::Spawn(name, args) => {
                if !self.actors.contains(name) {
                    self.err(format!("spawn of undefined actor '{}'", name));
                }
                for a in args {
                    self.check_expr(a);
                }
            }
            Expr::Call(name, args) => {
                if name == "borrow" || name == "borrow_mut" {
                    // Temporary borrow: exclusivity only; not registered persistently.
                    if args.len() == 1 {
                        if let Expr::Ident(src) = &args[0] {
                            if self.is_tensor(src) {
                                self.check_borrow_conflict(src, name == "borrow_mut");
                            } else {
                                self.err(format!("borrow()/borrow_mut() target '{}' is not a tensor", src));
                            }
                        } else {
                            self.check_expr(&args[0]);
                        }
                    }
                    return;
                }
                let is_borrow_fn = name == "print";
                for a in args {
                    if !is_borrow_fn && self.is_tensor_ident(a) {
                        // Passing a tensor by bare identifier moves it.
                        self.check_tensor_move(a, "function argument");
                    } else {
                        self.check_expr(a);
                    }
                }
                if is_borrow_fn {
                    return;
                }
                if let Some((_, arity)) = BUILTINS.iter().find(|(n, _)| n == name) {
                    if *arity != usize::MAX && *arity != args.len() {
                        self.err(format!(
                            "'{}' expects {} argument(s), got {}",
                            name, arity, args.len()
                        ));
                    }
                    return;
                }
                // Capitalized name = message-tag constructor, always OK.
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    return;
                }
                match self.functions.get(name) {
                    Some(arity) if *arity != args.len() => {
                        self.err(format!(
                            "'{}' expects {} argument(s), got {}",
                            name, arity, args.len()
                        ));
                    }
                    Some(_) => {}
                    None => {
                        // May be a local closure; confirm the name exists, skip arity.
                        if !self.is_declared(name) {
                            self.err(format!("call to undefined function '{}'", name));
                        }
                    }
                }
            }
        }
    }
}

fn literal_type(expr: &Expr) -> Option<String> {
    match expr {
        Expr::IntLit(_) => Some("int".to_string()),
        Expr::FloatLit(_) => Some("float".to_string()),
        Expr::StrLit(_) => Some("string".to_string()),
        Expr::BoolLit(_) => Some("bool".to_string()),
        _ => None, // calls/idents/binary exprs: not statically known here
    }
}
