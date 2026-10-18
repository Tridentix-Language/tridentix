//! Tree-walking interpreter and in-process actor runtime.

use crate::ast::{BinOp, Expr, Program, RestartStrategy, Stmt};
use crate::tridentix_rt_object;
use tridentix_actor_rt::mailbox as tridentix_rt_mailbox;
use tridentix_actor_rt::MailboxTable;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<Value>),
    Tensor { data: Vec<f64>, shape: Vec<usize> },
    Message { tag: String, args: Vec<Value> },
    ActorHandle(usize),
    Struct(String, std::sync::Arc<std::sync::Mutex<HashMap<String, Value>>>),
    Map(std::sync::Arc<std::sync::Mutex<HashMap<String, Value>>>),
    /// Cycle back-edge: does not keep the target cell alive.
    WeakStruct(String, std::sync::Weak<std::sync::Mutex<HashMap<String, Value>>>),
    WeakMap(std::sync::Weak<std::sync::Mutex<HashMap<String, Value>>>),
    /// Enum instance: type name, variant name, positional payload values.
    Enum(String, String, Vec<Value>),
    Closure(std::sync::Arc<ClosureData>),
    Future(Box<Value>),
    Unit,
}

#[derive(Debug)]
pub struct ClosureData {
    pub params: Vec<String>,
    pub body: Expr,
    pub captured_env: Vec<HashMap<String, Value>>,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        tridentix_rt_object::display_value(self, f)
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    UndefinedActor(String),
    TypeError(String),
    ArityMismatch { name: String, expected: usize, got: usize },
    /// User `raise <expr>`; `catch` binds the raised Value.
    UserRaised(Value),
    /// OS refused to create an actor/supervisor thread (thread cap, memory).
    ThreadSpawnFailed(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::UndefinedVariable(n) => write!(f, "undefined variable '{}'", n),
            RuntimeError::UndefinedFunction(n) => write!(f, "undefined function '{}'", n),
            RuntimeError::UndefinedActor(n) => write!(f, "undefined actor '{}'", n),
            RuntimeError::TypeError(msg) => write!(f, "type error: {}", msg),
            RuntimeError::ArityMismatch { name, expected, got } => write!(
                f,
                "'{}' expects {} argument(s), got {}",
                name, expected, got
            ),
            RuntimeError::UserRaised(v) => write!(f, "raised: {}", v),
            RuntimeError::ThreadSpawnFailed(msg) => write!(f, "failed to spawn thread: {}", msg),
        }
    }
}

type RResult<T> = Result<T, RuntimeError>;

#[derive(Clone)]
struct FnDef {
    params: Vec<String>,
    body: Vec<Stmt>,
    is_async: bool,
}

#[derive(Clone)]
struct ActorDef {
    state_vars: Vec<(String, Expr)>, // (field name, init expr) — type_ann dropped, not needed at runtime
    param_name: String,
    body: Vec<Stmt>,
}

/// Immutable program definitions shared across actor threads.
struct Globals {
    functions: HashMap<String, FnDef>,
    actors: HashMap<String, ActorDef>,
    /// Enum variants in declaration order (stable `enum_tag` index).
    enums: HashMap<String, Vec<String>>,
}

/// Shared runtime services needed to spawn actors and route messages.
struct RuntimeShared {
    globals: Arc<Globals>,
    mailboxes: MailboxTable<Value>,
    next_actor_id: AtomicUsize,
    handles: Mutex<Vec<thread::JoinHandle<()>>>,
    /// Supervisor coordinators notified on shutdown.
    supervisor_shutdown_senders: Mutex<Vec<Sender<SupervisorEvent>>>,
}

/// Control-flow signal used internally to unwind out of loops/blocks on `return`.
enum Flow {
    Normal,
    Return(Value),
}

/// Events a supervised worker reports to its coordinator.
enum SupervisorEvent {
    Crashed(usize),       // child index crashed (unhandled RuntimeError in a handler)
    ShutdownAck(usize),   // child index acknowledged a graceful __shutdown__
    GlobalShutdown,       // program is exiting; coordinator should wind down all children
}

pub struct Interpreter {
    runtime: Arc<RuntimeShared>,
    scopes: Vec<HashMap<String, Value>>,
    /// Recursion-depth guard (one OS stack frame per Tridentix call).
    call_depth: usize,
}

/// Max interpreter call depth (well below typical OS thread stacks).
const MAX_CALL_DEPTH: usize = 300;

impl Interpreter {
    pub fn new(program: &Program) -> RResult<Self> {
        let mut functions = HashMap::new();
        let mut actors = HashMap::new();
        let mut enums = HashMap::new();
        // Built-in Option/Result variants.
        enums.insert("Option".to_string(), vec!["Some".to_string(), "None".to_string()]);
        enums.insert("Result".to_string(), vec!["Ok".to_string(), "Err".to_string()]);

        for stmt in program {
            match stmt {
                Stmt::FnDef { name, params, body, is_async, .. } => {
                    functions.insert(
                        name.clone(),
                        FnDef {
                            params: params.iter().map(|p| p.name.clone()).collect(),
                            body: body.clone(),
                            is_async: *is_async,
                        },
                    );
                }
                Stmt::Actor { name, state_vars, param_name, body, .. } => {
                    actors.insert(
                        name.clone(),
                        ActorDef {
                            state_vars: state_vars.iter().map(|(n, _, e)| (n.clone(), e.clone())).collect(),
                            param_name: param_name.clone(),
                            body: body.clone(),
                        },
                    );
                }
                Stmt::EnumDef { name, variants } => {
                    enums.insert(name.clone(), variants.iter().map(|v| v.name.clone()).collect());
                }
                _ => {}
            }
        }

        let runtime = Arc::new(RuntimeShared {
            globals: Arc::new(Globals { functions, actors, enums }),
            mailboxes: MailboxTable::new(),
            next_actor_id: AtomicUsize::new(0),
            handles: Mutex::new(Vec::new()),
            supervisor_shutdown_senders: Mutex::new(Vec::new()),
        });

        Ok(Interpreter {
            runtime,
            scopes: vec![HashMap::new()],
            call_depth: 0,
        })
    }

    /// Run `main` (if any), then join spawned actor threads.
    pub fn run(&mut self, program: &Program) -> RResult<()> {
        for stmt in program {
            if let Stmt::FnDef { name, .. } = stmt {
                if name == "main" {
                    self.call_function("main", vec![])?;
                }
            }
        }
        self.shutdown_actors();
        Ok(())
    }

    fn shutdown_actors(&mut self) {
        self.runtime.mailboxes.broadcast(|| Value::Message {
            tag: "__shutdown__".into(),
            args: vec![],
        });

        // Notify coordinators directly so mid-restart children still shut down.
        let sup_senders = self.runtime.supervisor_shutdown_senders.lock().unwrap();
        for sender in sup_senders.iter() {
            let _ = sender.send(SupervisorEvent::GlobalShutdown);
        }
        drop(sup_senders);

        let mut handles = self.runtime.handles.lock().unwrap();
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }

    // ---- Scope helpers ----

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    fn define(&mut self, name: &str, value: Value) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), value);
    }
    fn assign(&mut self, name: &str, value: Value) -> RResult<()> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return Ok(());
            }
        }
        Err(RuntimeError::UndefinedVariable(name.to_string()))
    }
    fn lookup(&self, name: &str) -> RResult<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Ok(v.clone());
            }
        }
        Err(RuntimeError::UndefinedVariable(name.to_string()))
    }

    // ---- Statement execution ----

    fn exec_block(&mut self, block: &[Stmt]) -> RResult<Flow> {
        for stmt in block {
            match self.exec_stmt(stmt)? {
                Flow::Normal => {}
                flow @ Flow::Return(_) => return Ok(flow),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> RResult<Flow> {
        match stmt {
            Stmt::FnDef { .. } | Stmt::Actor { .. } | Stmt::StructDef { .. } | Stmt::EnumDef { .. } => {
                Ok(Flow::Normal) // already registered globally / don't need runtime state
            }
            Stmt::Supervisor { name, strategy, max_restarts, time_window_secs, children } => {
                let bindings = self.spawn_supervisor(name, strategy.clone(), *max_restarts, *time_window_secs, children)?;
                for (bind_name, handle) in bindings {
                    self.define(&bind_name, handle);
                }
                Ok(Flow::Normal)
            }
            Stmt::Let { name, value, .. } => {
                let v = self.eval(value)?;
                self.define(name, v);
                Ok(Flow::Normal)
            }
            Stmt::Assign { name, value } => {
                let v = self.eval(value)?;
                self.assign(name, v)?;
                Ok(Flow::Normal)
            }
            Stmt::IndexAssign { base, index, value } => {
                let idx = self.eval(index)?.as_int()?;
                let new_val = self.eval(value)?;
                let base_name = match base {
                    Expr::Ident(n) => n.clone(),
                    _ => {
                        return Err(RuntimeError::TypeError(
                            "index-assignment target must be a plain variable, e.g. `list[i] = x` (not a computed expression)".into(),
                        ))
                    }
                };
                let mut current = self.lookup(&base_name)?;
                match &mut current {
                    Value::List(items) => {
                        if idx < 0 || idx as usize >= items.len() {
                            return Err(RuntimeError::TypeError(format!(
                                "index {} out of bounds (length {}) assigning to '{}'",
                                idx, items.len(), base_name
                            )));
                        }
                        items[idx as usize] = new_val;
                    }
                    other => {
                        return Err(RuntimeError::TypeError(format!("cannot index-assign into {}", other)))
                    }
                }
                self.assign(&base_name, current)?;
                Ok(Flow::Normal)
            }
            Stmt::FieldAssign { base, field, value } => {
                let base_val = self.eval(base)?;
                let new_val = self.eval(value)?;
                let (type_name, cell) = match base_val {
                    Value::Struct(type_name, ref c) => (type_name, std::sync::Arc::clone(c)),
                    Value::WeakStruct(type_name, ref w) => {
                        let cell = w.upgrade().ok_or_else(|| {
                            RuntimeError::TypeError("struct cell already dropped".into())
                        })?;
                        (type_name, cell)
                    }
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "cannot assign field '.{}' on non-struct value {}",
                            field, other
                        )))
                    }
                };
                {
                    let guard = cell.lock().unwrap();
                    if !guard.contains_key(field) {
                        return Err(RuntimeError::TypeError(format!(
                            "struct '{}' has no field '{}'",
                            type_name, field
                        )));
                    }
                }
                tridentix_rt_object::install_field(&cell, field.clone(), new_val);
                Ok(Flow::Normal)
            }
            Stmt::ExprStmt(e) => {
                self.eval(e)?;
                Ok(Flow::Normal)
            }
            Stmt::Send { target, message } => {
                let target_val = self.eval(target)?;
                let msg_val = self.eval(message)?;
                let id = match target_val {
                    Value::ActorHandle(id) => id,
                    other => {
                        return Err(RuntimeError::TypeError(format!(
                            "send() target must be an actor handle, got {}",
                            other
                        )))
                    }
                };
                self.runtime.mailboxes.send_blocking(id, msg_val);
                Ok(Flow::Normal)
            }
            Stmt::Raise(e) => {
                let v = self.eval(e)?;
                Err(RuntimeError::UserRaised(v))
            }
            Stmt::Try { body, catch_var, catch_body } => {
                self.push_scope();
                let result = self.exec_block(body);
                self.pop_scope();
                match result {
                    Ok(flow) => Ok(flow),
                    Err(err) => {
                        // Any runtime error is catchable, not only explicit `raise`.
                        let bound = match err {
                            RuntimeError::UserRaised(v) => v,
                            other => Value::Str(other.to_string()),
                        };
                        self.push_scope();
                        self.define(catch_var, bound);
                        let flow = self.exec_block(catch_body);
                        self.pop_scope();
                        flow
                    }
                }
            }
            Stmt::Import(_) => {
                // Imports are spliced in by main.rs; this arm is exhaustive-only.
                Ok(Flow::Normal)
            }
            Stmt::If { cond, then_block, elif_blocks, else_block } => {
                if self.eval(cond)?.truthy() {
                    self.push_scope();
                    let flow = self.exec_block(then_block)?;
                    self.pop_scope();
                    return Ok(flow);
                }
                for (c, block) in elif_blocks {
                    if self.eval(c)?.truthy() {
                        self.push_scope();
                        let flow = self.exec_block(block)?;
                        self.pop_scope();
                        return Ok(flow);
                    }
                }
                if let Some(block) = else_block {
                    self.push_scope();
                    let flow = self.exec_block(block)?;
                    self.pop_scope();
                    return Ok(flow);
                }
                Ok(Flow::Normal)
            }
            Stmt::Loop { var, start, end, body } => {
                let start_v = self.eval(start)?.as_int()?;
                let end_v = self.eval(end)?.as_int()?;
                for i in start_v..end_v {
                    self.push_scope();
                    self.define(var, Value::Int(i));
                    let flow = self.exec_block(body)?;
                    self.pop_scope();
                    if let Flow::Return(v) = flow {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::While { cond, body } => {
                while self.eval(cond)?.truthy() {
                    self.push_scope();
                    let flow = self.exec_block(body)?;
                    self.pop_scope();
                    if let Flow::Return(v) = flow {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Return(val) => {
                let v = match val {
                    Some(e) => self.eval(e)?,
                    None => Value::Unit,
                };
                Ok(Flow::Return(v))
            }
        }
    }

    // ---- Expression evaluation ----

    fn eval(&mut self, expr: &Expr) -> RResult<Value> {
        match expr {
            Expr::IntLit(n) => Ok(Value::Int(*n)),
            Expr::FloatLit(f) => Ok(Value::Float(*f)),
            Expr::StrLit(s) => Ok(Value::Str(s.clone())),
            Expr::BoolLit(b) => Ok(Value::Bool(*b)),
            Expr::ListLit(items) => {
                let vals: RResult<Vec<Value>> = items.iter().map(|e| self.eval(e)).collect();
                Ok(Value::List(vals?))
            }
            Expr::Ident(name) => self.lookup(name),
            Expr::Binary(lhs, op, rhs) => {
                let l = self.eval(lhs)?;
                let r = self.eval(rhs)?;
                eval_binop(&l, op, &r)
            }
            Expr::Unary(op, e) => {
                let v = self.eval(e)?;
                match op {
                    crate::ast::UnOp::Neg => match v {
                        Value::Int(n) => n
                            .checked_neg()
                            .map(Value::Int)
                            .ok_or_else(|| RuntimeError::TypeError(format!("integer overflow: -({})", n))),
                        Value::Float(f) => Ok(Value::Float(-f)),
                        other => Err(RuntimeError::TypeError(format!("cannot negate {}", other))),
                    },
                    crate::ast::UnOp::Not => Ok(Value::Bool(!v.truthy())),
                }
            }
            Expr::Spawn(name, _args) => self.spawn_actor(name),
            Expr::StructLit(type_name, fields) => {
                let mut map = HashMap::new();
                for (fname, fexpr) in fields {
                    map.insert(fname.clone(), self.eval(fexpr)?);
                }
                Ok(Value::Struct(type_name.clone(), std::sync::Arc::new(std::sync::Mutex::new(map))))
            }
            Expr::FieldAccess(base, field) => {
                let v = self.eval(base)?;
                match v {
                    Value::Struct(type_name, fields) => {
                        let guard = fields.lock().unwrap();
                        guard.get(field).cloned().ok_or_else(|| {
                            RuntimeError::TypeError(format!("struct '{}' has no field '{}'", type_name, field))
                        })
                    }
                    other => Err(RuntimeError::TypeError(format!(
                        "field access '.{}' on non-struct value {}",
                        field, other
                    ))),
                }
            }
            Expr::EnumLit(type_name, variant, args) => {
                let vals: RResult<Vec<Value>> = args.iter().map(|e| self.eval(e)).collect();
                Ok(Value::Enum(type_name.clone(), variant.clone(), vals?))
            }
            Expr::Index(base, index) => {
                let base_val = self.eval(base)?;
                let idx = self.eval(index)?.as_int()?;
                match base_val {
                    Value::List(items) => {
                        if idx < 0 || idx as usize >= items.len() {
                            Err(RuntimeError::TypeError(format!(
                                "index {} out of bounds (length {})",
                                idx, items.len()
                            )))
                        } else {
                            Ok(items[idx as usize].clone())
                        }
                    }
                    other => Err(RuntimeError::TypeError(format!("cannot index into {}", other))),
                }
            }
            Expr::Closure(params, body) => {
                // Capture a snapshot of the current scope stack.
                Ok(Value::Closure(std::sync::Arc::new(ClosureData {
                    params: params.iter().map(|p| p.name.clone()).collect(),
                    body: (**body).clone(),
                    captured_env: self.scopes.clone(),
                })))
            }
            Expr::Await(inner) => {
                // `async`/`await` is eager: no scheduler; use actors for concurrency.
                let v = self.eval(inner)?;
                Ok(match v {
                    Value::Future(boxed) => *boxed,
                    other => other, // awaiting a non-future value just passes it through
                })
            }
            Expr::EnumIsVariant(subject, type_name, variant) => {
                let v = self.eval(subject)?;
                Ok(match v {
                    Value::Enum(t, var, _) => Value::Bool(&t == type_name && &var == variant),
                    _ => Value::Bool(false),
                })
            }
            Expr::EnumPayload(subject, type_name, variant, idx) => {
                let v = self.eval(subject)?;
                match v {
                    Value::Enum(t, var, args) if &t == type_name && &var == variant => args
                        .get(*idx)
                        .cloned()
                        .ok_or_else(|| RuntimeError::TypeError(format!("enum payload index {} out of bounds", idx))),
                    other => Err(RuntimeError::TypeError(format!(
                        "expected {}.{}, got {}",
                        type_name, variant, other
                    ))),
                }
            }
            Expr::Call(name, args) => self.eval_call(name, args),
        }
    }

    fn eval_call(&mut self, name: &str, arg_exprs: &[Expr]) -> RResult<Value> {
        let args: RResult<Vec<Value>> = arg_exprs.iter().map(|e| self.eval(e)).collect();
        let args = args?;

        // Local closures take priority over builtins and global functions.
        if let Ok(Value::Closure(closure)) = self.lookup(name) {
            return self.call_closure(&closure, args);
        }

        // Builtins
        match name {
            "print" => {
                let parts: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                println!("{}", parts.join(" "));
                return Ok(Value::Unit);
            }
            "tensor" => {
                let data = list_of_floats(&args)?;
                let len = data.len();
                return Ok(Value::Tensor { data, shape: vec![len] });
            }
            "tensor.zeros" => {
                let shape = match args.first() {
                    Some(Value::List(items)) => {
                        items.iter().map(|v| v.as_int().map(|n| n as usize)).collect::<RResult<Vec<_>>>()?
                    }
                    _ => {
                        return Err(RuntimeError::TypeError(
                            "tensor.zeros expects a shape list, e.g. tensor.zeros([3, 3])".into(),
                        ))
                    }
                };
                let total: usize = shape.iter().product();
                return Ok(Value::Tensor { data: vec![0.0; total], shape });
            }
            "tensor.sum" => {
                if let Some(Value::Tensor { data, .. }) = args.first() {
                    return Ok(Value::Float(data.iter().sum()));
                }
                return Err(RuntimeError::TypeError("tensor.sum expects a tensor".into()));
            }
            // Stdlib helpers
            "len" => {
                return Ok(match args.first() {
                    Some(Value::Str(s)) => Value::Int(s.chars().count() as i64),
                    Some(Value::List(items)) => Value::Int(items.len() as i64),
                    Some(Value::Tensor { data, .. }) => Value::Int(data.len() as i64),
                    Some(other) => {
                        return Err(RuntimeError::TypeError(format!("len() not supported for {}", other)))
                    }
                    None => return Err(RuntimeError::ArityMismatch { name: "len".into(), expected: 1, got: 0 }),
                });
            }
            "str_upper" => {
                return Ok(match args.first() {
                    Some(Value::Str(s)) => Value::Str(s.to_uppercase()),
                    _ => return Err(RuntimeError::TypeError("str_upper() expects a string".into())),
                });
            }
            "str_lower" => {
                return Ok(match args.first() {
                    Some(Value::Str(s)) => Value::Str(s.to_lowercase()),
                    _ => return Err(RuntimeError::TypeError("str_lower() expects a string".into())),
                });
            }
            "str_split" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(s)), Some(Value::Str(sep))) => {
                        let parts: Vec<Value> = s.split(sep.as_str()).map(|p| Value::Str(p.to_string())).collect();
                        Ok(Value::List(parts))
                    }
                    _ => Err(RuntimeError::TypeError("str_split() expects (string, string)".into())),
                };
            }
            "str_trim" => {
                return Ok(match args.first() {
                    Some(Value::Str(s)) => Value::Str(s.trim().to_string()),
                    _ => return Err(RuntimeError::TypeError("str_trim() expects a string".into())),
                });
            }
            "str_contains" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(s)), Some(Value::Str(needle))) => Ok(Value::Bool(s.contains(needle.as_str()))),
                    _ => Err(RuntimeError::TypeError("str_contains() expects (string, string)".into())),
                };
            }
            "int_to_str" => {
                return Ok(match args.first() {
                    Some(Value::Int(n)) => Value::Str(n.to_string()),
                    Some(Value::Float(f)) => Value::Str(f.to_string()),
                    _ => return Err(RuntimeError::TypeError("int_to_str() expects a number".into())),
                });
            }
            "str_to_int" => {
                return match args.first() {
                    Some(Value::Str(s)) => s
                        .trim()
                        .parse::<i64>()
                        .map(Value::Int)
                        .map_err(|_| RuntimeError::TypeError(format!("str_to_int(): '{}' is not a valid integer", s))),
                    _ => Err(RuntimeError::TypeError("str_to_int() expects a string".into())),
                };
            }
            "list_push" => {
                return match args.first() {
                    Some(Value::List(items)) => {
                        let mut new_items = items.clone();
                        if let Some(v) = args.get(1) {
                            new_items.push(v.clone());
                        }
                        Ok(Value::List(new_items))
                    }
                    _ => Err(RuntimeError::TypeError("list_push() expects a list as the first argument".into())),
                };
            }
            "list_get" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::List(items)), Some(Value::Int(idx))) => {
                        let i = *idx;
                        if i < 0 || i as usize >= items.len() {
                            Err(RuntimeError::TypeError(format!(
                                "list_get(): index {} out of bounds (length {})",
                                i, items.len()
                            )))
                        } else {
                            Ok(items[i as usize].clone())
                        }
                    }
                    _ => Err(RuntimeError::TypeError("list_get() expects (list, int)".into())),
                };
            }
            "range_list" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Int(start)), Some(Value::Int(end))) => {
                        Ok(Value::List((*start..*end).map(Value::Int).collect()))
                    }
                    _ => Err(RuntimeError::TypeError("range_list() expects (int, int)".into())),
                };
            }
            // Native stdlib (see stdlib.rs)
            "file_read" => {
                return match args.first() {
                    Some(Value::Str(path)) => crate::stdlib::file_read(path),
                    _ => Err(RuntimeError::TypeError("file_read() expects a string path".into())),
                };
            }
            "file_write" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(path)), Some(Value::Str(content))) => crate::stdlib::file_write(path, content),
                    _ => Err(RuntimeError::TypeError("file_write() expects (string path, string content)".into())),
                };
            }
            "file_append" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(path)), Some(Value::Str(content))) => crate::stdlib::file_append(path, content),
                    _ => Err(RuntimeError::TypeError("file_append() expects (string path, string content)".into())),
                };
            }
            "file_exists" => {
                return match args.first() {
                    Some(Value::Str(path)) => Ok(crate::stdlib::file_exists(path)),
                    _ => Err(RuntimeError::TypeError("file_exists() expects a string path".into())),
                };
            }
            "json_stringify" => {
                return match args.first() {
                    Some(v) => crate::stdlib::json_stringify(v),
                    None => Err(RuntimeError::ArityMismatch { name: "json_stringify".into(), expected: 1, got: 0 }),
                };
            }
            "json_parse" => {
                return match args.first() {
                    Some(Value::Str(s)) => crate::stdlib::json_parse(s),
                    _ => Err(RuntimeError::TypeError("json_parse() expects a string".into())),
                };
            }
            "regex_match" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(pat)), Some(Value::Str(text))) => crate::stdlib::regex_match(pat, text),
                    _ => Err(RuntimeError::TypeError("regex_match() expects (string pattern, string text)".into())),
                };
            }
            "regex_find" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(pat)), Some(Value::Str(text))) => crate::stdlib::regex_find(pat, text),
                    _ => Err(RuntimeError::TypeError("regex_find() expects (string pattern, string text)".into())),
                };
            }
            "regex_replace" => {
                return match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Str(pat)), Some(Value::Str(text)), Some(Value::Str(repl))) => {
                        crate::stdlib::regex_replace(pat, text, repl)
                    }
                    _ => Err(RuntimeError::TypeError(
                        "regex_replace() expects (string pattern, string text, string replacement)".into(),
                    )),
                };
            }
            "net_tcp_send" => {
                return match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Str(host)), Some(Value::Int(port)), Some(Value::Str(msg))) => {
                        crate::stdlib::net_tcp_send(host, *port, msg)
                    }
                    _ => Err(RuntimeError::TypeError(
                        "net_tcp_send() expects (string host, int port, string message)".into(),
                    )),
                };
            }
            "enum_tag" => {
                return match args.first() {
                    Some(Value::Enum(type_name, variant, _)) => {
                        match self.runtime.globals.enums.get(type_name) {
                            Some(variant_order) => variant_order
                                .iter()
                                .position(|v| v == variant)
                                .map(|idx| Value::Int(idx as i64))
                                .ok_or_else(|| {
                                    RuntimeError::TypeError(format!(
                                        "enum_tag(): '{}' has no variant '{}'",
                                        type_name, variant
                                    ))
                                }),
                            None => Err(RuntimeError::TypeError(format!(
                                "enum_tag(): undefined enum type '{}'",
                                type_name
                            ))),
                        }
                    }
                    _ => Err(RuntimeError::TypeError("enum_tag() expects an enum value".into())),
                };
            }
            "assert" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Bool(true)), _) => Ok(Value::Unit),
                    (Some(Value::Bool(false)), Some(Value::Str(msg))) => {
                        Err(RuntimeError::UserRaised(Value::Str(format!("assertion failed: {}", msg))))
                    }
                    (Some(Value::Bool(false)), None) => {
                        Err(RuntimeError::UserRaised(Value::Str("assertion failed".to_string())))
                    }
                    _ => Err(RuntimeError::TypeError("assert() expects (bool condition, optional string message)".into())),
                };
            }
            "map_new" => {
                return Ok(Value::Map(std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()))));
            }
            "map_set" => {
                return match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Map(m)), Some(Value::Str(k)), Some(v)) => {
                        m.lock().unwrap().insert(k.clone(), v.clone());
                        Ok(Value::Map(m.clone())) // same underlying map (Arc share), returned for chaining
                    }
                    _ => Err(RuntimeError::TypeError("map_set() expects (map, string key, value)".into())),
                };
            }
            "map_get" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Map(m)), Some(Value::Str(k))) => {
                        Ok(m.lock().unwrap().get(k).cloned().unwrap_or(Value::Unit))
                    }
                    _ => Err(RuntimeError::TypeError("map_get() expects (map, string key)".into())),
                };
            }
            "map_has" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Map(m)), Some(Value::Str(k))) => Ok(Value::Bool(m.lock().unwrap().contains_key(k))),
                    _ => Err(RuntimeError::TypeError("map_has() expects (map, string key)".into())),
                };
            }
            "map_delete" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Map(m)), Some(Value::Str(k))) => {
                        let removed = m.lock().unwrap().remove(k).is_some();
                        Ok(Value::Bool(removed))
                    }
                    _ => Err(RuntimeError::TypeError("map_delete() expects (map, string key)".into())),
                };
            }
            "map_keys" => {
                return match args.first() {
                    Some(Value::Map(m)) => {
                        Ok(Value::List(m.lock().unwrap().keys().map(|k| Value::Str(k.clone())).collect()))
                    }
                    _ => Err(RuntimeError::TypeError("map_keys() expects a map".into())),
                };
            }
            "map_len" => {
                return match args.first() {
                    Some(Value::Map(m)) => Ok(Value::Int(m.lock().unwrap().len() as i64)),
                    _ => Err(RuntimeError::TypeError("map_len() expects a map".into())),
                };
            }
            "env_get" => {
                return match args.first() {
                    Some(Value::Str(k)) => Ok(match std::env::var(k) {
                        Ok(v) => Value::Enum("Option".into(), "Some".into(), vec![Value::Str(v)]),
                        Err(_) => Value::Enum("Option".into(), "None".into(), vec![]),
                    }),
                    _ => Err(RuntimeError::TypeError("env_get() expects a string key".into())),
                };
            }
            "env_set" => {
                return match (args.first(), args.get(1)) {
                    (Some(Value::Str(k)), Some(Value::Str(v))) => {
                        std::env::set_var(k, v);
                        Ok(Value::Unit)
                    }
                    _ => Err(RuntimeError::TypeError("env_set() expects (string key, string value)".into())),
                };
            }
            "program_args" => {
                return Ok(Value::List(std::env::args().skip(1).map(Value::Str).collect()));
            }
            "process_exit" => {
                let code = match args.first() {
                    Some(Value::Int(n)) => *n as i32,
                    _ => 0,
                };
                // Immediate process exit; does not wait for actor shutdown.
                std::process::exit(code);
            }
            "process_run" => {
                return match args.first() {
                    Some(Value::Str(cmd)) => {
                        let cmd_args: Vec<String> = match args.get(1) {
                            Some(Value::List(items)) => items
                                .iter()
                                .map(|v| match v {
                                    Value::Str(s) => Ok(s.clone()),
                                    other => Err(RuntimeError::TypeError(format!(
                                        "process_run(): argument list must contain only strings, got {}",
                                        other
                                    ))),
                                })
                                .collect::<RResult<Vec<_>>>()?,
                            None => Vec::new(),
                            _ => return Err(RuntimeError::TypeError("process_run()'s 2nd argument must be a list of strings".into())),
                        };
                        crate::stdlib::process_run(cmd, &cmd_args)
                    }
                    _ => Err(RuntimeError::TypeError("process_run() expects a string command".into())),
                };
            }
            "udp_send" => {
                return match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Str(host)), Some(Value::Int(port)), Some(Value::Str(msg))) => {
                        crate::stdlib::udp_send(host, *port, msg)
                    }
                    _ => Err(RuntimeError::TypeError("udp_send() expects (string host, int port, string message)".into())),
                };
            }
            "http_get" => {
                return match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Str(host)), Some(Value::Int(port)), Some(Value::Str(path))) => {
                        crate::stdlib::http_get(host, *port, path)
                    }
                    _ => Err(RuntimeError::TypeError("http_get() expects (string host, int port, string path)".into())),
                };
            }
            "http_post" => {
                return match (args.first(), args.get(1), args.get(2), args.get(3)) {
                    (Some(Value::Str(host)), Some(Value::Int(port)), Some(Value::Str(path)), Some(Value::Str(body))) => {
                        crate::stdlib::http_post(host, *port, path, body)
                    }
                    _ => Err(RuntimeError::TypeError(
                        "http_post() expects (string host, int port, string path, string body)".into(),
                    )),
                };
            }
            "http_serve" => {
                return match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Int(port)), Some(Value::Closure(handler)), Some(Value::Int(max_requests))) => {
                        self.http_serve(*port as u16, handler.clone(), *max_requests as usize)
                    }
                    _ => Err(RuntimeError::TypeError(
                        "http_serve() expects (int port, closure handler(method, path) -> string, int max_requests)".into(),
                    )),
                };
            }
            "net_tcp_send_to_actor" => {
                return match (args.first(), args.get(1), args.get(2), args.get(3)) {
                    (Some(Value::ActorHandle(id)), Some(Value::Str(host)), Some(Value::Int(port)), Some(Value::Str(msg))) => {
                        self.net_tcp_send_to_actor(*id, host.clone(), *port, msg.clone())
                    }
                    _ => Err(RuntimeError::TypeError(
                        "net_tcp_send_to_actor() expects (actor handle, string host, int port, string message)".into(),
                    )),
                };
            }
            _ => {}
        }

        // Capitalized unknown name is a message-tag constructor.
        if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && !self.runtime.globals.functions.contains_key(name)
        {
            return Ok(Value::Message { tag: name.to_string(), args });
        }

        self.call_function(name, args)
    }

    fn call_function(&mut self, name: &str, args: Vec<Value>) -> RResult<Value> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(RuntimeError::TypeError(format!(
                "stack overflow: '{}' recursed past the maximum call depth ({}) — likely infinite or excessively deep recursion",
                name, MAX_CALL_DEPTH
            )));
        }

        let fn_def = self
            .runtime
            .globals
            .functions
            .get(name)
            .cloned()
            .ok_or_else(|| RuntimeError::UndefinedFunction(name.to_string()))?;

        if fn_def.params.len() != args.len() {
            return Err(RuntimeError::ArityMismatch {
                name: name.to_string(),
                expected: fn_def.params.len(),
                got: args.len(),
            });
        }

        self.call_depth += 1;
        self.push_scope();
        for (pname, pval) in fn_def.params.iter().zip(args) {
            self.define(pname, pval);
        }
        let flow = self.exec_block(&fn_def.body);
        self.pop_scope();
        self.call_depth -= 1;
        let flow = flow?;

        let result = match flow {
            Flow::Return(v) => v,
            Flow::Normal => Value::Unit,
        };
        Ok(if fn_def.is_async { Value::Future(Box::new(result)) } else { result })
    }

    /// Invoke a `Value::Closure` (single-expression body).
    fn call_closure(&mut self, closure: &ClosureData, args: Vec<Value>) -> RResult<Value> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(RuntimeError::TypeError(format!(
                "stack overflow: closure call recursed past the maximum call depth ({})",
                MAX_CALL_DEPTH
            )));
        }
        if closure.params.len() != args.len() {
            return Err(RuntimeError::ArityMismatch {
                name: "<closure>".to_string(),
                expected: closure.params.len(),
                got: args.len(),
            });
        }

        // Evaluate in the captured environment, then restore caller scopes.
        let caller_scopes = std::mem::replace(&mut self.scopes, closure.captured_env.clone());
        self.push_scope();
        for (pname, pval) in closure.params.iter().zip(args) {
            self.define(pname, pval);
        }
        self.call_depth += 1;
        let result = self.eval(&closure.body);
        self.call_depth -= 1;
        self.scopes = caller_scopes;
        result
    }

    /// Bind a local TCP server and invoke a user closure per request.
    fn http_serve(&mut self, port: u16, handler: std::sync::Arc<ClosureData>, max_requests: usize) -> RResult<Value> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| RuntimeError::TypeError(format!("http_serve(): bind to port {} failed: {}", port, e)))?;

        let mut handled = 0usize;
        for stream in listener.incoming() {
            if handled >= max_requests {
                break;
            }
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]);
            let mut parts = request.split_whitespace();
            let method = parts.next().unwrap_or("GET").to_string();
            let path = parts.next().unwrap_or("/").to_string();

            // Run the user closure as the HTTP response body.
            let response_body = match self.call_closure(&handler, vec![Value::Str(method), Value::Str(path)]) {
                Ok(v) => v.to_string(),
                Err(e) => format!("Internal Server Error: {}", e),
            };
            let http_response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(http_response.as_bytes());
            handled += 1;
        }
        Ok(Value::Int(handled as i64))
    }

    /// Fire TCP on a background thread and deliver the result to an actor.
    fn net_tcp_send_to_actor(&mut self, actor_id: usize, host: String, port: i64, msg: String) -> RResult<Value> {
        let runtime = Arc::clone(&self.runtime);
        let handle = thread::spawn(move || {
            let result = crate::stdlib::net_tcp_send(&host, port, &msg);
            let payload = match result {
                Ok(v) => Value::Message { tag: "NetworkResponse".to_string(), args: vec![v] },
                Err(e) => Value::Message { tag: "NetworkError".to_string(), args: vec![Value::Str(e.to_string())] },
            };
            runtime.mailboxes.send(actor_id, payload);
        });
        // Join this thread in `shutdown_actors` so the process waits for I/O.
        self.runtime.handles.lock().unwrap().push(handle);
        Ok(Value::Unit)
    }

    fn spawn_actor(&mut self, name: &str) -> RResult<Value> {
        let actor_def = self
            .runtime
            .globals
            .actors
            .get(name)
            .cloned()
            .ok_or_else(|| RuntimeError::UndefinedActor(name.to_string()))?;

        let id = self.runtime.next_actor_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tridentix_rt_mailbox::channel();
        self.runtime.mailboxes.register(id, tx);

        let runtime = Arc::clone(&self.runtime);
        let handle = thread::Builder::new()
            .stack_size(tridentix_rt_mailbox::actor_stack_size())
            .spawn(move || {
                run_actor_worker(runtime, actor_def, id, rx, None);
            })
            .map_err(|e| RuntimeError::ThreadSpawnFailed(e.to_string()))?;

        self.runtime.handles.lock().unwrap().push(handle);
        Ok(Value::ActorHandle(id))
    }

    /// Spawn a supervised child; crashes go to `events`.
    fn spawn_supervised_child(
        runtime: &Arc<RuntimeShared>,
        actor_name: &str,
        child_index: usize,
        events: Sender<SupervisorEvent>,
    ) -> RResult<usize> {
        let actor_def = runtime
            .globals
            .actors
            .get(actor_name)
            .cloned()
            .ok_or_else(|| RuntimeError::UndefinedActor(actor_name.to_string()))?;

        let id = runtime.next_actor_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tridentix_rt_mailbox::channel();
        runtime.mailboxes.register(id, tx);

        let runtime_clone = Arc::clone(runtime);
        let handle = thread::Builder::new()
            .stack_size(tridentix_rt_mailbox::actor_stack_size())
            .spawn(move || {
                run_actor_worker(runtime_clone, actor_def, id, rx, Some((events, child_index)));
            })
            .map_err(|e| RuntimeError::ThreadSpawnFailed(e.to_string()))?;
        runtime.handles.lock().unwrap().push(handle);

        Ok(id)
    }

    /// Spawn supervised children and a coordinator that restarts on crash.
    fn spawn_supervisor(
        &mut self,
        sup_name: &str,
        strategy: RestartStrategy,
        max_restarts: i64,
        time_window_secs: i64,
        children: &[crate::ast::SupervisedChild],
    ) -> RResult<Vec<(String, Value)>> {
        let runtime = Arc::clone(&self.runtime);
        let (events_tx, events_rx) = mpsc::channel::<SupervisorEvent>();

        let mut child_ids = Vec::with_capacity(children.len());
        let mut bindings = Vec::with_capacity(children.len());
        for (idx, child) in children.iter().enumerate() {
            let id = Self::spawn_supervised_child(&runtime, &child.actor_name, idx, events_tx.clone())?;
            child_ids.push(id);
            bindings.push((child.bind_name.clone(), Value::ActorHandle(id)));
        }
        let coordinator_events_tx = events_tx.clone();
        self.runtime
            .supervisor_shutdown_senders
            .lock()
            .unwrap()
            .push(coordinator_events_tx.clone());
        drop(events_tx); // children hold their own clones; original no longer needed here

        let sup_name_owned = sup_name.to_string();
        let children_owned: Vec<crate::ast::SupervisedChild> = children.to_vec();
        let coordinator_runtime = Arc::clone(&self.runtime);

        let config = SupervisorConfig {
            sup_name: sup_name_owned,
            strategy,
            max_restarts,
            time_window_secs,
            children: children_owned,
            child_ids,
        };
        let coordinator = thread::Builder::new()
            .stack_size(tridentix_rt_mailbox::actor_stack_size())
            .spawn(move || {
                run_supervisor_coordinator(
                    coordinator_runtime,
                    config,
                    coordinator_events_tx,
                    events_rx,
                );
            })
            .map_err(|e| RuntimeError::ThreadSpawnFailed(e.to_string()))?;

        self.runtime.handles.lock().unwrap().push(coordinator);
        Ok(bindings)
    }

}

/// Supervisor configuration passed to coordinator thread.
struct SupervisorConfig {
    sup_name: String,
    strategy: RestartStrategy,
    max_restarts: i64,
    time_window_secs: i64,
    children: Vec<crate::ast::SupervisedChild>,
    child_ids: Vec<usize>,
}

/// Shared actor receive loop (`events = None` is unsupervised / self-healing).
fn run_actor_worker(
    runtime: Arc<RuntimeShared>,
    actor_def: ActorDef,
    id: usize,
    rx: Receiver<Value>,
    events: Option<(Sender<SupervisorEvent>, usize)>,
) {
    let mut actor_interp = Interpreter {
        runtime,
        scopes: vec![HashMap::new()],
        call_depth: 0,
    };
    for (field_name, init_expr) in &actor_def.state_vars {
        match actor_interp.eval(init_expr) {
            Ok(v) => actor_interp.define(field_name, v),
            Err(e) => {
                eprintln!("[actor #{} state-init error] {}: {}", id, field_name, e);
                return; // can't safely run with broken state
            }
        }
    }
    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => break, // sender dropped, nothing left to do
        };
        if let Value::Message { tag, .. } = &msg {
            if tag == "__shutdown__" {
                if let Some((sender, idx)) = &events {
                    let _ = sender.send(SupervisorEvent::ShutdownAck(*idx));
                }
                break;
            }
            if tag == "__supervisor_restart__" {
                break; // coordinator-initiated restart; no event needed
            }
        }
        actor_interp.push_scope();
        actor_interp.define(&actor_def.param_name, msg);
        let result = actor_interp.exec_block(&actor_def.body);
        actor_interp.pop_scope();

        if let Err(e) = result {
            eprintln!("[actor #{} error] {}", id, e);
            match &events {
                Some((sender, idx)) => {
                    let _ = sender.send(SupervisorEvent::Crashed(*idx));
                    break; // supervised: crash ends this thread, coordinator decides what's next
                }
                None => continue, // unsupervised: self-heal, keep processing
            }
        }
    }
}

/// Supervisor coordinator: restart strategy, crash events, shutdown.
fn run_supervisor_coordinator(
    runtime: Arc<RuntimeShared>,
    config: SupervisorConfig,
    events_tx: Sender<SupervisorEvent>,
    events_rx: Receiver<SupervisorEvent>,
) {
    let SupervisorConfig {
        sup_name,
        strategy,
        max_restarts,
        time_window_secs,
        children,
        mut child_ids,
    } = config;
    let n = children.len();
    let mut done = vec![false; n];
    let mut given_up = vec![false; n];
    let mut restart_count = 0i64;
    let mut window_start = Instant::now();
    let window = Duration::from_secs(time_window_secs.max(0) as u64);
    // During shutdown, crashes finish the child instead of respawning.
    let mut shutting_down = false;

    let respawn = |runtime: &Arc<RuntimeShared>, idx: usize, old_id: usize, actor_name: &str| -> usize {
        // Keep the same actor id so existing handles keep working after restart.
        let actor_def = match runtime.globals.actors.get(actor_name).cloned() {
            Some(a) => a,
            None => return old_id, // shouldn't happen: validated at parse/spawn time
        };
        let (tx, rx) = tridentix_rt_mailbox::channel();
        runtime.mailboxes.register(old_id, tx);
        let runtime_clone = Arc::clone(runtime);
        let events_clone = events_tx.clone();
        match thread::Builder::new()
            .stack_size(tridentix_rt_mailbox::actor_stack_size())
            .spawn(move || {
                run_actor_worker(runtime_clone, actor_def, old_id, rx, Some((events_clone, idx)));
            }) {
            Ok(handle) => {
                runtime.handles.lock().unwrap().push(handle);
            }
            Err(e) => {
                eprintln!("[supervisor] failed to respawn actor thread: {}", e);
            }
        }
        old_id
    };

    loop {
        if done.iter().all(|d| *d) {
            break;
        }
        let event = match events_rx.recv() {
            Ok(e) => e,
            Err(_) => break, // all senders dropped -> nothing left to supervise
        };

        match event {
            SupervisorEvent::ShutdownAck(idx) => {
                done[idx] = true;
            }
            SupervisorEvent::GlobalShutdown => {
                shutting_down = true;
                for i in 0..n {
                    if done[i] || given_up[i] {
                        continue;
                    }
                    runtime.mailboxes.send_blocking(
                        child_ids[i],
                        Value::Message {
                            tag: "__shutdown__".to_string(),
                            args: vec![],
                        },
                    );
                }
            }
            SupervisorEvent::Crashed(idx) => {
                if done[idx] || given_up[idx] {
                    continue;
                }
                if shutting_down {
                    // Exiting: mark finished, do not respawn.
                    done[idx] = true;
                    continue;
                }

                // Sliding restart-budget window (Compiler doc §2.3).
                if window_start.elapsed() > window {
                    window_start = Instant::now();
                    restart_count = 0;
                }
                restart_count += 1;

                if restart_count > max_restarts {
                    eprintln!(
                        "[supervisor {}] child '{}' (index {}) exceeded max_restarts ({}) within {}s — escalating, giving up on this child",
                        sup_name, children[idx].bind_name, idx, max_restarts, time_window_secs
                    );
                    given_up[idx] = true;
                    done[idx] = true; // count it as finished so the coordinator can eventually exit
                    continue;
                }

                let targets: Vec<usize> = match strategy {
                    RestartStrategy::OneForOne => vec![idx],
                    RestartStrategy::OneForAll => (0..n).filter(|i| !done[*i] && !given_up[*i]).collect(),
                    RestartStrategy::RestForOne => (idx..n).filter(|i| !done[*i] && !given_up[*i]).collect(),
                };

                eprintln!(
                    "[supervisor {}] restarting {:?} (strategy={:?}, restart #{} in current window)",
                    sup_name,
                    targets.iter().map(|i| children[*i].bind_name.clone()).collect::<Vec<_>>(),
                    strategy,
                    restart_count
                );

                for &t in &targets {
                    if t != idx {
                        // Stop a live sibling before replacing it.
                        runtime.mailboxes.send_blocking(
                            child_ids[t],
                            Value::Message {
                                tag: "__supervisor_restart__".to_string(),
                                args: vec![],
                            },
                        );
                    }
                    let new_id = respawn(&runtime, t, child_ids[t], &children[t].actor_name);
                    child_ids[t] = new_id;
                }
            }
        }
    }
}


impl Value {
    fn truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(items) => !items.is_empty(),
            Value::Unit => false,
            _ => true,
        }
    }

    fn as_int(&self) -> RResult<i64> {
        match self {
            Value::Int(n) => Ok(*n),
            other => Err(RuntimeError::TypeError(format!("expected int, got {}", other))),
        }
    }

    fn as_float(&self) -> RResult<f64> {
        match self {
            Value::Int(n) => Ok(*n as f64),
            Value::Float(f) => Ok(*f),
            other => Err(RuntimeError::TypeError(format!("expected number, got {}", other))),
        }
    }
}

fn list_of_floats(args: &[Value]) -> RResult<Vec<f64>> {
    if args.len() == 1 {
        if let Value::List(items) = &args[0] {
            return items.iter().map(|v| v.as_float()).collect();
        }
    }
    args.iter().map(|v| v.as_float()).collect()
}

/// Structural equality for enum payloads (Value is not PartialEq).
fn values_structurally_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Enum(t1, v1, a1), Value::Enum(t2, v2, a2)) => {
            t1 == t2 && v1 == v2 && a1.len() == a2.len() && a1.iter().zip(a2.iter()).all(|(x, y)| values_structurally_equal(x, y))
        }
        _ => false,
    }
}

fn eval_binop(l: &Value, op: &BinOp, r: &Value) -> RResult<Value> {
    use BinOp::*;
    // String concatenation via '+' as a convenience for print-style code.
    if let (Value::Str(a), Add, Value::Str(b)) = (l, op, r) {
        return Ok(Value::Str(format!("{}{}", a, b)));
    }
    // String compare before numeric fallback (e.g. `msg == "crash"`).
    if let (Value::Str(a), Value::Str(b)) = (l, r) {
        return Ok(match op {
            Eq => Value::Bool(a == b),
            NotEq => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Gt => Value::Bool(a > b),
            LtEq => Value::Bool(a <= b),
            GtEq => Value::Bool(a >= b),
            Add => unreachable!("Add on two strings handled above"),
            Sub | Mul | Div => {
                return Err(RuntimeError::TypeError("operator not supported between two strings".to_string()))
            }
        });
    }
    // Enum == / != by type, variant, and payload.
    if let (Value::Enum(t1, v1, args1), Value::Enum(t2, v2, args2)) = (l, r) {
        let equal = t1 == t2 && v1 == v2 && args1.len() == args2.len() && args1
            .iter()
            .zip(args2.iter())
            .all(|(a, b)| values_structurally_equal(a, b));
        return Ok(match op {
            Eq => Value::Bool(equal),
            NotEq => Value::Bool(!equal),
            _ => {
                return Err(RuntimeError::TypeError(
                    "only == and != are supported between enum values".into(),
                ))
            }
        });
    }
    // Bool equality.
    if let (Value::Bool(a), Value::Bool(b)) = (l, r) {
        return Ok(match op {
            Eq => Value::Bool(a == b),
            NotEq => Value::Bool(a != b),
            _ => {
                return Err(RuntimeError::TypeError(
                    "only == and != are supported between bools".into(),
                ))
            }
        });
    }

    let is_float = matches!(l, Value::Float(_)) || matches!(r, Value::Float(_));
    if is_float {
        let a = l.as_float()?;
        let b = r.as_float()?;
        return Ok(match op {
            Add => Value::Float(a + b),
            Sub => Value::Float(a - b),
            Mul => Value::Float(a * b),
            Div => Value::Float(a / b),
            Eq => Value::Bool(a == b),
            NotEq => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Gt => Value::Bool(a > b),
            LtEq => Value::Bool(a <= b),
            GtEq => Value::Bool(a >= b),
        });
    }

    let a = l.as_int()?;
    let b = r.as_int()?;
    Ok(match op {
        // Use checked_* so overflow/div-zero become RuntimeError.
        Add => Value::Int(a.checked_add(b).ok_or_else(|| {
            RuntimeError::TypeError(format!("integer overflow: {} + {}", a, b))
        })?),
        Sub => Value::Int(a.checked_sub(b).ok_or_else(|| {
            RuntimeError::TypeError(format!("integer overflow: {} - {}", a, b))
        })?),
        Mul => Value::Int(a.checked_mul(b).ok_or_else(|| {
            RuntimeError::TypeError(format!("integer overflow: {} * {}", a, b))
        })?),
        Div => {
            if b == 0 {
                return Err(RuntimeError::TypeError(format!("division by zero: {} / 0", a)));
            }
            Value::Int(a.checked_div(b).ok_or_else(|| {
                RuntimeError::TypeError(format!("integer overflow: {} / {}", a, b))
            })?)
        }
        Eq => Value::Bool(a == b),
        NotEq => Value::Bool(a != b),
        Lt => Value::Bool(a < b),
        Gt => Value::Bool(a > b),
        LtEq => Value::Bool(a <= b),
        GtEq => Value::Bool(a >= b),
    })
}
