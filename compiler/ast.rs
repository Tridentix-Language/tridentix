//! Tridentix AST: functions, actors, supervisors, structs, enums, expressions.

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnOp {
    Neg, // -x
    Not, // not x
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub type_ann: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    /// Positional payload slot names, e.g. `Some(value)` → `["value"]`.
    pub payload_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    BoolLit(bool),
    ListLit(Vec<Expr>),
    Ident(String),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(String, Vec<Expr>),
    Spawn(String, Vec<Expr>),
    /// `Point { x: 1, y: 2 }` — struct literal construction.
    StructLit(String, Vec<(String, Expr)>),
    /// `point.x` — field access.
    FieldAccess(Box<Expr>, String),
    /// `Color.Red` or `Option.Some(5)` — enum variant construction.
    EnumLit(String, String, Vec<Expr>),
    /// `fn(a, b) => a + b` — single-expression closure body.
    Closure(Vec<Param>, Box<Expr>),
    Await(Box<Expr>),
    /// `list[index]` — read access.
    Index(Box<Expr>, Box<Expr>),
    EnumIsVariant(Box<Expr>, String, String),
    EnumPayload(Box<Expr>, String, String, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub type_ann: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RestartStrategy {
    OneForOne,
    OneForAll,
    RestForOne,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SupervisedChild {
    pub actor_name: String,
    pub bind_name: String, // the `as workerN` alias used to refer to it
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        mutable: bool,
        type_ann: Option<String>,
        value: Expr,
    },
    Assign {
        name: String,
        value: Expr,
    },
    /// `list[index] = value`.
    IndexAssign {
        base: Expr,
        index: Expr,
        value: Expr,
    },
    FieldAssign {
        base: Expr,
        field: String,
        value: Expr,
    },
    Send {
        target: Expr,
        message: Expr,
    },
    Raise(Expr),
    Try {
        body: Vec<Stmt>,
        catch_var: String,
        catch_body: Vec<Stmt>,
    },
    Import(String),
    Actor {
        name: String,
        state_vars: Vec<(String, Option<String>, Expr)>,
        param_name: String,
        param_type: Option<String>,
        body: Vec<Stmt>,
    },
    Supervisor {
        name: String,
        strategy: RestartStrategy,
        max_restarts: i64,
        time_window_secs: i64,
        children: Vec<SupervisedChild>,
    },
    ExprStmt(Expr),
    If {
        cond: Expr,
        then_block: Vec<Stmt>,
        elif_blocks: Vec<(Expr, Vec<Stmt>)>,
        else_block: Option<Vec<Stmt>>,
    },
    Loop {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    Return(Option<Expr>),
    FnDef {
        name: String,
        type_params: Vec<String>,
        params: Vec<Param>,
        ret_type: Option<String>,
        is_async: bool,
        body: Vec<Stmt>,
    },
    StructDef {
        name: String,
        fields: Vec<StructField>,
    },
    EnumDef {
        name: String,
        variants: Vec<EnumVariant>,
    },
}

pub type Program = Vec<Stmt>;

pub fn print_program(program: &Program) {
    for stmt in program {
        print_stmt(stmt, 0);
    }
}

fn indent(n: usize) -> String {
    "  ".repeat(n)
}

fn print_stmt(stmt: &Stmt, depth: usize) {
    match stmt {
        Stmt::Let {
            name,
            mutable,
            type_ann,
            value,
        } => {
            println!(
                "{}Let(mut={}, name={}, type={:?}) = {}",
                indent(depth),
                mutable,
                name,
                type_ann,
                fmt_expr(value)
            );
        }
        Stmt::Assign { name, value } => {
            println!("{}Assign {} = {}", indent(depth), name, fmt_expr(value));
        }
        Stmt::IndexAssign { base, index, value } => {
            println!("{}IndexAssign {}[{}] = {}", indent(depth), fmt_expr(base), fmt_expr(index), fmt_expr(value));
        }
        Stmt::FieldAssign { base, field, value } => {
            println!("{}FieldAssign {}.{} = {}", indent(depth), fmt_expr(base), field, fmt_expr(value));
        }
        Stmt::Send { target, message } => {
            println!("{}Send({}, {})", indent(depth), fmt_expr(target), fmt_expr(message));
        }
        Stmt::Raise(e) => {
            println!("{}Raise {}", indent(depth), fmt_expr(e));
        }
        Stmt::Try { body, catch_var, catch_body } => {
            println!("{}Try", indent(depth));
            for s in body {
                print_stmt(s, depth + 1);
            }
            println!("{}Catch({})", indent(depth), catch_var);
            for s in catch_body {
                print_stmt(s, depth + 1);
            }
        }
        Stmt::Import(path) => {
            println!("{}Import(\"{}\")", indent(depth), path);
        }
        Stmt::Actor {
            name,
            state_vars,
            param_name,
            param_type,
            body,
        } => {
            println!("{}Actor {}", indent(depth), name);
            for (sname, stype, sinit) in state_vars {
                println!(
                    "{}  state {}: {} = {}",
                    indent(depth), sname, stype.clone().unwrap_or_else(|| "?".into()), fmt_expr(sinit)
                );
            }
            println!(
                "{}  on receive({}: {})",
                indent(depth),
                param_name,
                param_type.clone().unwrap_or_else(|| "?".into())
            );
            for s in body {
                print_stmt(s, depth + 1);
            }
        }
        Stmt::Supervisor { name, strategy, max_restarts, time_window_secs, children } => {
            println!(
                "{}Supervisor {} (strategy={:?}, max_restarts={}, window={}s)",
                indent(depth), name, strategy, max_restarts, time_window_secs
            );
            for c in children {
                println!("{}  child: {} as {}", indent(depth), c.actor_name, c.bind_name);
            }
        }
        Stmt::ExprStmt(e) => {
            println!("{}ExprStmt: {}", indent(depth), fmt_expr(e));
        }
        Stmt::If {
            cond,
            then_block,
            elif_blocks,
            else_block,
        } => {
            println!("{}If ({})", indent(depth), fmt_expr(cond));
            for s in then_block {
                print_stmt(s, depth + 1);
            }
            for (c, block) in elif_blocks {
                println!("{}Elif ({})", indent(depth), fmt_expr(c));
                for s in block {
                    print_stmt(s, depth + 1);
                }
            }
            if let Some(block) = else_block {
                println!("{}Else", indent(depth));
                for s in block {
                    print_stmt(s, depth + 1);
                }
            }
        }
        Stmt::Loop {
            var,
            start,
            end,
            body,
        } => {
            println!(
                "{}Loop {} in range({}, {})",
                indent(depth),
                var,
                fmt_expr(start),
                fmt_expr(end)
            );
            for s in body {
                print_stmt(s, depth + 1);
            }
        }
        Stmt::While { cond, body } => {
            println!("{}While ({})", indent(depth), fmt_expr(cond));
            for s in body {
                print_stmt(s, depth + 1);
            }
        }
        Stmt::Return(val) => {
            println!("{}Return {}", indent(depth), val.as_ref().map(fmt_expr).unwrap_or_default());
        }
        Stmt::FnDef {
            name,
            type_params,
            params,
            ret_type,
            is_async,
            body,
        } => {
            let params_str: Vec<String> = params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.type_ann.clone().unwrap_or_else(|| "?".into())))
                .collect();
            let generics_str = if type_params.is_empty() {
                String::new()
            } else {
                format!("<{}>", type_params.join(", "))
            };
            println!(
                "{}{}Fn {}{}({}) -> {}",
                indent(depth),
                if *is_async { "async " } else { "" },
                name,
                generics_str,
                params_str.join(", "),
                ret_type.clone().unwrap_or_else(|| "?".into())
            );
            for s in body {
                print_stmt(s, depth + 1);
            }
        }
        Stmt::StructDef { name, fields } => {
            let fields_str: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", f.name, f.type_ann.clone().unwrap_or_else(|| "?".into())))
                .collect();
            println!("{}Struct {} {{ {} }}", indent(depth), name, fields_str.join(", "));
        }
        Stmt::EnumDef { name, variants } => {
            let variants_str: Vec<String> = variants.iter().map(|v| v.name.clone()).collect();
            println!("{}Enum {} {{ {} }}", indent(depth), name, variants_str.join(", "));
        }
    }
}

fn fmt_expr(e: &Expr) -> String {
    match e {
        Expr::IntLit(n) => n.to_string(),
        Expr::FloatLit(f) => f.to_string(),
        Expr::StrLit(s) => format!("\"{}\"", s),
        Expr::BoolLit(b) => b.to_string(),
        Expr::ListLit(items) => {
            let items_str: Vec<String> = items.iter().map(fmt_expr).collect();
            format!("[{}]", items_str.join(", "))
        }
        Expr::Ident(name) => name.clone(),
        Expr::Binary(lhs, op, rhs) => {
            format!("({} {} {})", fmt_expr(lhs), fmt_op(op), fmt_expr(rhs))
        }
        Expr::Unary(op, e) => {
            let sym = match op {
                crate::ast::UnOp::Neg => "-",
                crate::ast::UnOp::Not => "not ",
            };
            format!("({}{})", sym, fmt_expr(e))
        }
        Expr::Call(name, args) => {
            let args_str: Vec<String> = args.iter().map(fmt_expr).collect();
            format!("{}({})", name, args_str.join(", "))
        }
        Expr::Spawn(name, args) => {
            let args_str: Vec<String> = args.iter().map(fmt_expr).collect();
            format!("spawn {}({})", name, args_str.join(", "))
        }
        Expr::StructLit(name, fields) => {
            let fields_str: Vec<String> = fields.iter().map(|(n, e)| format!("{}: {}", n, fmt_expr(e))).collect();
            format!("{} {{ {} }}", name, fields_str.join(", "))
        }
        Expr::FieldAccess(base, field) => format!("{}.{}", fmt_expr(base), field),
        Expr::Index(base, idx) => format!("{}[{}]", fmt_expr(base), fmt_expr(idx)),
        Expr::EnumLit(enum_name, variant, args) => {
            if args.is_empty() {
                format!("{}.{}", enum_name, variant)
            } else {
                let args_str: Vec<String> = args.iter().map(fmt_expr).collect();
                format!("{}.{}({})", enum_name, variant, args_str.join(", "))
            }
        }
        Expr::Closure(params, body) => {
            let params_str: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
            format!("fn({}) => {}", params_str.join(", "), fmt_expr(body))
        }
        Expr::Await(e) => format!("await {}", fmt_expr(e)),
        Expr::EnumIsVariant(subj, t, v) => format!("__is_variant({}, {}.{})", fmt_expr(subj), t, v),
        Expr::EnumPayload(subj, t, v, idx) => format!("__payload({}, {}.{}, {})", fmt_expr(subj), t, v, idx),
    }
}

fn fmt_op(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Eq => "==",
        BinOp::NotEq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::LtEq => "<=",
        BinOp::GtEq => ">=",
    }
}
