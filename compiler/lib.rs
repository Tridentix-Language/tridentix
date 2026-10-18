//! Tridentix compiler: lexer, parser, AST, borrow checker, codegen, interpreter.

pub mod ast;
pub mod borrow_checker;
#[cfg(feature = "llvm")]
pub mod codegen;
#[cfg(not(feature = "llvm"))]
#[path = "codegen_stub.rs"]
pub mod codegen;
pub mod interpreter;
pub mod lexer;
pub mod parser;
pub mod stdlib;
pub mod tridentix_rt_object;
pub mod tridentix_rt_parser;
pub mod typechecker;
