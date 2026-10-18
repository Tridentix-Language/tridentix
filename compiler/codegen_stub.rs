//! LLVM-free backend used when the `llvm` cargo feature is off.

use crate::ast::Program;

pub enum CompileTarget {
    Jit,
    AheadOfTime {
        output_path: String,
        opt_passes: String,
    },
}

pub struct CodegenError(pub String);

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub fn compile_and_run(program: &Program) -> Result<(), CodegenError> {
    compile_and_execute(program, CompileTarget::Jit)
}

pub fn compile_and_execute(_program: &Program, _target: CompileTarget) -> Result<(), CodegenError> {
    Err(CodegenError(
        "LLVM codegen is disabled in this build. Rebuild with `--features llvm` after installing LLVM 20.1 \
(set LLVM_SYS_201_PREFIX). Use `tridentix run <file.trix>` for the interpreter."
            .into(),
    ))
}
