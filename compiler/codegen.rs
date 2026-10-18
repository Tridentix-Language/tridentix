//! LLVM 20 codegen: JIT (`build`) and AOT (`aot`) for the compiled subset.

use crate::ast::{BinOp, Expr, Program, Stmt, UnOp};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::{FloatValue, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use std::collections::HashMap;

/// JIT in-process, or AOT to a native binary.
pub enum CompileTarget {
    Jit,
    AheadOfTime {
        output_path: String,
        /// PassBuilder pipeline, e.g. `"default<O0>"` .. `"default<O3>"`.
        opt_passes: String,
    },
}

pub struct CodegenError(pub String);

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

type CResult<T> = Result<T, CodegenError>;

/// Recursion-depth cap for compiled native calls.
const MAX_NATIVE_CALL_DEPTH: i64 = 100_000;

#[derive(Clone, Copy)]
enum CgValue<'ctx> {
    Int(IntValue<'ctx>),
    Float(FloatValue<'ctx>),
}

#[derive(Clone, PartialEq)]
enum SlotKind<'ctx> {
    Int,
    Float,
    Str,
    /// Tridentix struct type name for field lookup.
    Struct(String),
    /// Holds the enum's Tridentix type name, analogous to `Struct(String)`.
    Enum(String),
    /// Lifted closure function plus float-param flags.
    Closure(FunctionValue<'ctx>, Vec<bool> /* true = float param */),
}

#[derive(Clone)]
struct VarSlot<'ctx> {
    ptr: PointerValue<'ctx>,
    kind: SlotKind<'ctx>,
}

/// Lifted closure with a heap-allocated capture environment.
struct ClosureInfo<'ctx> {
    fn_val: FunctionValue<'ctx>,
    env_struct_ty: inkwell::types::StructType<'ctx>,
    /// Free vars in env-struct field order (sorted).
    free_vars: Vec<String>,
    param_names: Vec<String>,
    param_is_float: Vec<bool>,
}

/// LLVM struct type; fields are currently int/float only.
struct StructTypeInfo<'ctx> {
    llvm_type: inkwell::types::StructType<'ctx>,
    /// Field names in declaration order (GEP indices).
    field_order: Vec<String>,
    field_kinds: Vec<SlotKind<'ctx>>, // one of SlotKind::Int / SlotKind::Float per field
}

/// Tagged union `{ i64 tag, i64 payload }`.
struct EnumTypeInfo<'ctx> {
    llvm_type: inkwell::types::StructType<'ctx>,
    /// Variant name -> its tag value (== its declaration-order index).
    variant_tags: HashMap<String, i64>,
    /// Variant name -> does it have a (single) payload slot?
    variant_has_payload: HashMap<String, bool>,
}

struct FnSpec<'a> {
    name: String,
    params: Vec<(String, Option<String>)>,
    body: &'a [Stmt],
}

/// JIT entry point used by `tridentix build`.
pub fn compile_and_run(program: &Program) -> CResult<()> {
    compile_and_execute(program, CompileTarget::Jit)
}

pub fn compile_and_execute(program: &Program, target: CompileTarget) -> CResult<()> {
    let context = Context::create();
    let module = context.create_module("tridentix_module");
    let builder = context.create_builder();

    let printf_fn = declare_printf(&context, &module);
    // Register libc `exit` for guarded integer division.
    let _exit_fn = declare_exit(&context, &module);
    let depth_global = declare_call_depth_global(&context, &module);

    let i64_type = context.i64_type();
    let f64_type = context.f64_type();

    // Lower int/float structs; skip other field types with a warning.
    let mut struct_types: HashMap<String, StructTypeInfo> = HashMap::new();
    let mut skipped_structs = Vec::new();
    for stmt in program {
        if let Stmt::StructDef { name, fields } = stmt {
            let mut field_order = Vec::new();
            let mut field_kinds = Vec::new();
            let mut llvm_field_types = Vec::new();
            let mut ok = true;
            for f in fields {
                match f.type_ann.as_deref() {
                    Some("float") => {
                        field_order.push(f.name.clone());
                        field_kinds.push(SlotKind::Float);
                        llvm_field_types.push(f64_type.into());
                    }
                    Some("int") | None => {
                        field_order.push(f.name.clone());
                        field_kinds.push(SlotKind::Int);
                        llvm_field_types.push(i64_type.into());
                    }
                    Some(other) => {
                        skipped_structs.push(format!(
                            "{} (field '{}' has unsupported codegen type '{}')",
                            name, f.name, other
                        ));
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                let llvm_type = context.struct_type(&llvm_field_types, false);
                struct_types.insert(name.clone(), StructTypeInfo { llvm_type, field_order, field_kinds });
            }
        }
    }

    // Uniform enum layout: `{ i64 tag, i64 payload }`.
    let mut enum_types: HashMap<String, EnumTypeInfo> = HashMap::new();
    for stmt in program {
        if let Stmt::EnumDef { name, variants } = stmt {
            let mut variant_tags = HashMap::new();
            let mut variant_has_payload = HashMap::new();
            for (idx, v) in variants.iter().enumerate() {
                variant_tags.insert(v.name.clone(), idx as i64);
                variant_has_payload.insert(v.name.clone(), !v.payload_names.is_empty());
            }
            let llvm_type = context.struct_type(&[i64_type.into(), i64_type.into()], false);
            enum_types.insert(name.clone(), EnumTypeInfo { llvm_type, variant_tags, variant_has_payload });
        }
    }

    let malloc_fn = declare_malloc(&context, &module);

    // Declare lifted functions for every closure before compiling bodies.
    let mut closure_literals: Vec<(&Vec<crate::ast::Param>, &Expr)> = Vec::new();
    for stmt in program {
        if let Stmt::FnDef { body, .. } = stmt {
            collect_closures_in_stmts(body, &mut closure_literals);
        }
    }

    let mut closure_infos: HashMap<*const Expr, ClosureInfo> = HashMap::new();
    for (idx, (params, body)) in closure_literals.iter().enumerate() {
        let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
        let mut free_vars_set = std::collections::BTreeSet::new();
        collect_free_vars(body, &param_names, &mut free_vars_set);
        let free_vars: Vec<String> = free_vars_set.into_iter().collect();

        let env_field_types: Vec<_> = free_vars.iter().map(|_| i64_type.into()).collect();
        let env_struct_ty = context.struct_type(&env_field_types, false);

        let param_is_float: Vec<bool> = params.iter().map(|p| p.type_ann.as_deref() == Some("float")).collect();
        let mut fn_param_types: Vec<inkwell::types::BasicMetadataTypeEnum> =
            vec![context.i8_type().ptr_type(AddressSpace::default()).into()];
        for is_float in &param_is_float {
            fn_param_types.push(if *is_float { f64_type.into() } else { i64_type.into() });
        }
        let fn_type = i64_type.fn_type(&fn_param_types, false);
        let fn_val = module.add_function(&format!("__closure_{}", idx), fn_type, None);

        closure_infos.insert(
            *body as *const Expr,
            ClosureInfo {
                fn_val,
                env_struct_ty,
                free_vars,
                param_names,
                param_is_float,
            },
        );
    }

    // Lift actor handlers (`extern "C" fn(i64) -> i64`); skip actors with `state:`.
    let (spawn_fn, send_fn, shutdown_fn) = declare_actor_runtime(&context, &module);
    let mut actor_infos: HashMap<String, ActorCodegenInfo> = HashMap::new();
    let mut skipped_actors = Vec::new();
    for stmt in program {
        if let Stmt::Actor { name, state_vars, param_name, .. } = stmt {
            if !state_vars.is_empty() {
                skipped_actors.push(format!(
                    "{} (actors with persistent `state:` aren't supported in native codegen yet — see ActorCodegenInfo's doc comment)",
                    name
                ));
                continue;
            }
            let handler_ty = i64_type.fn_type(&[i64_type.into()], false);
            let fn_val = module.add_function(&format!("__actor_{}", name), handler_ty, None);
            actor_infos.insert(name.clone(), ActorCodegenInfo { fn_val, param_name: param_name.clone() });
        }
    }

    let mut specs = Vec::new();
    for stmt in program {
        if let Stmt::FnDef { name, params, body, .. } = stmt {
            specs.push(FnSpec {
                name: name.clone(),
                params: params.iter().map(|p| (p.name.clone(), p.type_ann.clone())).collect(),
                body,
            });
        }
    }

    let mut fn_values: HashMap<String, FunctionValue> = HashMap::new();
    for spec in &specs {
        let param_types: Vec<_> = spec
            .params
            .iter()
            .map(|(_, ty)| if ty.as_deref() == Some("float") { f64_type.into() } else { i64_type.into() })
            .collect();
        let fn_type = i64_type.fn_type(&param_types, false);
        let fn_val = module.add_function(&spec.name, fn_type, None);
        fn_values.insert(spec.name.clone(), fn_val);
    }

    let mut compiled = 0;
    let mut skipped = skipped_structs;

    for spec in &specs {
        let fn_val = fn_values[&spec.name];
        match compile_function(&context, &module, &builder, &fn_values, printf_fn, depth_global, &struct_types, &enum_types, &closure_infos, malloc_fn, &actor_infos, spawn_fn, send_fn, shutdown_fn, spec, fn_val) {
            Ok(()) => compiled += 1,
            Err(e) => {
                skipped.push(format!("{} ({})", spec.name, e));
                unsafe { fn_val.delete() };
                fn_values.remove(&spec.name);
            }
        }
    }

    // Compile lifted closure bodies after ordinary functions.
    let mut compiled_closures = 0;
    for (idx, (_params, body)) in closure_literals.iter().enumerate() {
        let info = &closure_infos[&(*body as *const Expr)];
        match compile_closure_body(&context, &module, &builder, &fn_values, printf_fn, &struct_types, &enum_types, &closure_infos, malloc_fn, &actor_infos, spawn_fn, send_fn, info, body) {
            Ok(()) => compiled_closures += 1,
            Err(e) => {
                skipped.push(format!("__closure_{} ({})", idx, e));
                unsafe { info.fn_val.delete() };
            }
        }
    }
    if compiled_closures > 0 {
        println!("--- Compiled {} closure(s) ---", compiled_closures);
    }

    // Compile every actor's handler body into its lifted function.
    let mut compiled_actors = 0;
    for stmt in program {
        if let Stmt::Actor { name, body, .. } = stmt {
            if let Some(info) = actor_infos.get(name) {
                match compile_actor_handler(&context, &module, &builder, &fn_values, printf_fn, &struct_types, &enum_types, &closure_infos, malloc_fn, &actor_infos, spawn_fn, send_fn, info, body) {
                    Ok(()) => compiled_actors += 1,
                    Err(e) => {
                        skipped.push(format!("actor {} ({})", name, e));
                        unsafe { info.fn_val.delete() };
                    }
                }
            }
        }
    }
    if compiled_actors > 0 {
        println!("--- Compiled {} actor handler(s) ---", compiled_actors);
    }
    skipped.extend(skipped_actors);

    println!("--- LLVM IR (compiled {} function(s)) ---", compiled);
    println!("{}", module.print_to_string().to_string());

    if !skipped.is_empty() {
        println!("--- Skipped (unsupported in this codegen subset) ---");
        for s in &skipped {
            println!("  - {}", s);
        }
    }

    if let Err(e) = module.verify() {
        return Err(CodegenError(format!("module verification failed: {}", e)));
    }

    match target {
        CompileTarget::Jit => {
            if let Some(main_fn) = fn_values.get("main") {
                if main_fn.count_params() == 0 {
                    let engine = module
                        .create_jit_execution_engine(OptimizationLevel::None)
                        .map_err(|e| CodegenError(e.to_string()))?;

                    // Map JIT externs to host `tridentix_rt_*` function pointers.
                    if let Some(f) = module.get_function("tridentix_rt_spawn") {
                        engine.add_global_mapping(&f, tridentix_actor_rt::tridentix_rt_spawn as usize);
                    }
                    if let Some(f) = module.get_function("tridentix_rt_send") {
                        engine.add_global_mapping(&f, tridentix_actor_rt::tridentix_rt_send as usize);
                    }
                    if let Some(f) = module.get_function("tridentix_rt_shutdown_and_join") {
                        engine.add_global_mapping(&f, tridentix_actor_rt::tridentix_rt_shutdown_and_join as usize);
                    }

                    unsafe {
                        let jit_main: JitFunction<unsafe extern "C" fn() -> i64> = engine
                            .get_function("main")
                            .map_err(|e| CodegenError(e.to_string()))?;
                        println!("--- JIT execution ---");
                        let result = jit_main.call();
                        println!("main() => {}  (real machine code, executed via LLVM JIT)", result);
                    }
                }
            } else {
                println!("(no zero-argument int `main` found — skipping JIT execution)");
            }
        }
        CompileTarget::AheadOfTime { output_path, opt_passes } => {
            compile_to_binary(&module, &output_path, &opt_passes)?;
        }
    }

    Ok(())
}

/// AOT: optimize, emit an object file, and link with `cc`.
fn compile_to_binary(module: &Module, output_path: &str, opt_passes: &str) -> CResult<()> {
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError(format!("failed to initialize native target: {}", e)))?;

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| CodegenError(format!("unsupported target triple: {}", e)))?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let machine = target
        .create_target_machine(
            &triple,
            cpu.to_str().unwrap_or("generic"),
            features.to_str().unwrap_or(""),
            OptimizationLevel::Default,
            // PIC reloc so the object links into a PIE binary.
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError("failed to create target machine".into()))?;

    println!("--- Optimization pass pipeline: {} ---", opt_passes);
    let ir_before_len = module.print_to_string().to_string().len();
    module
        .run_passes(opt_passes, &machine, PassBuilderOptions::create())
        .map_err(|e| CodegenError(format!("optimization passes failed: {}", e)))?;
    let ir_after = module.print_to_string().to_string();
    println!(
        "IR size before optimization: {} chars, after: {} chars",
        ir_before_len,
        ir_after.len()
    );
    println!("--- Optimized LLVM IR ---");
    println!("{}", ir_after);

    let obj_path = format!("{}.o", output_path);
    machine
        .write_to_file(module, FileType::Object, std::path::Path::new(&obj_path))
        .map_err(|e| CodegenError(format!("failed to write object file: {}", e)))?;
    println!("Wrote object file: {}", obj_path);

    // Link with `cc` against `tridentix-runtime` (not libtridentix, which embeds LLVM).
    let current_exe = std::env::current_exe().map_err(|e| CodegenError(format!("couldn't locate running executable: {}", e)))?;
    let lib_dir = current_exe
        .parent()
        .ok_or_else(|| CodegenError("couldn't determine executable directory".into()))?;
    let candidates = [
        lib_dir.join("libtridentix_actor_rt.a"),
        lib_dir.join("tridentix_actor_rt.lib"),
        lib_dir.join("tridentix_actor_rt.dll.lib"),
        lib_dir.join("../../runtime/target/debug/libtridentix_actor_rt.a"),
        lib_dir.join("../../runtime/target/release/libtridentix_actor_rt.a"),
        lib_dir.join("../../target/debug/libtridentix_actor_rt.a"),
        lib_dir.join("../../target/release/libtridentix_actor_rt.a"),
        lib_dir.join("../../target/debug/tridentix_actor_rt.lib"),
        lib_dir.join("../../target/release/tridentix_actor_rt.lib"),
    ];
    let static_lib_path = candidates
        .iter()
        .find(|p| p.exists())
        .ok_or_else(|| {
            CodegenError(format!(
                "tridentix-runtime static lib not found (tried: {}) — build with `cargo build -p tridentix-runtime`",
                candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
            ))
        })?;

    let link_status = std::process::Command::new("cc")
        .arg(&obj_path)
        .arg(static_lib_path)
        // pthread/dl/m required by the actor runtime on Linux.
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-lm")
        .arg("-o")
        .arg(output_path)
        .status()
        .map_err(|e| CodegenError(format!("failed to invoke system linker 'cc': {}", e)))?;
    if !link_status.success() {
        return Err(CodegenError(format!("linking failed (cc exit status: {})", link_status)));
    }
    println!("Linked native executable: {}", output_path);

    Ok(())
}

fn declare_printf<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> FunctionValue<'ctx> {
    let i8_ptr_type = context.i8_type().ptr_type(AddressSpace::default());
    let printf_type = context.i32_type().fn_type(&[i8_ptr_type.into()], true);
    module.add_function("printf", printf_type, Some(Linkage::External))
}

/// Declare libc `exit(i32)` for runtime div-by-zero.
fn declare_exit<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> FunctionValue<'ctx> {
    let void_type = context.void_type();
    let exit_type = void_type.fn_type(&[context.i32_type().into()], false);
    module.add_function("exit", exit_type, Some(Linkage::External))
}

/// Shared i64 call-depth counter for compiled functions.
fn declare_call_depth_global<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> PointerValue<'ctx> {
    let i64_type = context.i64_type();
    let global = module.add_global(i64_type, None, "tridentix_call_depth");
    global.set_initializer(&i64_type.const_int(0, false));
    global.as_pointer_value()
}

/// Declare libc `malloc` for closure environment structs.
fn declare_malloc<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> FunctionValue<'ctx> {
    let i8_ptr_type = context.i8_type().ptr_type(AddressSpace::default());
    let malloc_type = i8_ptr_type.fn_type(&[context.i64_type().into()], false);
    module.add_function("malloc", malloc_type, Some(Linkage::External))
}

/// Declare `tridentix_rt_spawn` / `send` / `shutdown_and_join`.
fn declare_actor_runtime<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> (FunctionValue<'ctx>, FunctionValue<'ctx>, FunctionValue<'ctx>) {
    let i64_type = context.i64_type();
    let i8_ptr_type = context.i8_type().ptr_type(AddressSpace::default());
    let void_type = context.void_type();

    // extern "C" fn(i64) -> i64, passed BY POINTER to tridentix_rt_spawn.
    let handler_ty = i64_type.fn_type(&[i64_type.into()], false);
    let handler_ptr_ty = handler_ty.ptr_type(AddressSpace::default());

    let spawn_ty = i64_type.fn_type(&[handler_ptr_ty.into()], false);
    let spawn_fn = module.add_function("tridentix_rt_spawn", spawn_ty, Some(Linkage::External));

    let send_ty = void_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let send_fn = module.add_function("tridentix_rt_send", send_ty, Some(Linkage::External));

    let shutdown_ty = void_type.fn_type(&[], false);
    let shutdown_fn = module.add_function("tridentix_rt_shutdown_and_join", shutdown_ty, Some(Linkage::External));

    let _ = i8_ptr_type; // reserved: a future state-carrying ABI would use this
    (spawn_fn, send_fn, shutdown_fn)
}

/// Native lowering of one `actor` handler.
struct ActorCodegenInfo<'ctx> {
    fn_val: FunctionValue<'ctx>,
    param_name: String,
}

/// Collect `Expr::Closure` literals reachable from a function body.
fn collect_closures_in_stmts<'a>(stmts: &'a [Stmt], out: &mut Vec<(&'a Vec<crate::ast::Param>, &'a Expr)>) {
    for stmt in stmts {
        match stmt {
            Stmt::Let { value: Expr::Closure(params, body), .. } => {
                out.push((params, body));
                collect_closures_in_expr(body, out);
            }
            Stmt::Let { value, .. } => collect_closures_in_expr(value, out),
            Stmt::Assign { value, .. } => collect_closures_in_expr(value, out),
            Stmt::Return(Some(e)) => collect_closures_in_expr(e, out),
            Stmt::ExprStmt(e) => collect_closures_in_expr(e, out),
            Stmt::If { cond, then_block, elif_blocks, else_block } => {
                collect_closures_in_expr(cond, out);
                collect_closures_in_stmts(then_block, out);
                for (c, b) in elif_blocks {
                    collect_closures_in_expr(c, out);
                    collect_closures_in_stmts(b, out);
                }
                if let Some(b) = else_block {
                    collect_closures_in_stmts(b, out);
                }
            }
            Stmt::Loop { start, end, body, .. } => {
                collect_closures_in_expr(start, out);
                collect_closures_in_expr(end, out);
                collect_closures_in_stmts(body, out);
            }
            Stmt::While { cond, body } => {
                collect_closures_in_expr(cond, out);
                collect_closures_in_stmts(body, out);
            }
            _ => {}
        }
    }
}

fn collect_closures_in_expr<'a>(expr: &'a Expr, out: &mut Vec<(&'a Vec<crate::ast::Param>, &'a Expr)>) {
    match expr {
        Expr::Closure(params, body) => {
            out.push((params, body));
            collect_closures_in_expr(body, out);
        }
        Expr::Binary(l, _, r) => {
            collect_closures_in_expr(l, out);
            collect_closures_in_expr(r, out);
        }
        Expr::Unary(_, e) | Expr::Await(e) => collect_closures_in_expr(e, out),
        Expr::Call(_, args) | Expr::Spawn(_, args) => {
            for a in args {
                collect_closures_in_expr(a, out);
            }
        }
        _ => {}
    }
}

/// Idents in a closure body that are not its parameters.
fn collect_free_vars(expr: &Expr, params: &[String], out: &mut std::collections::BTreeSet<String>) {
    match expr {
        Expr::Ident(name) => {
            if !params.contains(name) {
                out.insert(name.clone());
            }
        }
        Expr::Binary(l, _, r) => {
            collect_free_vars(l, params, out);
            collect_free_vars(r, params, out);
        }
        Expr::Unary(_, e) | Expr::Await(e) => collect_free_vars(e, params, out),
        Expr::Call(_, args) | Expr::Spawn(_, args) => {
            for a in args {
                collect_free_vars(a, params, out);
            }
        }
        Expr::FieldAccess(base, _) => collect_free_vars(base, params, out),
        Expr::Index(base, idx) => {
            collect_free_vars(base, params, out);
            collect_free_vars(idx, params, out);
        }
        Expr::EnumLit(_, _, args) => {
            for a in args {
                collect_free_vars(a, params, out);
            }
        }
        Expr::StructLit(_, fields) => {
            for (_, e) in fields {
                collect_free_vars(e, params, out);
            }
        }
        Expr::ListLit(items) => {
            for e in items {
                collect_free_vars(e, params, out);
            }
        }
        Expr::Closure(inner_params, inner_body) => {
            let mut combined: Vec<String> = params.to_vec();
            combined.extend(inner_params.iter().map(|p| p.name.clone()));
            collect_free_vars(inner_body, &combined, out);
        }
        _ => {}
    }
}

fn compile_function<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    depth_global: PointerValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    shutdown_fn: FunctionValue<'ctx>,
    spec: &FnSpec,
    fn_val: FunctionValue<'ctx>,
) -> CResult<()> {
    let entry = context.append_basic_block(fn_val, "entry");
    builder.position_at_end(entry);

    // Check-then-increment the shared recursion-depth counter.
    let i64_type = context.i64_type();
    let cur_depth = builder.build_load(i64_type, depth_global, "depth_load").unwrap().into_int_value();
    let max_depth = i64_type.const_int(MAX_NATIVE_CALL_DEPTH as u64, false);
    let exceeded = builder.build_int_compare(IntPredicate::SGT, cur_depth, max_depth, "depth_exceeded").unwrap();

    let depth_err_bb = context.append_basic_block(fn_val, "stack_overflow_guard");
    let depth_ok_bb = context.append_basic_block(fn_val, "depth_ok");
    builder.build_conditional_branch(exceeded, depth_err_bb, depth_ok_bb).unwrap();

    builder.position_at_end(depth_err_bb);
    let exit_fn = module.get_function("exit").expect("exit() declared in compile_and_run");
    let msg = builder
        .build_global_string_ptr("Runtime error: stack overflow (recursion too deep)\n", "stackoverflow_msg")
        .unwrap();
    builder.build_call(printf_fn, &[msg.as_pointer_value().into()], "print_overflow_err").unwrap();
    let exit_code = context.i32_type().const_int(1, false);
    builder.build_call(exit_fn, &[exit_code.into()], "exit_call").unwrap();
    builder.build_unreachable().unwrap();

    builder.position_at_end(depth_ok_bb);
    let one = i64_type.const_int(1, false);
    let incremented = builder.build_int_add(cur_depth, one, "depth_inc").unwrap();
    builder.build_store(depth_global, incremented).unwrap();

    let mut vars: HashMap<String, VarSlot<'ctx>> = HashMap::new();
    for (i, (pname, ty)) in spec.params.iter().enumerate() {
        let is_float = ty.as_deref() == Some("float");
        let param = fn_val.get_nth_param(i as u32).ok_or_else(|| CodegenError("missing parameter".into()))?;
        if is_float {
            let slot = builder.build_alloca(context.f64_type(), pname).unwrap();
            builder.build_store(slot, param.into_float_value()).unwrap();
            vars.insert(pname.clone(), VarSlot { ptr: slot, kind: SlotKind::Float });
        } else {
            let slot = builder.build_alloca(context.i64_type(), pname).unwrap();
            builder.build_store(slot, param.into_int_value()).unwrap();
            vars.insert(pname.clone(), VarSlot { ptr: slot, kind: SlotKind::Int });
        }
    }

    let returned = compile_block(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, spec.body, fn_val, &mut vars)?;
    if !returned {
        // Join actors on implicit `main` return (explicit `return` skips this).
        if spec.name == "main" && !actor_infos.is_empty() {
            builder.build_call(shutdown_fn, &[], "actor_shutdown").unwrap();
        }
        let zero = context.i64_type().const_int(0, false);
        emit_return(context, builder, depth_global, zero);
    }
    Ok(())
}

/// Decrement the shared depth counter, then return.
fn emit_return<'ctx>(context: &'ctx Context, builder: &Builder<'ctx>, depth_global: PointerValue<'ctx>, value: IntValue<'ctx>) {
    let i64_type = context.i64_type();
    let cur = builder.build_load(i64_type, depth_global, "depth_load_ret").unwrap().into_int_value();
    let one = i64_type.const_int(1, false);
    let decremented = builder.build_int_sub(cur, one, "depth_dec").unwrap();
    builder.build_store(depth_global, decremented).unwrap();
    builder.build_return(Some(&value)).unwrap();
}

fn compile_block<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    depth_global: PointerValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    body: &[Stmt],
    fn_val: FunctionValue<'ctx>,
    vars: &mut HashMap<String, VarSlot<'ctx>>,
) -> CResult<bool> {
    for stmt in body {
        match stmt {
            Stmt::Let { name, value, .. } => {
                if let Expr::StrLit(s) = value {
                    let g = builder.build_global_string_ptr(s, "strlit").unwrap();
                    let i8_ptr_ty = context.i8_type().ptr_type(AddressSpace::default());
                    let slot = builder.build_alloca(i8_ptr_ty, name).unwrap();
                    builder.build_store(slot, g.as_pointer_value()).unwrap();
                    vars.insert(name.clone(), VarSlot { ptr: slot, kind: SlotKind::Str });
                } else if let Expr::StructLit(type_name, fields) = value {
                    let info = struct_types.get(type_name).ok_or_else(|| {
                        CodegenError(format!(
                            "struct '{}' isn't supported in the codegen subset (all fields must be int/float)",
                            type_name
                        ))
                    })?;
                    let alloca = builder.build_alloca(info.llvm_type, name).unwrap();
                    for (fname, fexpr) in fields {
                        let idx = info.field_order.iter().position(|f| f == fname).ok_or_else(|| {
                            CodegenError(format!("struct '{}' has no field '{}'", type_name, fname))
                        })?;
                        let field_ptr = builder
                            .build_struct_gep(info.llvm_type, alloca, idx as u32, fname)
                            .map_err(|_| CodegenError(format!("GEP failed for field '{}'", fname)))?;
                        let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, fexpr, vars)?;
                        match (&info.field_kinds[idx], v) {
                            (SlotKind::Int, CgValue::Int(iv)) => {
                                builder.build_store(field_ptr, iv).unwrap();
                            }
                            (SlotKind::Float, CgValue::Float(fv)) => {
                                builder.build_store(field_ptr, fv).unwrap();
                            }
                            (SlotKind::Int, CgValue::Float(fv)) => {
                                let truncated = builder.build_float_to_signed_int(fv, context.i64_type(), "trunc").unwrap();
                                builder.build_store(field_ptr, truncated).unwrap();
                            }
                            (SlotKind::Float, CgValue::Int(iv)) => {
                                let promoted = builder.build_signed_int_to_float(iv, context.f64_type(), "promo").unwrap();
                                builder.build_store(field_ptr, promoted).unwrap();
                            }
                            _ => return Err(CodegenError(format!("type mismatch initializing field '{}'", fname))),
                        }
                    }
                    vars.insert(name.clone(), VarSlot { ptr: alloca, kind: SlotKind::Struct(type_name.clone()) });
                } else if let Expr::EnumLit(type_name, variant, args) = value {
                    let info = enum_types.get(type_name).ok_or_else(|| {
                        CodegenError(format!("enum '{}' isn't supported in the codegen subset", type_name))
                    })?;
                    let tag = *info.variant_tags.get(variant).ok_or_else(|| {
                        CodegenError(format!("enum '{}' has no variant '{}'", type_name, variant))
                    })?;
                    if args.len() > 1 {
                        return Err(CodegenError(format!(
                            "enum variant '{}.{}' has {} payload values — the codegen subset supports at most 1 (see EnumTypeInfo's doc comment)",
                            type_name, variant, args.len()
                        )));
                    }

                    let alloca = builder.build_alloca(info.llvm_type, name).unwrap();
                    let tag_ptr = builder
                        .build_struct_gep(info.llvm_type, alloca, 0, "tag_ptr")
                        .map_err(|_| CodegenError("GEP failed for enum tag".into()))?;
                    builder.build_store(tag_ptr, context.i64_type().const_int(tag as u64, false)).unwrap();

                    let payload_ptr = builder
                        .build_struct_gep(info.llvm_type, alloca, 1, "payload_ptr")
                        .map_err(|_| CodegenError("GEP failed for enum payload".into()))?;
                    let payload_val = if let Some(arg_expr) = args.first() {
                        match compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, arg_expr, vars)? {
                            CgValue::Int(iv) => iv,
                            CgValue::Float(fv) => builder.build_float_to_signed_int(fv, context.i64_type(), "trunc").unwrap(),
                        }
                    } else {
                        context.i64_type().const_int(0, false)
                    };
                    builder.build_store(payload_ptr, payload_val).unwrap();

                    vars.insert(name.clone(), VarSlot { ptr: alloca, kind: SlotKind::Enum(type_name.clone()) });
                } else if let Expr::Closure(_params, body) = value {
                    let info = closure_infos
                        .get(&(body.as_ref() as *const Expr))
                        .ok_or_else(|| CodegenError("internal error: closure literal not pre-scanned".into()))?;

                    // Heap-allocate the capture environment.
                    let env_size = context.i64_type().const_int((info.free_vars.len() * 8) as u64, false);
                    let env_raw = builder
                        .build_call(malloc_fn, &[env_size.into()], "env_alloc")
                        .unwrap()
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError("malloc call produced no value".into()))?
                        .into_pointer_value();

                    // Store each captured free variable into the env struct.
                    for (i, fv_name) in info.free_vars.iter().enumerate() {
                        let fv_slot = vars.get(fv_name).ok_or_else(|| {
                            CodegenError(format!("closure captures undefined variable '{}'", fv_name))
                        })?;
                        let captured_val = match &fv_slot.kind {
                            SlotKind::Int => builder.build_load(context.i64_type(), fv_slot.ptr, fv_name).unwrap().into_int_value(),
                            SlotKind::Float => {
                                let fv = builder.build_load(context.f64_type(), fv_slot.ptr, fv_name).unwrap().into_float_value();
                                builder.build_float_to_signed_int(fv, context.i64_type(), "trunc").unwrap()
                            }
                            other => {
                                return Err(CodegenError(format!(
                                    "closure capture of '{}' ({:?}-kind) isn't supported in the codegen subset — only int/float free variables can be captured",
                                    fv_name,
                                    std::mem::discriminant(other)
                                )))
                            }
                        };
                        let field_ptr = builder
                            .build_struct_gep(info.env_struct_ty, env_raw, i as u32, fv_name)
                            .map_err(|_| CodegenError(format!("GEP failed for captured var '{}'", fv_name)))?;
                        builder.build_store(field_ptr, captured_val).unwrap();
                    }

                    // Closure value is the env pointer; the function is known statically.
                    let closure_slot = builder.build_alloca(context.i8_type().ptr_type(AddressSpace::default()), name).unwrap();
                    builder.build_store(closure_slot, env_raw).unwrap();
                    vars.insert(
                        name.clone(),
                        VarSlot { ptr: closure_slot, kind: SlotKind::Closure(info.fn_val, info.param_is_float.clone()) },
                    );
                } else {
                    let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, value, vars)?;
                    match v {
                        CgValue::Int(iv) => {
                            let slot = builder.build_alloca(context.i64_type(), name).unwrap();
                            builder.build_store(slot, iv).unwrap();
                            vars.insert(name.clone(), VarSlot { ptr: slot, kind: SlotKind::Int });
                        }
                        CgValue::Float(fv) => {
                            let slot = builder.build_alloca(context.f64_type(), name).unwrap();
                            builder.build_store(slot, fv).unwrap();
                            vars.insert(name.clone(), VarSlot { ptr: slot, kind: SlotKind::Float });
                        }
                    }
                }
            }
            Stmt::Assign { name, value } => {
                let slot = vars.get(name).cloned().ok_or_else(|| CodegenError(format!("assignment to undeclared variable '{}'", name)))?;
                if let Expr::StrLit(s) = value {
                    if slot.kind != SlotKind::Str {
                        return Err(CodegenError(format!("cannot assign a string to non-string variable '{}'", name)));
                    }
                    let g = builder.build_global_string_ptr(s, "strlit").unwrap();
                    builder.build_store(slot.ptr, g.as_pointer_value()).unwrap();
                } else {
                    let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, value, vars)?;
                    match (v, slot.kind) {
                        (CgValue::Int(iv), SlotKind::Int) => { builder.build_store(slot.ptr, iv).unwrap(); }
                        (CgValue::Float(fv), SlotKind::Float) => { builder.build_store(slot.ptr, fv).unwrap(); }
                        (CgValue::Int(iv), SlotKind::Float) => {
                            let promoted = builder.build_signed_int_to_float(iv, context.f64_type(), "promo").unwrap();
                            builder.build_store(slot.ptr, promoted).unwrap();
                        }
                        (CgValue::Float(fv), SlotKind::Int) => {
                            let truncated = builder.build_float_to_signed_int(fv, context.i64_type(), "trunc").unwrap();
                            builder.build_store(slot.ptr, truncated).unwrap();
                        }
                        (_, SlotKind::Str) => {
                            return Err(CodegenError(format!("cannot assign a number to string variable '{}'", name)))
                        }
                        (_, SlotKind::Struct(_)) => {
                            return Err(CodegenError(format!(
                                "reassigning struct variable '{}' isn't supported in the codegen subset yet (construct a new struct literal instead)",
                                name
                            )))
                        }
                        (_, SlotKind::Enum(_)) => {
                            return Err(CodegenError(format!(
                                "reassigning enum variable '{}' isn't supported in the codegen subset yet (construct a new enum literal instead)",
                                name
                            )))
                        }
                        (_, SlotKind::Closure(_, _)) => {
                            return Err(CodegenError(format!(
                                "reassigning closure variable '{}' isn't supported in the codegen subset yet",
                                name
                            )))
                        }
                    }
                }
            }
            Stmt::Return(val) => {
                let v = match val {
                    Some(e) => compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, e, vars)?,
                    None => CgValue::Int(context.i64_type().const_int(0, false)),
                };
                let as_i64 = match v {
                    CgValue::Int(iv) => iv,
                    CgValue::Float(fv) => builder.build_float_to_signed_int(fv, context.i64_type(), "ftoi").unwrap(),
                };
                emit_return(context, builder, depth_global, as_i64);
                return Ok(true);
            }
            Stmt::If { cond, then_block, elif_blocks, else_block } => {
                let returned = compile_if_chain(
                    context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, cond, then_block, elif_blocks, else_block.as_deref(),
                    fn_val, vars,
                )?;
                if returned {
                    return Ok(true);
                }
            }
            Stmt::Loop { var, start, end, body } => {
                compile_for_loop(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, var, start, end, body, fn_val, vars)?;
            }
            Stmt::While { cond, body } => {
                compile_while_loop(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, cond, body, fn_val, vars)?;
            }
            Stmt::ExprStmt(Expr::Call(name, args)) if name == "print" => {
                compile_print(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, args, vars)?;
            }
            Stmt::ExprStmt(e) => {
                compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, e, vars)?;
            }
            Stmt::Send { target, message } => {
                let target_val = expect_int(
                    compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, target, vars)?,
                    "send() target (actor handle)",
                )?;
                let msg_val = expect_int(
                    compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, message, vars)?,
                    "send() message (codegen subset: int messages only)",
                )?;
                builder.build_call(send_fn, &[target_val.into(), msg_val.into()], "send_call").unwrap();
            }
            _ => return Err(CodegenError(format!("statement not supported in codegen subset: {:?}", stmt))),
        }
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn compile_if_chain<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    depth_global: PointerValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    cond: &Expr,
    then_block: &[Stmt],
    elif_blocks: &[(Expr, Vec<Stmt>)],
    else_block: Option<&[Stmt]>,
    fn_val: FunctionValue<'ctx>,
    vars: &mut HashMap<String, VarSlot<'ctx>>,
) -> CResult<bool> {
    let cond_val = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, cond, vars)?;
    let cond_bool = as_bool_i1(context, builder, cond_val);

    let then_bb = context.append_basic_block(fn_val, "then");
    let else_bb = context.append_basic_block(fn_val, "else");
    let merge_bb = context.append_basic_block(fn_val, "merge");

    builder.build_conditional_branch(cond_bool, then_bb, else_bb).unwrap();

    builder.position_at_end(then_bb);
    let then_returned = compile_block(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, then_block, fn_val, vars)?;
    if !then_returned {
        builder.build_unconditional_branch(merge_bb).unwrap();
    }

    builder.position_at_end(else_bb);
    let else_returned = if !elif_blocks.is_empty() {
        let (next_cond, next_then) = &elif_blocks[0];
        compile_if_chain(
                    context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, next_cond, next_then, &elif_blocks[1..], else_block, fn_val, vars)?
    } else {
        match else_block {
            Some(b) => compile_block(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, b, fn_val, vars)?,
            None => false,
        }
    };
    if !else_returned {
        builder.build_unconditional_branch(merge_bb).unwrap();
    }

    builder.position_at_end(merge_bb);
    if then_returned && else_returned {
        let zero = context.i64_type().const_int(0, false);
        emit_return(context, builder, depth_global, zero);
        return Ok(true);
    }
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn compile_for_loop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    depth_global: PointerValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    var: &str,
    start: &Expr,
    end: &Expr,
    body: &[Stmt],
    fn_val: FunctionValue<'ctx>,
    vars: &mut HashMap<String, VarSlot<'ctx>>,
) -> CResult<()> {
    let i64_type = context.i64_type();
    let start_val = expect_int(compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, start, vars)?, "loop start")?;
    let end_val = expect_int(compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, end, vars)?, "loop end")?;

    let var_slot = builder.build_alloca(i64_type, var).unwrap();
    builder.build_store(var_slot, start_val).unwrap();
    vars.insert(var.to_string(), VarSlot { ptr: var_slot, kind: SlotKind::Int });

    let cond_bb = context.append_basic_block(fn_val, "loop_cond");
    let body_bb = context.append_basic_block(fn_val, "loop_body");
    let after_bb = context.append_basic_block(fn_val, "loop_end");

    builder.build_unconditional_branch(cond_bb).unwrap();

    builder.position_at_end(cond_bb);
    let current = builder.build_load(i64_type, var_slot, var).unwrap().into_int_value();
    let keep_going = builder.build_int_compare(IntPredicate::SLT, current, end_val, "loopcond").unwrap();
    builder.build_conditional_branch(keep_going, body_bb, after_bb).unwrap();

    builder.position_at_end(body_bb);
    let body_returned = compile_block(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, body, fn_val, vars)?;
    if !body_returned {
        let cur = builder.build_load(i64_type, var_slot, var).unwrap().into_int_value();
        let one = i64_type.const_int(1, false);
        let next = builder.build_int_add(cur, one, "loopnext").unwrap();
        builder.build_store(var_slot, next).unwrap();
        builder.build_unconditional_branch(cond_bb).unwrap();
    }

    builder.position_at_end(after_bb);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compile_while_loop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    depth_global: PointerValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    cond: &Expr,
    body: &[Stmt],
    fn_val: FunctionValue<'ctx>,
    vars: &mut HashMap<String, VarSlot<'ctx>>,
) -> CResult<()> {
    let cond_bb = context.append_basic_block(fn_val, "while_cond");
    let body_bb = context.append_basic_block(fn_val, "while_body");
    let after_bb = context.append_basic_block(fn_val, "while_end");

    builder.build_unconditional_branch(cond_bb).unwrap();

    builder.position_at_end(cond_bb);
    let cond_val = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, cond, vars)?;
    let keep_going = as_bool_i1(context, builder, cond_val);
    builder.build_conditional_branch(keep_going, body_bb, after_bb).unwrap();

    builder.position_at_end(body_bb);
    let body_returned = compile_block(context, module, builder, fn_values, printf_fn, depth_global, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, body, fn_val, vars)?;
    if !body_returned {
        builder.build_unconditional_branch(cond_bb).unwrap();
    }

    builder.position_at_end(after_bb);
    Ok(())
}

fn compile_print<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    args: &[Expr],
    vars: &HashMap<String, VarSlot<'ctx>>,
) -> CResult<()> {
    if args.len() != 1 {
        return Err(CodegenError("print() in the codegen subset supports exactly one argument".into()));
    }

    let str_ptr: Option<PointerValue<'ctx>> = match &args[0] {
        Expr::StrLit(s) => Some(builder.build_global_string_ptr(s, "fmt_str_lit").unwrap().as_pointer_value()),
        Expr::Ident(name) => match vars.get(name) {
            Some(slot) if slot.kind == SlotKind::Str => {
                let i8_ptr_ty = context.i8_type().ptr_type(AddressSpace::default());
                Some(builder.build_load(i8_ptr_ty, slot.ptr, name).unwrap().into_pointer_value())
            }
            _ => None,
        },
        _ => None,
    };

    if let Some(sptr) = str_ptr {
        let fmt = builder.build_global_string_ptr("%s\n", "fmt_str").unwrap();
        builder.build_call(printf_fn, &[fmt.as_pointer_value().into(), sptr.into()], "printf_call").unwrap();
        return Ok(());
    }

    let val = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, &args[0], vars)?;
    match val {
        CgValue::Int(iv) => {
            let fmt = builder.build_global_string_ptr("%lld\n", "fmt_int").unwrap();
            builder.build_call(printf_fn, &[fmt.as_pointer_value().into(), iv.into()], "printf_call").unwrap();
        }
        CgValue::Float(fv) => {
            let fmt = builder.build_global_string_ptr("%f\n", "fmt_float").unwrap();
            builder.build_call(printf_fn, &[fmt.as_pointer_value().into(), fv.into()], "printf_call").unwrap();
        }
    }
    Ok(())
}

fn expect_int<'ctx>(v: CgValue<'ctx>, context_msg: &str) -> CResult<IntValue<'ctx>> {
    match v {
        CgValue::Int(i) => Ok(i),
        CgValue::Float(_) => Err(CodegenError(format!("{} must be an int expression, got float", context_msg))),
    }
}

fn as_bool_i1<'ctx>(context: &'ctx Context, builder: &Builder<'ctx>, v: CgValue<'ctx>) -> IntValue<'ctx> {
    match v {
        CgValue::Int(iv) => {
            let zero = context.i64_type().const_int(0, false);
            builder.build_int_compare(IntPredicate::NE, iv, zero, "boolcheck").unwrap()
        }
        CgValue::Float(fv) => {
            let zero = context.f64_type().const_float(0.0);
            builder.build_float_compare(FloatPredicate::ONE, fv, zero, "boolcheck").unwrap()
        }
    }
}

fn compile_expr<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    expr: &Expr,
    vars: &HashMap<String, VarSlot<'ctx>>,
) -> CResult<CgValue<'ctx>> {
    match expr {
        Expr::IntLit(n) => Ok(CgValue::Int(context.i64_type().const_int(*n as u64, true))),
        Expr::FloatLit(f) => Ok(CgValue::Float(context.f64_type().const_float(*f))),
        Expr::BoolLit(b) => Ok(CgValue::Int(context.i64_type().const_int(*b as u64, false))),
        Expr::StrLit(_) => Err(CodegenError("string literal used in a numeric context — only print(<string>) is supported".into())),
        Expr::Ident(name) => {
            let slot = vars.get(name).ok_or_else(|| CodegenError(format!("undefined variable '{}' in codegen", name)))?;
            match &slot.kind {
                SlotKind::Int => Ok(CgValue::Int(builder.build_load(context.i64_type(), slot.ptr, name).unwrap().into_int_value())),
                SlotKind::Float => Ok(CgValue::Float(builder.build_load(context.f64_type(), slot.ptr, name).unwrap().into_float_value())),
                SlotKind::Str => Err(CodegenError(format!("string variable '{}' used in a numeric context", name))),
                SlotKind::Struct(type_name) => Err(CodegenError(format!(
                    "struct variable '{}' (type {}) used directly in a numeric context — access a field with '.' instead",
                    name, type_name
                ))),
                SlotKind::Enum(type_name) => Err(CodegenError(format!(
                    "enum variable '{}' (type {}) used directly in a numeric context — use match/EnumIsVariant/EnumPayload instead",
                    name, type_name
                ))),
                SlotKind::Closure(..) => Err(CodegenError(format!(
                    "closure variable '{}' used directly in a numeric context — call it instead, e.g. '{}(...)'",
                    name, name
                ))),
            }
        }
        Expr::StructLit(type_name, fields) => {
            // Stack-allocate the struct and store fields via GEP.
            let info = struct_types.get(type_name).ok_or_else(|| {
                CodegenError(format!(
                    "struct '{}' isn't supported in the codegen subset (see skipped-structs list, or it wasn't defined)",
                    type_name
                ))
            })?;
            let alloca = builder.build_alloca(info.llvm_type, type_name).unwrap();
            for (fname, fexpr) in fields {
                let idx = info.field_order.iter().position(|f| f == fname).ok_or_else(|| {
                    CodegenError(format!("struct '{}' has no field '{}'", type_name, fname))
                })?;
                let field_ptr = builder
                    .build_struct_gep(info.llvm_type, alloca, idx as u32, fname)
                    .map_err(|_| CodegenError(format!("GEP failed for field '{}'", fname)))?;
                let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, fexpr, vars)?;
                match (&info.field_kinds[idx], v) {
                    (SlotKind::Int, CgValue::Int(iv)) => {
                        builder.build_store(field_ptr, iv).unwrap();
                    }
                    (SlotKind::Float, CgValue::Float(fv)) => {
                        builder.build_store(field_ptr, fv).unwrap();
                    }
                    (SlotKind::Int, CgValue::Float(fv)) => {
                        let truncated = builder.build_float_to_signed_int(fv, context.i64_type(), "trunc").unwrap();
                        builder.build_store(field_ptr, truncated).unwrap();
                    }
                    (SlotKind::Float, CgValue::Int(iv)) => {
                        let promoted = builder.build_signed_int_to_float(iv, context.f64_type(), "promo").unwrap();
                        builder.build_store(field_ptr, promoted).unwrap();
                    }
                    _ => return Err(CodegenError(format!("type mismatch initializing field '{}'", fname))),
                }
            }
            // Struct literals are only lowered as the RHS of `let`.
            Err(CodegenError(
                "struct literals are only supported as the direct value of a `let` binding in the codegen subset".into(),
            ))
        }
        Expr::FieldAccess(base, field) => {
            // Only `ident.field` is supported (no computed bases or chains).
            let base_name = match base.as_ref() {
                Expr::Ident(n) => n,
                _ => {
                    return Err(CodegenError(
                        "field access on a non-identifier expression isn't supported in the codegen subset yet".into(),
                    ))
                }
            };
            let slot = vars
                .get(base_name)
                .ok_or_else(|| CodegenError(format!("undefined variable '{}' in codegen", base_name)))?;
            let type_name = match &slot.kind {
                SlotKind::Struct(t) => t.clone(),
                _ => return Err(CodegenError(format!("'{}' is not a struct in the codegen subset", base_name))),
            };
            let info = struct_types
                .get(&type_name)
                .ok_or_else(|| CodegenError(format!("internal error: struct type '{}' not registered", type_name)))?;
            let idx = info
                .field_order
                .iter()
                .position(|f| f == field)
                .ok_or_else(|| CodegenError(format!("struct '{}' has no field '{}'", type_name, field)))?;
            let field_ptr = builder
                .build_struct_gep(info.llvm_type, slot.ptr, idx as u32, field)
                .map_err(|_| CodegenError(format!("GEP failed for field '{}'", field)))?;
            match &info.field_kinds[idx] {
                SlotKind::Int => Ok(CgValue::Int(builder.build_load(context.i64_type(), field_ptr, field).unwrap().into_int_value())),
                SlotKind::Float => Ok(CgValue::Float(builder.build_load(context.f64_type(), field_ptr, field).unwrap().into_float_value())),
                _ => unreachable!("struct fields are always Int or Float in this codegen subset"),
            }
        }
        Expr::Unary(op, e) => {
            let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, e, vars)?;
            match op {
                UnOp::Neg => Ok(match v {
                    CgValue::Int(iv) => CgValue::Int(builder.build_int_neg(iv, "negtmp").unwrap()),
                    CgValue::Float(fv) => CgValue::Float(builder.build_float_neg(fv, "fnegtmp").unwrap()),
                }),
                UnOp::Not => {
                    let as_bool = as_bool_i1(context, builder, v);
                    let inverted = builder.build_not(as_bool, "nottmp").unwrap();
                    Ok(CgValue::Int(builder.build_int_z_extend(inverted, context.i64_type(), "notz").unwrap()))
                }
            }
        }
        Expr::Binary(lhs, op, rhs) => {
            let l = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, lhs, vars)?;
            let r = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, rhs, vars)?;
            compile_binop(context, module, builder, printf_fn, l, op, r)
        }
        Expr::Call(name, _) if name == "print" => {
            Err(CodegenError("print() used as an expression isn't supported in the codegen subset".into()))
        }
        Expr::Call(name, args) if matches!(vars.get(name).map(|s| &s.kind), Some(SlotKind::Closure(..))) => {
            let slot = vars.get(name).unwrap();
            let (closure_fn, param_is_float) = match &slot.kind {
                SlotKind::Closure(f, p) => (*f, p.clone()),
                _ => unreachable!("matched above"),
            };
            if args.len() != param_is_float.len() {
                return Err(CodegenError(format!(
                    "closure '{}' expects {} argument(s), got {}",
                    name, param_is_float.len(), args.len()
                )));
            }
            let i8_ptr_ty = context.i8_type().ptr_type(AddressSpace::default());
            let env_ptr = builder.build_load(i8_ptr_ty, slot.ptr, name).unwrap().into_pointer_value();

            let mut compiled_args: Vec<inkwell::values::BasicMetadataValueEnum> = vec![env_ptr.into()];
            for (a, expects_float) in args.iter().zip(param_is_float.iter()) {
                let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, a, vars)?;
                let coerced = match (v, *expects_float) {
                    (CgValue::Int(iv), true) => builder.build_signed_int_to_float(iv, context.f64_type(), "argf").unwrap().into(),
                    (CgValue::Float(fv), false) => builder.build_float_to_signed_int(fv, context.i64_type(), "argi").unwrap().into(),
                    (CgValue::Int(iv), false) => iv.into(),
                    (CgValue::Float(fv), true) => fv.into(),
                };
                compiled_args.push(coerced);
            }
            let call = builder.build_call(closure_fn, &compiled_args, "closure_call").unwrap();
            call.try_as_basic_value()
                .basic()
                .map(|v| CgValue::Int(v.into_int_value()))
                .ok_or_else(|| CodegenError(format!("closure '{}' has no return value", name)))
        }
        Expr::Call(name, args) => {
            let fn_val = *fn_values.get(name).ok_or_else(|| CodegenError(format!("call to unsupported/undefined function '{}'", name)))?;
            let mut compiled_args = Vec::new();
            for (i, a) in args.iter().enumerate() {
                let v = compile_expr(context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, a, vars)?;
                let expects_float = fn_val.get_nth_param(i as u32).map(|p| p.is_float_value()).unwrap_or(false);
                let coerced = match (v, expects_float) {
                    (CgValue::Int(iv), true) => builder.build_signed_int_to_float(iv, context.f64_type(), "argf").unwrap().into(),
                    (CgValue::Float(fv), false) => builder.build_float_to_signed_int(fv, context.i64_type(), "argi").unwrap().into(),
                    (CgValue::Int(iv), false) => iv.into(),
                    (CgValue::Float(fv), true) => fv.into(),
                };
                compiled_args.push(coerced);
            }
            let call = builder.build_call(fn_val, &compiled_args, "calltmp").unwrap();
            call.try_as_basic_value()
                .basic()
                .map(|v| CgValue::Int(v.into_int_value()))
                .ok_or_else(|| CodegenError(format!("function '{}' has no return value", name)))
        }
        Expr::EnumIsVariant(subject, type_name, variant) => {
            let base_name = match subject.as_ref() {
                Expr::Ident(n) => n,
                _ => {
                    return Err(CodegenError(
                        "match subject must be a plain variable in the codegen subset (e.g. `match shape:`, not `match get_shape():`)".into(),
                    ))
                }
            };
            let slot = vars.get(base_name).ok_or_else(|| CodegenError(format!("undefined variable '{}' in codegen", base_name)))?;
            match &slot.kind {
                SlotKind::Enum(actual_type) if actual_type == type_name => {}
                SlotKind::Enum(actual_type) => {
                    return Err(CodegenError(format!(
                        "'{}' is enum type '{}', but this pattern checks for '{}'",
                        base_name, actual_type, type_name
                    )))
                }
                _ => return Err(CodegenError(format!("'{}' is not an enum in the codegen subset", base_name))),
            }
            let info = enum_types
                .get(type_name)
                .ok_or_else(|| CodegenError(format!("internal error: enum type '{}' not registered", type_name)))?;
            let tag = *info
                .variant_tags
                .get(variant)
                .ok_or_else(|| CodegenError(format!("enum '{}' has no variant '{}'", type_name, variant)))?;
            let tag_ptr = builder
                .build_struct_gep(info.llvm_type, slot.ptr, 0, "tag_ptr")
                .map_err(|_| CodegenError("GEP failed for enum tag".into()))?;
            let actual_tag = builder.build_load(context.i64_type(), tag_ptr, "tag_load").unwrap().into_int_value();
            let expected_tag = context.i64_type().const_int(tag as u64, false);
            let is_match = builder.build_int_compare(IntPredicate::EQ, actual_tag, expected_tag, "variant_check").unwrap();
            Ok(CgValue::Int(zext(context, builder, is_match)))
        }
        Expr::EnumPayload(subject, type_name, _variant, _idx) => {
            let base_name = match subject.as_ref() {
                Expr::Ident(n) => n,
                _ => {
                    return Err(CodegenError(
                        "match subject must be a plain variable in the codegen subset".into(),
                    ))
                }
            };
            let slot = vars.get(base_name).ok_or_else(|| CodegenError(format!("undefined variable '{}' in codegen", base_name)))?;
            let info = enum_types
                .get(type_name)
                .ok_or_else(|| CodegenError(format!("internal error: enum type '{}' not registered", type_name)))?;
            // Single payload slot at struct index 1.
            let payload_ptr = builder
                .build_struct_gep(info.llvm_type, slot.ptr, 1, "payload_ptr")
                .map_err(|_| CodegenError("GEP failed for enum payload".into()))?;
            Ok(CgValue::Int(builder.build_load(context.i64_type(), payload_ptr, "payload_load").unwrap().into_int_value()))
        }
        Expr::Spawn(actor_name, _args) => {
            let info = actor_infos.get(actor_name).ok_or_else(|| {
                CodegenError(format!(
                    "actor '{}' isn't supported in the codegen subset (either undefined, or has a `state:` block — see ActorCodegenInfo's doc comment)",
                    actor_name
                ))
            })?;
            let fn_ptr = info.fn_val.as_global_value().as_pointer_value();
            let call = builder.build_call(spawn_fn, &[fn_ptr.into()], "spawn_call").unwrap();
            call.try_as_basic_value()
                .basic()
                .map(|v| CgValue::Int(v.into_int_value()))
                .ok_or_else(|| CodegenError("tridentix_rt_spawn produced no value".into()))
        }
        other => Err(CodegenError(format!("expression not supported in codegen subset: {:?}", other))),
    }
}

/// Compile an actor `on receive` body into its lifted function.
#[allow(clippy::too_many_arguments)]
fn compile_actor_handler<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    info: &ActorCodegenInfo<'ctx>,
    body: &[Stmt],
) -> CResult<()> {
    let entry = context.append_basic_block(info.fn_val, "entry");
    builder.position_at_end(entry);

    let mut vars: HashMap<String, VarSlot<'ctx>> = HashMap::new();
    let msg_param = info
        .fn_val
        .get_nth_param(0)
        .ok_or_else(|| CodegenError("actor handler missing message parameter".into()))?
        .into_int_value();
    let msg_slot = builder.build_alloca(context.i64_type(), &info.param_name).unwrap();
    builder.build_store(msg_slot, msg_param).unwrap();
    vars.insert(info.param_name.clone(), VarSlot { ptr: msg_slot, kind: SlotKind::Int });

    // Dummy depth alloca: handlers are not recursion-depth guarded.
    let unused_depth = builder.build_alloca(context.i64_type(), "unused_depth").unwrap();

    let returned = compile_block(
        context, module, builder, fn_values, printf_fn, unused_depth,
        struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn,
        body, info.fn_val, &mut vars,
    )?;
    if !returned {
        let zero = context.i64_type().const_int(0, false);
        builder.build_return(Some(&zero)).unwrap();
    }
    Ok(())
}

/// Compile a lifted closure (param 0 is the env pointer).
#[allow(clippy::too_many_arguments)]
fn compile_closure_body<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    fn_values: &HashMap<String, FunctionValue<'ctx>>,
    printf_fn: FunctionValue<'ctx>,
    struct_types: &HashMap<String, StructTypeInfo<'ctx>>,
    enum_types: &HashMap<String, EnumTypeInfo<'ctx>>,
    closure_infos: &HashMap<*const Expr, ClosureInfo<'ctx>>,
    malloc_fn: FunctionValue<'ctx>,
    actor_infos: &HashMap<String, ActorCodegenInfo<'ctx>>,
    spawn_fn: FunctionValue<'ctx>,
    send_fn: FunctionValue<'ctx>,
    info: &ClosureInfo<'ctx>,
    body: &Expr,
) -> CResult<()> {
    let entry = context.append_basic_block(info.fn_val, "entry");
    builder.position_at_end(entry);

    let mut vars: HashMap<String, VarSlot<'ctx>> = HashMap::new();

    let env_param = info
        .fn_val
        .get_nth_param(0)
        .ok_or_else(|| CodegenError("closure missing env parameter".into()))?
        .into_pointer_value();
    for (i, fv_name) in info.free_vars.iter().enumerate() {
        let field_ptr = builder
            .build_struct_gep(info.env_struct_ty, env_param, i as u32, fv_name)
            .map_err(|_| CodegenError(format!("GEP failed for captured var '{}'", fv_name)))?;
        // Captured free variables are int in this codegen subset.
        vars.insert(fv_name.clone(), VarSlot { ptr: field_ptr, kind: SlotKind::Int });
    }

    for (i, (pname, is_float)) in info.param_names.iter().zip(info.param_is_float.iter()).enumerate() {
        let param_val = info
            .fn_val
            .get_nth_param((i + 1) as u32)
            .ok_or_else(|| CodegenError("missing closure parameter".into()))?;
        if *is_float {
            let slot = builder.build_alloca(context.f64_type(), pname).unwrap();
            builder.build_store(slot, param_val.into_float_value()).unwrap();
            vars.insert(pname.clone(), VarSlot { ptr: slot, kind: SlotKind::Float });
        } else {
            let slot = builder.build_alloca(context.i64_type(), pname).unwrap();
            builder.build_store(slot, param_val.into_int_value()).unwrap();
            vars.insert(pname.clone(), VarSlot { ptr: slot, kind: SlotKind::Int });
        }
    }

    let result = compile_expr(
        context, module, builder, fn_values, printf_fn, struct_types, enum_types, closure_infos, malloc_fn, actor_infos, spawn_fn, send_fn, body, &vars,
    )?;
    let as_i64 = match result {
        CgValue::Int(iv) => iv,
        CgValue::Float(fv) => builder.build_float_to_signed_int(fv, context.i64_type(), "ftoi").unwrap(),
    };
    builder.build_return(Some(&as_i64)).unwrap();
    Ok(())
}

fn compile_binop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    printf_fn: FunctionValue<'ctx>,
    l: CgValue<'ctx>,
    op: &BinOp,
    r: CgValue<'ctx>,
) -> CResult<CgValue<'ctx>> {
    let (l, r, is_float) = match (l, r) {
        (CgValue::Float(lf), CgValue::Float(rf)) => (CgValue::Float(lf), CgValue::Float(rf), true),
        (CgValue::Float(lf), CgValue::Int(ri)) => {
            let rf = builder.build_signed_int_to_float(ri, context.f64_type(), "promo").unwrap();
            (CgValue::Float(lf), CgValue::Float(rf), true)
        }
        (CgValue::Int(li), CgValue::Float(rf)) => {
            let lf = builder.build_signed_int_to_float(li, context.f64_type(), "promo").unwrap();
            (CgValue::Float(lf), CgValue::Float(rf), true)
        }
        (CgValue::Int(li), CgValue::Int(ri)) => (CgValue::Int(li), CgValue::Int(ri), false),
    };

    if is_float {
        let (CgValue::Float(lf), CgValue::Float(rf)) = (l, r) else { unreachable!() };
        Ok(match op {
            BinOp::Add => CgValue::Float(builder.build_float_add(lf, rf, "faddtmp").unwrap()),
            BinOp::Sub => CgValue::Float(builder.build_float_sub(lf, rf, "fsubtmp").unwrap()),
            BinOp::Mul => CgValue::Float(builder.build_float_mul(lf, rf, "fmultmp").unwrap()),
            BinOp::Div => CgValue::Float(builder.build_float_div(lf, rf, "fdivtmp").unwrap()),
            BinOp::Eq => CgValue::Int(zext(context, builder, builder.build_float_compare(FloatPredicate::OEQ, lf, rf, "fcmp").unwrap())),
            BinOp::NotEq => CgValue::Int(zext(context, builder, builder.build_float_compare(FloatPredicate::ONE, lf, rf, "fcmp").unwrap())),
            BinOp::Lt => CgValue::Int(zext(context, builder, builder.build_float_compare(FloatPredicate::OLT, lf, rf, "fcmp").unwrap())),
            BinOp::Gt => CgValue::Int(zext(context, builder, builder.build_float_compare(FloatPredicate::OGT, lf, rf, "fcmp").unwrap())),
            BinOp::LtEq => CgValue::Int(zext(context, builder, builder.build_float_compare(FloatPredicate::OLE, lf, rf, "fcmp").unwrap())),
            BinOp::GtEq => CgValue::Int(zext(context, builder, builder.build_float_compare(FloatPredicate::OGE, lf, rf, "fcmp").unwrap())),
        })
    } else {
        let (CgValue::Int(li), CgValue::Int(ri)) = (l, r) else { unreachable!() };
        Ok(match op {
            BinOp::Add => CgValue::Int(builder.build_int_add(li, ri, "addtmp").unwrap()),
            BinOp::Sub => CgValue::Int(builder.build_int_sub(li, ri, "subtmp").unwrap()),
            BinOp::Mul => CgValue::Int(builder.build_int_mul(li, ri, "multmp").unwrap()),
            BinOp::Div => CgValue::Int(compile_guarded_sdiv(context, module, builder, printf_fn, li, ri)),
            BinOp::Eq => CgValue::Int(zext(context, builder, builder.build_int_compare(IntPredicate::EQ, li, ri, "cmp").unwrap())),
            BinOp::NotEq => CgValue::Int(zext(context, builder, builder.build_int_compare(IntPredicate::NE, li, ri, "cmp").unwrap())),
            BinOp::Lt => CgValue::Int(zext(context, builder, builder.build_int_compare(IntPredicate::SLT, li, ri, "cmp").unwrap())),
            BinOp::Gt => CgValue::Int(zext(context, builder, builder.build_int_compare(IntPredicate::SGT, li, ri, "cmp").unwrap())),
            BinOp::LtEq => CgValue::Int(zext(context, builder, builder.build_int_compare(IntPredicate::SLE, li, ri, "cmp").unwrap())),
            BinOp::GtEq => CgValue::Int(zext(context, builder, builder.build_int_compare(IntPredicate::SGE, li, ri, "cmp").unwrap())),
        })
    }
}

fn zext<'ctx>(context: &'ctx Context, builder: &Builder<'ctx>, cmp: IntValue<'ctx>) -> IntValue<'ctx> {
    builder.build_int_z_extend(cmp, context.i64_type(), "booltmp").unwrap()
}

/// Integer divide with a runtime zero check (unguarded sdiv can SIGFPE).
fn compile_guarded_sdiv<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    printf_fn: FunctionValue<'ctx>,
    numerator: IntValue<'ctx>,
    denominator: IntValue<'ctx>,
) -> IntValue<'ctx> {
    let fn_val = builder.get_insert_block().unwrap().get_parent().unwrap();
    let exit_fn = module.get_function("exit").expect("exit() declared in compile_and_run");

    let zero = context.i64_type().const_int(0, false);
    let is_zero = builder.build_int_compare(IntPredicate::EQ, denominator, zero, "divzero_check").unwrap();

    let err_bb = context.append_basic_block(fn_val, "div_by_zero");
    let ok_bb = context.append_basic_block(fn_val, "div_ok");
    let cont_bb = context.append_basic_block(fn_val, "div_cont");

    builder.build_conditional_branch(is_zero, err_bb, ok_bb).unwrap();

    builder.position_at_end(err_bb);
    let msg = builder.build_global_string_ptr("Runtime error: division by zero\n", "div_zero_msg").unwrap();
    builder.build_call(printf_fn, &[msg.as_pointer_value().into()], "print_div_err").unwrap();
    let exit_code = context.i32_type().const_int(1, false);
    builder.build_call(exit_fn, &[exit_code.into()], "exit_call").unwrap();
    builder.build_unreachable().unwrap();

    builder.position_at_end(ok_bb);
    let result = builder.build_int_signed_div(numerator, denominator, "divtmp").unwrap();
    builder.build_unconditional_branch(cont_bb).unwrap();

    builder.position_at_end(cont_bb);
    result
}
