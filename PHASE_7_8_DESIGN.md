# Phase 7-8: Python Interop & GPU/TPU — Design + Interface Stubs

> **Honest status: NOT runnable in this sandbox.** Everything below is real,
> compilable-in-principle Rust code following the Stdlib/AI Integration
> spec doc, but it needs things this sandbox genuinely doesn't have:
> a GPU (`nvidia-smi` confirmed absent), a CUDA toolkit, and — for the
> Python bridge — a full PyO3 build against a matching CPython. I did not
> fake test output for these; the honest thing is to hand you working
> *design + skeleton code* and be upfront that it's unverified here.

---

## 1. Python Interop Skeleton (`PyO3`-based)

This is what `Cargo.toml` would need:
```toml
[dependencies]
pyo3 = { version = "0.22", features = ["auto-initialize"] }
```

And the bridge module (matches Stdlib doc §1.2–§1.6):

```rust
// src/pybridge.rs — SKELETON, not wired into main.rs (needs PyO3 + a
// real Python 3 install to link against; untested in this sandbox).
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::interpreter::Value;

/// Marshals a Tridentix Value into a Python object (Stdlib doc §1.3 table).
/// Tensors are NOT copied element-by-element here — real code would use
/// PyO3's buffer-protocol / DLPack capsule APIs so PyTorch/NumPy can
/// read the same memory directly (Stdlib doc §1.4).
fn to_py(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    Ok(match v {
        Value::Int(n) => n.into_py(py),
        Value::Float(f) => f.into_py(py),
        Value::Str(s) => s.into_py(py),
        Value::Bool(b) => b.into_py(py),
        Value::List(items) => {
            let converted: PyResult<Vec<PyObject>> =
                items.iter().map(|i| to_py(py, i)).collect();
            converted?.into_py(py)
        }
        Value::Tensor { data, .. } => data.clone().into_py(py), // placeholder: real path is DLPack, not a copy
        other => format!("{}", other).into_py(py),
    })
}

/// Calls an arbitrary Python callable — the runtime-level implementation
/// of `import python:<module> as <alias>` (Stdlib doc §1.7).
pub fn call_python(module: &str, func: &str, args: Vec<Value>) -> PyResult<String> {
    Python::with_gil(|py| {
        let m = PyModule::import_bound(py, module)?;
        let f = m.getattr(func)?;
        let py_args: PyResult<Vec<PyObject>> = args.iter().map(|a| to_py(py, a)).collect();
        let result = f.call1(pyo3::types::PyTuple::new_bound(py, py_args?))?;
        Ok(result.str()?.to_string())
    })
}
```

**Why this isn't wired into `main.rs`:** linking against `libpython3.x` needs
the exact dev headers/lib present at build time, and the GIL-handling rule
from the Stdlib doc (§1.6 — "acquire GIL only for the call's duration")
needs to interact correctly with our actor threads (`interpreter.rs`
`spawn_actor`), which is a non-trivial integration test I can't validate
without a GPU-having, PyTorch-having machine to actually run
`torch.from_dlpack(...)` against.

## 2. GPU/TPU Skeleton (`cudarc`-based)

Matches Hardware Mapping doc §2.1–§2.3:

```rust
// src/gpu.rs — SKELETON, untested (no CUDA device in this sandbox).
// Cargo.toml would need: cudarc = "0.12"

use cudarc::driver::{CudaDevice, DriverError};
use std::sync::Arc;

pub enum DeviceKind {
    Cpu,
    Gpu(usize),
}

/// Mirrors the `Device.GPU(0)` abstraction from Hardware Mapping doc §2.1.
pub struct TridentixTensor {
    pub shape: Vec<usize>,
    pub device: DeviceKind,
    cpu_data: Option<Vec<f64>>,
    // gpu_data: Option<cudarc::driver::CudaSlice<f64>>,  // only valid with a real CUDA device
}

pub fn zeros_on_gpu(shape: Vec<usize>, gpu_index: usize) -> Result<TridentixTensor, DriverError> {
    let dev = CudaDevice::new(gpu_index)?; // this line requires an actual GPU + CUDA driver
    let total: usize = shape.iter().product();
    let _buf = dev.alloc_zeros::<f64>(total)?; // maps to Hardware Mapping doc §2.3's pool-allocator flow
    Ok(TridentixTensor {
        shape,
        device: DeviceKind::Gpu(gpu_index),
        cpu_data: None,
    })
}
```

`CudaDevice::new(0)` is the line that would fail immediately in this
sandbox — there's no `/dev/nvidia*` device node. This code is provided
as the correct **shape** of the implementation per the spec, to be
compiled and tested on a machine with an actual NVIDIA GPU + CUDA
toolkit installed.

## 3. What's Actually Verified vs. What Isn't

| Phase | Status | Evidence |
|---|---|---|
| 1-2: Lexer + Parser | ✅ Verified | `cargo run -- parse examples/hello.Tridentix` — real AST output |
| 3: Interpreter | ✅ Verified | `cargo run -- run examples/hello.Tridentix` — real "Hello, Tridentix!" + loop output |
| 4: Actor runtime | ✅ Verified | `cargo run -- run examples/actors.Tridentix` — real multi-threaded message passing, non-deterministic interleaving observed |
| 5: Type checker | ✅ Verified | `cargo run -- check /tmp/buggy.Tridentix` — caught 3 real errors (type mismatch, undefined var, arity) |
| 6: LLVM codegen | ✅ Verified (restricted subset) | `cargo run -- build examples/arithmetic.Tridentix` — real LLVM IR text + JIT-executed native code, `fib(10)+add(3,4) = 62` confirmed correct |
| 7: Python interop | ⚠️ Skeleton only | Code compiles in principle; not built/linked/run here |
| 8: GPU/TPU | ⚠️ Skeleton only | Code shown for correct shape; needs real GPU hardware to even attempt compiling `cudarc` calls meaningfully |

## 4. Realistic Next Steps for 7-8
1. Get access to a machine with an NVIDIA GPU + CUDA toolkit installed (cloud GPU instance, e.g.).
2. Add `pyo3` + `cudarc` to `Cargo.toml` there, wire `pybridge.rs`/`gpu.rs` into `main.rs` as new subcommands.
3. Write a real DLPack capsule exchange test: create a Tridentix tensor, hand it to `torch.from_dlpack()`, mutate it in PyTorch, read it back in Tridentix, and assert the underlying pointer never changed (proves zero-copy).
4. Only after that integration test passes should Phase 7-8 be marked "verified" like Phases 1-6 above.
