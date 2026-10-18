# Standard Library & AI Integration Spec
### Tridentix — Python Interop & Hardware Mapping (Draft v0.1)

> Ye document series ka teesra document hai. Pehle do documents ne bataya language kaisi dikhti hai (**Language Spec**) aur wo backend mein kaise chalti hai (**Compiler/VM Design**). Ye document batata hai ki wo **existing AI ecosystem (Python/PyTorch)** aur **hardware (GPU/TPU)** ke saath kaise connect hoti hai.

---

## 1. Python Interop

### 1.1 Goal
Tridentix ko duniya ke existing AI ecosystem (PyTorch, NumPy, HuggingFace) se **din-1 se compatible** hona zaroori hai — warna adoption impossible hai. Isliye naya wheel banane ke bajaye, Python ke saath **bidirectional bridge** banega.

### 1.2 Architecture: CPython C-API Bridge

```
 Tridentix source (.ae)
       |
       | import pytorch as torch
       v
 [Tridentix Compiler] --sees `import python:<module>`--> [Python Interop Layer]
       |
       v
 [ae_pybridge runtime module]
       |
       | libpython.so linked at runtime
       v
 [CPython Interpreter embedded in Tridentix runtime]
       |
       v
 [Real PyTorch / NumPy / HuggingFace Python objects]
```

- Tridentix runtime **CPython ko embed** karta hai (jaise `Py_Initialize()` C-API call se) — is process ke andar ek live Python interpreter chalta hai.
- `import pytorch as torch` likhne par compiler samajh jaata hai ye ek **foreign (Python) import** hai (naming convention ya explicit `import python:pytorch as torch` syntax se disambiguate hoga — decision needed, see §4).
- Har Python object (tensor, model, function) Tridentix runtime mein ek **opaque handle (`PyObjectRef`)** ke roop mein wrap hota hai.

### 1.3 Type Marshalling Rules

Python aur Tridentix ke beech data transfer karte waqt strict conversion rules:

| Tridentix Type | Python Equivalent | Conversion Cost |
|---|---|---|
| `int`, `float`, `bool`, `string` | `int`, `float`, `bool`, `str` | Cheap — direct C-API conversion (`PyLong_FromLong` etc.) |
| `vector<T>` | `list` / `numpy.ndarray` | Medium — element-wise copy unless zero-copy buffer protocol used |
| `tensor<dtype, shape>` | `torch.Tensor` / `numpy.ndarray` | **Zero-copy** via buffer protocol (`__array_interface__` / DLPack) — dekhein §1.4 |
| `graph`, `dataset`, `model` | Custom Python class wrapper | Handle-based reference, methods proxy hote hain |

### 1.4 Zero-Copy Tensor Bridge (Critical Rule)

> **Rule:** `tensor` type ko Python `torch.Tensor` mein convert karte waqt data **copy nahi hoga** — dono taraf **DLPack protocol** use hoga.

```
Tridentix tensor  --(export via DLPack capsule)-->  torch.Tensor
torch.Tensor     --(import via DLPack capsule)-->  Tridentix tensor
```

- DLPack ek **standard memory-layout descriptor** hai jo already PyTorch, NumPy, JAX, TensorFlow sab support karte hain.
- Tridentix tensor apna underlying buffer (pointer + shape + dtype + device info) ek `DLManagedTensor` struct ke roop mein expose karega.
- Isse `import pytorch as torch; let t2 = torch.from_aetheris(my_tensor)` jaisa call **memory copy ke bina** kaam karega — sirf metadata exchange hota hai.

### 1.5 Calling Python Functions
```
import python:pytorch as torch

fn train_with_torch(data: tensor):
    let torch_tensor = torch.from_dlpack(data)      # zero-copy handoff
    let model = torch.nn.Linear(128, 10)
    let output = model(torch_tensor)
    let result: tensor = tensor.from_dlpack(output)  # back to Tridentix
    return result
```

### 1.6 GIL (Global Interpreter Lock) Handling — Important
- CPython ka GIL Tridentix ke **actor concurrency model** se directly conflict karta hai (§2, Compiler doc).
- **Rule:** Jab bhi koi actor Python interop call karta hai, us actor ka runtime thread GIL acquire karta hai **sirf us call ki duration ke liye**, phir turant release karta hai — baaki actors block nahi honge.
- Heavy Python-bound compute (jaise pure-Python loops) actor scheduler ko starve kar sakta hai — isliye best practice: Python calls sirf **tensor-heavy library calls** (PyTorch/NumPy, jo GIL release karte hain internally C-level compute ke waqt) ke liye use karo, pure Python business-logic ke liye nahi.

### 1.7 AI ke liye Instruction
> Jab bhi Python library import karne wala code generate karo, hamesha `import python:<module> as <alias>` syntax use karo (explicit foreign-import marker). Tensor data Python ko bhejte/leते waqt hamesha `.from_dlpack()` / `.to_dlpack()` use karo — kabhi manual list-conversion se tensor data mat bhejo (performance loss).

---

## 2. Hardware Mapping (GPU/TPU Memory Rules)

### 2.1 Device Abstraction Model

Tridentix hardware ko ek **uniform `Device` abstraction** ke through expose karta hai:

```
enum DeviceKind:
    CPU
    GPU(index: int)
    TPU(index: int)

let dev: Device = Device.GPU(0)
let t: tensor<float, [1024, 1024]> = tensor.zeros([1024, 1024], device = dev)
```

- Har `tensor` ek `device` field carry karta hai — runtime ko pata hota hai data kahan physically resident hai.
- Cross-device operation (`cpu_tensor + gpu_tensor`) **compile-time error** hoga — explicit `.to(device)` call zaroori.

### 2.2 GPU Backend Mapping

| Layer | Technology |
|---|---|
| Codegen | LLVM **NVPTX** backend (CUDA) — primary; SPIR-V (Vulkan/OpenCL) — secondary, cross-vendor roadmap |
| Memory allocation | CUDA Unified Memory API (`cudaMallocManaged`) by default; explicit `cudaMalloc`/`cudaMemcpy` mode available for manual control |
| Kernel dispatch | Tridentix tensor ops compiler-generated PTX kernels; custom kernels via `@gpu_kernel` annotation |

### 2.3 Memory Allocation Rules (GPU)

```
let t = tensor.zeros([1000, 1000], device = Device.GPU(0))
```

Iske peeche runtime flow:
1. Tridentix runtime **memory pool allocator** check karta hai (pre-allocated GPU memory pool, taaki har allocation par `cudaMalloc` ka overhead na ho).
2. Agar pool mein jagah hai → seedha assign, **O(1)**.
3. Agar nahi → naya block `cudaMallocManaged` se allocate hota hai, pool mein add hota hai.
4. Tensor ka ownership rule (Language Spec §5.2) yahan bhi apply hota hai — jab tensor scope se bahar jaata hai (owner drop), runtime automatically GPU memory **pool ko return** karta hai (immediately free nahi karta — reuse ke liye).

### 2.4 Host ↔ Device Data Transfer

| Operation | Rule |
|---|---|
| `t.to(Device.GPU(0))` | Explicit copy — host se device memory, async by default (`cudaMemcpyAsync`) |
| `t.to(Device.CPU)` | Explicit copy back — blocking by default, `t.to_async(Device.CPU)` non-blocking variant |
| Actor message with GPU tensor | Message **sirf device-pointer + metadata** carry karta hai (zero host-copy) agar receiving actor bhi same GPU/node par hai |

### 2.5 TPU Mapping
- TPU support **XLA (Accelerated Linear Algebra) compiler** ke through hoga — Tridentix IR (AIR) ka ek subset **XLA HLO (High Level Operations)** mein lower hoga (LLVM path se alag, kyunki TPUs LLVM directly support nahi karte).
- V1 scope: TPU sirf **inference** ke liye supported; training-time TPU support roadmap mein hai (gradient/autodiff XLA integration extra complexity add karta hai).

### 2.6 Multi-GPU / Distributed Tensor Rule
```
let t = tensor.zeros([1000, 1000], device = Device.GPU(0))
let t2 = t.shard(devices = [Device.GPU(0), Device.GPU(1)])   # explicit sharding
```
- Auto-sharding **nahi** hoga by default (predictability ke liye) — user/AI ko explicit `.shard()` call karna hoga.

### 2.7 AI ke liye Instruction
> Jab bhi GPU/TPU-targeting code generate karo: (1) hamesha explicit `device =` parameter tensor creation mein do, (2) cross-device operations se pehle hamesha `.to(device)` call add karo, (3) memory-pool ka manual `free()` kabhi suggest mat karo — ownership drop hi cleanup trigger karta hai.

---

## 3. Summary — End-to-End Example

```
import python:pytorch as torch

fn run_inference(input: tensor):
    let dev = Device.GPU(0)
    let gpu_input = input.to(dev)

    # zero-copy handoff to PyTorch for a pretrained model
    let torch_input = torch.from_dlpack(gpu_input)
    let torch_model = torch.load("model.pt").to("cuda:0")
    let torch_output = torch_model(torch_input)

    let output: tensor = tensor.from_dlpack(torch_output)
    return output.to(Device.CPU)
```

---

## 4. Open Decisions (Aapko finalize karna hai)

1. Python import syntax — `import python:pytorch as torch` (explicit foreign marker) ya sirf `import pytorch as torch` (implicit, naming-convention se resolve)?
2. GPU memory allocation default — **Unified Memory** (simpler, kuch perf overhead) ya **explicit malloc/memcpy** (zyada control, zyada boilerplate) by default rakhein?
3. TPU training support v1 scope mein chahiye ya inference-only se shuru karein (§2.5)?
4. Multi-GPU sharding — sirf explicit (§2.6) rahe, ya future mein ek `auto_shard = true` opt-in flag bhi add karein?
5. Python interop error handling — Python exception aane par Tridentix `raise` mein convert ho ya alag `PyException` type ke roop mein propagate ho?

---

*Teeno documents (Language Spec + Compiler/VM Design + Stdlib/AI Integration) milkar poora foundation blueprint banate hain jisse AI code-generation tool ko diya ja sakta hai.*
