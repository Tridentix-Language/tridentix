# Compiler Architecture & VM Design Document
### Tridentix — Backend, Runtime & Concurrency Spec (Draft v0.1)

> Ye document **Language Specification Document (Blueprint)** ka follow-up hai. Wahan bataya gaya tha language kaisi *dikhegi*; yahan bataya jaayega compiler backend mein wo kaise *chalegi* — target format, process model, aur message passing.

---

## 1. Target Backend

### 1.1 Decision: Two-Tier Compilation Strategy

Tridentix **single backend** pe depend nahi karegi — do tiers honge, kyunki language do alag workloads serve karti hai (AI/tensor compute + actor concurrency):

| Tier | Target | Use case |
|---|---|---|
| **Tier 1 — Primary** | **LLVM IR** | Native compilation — tensor math, actor scheduling, CPU/GPU-bound production code |
| **Tier 2 — Secondary** | **WebAssembly (WASM)** | Portable/sandboxed deployment — browser-based AI agents, edge devices, untrusted plugin code |

- **Direct Rust/C++ transpile nahi hoga** — isse maintenance do gunaa ho jaata (do target languages sync rakhna). LLVM IR ek hi intermediate layer hai jisse dono native binaries aur (via `wasm32` LLVM target) WASM dono nikal sakte hain.
- AI code-generation tool ko ye samajhna zaroori hai: **Tridentix source → Tridentix AST → Tridentix IR (custom, high-level) → LLVM IR → native binary / WASM**.

### 1.2 Compilation Pipeline

```
[.ae source file]
      |
      v
 [Lexer] -> tokens
      |
      v
 [Parser] -> AST
      |
      v
 [Type Checker / Borrow Checker]   <-- tensor ownership rules yahan enforce honge
      |
      v
 [Tridentix IR (AIR)]               <-- custom high-level IR: actors, messages, tensors as first-class nodes
      |
      v
 [Lowering Pass] -> LLVM IR
      |
      v
      +--------------------+
      |                    |
      v                    v
 [Native Codegen]      [WASM Codegen]
 (x86_64 / ARM / GPU)  (wasm32)
```

### 1.3 Why LLVM IR as the core target
- Mature optimization passes (inlining, vectorization) — tensor loops directly benefit.
- GPU targeting possible via LLVM's NVPTX backend (CUDA) ya SPIR-V (Vulkan/OpenCL) — future roadmap.
- WASM export "free" milta hai, alag backend likhne ki zaroorat nahi.

### 1.4 AI ke liye Instruction
> Jab bhi code generate karo jo compiler internals touch kare (codegen, IR lowering), **hamesha LLVM IR conventions follow karo** — SSA form, typed values, explicit basic blocks. Actor spawn aur message-send calls ko **runtime library calls** (`ae_rt_spawn`, `ae_rt_send`) ke roop mein lower karo, inline mat karo.

---

## 2. Actor/Process Model (Erlang-Style Supervisor Trees)

### 2.1 Process = Lightweight, Isolated Unit
- Har `actor`/`agent` runtime mein ek **lightweight process (green thread)** ban jaata hai — OS thread nahi. Tridentix VM/runtime scheduler M:N model use karega (jaise Erlang BEAM / Go goroutines): lakhon processes, thodi si OS threads pe multiplex.
- Har process:
  - Apna isolated heap rakhta hai (Language Spec §5.1 wale rules).
  - Ek **mailbox** rakhta hai (incoming messages ke liye FIFO queue).
  - Crash hone par sirf apna heap discard hota hai — baaki system unaffected.

### 2.2 Supervisor Tree

Processes **flat** nahi spawn hote — hamesha ek **supervision tree** ke andar hote hain:

```
supervisor RootSupervisor:
    strategy = one_for_one     # crash hua child akela restart hoga
    children:
        spawn Trainer as worker1
        spawn Trainer as worker2
        spawn DataLoader as loader1
```

**Supervision Strategies (Erlang OTP-inspired):**

| Strategy | Behavior jab child crash ho |
|---|---|
| `one_for_one` | Sirf crashed child restart hoga, baaki unaffected |
| `one_for_all` | Sab siblings restart honge (jab shared state critical ho) |
| `rest_for_one` | Crashed child + uske baad spawn hue siblings restart honge |
| `escalate` | Supervisor khud apne parent ko crash report kar dega (bubble-up) |

### 2.3 Crash Handling Flow

```
[Actor crashes] 
      |
      v
 [Runtime traps exception] -> [Actor's own state discarded]
      |
      v
 [EXIT signal sent to Supervisor] 
      |
      v
 [Supervisor applies strategy] -> restart / escalate / ignore
      |
      v
 [Restart count check] -> agar max_restarts/time_window cross ho gaya:
                            supervisor khud crash + escalate to its parent
```

**Restart limits example:**
```
supervisor TrainerSupervisor:
    strategy = one_for_one
    max_restarts = 3
    time_window = 60s     # 60 sec mein 3 se zyada crash -> escalate

    on child_crash(child, reason):
        log("restarting", child, reason)
```

### 2.4 AI ke liye Instruction
> Koi bhi `actor`/`agent` bina supervisor tree ke top-level spawn **na kare** — production code mein har spawn kisi supervisor ke `children` block ke andar hona chahiye. Agar user explicitly "unsupervised/detached process" maange, tabhi bare `spawn` allowed hai — aur AI ko warning comment add karna chahiye.

---

## 3. Concurrency Rules — Message Passing Mechanism

### 3.1 Core Rule: **No Shared Memory, Only Message Passing**
- Processes ke beech data **kabhi bhi shared mutable reference se nahi** jaata.
- Sirf `send()` / mailbox-based `receive` se communicate hota hai.

### 3.2 Send / Receive Syntax
```
actor Trainer:
    on receive(msg: TrainBatch):
        let result = train_step(msg.data)
        send(msg.reply_to, TrainResult(result))

fn main():
    let t = spawn Trainer()
    send(t, TrainBatch(data = my_data, reply_to = self))
    let response = await receive(TrainResult)
```

### 3.3 Message Semantics
| Rule | Detail |
|---|---|
| **Ordering** | Ek sender→receiver pair ke beech messages FIFO order mein deliver honge |
| **Delivery guarantee** | At-most-once (Erlang jaisa) — agar receiver crash ho gaya, message drop ho jaata hai (supervisor restart handle karega) |
| **Data ownership on send** | Primitive/small data **copy** hoti hai; `tensor`/large buffers **move (ownership transfer)** hote hain — zero-copy (Language Spec §5.3 se linked) |
| **Blocking vs Async** | `send()` non-blocking hai (fire-and-forget mailbox push). `receive`/`await receive(Type)` blocking hai jab tak matching message na aaye |
| **Selective receive** | `receive` ek specific message-type/pattern filter kar sakta hai; baaki mailbox mein wait karte hain |

### 3.4 Pattern-Matched Receive
```
actor Coordinator:
    on receive(msg):
        match msg:
            TrainBatch(data) => handle_batch(data)
            Shutdown()       => stop_gracefully()
            _                => log("unknown message")
```

### 3.5 Cross-Node Messaging (Distributed, Future Roadmap)
- V1 mein sab processes single-node hote hain (in-memory mailbox, LLVM-native runtime).
- V2 roadmap: `send(remote_actor@node2, msg)` — network-transparent messaging, Erlang distribution protocol jaisa, serialized over the wire (tensors ke liye zero-copy shared-memory nahi milega, serialization overhead hoga — isliye large tensors cross-node avoid karne chahiye).

### 3.6 AI ke liye Instruction
> Jab bhi do actors ke beech data-sharing wala code generate karo, **kabhi shared mutable variable/global state suggest mat karo**. Hamesha `send`/`receive` pattern use karo. Agar tensor bheja ja raha hai to explicitly comment karo `# ownership moved` taaki clear rahe ki sender ab us tensor ko access nahi karega.

---

## 4. Summary — End-to-End Flow Example

```
supervisor MLPipeline:
    strategy = one_for_one
    max_restarts = 5
    time_window = 30s
    children:
        spawn DataLoader as loader
        spawn Trainer as trainer

actor DataLoader:
    on receive(Start()):
        loop batch in dataset.stream("train.csv"):
            send(trainer, TrainBatch(batch))   # tensor ownership moved

actor Trainer:
    on receive(msg: TrainBatch):
        let grad = compute_gradient(msg.data)
        update_weights(grad)
```

Compiler flow: is source ko Tier-1 (LLVM IR → native) mein compile karke production training cluster pe chalaya jaayega; wahi source Tier-2 (WASM) mein compile karke browser-based demo/inference agent ke roop mein bhi deploy ho sakta hai — bina code change ke.

---

## 5. Open Decisions (Aapko finalize karna hai)

1. GPU codegen ke liye LLVM ka **NVPTX (CUDA)** priority rahe ya **SPIR-V (cross-vendor)**?
2. Distributed/cross-node messaging (§3.5) **v1 mein hi** chahiye ya v2 roadmap tak defer kar sakte hain?
3. Default supervision strategy kya ho agar user kuch specify na kare — `one_for_one` ko default rakhein?
4. Mailbox ki **max size/backpressure policy** kya ho (unbounded ya bounded with drop/block)?
5. Message serialization format cross-node ke liye — custom binary, Protobuf, ya MessagePack?

---

*Ye document Language Specification Document ke saath milkar poora "Blueprint pair" banata hai — ek batata hai language kaisi dikhti hai, doosra batata hai wo internally kaise chalti hai.*
