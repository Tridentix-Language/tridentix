# Language Specification Document (The Blueprint)
### Custom AI-Native Programming Language — Draft v0.1

> **Language ka naam: "Tridentix"** (Sanskrit — "mind/consciousness"). AI-first language ke liye fitting naam, kyunki ye "thinking" agents/actors ko concurrently orchestrate karti hai. Ye document AI code-generation tools (jaise Claude/GPT) ko dene ke liye base spec hai, taaki wo consistent, valid code generate karein.

---

## 1. Design Philosophy

Tridentix ek **AI/ML-first, concurrency-friendly** language hai jo teen cheezon ko combine karti hai:

| Inspiration | Kya liya gaya |
|---|---|
| Python | Clean, indentation-based readable syntax |
| Erlang/Elixir | Actor-model concurrency, isolated memory, fault tolerance |
| Rust | Safety-critical tensor/GPU memory ke liye ownership & borrowing |

Goal: Data scientists ko Python jaisa comfort mile, aur production AI systems ko Erlang jaisi reliability + Rust jaisi memory safety mile.

---

## 2. Syntax & Grammar Rules

### 2.1 Block Structure
- **Indentation-based** (Python jaisa) — curly braces `{}` nahi honge.
- 4 spaces = 1 indentation level (tabs allowed nahi honge, taaki AI-generated code consistent rahe).

### 2.2 Statement Termination
- Semicolons `;` **optional** hain. Ek line = ek statement (Python jaisa).
- Multiple statements ek line mein likhne ho to `;` mandatory hoga:
  ```
  x = 5; y = 10
  ```

### 2.3 Comments
```
# single line comment
"""
multi-line
comment block
"""
```

### 2.4 Function Declaration
```
fn add(a: int, b: int) -> int:
    return a + b
```
- `fn` keyword mandatory.
- Return type `-> type` optional; agar diya nahi to compiler infer karega.
- Parameters ke type-hints **optional but recommended** — Tridentix gradually-typed language hai.

### 2.5 Variable Declaration
```
let x = 10          # immutable by default
let mut y = 20       # mutable, explicit
const PI = 3.14159   # compile-time constant
```

### 2.6 Control Flow
```
if x > 10:
    print("big")
elif x == 10:
    print("equal")
else:
    print("small")

loop i in range(0, 10):
    print(i)

while condition:
    do_something()
```

### 2.7 Naming Conventions
| Element | Convention | Example |
|---|---|---|
| Variables/functions | snake_case | `learning_rate` |
| Types/Actors/Agents | PascalCase | `NeuralNet`, `TrainerAgent` |
| Constants | UPPER_SNAKE | `MAX_EPOCHS` |

### 2.8 Module System
```
import math
import ai.tensor as t
module MyModel:
    fn forward(...):
        ...
```

---

## 3. Data Types

### 3.1 Primitive Types
| Type | Description | Example |
|---|---|---|
| `int` | 64-bit signed integer | `let x: int = 5` |
| `float` | 64-bit floating point | `let lr: float = 0.001` |
| `bool` | true/false | `let done: bool = false` |
| `string` | UTF-8 text | `let name: string = "run1"` |
| `none` | absence of value | `let x: none = none` |

### 3.2 AI-Native Composite Types

**tensor** — n-dimensional numeric array (core AI type, GPU-eligible)
```
let t: tensor<float, [3, 3]> = tensor.zeros([3, 3])
let t2 = tensor([1.0, 2.0, 3.0])          # shape inferred
```

**vector** — 1D fixed-type array, lighter than tensor (CPU-first)
```
let v: vector<float> = [1.0, 2.0, 3.0]
```

**graph** — node/edge structure for GNNs, computation graphs, agent topologies
```
let g: graph<Node, Edge> = graph()
g.add_node("A")
g.add_node("B")
g.connect("A", "B", weight = 0.5)
```

**dataset** — streaming/batchable data container
```
let d: dataset = dataset.load("train.csv")
```

**model** — trainable unit wrapping tensors + forward/backward logic
```
model NeuralNet:
    layers: vector<Layer>
    fn forward(x: tensor) -> tensor:
        ...
```

### 3.3 Type Declaration Rules
- Generic types use `<>` — e.g. `tensor<dtype, shape>`.
- Shape can be `?` for dynamic/unknown dimension: `tensor<float, [?, 128]>`.
- Type inference default hai; explicit typing sirf tensors/graphs jaise high-stakes types ke liye strongly recommended.

---

## 4. Keywords (Reserved Words)

| Category | Keywords |
|---|---|
| Declaration | `let`, `mut`, `const`, `fn`, `type`, `model`, `module` |
| Concurrency | `actor`, `agent`, `spawn`, `send`, `receive`, `on`, `monitor`, `kill`, `async`, `await` |
| Control Flow | `if`, `elif`, `else`, `loop`, `while`, `match`, `break`, `continue`, `return` |
| Data Types | `tensor`, `vector`, `graph`, `dataset`, `int`, `float`, `bool`, `string`, `none` |
| Error Handling | `try`, `catch`, `raise`, `finally` |
| Logical/Boolean | `and`, `or`, `not`, `true`, `false` |
| Import/Structure | `import`, `as`, `export` |

**Actor/Agent Concurrency Example:**
```
actor Trainer:
    on receive(msg: TrainSignal):
        let grad = compute_gradient(msg.batch)
        send(self.parent, grad)

fn main():
    let t = spawn Trainer()
    send(t, TrainSignal(batch))
```

`agent` vs `actor`: `actor` = generic concurrent unit (Erlang-style). `agent` = higher-level AI-specific actor with built-in state/policy/reward hooks for RL/multi-agent systems.

---

## 5. Memory Model

Tridentix **hybrid memory model** use karti hai — do alag zones:

### 5.1 Actor/Agent Memory — Isolated (Erlang-style)
- Har `actor`/`agent` ka apna **private heap** hota hai.
- Koi shared mutable state nahi — actors sirf **message-passing** (`send`/`receive`) se communicate karte hain.
- Har actor heap **independently garbage-collected** hota hai — ek actor crash ho to dusre unaffected rehte hain ("let it crash" philosophy, supervised by `monitor`).
- Isse concurrency bugs (race conditions, data races) design se hi eliminate ho jaate hain.

### 5.2 Tensor/GPU Memory — Ownership-based (Rust-style)
- `tensor` aur bade numeric buffers **ownership + borrowing rules** follow karte hain:
  - Har tensor ka ek hi **owner** hota hai.
  - `borrow(t)` se read-only reference milta hai; `borrow_mut(t)` se mutable reference — compile-time mein enforce hota hai ki ek time pe ek hi mutable borrow ho.
  - Ownership transfer ho sakta hai (`move`), copy nahi (heavy GPU buffers avoid karne ke liye).
- Isse **manual memory management ka safety-net** milta hai bina garbage collector overhead ke — training loops mein predictable, low-latency performance.

### 5.3 Interop Rule
- Jab ek actor kisi doosre actor ko tensor `send` karta hai, ownership **transfer** ho jaati hai (zero-copy move) — sender ke paas se access khatam ho jaata hai. Isse actor-isolation aur tensor-ownership dono rules ek saath maintain hote hain.

```
actor Worker:
    on receive(t: tensor):
        # ownership of `t` ab is actor ke paas hai
        let result = t.sum()
        send(self.parent, result)   # sirf result bhejo, tensor nahi
```

### 5.4 Summary Table
| Zone | Model | Inspired by | Purpose |
|---|---|---|---|
| Actor/Agent state | Isolated heap + GC + message passing | Erlang | Fault-tolerant concurrency |
| Tensor/GPU buffers | Ownership + borrow-checking, no GC | Rust | High-performance, safe numeric compute |

---

## 6. Open Decisions (Aapko finalize karna hai)

1. Kya `;` completely optional rahenge ya kabhi mandatory karna hai (multi-statement lines)?
2. `dtype` list kya-kya honge (`float16`, `float32`, `int8` etc.) — GPU quantization support chahiye?
3. Error handling: exceptions (`try/catch`) ya Rust-style `Result<T, E>` return type — ya dono?
4. Kya language compiled hogi, interpreted, ya JIT? *(Compiler doc mein ye already decide ho chuka hai — LLVM IR + WASM two-tier, dekhein Compiler Architecture doc §1.)*

---

*Ye document ek starting blueprint hai — jitna detailed aur unambiguous ye hoga, AI code-generation utna hi accurate aur consistent code likhega.*
