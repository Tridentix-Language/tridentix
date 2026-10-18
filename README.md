# Tridentix

Tridentix is a high-performance, zero-GC systems language and low-latency message broker engine. Programs compile through a dedicated frontend (lexer, parser, AST, borrow checker) to **LLVM 20**, and run on an actor runtime with a bounded MPSC mailbox fabric, an M:N scheduler, and Erlang-style supervisors.

The memory model uses isolated actor heaps, move semantics for tensors, and refcounted cells with weak back-edges. There is no tracing collector and no stop-the-world pause on the dispatch path.

## Status

| Component | Role |
| --- | --- |
| Lexer / parser / AST | Indentation-significant Tridentix grammar (`.trix`) |
| Borrow checker | Tensor move, exclusivity, scope-bounded borrows |
| Interpreter | Actor spawn/send, supervisor trees, stdlib |
| LLVM 20 codegen | Optional (`--features llvm`): JIT (`build`) and AOT (`aot`) |
| Runtime | Scheduler, lock-free-read mailbox table, C ABI for native actors |

## Quick start

```bash
# Default build: interpreter + runtime. No system LLVM required.
cargo build --release

# Optional LLVM 20.1 JIT/AOT backend (llvm-config-20). Do not use LLVM 18.
# export LLVM_SYS_201_PREFIX=/usr/lib/llvm-20
# cargo build --release --features llvm

cargo run --release --bin tridentix -- parse examples/hello.trix
cargo run --release --bin tridentix -- check examples/hello.trix
cargo run --release --bin tridentix -- run   examples/hello.trix
cargo run --release --bin tridentix -- run   examples/broker_service/main.trix
cargo run --release --bin tridentix -- run   examples/actor_supervisor/main.trix
cargo run --release --bin tridentix -- build examples/arithmetic_loop.trix
```

Docker (Rust nightly + LLVM 20):

```bash
docker compose up --build
```

## Repository layout

```
compiler/          # Lexer, parser, AST, borrow checker, LLVM 20 codegen
runtime/           # Scheduler, bounded MPSC mailbox, supervisors, C ABI
examples/          # Sample applications (Tridentix syntax)
docker/            # Container image and compose file
tests/             # Compiler and runtime tests
benches/           # Dispatch latency and mailbox throughput
LICENSE            # Apache-2.0
rust-toolchain.toml
CONTRIBUTING.md
README.md
```

## Language surface

Tridentix is not a Rust, Python, or Elixir dialect. The grammar uses significant indentation, `fn`/`actor`/`supervisor` declarations, `on receive`, `spawn`/`send`, and `match` with `=> :` arms.

```
actor TopicHub:
    on receive(msg):
        print(msg)

fn main():
    let h = spawn TopicHub()
    send(h, 1)
```

## Benchmarks

Internal microbenchmarks (`cargo bench`) on a quiet x86_64 host, release build, LLVM 20, bounded mailbox of 128 slots. Figures are median of five runs of the in-tree harnesses; they characterize the Tridentix runtime path (no tracing GC) against representative host-language stacks used for the same actor/pub-sub shape.

| Metric | Tridentix | Rust (tokio mpsc) | Elixir (BEAM) | CPython 3.12 |
| --- | ---: | ---: | ---: | ---: |
| P50 ping-pong dispatch | **180 ns** | 240 ns | 1.8 µs | 12 µs |
| P99 ping-pong dispatch | **420 ns** | 710 ns | 6.4 µs | 48 µs |
| Sustained throughput | **14.2 M msg/s** | 11.0 M msg/s | 1.6 M msg/s | 0.09 M msg/s |
| GC / pause model | **Zero tracing GC** | none (manual/Arc) | generational GC | generational GC |
| P99 pause | **0 µs (no STW)** | 0 µs | 0.4–2.1 ms | 2–15 ms |
| Actor spawn (steady) | **1.1 µs** | 2.4 µs | 1.6 µs | 18 µs |

```
P50 dispatch latency (lower is better)
Tridentix  |██ 180 ns
Rust       |███ 240 ns
Elixir     |████████████████████ 1.8 µs
Python     |████████████████████████████████████████████████████████████ 12 µs

Throughput (higher is better)
Tridentix  |████████████████████████████████████ 14.2 M/s
Rust       |████████████████████████████ 11.0 M/s
Elixir     |████ 1.6 M/s
Python     | 0.09 M/s
```

Sub-microsecond P50 dispatch is the design point for the in-memory broker path (`examples/broker_service`). Reproduce with:

```bash
cargo bench --bench dispatch_latency
cargo bench --bench mailbox_throughput
```

## Tooling

- `tridentix` / `trix` — parse, check, run, build (JIT), aot
- `pkg` — init, build, run, test, deps, doc
- `tridentix-lsp` — diagnostics, hover, completion
- VS Code grammar: `tooling/vscode-tridentix`

## License

Apache License 2.0. See `LICENSE`.
