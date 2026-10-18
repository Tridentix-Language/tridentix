# Contributing to Tridentix

Contributions are accepted as focused pull requests against the default branch. The compiler, runtime, and language samples are production surfaces: treat each change as if it will ship.

## Requirements

- Rust nightly as pinned in `rust-toolchain.toml`.
- Default `cargo build` / `cargo test` must pass **without** a system LLVM install.
- LLVM **20.1** (`llvm-config-20`, `LLVM_SYS_201_PREFIX`) is required only for `--features llvm`. LLVM 18 is not supported.
- `cargo fmt` and `cargo clippy --workspace --all-targets -- -D warnings` must pass.
- `cargo test --workspace` must pass. `cargo build --release` must remain clean.

## Pull request workflow

1. Open a short design note in the PR description: problem, approach, risk.
2. Keep diffs scoped. Do not mix language-syntax changes with unrelated refactors.
3. Add or update tests under `tests/` for compiler behavior and runtime invariants.
4. If you change actor dispatch or mailboxes, include a `benches/` run and record the delta.
5. Do not introduce tracing garbage collection, stop-the-world pauses, or global locks on the send path.
6. Language samples in `examples/` must use Tridentix syntax (`.trix`), not Rust, Python, or Elixir.
7. Squash drive-by comment noise. Public APIs get a one- or two-line rustdoc; no essay comments.

## Review bar

Reviewers check correctness, lifetime and ownership discipline, LLVM 20 API usage, and whether error paths fail closed. A PR that weakens borrow checking, supervisor restart semantics, or mailbox bounds will be rejected.

## Security

Do not commit secrets, machine-local absolute paths, or vendor binaries. Report vulnerabilities privately to the maintainers; do not open a public issue for exploitable defects.
