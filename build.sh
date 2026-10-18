#!/bin/bash
set -e

echo "=== [1/4] Checking and fixing folder path spaces ==="
if [[ "$PWD" == *"Tridentix language"* ]]; then
    cd ..
    mv "Tridentix language" "Tridentix_language"
    cd Tridentix_language/tridentix
fi

echo "=== [2/4] Resetting corrupt environment variables ==="
unset RUSTFLAGS
mkdir -p .cargo

echo "=== [3/4] Writing fixed .cargo/config.toml ==="
cat > .cargo/config.toml << 'CFG'
[build]
target = "x86_64-pc-windows-gnu"

[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"
rustflags = [
    "-Lnative=/ucrt64/lib",
    "-lffi",
    "-lstdc++",
    "-lLLVM-22",
]

[env]
LLVM_SYS_220_PREFIX = "/ucrt64"
LLVM_SYS_200_PREFIX = "/ucrt64"
LLVM_SYS_170_PREFIX = "/ucrt64"
LIBFFI_SYS_USE_PKG_CONFIG = "1"
PKG_CONFIG_PATH = "/ucrt64/lib/pkgconfig"
CFG

echo "=== [4/4] Running Clean Cargo Build ==="
cargo clean
cargo build --release --bin tridentix -j 2

echo "=== SUCCESS: Tridentix compiled successfully! ==="
