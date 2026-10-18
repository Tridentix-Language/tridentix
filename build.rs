//! Linker hints for the optional `llvm` feature (llvm-sys-201 / LLVM 20.1).
fn main() {
    println!("cargo:rerun-if-env-changed=LLVM_SYS_201_PREFIX");
    if std::env::var("CARGO_FEATURE_LLVM").is_err() {
        return;
    }
    if cfg!(all(target_os = "windows", target_env = "gnu")) {
        println!("cargo:rustc-link-search=native=/ucrt64/lib");
        println!("cargo:rustc-link-lib=ffi");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=LLVM-20");
    }
}
