use std::process::Command;

/// Compose the ABI fingerprint that gates external native-plugin loading.
/// It changes whenever anything that could break the dylib ABI changes:
/// the compiler (commit hash), the runtime version, the target, or a manual
/// epoch bump. A plugin bakes the runtime's fingerprint at compile time, so
/// a plugin built against a different runtime is rejected at load.
fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let vv = Command::new(&rustc)
        .arg("-vV")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let commit = vv
        .lines()
        .find(|l| l.starts_with("commit-hash:"))
        .unwrap_or("commit-hash:unknown")
        .trim();
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0".into());
    let epoch = std::env::var("DEAD_CST_ABI_EPOCH").unwrap_or_else(|_| "1".into());

    println!("cargo:rustc-env=RUNTIME_ABI_FINGERPRINT={epoch}|{commit}|v{version}|{target}");
    println!("cargo:rerun-if-env-changed=DEAD_CST_ABI_EPOCH");
}
