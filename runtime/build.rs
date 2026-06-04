use std::process::Command;

/// Epoch of the curated `plugin_api` surface external plugins compile against.
/// **Bump this** whenever that surface changes incompatibly — a trait method
/// signature (`ExternalPlugin::run`, `PerFilePlugin::run_on_file`), a
/// renamed/removed `PluginCtx`/`PluginOps` query, a changed re-exported type —
/// so a plugin built against the old API is rejected at load, distinct from a
/// plain version bump. Folded into the ABI fingerprint below and re-exported as
/// `native_plugins::plugin_api::PLUGIN_API_EPOCH`.
///
/// 1 → 2: added `ExternalPlugin::declare_node_flags` / `declare_edge_flags`,
/// `PluginCtx::node_flag` / `edge_flag`, the re-exported `FlagSpec`, and
/// narrowed edge `flags` from `u32` to `u8`.
/// 2 → 3: `PluginOps::keep_alive` / `FileOps::keep_alive` dropped their
/// `marker: String` parameter (seeds flag the decl directly now), and added
/// the re-exported `FLAG_INIT_SUBCLASS` edge-flag constant.
const PLUGIN_API_EPOCH: u32 = 3;

/// Compose the ABI fingerprint that gates external native-plugin loading.
/// It changes whenever anything that could break the dylib ABI changes:
/// the compiler (commit hash), the runtime version, the target, the curated
/// [`PLUGIN_API_EPOCH`], or a manual `DEAD_CST_ABI_EPOCH` bump. A plugin bakes
/// the runtime's fingerprint at compile time, so a plugin built against a
/// different runtime is rejected at load.
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

    println!("cargo:rustc-env=PLUGIN_API_EPOCH={PLUGIN_API_EPOCH}");
    println!(
        "cargo:rustc-env=RUNTIME_ABI_FINGERPRINT={epoch}|api{PLUGIN_API_EPOCH}|{commit}|v{version}|{target}"
    );
    println!("cargo:rerun-if-env-changed=DEAD_CST_ABI_EPOCH");
}
