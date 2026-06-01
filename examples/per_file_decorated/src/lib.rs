//! Example out-of-tree **per-file** native plugin that uses the ready-made
//! file-local query API instead of hand-rolling an AST walk.
//!
//! It keeps every top-level function decorated with a Click command
//! decorator (`@command` / `@group`, imported directly from `click`)
//! reachable — the kind of "decorated entrypoint" pattern a framework
//! plugin exists for. The whole match is two calls:
//!
//! * [`PluginFileCtx::imports_any_module`] — a cheap presence guard that
//!   skips files that don't import `click` at all.
//! * [`PluginFileCtx::decorated_decls`] — the file-local indices of the
//!   decorated decls, which pipe straight into [`FileOps::keep_alive`].
//!
//! No raw `parsed()` walk, no re-implementing import/alias resolution: the
//! query reuses the same decorator matcher core the in-tree project-wide
//! native plugins drive, so this per-file plugin and a project-wide twin
//! agree on what matches.
//!
//! Build it with
//! `dead-cst build-plugin examples/per_file_decorated/src/lib.rs` and load it
//! like any other plugin via `native.load_native_plugins(PATH)`.

use std::ffi::c_void;

use dead_cst_runtime::native_plugins::plugin_api::{
    ExternalPlugin, FileOps, PerFilePlugin, PluginFileCtx,
};
use dead_cst_runtime::native_plugins::{
    PluginDesc, PluginManifest, PLUGIN_ABI_FINGERPRINT, PLUGIN_MANIFEST_MAGIC,
};

const MARKER: &str = "<click-command>:per-file";

struct PerFileDecorated;

impl ExternalPlugin for PerFileDecorated {
    fn name(&self) -> &str {
        "ExternalPerFileDecoratedPlugin"
    }

    // Demonstrates the pre-graph hook — a real plugin might read a config
    // file under `project_root` here. We only assert it's wired (the host
    // forwards `NativePlugin.prepare` to this), so the body is a no-op.
    fn prepare(&self, _project_root: &str) {}

    // Opt into per-file dispatch.
    fn per_file(&self) -> Option<&dyn PerFilePlugin> {
        Some(self)
    }
}

impl PerFilePlugin for PerFileDecorated {
    fn run_on_file(&self, file: &PluginFileCtx<'_>, ops: &mut FileOps) {
        // Presence guard: nothing in a file that never imports `click`.
        if !file.imports_any_module(&["click"]) {
            return;
        }
        // File-local indices of decls decorated by `click.command` /
        // `click.group` (matched through this file's imports + aliases).
        for local_idx in file.decorated_decls(&["click"], &["command", "group"]) {
            ops.keep_alive(local_idx, MARKER.to_string());
        }
    }
}

extern "C" fn make_per_file_decorated_plugin() -> *mut c_void {
    let plugin: Box<dyn ExternalPlugin> = Box::new(PerFileDecorated);
    Box::into_raw(Box::new(plugin)) as *mut c_void
}

/// The self-contained manifest. Reads only inlined consts + plain data — no
/// hashed runtime call — so the host can gate on the fingerprint before
/// touching any version-hashed symbol.
#[no_mangle]
pub extern "C" fn _dead_cst_plugin_manifest_v1() -> *const PluginManifest {
    const NAME: &str = "ExternalPerFileDecoratedPlugin";
    let descs: Box<[PluginDesc]> = Box::new([PluginDesc {
        name: NAME.as_ptr(),
        name_len: NAME.len(),
        make: make_per_file_decorated_plugin,
    }]);
    let plugins = descs.as_ptr();
    let plugins_len = descs.len();
    Box::leak(descs);

    let manifest = Box::new(PluginManifest {
        magic: PLUGIN_MANIFEST_MAGIC,
        abi_fingerprint: PLUGIN_ABI_FINGERPRINT.as_ptr(),
        abi_fingerprint_len: PLUGIN_ABI_FINGERPRINT.len(),
        plugins,
        plugins_len,
    });
    Box::into_raw(manifest)
}
