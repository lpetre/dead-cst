//! Example out-of-tree native plugin, built as a separate cdylib that links
//! the `dead-cst-runtime` dylib. It reimplements `MainBlockPlugin` against
//! the public `plugin_api`: keep every top-level `if __name__ == "__main__":`
//! block (and its enclosing module) reachable.
//!
//! Build it with `dead-cst build-plugin main_block_plugin` (prints the
//! plugin path), then load it from Python:
//!
//! ```python
//! from dead_cst import _native as native
//! plugins = native.load_native_plugins("target/plugin-host/debug/libmain_block_plugin.dylib")
//! Analysis(root, plugins=plugins).materialize_all()
//! ```

use std::ffi::c_void;

use dead_cst_runtime::native_plugins::plugin_api::{ExternalPlugin, PluginCtx, PluginOps};
use dead_cst_runtime::native_plugins::{
    PluginDesc, PluginManifest, PLUGIN_ABI_FINGERPRINT, PLUGIN_MANIFEST_MAGIC,
};

const MARKER: &str = "<__main__>:external";

struct MainBlockPlugin;

impl ExternalPlugin for MainBlockPlugin {
    fn name(&self) -> &str {
        "ExternalMainBlockPlugin"
    }

    fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps) {
        for (module_idx, decl_indices) in ctx.main_blocks() {
            // Keep the module alive, plus every top-level decl inside the
            // `if __name__ == "__main__":` block.
            ops.keep_alive(module_idx, MARKER.to_string());
            for decl_idx in decl_indices {
                ops.keep_alive(decl_idx, MARKER.to_string());
            }
        }
    }
}

extern "C" fn make_main_block_plugin() -> *mut c_void {
    let plugin: Box<dyn ExternalPlugin> = Box::new(MainBlockPlugin);
    Box::into_raw(Box::new(plugin)) as *mut c_void
}

/// The self-contained manifest. Reads only inlined consts + plain data —
/// no hashed runtime call — so the host can gate on the fingerprint before
/// touching any version-hashed symbol.
#[no_mangle]
pub extern "C" fn _dead_cst_plugin_manifest_v1() -> *const PluginManifest {
    let descs: Box<[PluginDesc]> = Box::new([PluginDesc {
        name: "ExternalMainBlockPlugin".as_ptr(),
        name_len: "ExternalMainBlockPlugin".len(),
        make: make_main_block_plugin,
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
