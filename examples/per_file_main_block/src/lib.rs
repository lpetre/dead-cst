//! Example out-of-tree native plugin that opts into the **per-file** API.
//!
//! It's the same `MainBlockPlugin` behaviour as `examples/main_block_plugin`
//! — keep every top-level `if __name__ == "__main__":` block (and its
//! enclosing module) reachable — but instead of one project-wide `run(ctx)`
//! over the whole graph, it implements [`PerFilePlugin`] and returns
//! `Some(self)` from `per_file()`. The host then invokes `run_on_file` once
//! per project file through a salsa-cached query, so an unchanged file's ops
//! are reused across a `re_materialize` with zero re-run.
//!
//! The purity contract (see [`PerFilePlugin`]) is what buys the cache:
//! `run_on_file` reads only this file's nodes + parsed AST and emits ops in
//! the file's *local* index space — it never touches project-wide state.
//!
//! Build it with `dead-cst build-plugin examples/per_file_main_block/src/lib.rs`
//! and load it like any other plugin:
//!
//! ```python
//! from dead_cst import _native as native
//! plugins = native.load_native_plugins(PATH)
//! Analysis(root, plugins=plugins).materialize_all()
//! ```

use std::ffi::c_void;

use dead_cst_runtime::native_plugins::plugin_api::{
    ExternalPlugin, FileOps, PerFilePlugin, PluginFileCtx,
};
use dead_cst_runtime::native_plugins::{
    PluginDesc, PluginManifest, PLUGIN_ABI_FINGERPRINT, PLUGIN_MANIFEST_MAGIC,
};

struct PerFileMainBlock;

impl ExternalPlugin for PerFileMainBlock {
    fn name(&self) -> &str {
        "ExternalPerFileMainBlockPlugin"
    }

    // Opt into per-file dispatch — the host ignores `run` and calls
    // `run_on_file` (below) once per file through the salsa-cached query.
    fn per_file(&self) -> Option<&dyn PerFilePlugin> {
        Some(self)
    }
}

impl PerFilePlugin for PerFileMainBlock {
    fn run_on_file(&self, file: &PluginFileCtx<'_>, ops: &mut FileOps) {
        // Only files with a top-level `if __name__ == "__main__":` block
        // contribute anything.
        let Some(block) = file.main_block_range() else {
            return;
        };
        let (block_start, block_end) = file.line_span(block);

        // Keep the module node (local index 0) alive, plus every top-level
        // decl whose source span falls inside the block. `keep_alive` flags
        // each as a reachability seed directly — all indices are file-local,
        // and the host maps them to global indices at apply time.
        ops.keep_alive(file.module_local_idx());
        for node in file.nodes() {
            if node.local_idx != file.module_local_idx()
                && node.start_line >= block_start
                && node.end_line <= block_end
            {
                ops.keep_alive(node.local_idx);
            }
        }
    }
}

extern "C" fn make_per_file_main_block_plugin() -> *mut c_void {
    let plugin: Box<dyn ExternalPlugin> = Box::new(PerFileMainBlock);
    Box::into_raw(Box::new(plugin)) as *mut c_void
}

/// The self-contained manifest. Reads only inlined consts + plain data — no
/// hashed runtime call — so the host can gate on the fingerprint before
/// touching any version-hashed symbol.
#[no_mangle]
pub extern "C" fn _dead_cst_plugin_manifest_v1() -> *const PluginManifest {
    const NAME: &str = "ExternalPerFileMainBlockPlugin";
    let descs: Box<[PluginDesc]> = Box::new([PluginDesc {
        name: NAME.as_ptr(),
        name_len: NAME.len(),
        make: make_per_file_main_block_plugin,
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
