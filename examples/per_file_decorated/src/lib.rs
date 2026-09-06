//! Example out-of-tree plugin **pair** that demonstrates the flag and
//! topic/fact registries end to end, using the ready-made file-local query
//! API instead of hand-rolling an AST walk.
//!
//! Goal: keep every top-level function decorated with a Click command
//! decorator (`@command` / `@group`, imported directly from `click`)
//! reachable — but instead of a single plugin calling `keep_alive`, the work
//! is split across the per-file -> project-wide channel the runtime exposes:
//!
//! * [`ClickCommandScanner`] is a **per-file** plugin. For each click-decorated
//!   decl it [emits a fact](FileOps::emit_fact) under the topic
//!   `click/commands`, pinned to that decl. Per-file output must be a pure
//!   function of the file, so it emits by topic **name** (a salsa-stable
//!   string) — a per-file plugin can't resolve a registry handle or flag bit,
//!   which is exactly why the flag is stamped on the project-wide side, not
//!   here. It finds the decls with [`PluginFileCtx::imports_any_module`] +
//!   [`PluginFileCtx::decorated_decls`] (the same matcher core the in-tree
//!   project-wide plugins drive — no raw `parsed()` walk).
//! * [`ClickCommandKeeper`] is a **project-wide** plugin. It declares the
//!   `click/command` node flag, resolves the `click/commands` topic handle with
//!   [`PluginCtx::topic`], reads every collected fact via
//!   [`PluginCtx::facts_for_topic`], and stamps the registered flag on each
//!   fact's decl with [`PluginOps::flag_decl`] — keeping it alive.
//!
//! Both plugins declare the `click/commands` topic with an identical
//! [`TopicSpec`]; registration is idempotent, so they share one handle. The
//! manifest exports both, so loading this one cdylib wires up the whole round
//! trip.
//!
//! Build it with
//! `dead-cst build-plugin examples/per_file_decorated/src/lib.rs` and load it
//! like any other plugin via `native.load_native_plugins(PATH)`.

use std::ffi::c_void;

use dead_cst_runtime::native_plugins::plugin_api::{
    ExternalPlugin, FileOps, FlagSpec, PerFilePlugin, PluginCtx, PluginError, PluginFileCtx,
    PluginOps, TopicSpec,
};
use dead_cst_runtime::native_plugins::{
    PluginDesc, PluginManifest, PLUGIN_ABI_FINGERPRINT, PLUGIN_MANIFEST_MAGIC,
};

/// The shared topic both plugins declare (idempotently) and exchange facts
/// over: the per-file scanner publishes under it, the project-wide keeper
/// reads from it.
const TOPIC_NAME: &str = "click/commands";
const TOPIC_DESC: &str = "fqname of each top-level Click-decorated command/group decl";

/// The node flag the keeper declares and stamps on each command decl. It sets
/// both `seed` and `default_on`, putting it in the default keepalive mask, so
/// a flagged decl is a reachability root.
const FLAG_NAME: &str = "click/command";
const FLAG_DESC: &str = "top-level Click command/group entrypoint";

/// The one [`TopicSpec`] both sides register. Sharing a single builder keeps
/// the name + description byte-identical, which is what makes the second
/// registration idempotent rather than a conflicting re-declaration.
fn click_commands_topic() -> TopicSpec {
    TopicSpec {
        name: TOPIC_NAME.to_string(),
        description: TOPIC_DESC.to_string(),
    }
}

// --- per-file emitter -------------------------------------------------------

struct ClickCommandScanner;

impl ExternalPlugin for ClickCommandScanner {
    fn name(&self) -> &str {
        "ExternalClickCommandScanner"
    }

    fn declare_topics(&self) -> Vec<TopicSpec> {
        vec![click_commands_topic()]
    }

    // Opt into per-file dispatch — the host ignores `run` and calls
    // `run_on_file` (below) once per file through the salsa-cached query.
    fn per_file(&self) -> Option<&dyn PerFilePlugin> {
        Some(self)
    }
}

impl PerFilePlugin for ClickCommandScanner {
    fn run_on_file(&self, file: &PluginFileCtx<'_>, ops: &mut FileOps) {
        // Presence guard: nothing in a file that never imports `click`.
        if !file.imports_any_module(&["click"]) {
            return;
        }
        // File-local indices of decls decorated by `click.command` /
        // `click.group` (matched by fqname through this file's imports +
        // aliases, or a same-file definition).
        for local_idx in file.decorated_decls(&["click.command", "click.group"]) {
            // Publish a fact pinned to the decl. The host translates the
            // file-local index to a global node index when it collects the
            // fact (dropping it if the decl didn't survive assembly); the
            // value is the decl's fqname, for the reader to use or log.
            let fqname = file.node(local_idx).map(|n| n.fqname).unwrap_or_default();
            ops.emit_fact(TOPIC_NAME, Some(local_idx), fqname);
        }
    }
}

// --- project-wide reader ----------------------------------------------------

struct ClickCommandKeeper;

impl ExternalPlugin for ClickCommandKeeper {
    fn name(&self) -> &str {
        "ExternalClickCommandKeeper"
    }

    fn declare_node_flags(&self) -> Vec<FlagSpec> {
        vec![FlagSpec {
            name: FLAG_NAME.to_string(),
            seed: true,
            default_on: true,
            description: FLAG_DESC.to_string(),
        }]
    }

    // Declared on both sides (idempotent) so the handle resolves no matter
    // which plugin the host registers first.
    fn declare_topics(&self) -> Vec<TopicSpec> {
        vec![click_commands_topic()]
    }

    // The pre-graph hook — a real plugin might read a config file under
    // `project_root` here to stash project-wide `run` setup. No-op here.
    fn prepare(&self, _project_root: &str) {}

    fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps) -> Result<(), PluginError> {
        // The per-file scanner has already run (facts are collected during
        // graph assembly, before this project-wide pass), so the topic's
        // facts are ready to read.
        let Some(handle) = ctx.topic(TOPIC_NAME) else {
            return Ok(());
        };
        let flag = ctx
            .node_flag(FLAG_NAME)
            .expect("click/command is declared in declare_node_flags");
        for fact in ctx.facts_for_topic(handle) {
            // A fact pinned to a decl that survived assembly carries its
            // global node index; stamp the keepalive flag on it.
            if let Some(decl_idx) = fact.decl_idx {
                ops.flag_decl(decl_idx, flag);
            }
        }
        Ok(())
    }
}

// --- manifest ---------------------------------------------------------------

extern "C" fn make_click_command_scanner() -> *mut c_void {
    let plugin: Box<dyn ExternalPlugin> = Box::new(ClickCommandScanner);
    Box::into_raw(Box::new(plugin)) as *mut c_void
}

extern "C" fn make_click_command_keeper() -> *mut c_void {
    let plugin: Box<dyn ExternalPlugin> = Box::new(ClickCommandKeeper);
    Box::into_raw(Box::new(plugin)) as *mut c_void
}

/// The self-contained manifest. Reads only inlined consts + plain data — no
/// hashed runtime call — so the host can gate on the fingerprint before
/// touching any version-hashed symbol. Exports **both** plugins, so loading
/// this one cdylib registers the per-file emitter and the project-wide reader.
#[no_mangle]
pub extern "C" fn _dead_cst_plugin_manifest_v1() -> *const PluginManifest {
    const SCANNER: &str = "ExternalClickCommandScanner";
    const KEEPER: &str = "ExternalClickCommandKeeper";
    let descs: Box<[PluginDesc]> = Box::new([
        PluginDesc {
            name: SCANNER.as_ptr(),
            name_len: SCANNER.len(),
            make: make_click_command_scanner,
        },
        PluginDesc {
            name: KEEPER.as_ptr(),
            name_len: KEEPER.len(),
            make: make_click_command_keeper,
        },
    ]);
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
