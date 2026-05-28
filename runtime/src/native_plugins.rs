//! Native (rust-side) plugins — in-tree only.
//!
//! A native plugin is a rust implementation of the plugin contract
//! that skips the Python `.run(ctx)` call entirely: the harness
//! detects a :class:`NativePlugin` pyclass via downcast in
//! ``collect_prepared_plugin_ops``, invokes the inner
//! [`NativePluginImpl::run`] directly, and the impl pushes its
//! ops into the shared [`PreparedOp`] sink — no Python ``GraphOp``
//! instances are ever constructed, no per-op ``extract`` /
//! ``prepare_graph_op`` extraction step.
//!
//! **Scope.** Native plugins are an in-tree fast-path for bundled
//! plugins whose logic is fixed and hot. They're not a public
//! extension mechanism: the trait and its dependent types
//! ([`PreparedOp`], [`ProjectContext`]) are `pub(crate)` by design,
//! the `dead-cst-native` crate is not published to crates.io, and
//! the rust API has no stability commitment. Out-of-tree plugin
//! authors continue to use the Python :class:`Plugin` protocol —
//! they can still write hot code in rust, but they ship it as a
//! pyo3 extension that their `run(ctx)` body calls into, and they
//! emit ops via the public Python ``AddNodeByIdx`` / ``AddEdgeByIdx``
//! / ``AddEntrypointByIdx`` graph ops.
//!
//! Trade-off vs the Python-side plugin path (for in-tree authors):
//!
//! * **Lighter per-op cost** — every Python plugin op pays one
//!   ``Py::new(AddNodeByIdx { ... })`` allocation on the yield side,
//!   then one ``extract::<PyRef<...>>`` + ``.clone()`` of the inner
//!   fields on the harness side. Native plugins skip both ends.
//! * **No GIL release between op yields** — the impl runs entirely
//!   in rust under one GIL hold; rust queries inside the impl can
//!   still drop the GIL via ``py.allow_threads`` if they want to.
//! * **Costs flexibility** — a native plugin's logic is compiled
//!   into the wheel; it can't be authored / overridden / dataclass-
//!   configured externally. Python plugins remain the right home
//!   for anything user-configurable.
//!
//! The frozen-graph contract is identical on both paths: the impl
//! observes the base graph mid-pass, emits ops, ops apply in a
//! single batch after every plugin completes. Native plugins are
//! drop-in interchangeable with Python ones in
//! ``Analysis(plugins=[...])`` and inside the harness's
//! :class:`ThreadPoolExecutor` fan-out.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;
use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::line_index;
use ty_project::Db as ProjectDb;

use crate::builder::PreparedOp;
use crate::file_payload::{file_to_nodes, NodeData};
use crate::graph::{intern_kind, NodeFlags};
use crate::helpers::find_main_block_range;
use crate::project::ProjectContext;

/// Rust-side plugin contract. Each impl describes how to derive ops
/// from a frozen :class:`ProjectContext` and push them into the
/// shared [`PreparedOp`] sink the harness drains at the end of the
/// plugin pass.
///
/// Implementations don't construct Python ``GraphOp`` instances —
/// they push pure-rust [`PreparedOp`] variants directly, skipping
/// the prepare-from-Python round-trip. No ``Python<'_>`` parameter
/// either: every ctx accessor a native plugin needs (``node_attrs``,
/// ``find_main_blocks_indices``, the query DSL helpers) is GIL-free,
/// reading ``Sync`` ``#[pyclass(frozen)]`` data via ``Py::get`` rather
/// than ``Py::borrow``.
pub(crate) trait NativePluginImpl: Send + Sync {
    /// Human-readable name surfaced by ``NativePlugin.name`` and used
    /// for progress reporting (``plugin_start`` / ``plugin_end``
    /// events). Should match the conventional name of the equivalent
    /// Python plugin so existing harness logs read the same.
    //
    // Retained alongside the [`NativePluginKind::ProjectWide`] path for
    // future native plugins whose logic spans files (subclass walks,
    // dispatch handlers). The bundled `MainBlockPlugin` moved to the
    // per-file path, so no impl uses this today.
    #[allow(dead_code)]
    fn name(&self) -> &'static str;

    /// Walk the frozen ``ctx`` and append the plugin's ops to
    /// ``sink``. Same frozen-graph contract as the Python path: the
    /// impl observes the base graph only; its emissions are folded in
    /// by the apply pass after every plugin returns.
    #[allow(dead_code)]
    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()>;
}

// ---------------------------------------------------------------------------
// Per-file native plugins
//
// A *per-file* native plugin is invoked once per project file with a
// restricted [`FileContext`] — it sees only that file's nodes / parsed
// AST, nothing project-wide. The invocation is wrapped in the
// [`per_file_plugin_ops`] salsa-tracked query, so when a file's
// ``file_to_nodes`` / ``parsed_module`` is unchanged across a
// ``re_materialize``, the plugin's cached output is reused with zero
// re-run. Cache soundness comes from the restriction: a per-file
// plugin can only reference nodes in its own file, so its output is a
// pure function of that file's tracked inputs.
//
// Ops are emitted in a *file-local* index space ([`FileLocalOpData`]),
// positions into ``FileNodes(file).refs``. The harness translates
// those to global indices at apply time via the ``ref_to_global`` map.
// ---------------------------------------------------------------------------

/// File-local op produced by a [`PerFileNativePluginImpl`]. Mirrors
/// ``AddNodeByIdx`` but with ``edges_to_local_idx`` as positions into
/// the *file's own* ``FileNodes.refs`` array rather than global graph
/// indices. Salsa-cached as the per-file plugin's output, so it must
/// be pure rust + ``salsa::Update`` (no ``File`` handle, no global
/// idx — both would couple the cache to project-wide assemble order).
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileLocalOpData {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) flags: u32,
    /// Indices into the owning file's ``FileNodes.refs`` array.
    pub(crate) edges_to_local_idx: Vec<u32>,
}

/// Read-only, single-file view handed to a [`PerFileNativePluginImpl`].
/// Deliberately tiny: it exposes only this file's salsa-tracked
/// per-file payload + parsed AST. No project-wide queries, no
/// ``node_attrs(indices)`` over arbitrary nodes — that restriction is
/// what makes the plugin's output a pure function of the file and
/// therefore salsa-cacheable.
pub(crate) struct FileContext<'db> {
    db: &'db dyn ProjectDb,
    file: File,
}

impl<'db> FileContext<'db> {
    fn new(db: &'db dyn ProjectDb, file: File) -> Self {
        Self { db, file }
    }

    /// This file's nodes — index 0 is the synthetic module node, the
    /// rest are top-level decls. Indices line up with [`Self::refs`].
    pub(crate) fn nodes(&self) -> &'db [NodeData] {
        &file_to_nodes(self.db, self.file).nodes
    }

    /// The file's module fqname (``nodes()[0].fqname``).
    pub(crate) fn module_fqname(&self) -> &'db str {
        &file_to_nodes(self.db, self.file).nodes[0].fqname
    }

    /// Local index of the synthetic module node (always 0 — kept as a
    /// named accessor so impls don't hard-code the convention).
    pub(crate) fn module_local_idx(&self) -> u32 {
        0
    }

    /// 1-based ``(start_line, end_line)`` of the byte ``range`` in this
    /// file, via the salsa-cached line index. Used to map a TextRange
    /// (e.g. the ``if __name__`` block) onto the line numbers carried
    /// by [`NodeData`].
    pub(crate) fn line_span(&self, range: ruff_text_size::TextRange) -> (usize, usize) {
        let source = ruff_db::source::source_text(self.db, self.file);
        let idx = line_index(self.db, self.file);
        let start = idx.line_column(range.start(), &source).line.get() as usize;
        let end = idx.line_column(range.end(), &source).line.get() as usize;
        (start, end)
    }

    /// The parsed module for this file (salsa-cached).
    pub(crate) fn parsed(&self) -> ruff_db::parsed::ParsedModuleRef {
        parsed_module(self.db, self.file).load(self.db)
    }
}

/// Rust-side per-file plugin contract. Called once per project file
/// with a restricted [`FileContext`]; pushes [`FileLocalOpData`] into
/// the sink. Pure function of the file's tracked inputs — no
/// project-wide reads, no side effects — so the harness can cache the
/// result in salsa keyed on ``(file, name())``.
pub(crate) trait PerFileNativePluginImpl: Send + Sync {
    /// Walk ``file_ctx`` and append this file's ops to ``sink``.
    /// Naming + salsa keying live on [`PerFilePluginKind`], so the
    /// trait itself only carries the work method.
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOpData>);
}

/// Salsa cache-key discriminant for a per-file plugin. A cheap
/// ``Copy`` enum rather than a ``&'static str`` because salsa tracked
/// function arguments must be owned + ``'static`` (a borrowed key
/// would force ``'db: 'static``). One variant per configless per-file
/// impl; configured per-file plugins would carry a config-hash here
/// (out of scope).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PerFilePluginKind {
    MainBlock,
}

impl PerFilePluginKind {
    /// Human-readable name, matching the equivalent Python plugin.
    fn name(self) -> &'static str {
        match self {
            PerFilePluginKind::MainBlock => "MainBlockPlugin",
        }
    }

    /// Run the concrete impl for this kind against ``file_ctx``.
    fn run_on_file(self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOpData>) {
        match self {
            PerFilePluginKind::MainBlock => MainBlockPluginImpl.run_on_file(file_ctx, sink),
        }
    }
}

/// Salsa-tracked per-file plugin invocation. Keyed on ``(file,
/// kind)``; re-runs only when the file's tracked inputs
/// (``file_to_nodes`` / ``parsed_module`` / ``line_index``) change.
/// Returns the file-local ops the harness translates to global
/// indices at apply time.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn per_file_plugin_ops(
    db: &dyn ProjectDb,
    file: File,
    kind: PerFilePluginKind,
) -> Vec<FileLocalOpData> {
    let file_ctx = FileContext::new(db, file);
    let mut sink = Vec::new();
    kind.run_on_file(&file_ctx, &mut sink);
    sink
}

/// Internal classification of what a [`NativePlugin`] wraps. The
/// harness branches on this: project-wide plugins run once against the
/// whole ``ProjectContext``; per-file plugins run once per file
/// through the salsa-cached [`per_file_plugin_ops`] query.
pub(crate) enum NativePluginKind {
    /// Project-wide native plugin — one ``run`` against the whole
    /// graph. Retained for future non-file-local native plugins; the
    /// bundled `MainBlockPlugin` is per-file, so nothing constructs
    /// this variant today.
    #[allow(dead_code)]
    ProjectWide(Box<dyn NativePluginImpl>),
    PerFile(PerFilePluginKind),
    /// External native plugin loaded from a dylib through the ABI airlock
    /// (see [`load_native_plugins`]). Holds the boxed trait object and a
    /// refcount on the loaded library so its code stays mapped for the
    /// plugin's lifetime. Only meaningful in a `-C prefer-dynamic` build
    /// where the extension and the plugin share one `dead-cst-runtime`.
    External {
        name: String,
        plugin: Box<dyn plugin_api::ExternalPlugin>,
        _lib: Arc<libloading::Library>,
    },
}

/// Python-visible wrapper for a native plugin (project-wide or
/// per-file). Constructed via static factories (e.g.
/// :meth:`NativePlugin.main_block`); ``__init__`` is intentionally
/// unsupported.
///
/// Sendable so the harness can shuttle the handle between
/// :class:`ThreadPoolExecutor` workers alongside Python plugins.
#[pyclass(name = "NativePlugin")]
pub(crate) struct NativePlugin {
    pub(crate) kind: NativePluginKind,
}

#[pymethods]
impl NativePlugin {
    /// Plugin name. Matches the conventional name of the equivalent
    /// Python plugin (e.g. ``"MainBlockPlugin"``), so harness logs
    /// and ``progress_callback`` events look the same whether a
    /// native or Python instance is registered.
    #[getter]
    fn name(&self) -> String {
        match &self.kind {
            NativePluginKind::ProjectWide(inner) => inner.name().to_string(),
            NativePluginKind::PerFile(kind) => kind.name().to_string(),
            NativePluginKind::External { name, .. } => name.clone(),
        }
    }

    /// ``Plugin`` protocol's ``prepare(project_root)`` no-op hook.
    /// Native plugins don't have a pre-graph prepare phase today.
    fn prepare(&self, _project_root: PyObject) {}

    /// Construct the native ``MainBlockPlugin`` — same observable
    /// behaviour as ``dead_cst.plugins.MainBlockPlugin``, but
    /// implemented as a *per-file* plugin: invoked once per file
    /// through a salsa-cached query, so an unchanged file's marker is
    /// reused across ``re_materialize`` with zero re-run.
    #[staticmethod]
    fn main_block() -> Self {
        Self {
            kind: NativePluginKind::PerFile(PerFilePluginKind::MainBlock),
        }
    }
}

// ---------------------------------------------------------------------------
// MainBlockPlugin — first per-file impl. Equivalent to
// ``dead_cst.plugins.main_block.MainBlockPlugin``: emit one synthetic
// ``<__main__>:<module_fqname>`` entrypoint per file with a top-level
// ``if __name__ == "__main__":`` block, with edges to the containing
// module and to every top-level decl inside the block.
// ---------------------------------------------------------------------------

const MAIN_BLOCK_PREFIX: &str = "<__main__>:";

/// Test-only counter: number of times [`MainBlockPluginImpl::run_on_file`]
/// actually executed (i.e. salsa cache *misses* for the per-file
/// query). A salsa hit reuses the cached ops without touching the
/// impl, so this counter stays flat for unchanged files across a
/// ``re_materialize``. Surfaced to tests via
/// :func:`main_block_run_count` / :func:`reset_main_block_run_count`.
static MAIN_BLOCK_RUN_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Test helper — total `MainBlockPluginImpl::run_on_file` executions
/// since the last reset. Lets the cache-behaviour test assert that an
/// unchanged main-block file isn't re-run on ``re_materialize``.
#[pyfunction]
pub(crate) fn _main_block_run_count() -> usize {
    MAIN_BLOCK_RUN_COUNT.load(Ordering::Relaxed)
}

/// Test helper — zero the [`MAIN_BLOCK_RUN_COUNT`] counter.
#[pyfunction]
pub(crate) fn _reset_main_block_run_count() {
    MAIN_BLOCK_RUN_COUNT.store(0, Ordering::Relaxed);
}

pub(crate) struct MainBlockPluginImpl;

impl PerFileNativePluginImpl for MainBlockPluginImpl {
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOpData>) {
        MAIN_BLOCK_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
        let parsed = file_ctx.parsed();
        let Some(block_range) = find_main_block_range(&parsed) else {
            return;
        };
        let (block_start_line, block_end_line) = file_ctx.line_span(block_range);
        let nodes = file_ctx.nodes();
        // Local indices of every top-level decl whose source span falls
        // inside the ``if __name__`` block. Skip index 0 (the module
        // node) — it's added explicitly below as the first edge.
        let mut edges_to_local_idx: Vec<u32> = vec![file_ctx.module_local_idx()];
        for (local_idx, node) in nodes.iter().enumerate().skip(1) {
            if node.start_line >= block_start_line && node.end_line <= block_end_line {
                edges_to_local_idx.push(local_idx as u32);
            }
        }
        let synthetic_kind = intern_kind("synthetic").expect("'synthetic' is a valid kind");
        sink.push(FileLocalOpData {
            fqname: format!("{MAIN_BLOCK_PREFIX}{}", file_ctx.module_fqname()),
            kind: synthetic_kind,
            flags: NodeFlags::ENTRYPOINT,
            edges_to_local_idx,
        });
    }
}

// ===========================================================================
// External (dylib) native plugins
//
// An external native plugin is compiled in a *separate* crate that links the
// `dead-cst-runtime` *dylib* (under `-C prefer-dynamic`), so the plugin and
// the extension module share one runtime — one salsa db, one set of types.
// The plugin ships as a cdylib exporting a C-ABI manifest; the host loads it
// through an airlock that gates on an ABI fingerprint before touching any
// version-hashed runtime symbol.
//
// This is the sound version of an out-of-tree native plugin: it gives full
// rust-type fidelity (the plugin runs against a real `ProjectContext`) at
// the cost of recompiling against each runtime release — enforced by the
// fingerprint gate, which rejects a stale `.so` cleanly rather than crashing.
// ===========================================================================

/// The curated public API an external plugin crate compiles against. Kept
/// deliberately small: a plugin sees a restricted [`PluginCtx`] view of the
/// frozen graph and emits ops through [`PluginOps`] — it never touches the
/// internal `PreparedOp` / `ProjectContext` types directly, so the surface
/// an out-of-tree author depends on stays narrow.
pub mod plugin_api {
    use crate::builder::PreparedOp;
    use crate::project::ProjectContext;

    /// Contract an external native plugin implements. The host calls
    /// [`run`](ExternalPlugin::run) once per materialize against the frozen
    /// graph, then folds the emitted ops in as a single batch.
    pub trait ExternalPlugin: Send + Sync {
        /// Human-readable name (surfaced in progress logs).
        fn name(&self) -> &str;
        /// Inspect `ctx` and append ops to `ops`. Same frozen-graph
        /// contract as the in-tree native path.
        fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps);
    }

    /// Restricted, mostly-stable view of the frozen graph for external
    /// plugins. Wraps the internal `ProjectContext`; exposes only the
    /// index-based queries a plugin needs.
    pub struct PluginCtx<'a> {
        inner: &'a ProjectContext,
    }

    impl<'a> PluginCtx<'a> {
        pub(crate) fn new(inner: &'a ProjectContext) -> Self {
            Self { inner }
        }

        /// Each top-level `if __name__ == "__main__":` block as
        /// `(module_node_idx, [decl_node_idx, ...])`.
        pub fn main_blocks(&self) -> Vec<(usize, Vec<usize>)> {
            self.inner.find_main_blocks_indices().unwrap_or_default()
        }
    }

    /// Op sink for external plugins. Wraps the internal `PreparedOp` vec so
    /// plugins emit through named methods instead of constructing internals.
    pub struct PluginOps {
        sink: Vec<PreparedOp>,
    }

    impl PluginOps {
        pub(crate) fn new() -> Self {
            Self { sink: Vec::new() }
        }

        pub(crate) fn into_inner(self) -> Vec<PreparedOp> {
            self.sink
        }

        /// Keep `decl_idx` reachable via a synthetic entrypoint named
        /// `marker`. Mirrors the in-tree `AddEntrypointByIdx` graph op.
        pub fn keep_alive(&mut self, decl_idx: usize, marker: String) {
            self.sink
                .push(PreparedOp::EntrypointByIdx { decl_idx, marker });
        }
    }
}

/// ABI fingerprint this runtime accepts (see `build.rs`). An external plugin
/// bakes this exact string at compile time; the airlock rejects any plugin
/// whose baked fingerprint differs.
pub const PLUGIN_ABI_FINGERPRINT: &str = env!("RUNTIME_ABI_FINGERPRINT");

/// Magic number prefixing a valid plugin manifest.
pub const PLUGIN_MANIFEST_MAGIC: u64 = 0xDEAD_C570_0001;

/// One entry per plugin a dylib provides (N per dylib).
#[repr(C)]
pub struct PluginDesc {
    pub name: *const u8,
    pub name_len: usize,
    /// Constructs the plugin; returns `*mut Box<dyn ExternalPlugin>`.
    pub make: extern "C" fn() -> *mut c_void,
}

/// The self-contained airlock surface a plugin exposes via the
/// `_dead_cst_plugin_manifest_v1` symbol. Built from plain data + inlined
/// consts only — never a hashed runtime call — so even an ABI-incompatible
/// plugin can expose it for inspection before any version-hashed symbol is
/// touched.
#[repr(C)]
pub struct PluginManifest {
    pub magic: u64,
    pub abi_fingerprint: *const u8,
    pub abi_fingerprint_len: usize,
    pub plugins: *const PluginDesc,
    pub plugins_len: usize,
}

/// Load external native plugins from a dylib at `path` through the ABI
/// airlock. Returns one [`NativePlugin`] per plugin the dylib provides; each
/// is drop-in usable in ``Analysis(plugins=[...])`` alongside in-tree
/// plugins. Raises (clean rejection) on a missing manifest, bad magic, or an
/// ABI-fingerprint mismatch — never crashes the host.
#[pyfunction]
pub(crate) fn load_native_plugins(path: String) -> PyResult<Vec<NativePlugin>> {
    let err = pyo3::exceptions::PyRuntimeError::new_err::<String>;
    // SAFETY: libloading opens with RTLD_LAZY | RTLD_LOCAL, so unresolved
    // hashed symbols don't fault the load; we read the self-contained
    // manifest and gate on the fingerprint before calling any plugin code.
    unsafe {
        let lib =
            libloading::Library::new(&path).map_err(|e| err(format!("dlopen {path}: {e}")))?;

        let manifest_fn: libloading::Symbol<extern "C" fn() -> *const PluginManifest> =
            lib.get(b"_dead_cst_plugin_manifest_v1\0").map_err(|_| {
                err(format!(
                    "{path}: not a dead-cst plugin (no _dead_cst_plugin_manifest_v1) \
                     — or built before the manifest ABI; rebuild against this release"
                ))
            })?;
        let m = &*manifest_fn();
        if m.magic != PLUGIN_MANIFEST_MAGIC {
            return Err(err(format!(
                "{path}: bad manifest magic 0x{:x} (expected 0x{:x})",
                m.magic, PLUGIN_MANIFEST_MAGIC
            )));
        }
        let fp = std::str::from_utf8(std::slice::from_raw_parts(
            m.abi_fingerprint,
            m.abi_fingerprint_len,
        ))
        .unwrap_or("<invalid utf8>");
        if fp != PLUGIN_ABI_FINGERPRINT {
            return Err(err(format!(
                "{path}: ABI mismatch — plugin built against '{fp}', this runtime is \
                 '{PLUGIN_ABI_FINGERPRINT}'. Rebuild the plugin against this release."
            )));
        }

        // Accepted: instantiate each plugin. Hold a refcount on the library
        // so its code stays mapped for the plugins' lifetime.
        let lib = Arc::new(lib);
        let descs = std::slice::from_raw_parts(m.plugins, m.plugins_len);
        let mut out = Vec::with_capacity(descs.len());
        for d in descs {
            let name = std::str::from_utf8(std::slice::from_raw_parts(d.name, d.name_len))
                .unwrap_or("<?>")
                .to_string();
            let raw = (d.make)();
            let plugin_box: Box<Box<dyn plugin_api::ExternalPlugin>> =
                Box::from_raw(raw as *mut Box<dyn plugin_api::ExternalPlugin>);
            out.push(NativePlugin {
                kind: NativePluginKind::External {
                    name,
                    plugin: *plugin_box,
                    _lib: Arc::clone(&lib),
                },
            });
        }
        Ok(out)
    }
}
