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

/// File-local op produced by a [`PerFileNativePluginImpl`]. Indices
/// are positions into the *file's own* ``FileNodes.refs`` array rather
/// than global graph indices; the harness translates them to global
/// indices at apply time via the build's ``local_to_global`` map (a
/// local idx with no global entry is skipped). Salsa-cached as the
/// per-file plugin's output, so every field is pure rust +
/// ``salsa::Update`` (no ``File`` handle, no global idx — both would
/// couple the cache to project-wide assemble order).
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) enum FileLocalOp {
    /// Add a synthetic node with edges to / from file-local nodes.
    /// Mirrors ``PreparedOp::NodeByIdx`` in file-local index space.
    Node {
        fqname: String,
        kind: &'static str,
        flags: u32,
        /// Positions into the file's ``FileNodes.refs`` the new node
        /// points *at* (``node -> target``).
        edges_to_local_idx: Vec<u32>,
        /// Positions into the file's ``FileNodes.refs`` that point *at*
        /// the new node (``source -> node``).
        edges_from_local_idx: Vec<u32>,
    },
    /// Add an edge between two file-local nodes (``src -> dst``).
    /// Mirrors ``PreparedOp::EdgeByIdx``.
    Edge {
        src_local_idx: u32,
        dst_local_idx: u32,
    },
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

    /// The file's enclosing package name (``None`` for a top-level
    /// module). Needed to resolve relative imports (``from .pkg import
    /// App``) the same way the project-wide queries do.
    pub(crate) fn package(&self) -> Option<String> {
        crate::ingest::file_package_name(self.db, self.file)
    }

    /// File-local index of the live binding named ``name`` (the last
    /// reaching one when a name is rebound), or ``None`` if the file
    /// has no module-level binding of that name. This is the per-file
    /// analog of the project-wide ``vars_by_file`` lookup: it maps the
    /// textual owner of a ``@owner.deco`` decorator or the target of a
    /// ``owner = App(...)`` assignment onto its node in *this* file
    /// without any cross-file resolution. Backed by ty's
    /// end-of-scope bindings (``FileNodes.exports_by_name``), so it
    /// stays a pure function of the file.
    pub(crate) fn local_idx_for_name(&self, name: &str) -> Option<u32> {
        file_to_nodes(self.db, self.file)
            .exports_by_name
            .get(name)
            .and_then(|locals| locals.last().copied())
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
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>);
}

/// Salsa cache-key discriminant for a per-file plugin. A cheap
/// ``Copy`` enum rather than a ``&'static str`` because salsa tracked
/// function arguments must be owned + ``'static`` (a borrowed key
/// would force ``'db: 'static``).
///
/// A *configured* per-file plugin can't store its config inline (the
/// config — string lists, maps — is neither ``Copy`` nor cheap to hash
/// per file). Instead it carries a ``u32`` handle into the
/// process-global [`DISPATCH_CONFIGS`] registry: the config is
/// registered once when the Python plugin is constructed and is
/// thereafter immutable, so the handle is a sound, stable salsa key
/// (handle ``N`` always denotes the same config).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PerFilePluginKind {
    MainBlock,
    /// Per-file dispatch-app plugin, parameterised by the config at
    /// [`DISPATCH_CONFIGS`]``[id]``.
    DispatchApp(u32),
}

impl PerFilePluginKind {
    /// Human-readable name, matching the equivalent Python plugin.
    fn name(self) -> &'static str {
        match self {
            PerFilePluginKind::MainBlock => "MainBlockPlugin",
            PerFilePluginKind::DispatchApp(_) => "DispatchAppPlugin",
        }
    }

    /// Run the concrete impl for this kind against ``file_ctx``.
    fn run_on_file(self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>) {
        match self {
            PerFilePluginKind::MainBlock => MainBlockPluginImpl.run_on_file(file_ctx, sink),
            PerFilePluginKind::DispatchApp(id) => {
                DispatchAppPluginImpl::new(id).run_on_file(file_ctx, sink)
            }
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
) -> Vec<FileLocalOp> {
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

    /// Construct a *per-file* native dispatch-app plugin — the native
    /// counterpart of ``dead_cst.plugins.DispatchAppPlugin`` for the
    /// portion of the work that is genuinely per-file (direct
    /// construction promotion + same-file ``@app.deco`` handler
    /// wiring). Invoked once per file through a salsa-cached query, so
    /// an unchanged file's dispatch wiring is reused across
    /// ``re_materialize`` with zero re-run.
    ///
    /// ``module_to_names`` maps each framework module to the
    /// constructor bare-names treated as app classes — **already
    /// expanded over the subclass closure by the caller**, since that
    /// expansion needs the project-wide class hierarchy and so can't be
    /// per-file. The cross-file factory walk
    /// (``app = create_app()``) is likewise out of scope here; both
    /// remain on the Python plugin. See ``NATIVE_PLUGINS.md``.
    #[staticmethod]
    #[pyo3(signature = (marker_prefix, module_to_names, registration_decorators, seed_as_entrypoint))]
    fn dispatch_app(
        marker_prefix: String,
        module_to_names: std::collections::HashMap<String, Vec<String>>,
        registration_decorators: Vec<String>,
        seed_as_entrypoint: bool,
    ) -> Self {
        let id = register_dispatch_config(DispatchConfigData {
            marker_prefix,
            module_to_names: module_to_names.into_iter().collect(),
            registration_decorators,
            seed_as_entrypoint,
        });
        Self {
            kind: NativePluginKind::PerFile(PerFilePluginKind::DispatchApp(id)),
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
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>) {
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
        sink.push(FileLocalOp::Node {
            fqname: format!("{MAIN_BLOCK_PREFIX}{}", file_ctx.module_fqname()),
            kind: synthetic_kind,
            flags: NodeFlags::ENTRYPOINT,
            edges_to_local_idx,
            edges_from_local_idx: Vec::new(),
        });
    }
}

// ---------------------------------------------------------------------------
// DispatchAppPlugin — per-file native counterpart of
// ``dead_cst.plugins.decl_shapes.DispatchAppPlugin``.
//
// Operates on a *resolved* ``DispatchAppSpec``: the cross-file work
// (expanding ``app_classes`` over the subclass closure into a
// ``{module: {ctor_name, ...}}`` map) is the caller's job, because it
// needs the project-wide class hierarchy and so can't be a pure
// function of one file. What *is* per-file — and therefore lives here,
// salsa-cached — is everything downstream of that map:
//
//   * direct construction promotion  — ``app = App(...)`` → entrypoint
//     (when ``seed_as_entrypoint``);
//   * handler wiring                  — ``@app.deco(...)`` → the
//     same-file ``app`` binding.
//
// Deliberately *not* handled (genuinely cross-file, stays on the
// Python plugin): subclass-closure expansion of ``app_classes`` and
// the factory walk (``app = create_app()`` where ``create_app``
// returns an app instance — promoting it needs the factory decl's
// cross-file predecessors). See ``NATIVE_PLUGINS.md``.
// ---------------------------------------------------------------------------

/// Resolved, immutable configuration for one per-file dispatch plugin.
/// The string-heavy fields are why this lives in a side registry
/// instead of inline in [`PerFilePluginKind`]: a per-file salsa key
/// must be cheap-`Copy`, so the key carries only a [`u32`] handle into
/// [`DISPATCH_CONFIGS`] and the impl resolves the config from there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DispatchConfigData {
    /// Short label for the synthetic ``<{marker}-app>:`` fqnames.
    pub(crate) marker_prefix: String,
    /// ``module -> [ctor bare-name, ...]`` — already expanded over the
    /// subclass closure by the caller.
    pub(crate) module_to_names: Vec<(String, Vec<String>)>,
    /// Per-instance registration decorator attribute names
    /// (``route`` / ``get`` / ``task`` / ...).
    pub(crate) registration_decorators: Vec<String>,
    /// Promote each direct construction to an entrypoint (factory-aware
    /// frameworks); when false, only wire handlers (pure dispatch).
    pub(crate) seed_as_entrypoint: bool,
}

/// Process-global registry of resolved dispatch configs. Append-only:
/// a handle is minted once per ``NativePlugin.dispatch_app(...)`` call
/// and never reused or mutated, so using it as a salsa cache key is
/// sound (handle ``N`` always denotes the same config for the life of
/// the process). The salsa db is per-``ProjectContext``, so handles
/// only need to be stable, not unique across analyses.
static DISPATCH_CONFIGS: std::sync::OnceLock<std::sync::RwLock<Vec<Arc<DispatchConfigData>>>> =
    std::sync::OnceLock::new();

/// Register a resolved config and return its stable handle.
fn register_dispatch_config(cfg: DispatchConfigData) -> u32 {
    let reg = DISPATCH_CONFIGS.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    let mut guard = reg.write().expect("DISPATCH_CONFIGS poisoned");
    let id = guard.len() as u32;
    guard.push(Arc::new(cfg));
    id
}

/// Resolve a config handle minted by [`register_dispatch_config`].
fn dispatch_config(id: u32) -> Arc<DispatchConfigData> {
    DISPATCH_CONFIGS
        .get()
        .expect("DISPATCH_CONFIGS uninitialised")
        .read()
        .expect("DISPATCH_CONFIGS poisoned")[id as usize]
        .clone()
}

/// Test-only counter — number of times [`DispatchAppPluginImpl::run_on_file`]
/// actually executed (salsa cache *misses*). Lets the cache-behaviour
/// test assert an unchanged file isn't re-walked across a
/// ``re_materialize``.
static DISPATCH_RUN_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Test helper — total `DispatchAppPluginImpl::run_on_file` executions.
#[pyfunction]
pub(crate) fn _dispatch_app_run_count() -> usize {
    DISPATCH_RUN_COUNT.load(Ordering::Relaxed)
}

/// Test helper — zero the [`DISPATCH_RUN_COUNT`] counter.
#[pyfunction]
pub(crate) fn _reset_dispatch_app_run_count() {
    DISPATCH_RUN_COUNT.store(0, Ordering::Relaxed);
}

pub(crate) struct DispatchAppPluginImpl {
    config: Arc<DispatchConfigData>,
}

impl DispatchAppPluginImpl {
    fn new(id: u32) -> Self {
        Self {
            config: dispatch_config(id),
        }
    }
}

impl PerFileNativePluginImpl for DispatchAppPluginImpl {
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>) {
        DISPATCH_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
        let cfg = &*self.config;
        let parsed = file_ctx.parsed();
        let package = file_ctx.package();

        // Flatten the resolved map into the (modules, allowed-names)
        // shape the shared per-file matchers expect.
        let modules: Vec<String> = cfg.module_to_names.iter().map(|(m, _)| m.clone()).collect();
        let allowed: rustc_hash::FxHashSet<&str> = cfg
            .module_to_names
            .iter()
            .flat_map(|(_, names)| names.iter().map(String::as_str))
            .collect();
        let reg_attrs: rustc_hash::FxHashSet<&str> = cfg
            .registration_decorators
            .iter()
            .map(String::as_str)
            .collect();

        // --- Pass 1: direct constructions ``owner = Ctor(...)``. -------
        // Resolve the framework imports in *this* file, then match each
        // top-level ``owner = Ctor(...)`` whose callee binds to one of
        // the allowed ctor names. ``direct_owners`` maps the owner's
        // simple name to its file-local idx (the same name a
        // ``@owner.deco`` decorator references).
        let mut direct_owners: rustc_hash::FxHashMap<String, u32> =
            rustc_hash::FxHashMap::default();
        let imports = crate::helpers::collect_modules_imports_local(
            &parsed,
            &modules,
            &allowed,
            package.as_deref(),
        );
        if !imports.is_empty() {
            for stmt in &parsed.syntax().body {
                let Some((owner_name, value)) = top_level_assign_target_name(stmt) else {
                    continue;
                };
                let ruff_python_ast::Expr::Call(call) = value else {
                    continue;
                };
                if crate::helpers::matched_call_target_any(call, &imports, &modules, &allowed)
                    .is_none()
                {
                    continue;
                }
                let Some(owner_idx) = file_ctx.local_idx_for_name(owner_name) else {
                    continue;
                };
                direct_owners.insert(owner_name.to_string(), owner_idx);
                if cfg.seed_as_entrypoint {
                    let synthetic_kind =
                        intern_kind("synthetic").expect("'synthetic' is a valid kind");
                    sink.push(FileLocalOp::Node {
                        fqname: format!(
                            "<{}-app>:{}.{}",
                            cfg.marker_prefix,
                            file_ctx.module_fqname(),
                            owner_name
                        ),
                        kind: synthetic_kind,
                        flags: NodeFlags::ENTRYPOINT,
                        edges_to_local_idx: vec![owner_idx],
                        edges_from_local_idx: Vec::new(),
                    });
                }
            }
        }

        // --- Pass 2: handler wiring ``@owner.deco(...)``. --------------
        // For each top-level function decorated with ``@owner.<attr>``
        // (attr a registration decorator), wire the owner binding ->
        // the handler so reachability flows app -> handler.
        //
        //   * seed_as_entrypoint=True  (factory-aware): wire to any
        //     same-file binding named ``owner`` (mirrors the Python
        //     plugin's ``vars_by_file`` path, so ``app = create_app()``
        //     handlers still wire even though the construction itself
        //     isn't matched here).
        //   * seed_as_entrypoint=False (pure dispatch): wire only to an
        //     owner that came from a *direct construction* above, so a
        //     star-imported ``app = App()`` the matcher can't see stays
        //     invisible — same conservative rule as the Python plugin.
        if reg_attrs.is_empty() {
            return;
        }
        for stmt in &parsed.syntax().body {
            let ruff_python_ast::Stmt::FunctionDef(func) = stmt else {
                continue;
            };
            let mut seen_owners: rustc_hash::FxHashSet<&str> = rustc_hash::FxHashSet::default();
            for dec in &func.decorator_list {
                let root = match &dec.expression {
                    ruff_python_ast::Expr::Call(call) => &*call.func,
                    other => other,
                };
                let root = crate::helpers::unwrap_subscripted_callee(root);
                let ruff_python_ast::Expr::Attribute(attr) = root else {
                    continue;
                };
                if !reg_attrs.contains(attr.attr.as_str()) {
                    continue;
                }
                let ruff_python_ast::Expr::Name(owner) = attr.value.as_ref() else {
                    continue;
                };
                let owner_name = owner.id.as_str();
                if !seen_owners.insert(owner_name) {
                    continue;
                }
                let owner_idx = if cfg.seed_as_entrypoint {
                    file_ctx.local_idx_for_name(owner_name)
                } else {
                    direct_owners.get(owner_name).copied()
                };
                let Some(owner_idx) = owner_idx else {
                    continue;
                };
                let Some(handler_idx) = file_ctx.local_idx_for_name(func.name.as_str()) else {
                    continue;
                };
                sink.push(FileLocalOp::Edge {
                    src_local_idx: owner_idx,
                    dst_local_idx: handler_idx,
                });
            }
        }
    }
}

/// Top-level ``owner = <value>`` / ``owner: T = <value>`` with a single
/// bare-`Name` target — returns ``(owner_name, value_expr)``. Mirrors
/// the acceptance rules of ``helpers::top_level_assign_to_name`` (which
/// returns the target's range) but yields the name string directly, so
/// the dispatch matcher can look it up in
/// [`FileContext::local_idx_for_name`] without a range→text slice.
fn top_level_assign_target_name(
    stmt: &ruff_python_ast::Stmt,
) -> Option<(&str, &ruff_python_ast::Expr)> {
    match stmt {
        ruff_python_ast::Stmt::Assign(assign) => {
            let [target] = assign.targets.as_slice() else {
                return None;
            };
            let ruff_python_ast::Expr::Name(name) = target else {
                return None;
            };
            Some((name.id.as_str(), assign.value.as_ref()))
        }
        ruff_python_ast::Stmt::AnnAssign(ann) => {
            let value = ann.value.as_deref()?;
            let ruff_python_ast::Expr::Name(name) = ann.target.as_ref() else {
                return None;
            };
            Some((name.id.as_str(), value))
        }
        _ => None,
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
