//! Native (rust-side) plugins — the only plugin mechanism.
//!
//! A native plugin is a rust implementation of the plugin contract:
//! the harness detects a :class:`NativePlugin` pyclass via downcast in
//! ``collect_prepared_plugin_ops``, invokes the inner
//! [`NativePluginImpl::run`] directly, and the impl pushes its ops into
//! the shared [`PreparedOp`] sink the harness drains at the end of the
//! plugin pass. There is no Python ``.run(ctx)`` call and no per-op
//! ``extract`` round-trip — the impl emits pure-rust [`PreparedOp`]
//! variants directly.
//!
//! **Scope.** The trait and its dependent types ([`PreparedOp`],
//! [`ProjectContext`]) are `pub(crate)` by design, the
//! `dead-cst-native` crate is not published to crates.io, and the rust
//! API has no stability commitment. In-tree plugins implement
//! [`NativePluginImpl`] directly; out-of-tree authors ship an external
//! native plugin compiled against the shipped runtime dylib and loaded
//! via [`load_native_plugins`] (see ``NATIVE_PLUGINS.md``).
//!
//! Design properties:
//!
//! * **Pure-rust ops** — the impl pushes [`PreparedOp`] variants
//!   straight into the sink; no ``Py::new`` per op and no
//!   ``extract::<PyRef<...>>`` on the harness side.
//! * **One GIL hold** — the impl runs entirely in rust under a single
//!   GIL hold; rust queries inside the impl can still drop the GIL via
//!   ``py.allow_threads`` when they want to.
//! * **Fixed at build time** — a native plugin's logic is compiled into
//!   the wheel; configuration is supplied through its factory (e.g.
//!   ``NativePlugin::server_config(filenames=...)``).
//!
//! Frozen-graph contract: the impl observes the base graph mid-pass,
//! emits ops, and ops apply in a single batch after every plugin
//! completes. Native plugins run in ``Analysis(plugins=[...])`` and
//! inside the harness's :class:`ThreadPoolExecutor` fan-out.

use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;
use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::line_index;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::{Expr, Stmt};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use ty_project::Db as ProjectDb;

use crate::builder::PreparedOp;
use crate::file_payload::{file_to_nodes, NodeData, NodeKind};
use crate::graph::{intern_kind, EdgeFlags, NodeFlags};
use crate::helpers::{
    collect_modules_imports_local, decorators_match_imports, find_main_block_range, is_dunder_name,
    matched_call_target_any, top_level_assign_to_name, ArgValue, CallArgs, FactoryCallFinder,
};
use crate::ingest::file_package_name;
use crate::project::ProjectContext;

/// Rust-side plugin contract. Each impl describes how to derive ops
/// from a frozen :class:`ProjectContext` and push them into the
/// shared [`PreparedOp`] sink the harness drains at the end of the
/// plugin pass.
///
/// Implementations push pure-rust [`PreparedOp`] variants directly
/// into the sink. No ``Python<'_>`` parameter
/// either: every ctx accessor a native plugin needs (``node_attrs``,
/// ``find_main_blocks_indices``, the ``find_*`` query helpers) is GIL-free,
/// reading ``Sync`` ``#[pyclass(frozen)]`` data via ``Py::get`` rather
/// than ``Py::borrow``.
pub(crate) trait NativePluginImpl: Send + Sync {
    /// Human-readable name surfaced by ``NativePlugin.name`` and used
    /// for progress reporting (``plugin_start`` / ``plugin_end``
    /// events). Should match the conventional name of the equivalent
    /// Python plugin so existing harness logs read the same.
    //
    // Borrowed (not `&'static`) so an impl can own a runtime-built name —
    // the custom `DispatchAppPluginImpl` carries a caller-supplied `String`.
    // The bundled impls still return string literals.
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Walk the frozen ``ctx`` and append the plugin's ops to
    /// ``sink``. Same frozen-graph contract as the Python path: the
    /// impl observes the base graph only; its emissions are folded in
    /// by the apply pass after every plugin returns.
    #[allow(dead_code)]
    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()>;

    /// Pre-graph hook, mirroring the Python ``Plugin.prepare`` contract:
    /// called once with the project root before any graph construction, so
    /// the impl can scan for config files / framework manifests. Default
    /// no-op. Runs before the graph exists, so it must not touch
    /// ``ProjectContext``.
    #[allow(dead_code)]
    fn prepare(&self, _project_root: &str) {}
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
// Ops are emitted in a *file-local* index space ([`FileLocalOp`]),
// positions into ``FileNodes(file).refs``. The harness translates
// those to global indices at apply time via the ``ref_to_global`` map.
// ---------------------------------------------------------------------------

/// File-local op produced by a [`PerFileNativePluginImpl`]. Every
/// endpoint is a *file-local* index (position in the file's own
/// ``FileNodes.refs`` array), never a global graph index — the harness
/// translates to global indices at apply time. Salsa-cached as the
/// per-file plugin's output, so it must be pure rust + ``salsa::Update``
/// (no ``File`` handle, no global idx — both would couple the cache to
/// project-wide assemble order). Mirrors the three project-wide
/// [`PreparedOp`] variants, restricted to a single file's index space.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) enum FileLocalOp {
    /// Mint a synthetic node in this file (`PreparedOp::Node` shape) with an
    /// out-edge to each file-local index in ``edges_to_local_idx`` and an
    /// in-edge from each file-local index in ``edges_from_local_idx``.
    Node {
        fqname: String,
        kind: &'static str,
        flags: u32,
        /// File-local indices this node points *to* (out-edges).
        edges_to_local_idx: Vec<u32>,
        /// File-local indices that point *to* this node (in-edges).
        edges_from_local_idx: Vec<u32>,
    },
    /// Reachability edge between two nodes in this file (`PreparedOp::Edge`
    /// shape). Both endpoints are file-local indices.
    Edge {
        src_local_idx: u32,
        dst_local_idx: u32,
        flags: u32,
    },
    /// Keep a node in this file alive via a synthetic entrypoint marker
    /// (`PreparedOp::Entrypoint` shape). ``decl_local_idx`` is a file-local
    /// index.
    Entrypoint { decl_local_idx: u32, marker: String },
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

    // --- file-local query helpers ------------------------------------------
    //
    // Ready-made matching so a per-file plugin needn't re-implement the
    // import / decorator / construction / call AST walk by hand. Each
    // returns *file-local* indices (positions in [`Self::nodes`]) — exactly
    // the index space [`FileLocalOp`] / [`plugin_api::FileOps`] emit against.
    // All are pure functions of the file's tracked inputs (parsed AST +
    // nodes), so they preserve the per-file salsa-cache contract. They reuse
    // the same decorator / construction / call matchers the project-wide
    // ``ProjectContext`` finders are built on, so a per-file plugin and its
    // project-wide twin match identically.

    /// This file's enclosing package, for resolving relative imports.
    fn file_package(&self) -> Option<String> {
        file_package_name(self.db, self.file)
    }

    /// File-local index of the top-level decl whose name occupies
    /// `name_range`, found by source line. Top-level decls never share a
    /// line, so the line is a unique key (a decl's [`NodeData::start_line`]
    /// is snapped to its keyword/name line). The module node (index 0) and
    /// `import` nodes are skipped so an assignment / def on an import's line
    /// can't be misattributed.
    fn local_idx_for_name_range(&self, name_range: TextRange) -> Option<u32> {
        let (line, _) = self.line_span(name_range);
        self.nodes()
            .iter()
            .position(|n| {
                n.start_line == line && !matches!(n.kind, NodeKind::Module | NodeKind::Import)
            })
            .map(|i| i as u32)
    }

    /// Merged ``{local_name -> upstream_target}`` import map for `names`
    /// imported from any of `modules`. Empty when this file imports nothing
    /// matching — callers use that as the cheap early-out.
    fn matching_imports(
        &self,
        parsed: &ruff_db::parsed::ParsedModuleRef,
        modules: &[String],
        names: &FxHashSet<&str>,
    ) -> rustc_hash::FxHashMap<String, String> {
        collect_modules_imports_local(parsed, modules, names, self.file_package().as_deref())
    }

    /// True if this file imports any of `modules` — the per-file twin of
    /// the project-wide ``ProjectContext::has_imports_of``. A presence guard
    /// a plugin can check before doing heavier per-file work.
    pub(crate) fn imports_any_module(&self, modules: &[&str]) -> bool {
        let parsed = self.parsed();
        let file_package = self.file_package();
        parsed.syntax().body.iter().any(|stmt| match stmt {
            Stmt::Import(im) => im.names.iter().any(|alias| {
                let name = alias.name.as_str();
                modules
                    .iter()
                    .any(|m| name == *m || name.starts_with(&format!("{m}.")))
            }),
            Stmt::ImportFrom(im) => {
                let tail = im.module.as_ref().map(|n| n.as_str()).unwrap_or("");
                let absolute = if im.level == 0 {
                    if tail.is_empty() {
                        return false;
                    }
                    tail.to_string()
                } else {
                    match crate::helpers::resolve_relative_import(
                        im.level,
                        tail,
                        file_package.as_deref(),
                    ) {
                        Some(a) => a,
                        None => return false,
                    }
                };
                modules.iter().any(|m| {
                    absolute == *m
                        || absolute.starts_with(&format!("{m}."))
                        || m.starts_with(&format!("{absolute}."))
                })
            }
            _ => false,
        })
    }

    /// File-local indices of top-level function / class decls carrying a
    /// decorator that resolves (through this file's imports) to one of
    /// `names` imported from one of `modules`. The per-file twin of the
    /// project-wide ``ProjectContext::find_decorated_decls``.
    pub(crate) fn decorated_decls(&self, modules: &[&str], names: &[&str]) -> Vec<u32> {
        let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
        let names_set: FxHashSet<&str> = names.iter().copied().collect();
        let parsed = self.parsed();
        let imports = self.matching_imports(&parsed, &modules_owned, &names_set);
        if imports.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for stmt in &parsed.syntax().body {
            let (decorators, name_range) = match stmt {
                Stmt::FunctionDef(f) => (&f.decorator_list, f.name.range()),
                Stmt::ClassDef(c) => (&c.decorator_list, c.name.range()),
                _ => continue,
            };
            if decorators_match_imports(decorators, &imports, &names_set).is_some() {
                if let Some(idx) = self.local_idx_for_name_range(name_range) {
                    out.push(idx);
                }
            }
        }
        out
    }

    /// File-local indices of top-level ``X = Ctor(...)`` / ``X: T =
    /// Ctor(...)`` variable decls whose `Ctor` resolves to one of `names`
    /// imported from one of `modules`. The per-file twin of the project-wide
    /// ``ProjectContext::find_instance_constructions``.
    pub(crate) fn constructions(&self, modules: &[&str], names: &[&str]) -> Vec<u32> {
        let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
        let names_set: FxHashSet<&str> = names.iter().copied().collect();
        let parsed = self.parsed();
        let imports = self.matching_imports(&parsed, &modules_owned, &names_set);
        if imports.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for stmt in &parsed.syntax().body {
            let Some((name_range, value)) = top_level_assign_to_name(stmt) else {
                continue;
            };
            let Expr::Call(call) = value else {
                continue;
            };
            if matched_call_target_any(call, &imports, &modules_owned, &names_set).is_some() {
                if let Some(idx) = self.local_idx_for_name_range(name_range) {
                    out.push(idx);
                }
            }
        }
        out
    }

    /// File-local indices of the decls whose body contains a call to one of
    /// `names` imported from one of `modules`. A top-level function / class
    /// owns calls anywhere in its subtree; module-scope calls attribute to
    /// the module node (index 0). Returns each owner once.
    pub(crate) fn calls(&self, modules: &[&str], names: &[&str]) -> Vec<u32> {
        let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
        let names_set: FxHashSet<&str> = names.iter().copied().collect();
        let parsed = self.parsed();
        let imports = self.matching_imports(&parsed, &modules_owned, &names_set);
        if imports.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut push_owner = |idx: u32, out: &mut Vec<u32>| {
            if seen.insert(idx) {
                out.push(idx);
            }
        };
        for stmt in &parsed.syntax().body {
            let (name_range, body): (Option<TextRange>, &[Stmt]) = match stmt {
                Stmt::FunctionDef(f) => (Some(f.name.range()), &f.body),
                Stmt::ClassDef(c) => (Some(c.name.range()), &c.body),
                other => {
                    let mut finder = FactoryCallFinder {
                        imports: &imports,
                        modules: &modules_owned,
                        allowed: &names_set,
                        kinds: FxHashSet::default(),
                    };
                    finder.visit_stmt(other);
                    if !finder.kinds.is_empty() {
                        push_owner(self.module_local_idx(), &mut out);
                    }
                    continue;
                }
            };
            let mut finder = FactoryCallFinder {
                imports: &imports,
                modules: &modules_owned,
                allowed: &names_set,
                kinds: FxHashSet::default(),
            };
            for inner in body {
                finder.visit_stmt(inner);
            }
            if finder.kinds.is_empty() {
                continue;
            }
            if let Some(nr) = name_range {
                if let Some(idx) = self.local_idx_for_name_range(nr) {
                    push_owner(idx, &mut out);
                }
            }
        }
        out
    }
}

/// Rust-side per-file plugin contract. Called once per project file
/// with a restricted [`FileContext`]; pushes [`FileLocalOp`] into
/// the sink. Pure function of the file's tracked inputs — no
/// project-wide reads, no side effects — so the harness can cache the
/// result in salsa keyed on ``(file, name())``.
pub(crate) trait PerFileNativePluginImpl: Send + Sync {
    /// Walk ``file_ctx`` and append this file's ops to ``sink``.
    /// Naming + salsa keying live on [`PerFilePluginKind`], so the
    /// trait itself only carries the work method.
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>);
}

/// Salsa cache-key discriminant for a *configless* per-file plugin. A cheap
/// ``Copy`` enum rather than a ``&'static str`` because salsa tracked
/// function arguments must be owned + ``'static`` (a borrowed key
/// would force ``'db: 'static``). One variant per configless per-file
/// impl; per-file plugins that carry config use
/// [`PerFilePluginId::Configured`] + [`ConfiguredPerFile`] instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PerFilePluginKind {
    MainBlock,
    ModuleDunders,
}

impl PerFilePluginKind {
    /// Human-readable name, matching the equivalent Python plugin.
    fn name(self) -> &'static str {
        match self {
            PerFilePluginKind::MainBlock => "MainBlockPlugin",
            PerFilePluginKind::ModuleDunders => "ModuleDundersPlugin",
        }
    }

    /// Run the concrete impl for this kind against ``file_ctx``.
    fn run_on_file(self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>) {
        match self {
            PerFilePluginKind::MainBlock => MainBlockPluginImpl.run_on_file(file_ctx, sink),
            PerFilePluginKind::ModuleDunders => ModuleDundersPluginImpl.run_on_file(file_ctx, sink),
        }
    }
}

/// Salsa cache-key identifying *which* per-file plugin a
/// [`per_file_plugin_ops`] invocation is for. `Copy + 'static` so it can be
/// a salsa tracked-function argument. Configless builtin impls name themselves
/// by [`PerFilePluginKind`]; *configured* builtins and external dylib plugins
/// by a process-stable id assigned on registration (hash-interned for
/// configured plugins, see [`register_configured_per_file`] /
/// [`register_external_per_file`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PerFilePluginId {
    Builtin(PerFilePluginKind),
    /// Configured built-in per-file plugin — carries config (e.g.
    /// `ServerConfig`'s filename set) behind a process-stable id into
    /// [`CONFIGURED_PER_FILE_PLUGINS`]. The id is hash-interned on the config,
    /// so identical configs share an id and thus a salsa cache entry.
    Configured(u32),
    External(u32),
}

impl PerFilePluginId {
    /// Human-readable plugin name for harness logs / `progress_callback`.
    fn name(self) -> String {
        match self {
            PerFilePluginId::Builtin(kind) => kind.name().to_string(),
            PerFilePluginId::Configured(id) => configured_per_file_plugin(id)
                .map(|cfg| cfg.name().to_string())
                .unwrap_or_default(),
            // External per-file plugins flow through
            // `NativePluginKind::External`, never `PerFile`, so this arm is
            // unreachable under the name getter.
            PerFilePluginId::External(_) => {
                debug_assert!(false, "External per-file id has no PerFile name");
                String::new()
            }
        }
    }
}

/// Process-global registry of per-file external plugins, indexed by the id
/// baked into [`PerFilePluginId::External`].
///
/// Why a global: [`per_file_plugin_ops`] is a salsa tracked function, so its
/// key must be `Copy + 'static` — it cannot carry a `&dyn ExternalPlugin`.
/// We give each per-file external plugin a small stable id at load and look
/// the trait object back up here from inside the query. The mapping is
/// append-only and immutable for a given id for the process lifetime, so the
/// untracked read is sound: a memoized result stays valid exactly as long as
/// the file's tracked inputs are unchanged, identical to the builtin
/// enum-dispatch path. (Ids are never reused; a plugin's `Arc` is retained
/// for the session.)
static EXTERNAL_PER_FILE_PLUGINS: std::sync::OnceLock<
    std::sync::RwLock<Vec<Arc<dyn plugin_api::ExternalPlugin>>>,
> = std::sync::OnceLock::new();

/// Register a per-file external plugin and return its process-stable id.
fn register_external_per_file(plugin: Arc<dyn plugin_api::ExternalPlugin>) -> u32 {
    let reg = EXTERNAL_PER_FILE_PLUGINS.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    let mut guard = reg.write().expect("external per-file registry poisoned");
    let id = guard.len() as u32;
    guard.push(plugin);
    id
}

/// Look up a per-file external plugin by id.
fn external_per_file_plugin(id: u32) -> Option<Arc<dyn plugin_api::ExternalPlugin>> {
    EXTERNAL_PER_FILE_PLUGINS
        .get()?
        .read()
        .expect("external per-file registry poisoned")
        .get(id as usize)
        .map(Arc::clone)
}

/// A *configured* built-in per-file plugin: carries its config inline and
/// dispatches to the matching impl. Unlike [`PerFilePluginKind`] (a configless
/// `Copy` discriminant), these hold owned config, so they live behind an id in
/// [`CONFIGURED_PER_FILE_PLUGINS`] and are reached from inside the salsa query.
#[derive(Debug)]
pub(crate) enum ConfiguredPerFile {
    /// `ServerConfigPlugin` with a caller-supplied filename set.
    ServerConfig { filenames: Vec<String> },
}

impl ConfiguredPerFile {
    /// Human-readable name, matching the equivalent Python plugin.
    fn name(&self) -> &'static str {
        match self {
            ConfiguredPerFile::ServerConfig { .. } => "ServerConfigPlugin",
        }
    }

    /// Stable hash of the config, used to intern identical configs to one id.
    /// Canonicalizes collection fields (sort + dedup) so logically-equal
    /// configs hash equal and share a salsa cache entry; the leading
    /// discriminant byte keeps variants from colliding. A hash collision only
    /// costs a needless recompute, never correctness.
    fn config_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        match self {
            ConfiguredPerFile::ServerConfig { filenames } => {
                0u8.hash(&mut hasher);
                let mut canon: Vec<&str> = filenames.iter().map(String::as_str).collect();
                canon.sort_unstable();
                canon.dedup();
                canon.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Run the configured impl against ``file_ctx``.
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>) {
        match self {
            ConfiguredPerFile::ServerConfig { filenames } => {
                server_config_run_on_file(filenames, file_ctx, sink)
            }
        }
    }
}

/// Process-global registry of *configured* built-in per-file plugins, indexed
/// by the id baked into [`PerFilePluginId::Configured`]. Mirrors
/// [`EXTERNAL_PER_FILE_PLUGINS`] — append-only and immutable for a given id, so
/// the untracked read from inside [`per_file_plugin_ops`] is sound (a memoized
/// result stays valid exactly as long as the file's tracked inputs are
/// unchanged). It additionally *hash-interns* by
/// [`ConfiguredPerFile::config_hash`]: registering an equal config returns the
/// existing id, so a config reconstructed across a `re_materialize` reuses the
/// same salsa cache entry rather than minting a fresh, cold key.
static CONFIGURED_PER_FILE_PLUGINS: std::sync::OnceLock<std::sync::RwLock<ConfiguredRegistry>> =
    std::sync::OnceLock::new();

#[derive(Default)]
struct ConfiguredRegistry {
    plugins: Vec<Arc<ConfiguredPerFile>>,
    by_hash: FxHashMap<u64, u32>,
}

/// Register (intern) a configured per-file plugin and return its
/// process-stable id. Identical config (same [`ConfiguredPerFile::config_hash`])
/// returns the id assigned on first registration.
fn register_configured_per_file(cfg: ConfiguredPerFile) -> u32 {
    let reg = CONFIGURED_PER_FILE_PLUGINS
        .get_or_init(|| std::sync::RwLock::new(ConfiguredRegistry::default()));
    let hash = cfg.config_hash();
    let mut guard = reg.write().expect("configured per-file registry poisoned");
    if let Some(&id) = guard.by_hash.get(&hash) {
        return id;
    }
    let id = guard.plugins.len() as u32;
    guard.plugins.push(Arc::new(cfg));
    guard.by_hash.insert(hash, id);
    id
}

/// Look up a configured per-file plugin by id.
fn configured_per_file_plugin(id: u32) -> Option<Arc<ConfiguredPerFile>> {
    CONFIGURED_PER_FILE_PLUGINS
        .get()?
        .read()
        .expect("configured per-file registry poisoned")
        .plugins
        .get(id as usize)
        .map(Arc::clone)
}

/// Salsa-tracked per-file plugin invocation. Keyed on ``(file, id)``;
/// re-runs only when the file's tracked inputs (``file_to_nodes`` /
/// ``parsed_module`` / ``line_index``) change. Returns the file-local ops
/// the harness translates to global indices at apply time.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn per_file_plugin_ops(
    db: &dyn ProjectDb,
    file: File,
    id: PerFilePluginId,
) -> Vec<FileLocalOp> {
    let file_ctx = FileContext::new(db, file);
    let mut sink = Vec::new();
    match id {
        PerFilePluginId::Builtin(kind) => kind.run_on_file(&file_ctx, &mut sink),
        PerFilePluginId::Configured(plugin_id) => {
            // Resolve the configured plugin from the registry; a stale id
            // (shouldn't happen) yields no ops.
            if let Some(cfg) = configured_per_file_plugin(plugin_id) {
                cfg.run_on_file(&file_ctx, &mut sink);
            }
        }
        PerFilePluginId::External(plugin_id) => {
            // Resolve the plugin and its per-file capability from the
            // registry; a stale id (shouldn't happen) yields no ops.
            if let Some(plugin) = external_per_file_plugin(plugin_id) {
                if let Some(per_file) = plugin.per_file() {
                    let pctx = plugin_api::PluginFileCtx::new(&file_ctx);
                    let mut ops = plugin_api::FileOps::new();
                    per_file.run_on_file(&pctx, &mut ops);
                    sink = ops.into_inner();
                }
            }
        }
    }
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
    PerFile(PerFilePluginId),
    /// External native plugin loaded from a dylib through the ABI airlock
    /// (see [`load_native_plugins`]). Holds the trait object and a refcount
    /// on the loaded library so its code stays mapped for the plugin's
    /// lifetime. Only meaningful in a `-C prefer-dynamic` build where the
    /// extension and the plugin share one `dead-cst-runtime`.
    ///
    /// `per_file_id` is `Some(id)` when the plugin opted into per-file
    /// dispatch (`ExternalPlugin::per_file() -> Some`) — the host then routes
    /// it through the salsa-cached [`per_file_plugin_ops`] query keyed on
    /// [`PerFilePluginId::External`] instead of the project-wide `run`.
    External {
        name: String,
        plugin: Arc<dyn plugin_api::ExternalPlugin>,
        per_file_id: Option<u32>,
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
            NativePluginKind::PerFile(id) => id.name(),
            NativePluginKind::External { name, .. } => name.clone(),
        }
    }

    /// ``Plugin`` protocol's ``prepare(project_root)`` pre-graph hook.
    /// The harness calls this on every plugin before graph construction;
    /// we forward it to the underlying native impl (project-wide builtin or
    /// external dylib plugin). Per-file plugins are pure functions of their
    /// file and take no prepare step, so that arm is a no-op. `project_root`
    /// is coerced to its string form (it arrives as a ``pathlib.Path``).
    fn prepare(&self, py: Python<'_>, project_root: PyObject) -> PyResult<()> {
        let root = project_root.bind(py).str()?.to_string();
        match &self.kind {
            NativePluginKind::ProjectWide(inner) => inner.prepare(&root),
            NativePluginKind::PerFile(_) => {}
            NativePluginKind::External { plugin, .. } => plugin.prepare(&root),
        }
        Ok(())
    }

    /// Construct the native ``MainBlockPlugin`` — same observable
    /// behaviour as ``dead_cst.plugins.MainBlockPlugin``, but
    /// implemented as a *per-file* plugin: invoked once per file
    /// through a salsa-cached query, so an unchanged file's marker is
    /// reused across ``re_materialize`` with zero re-run.
    #[staticmethod]
    fn main_block() -> Self {
        Self {
            kind: NativePluginKind::PerFile(PerFilePluginId::Builtin(PerFilePluginKind::MainBlock)),
        }
    }

    /// Native `ModuleDundersPlugin` — pin module-level dunder names and
    /// `__future__` imports as entrypoints. Per-file (salsa-cached): a file's
    /// dunders + `__future__` imports are entirely local to it.
    #[staticmethod]
    fn module_dunders() -> Self {
        Self {
            kind: NativePluginKind::PerFile(PerFilePluginId::Builtin(
                PerFilePluginKind::ModuleDunders,
            )),
        }
    }

    /// Native `InitSubclassPlugin` — keep transitive subclasses of
    /// `__init_subclass__`-defining classes alive via a marker node.
    #[staticmethod]
    fn init_subclass() -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(InitSubclassPluginImpl)),
        }
    }

    /// Native `ServerConfigPlugin` — mark conventional WSGI/ASGI server
    /// config modules (`gunicorn.conf.py`, `hypercorn.conf.py`, …) as
    /// entrypoints. Per-file (salsa-cached): a file matches purely on its own
    /// basename and keeps its own top-level surface alive. `filenames` defaults
    /// to the conventional Gunicorn/Hypercorn set; pass a custom list to match
    /// other server-config basenames. Identical filename sets intern to one
    /// salsa cache key (see [`register_configured_per_file`]).
    #[staticmethod]
    #[pyo3(signature = (filenames = None))]
    fn server_config(filenames: Option<Vec<String>>) -> Self {
        let filenames = filenames.unwrap_or_else(|| {
            SERVER_CONFIG_FILENAMES
                .iter()
                .map(|s| (*s).to_string())
                .collect()
        });
        let id = register_configured_per_file(ConfiguredPerFile::ServerConfig { filenames });
        Self {
            kind: NativePluginKind::PerFile(PerFilePluginId::Configured(id)),
        }
    }

    /// Native `UnittestPlugin` — keep stdlib `unittest` test classes and
    /// lifecycle hooks alive. Project-wide (the subclass walk spans files).
    #[staticmethod]
    fn unittest() -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(UnittestPluginImpl)),
        }
    }

    /// Native Flask dispatch-app plugin (port of
    /// `dead_cst.contrib.flask.flask_plugin`). Mark Flask apps as entrypoints
    /// and wire `@app.route(...)` &c. handlers through them. Project-wide
    /// (a handler and its app may live in different files).
    #[staticmethod]
    fn flask() -> Self {
        Self::from_dispatch_config("flask".to_string(), flask_config())
    }

    /// Native FastAPI dispatch-app plugin (port of
    /// `dead_cst.contrib.fastapi.fastapi_plugin`).
    #[staticmethod]
    fn fastapi() -> Self {
        Self::from_dispatch_config("fastapi".to_string(), fastapi_config())
    }

    /// Native Typer dispatch-app plugin (port of
    /// `dead_cst.contrib.typer.typer_plugin`). Pure dispatch — apps are not
    /// entrypoint-promoted, so unused sub-typers surface as dead.
    #[staticmethod]
    fn typer() -> Self {
        Self::from_dispatch_config("typer".to_string(), typer_config())
    }

    /// Native cyclopts dispatch-app plugin (port of
    /// `dead_cst.contrib.cyclopts.cyclopts_plugin`). Pure dispatch (see
    /// :meth:`typer`).
    #[staticmethod]
    fn cyclopts() -> Self {
        Self::from_dispatch_config("cyclopts".to_string(), cyclopts_config())
    }

    /// Native Slack Bolt dispatch-app plugin (port of
    /// `dead_cst.contrib.slack_bolt.slack_bolt_plugin`). Recognises both the
    /// sync `App` and async `AsyncApp` bases.
    #[staticmethod]
    fn slack_bolt() -> Self {
        Self::from_dispatch_config("slack_bolt".to_string(), slack_bolt_config())
    }

    /// Native FastMCP dispatch-app plugin (port of
    /// `dead_cst.contrib.fastmcp.fastmcp_plugin`). Recognises the standalone
    /// `fastmcp` package and the MCP SDK's `mcp.server.fastmcp` layer.
    #[staticmethod]
    fn fastmcp() -> Self {
        Self::from_dispatch_config("fastmcp".to_string(), fastmcp_config())
    }

    /// Native Celery dispatch-app plugin (port of
    /// `dead_cst.contrib.celery.CeleryPlugin`). Wires `@app.task` handlers and
    /// additionally fans out appless `@shared_task` callables as entrypoints.
    #[staticmethod]
    fn celery() -> Self {
        Self::from_dispatch_config("celery".to_string(), celery_config())
    }

    /// Build a dispatch-app plugin from a caller-supplied config — the
    /// generalized form behind :meth:`flask` … :meth:`celery`, for a
    /// framework `dead-cst` doesn't bundle. `name` labels the plugin in
    /// progress logs; `marker_prefix` namespaces the synthetic entrypoint /
    /// factory nodes it mints. `app_classes` are dotted fqnames of the
    /// application classes (e.g. `["myframework.App"]`) whose instances —
    /// and transitive subclasses — anchor handler wiring;
    /// `registration_decorators` are the bare method names a handler is
    /// decorated with on such an instance (`@app.route` → `"route"`).
    ///
    /// When `seed_as_entrypoint` is true the discovered app instances (and
    /// factory functions returning them) are themselves kept alive — the
    /// web/task-framework default (flask/fastapi/celery). Pass false for a
    /// pure-dispatch CLI (typer/cyclopts), where an unused app surfaces as
    /// dead. The celery-style appless `@shared_task` fan-out is not exposed
    /// here; use :meth:`celery`.
    #[staticmethod]
    fn dispatch_app(
        name: String,
        marker_prefix: String,
        app_classes: Vec<String>,
        registration_decorators: Vec<String>,
        seed_as_entrypoint: bool,
    ) -> Self {
        Self::from_dispatch_config(
            name,
            DispatchAppConfig {
                marker_prefix,
                app_classes,
                registration_decorators,
                seed_as_entrypoint,
                shared_task: None,
            },
        )
    }

    /// The Click CLI plugin. Wires `@<group>.command` / `@<group>.group` /
    /// `@<group>.result_callback` handlers to their owning Click group, with a
    /// fixpoint so a `@<group>.group()` handler is itself promoted to a group.
    /// Groups are *not* seeded as entrypoints — reachability enters through
    /// `[project.scripts]` / `__main__` / `add_command`. Project-wide because a
    /// group declared in one file may collect handlers registered in another.
    #[staticmethod]
    fn click() -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(ClickPluginImpl)),
        }
    }

    /// Native `MockPatchPlugin` (port of `dead_cst.contrib.mock_patch`).
    /// Resolve string-fqname `patch(...)` / `mocker.patch(...)` /
    /// `monkeypatch.setattr(...)` / `monkeypatch.delattr(...)` targets to their
    /// decls and keep them alive. Project-wide (a patch string in a test file
    /// targets a decl in another).
    #[staticmethod]
    fn mock_patch() -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(MockPatchPluginImpl)),
        }
    }

    /// Native `DiscordPyPlugin` (port of `dead_cst.contrib.discordpy`). Wire
    /// discord.py bot/client handlers, Cogs, and extension hooks into
    /// reachability. Project-wide (a `@bot.command` handler and its `bot = Bot()`
    /// may live in different files; Cog subclasses span files).
    #[staticmethod]
    fn discordpy() -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(DiscordPyPluginImpl)),
        }
    }

    /// Native `PytestPlugin` (port of `dead_cst.contrib.pytest`). Seed
    /// pytest-discovered tests / conftest decls / `@pytest.fixture` functions
    /// and wire test→fixture edges by parameter-name matching. Project-wide
    /// (the subclass / fixture walk spans files).
    #[staticmethod]
    fn pytest() -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(PytestPluginImpl)),
        }
    }

    /// Native `ProjectScriptsPlugin` (port of
    /// `dead_cst.plugins.project_scripts`). Mark every `[project.scripts]` entry
    /// in `pyproject.toml` as an entrypoint. `pyproject_path` defaults to
    /// `<project_root>/pyproject.toml`. Project-wide (a script target resolves
    /// against the whole graph).
    #[staticmethod]
    #[pyo3(signature = (pyproject_path = None))]
    fn project_scripts(pyproject_path: Option<String>) -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(ProjectScriptsPluginImpl {
                pyproject_path,
            })),
        }
    }

    /// Native `DynamicImportFallbackPlugin` (port of
    /// `dead_cst.plugins.dynamic_import`). Fan out `EdgeFlags.DYNAMIC_IMPORT`
    /// edges to each target module's exports. The three-stage rollout knobs
    /// mirror the Python dataclass; with no filters every dynamic-import edge
    /// fans out (the catch-all default). Project-wide (edges span files).
    #[staticmethod]
    #[pyo3(signature = (
        *,
        include_underscore = false,
        respect_dunder_all = true,
        exclude_sources = None,
        exclude_targets = None,
        include_sources = None,
        include_targets = None,
    ))]
    fn dynamic_import_fallback(
        include_underscore: bool,
        respect_dunder_all: bool,
        exclude_sources: Option<Vec<String>>,
        exclude_targets: Option<Vec<String>>,
        include_sources: Option<Vec<String>>,
        include_targets: Option<Vec<String>>,
    ) -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(DynamicImportFallbackPluginImpl {
                include_underscore,
                respect_dunder_all,
                exclude_sources: exclude_sources.unwrap_or_default(),
                exclude_targets: exclude_targets.unwrap_or_default(),
                include_sources: include_sources.unwrap_or_default(),
                include_targets: include_targets.unwrap_or_default(),
            })),
        }
    }

    /// Native `ExplicitEntrypointPlugin` (port of
    /// `dead_cst.plugins.explicit_entrypoint`). Mark user-specified symbols
    /// (file paths, FQNs, regexes) as entrypoints. Specs arrive pre-bucketed:
    /// `regexes` (matched against the project-relative path), `str_specs`
    /// (exact fqname or project-relative path), `abs_paths` (exact absolute
    /// path). Not in `_builtin_native_plugin` — it needs caller-supplied specs.
    #[staticmethod]
    fn explicit(regexes: Vec<String>, str_specs: Vec<String>, abs_paths: Vec<String>) -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(ExplicitEntrypointPluginImpl {
                regexes,
                str_specs,
                abs_paths,
            })),
        }
    }
}

impl NativePlugin {
    /// Wrap a [`DispatchAppConfig`] in a project-wide native plugin. Shared by
    /// the per-framework factories and the custom `dispatch_app` factory.
    fn from_dispatch_config(name: String, config: DispatchAppConfig) -> Self {
        Self {
            kind: NativePluginKind::ProjectWide(Box::new(DispatchAppPluginImpl { name, config })),
        }
    }
}

/// Resolve a built-in plugin name to its native implementation, or `None`
/// if no native plugin owns that name (yet). The CLI's `_load_plugin`
/// consults this before the Python builtin map — as plugins are ported to
/// Rust they move from that map into this registry.
#[pyfunction]
pub(crate) fn _builtin_native_plugin(name: &str) -> Option<NativePlugin> {
    Some(match name {
        "main_block" => NativePlugin::main_block(),
        "module_dunders" => NativePlugin::module_dunders(),
        "init_subclass" => NativePlugin::init_subclass(),
        "server_config" => NativePlugin::server_config(None),
        "unittest" => NativePlugin::unittest(),
        "flask" => NativePlugin::flask(),
        "fastapi" => NativePlugin::fastapi(),
        "typer" => NativePlugin::typer(),
        "cyclopts" => NativePlugin::cyclopts(),
        "slack_bolt" => NativePlugin::slack_bolt(),
        "fastmcp" => NativePlugin::fastmcp(),
        "celery" => NativePlugin::celery(),
        "click" => NativePlugin::click(),
        "mock_patch" => NativePlugin::mock_patch(),
        "discordpy" => NativePlugin::discordpy(),
        "pytest" => NativePlugin::pytest(),
        "project_scripts" => NativePlugin::project_scripts(None),
        "dynamic_import_fallback" => {
            NativePlugin::dynamic_import_fallback(false, true, None, None, None, None)
        }
        _ => return None,
    })
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
///
/// The view is **index-based**: every query returns positional indices into
/// the frozen node list (the same index space [`PluginOps`] emits against),
/// and [`PluginCtx::node`] turns an index into an owned [`NodeView`]. No
/// `Python<'_>` token is ever exposed — the methods read `#[pyclass(frozen)]`
/// data directly and acquire the GIL internally only where the underlying
/// query needs it.
pub mod plugin_api {
    use pyo3::Python;

    use super::{FileContext, FileLocalOp};
    use crate::builder::PreparedOp;
    use crate::graph::{intern_kind, EdgeFlags, NodeFlags};
    use crate::project::ProjectContext;

    // Re-exported so a per-file plugin can name the raw AST / range types
    // [`PluginFileCtx::parsed`] and [`PluginFileCtx::line_span`] traffic in
    // without depending on the exact ruff crate paths.
    pub use ruff_db::parsed::ParsedModuleRef;
    pub use ruff_text_size::TextRange;

    /// `NodeFlags::ENTRYPOINT` re-exported for plugin authors: set it on a
    /// node minted via [`PluginOps::add_synthetic_node`] /
    /// [`FileOps::add_synthetic_node`] to make that node a reachability seed.
    /// Single source of truth — tracks the internal bit.
    pub const FLAG_ENTRYPOINT: u32 = NodeFlags::ENTRYPOINT;

    /// `EdgeFlags::DEAD_BRANCH` re-exported for the `flags` argument of
    /// [`PluginOps::add_edge`] / [`FileOps::add_edge`]: mark a plugin-added
    /// edge as originating in a statically-dead region. Metadata only — the
    /// edge still participates in default reachability.
    pub const FLAG_DEAD_BRANCH: u32 = EdgeFlags::DEAD_BRANCH;

    /// `EdgeFlags::DYNAMIC_IMPORT` re-exported for the `flags` argument of
    /// [`PluginOps::add_edge`] / [`FileOps::add_edge`]: mark a plugin-added
    /// edge as a runtime-import (`__import__` / `importlib.import_module`)
    /// fan-out, the same bit the visitor stamps on dynamic-import edges.
    pub const FLAG_DYNAMIC_IMPORT: u32 = EdgeFlags::DYNAMIC_IMPORT;

    /// Contract an external native plugin implements. The host calls
    /// [`run`](ExternalPlugin::run) once per materialize against the frozen
    /// graph, then folds the emitted ops in as a single batch.
    ///
    /// A plugin can optionally opt into **per-file** dispatch by also
    /// implementing [`PerFilePlugin`] and returning `Some(self)` from
    /// [`per_file`](ExternalPlugin::per_file). When it does, the host ignores
    /// [`run`](ExternalPlugin::run) and instead invokes
    /// [`PerFilePlugin::run_on_file`] once per project file through a
    /// salsa-cached query — so an unchanged file's ops are reused across a
    /// `re_materialize` with zero re-run. (Same fast-path the in-tree
    /// `MainBlockPlugin` rides.)
    pub trait ExternalPlugin: Send + Sync {
        /// Human-readable name (surfaced in progress logs).
        fn name(&self) -> &str;

        /// Inspect the whole frozen graph and append ops to `ops`. Same
        /// frozen-graph contract as the in-tree native path. Default no-op
        /// so a pure per-file plugin needn't implement it; the host skips it
        /// entirely when [`per_file`](ExternalPlugin::per_file) is `Some`.
        fn run(&self, _ctx: &PluginCtx<'_>, _ops: &mut PluginOps) {}

        /// Opt into per-file (salsa-cached) dispatch by returning
        /// `Some(self)`. The default `None` keeps the plugin project-wide.
        ///
        /// Whether a plugin is per-file is read **once, at load** — it must
        /// not depend on runtime state.
        fn per_file(&self) -> Option<&dyn PerFilePlugin> {
            None
        }

        /// Pre-graph hook, mirroring the Python `Plugin.prepare` contract:
        /// the host calls it once with the project root (as a string) before
        /// any graph construction, so a plugin can scan for config files or
        /// framework manifests up front. Default no-op.
        ///
        /// Runs before the graph exists — it gets no [`PluginCtx`]. A
        /// **per-file** plugin must keep `run_on_file` a pure function of its
        /// file, so anything `prepare` stashes must not leak into per-file
        /// output (that would break the salsa cache); use it for project-wide
        /// `run` setup only.
        fn prepare(&self, _project_root: &str) {}
    }

    /// Per-file capability for an [`ExternalPlugin`]. Invoked once per
    /// project file with a restricted [`PluginFileCtx`]; emits file-local
    /// ops into [`FileOps`].
    ///
    /// **Purity contract.** `run_on_file` must be a pure function of its
    /// `file` — it may read only that file's nodes / parsed AST (everything
    /// [`PluginFileCtx`] exposes) and must not consult project-wide state,
    /// other files, globals, the clock, or the filesystem. The host caches
    /// the result in salsa keyed on the file's tracked inputs, so an impure
    /// `run_on_file` would serve stale ops after an unrelated edit. Ops may
    /// only reference nodes *in this file* (file-local indices) — that
    /// restriction is what makes the output cacheable.
    pub trait PerFilePlugin: Send + Sync {
        /// Walk `file` and append this file's ops to `ops`.
        fn run_on_file(&self, file: &PluginFileCtx<'_>, ops: &mut FileOps);
    }

    /// Owned, plain-data snapshot of one graph node, returned by
    /// [`PluginCtx::node`]. Decoupled from the internal `SymbolNode`
    /// pyclass so the author-facing shape stays stable across releases.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct NodeView {
        /// Positional index of this node in the frozen graph.
        pub idx: usize,
        /// Fully-qualified name (e.g. `pkg.mod.func`).
        pub fqname: String,
        /// One of `function`, `class`, `variable`, `import`, `type_alias`,
        /// `module`, `synthetic`.
        pub kind: String,
        /// Source path the node was declared in (empty for synthetics).
        pub path: String,
        /// 1-based start line, or 0 for synthetic nodes.
        pub start_line: usize,
        /// 1-based end line, or 0 for synthetic nodes.
        pub end_line: usize,
        /// `NodeFlags` bitset (see [`FLAG_ENTRYPOINT`]).
        pub flags: u32,
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

        // --- node reads -----------------------------------------------------

        /// Total number of nodes in the frozen graph. Indices in
        /// `0..node_count()` are valid arguments to [`Self::node`].
        pub fn node_count(&self) -> usize {
            self.inner
                .materialized("plugin")
                .map(|o| o.builder.nodes.len())
                .unwrap_or(0)
        }

        /// Resolve an index to an owned [`NodeView`], or `None` if out of
        /// range. Reads frozen pyclass data directly — no GIL, no clone of
        /// the underlying `SymbolNode`.
        pub fn node(&self, idx: usize) -> Option<NodeView> {
            let outputs = self.inner.materialized("plugin").ok()?;
            let node = outputs.builder.nodes.get(idx)?;
            Some(NodeView {
                idx,
                fqname: node.fqname.clone(),
                kind: node.kind.to_string(),
                path: node.path.clone(),
                start_line: node.start_line,
                end_line: node.end_line,
                flags: node.flags,
            })
        }

        // --- structural lookups (index-returning, GIL-free) -----------------

        /// O(1) module-node lookup by dotted fqname.
        pub fn find_module(&self, fqname: &str) -> Option<usize> {
            self.inner.find_module_idx(fqname).unwrap_or(None)
        }

        /// Every declaration node with this exact fqname (more than one
        /// when a name is shadowed; see the graph-model invariants).
        pub fn find_declarations(&self, fqname: &str) -> Vec<usize> {
            self.inner
                .find_declarations_indices(fqname)
                .unwrap_or_default()
        }

        /// O(1) module-node lookup by source path.
        pub fn module_for(&self, path: &str) -> Option<usize> {
            self.inner.module_for_indices(path).unwrap_or(None)
        }

        /// Resolve a dotted fqname to a decl or module node, walking back
        /// through dotted segments for method/attribute fqnames. `None`
        /// when nothing matches.
        pub fn resolve(&self, fqname: &str) -> Option<usize> {
            self.inner.resolve_idx(fqname).unwrap_or(None)
        }

        /// Every node whose source path starts with `path_prefix` — a
        /// cheap way to scope a plugin to one package/directory.
        pub fn decls_under(&self, path_prefix: &str) -> Vec<usize> {
            self.inner
                .decls_under_indices(path_prefix)
                .unwrap_or_default()
        }

        /// Transitive subclasses of the class at `class_idx` (empty when
        /// `class_idx` isn't a class node).
        pub fn find_subclasses_of(&self, class_idx: usize) -> Vec<usize> {
            self.inner
                .find_subclasses_of_idx(class_idx)
                .unwrap_or_default()
        }

        /// Subclasses of the class named by dotted `base_fqn`, resolved
        /// through ty's module resolver — so the base may live in another
        /// file or a dependency (e.g. `"unittest.TestCase"`). `transitive`
        /// walks the whole hierarchy; `false` returns only direct subclasses.
        /// The by-fqname twin of [`Self::find_subclasses_of`] (which takes a
        /// node index). Empty when nothing resolves.
        pub fn find_subclasses_of_fqn(&self, base_fqn: &str, transitive: bool) -> Vec<usize> {
            Python::with_gil(|py| self.inner.find_subclasses_indices(py, base_fqn, transitive))
                .unwrap_or_default()
        }

        // --- reachability ---------------------------------------------------

        /// Forward closure: every node reachable from `root_idx` by
        /// following graph edges (dead-branch edges included, matching the
        /// default traversal).
        pub fn descendants(&self, root_idx: usize) -> Vec<usize> {
            self.inner
                .descendants_indices(root_idx, 0)
                .unwrap_or_default()
        }

        /// Reverse closure: every node that can reach `decl_idx`.
        pub fn ancestors(&self, decl_idx: usize) -> Vec<usize> {
            self.inner
                .ancestors_indices(decl_idx, 0)
                .unwrap_or_default()
        }

        /// One-hop reverse step: every node with an edge directly into
        /// `idx`.
        pub fn direct_predecessors(&self, idx: usize) -> Vec<usize> {
            self.inner
                .direct_predecessors_idx(idx, 0)
                .unwrap_or_default()
        }

        /// Each top-level `if __name__ == "__main__":` block as
        /// `(module_node_idx, [decl_node_idx, ...])`.
        pub fn main_blocks(&self) -> Vec<(usize, Vec<usize>)> {
            self.inner.find_main_blocks_indices().unwrap_or_default()
        }

        // --- project-wide matchers ------------------------------------------
        //
        // Index-returning project-wide finders, so a project-wide native
        // plugin can find its seeds without hand-rolling the import /
        // decorator / construction AST walk. They share the exact matchers
        // the per-file [`PluginFileCtx`] helpers use, so the per-file and
        // project-wide surfaces agree. Each returns project-wide node
        // indices.

        /// The names `from module_fqn import *` would bind into the
        /// importing scope — every top-level decl on `module_fqn`'s public
        /// surface (honoring `__all__` when present). Empty when the module
        /// isn't found.
        pub fn module_surface(&self, module_fqn: &str) -> Vec<usize> {
            self.inner
                .module_surface_indices(module_fqn)
                .unwrap_or_default()
        }

        /// The decl nodes named by `module_fqn`'s `__all__`, or `None` when
        /// the module has no `__all__` / it isn't a literal string list.
        pub fn dunder_all_exports(&self, module_fqn: &str) -> Option<Vec<usize>> {
            self.inner
                .find_module_dunder_all_exports_indices(module_fqn)
                .unwrap_or(None)
        }

        /// The string entries of a top-level `X = ["a", "b"]` /
        /// `X: tuple[str, ...] = ("a", "b")` literal-list assignment named
        /// `var_fqn`, or `None` when the variable isn't a literal string
        /// list. The read the `LiteralListPlugin` shape is built on.
        pub fn literal_list_entries(&self, var_fqn: &str) -> Option<Vec<String>> {
            self.inner
                .find_literal_list_entries(var_fqn)
                .unwrap_or(None)
        }

        /// Every top-level decl (function / class / variable / import /
        /// type_alias) whose simple name matches the regular expression
        /// `pattern`. An invalid pattern yields an empty result.
        pub fn decls_matching_name(&self, pattern: &str) -> Vec<usize> {
            self.inner
                .decls_matching_name_indices(pattern)
                .unwrap_or_default()
        }

        /// Project-wide twin of [`PluginFileCtx::decorated_decls`]:
        /// function / class decls anywhere in the project carrying a
        /// decorator that resolves to one of `names` imported from one of
        /// `modules`.
        pub fn decorated_decls(&self, modules: &[&str], names: &[&str]) -> Vec<usize> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            let names_owned: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            Python::with_gil(|py| {
                self.inner
                    .find_decorated_decls(py, &modules_owned, names_owned, None, false)
            })
            .map(|pairs| pairs.into_iter().map(|(idx, _)| idx).collect())
            .unwrap_or_default()
        }

        /// Project-wide twin of [`PluginFileCtx::constructions`]: top-level
        /// `X = Ctor(...)` decls anywhere in the project whose `Ctor`
        /// resolves to one of `names` imported from one of `modules`.
        pub fn constructions(&self, modules: &[&str], names: &[&str]) -> Vec<usize> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            let names_owned: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            Python::with_gil(|py| {
                self.inner
                    .find_instance_constructions(py, &modules_owned, names_owned, None, false)
            })
            .map(|triples| triples.into_iter().map(|(idx, _, _)| idx).collect())
            .unwrap_or_default()
        }

        /// Top-level functions decorated `@<owner>.<attr>(...)` where `attr`
        /// is one of `decorator_attrs`. Returns `(owner_name, decl_idx)` —
        /// the raw textual decorator owner (`"app"` for `@app.route`, *not*
        /// resolved to a node: the caller decides which owners map to real
        /// framework instances) and the decorated decl's node index. This
        /// owner-attr read powers the dispatch-app / click handler wiring.
        /// Multiple matching decorators on one function yield multiple rows.
        pub fn handler_decorators(&self, decorator_attrs: &[&str]) -> Vec<(String, usize)> {
            let attrs: Vec<String> = decorator_attrs.iter().map(|s| s.to_string()).collect();
            Python::with_gil(|py| self.inner.find_handler_decorators(py, attrs, None, false))
                .map(|triples| {
                    triples
                        .into_iter()
                        .map(|(owner, idx, _args)| (owner, idx))
                        .collect()
                })
                .unwrap_or_default()
        }

        /// Calls to `name` (imported from one of `modules`) whose argument at
        /// position `arg_index` is a string literal. Returns
        /// `(owning_decl_idx, literal)` — the top-level decl whose body makes
        /// the call (module-scope calls map to the module node) and the
        /// literal's value. This string-arg read powers FQN/path-string
        /// reads like `mock.patch("pkg.mod.target")`. Calls whose arg isn't a
        /// plain string literal are skipped.
        pub fn calls_with_string_arg(
            &self,
            modules: &[&str],
            name: &str,
            arg_index: usize,
        ) -> Vec<(usize, String)> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            Python::with_gil(|py| {
                self.inner
                    .find_calls_to_imported(py, &modules_owned, name, arg_index, None, false)
            })
            .map(|triples| {
                triples
                    .into_iter()
                    .map(|(idx, literal, _args)| (idx, literal))
                    .collect()
            })
            .unwrap_or_default()
        }

        /// Attr-method twin of [`PluginCtx::calls_with_string_arg`]: calls of
        /// the form `<recv>.<attr>(...)` — *any* receiver shape
        /// (`bot.load_extension(x)`, `self.bot.load_extension(x)`,
        /// `get_bot().load_extension(x)`) — whose argument at `arg_index` is a
        /// string literal. A list/tuple of string literals fans out to one row
        /// per element. Returns `(owning_decl_idx, literal)` — the top-level
        /// decl whose body makes the call (module-scope calls map to the module
        /// node) and the literal's value. Keyed on the method name, so the
        /// receiver is the plugin's concern (typically gated by a per-file
        /// import check). Calls whose arg isn't a string literal (or string
        /// collection) are skipped.
        pub fn calls_on_attr(&self, attr: &str, arg_index: usize) -> Vec<(usize, String)> {
            Python::with_gil(|py| {
                self.inner
                    .find_calls_on_attr(py, attr, arg_index, None, false)
            })
            .map(|triples| {
                triples
                    .into_iter()
                    .map(|(idx, literal, _args)| (idx, literal))
                    .collect()
            })
            .unwrap_or_default()
        }
    }

    /// Op sink for external plugins. Wraps the internal `PreparedOp` vec so
    /// plugins emit through named methods instead of constructing internals.
    /// Exposes the three [`PreparedOp`] variants —
    /// `Entrypoint` / `Edge` / `Node` — as `keep_alive` / `add_edge` /
    /// `add_synthetic_node`.
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
        /// `marker`. Emits a [`PreparedOp::Entrypoint`].
        pub fn keep_alive(&mut self, decl_idx: usize, marker: String) {
            self.sink.push(PreparedOp::Entrypoint { decl_idx, marker });
        }

        /// Add a reachability edge `src_idx -> dst_idx` between two
        /// existing nodes, stamped with `flags` (`0` for a plain edge, or
        /// one of [`FLAG_DEAD_BRANCH`] / [`FLAG_DYNAMIC_IMPORT`]). Emits a
        /// [`PreparedOp::Edge`].
        pub fn add_edge(&mut self, src_idx: usize, dst_idx: usize, flags: u32) {
            self.sink.push(PreparedOp::Edge {
                src_idx,
                dst_idx,
                flags,
            });
        }

        /// Mint a new `kind="synthetic"` node named `fqname`, attributed to
        /// source `path` (empty for a placeless marker), with `flags` (see
        /// [`FLAG_ENTRYPOINT`]), an out-edge to each index in `edges_to_idx`,
        /// and an in-edge from each index in `edges_from_idx`. Emits a
        /// [`PreparedOp::Node`]; the host bounds-checks every endpoint at apply
        /// time and rejects a dangling index rather than minting an unconnected
        /// node.
        pub fn add_synthetic_node(
            &mut self,
            fqname: String,
            path: String,
            flags: u32,
            edges_to_idx: Vec<usize>,
            edges_from_idx: Vec<usize>,
        ) {
            self.sink.push(PreparedOp::Node {
                fqname,
                kind: intern_kind("synthetic").expect("'synthetic' is a valid kind"),
                path,
                flags,
                edges_from_idx,
                edges_to_idx,
            });
        }
    }

    // ---- per-file surface --------------------------------------------------

    /// Owned, plain-data snapshot of one node in a single file, returned by
    /// [`PluginFileCtx::nodes`] / [`PluginFileCtx::node`]. The `local_idx`
    /// addresses this file only — it's the index space [`FileOps`] emits
    /// against, *not* the project-wide [`NodeView::idx`].
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct FileNodeView {
        /// File-local index (0 is always the synthetic module node).
        pub local_idx: u32,
        /// Fully-qualified name.
        pub fqname: String,
        /// One of `function`, `class`, `variable`, `import`, `type_alias`,
        /// `module`, `synthetic`.
        pub kind: String,
        /// Source path the node was declared in.
        pub path: String,
        /// 1-based start line.
        pub start_line: usize,
        /// 1-based end line.
        pub end_line: usize,
        /// `NodeFlags` bitset.
        pub flags: u32,
    }

    /// Restricted, single-file view handed to [`PerFilePlugin::run_on_file`].
    /// Exposes only this file's salsa-tracked nodes + parsed AST — no
    /// project-wide queries — which is exactly what makes a per-file plugin's
    /// output a pure function of the file (and therefore cacheable).
    pub struct PluginFileCtx<'a> {
        inner: &'a FileContext<'a>,
    }

    impl<'a> PluginFileCtx<'a> {
        pub(crate) fn new(inner: &'a FileContext<'a>) -> Self {
            Self { inner }
        }

        /// This file's module fqname.
        pub fn module_fqname(&self) -> &str {
            self.inner.module_fqname()
        }

        /// File-local index of the synthetic module node (always 0).
        pub fn module_local_idx(&self) -> u32 {
            self.inner.module_local_idx()
        }

        /// Number of nodes in this file (index 0 is the module node, the
        /// rest are top-level decls). Valid `local_idx` are `0..node_count()`.
        pub fn node_count(&self) -> usize {
            self.inner.nodes().len()
        }

        /// Resolve a file-local index to an owned [`FileNodeView`].
        pub fn node(&self, local_idx: u32) -> Option<FileNodeView> {
            let node = self.inner.nodes().get(local_idx as usize)?;
            Some(FileNodeView {
                local_idx,
                fqname: node.fqname.clone(),
                kind: node.kind.as_static_str().to_string(),
                path: node.path.clone(),
                start_line: node.start_line,
                end_line: node.end_line,
                flags: node.flags,
            })
        }

        /// Every node in this file as a [`FileNodeView`], in file-local
        /// index order.
        pub fn nodes(&self) -> Vec<FileNodeView> {
            self.inner
                .nodes()
                .iter()
                .enumerate()
                .map(|(i, node)| FileNodeView {
                    local_idx: i as u32,
                    fqname: node.fqname.clone(),
                    kind: node.kind.as_static_str().to_string(),
                    path: node.path.clone(),
                    start_line: node.start_line,
                    end_line: node.end_line,
                    flags: node.flags,
                })
                .collect()
        }

        /// 1-based `(start_line, end_line)` of a byte `range` in this file,
        /// via the salsa-cached line index. Use it to map a [`TextRange`]
        /// from [`Self::parsed`] onto the line numbers [`FileNodeView`]
        /// carries.
        pub fn line_span(&self, range: TextRange) -> (usize, usize) {
            self.inner.line_span(range)
        }

        /// This file's parsed module (salsa-cached) for raw AST walks.
        pub fn parsed(&self) -> ParsedModuleRef {
            self.inner.parsed()
        }

        /// Convenience: the byte range of this file's top-level
        /// `if __name__ == "__main__":` block, if any — the same helper the
        /// in-tree `MainBlockPlugin` uses, so the common case needs no raw
        /// AST walk.
        pub fn main_block_range(&self) -> Option<TextRange> {
            crate::helpers::find_main_block_range(&self.parsed())
        }

        // --- ready-made queries ---------------------------------------------
        //
        // File-local matching that saves a per-file plugin from hand-rolling
        // the import / decorator / construction / call AST walk. Each returns
        // file-local indices (the index space [`FileOps`] emits against), so
        // a plugin pipes the result straight into `ops.keep_alive(idx, ...)`
        // / `ops.add_edge(...)`. Same matcher cores the in-tree project-wide
        // queries use — a per-file plugin and its project-wide twin agree.

        /// True if this file imports any of `modules`. A cheap presence
        /// guard to short-circuit before heavier per-file work.
        pub fn imports_any_module(&self, modules: &[&str]) -> bool {
            self.inner.imports_any_module(modules)
        }

        /// File-local indices of top-level function / class decls decorated
        /// by one of `names` imported from one of `modules`
        /// (e.g. `modules=["flask"], names=["route"]` for `@app.route` via
        /// a `Flask` instance is *not* this — this matches decorators bound
        /// directly to an imported name; instance-method decorators need the
        /// project-wide query).
        pub fn decorated_decls(&self, modules: &[&str], names: &[&str]) -> Vec<u32> {
            self.inner.decorated_decls(modules, names)
        }

        /// File-local indices of top-level ``X = Ctor(...)`` decls whose
        /// `Ctor` is one of `names` imported from one of `modules`.
        pub fn constructions(&self, modules: &[&str], names: &[&str]) -> Vec<u32> {
            self.inner.constructions(modules, names)
        }

        /// File-local indices of decls whose body calls one of `names`
        /// imported from one of `modules`; module-scope calls map to the
        /// module node (index 0). Each owner appears once.
        pub fn calls(&self, modules: &[&str], names: &[&str]) -> Vec<u32> {
            self.inner.calls(modules, names)
        }
    }

    /// File-local op sink for a [`PerFilePlugin`]. Mirrors
    /// [`PluginOps::add_synthetic_node`] but in this file's *local* index
    /// space — endpoints are positions in [`PluginFileCtx::nodes`], which
    /// the host translates to global indices at apply time.
    pub struct FileOps {
        sink: Vec<FileLocalOp>,
    }

    impl FileOps {
        pub(crate) fn new() -> Self {
            Self { sink: Vec::new() }
        }

        pub(crate) fn into_inner(self) -> Vec<FileLocalOp> {
            self.sink
        }

        /// Mint a `kind="synthetic"` node named `fqname` with `flags` (see
        /// [`FLAG_ENTRYPOINT`]), an out-edge to each file-local index in
        /// `edges_to_local_idx`, and an in-edge from each file-local index
        /// in `edges_from_local_idx`. References must stay within this
        /// file — a local index with no node is dropped at apply time.
        pub fn add_synthetic_node(
            &mut self,
            fqname: String,
            flags: u32,
            edges_to_local_idx: Vec<u32>,
            edges_from_local_idx: Vec<u32>,
        ) {
            self.sink.push(FileLocalOp::Node {
                fqname,
                kind: intern_kind("synthetic").expect("'synthetic' is a valid kind"),
                flags,
                edges_to_local_idx,
                edges_from_local_idx,
            });
        }

        /// Keep the node at file-local index `decl_local_idx` alive by
        /// seeding it from a synthetic entrypoint tagged `marker`. The index
        /// is a position in [`PluginFileCtx::nodes`]; an index with no node
        /// is dropped at apply time.
        pub fn keep_alive(&mut self, decl_local_idx: u32, marker: String) {
            self.sink.push(FileLocalOp::Entrypoint {
                decl_local_idx,
                marker,
            });
        }

        /// Add a reachability edge from `src_local_idx` to `dst_local_idx`,
        /// stamped with `flags` (`0` for a plain edge, or one of
        /// [`FLAG_DEAD_BRANCH`] / [`FLAG_DYNAMIC_IMPORT`]). Both endpoints
        /// are positions in [`PluginFileCtx::nodes`]; an edge with an
        /// unresolvable endpoint is dropped at apply time.
        pub fn add_edge(&mut self, src_local_idx: u32, dst_local_idx: u32, flags: u32) {
            self.sink.push(FileLocalOp::Edge {
                src_local_idx,
                dst_local_idx,
                flags,
            });
        }
    }

    // The read methods are thin, infallible delegations to `pub(crate)`
    // `ProjectContext` methods that already carry Python-level test coverage
    // (the native-plugin and per-file query suites), so they don't get a
    // bespoke harness here — the end-to-end airlock path is exercised by the
    // gated `tests/test_plugins/test_external_dylib_plugin.py` in CI. These
    // tests pin the *writer* widening: that `flags` and the in-edge list
    // thread through to the emitted op rather than being dropped.
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::builder::PreparedOp;

        #[test]
        fn plugin_ops_add_edge_carries_flags() {
            let mut ops = PluginOps::new();
            ops.add_edge(1, 2, FLAG_DYNAMIC_IMPORT);
            assert!(matches!(
                ops.into_inner().as_slice(),
                [PreparedOp::Edge { src_idx: 1, dst_idx: 2, flags }]
                    if *flags == FLAG_DYNAMIC_IMPORT
            ));
        }

        #[test]
        fn plugin_ops_add_synthetic_node_carries_edges_from() {
            let mut ops = PluginOps::new();
            ops.add_synthetic_node(
                "syn".to_string(),
                "pkg/mod.py".to_string(),
                FLAG_ENTRYPOINT,
                vec![3],
                vec![4, 5],
            );
            let sink = ops.into_inner();
            let [PreparedOp::Node {
                fqname,
                kind,
                path,
                flags,
                edges_from_idx,
                edges_to_idx,
            }] = sink.as_slice()
            else {
                panic!("expected a single Node op");
            };
            assert_eq!(fqname, "syn");
            assert_eq!(*kind, "synthetic");
            assert_eq!(path, "pkg/mod.py");
            assert_eq!(*flags, FLAG_ENTRYPOINT);
            assert_eq!(edges_to_idx.as_slice(), &[3]);
            assert_eq!(edges_from_idx.as_slice(), &[4, 5]);
        }

        #[test]
        fn file_ops_add_edge_carries_flags() {
            let mut ops = FileOps::new();
            ops.add_edge(0, 1, FLAG_DEAD_BRANCH);
            assert!(matches!(
                ops.into_inner().as_slice(),
                [FileLocalOp::Edge { src_local_idx: 0, dst_local_idx: 1, flags }]
                    if *flags == FLAG_DEAD_BRANCH
            ));
        }

        #[test]
        fn file_ops_add_synthetic_node_carries_edges_from() {
            let mut ops = FileOps::new();
            ops.add_synthetic_node("syn".to_string(), FLAG_ENTRYPOINT, vec![2], vec![3]);
            let sink = ops.into_inner();
            let [FileLocalOp::Node {
                fqname,
                edges_to_local_idx,
                edges_from_local_idx,
                ..
            }] = sink.as_slice()
            else {
                panic!("expected a single Node op, got {sink:?}");
            };
            assert_eq!(fqname, "syn");
            assert_eq!(edges_to_local_idx.as_slice(), &[2]);
            assert_eq!(edges_from_local_idx.as_slice(), &[3]);
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
            let plugin: Arc<dyn plugin_api::ExternalPlugin> = Arc::from(*plugin_box);
            // Read the per-file capability once, at load: a per-file plugin
            // is registered for salsa-cached dispatch and carries its id; a
            // project-wide one keeps `None`.
            let per_file_id = plugin
                .per_file()
                .is_some()
                .then(|| register_external_per_file(Arc::clone(&plugin)));
            out.push(NativePlugin {
                kind: NativePluginKind::External {
                    name,
                    plugin,
                    per_file_id,
                    _lib: Arc::clone(&lib),
                },
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Native plugins ported from Python. `module_dunders` is *per-file* (its
// output for a file is a pure function of that file → salsa-cached);
// `init_subclass` is project-wide (subclasses live in other files, so its
// output isn't file-local). The project-wide `py`-taking ctx accessors are
// reached via `Python::with_gil` (the harness already holds the GIL).
// ---------------------------------------------------------------------------

const MODULE_DUNDERS_PREFIX: &str = "<dunder>:";
const INIT_SUBCLASS_PREFIX: &str = "<__init_subclass__>:";
const SERVER_CONFIG_PREFIX: &str = "<server-config>:";
const UNITTEST_PREFIX: &str = "<unittest>:";

/// Conventional filenames Gunicorn / Hypercorn load at startup. The default
/// the [`NativePlugin::server_config`] factory supplies when no `filenames`
/// are given; callers pass a custom set for other server-config basenames.
const SERVER_CONFIG_FILENAMES: [&str; 4] = [
    "gunicorn.conf.py",
    "gunicorn_conf.py",
    "hypercorn.conf.py",
    "hypercorn_conf.py",
];

/// stdlib `unittest` lifecycle hooks kept alive in any file importing
/// `unittest` (mirrors `dead_cst.contrib.unittest._MODULE_HOOKS`).
const UNITTEST_MODULE_HOOKS: [&str; 3] = ["setUpModule", "tearDownModule", "load_tests"];

/// Base classes whose transitive subclasses are kept alive (mirrors
/// `dead_cst.contrib.unittest._UNITTEST_BASE_FQNAMES`).
const UNITTEST_BASE_FQNAMES: [&str; 2] = ["unittest.TestCase", "unittest.IsolatedAsyncioTestCase"];

/// Per-file port of `dead_cst.plugins.module_dunders.ModuleDundersPlugin`:
/// pin a file's module-level dunders (variables + PEP 562 functions) and its
/// `__future__` imports as entrypoints. All file-local — one synthetic
/// entrypoint node per file with edges to those decls.
pub(crate) struct ModuleDundersPluginImpl;

impl PerFileNativePluginImpl for ModuleDundersPluginImpl {
    fn run_on_file(&self, file_ctx: &FileContext<'_>, sink: &mut Vec<FileLocalOp>) {
        let mut targets: Vec<u32> = Vec::new();
        for (local_idx, node) in file_ctx.nodes().iter().enumerate() {
            let is_dunder_decl = matches!(node.kind.as_static_str(), "variable" | "function")
                && is_dunder_name(&node.fqname);
            let is_future = node
                .imports
                .as_ref()
                .is_some_and(|imp| imp.module == "__future__");
            if is_dunder_decl || is_future {
                targets.push(local_idx as u32);
            }
        }
        if targets.is_empty() {
            return;
        }
        let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
        sink.push(FileLocalOp::Node {
            fqname: format!("{MODULE_DUNDERS_PREFIX}{}", file_ctx.module_fqname()),
            kind: synthetic,
            flags: NodeFlags::ENTRYPOINT,
            edges_to_local_idx: targets,
            edges_from_local_idx: Vec::new(),
        });
    }
}

/// Port of `dead_cst.plugins.init_subclass.InitSubclassPlugin`: for each
/// class defining `__init_subclass__`, emit a marker node reachable from the
/// parent that keeps every transitive subclass alive.
pub(crate) struct InitSubclassPluginImpl;

impl NativePluginImpl for InitSubclassPluginImpl {
    fn name(&self) -> &str {
        "InitSubclassPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            let parents = ctx.find_classes_defining_method_indices(py, "__init_subclass__")?;
            let attrs = ctx.node_attrs(parents.clone())?;
            let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
            for (parent_idx, attr) in parents.iter().zip(attrs.iter()) {
                let subclass_idxs = ctx.find_subclasses_of_idx(*parent_idx)?;
                sink.push(PreparedOp::Node {
                    fqname: format!("{INIT_SUBCLASS_PREFIX}{}", attr.fqname),
                    kind: synthetic,
                    path: attr.path.clone(),
                    flags: 0,
                    edges_from_idx: vec![*parent_idx],
                    edges_to_idx: subclass_idxs,
                });
            }
            Ok(())
        })
    }
}

/// Test-only counter: number of times [`server_config_run_on_file`] actually
/// executed (i.e. salsa cache *misses* for the configured per-file query). A
/// salsa hit reuses the cached ops without touching the body, so this counter
/// stays flat for unchanged files across a ``re_materialize``. Surfaced to
/// tests via :func:`_server_config_run_count` /
/// :func:`_reset_server_config_run_count`.
static SERVER_CONFIG_RUN_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Test helper — total `server_config_run_on_file` executions since the last
/// reset. Lets the cache-behaviour test assert that an unchanged server-config
/// file isn't re-run on ``re_materialize`` and that distinct filename configs
/// key separately.
#[pyfunction]
pub(crate) fn _server_config_run_count() -> usize {
    SERVER_CONFIG_RUN_COUNT.load(Ordering::Relaxed)
}

/// Test helper — zero the [`SERVER_CONFIG_RUN_COUNT`] counter.
#[pyfunction]
pub(crate) fn _reset_server_config_run_count() {
    SERVER_CONFIG_RUN_COUNT.store(0, Ordering::Relaxed);
}

/// Per-file body for `ServerConfigPlugin` (see
/// [`ConfiguredPerFile::ServerConfig`], ported from
/// `dead_cst.contrib.server_config.ServerConfigPlugin`): when a file's basename
/// is one of `filenames`, mint a `<server-config>:` entrypoint keeping that
/// file's whole top-level surface alive. File-local — the match and the targets
/// are both functions of the single file.
fn server_config_run_on_file(
    filenames: &[String],
    file_ctx: &FileContext<'_>,
    sink: &mut Vec<FileLocalOp>,
) {
    SERVER_CONFIG_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
    let nodes = file_ctx.nodes();
    let path = nodes[0].path.as_str();
    let basename = path
        .rsplit_once(std::path::MAIN_SEPARATOR)
        .map(|(_, name)| name)
        .unwrap_or(path);
    if !filenames.iter().any(|f| f == basename) {
        return;
    }
    // `_TARGET_KINDS` from the Python plugin: every top-level surface
    // node (the module node included), excluding `type_alias`.
    let targets: Vec<u32> = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(
                node.kind.as_static_str(),
                "module" | "function" | "class" | "variable" | "import"
            )
        })
        .map(|(local_idx, _)| local_idx as u32)
        .collect();
    if targets.is_empty() {
        return;
    }
    let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
    sink.push(FileLocalOp::Node {
        fqname: format!("{SERVER_CONFIG_PREFIX}{}", file_ctx.module_fqname()),
        kind: synthetic,
        flags: NodeFlags::ENTRYPOINT,
        edges_to_local_idx: targets,
        edges_from_local_idx: Vec::new(),
    });
}

/// Project-wide port of `dead_cst.contrib.unittest.UnittestPlugin`: keep
/// every transitive subclass of `unittest.TestCase` /
/// `IsolatedAsyncioTestCase` alive, plus module-level lifecycle hooks
/// (`setUpModule` / `tearDownModule` / `load_tests`) in any file that
/// imports `unittest`. Cross-file (the subclass walk spans files), so it
/// can't be per-file.
pub(crate) struct UnittestPluginImpl;

impl NativePluginImpl for UnittestPluginImpl {
    fn name(&self) -> &str {
        "UnittestPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        // O(1) presence probe — short-circuit before the subclass walk.
        if !ctx.has_imports_of("unittest")? {
            return Ok(());
        }
        let import_idxs = ctx.find_imports_of_indices("unittest")?;
        let importer_paths: std::collections::HashSet<String> =
            ctx.node_paths(import_idxs)?.into_iter().collect();

        // path -> [decl_idx, ...] seeds to keep alive.
        let mut decls_by_path: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        Python::with_gil(|py| -> PyResult<()> {
            for base in UNITTEST_BASE_FQNAMES {
                let sub_idxs = ctx.find_subclasses_indices(py, base, true)?;
                if sub_idxs.is_empty() {
                    continue;
                }
                let paths = ctx.node_paths(sub_idxs.clone())?;
                for (idx, path) in sub_idxs.into_iter().zip(paths) {
                    decls_by_path.entry(path).or_default().push(idx);
                }
            }
            Ok(())
        })?;

        // Lifecycle hooks: function nodes in an importer file whose
        // trailing fqname segment is one of the module hooks.
        {
            let outputs = ctx.materialized("UnittestPlugin")?;
            for (idx, node) in outputs.builder.nodes.iter().enumerate() {
                if node.kind != "function" || !importer_paths.contains(node.path.as_str()) {
                    continue;
                }
                let simple = node
                    .fqname
                    .rsplit_once('.')
                    .map(|(_, n)| n)
                    .unwrap_or(node.fqname.as_str());
                if UNITTEST_MODULE_HOOKS.contains(&simple) {
                    decls_by_path
                        .entry(node.path.clone())
                        .or_default()
                        .push(idx);
                }
            }
        }

        if decls_by_path.is_empty() {
            return Ok(());
        }
        let paths: Vec<String> = decls_by_path.keys().cloned().collect();
        let module_idxs = ctx.modules_for_paths(paths.clone())?;
        let present: Vec<(String, usize)> = paths
            .into_iter()
            .zip(module_idxs)
            .filter_map(|(p, idx)| idx.map(|i| (p, i)))
            .collect();
        if present.is_empty() {
            return Ok(());
        }
        let module_attrs = ctx.node_attrs(present.iter().map(|(_, i)| *i).collect())?;
        let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
        for ((path, _idx), attr) in present.iter().zip(module_attrs.iter()) {
            sink.push(PreparedOp::Node {
                fqname: format!("{UNITTEST_PREFIX}{}", attr.fqname),
                kind: synthetic,
                path: path.clone(),
                flags: NodeFlags::TESTCASE,
                edges_from_idx: Vec::new(),
                edges_to_idx: decls_by_path[path].clone(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DispatchAppPlugin — project-wide port of
// `dead_cst.plugins.decl_shapes.DispatchAppPlugin` plus the seven framework
// configs that drove it (flask / fastapi / typer / cyclopts / slack_bolt /
// fastmcp / celery). Cross-file by nature — a handler `@app.route(...)` in one
// file wires to an `app = Flask()` variable that may live in another file — so
// this is a project-wide `NativePluginImpl`, not a per-file one.
//
// Verbatim port of `DispatchAppPlugin._gather_one` (discovery) + `.policy`
// (steps 3-6 of emission). The Python harness auto-batched the gather across
// plugins purely for speed; native plugins run independently with identical
// per-plugin output (the batching was a perf optimization, not a semantic one).
// ---------------------------------------------------------------------------

/// Celery-style appless `@shared_task` fan-out layered on top of the standard
/// dispatch policy. Mirrors `CeleryPlugin.policy`'s `super().policy(...)` then
/// shared-task pass: every top-level function decorated with one of `names`
/// (imported from `module`) is kept alive via one `<marker_prefix><basename>`
/// entrypoint per file. `None` for non-celery configs.
struct SharedTaskFanout {
    module: String,
    names: Vec<String>,
    marker_prefix: String,
}

/// Pure-data description of a dispatch-app framework — the Rust twin of the
/// Python `DispatchAppSpec` (plus the optional celery `shared_task` extension).
/// The seven bundled framework configs are built by the `*_config()` helpers
/// below; the Python `NativePlugin.dispatch_app(...)` factory builds one from
/// caller-supplied values (owned, so the config need not be `'static`).
pub(crate) struct DispatchAppConfig {
    marker_prefix: String,
    app_classes: Vec<String>,
    registration_decorators: Vec<String>,
    seed_as_entrypoint: bool,
    shared_task: Option<SharedTaskFanout>,
}

/// Project-wide native plugin wrapping a [`DispatchAppConfig`]. One instance
/// per framework, constructed via the `NativePlugin::flask()` &c. factories
/// or the custom `NativePlugin::dispatch_app(...)` factory.
pub(crate) struct DispatchAppPluginImpl {
    name: String,
    config: DispatchAppConfig,
}

impl DispatchAppPluginImpl {
    /// Cheap import-presence + config-completeness guard (port of
    /// `DispatchAppPlugin._is_active`). Skips the subclass walk when no file
    /// imports any app class's root module.
    fn is_active(&self, ctx: &ProjectContext) -> PyResult<bool> {
        let cfg = &self.config;
        if cfg.app_classes.is_empty() || cfg.registration_decorators.is_empty() {
            return Ok(false);
        }
        for fqn in &cfg.app_classes {
            if let Some((module, _)) = fqn.rsplit_once('.') {
                if !module.is_empty() && ctx.has_imports_of(module)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

/// Trailing path component (`Path(path).name`) for the celery shared-task
/// marker; matches `server_config_run_on_file`'s basename handling.
fn path_basename(path: &str) -> &str {
    path.rsplit_once(std::path::MAIN_SEPARATOR)
        .map(|(_, name)| name)
        .unwrap_or(path)
}

/// Trailing dotted segment of an fqname (`fqname.rsplit(".", 1)[-1]`).
fn simple_name(fqname: &str) -> &str {
    fqname.rsplit_once('.').map(|(_, n)| n).unwrap_or(fqname)
}

impl NativePluginImpl for DispatchAppPluginImpl {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        let cfg = &self.config;
        Python::with_gil(|py| -> PyResult<()> {
            if !self.is_active(ctx)? {
                return Ok(());
            }

            // --- Phase 1: gather indices. Each ctx call takes (and releases)
            // its own `materialized` read guard; none is held across another,
            // matching the convention in `UnittestPluginImpl`. ---

            // module_to_names: {module -> {ctor simple name}}, expanded
            // transitively over subclasses of each app class.
            let mut module_to_names: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
            for fqn in &cfg.app_classes {
                if let Some((module, name)) = fqn.rsplit_once('.') {
                    if !module.is_empty() && !name.is_empty() {
                        module_to_names
                            .entry(module.to_string())
                            .or_default()
                            .insert(name.to_string());
                    }
                }
            }
            for fqn in &cfg.app_classes {
                let sub_idxs = ctx.find_subclasses_indices(py, fqn, true)?;
                if sub_idxs.is_empty() {
                    continue;
                }
                for attr in ctx.node_attrs(sub_idxs)? {
                    if let Some((sub_module, sub_name)) = attr.fqname.rsplit_once('.') {
                        if !sub_module.is_empty() && !sub_name.is_empty() {
                            module_to_names
                                .entry(sub_module.to_string())
                                .or_default()
                                .insert(sub_name.to_string());
                        }
                    }
                }
            }
            if module_to_names.is_empty() {
                return Ok(());
            }

            // direct: top-level `X = Ctor(...)` constructions, deduped by var
            // idx across modules (the single-plugin path ignores `class_name`).
            let mut direct_seen: FxHashSet<usize> = FxHashSet::default();
            let mut direct: Vec<usize> = Vec::new();
            for (module, names) in &module_to_names {
                let mut names_vec: Vec<String> = names.iter().cloned().collect();
                names_vec.sort();
                let rows = ctx.find_instance_constructions(
                    py,
                    std::slice::from_ref(module),
                    names_vec,
                    None,
                    false,
                )?;
                for (var_idx, _class_name, _args) in rows {
                    if direct_seen.insert(var_idx) {
                        direct.push(var_idx);
                    }
                }
            }

            // factory_decls (seed_as_entrypoint only): functions/classes that
            // return an app instance, deduped by (decl idx, kind).
            let mut factory_decls: Vec<(usize, String)> = Vec::new();
            if cfg.seed_as_entrypoint {
                let mut factory_seen: FxHashSet<(usize, String)> = FxHashSet::default();
                for (module, names) in &module_to_names {
                    let mut names_vec: Vec<String> = names.iter().cloned().collect();
                    names_vec.sort();
                    let rows =
                        ctx.find_factory_decls(py, std::slice::from_ref(module), names_vec)?;
                    for (decl_idx, kinds) in rows {
                        for kind in kinds {
                            if factory_seen.insert((decl_idx, kind.clone())) {
                                factory_decls.push((decl_idx, kind));
                            }
                        }
                    }
                }
            }

            // handlers: `@<owner>.<reg_decorator>(...)`-decorated functions.
            let reg_decorators: Vec<String> = cfg.registration_decorators.clone();
            let handlers: Vec<(String, usize)> = ctx
                .find_handler_decorators(py, reg_decorators, None, false)?
                .into_iter()
                .map(|(owner, decorated_idx, _args)| (owner, decorated_idx))
                .collect();

            // factory_reachers (seed + factory): union of every factory decl's
            // direct predecessors. Inverts step 6's question (is this var's
            // *direct* successor a factory?) into one membership set.
            let mut factory_reachers: FxHashSet<usize> = FxHashSet::default();
            if cfg.seed_as_entrypoint && !factory_decls.is_empty() {
                let mut decl_seen: FxHashSet<usize> = FxHashSet::default();
                for &(decl_idx, _) in &factory_decls {
                    if !decl_seen.insert(decl_idx) {
                        continue;
                    }
                    for pred in ctx.direct_predecessors_idx(decl_idx, 0)? {
                        factory_reachers.insert(pred);
                    }
                }
            }

            // shared_task (celery): `@shared_task`-decorated decls.
            let shared_idxs: Vec<usize> = match &cfg.shared_task {
                Some(st) => {
                    let modules = [st.module.clone()];
                    let names: Vec<String> = st.names.clone();
                    ctx.find_decorated_decls(py, &modules, names, None, false)?
                        .into_iter()
                        .map(|(decl_idx, _args)| decl_idx)
                        .collect()
                }
                None => Vec::new(),
            };

            // --- Phase 2: resolve paths/fqnames + assemble ops under one
            // `materialized` read guard. No ctx call that re-acquires the guard
            // runs in this block (all index gathering happened above). ---
            let outputs = ctx.materialized("DispatchAppPlugin")?;
            let nodes = &outputs.builder.nodes;
            let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");

            let app_prefix = format!("<{}-app>:", cfg.marker_prefix);
            let factory_prefix = format!("<{}-factory>:", cfg.marker_prefix);

            // vars_by_file: (path, simple var name) -> first var idx wins.
            let mut vars_by_file: FxHashMap<(String, String), usize> = FxHashMap::default();
            for (idx, node) in nodes.iter().enumerate() {
                if node.kind != "variable" {
                    continue;
                }
                vars_by_file
                    .entry((node.path.clone(), simple_name(&node.fqname).to_string()))
                    .or_insert(idx);
            }

            // Steps 1 + 3: build direct_by_owner and entrypoint-promote each
            // direct construction (the latter only under seed_as_entrypoint).
            let mut direct_by_owner: FxHashMap<(String, String), Vec<usize>> = FxHashMap::default();
            for &var_idx in &direct {
                let node = &nodes[var_idx];
                direct_by_owner
                    .entry((node.path.clone(), simple_name(&node.fqname).to_string()))
                    .or_default()
                    .push(var_idx);
                if cfg.seed_as_entrypoint {
                    sink.push(PreparedOp::Node {
                        fqname: format!("{app_prefix}{}", node.fqname),
                        kind: synthetic,
                        path: node.path.clone(),
                        flags: NodeFlags::ENTRYPOINT,
                        edges_from_idx: Vec::new(),
                        edges_to_idx: vec![var_idx],
                    });
                }
            }

            // Step 4: factory markers so step 6's reachability walk can find
            // them. `factory_decls` is empty unless seed_as_entrypoint.
            for (decl_idx, kind) in &factory_decls {
                let node = &nodes[*decl_idx];
                sink.push(PreparedOp::Node {
                    fqname: format!("{factory_prefix}{kind}:{}", node.fqname),
                    kind: synthetic,
                    path: node.path.clone(),
                    flags: 0,
                    edges_from_idx: vec![*decl_idx],
                    edges_to_idx: Vec::new(),
                });
            }

            // Step 5: wire handler decorators to their owner var. Seed mode
            // uses vars_by_file (so `app = create_app()` factory chains pick up
            // edges); pure-dispatch mode wires only to direct constructions (so
            // a star-imported `app = App()` stays invisible).
            for (owner, decorated_idx) in &handlers {
                let key = (nodes[*decorated_idx].path.clone(), owner.clone());
                if cfg.seed_as_entrypoint {
                    if let Some(&var_idx) = vars_by_file.get(&key) {
                        sink.push(PreparedOp::Edge {
                            src_idx: var_idx,
                            dst_idx: *decorated_idx,
                            flags: 0,
                        });
                    }
                } else if let Some(var_idxs) = direct_by_owner.get(&key) {
                    for &var_idx in var_idxs {
                        sink.push(PreparedOp::Edge {
                            src_idx: var_idx,
                            dst_idx: *decorated_idx,
                            flags: 0,
                        });
                    }
                }
            }

            // Step 6: factory walk — entrypoint-promote vars whose *direct*
            // successor is a factory decl (seed + factory only). The
            // `factory_reachers` membership test rules out the two-hop
            // `wrapper = app; app = create_app()` over-promotion case.
            if cfg.seed_as_entrypoint && !factory_decls.is_empty() {
                let mut classified: FxHashSet<(String, String)> = FxHashSet::default();
                for (owner, decorated_idx) in &handlers {
                    let key = (nodes[*decorated_idx].path.clone(), owner.clone());
                    if direct_by_owner.contains_key(&key) || classified.contains(&key) {
                        continue;
                    }
                    let Some(&var_idx) = vars_by_file.get(&key) else {
                        continue;
                    };
                    if !factory_reachers.contains(&var_idx) {
                        continue;
                    }
                    classified.insert(key);
                    let node = &nodes[var_idx];
                    sink.push(PreparedOp::Node {
                        fqname: format!("{app_prefix}{}", node.fqname),
                        kind: synthetic,
                        path: node.path.clone(),
                        flags: NodeFlags::ENTRYPOINT,
                        edges_from_idx: Vec::new(),
                        edges_to_idx: vec![var_idx],
                    });
                }
            }

            // Celery `@shared_task` fan-out: one `<marker_prefix><basename>`
            // entrypoint per file, keeping every appless task callable alive.
            if let Some(st) = &cfg.shared_task {
                let mut by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
                let mut path_order: Vec<String> = Vec::new();
                for &decl_idx in &shared_idxs {
                    let path = nodes[decl_idx].path.clone();
                    if !by_path.contains_key(&path) {
                        path_order.push(path.clone());
                    }
                    by_path.entry(path).or_default().push(decl_idx);
                }
                for path in path_order {
                    let target_idxs = by_path.remove(&path).unwrap_or_default();
                    sink.push(PreparedOp::Node {
                        fqname: format!("{}{}", st.marker_prefix, path_basename(&path)),
                        kind: synthetic,
                        path: path.clone(),
                        flags: NodeFlags::ENTRYPOINT,
                        edges_from_idx: Vec::new(),
                        edges_to_idx: target_idxs,
                    });
                }
            }

            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Baked framework configs. Values mirror the deleted Python contrib modules
// (`dead_cst.contrib.{flask,fastapi,typer,cyclopts,slack_bolt,fastmcp,celery}`)
// verbatim. `seed_as_entrypoint` defaults to true for factory-aware web/task
// frameworks; the pure-dispatch CLIs (typer / cyclopts) set it false.
// ---------------------------------------------------------------------------

/// Own a slice of string literals as a `Vec<String>` — keeps the baked
/// config builders below readable now that the fields are owned.
fn owned(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

fn flask_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "flask".to_string(),
        app_classes: owned(&["flask.Flask"]),
        registration_decorators: owned(&[
            "route",
            "get",
            "post",
            "put",
            "delete",
            "patch",
            "before_request",
            "after_request",
            "teardown_request",
            "teardown_appcontext",
            "before_first_request",
            "before_app_request",
            "after_app_request",
            "teardown_app_request",
            "before_app_first_request",
            "errorhandler",
            "app_errorhandler",
            "context_processor",
            "app_context_processor",
            "template_filter",
            "app_template_filter",
            "template_test",
            "app_template_test",
            "template_global",
            "app_template_global",
            "url_value_preprocessor",
            "app_url_value_preprocessor",
            "url_defaults",
            "app_url_defaults",
            "shell_context_processor",
            "record",
            "record_once",
        ]),
        seed_as_entrypoint: true,
        shared_task: None,
    }
}

fn fastapi_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "fastapi".to_string(),
        app_classes: owned(&["fastapi.FastAPI"]),
        registration_decorators: owned(&[
            "get",
            "post",
            "put",
            "delete",
            "patch",
            "options",
            "head",
            "trace",
            "api_route",
            "websocket",
            "websocket_route",
            "middleware",
            "exception_handler",
            "on_event",
        ]),
        seed_as_entrypoint: true,
        shared_task: None,
    }
}

fn typer_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "typer".to_string(),
        app_classes: owned(&["typer.Typer"]),
        registration_decorators: owned(&["command", "callback"]),
        seed_as_entrypoint: false,
        shared_task: None,
    }
}

fn cyclopts_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "cyclopts".to_string(),
        app_classes: owned(&["cyclopts.App"]),
        registration_decorators: owned(&["command", "default"]),
        seed_as_entrypoint: false,
        shared_task: None,
    }
}

fn slack_bolt_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "slack-bolt".to_string(),
        app_classes: owned(&["slack_bolt.App", "slack_bolt.async_app.AsyncApp"]),
        registration_decorators: owned(&[
            "event", "message", "command", "action", "shortcut", "view", "options", "error",
            "step", "function",
        ]),
        seed_as_entrypoint: true,
        shared_task: None,
    }
}

fn fastmcp_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "fastmcp".to_string(),
        app_classes: owned(&["fastmcp.FastMCP", "mcp.server.fastmcp.FastMCP"]),
        registration_decorators: owned(&["tool", "resource", "prompt", "completion"]),
        seed_as_entrypoint: true,
        shared_task: None,
    }
}

fn celery_config() -> DispatchAppConfig {
    DispatchAppConfig {
        marker_prefix: "celery".to_string(),
        app_classes: owned(&["celery.Celery"]),
        registration_decorators: owned(&["task"]),
        seed_as_entrypoint: true,
        shared_task: Some(SharedTaskFanout {
            module: "celery".to_string(),
            names: owned(&["shared_task"]),
            marker_prefix: "<celery-shared>:".to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// ClickPlugin — project-wide port of `dead_cst.contrib.click.ClickPlugin`.
//
// Click is *not* a dispatch-app framework: groups are never seeded as
// entrypoints (reachability enters via `[project.scripts]` / `__main__` /
// `add_command`), and a command handler can itself *become* a group
// (`@group.group()` defines a nested group whose own `@subgroup.command()`
// handlers must then be wired). That second property needs a group->handler
// *fixpoint* — a single registration pass over a config (as the dispatch-app
// engine does) can't express it — so Click gets its own impl instead of a
// `DispatchAppConfig`. Cross-file by nature (a group in one file, handlers
// registered on it in another), hence project-wide.
// ---------------------------------------------------------------------------

const CLICK_GROUP_DECORATORS: [&str; 2] = ["group", "Group"];
const CLICK_REGISTRATION_DECORATORS: [&str; 3] = ["command", "group", "result_callback"];
const CLICK_SUBGROUP_DECORATOR: &str = "group";

pub(crate) struct ClickPluginImpl;

impl NativePluginImpl for ClickPluginImpl {
    fn name(&self) -> &str {
        "click"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            // Cheap import-presence guard (`has_imports_of("click")`).
            if !ctx.has_imports_of("click")? {
                return Ok(());
            }
            let click_mod = ["click".to_string()];

            // --- Phase 1: gather indices. Each ctx call takes (and releases)
            // its own `materialized` read guard; mirrors DispatchAppPluginImpl. ---

            // Groups via `@click.group` / `@click.Group`-decorated decls.
            let group_decls: Vec<usize> = ctx
                .find_decorated_decls(
                    py,
                    &click_mod,
                    CLICK_GROUP_DECORATORS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    None,
                    false,
                )?
                .into_iter()
                .map(|(idx, _)| idx)
                .collect();

            // Groups via `X = click.Group(...)` constructions.
            let group_ctors: Vec<usize> = ctx
                .find_instance_constructions(
                    py,
                    &click_mod,
                    vec!["Group".to_string()],
                    None,
                    false,
                )?
                .into_iter()
                .map(|(idx, _, _)| idx)
                .collect();

            // No groups => no wiring to do (the fixpoint would no-op anyway).
            if group_decls.is_empty() && group_ctors.is_empty() {
                return Ok(());
            }

            // Handlers: `@<owner>.{command,group,result_callback}(...)`.
            let handlers: Vec<(String, usize)> = ctx
                .find_handler_decorators(
                    py,
                    CLICK_REGISTRATION_DECORATORS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    None,
                    false,
                )?
                .into_iter()
                .map(|(owner, idx, _)| (owner, idx))
                .collect();

            // Subgroup links: handlers decorated specifically with
            // `@<owner>.group(...)`. Wiring such a handler promotes it to a
            // group so its own `@<handler>.command()` handlers wire next pass.
            let subgroup_links: FxHashSet<(usize, String)> = ctx
                .find_handler_decorators(
                    py,
                    vec![CLICK_SUBGROUP_DECORATOR.to_string()],
                    None,
                    false,
                )?
                .into_iter()
                .map(|(owner, idx, _)| (idx, owner))
                .collect();

            // --- Phase 2: resolve paths/fqnames + run the group->handler
            // fixpoint under one `materialized` read guard. ---
            let outputs = ctx.materialized("ClickPlugin")?;
            let nodes = &outputs.builder.nodes;

            // groups_by_owner: (path, simple name) -> [group idx, ...].
            let mut groups_by_owner: FxHashMap<(String, String), Vec<usize>> = FxHashMap::default();
            for &idx in group_decls.iter().chain(group_ctors.iter()) {
                let node = &nodes[idx];
                groups_by_owner
                    .entry((node.path.clone(), simple_name(&node.fqname).to_string()))
                    .or_default()
                    .push(idx);
            }

            // Fixpoint: wire each handler to its owning group(s); a newly wired
            // subgroup handler becomes a group, exposing its own handlers on the
            // next pass. `emitted` dedups edges so the loop terminates once no
            // new group is discovered. Verbatim port of `ClickPlugin.run`'s loop.
            let mut emitted: FxHashSet<(usize, usize)> = FxHashSet::default();
            let mut changed = true;
            while changed {
                changed = false;
                for (owner_name, decorated_idx) in &handlers {
                    let path = nodes[*decorated_idx].path.clone();
                    // Snapshot the owner's group idxs: `add_group` below may
                    // mutate `groups_by_owner`, and the dedup + outer loop make
                    // deferring a same-pass insertion to the next pass
                    // fixpoint-equivalent.
                    let Some(owner_idxs) = groups_by_owner.get(&(path.clone(), owner_name.clone()))
                    else {
                        continue;
                    };
                    for owner_idx in owner_idxs.clone() {
                        if !emitted.insert((owner_idx, *decorated_idx)) {
                            continue;
                        }
                        sink.push(PreparedOp::Edge {
                            src_idx: owner_idx,
                            dst_idx: *decorated_idx,
                            flags: 0,
                        });
                        if subgroup_links.contains(&(*decorated_idx, owner_name.clone())) {
                            let simple = simple_name(&nodes[*decorated_idx].fqname).to_string();
                            groups_by_owner
                                .entry((path.clone(), simple))
                                .or_default()
                                .push(*decorated_idx);
                            changed = true;
                        }
                    }
                }
            }

            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// MockPatchPlugin — project-wide port of `dead_cst.contrib.mock_patch`.
// Resolve string-fqname patch targets (`unittest.mock.patch` / `mock.patch`,
// pytest-mock's `mocker.patch`, pytest's `monkeypatch.setattr` / `.delattr`)
// to their decls and emit keep-alive edges. Cross-file (a patch string in a
// test file targets a decl in another), hence project-wide.
// ---------------------------------------------------------------------------

const PATCH_TARGET_PREFIX: &str = "<patch-target>:";

pub(crate) struct MockPatchPluginImpl;

impl NativePluginImpl for MockPatchPluginImpl {
    fn name(&self) -> &str {
        "MockPatchPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        // Phase 1: gather (enclosing decl idx, target fqname) rows from the
        // four patch-call shapes. Each query takes/releases its own guard.
        let rows: Vec<(usize, String)> =
            Python::with_gil(|py| -> PyResult<Vec<(usize, String)>> {
                let mut rows: Vec<(usize, String)> = Vec::new();
                // `patch("X.Y")` imported from unittest.mock / mock.
                let mock_modules = ["unittest.mock".to_string(), "mock".to_string()];
                for (owner_idx, target, _args) in
                    ctx.find_calls_to_imported(py, &mock_modules, "patch", 0, None, false)?
                {
                    rows.push((owner_idx, target));
                }
                // `mocker.patch("X.Y")` (pytest-mock fixture).
                for (owner_idx, target, _args) in
                    ctx.find_calls_on_var(py, "mocker", "patch", 0, None, None, false)?
                {
                    rows.push((owner_idx, target));
                }
                // `monkeypatch.setattr("X.Y", v)` (2 positional) /
                // `monkeypatch.delattr("X.Y")` (1 positional). The
                // required-positional count distinguishes the fqname form from the
                // object form (`setattr(obj, "name", v)`).
                for (attr, required) in [("setattr", 2usize), ("delattr", 1usize)] {
                    for (owner_idx, target, _args) in ctx.find_calls_on_var(
                        py,
                        "monkeypatch",
                        attr,
                        0,
                        Some(required),
                        None,
                        false,
                    )? {
                        rows.push((owner_idx, target));
                    }
                }
                Ok(rows)
            })?;

        if rows.is_empty() {
            return Ok(());
        }

        // Bucket owners by target fqname, first-seen order preserved (matches
        // the Python dict insertion order).
        let mut order: Vec<String> = Vec::new();
        let mut owners_by_fqname: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for (owner_idx, fqname) in rows {
            if !owners_by_fqname.contains_key(&fqname) {
                order.push(fqname.clone());
            }
            owners_by_fqname.entry(fqname).or_default().push(owner_idx);
        }

        // Resolve targets per fqname (decls + the module node if the fqname is
        // itself a module). One synthetic `<patch-target>:<fqname>` per fqname,
        // emitted unconditionally — an empty target set still records the
        // patch site for introspection, matching the Python plugin.
        let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
        for fqname in order {
            let owners = &owners_by_fqname[&fqname];
            let mut target_idxs = ctx.find_declarations_indices(&fqname)?;
            if let Some(mod_idx) = ctx.find_module_idx(&fqname)? {
                target_idxs.push(mod_idx);
            }
            let owner_path = ctx
                .node_paths(vec![owners[0]])?
                .into_iter()
                .next()
                .unwrap_or_default();
            sink.push(PreparedOp::Node {
                fqname: format!("{PATCH_TARGET_PREFIX}{fqname}"),
                kind: synthetic,
                path: owner_path,
                flags: 0,
                edges_from_idx: owners.clone(),
                edges_to_idx: target_idxs,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DiscordPyPlugin — project-wide port of `dead_cst.contrib.discordpy`. Wire
// discord.py bots/clients, their `@bot.<verb>` / `@bot.tree.<verb>` handlers,
// Cog subclasses + module setup/teardown hooks, and `load_extension(...)`
// string targets into reachability. Cross-file by nature, hence project-wide.
// ---------------------------------------------------------------------------

const DISCORDPY_COG_PREFIX: &str = "<discordpy-cog>:";
const DISCORDPY_EXTENSION_PREFIX: &str = "<discordpy-extension>:";
const DISCORDPY_APP_MARKER: &str = "<discordpy-app>";
const DISCORD_PROBE_MODULES: [&str; 3] = ["discord", "discord.ext", "discord.ext.commands"];
const DISCORD_COMMANDS_BOT_KINDS: [&str; 2] = ["Bot", "AutoShardedBot"];
const DISCORD_CLIENT_KINDS: [&str; 2] = ["Client", "AutoShardedClient"];
const DISCORD_BOT_DECORATORS: [&str; 10] = [
    "command",
    "event",
    "listen",
    "group",
    "hybrid_command",
    "hybrid_group",
    "check",
    "check_once",
    "before_invoke",
    "after_invoke",
];
const DISCORD_TREE_DECORATORS: [&str; 2] = ["command", "context_menu"];
const DISCORD_COG_BASES: [&str; 2] = ["discord.ext.commands.Cog", "discord.ext.commands.GroupCog"];

pub(crate) struct DiscordPyPluginImpl;

impl NativePluginImpl for DiscordPyPluginImpl {
    fn name(&self) -> &str {
        "DiscordPyPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        // Cheap presence probe — short-circuit before any walk.
        let mut imports_discord = false;
        for m in DISCORD_PROBE_MODULES {
            if ctx.has_imports_of(m)? {
                imports_discord = true;
                break;
            }
        }
        if !imports_discord {
            return Ok(());
        }

        // Per-file gate: files that import discord.
        let mut discord_paths: FxHashSet<String> = FxHashSet::default();
        for m in DISCORD_PROBE_MODULES {
            let idxs = ctx.find_imports_of_indices(m)?;
            for p in ctx.node_paths(idxs)? {
                discord_paths.insert(p);
            }
        }

        let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");

        Python::with_gil(|py| -> PyResult<()> {
            // --- Phase 1: gather indices. Each ctx call takes its own guard. ---

            // 1. Bot / Client constructions (var idxs, in discovery order).
            let mut bot_var_idxs: Vec<usize> = Vec::new();
            let commands_mod = ["discord.ext.commands".to_string()];
            let mut bot_names: Vec<String> = DISCORD_COMMANDS_BOT_KINDS
                .iter()
                .map(|s| s.to_string())
                .collect();
            bot_names.sort();
            for (var_idx, _ctor, _args) in
                ctx.find_instance_constructions(py, &commands_mod, bot_names, None, false)?
            {
                bot_var_idxs.push(var_idx);
            }
            let discord_mod = ["discord".to_string()];
            let mut client_names: Vec<String> =
                DISCORD_CLIENT_KINDS.iter().map(|s| s.to_string()).collect();
            client_names.sort();
            for (var_idx, _ctor, _args) in
                ctx.find_instance_constructions(py, &discord_mod, client_names, None, false)?
            {
                bot_var_idxs.push(var_idx);
            }

            // 2 + 3. Handler decorators: single-attr `@<bot>.<verb>` and
            // two-level `@<bot>.tree.<verb>` slash commands.
            let bot_handlers: Vec<(String, usize)> = ctx
                .find_handler_decorators(
                    py,
                    DISCORD_BOT_DECORATORS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    None,
                    false,
                )?
                .into_iter()
                .map(|(owner, idx, _)| (owner, idx))
                .collect();
            let tree_handlers: Vec<(String, usize)> = ctx
                .find_handler_decorators_via(
                    py,
                    "tree",
                    DISCORD_TREE_DECORATORS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    None,
                    false,
                )?
                .into_iter()
                .map(|(owner, idx, _)| (owner, idx))
                .collect();

            // 4. Cog subclasses + their files; module setup/teardown hooks.
            let mut cogs_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
            let mut cog_path_order: Vec<String> = Vec::new();
            for base in DISCORD_COG_BASES {
                let cog_idxs = ctx.find_subclasses_indices(py, base, true)?;
                if cog_idxs.is_empty() {
                    continue;
                }
                let cog_paths = ctx.node_paths(cog_idxs.clone())?;
                for (idx, path) in cog_idxs.into_iter().zip(cog_paths) {
                    if !cogs_by_path.contains_key(&path) {
                        cog_path_order.push(path.clone());
                    }
                    cogs_by_path.entry(path).or_default().push(idx);
                }
            }
            let mut hook_funcs_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
            if !cogs_by_path.is_empty() {
                let hook_idxs = ctx.indices_where(
                    Some("function".to_string()),
                    None,
                    None,
                    None,
                    None,
                    Some(vec!["setup".to_string(), "teardown".to_string()]),
                    Some(cog_path_order.clone()),
                    None,
                    None,
                    None,
                    None,
                )?;
                if !hook_idxs.is_empty() {
                    let hook_paths = ctx.node_paths(hook_idxs.clone())?;
                    for (idx, path) in hook_idxs.into_iter().zip(hook_paths) {
                        hook_funcs_by_path.entry(path).or_default().push(idx);
                    }
                }
            }

            // 6. load_extension / load_extensions string-literal targets
            // (owner decl idx + extension fqname), raw — gated/deduped below.
            let mut raw_ext_calls: Vec<(usize, String)> = Vec::new();
            for attr in ["load_extension", "load_extensions"] {
                for (owner_idx, ext_fqname, _args) in
                    ctx.find_calls_on_attr(py, attr, 0, None, false)?
                {
                    raw_ext_calls.push((owner_idx, ext_fqname));
                }
            }

            // --- Phase 2a: build bot var map, emit bot entrypoints + handler
            // edges, gate + dedup extension calls. One guard, dropped after. ---
            let mut pending_extensions: Vec<(String, String)> = Vec::new();
            {
                let outputs = ctx.materialized("DiscordPyPlugin")?;
                let nodes = &outputs.builder.nodes;

                // bot_vars_by_file: path -> { simple var name -> var idx }.
                let mut bot_vars_by_file: FxHashMap<String, FxHashMap<String, usize>> =
                    FxHashMap::default();
                for &var_idx in &bot_var_idxs {
                    let node = &nodes[var_idx];
                    if !discord_paths.contains(node.path.as_str()) {
                        continue;
                    }
                    bot_vars_by_file
                        .entry(node.path.clone())
                        .or_default()
                        .insert(simple_name(&node.fqname).to_string(), var_idx);
                    sink.push(PreparedOp::Entrypoint {
                        decl_idx: var_idx,
                        marker: DISCORDPY_APP_MARKER.to_string(),
                    });
                }

                // Wire single-attr + two-level handler decorators to their bot.
                for (owner, decorated_idx) in bot_handlers.iter().chain(tree_handlers.iter()) {
                    let path = nodes[*decorated_idx].path.as_str();
                    if let Some(&owner_idx) = bot_vars_by_file.get(path).and_then(|m| m.get(owner))
                    {
                        sink.push(PreparedOp::Edge {
                            src_idx: owner_idx,
                            dst_idx: *decorated_idx,
                            flags: 0,
                        });
                    }
                }

                // Gate extension calls on importer files; dedup by fqname.
                let mut seen_extensions: FxHashSet<String> = FxHashSet::default();
                for (owner_idx, ext_fqname) in &raw_ext_calls {
                    let path = nodes[*owner_idx].path.clone();
                    if !discord_paths.contains(path.as_str()) {
                        continue;
                    }
                    if !seen_extensions.insert(ext_fqname.clone()) {
                        continue;
                    }
                    pending_extensions.push((path, ext_fqname.clone()));
                }
            }

            // --- Phase 2b: Cog entrypoint nodes (one per cog file). ---
            for path in &cog_path_order {
                let mut targets = cogs_by_path[path].clone();
                if let Some(hooks) = hook_funcs_by_path.get(path) {
                    targets.extend(hooks.iter().copied());
                }
                sink.push(PreparedOp::Node {
                    fqname: format!("{DISCORDPY_COG_PREFIX}{}", path_basename(path)),
                    kind: synthetic,
                    path: path.clone(),
                    flags: NodeFlags::ENTRYPOINT,
                    edges_from_idx: Vec::new(),
                    edges_to_idx: targets,
                });
            }

            // --- Phase 2c: extension surface fan-out (re-acquires a guard). ---
            if !pending_extensions.is_empty() {
                let ext_fqnames: Vec<String> =
                    pending_extensions.iter().map(|(_, e)| e.clone()).collect();
                let surfaces = ctx.module_surfaces_indices(ext_fqnames)?;
                for (owner_path, ext_fqname) in &pending_extensions {
                    let targets = surfaces.get(ext_fqname).cloned().unwrap_or_default();
                    if targets.is_empty() {
                        continue;
                    }
                    sink.push(PreparedOp::Node {
                        fqname: format!("{DISCORDPY_EXTENSION_PREFIX}{ext_fqname}"),
                        kind: synthetic,
                        path: owner_path.clone(),
                        flags: NodeFlags::ENTRYPOINT,
                        edges_from_idx: Vec::new(),
                        edges_to_idx: targets,
                    });
                }
            }

            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// DynamicImportFallbackPlugin — project-wide port of
// `dead_cst.plugins.dynamic_import`. Fan out each `EdgeFlags.DYNAMIC_IMPORT`
// edge whose dst is a module node to that module's exports. The include/exclude
// glob knobs gate the fan-out; with no filters every such edge fans out.
// Glob matching reuses Python's `pathlib.PurePosixPath.match` (sources) +
// `fnmatch.fnmatchcase` (targets) for byte-parity, but only when a filter is
// configured — the zero-config catch-all never touches Python.
// ---------------------------------------------------------------------------

pub(crate) struct DynamicImportFallbackPluginImpl {
    include_underscore: bool,
    respect_dunder_all: bool,
    exclude_sources: Vec<String>,
    exclude_targets: Vec<String>,
    include_sources: Vec<String>,
    include_targets: Vec<String>,
}

impl DynamicImportFallbackPluginImpl {
    fn needs_matching(&self) -> bool {
        !self.include_sources.is_empty()
            || !self.include_targets.is_empty()
            || !self.exclude_sources.is_empty()
            || !self.exclude_targets.is_empty()
    }

    /// Port of `_allowed`: include filters (if set) must all match; exclude
    /// filters (if set) must none match. `rel_pp` is the source path made
    /// relative to the project root as a `PurePosixPath`; `target_fqname` is
    /// the imported module's fqname.
    fn is_allowed(
        &self,
        fnmatch: &Bound<'_, PyAny>,
        pure_posix: &Bound<'_, PyAny>,
        src_path: &str,
        project_root: &str,
        target_fqname: &str,
    ) -> PyResult<bool> {
        let rel = std::path::Path::new(src_path)
            .strip_prefix(project_root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| src_path.to_string());
        let rel_pp = pure_posix.call1((rel,))?;
        if !self.include_sources.is_empty() && !match_pure_path(&rel_pp, &self.include_sources)? {
            return Ok(false);
        }
        if !self.include_targets.is_empty()
            && !match_fnmatch(fnmatch, target_fqname, &self.include_targets)?
        {
            return Ok(false);
        }
        if !self.exclude_sources.is_empty() && match_pure_path(&rel_pp, &self.exclude_sources)? {
            return Ok(false);
        }
        if !self.exclude_targets.is_empty()
            && match_fnmatch(fnmatch, target_fqname, &self.exclude_targets)?
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Exports of `module_fqname` per `_exports_indices_for`: `__all__` when
    /// `respect_dunder_all` and present, else the module's top-level decls
    /// (dropping `_`-prefixed names unless `include_underscore`).
    fn exports_for(&self, ctx: &ProjectContext, module_fqname: &str) -> PyResult<Vec<usize>> {
        if self.respect_dunder_all {
            if let Some(exports) = ctx.find_module_dunder_all_exports_indices(module_fqname)? {
                return Ok(exports);
            }
        }
        let decls = ctx.find_module_top_level_decls_indices(module_fqname)?;
        if self.include_underscore || decls.is_empty() {
            return Ok(decls);
        }
        let attrs = ctx.node_attrs(decls.clone())?;
        Ok(decls
            .into_iter()
            .zip(attrs)
            .filter(|(_idx, attr)| !simple_name(&attr.fqname).starts_with('_'))
            .map(|(idx, _attr)| idx)
            .collect())
    }
}

/// `any(rel_pp.match(p) for p in patterns)`.
fn match_pure_path(rel_pp: &Bound<'_, PyAny>, patterns: &[String]) -> PyResult<bool> {
    for p in patterns {
        if rel_pp.call_method1("match", (p,))?.extract::<bool>()? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `any(fnmatch.fnmatchcase(fqname, p) for p in patterns)`.
fn match_fnmatch(fnmatch: &Bound<'_, PyAny>, fqname: &str, patterns: &[String]) -> PyResult<bool> {
    for p in patterns {
        if fnmatch
            .call_method1("fnmatchcase", (fqname, p))?
            .extract::<bool>()?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

impl NativePluginImpl for DynamicImportFallbackPluginImpl {
    fn name(&self) -> &str {
        "DynamicImportFallbackPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        let project_root = ctx.project_root().to_string();

        // Snapshot DYNAMIC_IMPORT edges whose dst is a module node:
        // (src idx, src path, dst module fqname). One guard, dropped after.
        let candidates: Vec<(usize, String, String)> = {
            let outputs = ctx.materialized("DynamicImportFallbackPlugin")?;
            let b = &outputs.builder;
            b.edges
                .iter()
                .filter_map(|&(src, dst, flags)| {
                    if flags & EdgeFlags::DYNAMIC_IMPORT != 0 && b.nodes[dst].kind == "module" {
                        Some((src, b.nodes[src].path.clone(), b.nodes[dst].fqname.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        };
        if candidates.is_empty() {
            return Ok(());
        }

        // Apply include/exclude filters. Zero-config => every edge passes and
        // Python is never entered.
        let allowed: Vec<(usize, String)> = if self.needs_matching() {
            Python::with_gil(|py| -> PyResult<Vec<(usize, String)>> {
                let fnmatch = py.import_bound("fnmatch")?.into_any();
                let pure_posix = py.import_bound("pathlib")?.getattr("PurePosixPath")?;
                let mut out: Vec<(usize, String)> = Vec::new();
                for (src_idx, src_path, dst_fqname) in &candidates {
                    if self.is_allowed(
                        &fnmatch,
                        &pure_posix,
                        src_path,
                        &project_root,
                        dst_fqname,
                    )? {
                        out.push((*src_idx, dst_fqname.clone()));
                    }
                }
                Ok(out)
            })?
        } else {
            candidates
                .into_iter()
                .map(|(src, _path, fqname)| (src, fqname))
                .collect()
        };

        // Resolve exports per distinct module fqname (cached), fan each
        // matched edge out to every export.
        let mut cache: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for (src_idx, dst_fqname) in allowed {
            if !cache.contains_key(&dst_fqname) {
                let exports = self.exports_for(ctx, &dst_fqname)?;
                cache.insert(dst_fqname.clone(), exports);
            }
            for &export_idx in &cache[&dst_fqname] {
                sink.push(PreparedOp::Edge {
                    src_idx,
                    dst_idx: export_idx,
                    flags: 0,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ExplicitEntrypointPlugin — project-wide port of
// `dead_cst.plugins.explicit_entrypoint`. Mark user-specified symbols as
// entrypoints. The Python plugin buckets its `specs` (str / Path / re.Pattern)
// into the three pre-bucketed lists this impl carries; the matching itself is
// `find_nodes_matching_specs_indices` (already GIL-free).
// ---------------------------------------------------------------------------

const EXPLICIT_ENTRYPOINT_MARKER: &str = "<entrypoint>";

pub(crate) struct ExplicitEntrypointPluginImpl {
    regexes: Vec<String>,
    str_specs: Vec<String>,
    abs_paths: Vec<String>,
}

impl NativePluginImpl for ExplicitEntrypointPluginImpl {
    fn name(&self) -> &str {
        "ExplicitEntrypointPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        if self.regexes.is_empty() && self.str_specs.is_empty() && self.abs_paths.is_empty() {
            return Ok(());
        }
        let project_root = ctx.project_root().to_string();
        let idxs = ctx.find_nodes_matching_specs_indices(
            &project_root,
            self.regexes.clone(),
            self.str_specs.clone(),
            self.abs_paths.clone(),
        )?;
        for idx in idxs {
            sink.push(PreparedOp::Entrypoint {
                decl_idx: idx,
                marker: EXPLICIT_ENTRYPOINT_MARKER.to_string(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProjectScriptsPlugin — project-wide port of
// `dead_cst.plugins.project_scripts`. Mark every `[project.scripts]` entry in
// `pyproject.toml` as an entrypoint. TOML is parsed via Python's stdlib
// `tomllib` (no Rust toml dependency); a missing file is a no-op.
// ---------------------------------------------------------------------------

const PROJECT_SCRIPTS_PREFIX: &str = "<project.scripts>:";

pub(crate) struct ProjectScriptsPluginImpl {
    pyproject_path: Option<String>,
}

impl NativePluginImpl for ProjectScriptsPluginImpl {
    fn name(&self) -> &str {
        "ProjectScriptsPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        let pyproject = match &self.pyproject_path {
            Some(p) => p.clone(),
            None => std::path::Path::new(ctx.project_root())
                .join("pyproject.toml")
                .to_string_lossy()
                .into_owned(),
        };
        // Missing file => no-op (mirrors `load_toml`'s OSError -> None).
        let text = match std::fs::read_to_string(&pyproject) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        // Parse `[project.scripts]` (name -> "pkg.mod:func") via tomllib.
        let scripts: Vec<(String, String)> =
            Python::with_gil(|py| -> PyResult<Vec<(String, String)>> {
                let tomllib = py.import_bound("tomllib")?;
                let data = tomllib.call_method1("loads", (text,))?;
                let project = data.call_method1("get", ("project",))?;
                if project.is_none() {
                    return Ok(Vec::new());
                }
                let scripts_obj = project.call_method1("get", ("scripts",))?;
                if scripts_obj.is_none() {
                    return Ok(Vec::new());
                }
                let mut out: Vec<(String, String)> = Vec::new();
                for item in scripts_obj.call_method0("items")?.iter()? {
                    let (name, target): (String, String) = item?.extract()?;
                    out.push((name, target));
                }
                Ok(out)
            })?;

        let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
        for (script_name, target) in scripts {
            let (module_part, decl_part) = match target.split_once(':') {
                Some((m, d)) => (m, d),
                None => (target.as_str(), ""),
            };
            let fqname = if decl_part.is_empty() {
                module_part.to_string()
            } else {
                format!("{module_part}.{decl_part}")
            };
            let mut target_idxs = ctx.find_declarations_indices(&fqname)?;
            if target_idxs.is_empty() {
                if let Some(module_idx) = ctx.find_module_idx(module_part)? {
                    target_idxs.push(module_idx);
                }
            }
            if target_idxs.is_empty() {
                continue;
            }
            sink.push(PreparedOp::Node {
                fqname: format!("{PROJECT_SCRIPTS_PREFIX}{script_name}"),
                kind: synthetic,
                path: pyproject.clone(),
                flags: NodeFlags::ENTRYPOINT,
                edges_from_idx: Vec::new(),
                edges_to_idx: target_idxs,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PytestPlugin — project-wide port of `dead_cst.contrib.pytest`. Seed
// conftest decls, `test_*` functions / `Test*` classes, and
// `@pytest.fixture` functions as test-cases, then wire test/class -> fixture
// edges by parameter-name matching. Cross-file (fixtures in conftest serve
// tests anywhere), hence project-wide.
// ---------------------------------------------------------------------------

const PYTEST_CONFTEST_PREFIX: &str = "<pytest:conftest>:";
const PYTEST_TESTS_PREFIX: &str = "<pytest:tests>:";
const PYTEST_FIXTURES_PREFIX: &str = "<pytest:fixtures>:";

fn is_test_filename(name: &str) -> bool {
    (name.starts_with("test_") && name.ends_with(".py")) || name.ends_with("_test.py")
}

fn is_test_decl(kind: &str, fqname: &str) -> bool {
    let simple = simple_name(fqname);
    (kind == "function" && simple.starts_with("test_"))
        || (kind == "class" && simple.starts_with("Test"))
}

/// Emit a `TESTCASE` seed node keeping `target_idxs` alive (no-op when empty).
fn mark_seed(
    sink: &mut Vec<PreparedOp>,
    kind: &'static str,
    fqname: String,
    path: String,
    target_idxs: Vec<usize>,
) {
    if target_idxs.is_empty() {
        return;
    }
    sink.push(PreparedOp::Node {
        fqname,
        kind,
        path,
        flags: NodeFlags::TESTCASE,
        edges_from_idx: Vec::new(),
        edges_to_idx: target_idxs,
    });
}

/// `{ path: module_fqname }` for every path resolving to a project module.
fn module_fqnames_for(
    ctx: &ProjectContext,
    paths: &[String],
) -> PyResult<FxHashMap<String, String>> {
    if paths.is_empty() {
        return Ok(FxHashMap::default());
    }
    let module_idxs = ctx.modules_for_paths(paths.to_vec())?;
    let present: Vec<(String, usize)> = paths
        .iter()
        .cloned()
        .zip(module_idxs)
        .filter_map(|(p, m)| m.map(|i| (p, i)))
        .collect();
    if present.is_empty() {
        return Ok(FxHashMap::default());
    }
    let attrs = ctx.node_attrs(present.iter().map(|(_, i)| *i).collect())?;
    Ok(present
        .into_iter()
        .zip(attrs)
        .map(|((path, _), attr)| (path, attr.fqname))
        .collect())
}

pub(crate) struct PytestPluginImpl;

impl NativePluginImpl for PytestPluginImpl {
    fn name(&self) -> &str {
        "PytestPlugin"
    }

    fn run(&self, ctx: &ProjectContext, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        // Every top-level function / class / variable decl + its path.
        let idxs = ctx.indices_where(
            None,
            Some(vec![
                "function".to_string(),
                "class".to_string(),
                "variable".to_string(),
            ]),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        if idxs.is_empty() {
            return Ok(());
        }
        let paths = ctx.node_paths(idxs.clone())?;

        // Bucket conftest decls; collect test-file candidates (idx + path).
        let mut conftest_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut conftest_order: Vec<String> = Vec::new();
        let mut cand_idxs: Vec<usize> = Vec::new();
        let mut cand_paths: Vec<String> = Vec::new();
        for (idx, path) in idxs.iter().zip(paths.iter()) {
            let filename = path_basename(path);
            if filename == "conftest.py" {
                if !conftest_by_path.contains_key(path) {
                    conftest_order.push(path.clone());
                }
                conftest_by_path.entry(path.clone()).or_default().push(*idx);
            } else if is_test_filename(filename) {
                cand_idxs.push(*idx);
                cand_paths.push(path.clone());
            }
        }

        // Filter test candidates by `_is_test_decl`; track function/class kinds.
        let mut tests_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut tests_order: Vec<String> = Vec::new();
        let mut test_function_idxs: Vec<usize> = Vec::new();
        let mut test_class_idxs: Vec<usize> = Vec::new();
        if !cand_idxs.is_empty() {
            let attrs = ctx.node_attrs(cand_idxs.clone())?;
            for ((idx, path), attr) in cand_idxs.iter().zip(cand_paths.iter()).zip(attrs.iter()) {
                if !is_test_decl(&attr.kind, &attr.fqname) {
                    continue;
                }
                if !tests_by_path.contains_key(path) {
                    tests_order.push(path.clone());
                }
                tests_by_path.entry(path.clone()).or_default().push(*idx);
                if attr.kind == "function" {
                    test_function_idxs.push(*idx);
                } else if attr.kind == "class" {
                    test_class_idxs.push(*idx);
                }
            }
        }

        // Module fqnames for the paths we'll seed (conftest + filtered tests).
        let synthetic = intern_kind("synthetic").expect("'synthetic' is a valid kind");
        let mut seed_paths: Vec<String> = conftest_order.clone();
        for p in &tests_order {
            if !conftest_by_path.contains_key(p) {
                seed_paths.push(p.clone());
            }
        }
        let seed_module_fqnames = module_fqnames_for(ctx, &seed_paths)?;
        for path in &conftest_order {
            if let Some(module_fqname) = seed_module_fqnames.get(path) {
                mark_seed(
                    sink,
                    synthetic,
                    format!("{PYTEST_CONFTEST_PREFIX}{module_fqname}"),
                    path.clone(),
                    conftest_by_path[path].clone(),
                );
            }
        }
        for path in &tests_order {
            if let Some(module_fqname) = seed_module_fqnames.get(path) {
                mark_seed(
                    sink,
                    synthetic,
                    format!("{PYTEST_TESTS_PREFIX}{module_fqname}"),
                    path.clone(),
                    tests_by_path[path].clone(),
                );
            }
        }

        // `@pytest.fixture`-decorated decls (args extracted for the `name=`
        // alias). Needs the GIL for the parallel decorator scan.
        let fixture_refs: Vec<(usize, CallArgs)> = Python::with_gil(|py| {
            ctx.find_decorated_decls(
                py,
                &["pytest".to_string()],
                vec!["fixture".to_string()],
                None,
                true,
            )
        })?;
        if fixture_refs.is_empty() {
            return Ok(());
        }
        let fixture_idxs_all: Vec<usize> = fixture_refs.iter().map(|(idx, _)| *idx).collect();

        // Per-file fixture seed (every `@pytest.fixture` stays alive — the
        // conservative rule that catches autouse / usefixtures / indirect).
        let fixture_paths = ctx.node_paths(fixture_idxs_all.clone())?;
        let mut fixtures_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut fixtures_path_order: Vec<String> = Vec::new();
        for (idx, path) in fixture_idxs_all.iter().zip(fixture_paths.iter()) {
            if !fixtures_by_path.contains_key(path) {
                fixtures_path_order.push(path.clone());
            }
            fixtures_by_path.entry(path.clone()).or_default().push(*idx);
        }
        let fixture_module_fqnames = module_fqnames_for(ctx, &fixtures_path_order)?;
        for path in &fixtures_path_order {
            if let Some(module_fqname) = fixture_module_fqnames.get(path) {
                mark_seed(
                    sink,
                    synthetic,
                    format!("{PYTEST_FIXTURES_PREFIX}{module_fqname}"),
                    path.clone(),
                    fixtures_by_path[path].clone(),
                );
            }
        }

        // Parameter-name edges. Skip when no test signature to inspect.
        if test_function_idxs.is_empty() && test_class_idxs.is_empty() {
            return Ok(());
        }
        // binding name -> [fixture idx]. Binding is the `name=` kwarg literal
        // when present, else the function's simple name.
        let fixture_attrs = ctx.node_attrs(fixture_idxs_all.clone())?;
        let mut fixtures_by_name: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for ((idx, args), attr) in fixture_refs.iter().zip(fixture_attrs.iter()) {
            let binding = match args.kwargs.get("name") {
                Some(ArgValue::Str(alias)) => alias.clone(),
                _ => simple_name(&attr.fqname).to_string(),
            };
            fixtures_by_name.entry(binding).or_default().push(*idx);
        }

        Python::with_gil(|py| -> PyResult<()> {
            if !test_function_idxs.is_empty() {
                let params = ctx.function_parameters(py, test_function_idxs.clone())?;
                for (test_idx, names) in test_function_idxs.iter().zip(params) {
                    for name in names {
                        if let Some(fixture_idxs) = fixtures_by_name.get(&name) {
                            for &fixture_idx in fixture_idxs {
                                sink.push(PreparedOp::Edge {
                                    src_idx: *test_idx,
                                    dst_idx: fixture_idx,
                                    flags: 0,
                                });
                            }
                        }
                    }
                }
            }
            if !test_class_idxs.is_empty() {
                let params = ctx.class_method_parameters(py, test_class_idxs.clone())?;
                for (cls_idx, names) in test_class_idxs.iter().zip(params) {
                    for name in names {
                        if let Some(fixture_idxs) = fixtures_by_name.get(&name) {
                            for &fixture_idx in fixture_idxs {
                                sink.push(PreparedOp::Edge {
                                    src_idx: *cls_idx,
                                    dst_idx: fixture_idx,
                                    flags: 0,
                                });
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    }
}
