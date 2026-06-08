//! Native (rust-side) plugins — the only plugin mechanism.
//!
//! A native plugin is a rust implementation of the curated
//! [`plugin_api::ExternalPlugin`] contract: the harness detects a
//! :class:`NativePlugin` pyclass via downcast in
//! ``collect_prepared_plugin_ops``, invokes the inner
//! [`plugin_api::ExternalPlugin::run`] directly, and the impl pushes its ops
//! into the shared [`PreparedOp`] sink the harness drains at the end of the
//! plugin pass. There is no Python ``.run(ctx)`` call and no per-op
//! ``extract`` round-trip — the impl emits ops through the GIL-free
//! [`plugin_api::PluginOps`] surface, lowered to pure-rust [`PreparedOp`]
//! variants.
//!
//! **One surface.** In-tree built-ins and out-of-tree authors implement the
//! *same* curated [`plugin_api`] traits ([`plugin_api::ExternalPlugin`] /
//! [`plugin_api::PerFilePlugin`]) — there is no separate internal trait. An
//! out-of-tree author ships an external native plugin compiled against the
//! shipped runtime dylib and loaded via [`load_native_plugins`] (see
//! ``NATIVE_PLUGINS.md``); a built-in is the same trait object constructed
//! in-process. [`PreparedOp`] and [`ProjectContext`] stay `pub(crate)`, the
//! `dead-cst-native` crate is not published to crates.io, and the rust API
//! has no stability commitment.
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
use crate::flag_registry::FlagRegistry;
use crate::graph::EdgeFlags;
use crate::helpers::{
    collect_modules_imports_local, decorators_match_imports, file_path_string,
    matched_call_target_any, top_level_assign_to_name, ArgValue, CallArgs, FactoryCallFinder,
};
use crate::ingest::file_package_name;
use crate::project::FrozenView;
use crate::topic_registry::TopicRegistry;

/// Map a curated [`plugin_api::PluginError`] onto a Python exception. The
/// curated surface stays pyo3-free; this is the single crossing point where a
/// plugin failure becomes a `PyErr` (raised out of `materialize`).
impl From<plugin_api::PluginError> for PyErr {
    fn from(err: plugin_api::PluginError) -> Self {
        let msg = err.message().to_string();
        match err.kind() {
            plugin_api::PluginErrorKind::Value => pyo3::exceptions::PyValueError::new_err(msg),
            plugin_api::PluginErrorKind::Runtime => pyo3::exceptions::PyRuntimeError::new_err(msg),
        }
    }
}

// Gated to the libpython-linked build (CI's `--no-default-features`); under the
// default `extension-module` feature pyo3 doesn't link an interpreter, so the
// GIL calls below would not link.
#[cfg(all(test, not(feature = "extension-module")))]
mod plugin_error_boundary_tests {
    use super::*;

    // Regression guard for the swallow bug: a plugin failure must cross the
    // boundary as a *typed* `PyErr` (the conversion `PluginJob::run`'s `?`
    // relies on), not be silently dropped. Locks the kind→exception mapping.
    #[test]
    fn plugin_error_maps_to_typed_pyerr() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let runtime: PyErr = plugin_api::PluginError::runtime("boom").into();
            assert!(runtime.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(runtime.to_string().contains("boom"));

            let value: PyErr = plugin_api::PluginError::value("bad toml").into();
            assert!(value.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(value.to_string().contains("bad toml"));
        });
    }
}

/// One registered plugin's GIL-free unit of work for the
/// [`crate::project::ProjectContext::materialize`] fan-out. Extracted
/// under the GIL by [`extract_plugin_jobs`] (so the [`NativePlugin`]
/// pyclass borrow is dropped before [`pyo3::Python::allow_threads`]),
/// then run on a rayon worker that owns its own [`FrozenView`]. The
/// trait objects are held in [`Arc`]s (the traits are `Send + Sync`),
/// so a handle clones cheaply into each worker.
///
/// Per-file plugins (builtin + external-with-per-file dispatch) fold
/// their ops into the graph during the build itself, so they carry no
/// post-build work — [`PluginJob::Noop`] keeps the progress-slot
/// indices aligned with registration order.
pub(crate) enum PluginJob {
    External(Arc<dyn plugin_api::ExternalPlugin>),
    Noop,
}

impl PluginJob {
    /// Run this plugin against ``view`` and append its ops to ``sink``.
    /// Mirrors the per-kind dispatch the serial path used: a
    /// project-wide impl runs directly; an external dylib plugin runs
    /// once against a restricted [`plugin_api::PluginCtx`] and its ops
    /// are folded in; a per-file no-op contributes nothing.
    pub(crate) fn run(&self, view: &FrozenView<'_>, sink: &mut Vec<PreparedOp>) -> PyResult<()> {
        match self {
            PluginJob::External(plugin) => {
                let pctx = plugin_api::PluginCtx::new(view);
                let mut ops = plugin_api::PluginOps::new();
                // Propagate (no longer swallow) a plugin failure: `PluginError`
                // converts to `PyErr` at this boundary and aborts the apply.
                plugin.run(&pctx, &mut ops)?;
                sink.extend(ops.into_inner());
                Ok(())
            }
            PluginJob::Noop => Ok(()),
        }
    }
}

/// Output of [`extract_plugin_jobs`]: the registration-order plugin jobs and
/// their display names, plus the post-declaration node/edge flag registries
/// and the topic registry, all built in the one GIL-held pass.
pub(crate) type ExtractedPlugins = (
    Vec<PluginJob>,
    Vec<String>,
    FlagRegistry,
    FlagRegistry,
    TopicRegistry,
);

/// Pull every registered plugin into a `Send` [`PluginJob`] paired with
/// its display name, in registration order, under the GIL. Doing the
/// downcast + kind dispatch here (rather than inside the fan-out) keeps
/// the not-a-`NativePlugin` diagnostic — and the pyclass borrow — on
/// the calling thread, so the rayon workers touch no Python state. The
/// names match [`NativePlugin::name`] so the rust-side progress slabs
/// agree with the Python poller's plugin labels.
pub(crate) fn extract_plugin_jobs(
    py: Python<'_>,
    plugins: &[PyObject],
) -> PyResult<ExtractedPlugins> {
    let mut jobs = Vec::with_capacity(plugins.len());
    let mut names = Vec::with_capacity(plugins.len());
    // Seed both registries with the engine built-ins, then fold in each
    // plugin's declared flags in registration order (so bit allocation is a
    // pure function of plugin order). A conflicting / reserved / overflowing
    // declaration surfaces here, on the GIL thread, before the parallel run.
    let mut node_reg = FlagRegistry::with_node_builtins();
    let mut edge_reg = FlagRegistry::with_edge_builtins();
    // Topics carry no engine built-ins (they're a plugin-only channel); each
    // plugin's declared topics fold in in the same registration-order pass, so
    // handle allocation is likewise a pure function of plugin order.
    let mut topic_reg = TopicRegistry::new();
    for plugin in plugins {
        let bound = plugin.bind(py);
        let native = match bound.downcast::<NativePlugin>() {
            Ok(native) => native,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "expected a dead_cst._native.NativePlugin, got {:?}; the Python \
                     plugin protocol (plugin.run(ctx) yielding GraphOps) has been removed",
                    bound.get_type().name()?,
                )));
            }
        };
        let native_ref = native.borrow();
        // Any plugin carrying an `ExternalPlugin` (project-wide or
        // per-file-capable) may declare flags; pure per-file built-ins don't.
        if let NativePluginKind::External { plugin, .. } = &native_ref.kind {
            for spec in plugin.declare_node_flags() {
                node_reg.register_plugin(spec)?;
            }
            for spec in plugin.declare_edge_flags() {
                edge_reg.register_plugin(spec)?;
            }
            for spec in plugin.declare_topics() {
                topic_reg.register_plugin(spec)?;
            }
        }
        let (job, name) = match &native_ref.kind {
            // Every `ExternalPlugin` runs its project-wide `run` — including
            // one that also opts into per-file dispatch (`per_file_id: Some`),
            // which additionally emits facts during the build walk. The two
            // phases compose: the per-file pass publishes, `run` resolves. A
            // plugin with no project-wide work just inherits the no-op default
            // `run`, so this costs it nothing.
            NativePluginKind::External { plugin, name, .. } => {
                (PluginJob::External(Arc::clone(plugin)), name.clone())
            }
            NativePluginKind::PerFile(id) => (PluginJob::Noop, id.name()),
        };
        jobs.push(job);
        names.push(name);
    }
    Ok((jobs, names, node_reg, edge_reg, topic_reg))
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

/// File-local op produced by a [`plugin_api::PerFilePlugin`]. Every
/// endpoint is a *file-local* index (position in the file's own
/// ``FileNodes.refs`` array), never a global graph index — the harness
/// translates to global indices at apply time. Salsa-cached as the
/// per-file plugin's output, so it must be pure rust + ``salsa::Update``
/// (no ``File`` handle, no global idx — both would couple the cache to
/// project-wide assemble order). Mirrors the project-wide [`PreparedOp`]
/// variants, restricted to a single file's index space.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) enum FileLocalOp {
    /// Reachability edge between two nodes in this file (`PreparedOp::Edge`
    /// shape). Both endpoints are file-local indices.
    Edge {
        src_local_idx: u32,
        dst_local_idx: u32,
        flags: u8,
    },
    /// Keep a node in this file alive by flagging it an entrypoint seed
    /// (`PreparedOp::Entrypoint` shape). ``decl_local_idx`` is a file-local
    /// index.
    Entrypoint { decl_local_idx: u32 },
    /// OR `flags` onto an existing node in this file (`PreparedOp::FlagDecl`
    /// shape). ``decl_local_idx`` is a file-local index; unlike
    /// [`FileLocalOp::Entrypoint`] the bits are caller-supplied.
    FlagDecl { decl_local_idx: u32, flags: u32 },
    /// Publish a fact under a topic **name** (not a registry handle — a handle
    /// is allocated from the whole plugin set and would couple this file's
    /// salsa cache to that set). ``decl_local_idx`` optionally pins the fact to
    /// a file-local decl, translated to a global index at assemble time; a
    /// fact whose decl fails to translate is dropped. Unlike the other
    /// variants this does not mutate the graph — it rides the per-file salsa
    /// cache into [`crate::project::BuildOutputs`] for the topic's project-wide
    /// reader.
    Fact {
        topic: compact_str::CompactString,
        decl_local_idx: Option<u32>,
        value: compact_str::CompactString,
    },
}

/// Read-only, single-file view a [`plugin_api::PerFilePlugin`] is run
/// against (wrapped by [`plugin_api::PluginFileCtx`]).
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

    /// This file's nodes — index 0 is the module node, the
    /// rest are top-level decls. Indices line up with [`Self::refs`].
    pub(crate) fn nodes(&self) -> &'db [NodeData] {
        &file_to_nodes(self.db, self.file).nodes
    }

    /// The file's module fqname (``nodes()[0].fqname``).
    pub(crate) fn module_fqname(&self) -> &'db str {
        &file_to_nodes(self.db, self.file).nodes[0].fqname
    }

    /// This file's source path, re-derived from its [`File`]. Every node
    /// in the file shares it, so [`PluginFileCtx`] computes it once per
    /// call and lends it to [`file_node_view`].
    pub(crate) fn path(&self) -> String {
        file_path_string(self.db, self.file)
    }

    /// Local index of the module node (always 0 — kept as a
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
    ) -> crate::helpers::LocalImports {
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
                    compact_str::CompactString::from(tail)
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

    // --- fact-emission helpers (dual-mode per-file plugins) -----------------
    //
    // These read the same salsa-cached [`file_extraction`] rows the
    // project-wide finders join against `decl_by_name_range`, but key the
    // result on *file-local* indices (via each node's `name_range`, which is
    // exactly `range_key(def.name.range())` — the key those rows carry). So a
    // dual-mode plugin's per-file fact emission matches its old project-wide
    // walk decl-for-decl, while riding the per-file salsa cache.

    /// `name_range → file-local idx` for every non-module / non-import node.
    /// `decl_by_name_range`'s per-file restriction: the decl rows in
    /// [`file_extraction`] (`decorator_rows`, `function_params`, …) are keyed
    /// by the same `name_range`, so this is the local-space join table.
    fn name_range_to_local(&self) -> rustc_hash::FxHashMap<(u32, u32), u32> {
        self.nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| !matches!(n.kind, NodeKind::Module | NodeKind::Import))
            .map(|(i, n)| (n.name_range, i as u32))
            .collect()
    }

    /// Per-file twin of `find_decorated_decls(extract_args = true)`: file-local
    /// indices of decls decorated by one of `names` from one of `modules`,
    /// paired with the decorator's captured kwargs. Mirrors
    /// `find_decorated_decls_core`'s inner loop (first matching decorator wins).
    fn decorated_decls_with_args(&self, modules: &[&str], names: &[&str]) -> Vec<(u32, CallArgs)> {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let names_set: FxHashSet<&str> = names.iter().copied().collect();
        let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
        let imports = crate::file_extraction::imports_local_from_facts(
            &facts.import_facts,
            &modules_owned,
            &names_set,
        );
        if imports.is_empty() {
            return Vec::new();
        }
        let by_range = self.name_range_to_local();
        let mut out: Vec<(u32, CallArgs)> = Vec::new();
        for (rk, descriptors) in &facts.decorator_rows {
            let Some(&local) = by_range.get(rk) else {
                continue;
            };
            for desc in descriptors {
                let matched = match desc.attrs.as_slice() {
                    [] => imports
                        .get(&desc.root_name)
                        .is_some_and(|target| names_set.contains(target.as_str())),
                    [attr] => {
                        imports
                            .get(&desc.root_name)
                            .map(compact_str::CompactString::as_str)
                            == Some(crate::helpers::MODULE_ALIAS_MARKER)
                            && names_set.contains(attr.as_str())
                    }
                    _ => false,
                };
                if matched {
                    out.push((local, desc.kwargs.clone()));
                    break;
                }
            }
        }
        out
    }

    /// `file-local idx → parameter names` for this file's top-level functions,
    /// read from the salsa-cached `function_params` rows (the same data
    /// `ProjectContext::function_parameters` serves project-wide).
    fn function_params_by_local(
        &self,
    ) -> rustc_hash::FxHashMap<u32, Vec<compact_str::CompactString>> {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let by_range = self.name_range_to_local();
        facts
            .function_params
            .iter()
            .filter_map(|(rk, names)| by_range.get(rk).map(|&local| (local, names.clone())))
            .collect()
    }

    /// `file-local idx → method parameter names` for this file's top-level
    /// classes (every method's parameters, the union pytest matches against),
    /// read from the salsa-cached `class_method_params` rows.
    fn class_method_params_by_local(
        &self,
    ) -> rustc_hash::FxHashMap<u32, Vec<compact_str::CompactString>> {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let by_range = self.name_range_to_local();
        facts
            .class_method_params
            .iter()
            .filter_map(|(rk, names)| by_range.get(rk).map(|&local| (local, names.clone())))
            .collect()
    }

    /// File-local indices of top-level classes that define a method named
    /// `method_name`, from the salsa-cached `class_method_defs` rows — the
    /// per-file twin of `find_classes_defining_method_indices`.
    fn classes_defining_method(&self, method_name: &str) -> Vec<u32> {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let by_range = self.name_range_to_local();
        facts
            .class_method_defs
            .iter()
            .filter(|(_, methods)| methods.iter().any(|m| m == method_name))
            .filter_map(|(rk, _)| by_range.get(rk).copied())
            .collect()
    }

    /// Per-file twin of `find_handler_decorators`: `(owner name, file-local
    /// idx)` for every decl carrying an `@<owner>.<attr>(...)` decorator whose
    /// `attr` is one of `decorator_attrs` — one row per distinct owner in
    /// source order. Reads the salsa-cached `decorator_rows` directly.
    fn handler_decorators_local(&self, decorator_attrs: &[&str]) -> Vec<(String, u32)> {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let attrs: rustc_hash::FxHashSet<&str> = decorator_attrs.iter().copied().collect();
        let by_range = self.name_range_to_local();
        let mut out: Vec<(String, u32)> = Vec::new();
        for (rk, descriptors) in &facts.decorator_rows {
            let Some(&local) = by_range.get(rk) else {
                continue;
            };
            let mut seen_owners: rustc_hash::FxHashSet<&str> = rustc_hash::FxHashSet::default();
            for desc in descriptors {
                let [attr] = desc.attrs.as_slice() else {
                    continue;
                };
                if !attrs.contains(attr.as_str()) {
                    continue;
                }
                if !seen_owners.insert(desc.root_name.as_str()) {
                    continue;
                }
                out.push((desc.root_name.to_string(), local));
            }
        }
        out
    }

    /// Visit every call site in this file (decl-owned and module-scope),
    /// yielding the *file-local* owner index (module-scope → 0). The per-file
    /// twin of [`for_each_call_site`](crate::project) keyed on local indices.
    fn for_each_call_site_local(
        &self,
        mut f: impl FnMut(u32, &crate::file_extraction::CallSiteFact),
    ) {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let by_range = self.name_range_to_local();
        for (rk, sites) in &facts.call_sites_by_decl {
            // A call site whose enclosing decl isn't a node falls back to the
            // module node (idx 0) — same `.or(module_idx)` the global walk uses.
            let owner = by_range.get(rk).copied().unwrap_or(0);
            for site in sites {
                f(owner, site);
            }
        }
        for site in &facts.module_call_sites {
            f(0, site);
        }
    }

    /// Per-file twin of `find_calls_to_imported`: `(local owner, literal)` for
    /// calls to `name` imported from one of `modules`, taking the `arg_index`
    /// positional string literal.
    fn calls_to_imported_local(
        &self,
        modules: &[&str],
        name: &str,
        arg_index: usize,
    ) -> Vec<(u32, String)> {
        let facts = crate::file_extraction::file_extraction(self.db, self.file);
        let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
        let allowed: rustc_hash::FxHashSet<&str> = [name].into_iter().collect();
        let imports = crate::file_extraction::imports_local_from_facts(
            &facts.import_facts,
            &modules_owned,
            &allowed,
        );
        if imports.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<(u32, String)> = Vec::new();
        self.for_each_call_site_local(|owner, site| {
            let Some(chain) = &site.callee else {
                return;
            };
            if crate::file_extraction::match_callee_chain(
                &chain.root_name,
                &chain.attrs,
                &imports,
                &modules_owned,
                &allowed,
            )
            .is_none()
            {
                return;
            }
            if let Some(arg) = site.nth_positional_string(arg_index) {
                out.push((owner, arg.to_string()));
            }
        });
        out
    }

    /// Per-file twin of `find_calls_on_var`: `(local owner, literal)` for
    /// `<owner>.<attr>(...)` calls (bare-name receiver), taking the
    /// `arg_index` positional string literal, optionally gated on an exact
    /// positional-argument count.
    fn calls_on_var_local(
        &self,
        owner_name: &str,
        attr: &str,
        arg_index: usize,
        required_positional: Option<usize>,
    ) -> Vec<(u32, String)> {
        let mut out: Vec<(u32, String)> = Vec::new();
        self.for_each_call_site_local(|owner, site| {
            let Some(chain) = &site.callee else {
                return;
            };
            let [only] = chain.attrs.as_slice() else {
                return;
            };
            if chain.root_name.as_str() != owner_name || only.as_str() != attr {
                return;
            }
            if let Some(expected) = required_positional {
                if site.positional_len != expected {
                    return;
                }
            }
            if let Some(arg) = site.nth_positional_string(arg_index) {
                out.push((owner, arg.to_string()));
            }
        });
        out
    }
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
}

impl PerFilePluginKind {
    /// Human-readable name, matching the equivalent Python plugin.
    fn name(self) -> &'static str {
        match self {
            PerFilePluginKind::MainBlock => "MainBlockPlugin",
        }
    }

    /// The `'static` curated [`plugin_api::PerFilePlugin`] impl for this
    /// configless built-in kind. Unit structs promote to `'static`, so the
    /// dispatch in [`per_file_plugin_ops`] hands back a borrowed trait object.
    fn plugin(self) -> &'static dyn plugin_api::PerFilePlugin {
        match self {
            PerFilePluginKind::MainBlock => &MainBlockPluginImpl,
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
}

impl plugin_api::PerFilePlugin for ConfiguredPerFile {
    /// Run the configured impl against ``file``.
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        match self {
            ConfiguredPerFile::ServerConfig { filenames } => {
                server_config_run_on_file(filenames, file, ops)
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

/// Process-global registry interning an *ordered* per-file plugin id list to a
/// stable `set_id`. This is what lets [`per_file_plugin_ops`] be keyed
/// `(file, set_id)` — **one** salsa query per file that runs the whole set —
/// rather than `(file, plugin)`, one per plugin. Salsa's per-build memo work
/// is then O(files), not O(files × plugins): the per-plugin loop lives inside
/// the query body, costing nothing in salsa. Interning by the ordered list
/// keeps it sound across `Analysis`es with different plugin sets (a different
/// list → a different `set_id` → separate memo entries), and an identical set
/// reconstructed across a `re_materialize` reuses its `set_id` (and so its
/// cache entries). Append-only and immutable for a given `set_id`, so the
/// untracked lookup from inside the query is sound — same contract as
/// [`CONFIGURED_PER_FILE_PLUGINS`] / [`EXTERNAL_PER_FILE_PLUGINS`].
static PER_FILE_PLUGIN_SETS: std::sync::OnceLock<std::sync::RwLock<PluginSetRegistry>> =
    std::sync::OnceLock::new();

#[derive(Default)]
struct PluginSetRegistry {
    sets: Vec<Arc<[PerFilePluginId]>>,
    by_hash: FxHashMap<u64, u32>,
}

/// Register (intern) an ordered per-file plugin id list, returning its
/// process-stable `set_id`. An identical list returns the existing id.
pub(crate) fn register_per_file_set(ids: Vec<PerFilePluginId>) -> u32 {
    let reg =
        PER_FILE_PLUGIN_SETS.get_or_init(|| std::sync::RwLock::new(PluginSetRegistry::default()));
    let mut hasher = FxHasher::default();
    ids.hash(&mut hasher);
    let hash = hasher.finish();
    let mut guard = reg.write().expect("per-file plugin set registry poisoned");
    if let Some(&id) = guard.by_hash.get(&hash) {
        return id;
    }
    let set_id = guard.sets.len() as u32;
    guard.sets.push(ids.into());
    guard.by_hash.insert(hash, set_id);
    set_id
}

/// Look up an interned per-file plugin id list by `set_id`.
fn per_file_set(set_id: u32) -> Option<Arc<[PerFilePluginId]>> {
    PER_FILE_PLUGIN_SETS
        .get()?
        .read()
        .expect("per-file plugin set registry poisoned")
        .sets
        .get(set_id as usize)
        .map(Arc::clone)
}

/// Run one per-file plugin (configless builtin, configured builtin, or
/// external dylib) into `ops` — the single curated [`plugin_api::PerFilePlugin`]
/// dispatch, shared by the whole-set loop in [`per_file_plugin_ops`].
fn run_one_per_file(
    id: PerFilePluginId,
    pctx: &plugin_api::PluginFileCtx<'_>,
    ops: &mut plugin_api::FileOps,
) {
    match id {
        PerFilePluginId::Builtin(kind) => kind.plugin().run_on_file(pctx, ops),
        PerFilePluginId::Configured(plugin_id) => {
            // Resolve the configured plugin from the registry; a stale id
            // (shouldn't happen) yields no ops.
            if let Some(cfg) = configured_per_file_plugin(plugin_id) {
                let per_file: &dyn plugin_api::PerFilePlugin = cfg.as_ref();
                per_file.run_on_file(pctx, ops);
            }
        }
        PerFilePluginId::External(plugin_id) => {
            // Resolve the plugin and its per-file capability from the
            // registry; a stale id (shouldn't happen) yields no ops.
            if let Some(plugin) = external_per_file_plugin(plugin_id) {
                if let Some(per_file) = plugin.per_file() {
                    per_file.run_on_file(pctx, ops);
                }
            }
        }
    }
}

/// Salsa-tracked per-file plugin invocation. Keyed on ``(file, set_id)`` — the
/// whole registered per-file plugin *set* runs in one query, so re-runs are
/// O(files) not O(files × plugins). Re-runs only when the file's tracked
/// inputs (``file_to_nodes`` / ``parsed_module`` / ``line_index``) change.
/// Returns every plugin's file-local ops concatenated in registration order
/// (the order the harness translates + folds at apply time).
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn per_file_plugin_ops(db: &dyn ProjectDb, file: File, set_id: u32) -> Vec<FileLocalOp> {
    let file_ctx = FileContext::new(db, file);
    let pctx = plugin_api::PluginFileCtx::new(&file_ctx);
    let mut ops = plugin_api::FileOps::new();
    // Every per-file plugin in the set — configless builtin, configured
    // builtin, or external dylib — runs through the one curated
    // `PerFilePlugin` surface, in registration order.
    if let Some(ids) = per_file_set(set_id) {
        for &id in ids.iter() {
            run_one_per_file(id, &pctx, &mut ops);
        }
    }
    ops.into_inner()
}

/// Internal classification of what a [`NativePlugin`] wraps. The
/// harness branches on this: project-wide plugins run once against the
/// whole ``ProjectContext``; per-file plugins run once per file
/// through the salsa-cached [`per_file_plugin_ops`] query.
pub(crate) enum NativePluginKind {
    /// Built-in *per-file* plugin selected by a configless/configured id;
    /// dispatched through the salsa-cached [`per_file_plugin_ops`] query.
    PerFile(PerFilePluginId),
    /// A plugin backed by the curated [`plugin_api::ExternalPlugin`] trait —
    /// covers *both* every in-tree project-wide built-in and every dylib-loaded
    /// external plugin, so the two surfaces share one dispatch path. A
    /// project-wide `run` fans out against a frozen, ``Send`` view of the whole
    /// graph; the [`Arc`] lets the materialize fan-out clone a handle into each
    /// GIL-free rayon worker (the trait is ``Send + Sync``).
    ///
    /// `per_file_id` is `Some(id)` when the plugin opted into per-file
    /// dispatch (`ExternalPlugin::per_file() -> Some`) — the host then routes
    /// it through the salsa-cached [`per_file_plugin_ops`] query keyed on
    /// [`PerFilePluginId::External`] instead of the project-wide `run`.
    ///
    /// `_lib` holds a refcount on the loaded library so its code stays mapped
    /// for the plugin's lifetime (only meaningful in a `-C prefer-dynamic`
    /// build where the extension and the plugin share one `dead-cst-runtime`).
    /// It is `None` for in-tree built-ins — statically linked, nothing to keep
    /// mapped.
    External {
        name: String,
        plugin: Arc<dyn plugin_api::ExternalPlugin>,
        per_file_id: Option<u32>,
        _lib: Option<Arc<libloading::Library>>,
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

    /// Native `InitSubclassPlugin` — keep transitive subclasses of
    /// `__init_subclass__`-defining classes alive via a marker node.
    #[staticmethod]
    fn init_subclass() -> Self {
        Self::dual_mode(Arc::new(InitSubclassPluginImpl))
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
        Self::dual_mode(Arc::new(UnittestPluginImpl))
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
    /// progress logs. `app_classes` are dotted fqnames of the
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
        app_classes: Vec<String>,
        registration_decorators: Vec<String>,
        seed_as_entrypoint: bool,
    ) -> Self {
        Self::from_dispatch_config(
            name,
            DispatchAppConfig {
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
        Self::dual_mode(Arc::new(ClickPluginImpl))
    }

    /// Native `MockPatchPlugin` (port of `dead_cst.contrib.mock_patch`).
    /// Resolve string-fqname `patch(...)` / `mocker.patch(...)` /
    /// `monkeypatch.setattr(...)` / `monkeypatch.delattr(...)` targets to their
    /// decls and keep them alive. Project-wide (a patch string in a test file
    /// targets a decl in another).
    #[staticmethod]
    fn mock_patch() -> Self {
        Self::dual_mode(Arc::new(MockPatchPluginImpl))
    }

    /// Native `DiscordPyPlugin` (port of `dead_cst.contrib.discordpy`). Wire
    /// discord.py bot/client handlers, Cogs, and extension hooks into
    /// reachability. Project-wide (a `@bot.command` handler and its `bot = Bot()`
    /// may live in different files; Cog subclasses span files).
    #[staticmethod]
    fn discordpy() -> Self {
        Self::project_wide(Arc::new(DiscordPyPluginImpl))
    }

    /// Native `PytestPlugin` (port of `dead_cst.contrib.pytest`). Seed
    /// pytest-discovered tests / conftest decls / `@pytest.fixture` functions
    /// and wire test→fixture edges by parameter-name matching. Project-wide
    /// (the subclass / fixture walk spans files).
    #[staticmethod]
    fn pytest() -> Self {
        Self::dual_mode(Arc::new(PytestPluginImpl))
    }

    /// Native `ProjectScriptsPlugin` (port of
    /// `dead_cst.plugins.project_scripts`). Mark every `[project.scripts]` entry
    /// in `pyproject.toml` as an entrypoint. `pyproject_path` defaults to
    /// `<project_root>/pyproject.toml`. Project-wide (a script target resolves
    /// against the whole graph).
    #[staticmethod]
    #[pyo3(signature = (pyproject_path = None))]
    fn project_scripts(pyproject_path: Option<String>) -> Self {
        Self::project_wide(Arc::new(ProjectScriptsPluginImpl { pyproject_path }))
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
        Self::project_wide(Arc::new(DynamicImportFallbackPluginImpl {
            include_underscore,
            respect_dunder_all,
            exclude_sources: exclude_sources.unwrap_or_default(),
            exclude_targets: exclude_targets.unwrap_or_default(),
            include_sources: include_sources.unwrap_or_default(),
            include_targets: include_targets.unwrap_or_default(),
        }))
    }

    /// Native `ExplicitEntrypointPlugin` (port of
    /// `dead_cst.plugins.explicit_entrypoint`). Mark user-specified symbols
    /// (file paths, FQNs, regexes) as entrypoints. Specs arrive pre-bucketed:
    /// `regexes` (matched against the project-relative path), `str_specs`
    /// (exact fqname or project-relative path), `abs_paths` (exact absolute
    /// path). Not in `_builtin_native_plugin` — it needs caller-supplied specs.
    #[staticmethod]
    fn explicit(regexes: Vec<String>, str_specs: Vec<String>, abs_paths: Vec<String>) -> Self {
        Self::project_wide(Arc::new(ExplicitEntrypointPluginImpl {
            regexes,
            str_specs,
            abs_paths,
        }))
    }
}

impl NativePlugin {
    /// Wrap a curated project-wide [`plugin_api::ExternalPlugin`] built-in: no
    /// per-file dispatch, no dylib to keep mapped. The in-tree twin of the
    /// dylib-loaded `External` variant the ABI airlock builds — both share the
    /// one project-wide dispatch path now.
    fn project_wide(plugin: Arc<dyn plugin_api::ExternalPlugin>) -> Self {
        Self {
            kind: NativePluginKind::External {
                name: plugin.name().to_string(),
                plugin,
                per_file_id: None,
                _lib: None,
            },
        }
    }

    /// Wrap a curated **dual-mode** built-in: it emits facts per-file
    /// (salsa-cached via [`per_file_plugin_ops`], so a clean file's facts are
    /// reused with zero re-run) and resolves them in its project-wide `run`.
    /// Registers the plugin for per-file dispatch and keeps it `External`, so
    /// the host runs *both* phases (every `ExternalPlugin` runs its `run`; a
    /// `Some` `per_file()` adds the build-walk emit). The plugin must return
    /// `Some` from `per_file()`.
    fn dual_mode(plugin: Arc<dyn plugin_api::ExternalPlugin>) -> Self {
        debug_assert!(
            plugin.per_file().is_some(),
            "dual_mode plugin must implement per_file()"
        );
        let per_file_id = Some(register_external_per_file(Arc::clone(&plugin)));
        Self {
            kind: NativePluginKind::External {
                name: plugin.name().to_string(),
                plugin,
                per_file_id,
                _lib: None,
            },
        }
    }

    /// Wrap a [`DispatchAppConfig`] in a project-wide native plugin. Shared by
    /// the per-framework factories and the custom `dispatch_app` factory.
    fn from_dispatch_config(name: String, config: DispatchAppConfig) -> Self {
        Self::project_wide(Arc::new(DispatchAppPluginImpl { name, config }))
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
// MainBlockPlugin — first per-file impl. For every file with a top-level
// ``if __name__ == "__main__":`` block, keep the module node alive plus
// every top-level decl whose source span falls inside the block (via
// ``keep_alive``, which stamps ENTRYPOINT directly on each target).
// ---------------------------------------------------------------------------

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

impl plugin_api::PerFilePlugin for MainBlockPluginImpl {
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        MAIN_BLOCK_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
        let Some(block_range) = file.main_block_range() else {
            return;
        };
        let (block_start_line, block_end_line) = file.line_span(block_range);
        // Keep the module node alive, plus every top-level decl whose source
        // span falls inside the ``if __name__`` block. Skip index 0 (the
        // module node) — it's kept alive explicitly first.
        ops.keep_alive(file.module_local_idx());
        for node in file.nodes().iter().skip(1) {
            if node.start_line >= block_start_line && node.end_line <= block_end_line {
                ops.keep_alive(node.local_idx);
            }
        }
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
    use super::{FileContext, FileLocalOp};
    use crate::builder::PreparedOp;
    use crate::graph::{EdgeFlags, NodeFlags};
    use crate::project::FrozenView;

    // Re-exported so a per-file plugin can name the raw AST / range types
    // [`PluginFileCtx::parsed`] and [`PluginFileCtx::line_span`] traffic in
    // without depending on the exact ruff crate paths.
    pub use ruff_db::parsed::ParsedModuleRef;
    pub use ruff_text_size::TextRange;

    // Decorator / constructor argument captures (plain Rust, pyo3-free), so a
    // plugin reading the args returned by [`PluginCtx::decorated_decls_with_args`]
    // need not reach into crate internals. `kwargs` is crate-private; read
    // values via [`CallArgs::get`] / [`CallArgs::str_value`].
    pub use crate::helpers::{ArgValue, CallArgs};

    // Flag declaration vocabulary. A plugin returns these from
    // [`ExternalPlugin::declare_node_flags`] / [`declare_edge_flags`]; the host
    // allocates each a bit and the plugin reads it back via
    // [`PluginCtx::node_flag`] / [`edge_flag`]. Pyo3-free.
    pub use crate::flag_registry::FlagSpec;

    // Topic declaration vocabulary. A plugin returns these from
    // [`ExternalPlugin::declare_topics`]; the host assigns each a handle and
    // the plugin resolves a name to its handle via [`PluginCtx::topic`], then
    // reads the collected [`Fact`]s with [`PluginCtx::facts_for_topic`].
    // Pyo3-free.
    pub use crate::topic_registry::TopicSpec;

    /// `NodeFlags::ENTRYPOINT` re-exported for plugin authors. The flag
    /// [`PluginOps::keep_alive`] / [`FileOps::keep_alive`] stamp on a decl to
    /// make it a reachability seed; also usable as a `flag_decl` bit.
    /// Single source of truth — tracks the internal bit.
    pub const FLAG_ENTRYPOINT: u32 = NodeFlags::ENTRYPOINT;

    /// `EdgeFlags::DEAD_BRANCH` re-exported for the `flags` argument of
    /// [`PluginOps::add_edge`] / [`FileOps::add_edge`]: mark a plugin-added
    /// edge as originating in a statically-dead region. Metadata only — the
    /// edge still participates in default reachability.
    pub const FLAG_DEAD_BRANCH: u8 = EdgeFlags::DEAD_BRANCH;

    /// `EdgeFlags::DYNAMIC_IMPORT` re-exported for the `flags` argument of
    /// [`PluginOps::add_edge`] / [`FileOps::add_edge`]: mark a plugin-added
    /// edge as a runtime-import (`__import__` / `importlib.import_module`)
    /// fan-out, the same bit the visitor stamps on dynamic-import edges.
    pub const FLAG_DYNAMIC_IMPORT: u8 = EdgeFlags::DYNAMIC_IMPORT;

    /// `EdgeFlags::INIT_SUBCLASS` re-exported for the `flags` argument of
    /// [`PluginOps::add_edge`] / [`FileOps::add_edge`]: mark a plugin-added
    /// edge as a base-class → `__init_subclass__`-discovered subclass link,
    /// the same bit the built-in init-subclass plugin stamps.
    pub const FLAG_INIT_SUBCLASS: u8 = EdgeFlags::INIT_SUBCLASS;

    /// Epoch of this curated `plugin_api` surface — the author-facing contract
    /// version, baked into the [ABI fingerprint](super::PLUGIN_ABI_FINGERPRINT)
    /// (the `api<N>` segment). It is **bumped in `runtime/build.rs`** whenever
    /// this surface changes incompatibly, so a plugin compiled against an older
    /// `plugin_api` is rejected at load even if nothing else (rustc, version,
    /// target) changed. Exposed for plugin authors and diagnostics; the
    /// load-time gate compares the whole fingerprint, not this value alone.
    pub const PLUGIN_API_EPOCH: &str = env!("PLUGIN_API_EPOCH");

    /// Error a plugin returns from [`ExternalPlugin::run`]. **Pyo3-free by
    /// design** — the curated surface never names a `PyErr`, so a plugin author
    /// depends only on this crate's API, not on pyo3. The host maps it to a
    /// Python exception once, at the `materialize` boundary, by [`kind`]
    /// ([`Value`](PluginErrorKind::Value) → `ValueError`,
    /// [`Runtime`](PluginErrorKind::Runtime) → `RuntimeError`).
    #[derive(Debug, Clone)]
    pub struct PluginError {
        message: String,
        kind: PluginErrorKind,
    }

    /// Coarse classification carried by a [`PluginError`]; selects which Python
    /// exception type the host raises. A plain enum (no pyo3) so the airlock
    /// stays dependency-encapsulated.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PluginErrorKind {
        /// Bad plugin input or configuration; surfaces as a `ValueError`.
        Value,
        /// Any other failure; surfaces as a `RuntimeError`.
        Runtime,
    }

    impl PluginError {
        /// A [`Value`](PluginErrorKind::Value)-kind error (bad config/input).
        pub fn value(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                kind: PluginErrorKind::Value,
            }
        }

        /// A [`Runtime`](PluginErrorKind::Runtime)-kind error.
        pub fn runtime(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                kind: PluginErrorKind::Runtime,
            }
        }

        /// The human-readable message.
        pub fn message(&self) -> &str {
            &self.message
        }

        /// The error's classification (drives the host's exception mapping).
        pub fn kind(&self) -> PluginErrorKind {
            self.kind
        }
    }

    impl std::fmt::Display for PluginError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.message)
        }
    }

    impl std::error::Error for PluginError {}

    impl From<String> for PluginError {
        fn from(message: String) -> Self {
            Self::runtime(message)
        }
    }

    impl From<&str> for PluginError {
        fn from(message: &str) -> Self {
            Self::runtime(message)
        }
    }

    /// Contract an external native plugin implements. The host calls
    /// [`run`](ExternalPlugin::run) once per materialize against the frozen
    /// graph, then folds the emitted ops in as a single batch.
    ///
    /// A plugin can additionally opt into **per-file** dispatch by also
    /// implementing [`PerFilePlugin`] and returning `Some(self)` from
    /// [`per_file`](ExternalPlugin::per_file): the host then invokes
    /// [`PerFilePlugin::run_on_file`] once per project file through a
    /// salsa-cached query — so an unchanged file's ops are reused across a
    /// `re_materialize` with zero re-run (same fast-path the in-tree
    /// `MainBlockPlugin` rides) — **and still calls [`run`](ExternalPlugin::run)**
    /// for the project-wide pass. The two compose: the per-file pass typically
    /// emits facts ([`FileOps::emit_fact`]) and `run` resolves them
    /// ([`PluginCtx::facts_for_topic`]). A plugin with no project-wide work
    /// simply leaves `run` as the no-op default, which costs nothing.
    pub trait ExternalPlugin: Send + Sync {
        /// Human-readable name (surfaced in progress logs).
        fn name(&self) -> &str;

        /// Inspect the whole frozen graph and append ops to `ops`. Same
        /// frozen-graph contract as every built-in. Return
        /// `Err(`[`PluginError`]`)` to fail the whole materialize with that
        /// message; the host raises the mapped Python exception. Default no-op
        /// (`Ok(())`), so a pure per-file plugin needn't implement it — the host
        /// still calls it, but the default does nothing.
        fn run(&self, _ctx: &PluginCtx<'_>, _ops: &mut PluginOps) -> Result<(), PluginError> {
            Ok(())
        }

        /// Opt into per-file (salsa-cached) dispatch by returning
        /// `Some(self)`, *in addition to* the project-wide [`run`](Self::run)
        /// the host always calls. The default `None` keeps the plugin
        /// project-wide only.
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

        /// Node flags this plugin contributes to the registry. Each returned
        /// [`FlagSpec`] is assigned a bit by the host (above the engine flags,
        /// in registration order) before [`run`](Self::run); read the bit back
        /// with [`PluginCtx::node_flag`]. Declaring the same `owner/name` spec
        /// from two plugins is idempotent (they share one bit); a conflicting
        /// re-declaration, an `engine/`-prefixed name, or exhausting the 32-bit
        /// node width fails the whole materialize. Default none.
        fn declare_node_flags(&self) -> Vec<FlagSpec> {
            Vec::new()
        }

        /// Edge flags this plugin contributes to the registry — the edge-space
        /// twin of [`declare_node_flags`](Self::declare_node_flags), read back
        /// with [`PluginCtx::edge_flag`]. Edge flags share an 8-bit width.
        /// Default none.
        fn declare_edge_flags(&self) -> Vec<FlagSpec> {
            Vec::new()
        }

        /// Topics this plugin publishes facts under. Each returned
        /// [`TopicSpec`] is assigned a handle by the host (in registration
        /// order) before any run; a per-file run emits facts under the topic
        /// **name** via [`FileOps::emit_fact`], and the project-wide
        /// [`run`](Self::run) resolves a name to its handle with
        /// [`PluginCtx::topic`] and reads the collected facts via
        /// [`PluginCtx::facts_for_topic`]. Declaring the same `owner/name` spec
        /// from two plugins is idempotent (they share one handle); a
        /// conflicting re-declaration or an `engine/`-prefixed name fails the
        /// whole materialize. Default none.
        fn declare_topics(&self) -> Vec<TopicSpec> {
            Vec::new()
        }
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

    /// One collected fact, returned by [`PluginCtx::facts_for_topic`] (and the
    /// Python `ProjectContext.facts_for_topic`). `path` is the file the fact
    /// was published from; `decl_idx`, when present, is the **global** node
    /// index the publishing per-file plugin pinned the fact to (its file-local
    /// index translated at assemble time — a fact whose decl failed to
    /// translate is dropped, so a `Some` here always names a live node);
    /// `value` is the plugin-defined payload.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Fact {
        pub path: String,
        pub decl_idx: Option<usize>,
        pub value: String,
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
        /// `module`, `external`.
        pub kind: String,
        /// Source path the node was declared in (empty for some external nodes).
        pub path: String,
        /// 1-based start line, or 0 for nodes with no source range
        /// (`module` / `external`).
        pub start_line: usize,
        /// 1-based end line, or 0 for nodes with no source range
        /// (`module` / `external`).
        pub end_line: usize,
        /// `NodeFlags` bitset (see [`FLAG_ENTRYPOINT`]).
        pub flags: u32,
    }

    /// Borrowing view of one graph node, yielded by [`PluginCtx::nodes`].
    /// Unlike the owned [`NodeView`], the string fields borrow the frozen
    /// graph — so a full-graph scan filtering on `kind` / `path` costs no
    /// allocation until the plugin keeps a match. Use [`PluginCtx::node`]
    /// for the owned single-node read.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NodeRef<'a> {
        /// Positional index of this node in the frozen graph.
        pub idx: usize,
        /// Fully-qualified name.
        pub fqname: &'a str,
        /// One of `function`, `class`, `variable`, `import`, `type_alias`,
        /// `module`, `external`.
        pub kind: &'a str,
        /// Source path (empty for some external nodes).
        pub path: &'a str,
        /// 1-based start line, or 0 for nodes with no source range
        /// (`module` / `external`).
        pub start_line: usize,
        /// 1-based end line, or 0 for nodes with no source range
        /// (`module` / `external`).
        pub end_line: usize,
        /// `NodeFlags` bitset (see [`FLAG_ENTRYPOINT`]).
        pub flags: u32,
    }

    /// One directed graph edge, yielded by [`PluginCtx::edges`]: `src -> dst`
    /// (both positional node indices) stamped with an `EdgeFlags` bitset (see
    /// [`FLAG_DEAD_BRANCH`] / [`FLAG_DYNAMIC_IMPORT`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EdgeRef {
        /// Source node index.
        pub src: usize,
        /// Destination node index.
        pub dst: usize,
        /// `EdgeFlags` bitset.
        pub flags: u8,
    }

    /// Conjunctive filter for [`PluginCtx::nodes_matching`] — a node must
    /// satisfy every set criterion. All fields default to "no constraint"
    /// (`Default`), so a plugin sets only what it needs:
    /// `NodeFilter { kinds: &["function"], simple_names: &["setup", "teardown"], paths: &cog_paths, ..Default::default() }`.
    /// Empty slices mean "unconstrained" (not "matches nothing").
    #[derive(Debug, Clone, Default)]
    pub struct NodeFilter<'a> {
        /// Match only these node kinds (`function` / `class` / …). Empty = any.
        pub kinds: &'a [&'a str],
        /// Match only nodes whose trailing fqname segment is one of these.
        pub simple_names: &'a [&'a str],
        /// Match only nodes declared in one of these exact source paths.
        pub paths: &'a [&'a str],
        /// Match only nodes whose fqname starts with this dotted prefix.
        pub fqname_prefix: Option<&'a str>,
        /// Match only nodes whose flags contain *all* of these bits.
        pub flags_all: Option<u32>,
        /// Match only nodes whose flags contain *any* of these bits.
        pub flags_any: Option<u32>,
    }

    /// Restricted, mostly-stable view of the frozen graph for external
    /// plugins. Wraps the internal `ProjectContext`; exposes only the
    /// index-based queries a plugin needs.
    pub struct PluginCtx<'a> {
        inner: &'a FrozenView<'a>,
    }

    impl<'a> PluginCtx<'a> {
        pub(crate) fn new(inner: &'a FrozenView<'a>) -> Self {
            Self { inner }
        }

        // --- node reads -----------------------------------------------------

        /// Total number of nodes in the frozen graph. Indices in
        /// `0..node_count()` are valid arguments to [`Self::node`].
        pub fn node_count(&self) -> usize {
            self.inner.node_count()
        }

        /// Resolve an index to an owned [`NodeView`], or `None` if out of
        /// range. Reads frozen pyclass data directly — no GIL, no clone of
        /// the underlying `SymbolNode`.
        pub fn node(&self, idx: usize) -> Option<NodeView> {
            let node = self.inner.outputs.builder.nodes.get(idx)?;
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
            self.inner.find_module_idx(fqname)
        }

        /// Every declaration node with this exact fqname (more than one
        /// when a name is shadowed; see the graph-model invariants).
        pub fn find_declarations(&self, fqname: &str) -> Vec<usize> {
            self.inner.find_declarations_indices(fqname)
        }

        /// O(1) module-node lookup by source path.
        pub fn module_for(&self, path: &str) -> Option<usize> {
            self.inner.module_for_idx(path)
        }

        /// Resolve a dotted fqname to a decl or module node, walking back
        /// through dotted segments for method/attribute fqnames. `None`
        /// when nothing matches.
        pub fn resolve(&self, fqname: &str) -> Option<usize> {
            self.inner.resolve_idx(fqname)
        }

        /// Every node whose source path starts with `path_prefix` — a
        /// cheap way to scope a plugin to one package/directory.
        pub fn decls_under(&self, path_prefix: &str) -> Vec<usize> {
            self.inner.decls_under_indices(path_prefix)
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
            self.inner.find_subclasses_indices(base_fqn, transitive)
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
            self.inner.find_main_blocks_indices()
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
            self.inner.module_surface_indices(module_fqn)
        }

        /// Bulk twin of [`Self::module_surface`]: the public surface of each
        /// `module_fqn`, keyed by fqname, resolved in a single sweep. The shape
        /// discord.py's `load_extension` fan-out reads. A module that isn't
        /// found maps to an empty vec.
        pub fn module_surfaces(
            &self,
            module_fqns: &[String],
        ) -> std::collections::HashMap<String, Vec<usize>> {
            self.inner
                .module_surfaces_indices(module_fqns)
                .into_iter()
                .collect()
        }

        /// The decl nodes named by `module_fqn`'s `__all__`, or `None` when
        /// the module has no `__all__` / it isn't a literal string list.
        pub fn dunder_all_exports(&self, module_fqn: &str) -> Option<Vec<usize>> {
            self.inner
                .find_module_dunder_all_exports_indices(module_fqn)
        }

        /// The string entries of a top-level `X = ["a", "b"]` /
        /// `X: tuple[str, ...] = ("a", "b")` literal-list assignment named
        /// `var_fqn`, or `None` when the variable isn't a literal string
        /// list. The read the `LiteralListPlugin` shape is built on.
        pub fn literal_list_entries(&self, var_fqn: &str) -> Option<Vec<String>> {
            self.inner.find_literal_list_entries(var_fqn)
        }

        /// Every top-level decl (function / class / variable / import /
        /// type_alias) whose simple name matches the regular expression
        /// `pattern`. An invalid pattern yields an empty result.
        pub fn decls_matching_name(&self, pattern: &str) -> Vec<usize> {
            self.inner.decls_matching_name_indices(pattern)
        }

        /// Project-wide twin of [`PluginFileCtx::decorated_decls`]:
        /// function / class decls anywhere in the project carrying a
        /// decorator that resolves to one of `names` imported from one of
        /// `modules`.
        pub fn decorated_decls(&self, modules: &[&str], names: &[&str]) -> Vec<usize> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            let names_owned: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            self.inner
                .find_decorated_decls(&modules_owned, &names_owned, false)
                .into_iter()
                .map(|(idx, _)| idx)
                .collect()
        }

        /// Project-wide twin of [`PluginFileCtx::constructions`]: top-level
        /// `X = Ctor(...)` decls anywhere in the project whose `Ctor`
        /// resolves to one of `names` imported from one of `modules`.
        pub fn constructions(&self, modules: &[&str], names: &[&str]) -> Vec<usize> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            let names_owned: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            self.inner
                .find_instance_constructions(&modules_owned, &names_owned, false)
                .into_iter()
                .map(|(idx, _, _)| idx)
                .collect()
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
            self.inner
                .find_handler_decorators(&attrs, false)
                .into_iter()
                .map(|(owner, idx, _args)| (owner, idx))
                .collect()
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
            self.inner
                .find_calls_to_imported(&modules_owned, name, arg_index, false)
                .into_iter()
                .map(|(idx, literal, _args)| (idx, literal))
                .collect()
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
            self.inner
                .find_calls_on_attr(attr, arg_index, false)
                .into_iter()
                .map(|(idx, literal, _args)| (idx, literal))
                .collect()
        }

        /// Var-method twin of [`Self::calls_with_string_arg`]: calls
        /// `<owner>.<attr>(...)` whose argument at `arg_index` is a string
        /// literal, where `owner` is a bare variable name (`mocker`,
        /// `monkeypatch`). `required_positional`, when set, restricts to
        /// calls with exactly that many positional args — the
        /// `monkeypatch.setattr("x.y", v)` (2) vs `setattr(obj, "n", v)` (3)
        /// disambiguation. Returns `(owning_decl_idx, literal)`.
        pub fn calls_on_var(
            &self,
            owner: &str,
            attr: &str,
            arg_index: usize,
            required_positional: Option<usize>,
        ) -> Vec<(usize, String)> {
            self.inner
                .find_calls_on_var(owner, attr, arg_index, required_positional, false)
                .into_iter()
                .map(|(idx, literal, _args)| (idx, literal))
                .collect()
        }

        /// Two-level twin of [`Self::handler_decorators`]: top-level functions
        /// decorated `@<owner>.<via_attr>.<attr>(...)` where `attr` is one of
        /// `decorator_attrs` (e.g. `via_attr="tree"` for discord.py
        /// `@bot.tree.command`). Returns `(owner_name, decl_idx)` — the raw
        /// textual root owner and the decorated decl's index.
        pub fn handler_decorators_via(
            &self,
            via_attr: &str,
            decorator_attrs: &[&str],
        ) -> Vec<(String, usize)> {
            let attrs: Vec<String> = decorator_attrs.iter().map(|s| s.to_string()).collect();
            self.inner
                .find_handler_decorators_via(via_attr, &attrs, false)
                .into_iter()
                .map(|(owner, idx, _args)| (owner, idx))
                .collect()
        }

        /// Args-capturing twin of [`Self::decorated_decls`]: each match paired
        /// with its decorator's captured keyword arguments
        /// ([`CallArgs`]). Read values via [`CallArgs::str_value`] — e.g.
        /// the `name=` alias on `@pytest.fixture(name="…")`.
        pub fn decorated_decls_with_args(
            &self,
            modules: &[&str],
            names: &[&str],
        ) -> Vec<(usize, CallArgs)> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            let names_owned: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            self.inner
                .find_decorated_decls(&modules_owned, &names_owned, true)
        }

        /// Functions / classes that *return* an instance of one of `names`
        /// imported from one of `modules` (app-factory shape). Returns
        /// `(decl_idx, return_kinds)` where `return_kinds` lists the node
        /// kinds the factory yields.
        pub fn factory_decls(
            &self,
            modules: &[&str],
            ctor_names: &[&str],
        ) -> Vec<(usize, Vec<String>)> {
            let modules_owned: Vec<String> = modules.iter().map(|m| m.to_string()).collect();
            let names_owned: Vec<String> = ctor_names.iter().map(|n| n.to_string()).collect();
            self.inner.find_factory_decls(&modules_owned, &names_owned)
        }

        /// Class nodes that define a method named `method_name` directly in
        /// their body (e.g. `"__init_subclass__"`).
        pub fn classes_defining_method(&self, method_name: &str) -> Vec<usize> {
            self.inner.find_classes_defining_method_indices(method_name)
        }

        // --- import presence ------------------------------------------------

        /// True if any file imports `module` (or a submodule of it). A cheap
        /// O(1) presence probe to short-circuit before heavier walks.
        pub fn has_imports_of(&self, module: &str) -> bool {
            self.inner.has_imports_of(module)
        }

        /// The `kind="import"` node indices that import `module` (or a
        /// submodule). Pair with [`Self::node_paths`] to learn which files
        /// import it.
        pub fn imports_of(&self, module: &str) -> Vec<usize> {
            self.inner.find_imports_of_indices(module)
        }

        // --- bulk node reads ------------------------------------------------

        /// Owned [`NodeView`] for each index, skipping any out of range. The
        /// bulk twin of [`Self::node`].
        pub fn nodes_at(&self, idxs: &[usize]) -> Vec<NodeView> {
            idxs.iter().filter_map(|&i| self.node(i)).collect()
        }

        /// Source path for each index, skipping any out of range. Convenience
        /// over [`Self::nodes_at`] when only the path is needed.
        pub fn node_paths(&self, idxs: &[usize]) -> Vec<String> {
            idxs.iter()
                .filter_map(|&i| self.node(i).map(|v| v.path))
                .collect()
        }

        /// The module node index for each source `path`, `None` where the
        /// path isn't a known project module. Order matches `paths`.
        pub fn modules_for_paths(&self, paths: &[String]) -> Vec<Option<usize>> {
            self.inner.modules_for_paths(paths)
        }

        /// Every top-level decl (function / class / variable / import /
        /// type_alias) of `module_fqn`, ignoring `__all__`. Empty when the
        /// module isn't found.
        pub fn module_top_level_decls(&self, module_fqn: &str) -> Vec<usize> {
            self.inner.find_module_top_level_decls_indices(module_fqn)
        }

        /// Parameter names of each function index (positional + keyword, in
        /// source order). Order matches `idxs`; a non-function index yields an
        /// empty list.
        pub fn function_parameters(&self, idxs: &[usize]) -> Vec<Vec<String>> {
            self.inner.function_parameters(idxs).unwrap_or_default()
        }

        /// Union of every method's parameter names per class index (excluding
        /// `self` / `cls`). Order matches `idxs`; a non-class index yields an
        /// empty list. Powers pytest's fixture-by-parameter wiring.
        pub fn class_method_parameters(&self, idxs: &[usize]) -> Vec<Vec<String>> {
            self.inner.class_method_parameters(idxs).unwrap_or_default()
        }

        // --- whole-graph iteration ------------------------------------------

        /// Borrowing iterator over every node in the frozen graph, in index
        /// order. Filter on the borrowed `kind` / `path` without allocating;
        /// keep [`NodeRef::idx`] for the matches you act on. For a known index
        /// use [`Self::node`] (owned) instead.
        pub fn nodes(&self) -> impl Iterator<Item = NodeRef<'_>> {
            self.inner
                .outputs
                .builder
                .nodes
                .iter()
                .enumerate()
                .map(|(idx, n)| NodeRef {
                    idx,
                    fqname: &n.fqname,
                    kind: n.kind,
                    path: &n.path,
                    start_line: n.start_line,
                    end_line: n.end_line,
                    flags: n.flags,
                })
        }

        /// Iterator over every directed edge in the frozen graph. Combine with
        /// [`Self::node`] to inspect endpoints — e.g. fan out each
        /// [`FLAG_DYNAMIC_IMPORT`] edge whose `dst` is a module node.
        pub fn edges(&self) -> impl Iterator<Item = EdgeRef> + '_ {
            self.inner
                .outputs
                .builder
                .edges
                .iter()
                .map(|&(src, dst, flags)| EdgeRef { src, dst, flags })
        }

        /// Node indices matching every set criterion of `filter` (see
        /// [`NodeFilter`]). The general-purpose structural finder behind the
        /// narrower `find_*` helpers.
        pub fn nodes_matching(&self, filter: &NodeFilter<'_>) -> Vec<usize> {
            let to_owned = |s: &[&str]| -> Option<Vec<String>> {
                if s.is_empty() {
                    None
                } else {
                    Some(s.iter().map(|x| x.to_string()).collect())
                }
            };
            self.inner.indices_where(
                None,
                to_owned(filter.kinds),
                None,
                None,
                None,
                to_owned(filter.simple_names),
                to_owned(filter.paths),
                None,
                filter.flags_all,
                filter.flags_any,
                filter.fqname_prefix.map(|s| s.to_string()),
            )
        }

        /// Node indices matching the pre-bucketed entrypoint specs of
        /// [`crate::native_plugins`]' explicit-entrypoint plugin: `regexes`
        /// match a node's fqname (`re.match` semantics), `str_specs` are exact
        /// fqname / module / path matches, `abs_paths` match a node's source
        /// path. A regex that doesn't compile is dropped.
        pub fn nodes_matching_specs(
            &self,
            regexes: &[String],
            str_specs: &[String],
            abs_paths: &[String],
        ) -> Vec<usize> {
            self.inner
                .find_nodes_matching_specs_indices(regexes, str_specs, abs_paths)
        }

        /// The project root passed to the analysis, as a path string. Use it
        /// to locate config files or to relativize node paths.
        pub fn project_root(&self) -> &str {
            self.inner.project_root()
        }

        // --- flag registry --------------------------------------------------

        /// The bit a node flag named `name` (`owner/name`) was assigned, or
        /// `None` if no plugin/engine declared it. A plugin reads its own
        /// declared flag this way: the host allocated the bit from
        /// [`ExternalPlugin::declare_node_flags`] before `run`, so a plugin
        /// that declared `name` can `.expect()` the lookup.
        pub fn node_flag(&self, name: &str) -> Option<u32> {
            self.inner.node_flags.get(name).map(|bit| bit as u32)
        }

        /// The bit an edge flag named `name` was assigned (edge-space twin of
        /// [`Self::node_flag`]), or `None` if undeclared.
        pub fn edge_flag(&self, name: &str) -> Option<u8> {
            self.inner.edge_flags.get(name).map(|bit| bit as u8)
        }

        // --- topic registry -------------------------------------------------

        /// The handle a topic named `name` (`owner/name`) was assigned, or
        /// `None` if no plugin declared it. A plugin resolves its own declared
        /// topic this way: the host assigned the handle from
        /// [`ExternalPlugin::declare_topics`] before `run`, so a plugin that
        /// declared `name` can `.expect()` the lookup, then pass the handle to
        /// [`Self::facts_for_topic`].
        pub fn topic(&self, name: &str) -> Option<u32> {
            self.inner.topics.get(name)
        }

        /// Every [`Fact`] published under the topic `handle` across the whole
        /// project — the per-file plugins' [`FileOps::emit_fact`] output,
        /// collected during graph assembly. Returns an empty vec for an
        /// out-of-range handle or a topic nothing published under.
        pub fn facts_for_topic(&self, handle: u32) -> Vec<Fact> {
            match self.inner.topics.name_of(handle) {
                Some(name) => self
                    .inner
                    .outputs
                    .topic_facts
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
                None => Vec::new(),
            }
        }
    }

    /// Op sink for external plugins. Wraps the internal `PreparedOp` vec so
    /// plugins emit through named methods instead of constructing internals.
    /// Exposes the [`PreparedOp`] variants — `Entrypoint` / `FlagDecl` /
    /// `Edge` — as `keep_alive` / `flag_decl` / `add_edge`.
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

        /// Keep `decl_idx` reachable by flagging it an entrypoint seed.
        /// Emits a [`PreparedOp::Entrypoint`].
        pub fn keep_alive(&mut self, decl_idx: usize) {
            self.sink.push(PreparedOp::Entrypoint { decl_idx });
        }

        /// OR `flags` onto the existing decl at `decl_idx` (no new node).
        /// Use this to stamp a registered node flag — the bit resolved from
        /// [`PluginCtx::node_flag`] — directly on a discovered decl, instead
        /// of routing it through a seed marker node. Emits a
        /// [`PreparedOp::FlagDecl`].
        pub fn flag_decl(&mut self, decl_idx: usize, flags: u32) {
            self.sink.push(PreparedOp::FlagDecl { decl_idx, flags });
        }

        /// Add a reachability edge `src_idx -> dst_idx` between two
        /// existing nodes, stamped with `flags` (`0` for a plain edge, or
        /// one of [`FLAG_DEAD_BRANCH`] / [`FLAG_DYNAMIC_IMPORT`]). Emits a
        /// [`PreparedOp::Edge`].
        pub fn add_edge(&mut self, src_idx: usize, dst_idx: usize, flags: u8) {
            self.sink.push(PreparedOp::Edge {
                src_idx,
                dst_idx,
                flags,
            });
        }
    }

    // ---- per-file surface --------------------------------------------------

    /// The import a `kind="import"` node binds, exposed on
    /// [`FileNodeView::imports`]. Mirrors the per-file `ImportPayload`:
    /// `import foo` → `{module: "foo", decl: None, star: false}`;
    /// `from foo import bar` → `{module: "foo", decl: Some("bar"), …}`;
    /// `from foo import *` → `{module: "foo", decl: None, star: true}`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ImportRef {
        /// The imported-from module (absolute where ty could resolve it).
        pub module: String,
        /// The specific name pulled in, for `from … import name`; `None`
        /// for plain `import module` and for star imports.
        pub decl: Option<String>,
        /// True when this alias was bound by `from … import *`.
        pub star: bool,
    }

    /// Owned, plain-data snapshot of one node in a single file, returned by
    /// [`PluginFileCtx::nodes`] / [`PluginFileCtx::node`]. The `local_idx`
    /// addresses this file only — it's the index space [`FileOps`] emits
    /// against, *not* the project-wide [`NodeView::idx`].
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct FileNodeView {
        /// File-local index (0 is always the module node).
        pub local_idx: u32,
        /// Fully-qualified name.
        pub fqname: String,
        /// One of `function`, `class`, `variable`, `import`, `type_alias`,
        /// `module`.
        pub kind: String,
        /// Source path the node was declared in.
        pub path: String,
        /// 1-based start line.
        pub start_line: usize,
        /// 1-based end line.
        pub end_line: usize,
        /// `NodeFlags` bitset.
        pub flags: u32,
        /// For `kind == "import"` nodes, the import this alias binds;
        /// `None` for every other kind.
        pub imports: Option<ImportRef>,
    }

    /// Project the crate-internal per-file `NodeData` onto the curated,
    /// owned [`FileNodeView`]. Shared by [`PluginFileCtx::node`] and
    /// [`PluginFileCtx::nodes`] so the two stay in lockstep.
    fn file_node_view(
        local_idx: u32,
        node: &crate::file_payload::NodeData,
        path: &str,
    ) -> FileNodeView {
        FileNodeView {
            local_idx,
            fqname: node.fqname.to_string(),
            kind: node.kind.as_static_str().to_string(),
            path: path.to_string(),
            start_line: node.start_line,
            end_line: node.end_line,
            flags: node.flags,
            imports: node.imports.as_ref().map(|i| ImportRef {
                module: i.module.to_string(),
                decl: i.decl.as_ref().map(|d| d.to_string()),
                star: i.star,
            }),
        }
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

        /// File-local index of the module node (always 0).
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
            Some(file_node_view(local_idx, node, &self.inner.path()))
        }

        /// Every node in this file as a [`FileNodeView`], in file-local
        /// index order.
        pub fn nodes(&self) -> Vec<FileNodeView> {
            let path = self.inner.path();
            self.inner
                .nodes()
                .iter()
                .enumerate()
                .map(|(i, node)| file_node_view(i as u32, node, &path))
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
        // a plugin pipes the result straight into `ops.keep_alive(idx)`
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

        // --- dual-mode (fact-emission) helpers, in-tree only ----------------
        //
        // `pub(crate)` so the surface external dylib authors compile against
        // stays unchanged; the in-tree dual-mode built-ins use these to emit
        // facts pinned to file-local decls, matching their old project-wide
        // walk while riding the per-file salsa cache.

        /// This file's source path (every node in it shares it).
        pub(crate) fn path_string(&self) -> String {
            self.inner.path()
        }

        /// File-local decls decorated by one of `names` from one of `modules`,
        /// paired with the decorator's captured kwargs. See
        /// [`FileContext::decorated_decls_with_args`].
        pub(crate) fn decorated_decls_with_args(
            &self,
            modules: &[&str],
            names: &[&str],
        ) -> Vec<(u32, CallArgs)> {
            self.inner.decorated_decls_with_args(modules, names)
        }

        /// `file-local idx → function parameter names`.
        pub(crate) fn function_params_by_local(
            &self,
        ) -> rustc_hash::FxHashMap<u32, Vec<compact_str::CompactString>> {
            self.inner.function_params_by_local()
        }

        /// `file-local idx → class method parameter names`.
        pub(crate) fn class_method_params_by_local(
            &self,
        ) -> rustc_hash::FxHashMap<u32, Vec<compact_str::CompactString>> {
            self.inner.class_method_params_by_local()
        }

        /// File-local indices of classes defining a method named `method_name`.
        pub(crate) fn classes_defining_method(&self, method_name: &str) -> Vec<u32> {
            self.inner.classes_defining_method(method_name)
        }

        /// `(owner name, file-local idx)` for `@<owner>.<attr>(...)`-decorated
        /// decls whose `attr` is one of `decorator_attrs`.
        pub(crate) fn handler_decorators_local(
            &self,
            decorator_attrs: &[&str],
        ) -> Vec<(String, u32)> {
            self.inner.handler_decorators_local(decorator_attrs)
        }

        /// `(local owner, literal)` for calls to `name` imported from one of
        /// `modules`, taking the `arg_index` positional string literal.
        pub(crate) fn calls_to_imported_local(
            &self,
            modules: &[&str],
            name: &str,
            arg_index: usize,
        ) -> Vec<(u32, String)> {
            self.inner.calls_to_imported_local(modules, name, arg_index)
        }

        /// `(local owner, literal)` for `<owner>.<attr>(...)` calls.
        pub(crate) fn calls_on_var_local(
            &self,
            owner_name: &str,
            attr: &str,
            arg_index: usize,
            required_positional: Option<usize>,
        ) -> Vec<(u32, String)> {
            self.inner
                .calls_on_var_local(owner_name, attr, arg_index, required_positional)
        }
    }

    /// File-local op sink for a [`PerFilePlugin`]. Emits in this file's
    /// *local* index space — endpoints are positions in
    /// [`PluginFileCtx::nodes`], which the host translates to global indices
    /// at apply time.
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

        /// Keep the node at file-local index `decl_local_idx` alive by
        /// flagging it an entrypoint seed. The index is a position in
        /// [`PluginFileCtx::nodes`]; an index with no node is dropped at
        /// apply time.
        pub fn keep_alive(&mut self, decl_local_idx: u32) {
            self.sink.push(FileLocalOp::Entrypoint { decl_local_idx });
        }

        /// OR `flags` onto the existing node at file-local index
        /// `decl_local_idx` (no new node) — the per-file twin of
        /// [`PluginOps::flag_decl`]. The index is a position in
        /// [`PluginFileCtx::nodes`]; an index with no node is dropped at
        /// apply time. Emits a [`FileLocalOp::FlagDecl`].
        pub fn flag_decl(&mut self, decl_local_idx: u32, flags: u32) {
            self.sink.push(FileLocalOp::FlagDecl {
                decl_local_idx,
                flags,
            });
        }

        /// Add a reachability edge from `src_local_idx` to `dst_local_idx`,
        /// stamped with `flags` (`0` for a plain edge, or one of
        /// [`FLAG_DEAD_BRANCH`] / [`FLAG_DYNAMIC_IMPORT`]). Both endpoints
        /// are positions in [`PluginFileCtx::nodes`]; an edge with an
        /// unresolvable endpoint is dropped at apply time.
        pub fn add_edge(&mut self, src_local_idx: u32, dst_local_idx: u32, flags: u8) {
            self.sink.push(FileLocalOp::Edge {
                src_local_idx,
                dst_local_idx,
                flags,
            });
        }

        /// Publish a fact under topic `topic` (a name declared in
        /// [`ExternalPlugin::declare_topics`] — emitted by name, not handle, so
        /// the per-file salsa cache stays independent of the plugin set).
        /// `decl_local_idx`, when `Some`, pins the fact to a position in
        /// [`PluginFileCtx::nodes`], translated to a global node index at
        /// assemble time (a fact whose decl fails to translate is dropped);
        /// `None` leaves the fact file-scoped. The project-wide side reads it
        /// back via [`PluginCtx::facts_for_topic`]. Emits a
        /// [`FileLocalOp::Fact`].
        pub fn emit_fact(
            &mut self,
            topic: impl Into<String>,
            decl_local_idx: Option<u32>,
            value: impl Into<String>,
        ) {
            self.sink.push(FileLocalOp::Fact {
                topic: topic.into().into(),
                decl_local_idx,
                value: value.into().into(),
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
        fn file_ops_add_edge_carries_flags() {
            let mut ops = FileOps::new();
            ops.add_edge(0, 1, FLAG_DEAD_BRANCH);
            assert!(matches!(
                ops.into_inner().as_slice(),
                [FileLocalOp::Edge { src_local_idx: 0, dst_local_idx: 1, flags }]
                    if *flags == FLAG_DEAD_BRANCH
            ));
        }
    }
}

/// ABI fingerprint this runtime accepts (see `build.rs`):
/// `<abi-epoch>|api<plugin-api-epoch>|<rustc-commit>|v<version>|<target>`. An
/// external plugin bakes this exact string at compile time; the airlock rejects
/// any plugin whose baked fingerprint differs. The `api<N>` segment tracks
/// [`plugin_api::PLUGIN_API_EPOCH`].
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
                    _lib: Some(Arc::clone(&lib)),
                },
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Native plugins ported from Python. `init_subclass` is project-wide
// (subclasses live in other files, so its output isn't file-local). The
// project-wide `py`-taking ctx accessors are reached via `Python::with_gil`
// (the harness already holds the GIL).
// ---------------------------------------------------------------------------

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

/// Port of `dead_cst.plugins.init_subclass.InitSubclassPlugin`: for each
/// class defining `__init_subclass__`, emit a marker node reachable from the
/// parent that keeps every transitive subclass alive.
/// Topic pinning each class that defines `__init_subclass__` (file-local
/// detection); the project-wide resolve walks its subclasses.
const INIT_SUBCLASS_TOPIC_PARENT: &str = "init_subclass/parent";

/// Init-subclass, decomposed: the per-file pass detects `__init_subclass__`
/// definers (a pure file-local read of `class_method_defs`); the project-wide
/// resolve does the cross-file subclass walk those parents need.
pub(crate) struct InitSubclassPluginImpl;

impl plugin_api::PerFilePlugin for InitSubclassPluginImpl {
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        for local in file.classes_defining_method("__init_subclass__") {
            ops.emit_fact(INIT_SUBCLASS_TOPIC_PARENT, Some(local), "");
        }
    }
}

impl plugin_api::ExternalPlugin for InitSubclassPluginImpl {
    fn name(&self) -> &str {
        "InitSubclassPlugin"
    }

    fn declare_topics(&self) -> Vec<crate::topic_registry::TopicSpec> {
        vec![crate::topic_registry::TopicSpec {
            name: INIT_SUBCLASS_TOPIC_PARENT.to_string(),
            description: "A class defining __init_subclass__, pinned to the class decl; \
                          its transitive subclasses are kept reachable from it."
                .to_string(),
        }]
    }

    fn per_file(&self) -> Option<&dyn plugin_api::PerFilePlugin> {
        Some(self)
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        let Some(handle) = ctx.topic(INIT_SUBCLASS_TOPIC_PARENT) else {
            return Ok(());
        };
        for fact in ctx.facts_for_topic(handle) {
            let Some(parent_idx) = fact.decl_idx else {
                continue;
            };
            for subclass_idx in ctx.find_subclasses_of(parent_idx) {
                ops.add_edge(parent_idx, subclass_idx, plugin_api::FLAG_INIT_SUBCLASS);
            }
        }
        Ok(())
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
/// is one of `filenames`, keep that file's whole top-level surface alive (via
/// `keep_alive`, stamping ENTRYPOINT on each node). File-local — the match and
/// the targets are both functions of the single file.
fn server_config_run_on_file(
    filenames: &[String],
    file: &plugin_api::PluginFileCtx<'_>,
    ops: &mut plugin_api::FileOps,
) {
    SERVER_CONFIG_RUN_COUNT.fetch_add(1, Ordering::Relaxed);
    let nodes = file.nodes();
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
        .filter(|node| {
            matches!(
                node.kind.as_str(),
                "module" | "function" | "class" | "variable" | "import"
            )
        })
        .map(|node| node.local_idx)
        .collect();
    for local in targets {
        ops.keep_alive(local);
    }
}

/// Project-wide port of `dead_cst.contrib.unittest.UnittestPlugin`: keep
/// every transitive subclass of `unittest.TestCase` /
/// `IsolatedAsyncioTestCase` alive, plus module-level lifecycle hooks
/// (`setUpModule` / `tearDownModule` / `load_tests`) in any file that
/// imports `unittest`. Cross-file (the subclass walk spans files), so it
/// can't be per-file.
/// File-scoped fact (no pinned decl) signalling that this file imports
/// `unittest` — the project-wide resolve runs the TestCase subclass walk iff
/// any file published it (preserving the old `has_imports_of` short-circuit).
const UNITTEST_TOPIC_IMPORTS: &str = "unittest/imports";

/// Unittest, decomposed. The module-hook wiring is purely file-local — a
/// `module -> setUpModule/tearDownModule/load_tests` edge within one file — so
/// it moves wholesale into the per-file pass (salsa-cached; the old version
/// rescanned every project node each build). The TestCase subclass flagging is
/// genuinely cross-file (a base resolved through ty, transitive descendants in
/// other files), so it stays in the project-wide resolve, gated on the
/// per-file `unittest/imports` signal.
pub(crate) struct UnittestPluginImpl;

impl plugin_api::PerFilePlugin for UnittestPluginImpl {
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        if !file.imports_any_module(&["unittest"]) {
            return;
        }
        // Signal the project-wide subclass walk that unittest is in play.
        ops.emit_fact(UNITTEST_TOPIC_IMPORTS, None, "");
        // Module lifecycle hooks ride a `module -> hook` edge (alive iff the
        // module is reached) — both endpoints are in this file, so it's a
        // file-local edge, no project query needed.
        let module_idx = file.module_local_idx();
        for node in file.nodes().iter().skip(1) {
            if node.kind != "function" {
                continue;
            }
            let simple = node
                .fqname
                .rsplit_once('.')
                .map(|(_, n)| n)
                .unwrap_or(node.fqname.as_str());
            if UNITTEST_MODULE_HOOKS.contains(&simple) {
                ops.add_edge(module_idx, node.local_idx, 0);
            }
        }
    }
}

impl plugin_api::ExternalPlugin for UnittestPluginImpl {
    fn name(&self) -> &str {
        "UnittestPlugin"
    }

    fn declare_node_flags(&self) -> Vec<plugin_api::FlagSpec> {
        vec![testcase_flag_spec()]
    }

    fn declare_topics(&self) -> Vec<crate::topic_registry::TopicSpec> {
        vec![crate::topic_registry::TopicSpec {
            name: UNITTEST_TOPIC_IMPORTS.to_string(),
            description: "Published (file-scoped) by each file importing unittest; gates \
                          the project-wide TestCase subclass walk."
                .to_string(),
        }]
    }

    fn per_file(&self) -> Option<&dyn plugin_api::PerFilePlugin> {
        Some(self)
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        // Short-circuit before the subclass walk when nothing imports unittest
        // (the per-file pass would have published at least one fact).
        let imports = ctx
            .topic(UNITTEST_TOPIC_IMPORTS)
            .map(|h| ctx.facts_for_topic(h));
        if imports.is_none_or(|f| f.is_empty()) {
            return Ok(());
        }
        // The bit the host allocated for our declared `test/testcase` flag.
        let testcase_flag = ctx
            .node_flag("test/testcase")
            .expect("test/testcase is declared in declare_node_flags");

        // TestCase subclasses are unconditional roots — a test class stays
        // alive even if nothing imports its module — so stamp the
        // `test/testcase` seed flag directly on the decl. Deduped because a
        // class can match two bases (`IsolatedAsyncioTestCase` is itself a
        // `TestCase` subclass, so its descendants resolve under both).
        let mut flagged: FxHashSet<usize> = FxHashSet::default();
        for base in UNITTEST_BASE_FQNAMES {
            for idx in ctx.find_subclasses_of_fqn(base, true) {
                if flagged.insert(idx) {
                    ops.flag_decl(idx, testcase_flag);
                }
            }
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
// this is a project-wide `ExternalPlugin`, not a per-file one.
//
// Verbatim port of `DispatchAppPlugin._gather_one` (discovery) + `.policy`
// (steps 3-6 of emission). The Python harness auto-batched the gather across
// plugins purely for speed; native plugins run independently with identical
// per-plugin output (the batching was a perf optimization, not a semantic one).
// ---------------------------------------------------------------------------

/// Celery-style appless `@shared_task` fan-out layered on top of the standard
/// dispatch policy. Mirrors `CeleryPlugin.policy`'s `super().policy(...)` then
/// shared-task pass: every top-level function decorated with one of `names`
/// (imported from `module`) is kept alive directly. `None` for non-celery
/// configs.
struct SharedTaskFanout {
    module: String,
    names: Vec<String>,
}

/// Pure-data description of a dispatch-app framework — the Rust twin of the
/// Python `DispatchAppSpec` (plus the optional celery `shared_task` extension).
/// The seven bundled framework configs are built by the `*_config()` helpers
/// below; the Python `NativePlugin.dispatch_app(...)` factory builds one from
/// caller-supplied values (owned, so the config need not be `'static`).
pub(crate) struct DispatchAppConfig {
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
    fn is_active(&self, ctx: &plugin_api::PluginCtx<'_>) -> bool {
        let cfg = &self.config;
        if cfg.app_classes.is_empty() || cfg.registration_decorators.is_empty() {
            return false;
        }
        for fqn in &cfg.app_classes {
            if let Some((module, _)) = fqn.rsplit_once('.') {
                if !module.is_empty() && ctx.has_imports_of(module) {
                    return true;
                }
            }
        }
        false
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

impl plugin_api::ExternalPlugin for DispatchAppPluginImpl {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        let cfg = &self.config;
        if !self.is_active(ctx) {
            return Ok(());
        }

        // --- Phase 1: gather indices. ---

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
            let sub_idxs = ctx.find_subclasses_of_fqn(fqn, true);
            if sub_idxs.is_empty() {
                continue;
            }
            for attr in ctx.nodes_at(&sub_idxs) {
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
            let names_ref: Vec<&str> = names_vec.iter().map(String::as_str).collect();
            for var_idx in ctx.constructions(&[module.as_str()], &names_ref) {
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
                let names_ref: Vec<&str> = names_vec.iter().map(String::as_str).collect();
                for (decl_idx, kinds) in ctx.factory_decls(&[module.as_str()], &names_ref) {
                    for kind in kinds {
                        if factory_seen.insert((decl_idx, kind.clone())) {
                            factory_decls.push((decl_idx, kind));
                        }
                    }
                }
            }
        }

        // handlers: `@<owner>.<reg_decorator>(...)`-decorated functions.
        let reg_ref: Vec<&str> = cfg
            .registration_decorators
            .iter()
            .map(String::as_str)
            .collect();
        let handlers: Vec<(String, usize)> = ctx.handler_decorators(&reg_ref);

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
                for pred in ctx.direct_predecessors(decl_idx) {
                    factory_reachers.insert(pred);
                }
            }
        }

        // shared_task (celery): `@shared_task`-decorated decls.
        let shared_idxs: Vec<usize> = match &cfg.shared_task {
            Some(st) => {
                let names_ref: Vec<&str> = st.names.iter().map(String::as_str).collect();
                ctx.decorated_decls(&[st.module.as_str()], &names_ref)
            }
            None => Vec::new(),
        };

        // --- Phase 2: resolve paths/fqnames + assemble ops. All index
        // gathering happened above; bulk-fetch the node attrs the emission
        // loops need (parallel to their index vecs), then emit through `ops`.

        // vars_by_file: (path, simple var name) -> first var idx wins.
        let mut vars_by_file: FxHashMap<(String, String), usize> = FxHashMap::default();
        for node in ctx.nodes() {
            if node.kind != "variable" {
                continue;
            }
            vars_by_file
                .entry((node.path.to_string(), simple_name(node.fqname).to_string()))
                .or_insert(node.idx);
        }

        // Steps 1 + 3: build direct_by_owner and entrypoint-promote each
        // direct construction (the latter only under seed_as_entrypoint).
        let direct_attrs = ctx.nodes_at(&direct);
        let mut direct_by_owner: FxHashMap<(String, String), Vec<usize>> = FxHashMap::default();
        for (&var_idx, node) in direct.iter().zip(direct_attrs.iter()) {
            direct_by_owner
                .entry((node.path.clone(), simple_name(&node.fqname).to_string()))
                .or_default()
                .push(var_idx);
            if cfg.seed_as_entrypoint {
                ops.keep_alive(var_idx);
            }
        }

        // Step 5: wire handler decorators to their owner var. Seed mode
        // uses vars_by_file (so `app = create_app()` factory chains pick up
        // edges); pure-dispatch mode wires only to direct constructions (so
        // a star-imported `app = App()` stays invisible).
        let handler_idxs: Vec<usize> = handlers.iter().map(|(_, i)| *i).collect();
        let handler_paths = ctx.node_paths(&handler_idxs);
        for ((owner, decorated_idx), dpath) in handlers.iter().zip(handler_paths.iter()) {
            let key = (dpath.clone(), owner.clone());
            if cfg.seed_as_entrypoint {
                if let Some(&var_idx) = vars_by_file.get(&key) {
                    ops.add_edge(var_idx, *decorated_idx, 0);
                }
            } else if let Some(var_idxs) = direct_by_owner.get(&key) {
                for &var_idx in var_idxs {
                    ops.add_edge(var_idx, *decorated_idx, 0);
                }
            }
        }

        // Step 6: factory walk — entrypoint-promote vars whose *direct*
        // successor is a factory decl (seed + factory only). The
        // `factory_reachers` membership test rules out the two-hop
        // `wrapper = app; app = create_app()` over-promotion case.
        if cfg.seed_as_entrypoint && !factory_decls.is_empty() {
            let mut classified: FxHashSet<(String, String)> = FxHashSet::default();
            for ((owner, _decorated_idx), dpath) in handlers.iter().zip(handler_paths.iter()) {
                let key = (dpath.clone(), owner.clone());
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
                ops.keep_alive(var_idx);
            }
        }

        // Celery `@shared_task` fan-out: keep every appless task callable
        // alive directly (no per-file marker node).
        if cfg.shared_task.is_some() {
            for &decl_idx in &shared_idxs {
                ops.keep_alive(decl_idx);
            }
        }

        Ok(())
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
        app_classes: owned(&["typer.Typer"]),
        registration_decorators: owned(&["command", "callback"]),
        seed_as_entrypoint: false,
        shared_task: None,
    }
}

fn cyclopts_config() -> DispatchAppConfig {
    DispatchAppConfig {
        app_classes: owned(&["cyclopts.App"]),
        registration_decorators: owned(&["command", "default"]),
        seed_as_entrypoint: false,
        shared_task: None,
    }
}

fn slack_bolt_config() -> DispatchAppConfig {
    DispatchAppConfig {
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
        app_classes: owned(&["fastmcp.FastMCP", "mcp.server.fastmcp.FastMCP"]),
        registration_decorators: owned(&["tool", "resource", "prompt", "completion"]),
        seed_as_entrypoint: true,
        shared_task: None,
    }
}

fn celery_config() -> DispatchAppConfig {
    DispatchAppConfig {
        app_classes: owned(&["celery.Celery"]),
        registration_decorators: owned(&["task"]),
        seed_as_entrypoint: true,
        shared_task: Some(SharedTaskFanout {
            module: "celery".to_string(),
            names: owned(&["shared_task"]),
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

impl plugin_api::PerFilePlugin for ClickPluginImpl {
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        // Cheap import-presence guard.
        if !file.imports_any_module(&["click"]) {
            return;
        }

        // --- Phase 1: gather file-local indices. ---

        // Groups via `@click.group` / `@click.Group`-decorated decls.
        let group_decls = file.decorated_decls(&["click"], &CLICK_GROUP_DECORATORS);
        // Groups via `X = click.Group(...)` constructions.
        let group_ctors = file.constructions(&["click"], &["Group"]);
        // No groups => no wiring to do (the fixpoint would no-op anyway).
        if group_decls.is_empty() && group_ctors.is_empty() {
            return;
        }

        // Handlers: `@<owner>.{command,group,result_callback}(...)`.
        let handlers = file.handler_decorators_local(&CLICK_REGISTRATION_DECORATORS);
        // Subgroup links: handlers decorated specifically with
        // `@<owner>.group(...)`. Wiring such a handler promotes it to a group
        // so its own `@<handler>.command()` handlers wire next pass.
        let subgroup_links: FxHashSet<(u32, String)> = file
            .handler_decorators_local(&[CLICK_SUBGROUP_DECORATOR])
            .into_iter()
            .map(|(owner, idx)| (idx, owner))
            .collect();

        // --- Phase 2: group->handler fixpoint. Everything is keyed within
        // this one file, so the old `(path, name)` key reduces to the simple
        // name (path is constant), and every wired edge is file-local. ---

        // Local idx -> simple name, for every node in the file.
        let nodes = file.nodes();
        let simple_of = |idx: u32| -> String {
            nodes
                .get(idx as usize)
                .map(|n| simple_name(&n.fqname).to_string())
                .unwrap_or_default()
        };

        // groups_by_name: simple name -> [group local idx, ...].
        let mut groups_by_name: FxHashMap<String, Vec<u32>> = FxHashMap::default();
        for &g in group_decls.iter().chain(group_ctors.iter()) {
            groups_by_name.entry(simple_of(g)).or_default().push(g);
        }

        // Fixpoint: wire each handler to its owning group(s); a newly wired
        // subgroup handler becomes a group, exposing its own handlers next
        // pass. `emitted` dedups edges so the loop terminates once no new group
        // is discovered. Verbatim port of `ClickPlugin.run`'s loop.
        let mut emitted: FxHashSet<(u32, u32)> = FxHashSet::default();
        let mut changed = true;
        while changed {
            changed = false;
            for (owner_name, decorated_idx) in &handlers {
                // Snapshot the owner's group idxs: the insert below may mutate
                // `groups_by_name`, and the dedup + outer loop make deferring a
                // same-pass insertion to the next pass fixpoint-equivalent.
                let Some(owner_idxs) = groups_by_name.get(owner_name) else {
                    continue;
                };
                for owner_idx in owner_idxs.clone() {
                    if !emitted.insert((owner_idx, *decorated_idx)) {
                        continue;
                    }
                    ops.add_edge(owner_idx, *decorated_idx, 0);
                    if subgroup_links.contains(&(*decorated_idx, owner_name.clone())) {
                        groups_by_name
                            .entry(simple_of(*decorated_idx))
                            .or_default()
                            .push(*decorated_idx);
                        changed = true;
                    }
                }
            }
        }
    }
}

impl plugin_api::ExternalPlugin for ClickPluginImpl {
    fn name(&self) -> &str {
        "click"
    }

    fn per_file(&self) -> Option<&dyn plugin_api::PerFilePlugin> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// MockPatchPlugin — project-wide port of `dead_cst.contrib.mock_patch`.
// Resolve string-fqname patch targets (`unittest.mock.patch` / `mock.patch`,
// pytest-mock's `mocker.patch`, pytest's `monkeypatch.setattr` / `.delattr`)
// to their decls and emit keep-alive edges. Cross-file (a patch string in a
// test file targets a decl in another), hence project-wide.
// ---------------------------------------------------------------------------

/// Topic carrying one patch site: pinned to the enclosing decl (the `owner`),
/// value = the patch target's fqname string. The project-wide resolve groups
/// by fqname, resolves it cross-file, and wires the keep-alive edges.
const MOCKPATCH_TOPIC_SITE: &str = "mockpatch/site";

/// Mock-patch, decomposed. Detecting the four patch-call shapes — `patch("X")`
/// (from unittest.mock / mock), `mocker.patch("X")`,
/// `monkeypatch.setattr("X", v)` / `.delattr("X")` — is a pure read of this
/// file's salsa-cached call sites, emitted as `mockpatch/site` facts. The
/// project-wide run does only the cross-file part: resolving each target
/// fqname to its decls / module and wiring `owner -> target` keep-alive edges.
pub(crate) struct MockPatchPluginImpl;

impl plugin_api::PerFilePlugin for MockPatchPluginImpl {
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        let mut emit = |owner: u32, target: String| {
            ops.emit_fact(MOCKPATCH_TOPIC_SITE, Some(owner), target);
        };
        // `patch("X.Y")` imported from unittest.mock / mock.
        for (owner, target) in file.calls_to_imported_local(&["unittest.mock", "mock"], "patch", 0)
        {
            emit(owner, target);
        }
        // `mocker.patch("X.Y")` (pytest-mock fixture).
        for (owner, target) in file.calls_on_var_local("mocker", "patch", 0, None) {
            emit(owner, target);
        }
        // `monkeypatch.setattr("X.Y", v)` (2 positional) /
        // `monkeypatch.delattr("X.Y")` (1 positional). The required-positional
        // count distinguishes the fqname form from the object form
        // (`setattr(obj, "name", v)`).
        for (attr, required) in [("setattr", 2usize), ("delattr", 1usize)] {
            for (owner, target) in file.calls_on_var_local("monkeypatch", attr, 0, Some(required)) {
                emit(owner, target);
            }
        }
    }
}

impl plugin_api::ExternalPlugin for MockPatchPluginImpl {
    fn name(&self) -> &str {
        "MockPatchPlugin"
    }

    fn declare_topics(&self) -> Vec<crate::topic_registry::TopicSpec> {
        vec![crate::topic_registry::TopicSpec {
            name: MOCKPATCH_TOPIC_SITE.to_string(),
            description: "A patch/monkeypatch call site: pinned to the enclosing decl, \
                          value = the target fqname string to keep alive."
                .to_string(),
        }]
    }

    fn per_file(&self) -> Option<&dyn plugin_api::PerFilePlugin> {
        Some(self)
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        let Some(handle) = ctx.topic(MOCKPATCH_TOPIC_SITE) else {
            return Ok(());
        };
        // Bucket owners by target fqname (first-seen order preserved for
        // determinism; facts arrive in file order).
        let mut order: Vec<String> = Vec::new();
        let mut owners_by_fqname: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for fact in ctx.facts_for_topic(handle) {
            let Some(owner_idx) = fact.decl_idx else {
                continue;
            };
            if !owners_by_fqname.contains_key(&fact.value) {
                order.push(fact.value.clone());
            }
            owners_by_fqname
                .entry(fact.value)
                .or_default()
                .push(owner_idx);
        }

        // Resolve targets per fqname (decls + the module node if the fqname is
        // itself a module) and wire a direct `owner -> target` keep-alive edge
        // for each patch site. An unresolved fqname has no targets and emits no
        // edge — there is nothing to keep alive.
        for fqname in order {
            let owners = &owners_by_fqname[&fqname];
            let mut target_idxs = ctx.find_declarations(&fqname);
            if let Some(mod_idx) = ctx.find_module(&fqname) {
                target_idxs.push(mod_idx);
            }
            for &owner_idx in owners {
                for &target_idx in &target_idxs {
                    ops.add_edge(owner_idx, target_idx, 0);
                }
            }
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

impl plugin_api::ExternalPlugin for DiscordPyPluginImpl {
    fn name(&self) -> &str {
        "DiscordPyPlugin"
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        // Cheap presence probe — short-circuit before any walk.
        let mut imports_discord = false;
        for m in DISCORD_PROBE_MODULES {
            if ctx.has_imports_of(m) {
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
            let idxs = ctx.imports_of(m);
            for p in ctx.node_paths(&idxs) {
                discord_paths.insert(p);
            }
        }

        // --- Phase 1: gather indices. ---

        // 1. Bot / Client constructions (var idxs, in discovery order).
        let mut bot_var_idxs: Vec<usize> = Vec::new();
        let mut bot_names: Vec<String> = DISCORD_COMMANDS_BOT_KINDS
            .iter()
            .map(|s| s.to_string())
            .collect();
        bot_names.sort();
        let bot_names_ref: Vec<&str> = bot_names.iter().map(String::as_str).collect();
        bot_var_idxs.extend(ctx.constructions(&["discord.ext.commands"], &bot_names_ref));
        let mut client_names: Vec<String> =
            DISCORD_CLIENT_KINDS.iter().map(|s| s.to_string()).collect();
        client_names.sort();
        let client_names_ref: Vec<&str> = client_names.iter().map(String::as_str).collect();
        bot_var_idxs.extend(ctx.constructions(&["discord"], &client_names_ref));

        // 2 + 3. Handler decorators: single-attr `@<bot>.<verb>` and
        // two-level `@<bot>.tree.<verb>` slash commands.
        let bot_handlers: Vec<(String, usize)> = ctx.handler_decorators(&DISCORD_BOT_DECORATORS);
        let tree_handlers: Vec<(String, usize)> =
            ctx.handler_decorators_via("tree", &DISCORD_TREE_DECORATORS);

        // 4. Cog subclasses + their files; module setup/teardown hooks.
        let mut cogs_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut cog_path_order: Vec<String> = Vec::new();
        for base in DISCORD_COG_BASES {
            let cog_idxs = ctx.find_subclasses_of_fqn(base, true);
            if cog_idxs.is_empty() {
                continue;
            }
            let cog_paths = ctx.node_paths(&cog_idxs);
            for (idx, path) in cog_idxs.into_iter().zip(cog_paths) {
                if !cogs_by_path.contains_key(&path) {
                    cog_path_order.push(path.clone());
                }
                cogs_by_path.entry(path).or_default().push(idx);
            }
        }
        let mut hook_funcs_by_path: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        if !cogs_by_path.is_empty() {
            let cog_path_refs: Vec<&str> = cog_path_order.iter().map(String::as_str).collect();
            let hook_idxs = ctx.nodes_matching(&plugin_api::NodeFilter {
                kinds: &["function"],
                simple_names: &["setup", "teardown"],
                paths: &cog_path_refs,
                fqname_prefix: None,
                flags_all: None,
                flags_any: None,
            });
            if !hook_idxs.is_empty() {
                let hook_paths = ctx.node_paths(&hook_idxs);
                for (idx, path) in hook_idxs.into_iter().zip(hook_paths) {
                    hook_funcs_by_path.entry(path).or_default().push(idx);
                }
            }
        }

        // 6. load_extension / load_extensions string-literal targets
        // (owner decl idx + extension fqname), raw — gated/deduped below.
        let mut raw_ext_calls: Vec<(usize, String)> = Vec::new();
        for attr in ["load_extension", "load_extensions"] {
            for (owner_idx, ext_fqname) in ctx.calls_on_attr(attr, 0) {
                raw_ext_calls.push((owner_idx, ext_fqname));
            }
        }

        // --- Phase 2a: build bot var map, emit bot entrypoints + handler
        // edges, gate + dedup extension calls. Node attrs (path / simple name)
        // are pre-fetched in bulk so the loops need no random node access. ---
        let mut pending_extensions: Vec<String> = Vec::new();

        // bot_vars_by_file: path -> { simple var name -> var idx }.
        let mut bot_vars_by_file: FxHashMap<String, FxHashMap<String, usize>> =
            FxHashMap::default();
        for nv in ctx.nodes_at(&bot_var_idxs) {
            if !discord_paths.contains(nv.path.as_str()) {
                continue;
            }
            bot_vars_by_file
                .entry(nv.path.clone())
                .or_default()
                .insert(simple_name(&nv.fqname).to_string(), nv.idx);
            ops.keep_alive(nv.idx);
        }

        // Wire single-attr + two-level handler decorators to their bot.
        let handler_idxs: Vec<usize> = bot_handlers
            .iter()
            .chain(tree_handlers.iter())
            .map(|(_, idx)| *idx)
            .collect();
        let handler_paths: FxHashMap<usize, String> = handler_idxs
            .iter()
            .copied()
            .zip(ctx.node_paths(&handler_idxs))
            .collect();
        for (owner, decorated_idx) in bot_handlers.iter().chain(tree_handlers.iter()) {
            let Some(path) = handler_paths.get(decorated_idx) else {
                continue;
            };
            if let Some(&owner_idx) = bot_vars_by_file.get(path).and_then(|m| m.get(owner)) {
                ops.add_edge(owner_idx, *decorated_idx, 0);
            }
        }

        // Gate extension calls on importer files; dedup by fqname.
        let ext_owner_idxs: Vec<usize> = raw_ext_calls.iter().map(|(idx, _)| *idx).collect();
        let ext_owner_paths: FxHashMap<usize, String> = ext_owner_idxs
            .iter()
            .copied()
            .zip(ctx.node_paths(&ext_owner_idxs))
            .collect();
        let mut seen_extensions: FxHashSet<String> = FxHashSet::default();
        for (owner_idx, ext_fqname) in &raw_ext_calls {
            let Some(path) = ext_owner_paths.get(owner_idx) else {
                continue;
            };
            if !discord_paths.contains(path.as_str()) {
                continue;
            }
            if !seen_extensions.insert(ext_fqname.clone()) {
                continue;
            }
            pending_extensions.push(ext_fqname.clone());
        }

        // --- Phase 2b: keep every cog file's Cog subclasses + setup/teardown
        // hooks alive (ENTRYPOINT stamped directly on each via keep_alive). ---
        for path in &cog_path_order {
            let mut targets = cogs_by_path[path].clone();
            if let Some(hooks) = hook_funcs_by_path.get(path) {
                targets.extend(hooks.iter().copied());
            }
            for t in targets {
                ops.keep_alive(t);
            }
        }

        // --- Phase 2c: keep each loaded extension module's surface alive
        // (ENTRYPOINT stamped directly on each surface node via keep_alive). ---
        if !pending_extensions.is_empty() {
            let surfaces = ctx.module_surfaces(&pending_extensions);
            for ext_fqname in &pending_extensions {
                let targets = surfaces.get(ext_fqname).cloned().unwrap_or_default();
                for t in targets {
                    ops.keep_alive(t);
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DynamicImportFallbackPlugin — project-wide port of
// `dead_cst.plugins.dynamic_import`. Fan out each `EdgeFlags.DYNAMIC_IMPORT`
// edge whose dst is a module node to that module's exports. The include/exclude
// glob knobs gate the fan-out; with no filters every such edge fans out.
// Source globs match like `pathlib.PurePosixPath.match` (componentwise, from
// the right, case-sensitive); target globs match like `fnmatch.fnmatchcase`.
// Both are reimplemented in Rust over the `regex` crate (`fnmatch_regex` /
// `PathGlob`) — no host-Python call on any path.
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

    /// Exports of `module_fqname` per `_exports_indices_for`: `__all__` when
    /// `respect_dunder_all` and present, else the module's top-level decls
    /// (dropping `_`-prefixed names unless `include_underscore`).
    fn exports_for(&self, ctx: &plugin_api::PluginCtx<'_>, module_fqname: &str) -> Vec<usize> {
        if self.respect_dunder_all {
            if let Some(exports) = ctx.dunder_all_exports(module_fqname) {
                return exports;
            }
        }
        let decls = ctx.module_top_level_decls(module_fqname);
        if self.include_underscore || decls.is_empty() {
            return decls;
        }
        let attrs = ctx.nodes_at(&decls);
        decls
            .into_iter()
            .zip(attrs)
            .filter(|(_idx, attr)| !simple_name(&attr.fqname).starts_with('_'))
            .map(|(idx, _attr)| idx)
            .collect()
    }
}

/// A shell glob compiled to a regex, matching like `fnmatch.fnmatchcase`
/// (case-sensitive, anchored to the whole string). A pattern that fails to
/// compile never matches (degenerate input; real globs always compile).
struct FnGlob(Option<regex::Regex>);

impl FnGlob {
    fn new(pat: &str) -> Self {
        FnGlob(regex::Regex::new(&fnmatch_regex(pat)).ok())
    }

    fn is_match(&self, s: &str) -> bool {
        self.0.as_ref().is_some_and(|re| re.is_match(s))
    }
}

/// A path glob matching like `pathlib.PurePosixPath.match`: split into `/`
/// components, compared from the right, each component matched case-sensitively
/// as an `fnmatch` glob (so `*`/`?` never cross a `/`). A relative pattern
/// matches any path whose trailing components match; an absolute pattern must
/// align component-for-component.
struct PathGlob {
    absolute: bool,
    components: Vec<FnGlob>,
}

impl PathGlob {
    fn new(pattern: &str) -> Self {
        PathGlob {
            absolute: pattern.starts_with('/'),
            components: posix_parts(pattern)
                .iter()
                .map(|c| FnGlob::new(c))
                .collect(),
        }
    }

    fn matches(&self, rel: &str) -> bool {
        if self.components.is_empty() {
            // `PurePosixPath("").match(_)` raises; an empty glob matches nothing.
            return false;
        }
        if self.absolute != rel.starts_with('/') {
            return false;
        }
        let parts = posix_parts(rel);
        if self.absolute {
            if self.components.len() != parts.len() {
                return false;
            }
        } else if self.components.len() > parts.len() {
            return false;
        }
        parts
            .iter()
            .rev()
            .zip(self.components.iter().rev())
            .all(|(part, glob)| glob.is_match(part))
    }
}

/// Split a posix path/pattern into components, dropping empty segments and `.`
/// (mirrors `pathlib`'s `parse_parts`). Backslashes are literal on posix.
fn posix_parts(p: &str) -> Vec<&str> {
    p.split('/')
        .filter(|c| !c.is_empty() && *c != ".")
        .collect()
}

/// Translate an `fnmatch` glob to an anchored regex with the same match set as
/// CPython's `fnmatch.translate`. `*` becomes a greedy `.*` (CPython's atomic
/// groups only bound backtracking; the accepted language is identical and the
/// `regex` crate already matches in linear time), `?` becomes `.`, `[...]`
/// becomes a character class (`!` negation, ranges), everything else is
/// escaped. Exotic class set-operations are not reproduced — no real import
/// glob uses them.
fn fnmatch_regex(pat: &str) -> String {
    let chars: Vec<char> = pat.chars().collect();
    let n = chars.len();
    let mut out = String::from("(?s)^(?:");
    let mut i = 0;
    while i < n {
        let c = chars[i];
        i += 1;
        match c {
            '*' => out.push_str(".*"),
            '?' => out.push('.'),
            '[' => {
                let mut j = i;
                if j < n && chars[j] == '!' {
                    j += 1;
                }
                if j < n && chars[j] == ']' {
                    j += 1;
                }
                while j < n && chars[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    // No closing `]`: a literal `[`.
                    out.push_str("\\[");
                } else {
                    out.push('[');
                    let body = &chars[i..j];
                    let mut k = 0;
                    if body.first() == Some(&'!') {
                        out.push('^');
                        k = 1;
                    }
                    while k < body.len() {
                        let ch = body[k];
                        // Escape regex class metacharacters so `&&`/`~~`/nested
                        // `[` are not read as set operations; `-` stays literal
                        // so ranges keep working.
                        if matches!(ch, '\\' | ']' | '^' | '&' | '~' | '|' | '[') {
                            out.push('\\');
                        }
                        out.push(ch);
                        k += 1;
                    }
                    out.push(']');
                    i = j + 1;
                }
            }
            _ => out.push_str(&regex::escape(&c.to_string())),
        }
    }
    out.push_str(")\\z");
    out
}

impl plugin_api::ExternalPlugin for DynamicImportFallbackPluginImpl {
    fn name(&self) -> &str {
        "DynamicImportFallbackPlugin"
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        let project_root = ctx.project_root().to_string();

        // Snapshot DYNAMIC_IMPORT edges whose dst is a module node:
        // (src idx, src path, dst module fqname).
        let candidates: Vec<(usize, String, String)> = ctx
            .edges()
            .filter_map(|e| {
                if e.flags & EdgeFlags::DYNAMIC_IMPORT == 0 {
                    return None;
                }
                let dst = ctx.node(e.dst)?;
                if dst.kind != "module" {
                    return None;
                }
                let src = ctx.node(e.src)?;
                Some((e.src, src.path, dst.fqname))
            })
            .collect();
        if candidates.is_empty() {
            return Ok(());
        }

        // Apply include/exclude filters (compiled once). Zero-config => every
        // edge passes with no glob work at all.
        let allowed: Vec<(usize, String)> = if self.needs_matching() {
            let include_src: Vec<PathGlob> = self
                .include_sources
                .iter()
                .map(|p| PathGlob::new(p))
                .collect();
            let include_tgt: Vec<FnGlob> = self
                .include_targets
                .iter()
                .map(|p| FnGlob::new(p))
                .collect();
            let exclude_src: Vec<PathGlob> = self
                .exclude_sources
                .iter()
                .map(|p| PathGlob::new(p))
                .collect();
            let exclude_tgt: Vec<FnGlob> = self
                .exclude_targets
                .iter()
                .map(|p| FnGlob::new(p))
                .collect();
            candidates
                .into_iter()
                .filter_map(|(src_idx, src_path, dst_fqname)| {
                    // Source path relative to the project root, matched as a
                    // `PurePosixPath`; falls back to the raw path when it is not
                    // under the root (mirrors the original strip_prefix).
                    let rel = std::path::Path::new(&src_path)
                        .strip_prefix(&project_root)
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| src_path.clone());
                    // Include filters (if any) must all match; exclude filters
                    // (if any) must none match.
                    if !include_src.is_empty() && !include_src.iter().any(|g| g.matches(&rel)) {
                        return None;
                    }
                    if !include_tgt.is_empty()
                        && !include_tgt.iter().any(|g| g.is_match(&dst_fqname))
                    {
                        return None;
                    }
                    if exclude_src.iter().any(|g| g.matches(&rel)) {
                        return None;
                    }
                    if exclude_tgt.iter().any(|g| g.is_match(&dst_fqname)) {
                        return None;
                    }
                    Some((src_idx, dst_fqname))
                })
                .collect()
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
                let exports = self.exports_for(ctx, &dst_fqname);
                cache.insert(dst_fqname.clone(), exports);
            }
            for &export_idx in &cache[&dst_fqname] {
                ops.add_edge(src_idx, export_idx, 0);
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

pub(crate) struct ExplicitEntrypointPluginImpl {
    regexes: Vec<String>,
    str_specs: Vec<String>,
    abs_paths: Vec<String>,
}

impl plugin_api::ExternalPlugin for ExplicitEntrypointPluginImpl {
    fn name(&self) -> &str {
        "ExplicitEntrypointPlugin"
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        if self.regexes.is_empty() && self.str_specs.is_empty() && self.abs_paths.is_empty() {
            return Ok(());
        }
        let idxs = ctx.nodes_matching_specs(&self.regexes, &self.str_specs, &self.abs_paths);
        for idx in idxs {
            ops.keep_alive(idx);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProjectScriptsPlugin — project-wide port of
// `dead_cst.plugins.project_scripts`. Mark every `[project.scripts]` entry in
// `pyproject.toml` as an entrypoint. TOML is parsed in Rust via the `toml`
// crate; a missing file is a no-op.
// ---------------------------------------------------------------------------

pub(crate) struct ProjectScriptsPluginImpl {
    pyproject_path: Option<String>,
}

impl plugin_api::ExternalPlugin for ProjectScriptsPluginImpl {
    fn name(&self) -> &str {
        "ProjectScriptsPlugin"
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
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

        // Parse `[project.scripts]` (name -> "pkg.mod:func") in Rust. A parse
        // error surfaces as a `ValueError` (mirroring `tomllib.loads`); a
        // missing `[project]` / `[project.scripts]` (or a non-string value) is
        // simply skipped.
        let table: toml::Table = text.parse().map_err(|e| {
            plugin_api::PluginError::value(format!("{pyproject}: invalid TOML: {e}"))
        })?;
        let scripts: Vec<(String, String)> = table
            .get("project")
            .and_then(|project| project.get("scripts"))
            .and_then(|scripts| scripts.as_table())
            .map(|scripts| {
                scripts
                    .iter()
                    .filter_map(|(name, target)| {
                        target.as_str().map(|t| (name.clone(), t.to_string()))
                    })
                    .collect::<Vec<(String, String)>>()
            })
            .unwrap_or_default();

        for (_script_name, target) in scripts {
            let (module_part, decl_part) = match target.split_once(':') {
                Some((m, d)) => (m, d),
                None => (target.as_str(), ""),
            };
            let fqname = if decl_part.is_empty() {
                module_part.to_string()
            } else {
                format!("{module_part}.{decl_part}")
            };
            let mut target_idxs = ctx.find_declarations(&fqname);
            if target_idxs.is_empty() {
                if let Some(module_idx) = ctx.find_module(module_part) {
                    target_idxs.push(module_idx);
                }
            }
            for idx in target_idxs {
                ops.keep_alive(idx);
            }
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

fn is_test_filename(name: &str) -> bool {
    (name.starts_with("test_") && name.ends_with(".py")) || name.ends_with("_test.py")
}

fn is_test_decl(kind: &str, fqname: &str) -> bool {
    let simple = simple_name(fqname);
    (kind == "function" && simple.starts_with("test_"))
        || (kind == "class" && simple.starts_with("Test"))
}

/// The shared `test/testcase` node-flag spec declared by both the pytest and
/// unittest plugins. Identical from both, so the registry collapses it to a
/// single bit (idempotent registration); `seed + default_on` puts it in the
/// default keepalive mask whenever a test plugin is registered. Scoped to
/// *genuine* test roots (things a framework collects and runs) — the
/// conservatively-kept pytest support code (fixtures, conftest) rides the
/// separate [`fixture_flag_spec`] instead.
fn testcase_flag_spec() -> plugin_api::FlagSpec {
    plugin_api::FlagSpec {
        name: "test/testcase".to_string(),
        seed: true,
        default_on: true,
        description: "Symbol a test framework collects and runs directly: a pytest \
                      `test_*` function / `Test*` class, or a `unittest.TestCase` \
                      subclass."
            .to_string(),
    }
}

/// The pytest plugin's provisional `test/fixture` node-flag spec, stamped on
/// symbols kept alive *conservatively* — every `@pytest.fixture` and every
/// `conftest.py` decl — because their real usage (autouse / usefixtures /
/// indirect parametrization, conftest auto-loading) is not modeled as edges
/// yet. `seed + default_on` preserves today's keep-everything behavior; it is
/// kept distinct from [`testcase_flag_spec`] so the conservative blast radius
/// is measurable (`kept_alive_by_flags_only`) and can be flipped to
/// `seed: false` in isolation once that usage is traced precisely. Declared by
/// pytest only.
fn fixture_flag_spec() -> plugin_api::FlagSpec {
    plugin_api::FlagSpec {
        name: "test/fixture".to_string(),
        seed: true,
        default_on: true,
        description: "Conservatively kept alive by the pytest plugin (a @pytest.fixture \
                      or a conftest.py decl) pending precise fixture/conftest-usage \
                      modeling; provisional, flips to non-seeding once that usage is \
                      traced as edges."
            .to_string(),
    }
}

/// Topics pytest publishes per-file and resolves project-wide. Names are
/// stable (the per-file salsa cache emits by name); descriptions must stay
/// fixed (a conflicting re-registration fails the materialize).
const PYTEST_TOPIC_TESTCASE: &str = "pytest/testcase";
const PYTEST_TOPIC_FIXTURE_FLAG: &str = "pytest/fixture_flag";
const PYTEST_TOPIC_FIXTURE_NAME: &str = "pytest/fixture_name";
const PYTEST_TOPIC_TESTPARAM: &str = "pytest/testparam";

fn pytest_topic_specs() -> Vec<crate::topic_registry::TopicSpec> {
    use crate::topic_registry::TopicSpec;
    vec![
        TopicSpec {
            name: PYTEST_TOPIC_TESTCASE.to_string(),
            description: "A pytest test decl (test_* function / Test* class in a test \
                          file); resolved to a test/testcase flag."
                .to_string(),
        },
        TopicSpec {
            name: PYTEST_TOPIC_FIXTURE_FLAG.to_string(),
            description: "A decl kept alive as a fixture (a @pytest.fixture or any \
                          conftest.py top-level decl); resolved to a test/fixture flag."
                .to_string(),
        },
        TopicSpec {
            name: PYTEST_TOPIC_FIXTURE_NAME.to_string(),
            description: "A @pytest.fixture's binding name (its `name=` kwarg or simple \
                          name); the value is the binding, pinned to the fixture decl."
                .to_string(),
        },
        TopicSpec {
            name: PYTEST_TOPIC_TESTPARAM.to_string(),
            description: "One parameter name of a test function / test class, pinned to \
                          the test decl; resolved to a test→fixture edge when it names a \
                          fixture binding."
                .to_string(),
        },
    ]
}

/// Pytest, decomposed into a per-file salsa-cached fact emitter
/// ([`plugin_api::PerFilePlugin`]) and a project-wide resolve
/// ([`plugin_api::ExternalPlugin::run`]). The per-file pass observes only this
/// file's decls/decorators/params (all from the salsa-cached
/// [`crate::file_extraction::file_extraction`]) and publishes facts; the
/// project-wide pass aggregates them — building the cross-file fixture-name
/// map and matching test parameters to fixtures — with no project walk. On an
/// incremental edit, a clean file's facts are reused verbatim.
pub(crate) struct PytestPluginImpl;

impl plugin_api::PerFilePlugin for PytestPluginImpl {
    fn run_on_file(&self, file: &plugin_api::PluginFileCtx<'_>, ops: &mut plugin_api::FileOps) {
        let path = file.path_string();
        let filename = path_basename(&path);
        let nodes = file.nodes();

        if filename == "conftest.py" {
            // Every top-level function / class / variable decl is kept alive as
            // a fixture (conftest auto-loading isn't modeled as edges).
            for node in nodes.iter().skip(1) {
                if matches!(node.kind.as_str(), "function" | "class" | "variable") {
                    ops.emit_fact(PYTEST_TOPIC_FIXTURE_FLAG, Some(node.local_idx), "");
                }
            }
        } else if is_test_filename(filename) {
            // Genuine test decls → testcase; their parameter names → testparam
            // facts for the project-wide fixture match.
            let fn_params = file.function_params_by_local();
            let cls_params = file.class_method_params_by_local();
            for node in nodes.iter().skip(1) {
                if !is_test_decl(&node.kind, &node.fqname) {
                    continue;
                }
                ops.emit_fact(PYTEST_TOPIC_TESTCASE, Some(node.local_idx), "");
                let params = match node.kind.as_str() {
                    "function" => fn_params.get(&node.local_idx),
                    "class" => cls_params.get(&node.local_idx),
                    _ => None,
                };
                if let Some(names) = params {
                    for name in names {
                        ops.emit_fact(PYTEST_TOPIC_TESTPARAM, Some(node.local_idx), name.as_str());
                    }
                }
            }
        }

        // `@pytest.fixture`-decorated decls (in any file): flagged a fixture,
        // and their binding name published for the cross-file param match.
        for (local, args) in file.decorated_decls_with_args(&["pytest"], &["fixture"]) {
            ops.emit_fact(PYTEST_TOPIC_FIXTURE_FLAG, Some(local), "");
            let binding: compact_str::CompactString = match args.kwargs.get("name") {
                Some(ArgValue::Str(alias)) => alias.clone(),
                _ => nodes
                    .get(local as usize)
                    .map(|n| simple_name(&n.fqname).into())
                    .unwrap_or_default(),
            };
            ops.emit_fact(PYTEST_TOPIC_FIXTURE_NAME, Some(local), binding.as_str());
        }
    }
}

impl plugin_api::ExternalPlugin for PytestPluginImpl {
    fn name(&self) -> &str {
        "PytestPlugin"
    }

    fn declare_node_flags(&self) -> Vec<plugin_api::FlagSpec> {
        vec![testcase_flag_spec(), fixture_flag_spec()]
    }

    fn declare_topics(&self) -> Vec<crate::topic_registry::TopicSpec> {
        pytest_topic_specs()
    }

    fn per_file(&self) -> Option<&dyn plugin_api::PerFilePlugin> {
        Some(self)
    }

    fn run(
        &self,
        ctx: &plugin_api::PluginCtx<'_>,
        ops: &mut plugin_api::PluginOps,
    ) -> Result<(), plugin_api::PluginError> {
        // The bits the host allocated for our declared flags.
        let testcase_flag = ctx
            .node_flag("test/testcase")
            .expect("test/testcase is declared in declare_node_flags");
        let fixture_flag = ctx
            .node_flag("test/fixture")
            .expect("test/fixture is declared in declare_node_flags");

        // Resolve our topic handles once. A `None` means the topic collected
        // no facts this build (nothing to do for it).
        let facts = |name: &str| {
            ctx.topic(name)
                .map(|h| ctx.facts_for_topic(h))
                .unwrap_or_default()
        };

        // Flag passes — idempotent OR, so the conftest / @pytest.fixture
        // overlap that the old pass dedup-tracked is harmless here.
        for fact in facts(PYTEST_TOPIC_FIXTURE_FLAG) {
            if let Some(idx) = fact.decl_idx {
                ops.flag_decl(idx, fixture_flag);
            }
        }
        for fact in facts(PYTEST_TOPIC_TESTCASE) {
            if let Some(idx) = fact.decl_idx {
                ops.flag_decl(idx, testcase_flag);
            }
        }

        // binding name -> [fixture idx], from the per-file fixture-name facts.
        let mut fixtures_by_name: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for fact in facts(PYTEST_TOPIC_FIXTURE_NAME) {
            if let Some(idx) = fact.decl_idx {
                fixtures_by_name.entry(fact.value).or_default().push(idx);
            }
        }
        if fixtures_by_name.is_empty() {
            return Ok(());
        }

        // Parameter-name edges: each test-parameter fact whose value names a
        // fixture binding wires the test decl to that fixture.
        for fact in facts(PYTEST_TOPIC_TESTPARAM) {
            let Some(test_idx) = fact.decl_idx else {
                continue;
            };
            if let Some(fixture_idxs) = fixtures_by_name.get(&fact.value) {
                for &fixture_idx in fixture_idxs {
                    ops.add_edge(test_idx, fixture_idx, 0);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod dynamic_import_glob_tests {
    use super::{FnGlob, PathGlob};

    fn fn_match(name: &str, pat: &str) -> bool {
        FnGlob::new(pat).is_match(name)
    }

    fn path_match(rel: &str, pat: &str) -> bool {
        PathGlob::new(pat).matches(rel)
    }

    #[test]
    fn fnmatch_star_and_question() {
        assert!(fn_match("foo.py", "*.py"));
        assert!(!fn_match("foo.txt", "*.py"));
        assert!(fn_match("a", "?"));
        assert!(!fn_match("ab", "?"));
        assert!(fn_match("ab", "??"));
    }

    #[test]
    fn fnmatch_dot_is_literal() {
        // The glob `.` is a literal dot, not "any char".
        assert!(fn_match("a.b", "a.b"));
        assert!(!fn_match("axb", "a.b"));
    }

    #[test]
    fn fnmatch_prefix_glob() {
        assert!(fn_match("pkg.vendored.x", "pkg.vendored.*"));
        assert!(fn_match("pkg.vendored.sub.y", "pkg.vendored.*"));
        assert!(!fn_match("pkg.other.x", "pkg.vendored.*"));
    }

    #[test]
    fn fnmatch_char_class() {
        assert!(fn_match("a1", "a[0-9]"));
        assert!(!fn_match("aX", "a[0-9]"));
        assert!(fn_match("aX", "a[!0-9]"));
    }

    #[test]
    fn path_match_componentwise() {
        assert!(path_match("pkg/loaders/foo.py", "pkg/loaders/*.py"));
        assert!(!path_match("pkg/loaders/foo.txt", "pkg/loaders/*.py"));
        // `*` does not cross a `/`.
        assert!(!path_match("pkg/loaders/sub/foo.py", "pkg/loaders/*.py"));
        // `*` matches a whole single component.
        assert!(path_match("pkg/loaders/foo.py", "pkg/*/foo.py"));
    }

    #[test]
    fn path_match_relative_from_the_right() {
        assert!(path_match("pkg/loaders/foo.py", "loaders/*.py"));
        assert!(path_match("a/b/c.py", "*.py"));
        // A pattern longer than the path cannot match.
        assert!(!path_match("c.py", "b/c.py"));
    }

    #[test]
    fn path_match_literal_filename() {
        assert!(path_match("pkg/legacy/b.py", "pkg/legacy/b.py"));
        assert!(!path_match("pkg/legacy/a.py", "pkg/legacy/b.py"));
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::plugin_api::PLUGIN_API_EPOCH;
    use super::PLUGIN_ABI_FINGERPRINT;

    #[test]
    fn fingerprint_embeds_plugin_api_epoch() {
        // The curated-API epoch is part of the load-time gate, so bumping it
        // invalidates plugins built against an older `plugin_api`. Lock the
        // wiring: build.rs must keep the `api<N>` segment in the fingerprint.
        let seg = format!("|api{PLUGIN_API_EPOCH}|");
        assert!(
            PLUGIN_ABI_FINGERPRINT.contains(&seg),
            "fingerprint {PLUGIN_ABI_FINGERPRINT:?} missing segment {seg:?}"
        );
    }
}
