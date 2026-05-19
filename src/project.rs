//! `Project`, `BuildOutputs`, `ProgressBars`, `ProjectContext`, and the
//! `build_project_graph` pipeline entrypoint. This module owns the
//! Salsa-backed analysis context that the rest of the crate operates on.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use ruff_db::system::{OsSystem, SystemPathBuf};
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::{Expr, Stmt};
use ruff_text_size::{Ranged, TextRange};
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::RangedValue;
use ty_project::{Db as ProjectDb, ProjectDatabase, ProjectMetadata};
use ty_python_core::definition::DefinitionState;
use ty_python_core::program::UseDefaultStrategy;
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;

use crate::builder::{apply_graph_op, bfs, lookup_idx, not_materialized, Direction, GraphBuilder};
use crate::graph::{
    DeclIndex, GlobalsByName, ImportSpec, LiveDeclIndex, MainBlock, NativeGraph, StarReexports,
    SymbolNode,
};
use crate::helpers::{
    call_callee_matches_var, class_body_defines_method, collect_all_imports_local,
    collect_module_imports_local, decorators_match_imports, extract_call_args_kwargs,
    file_decl_sites, file_path_string, find_main_block_range, find_subclass_indices_via_refs,
    is_dunder_name, locate_class_def, locate_class_seed, matched_call_target,
    owner_idx_for_stmt_with, range_key, rel_path, top_level_assign_to_name, AttrCallFinder,
    CallArgs, FactoryCallFinder, StringArgCallFinder, NODE_FLAG_ENTRYPOINT,
};
use crate::ingest::{
    build_dist_lookup, emit_import_edges, emit_module_hierarchy, emit_reference_edges,
    ingest_decls, string_literal_list,
};
use crate::query::{
    _compile_path_regex, _contains_any_identifier, _contains_identifier, par_scan_files,
    QueryBuilder,
};

/// A ty-backed analysis project with explicitly-injected configuration.
#[pyclass(unsendable)]
pub(crate) struct Project {
    pub(crate) db: ProjectDatabase,
}

#[pymethods]
impl Project {
    #[new]
    #[pyo3(signature = (
        root,
        *,
        src_roots = None,
        extra_paths = None,
        python_env = None,
        python_version = None,
        typeshed = None,
    ))]
    pub(crate) fn new(
        root: &str,
        src_roots: Option<Vec<String>>,
        extra_paths: Option<Vec<String>>,
        python_env: Option<&str>,
        python_version: Option<&str>,
        typeshed: Option<&str>,
    ) -> PyResult<Self> {
        let db = make_db(
            root,
            src_roots,
            extra_paths,
            python_env,
            python_version,
            typeshed,
        )?;
        Ok(Self { db })
    }

    /// Build the project-wide symbol graph.
    pub(crate) fn build(&self, py: Python<'_>) -> PyResult<NativeGraph> {
        let outputs = build_project_graph(py, &self.db, false)?;
        Ok(NativeGraph {
            nodes: outputs.builder.nodes,
            edges: outputs.builder.edges,
        })
    }
}

/// All state produced by one project-wide build pass.
///
/// Owned by [`ProjectContext`] across the materialize call so plugin
/// queries can re-read indices (e.g. live_decls, global_index) without
/// having to re-derive them from the parsed modules.
pub(crate) struct BuildOutputs {
    pub(crate) builder: GraphBuilder,
    pub(crate) project_files: Vec<File>,
    pub(crate) global_index: DeclIndex,
    /// `file_path_string(file) -> File` so seed lookups don't have to
    /// linear-scan `project_files`. Populated alongside ingest.
    pub(crate) path_to_file: HashMap<String, File>,
    /// `(file, class_target_range_key) -> class node idx`. Lets
    /// `find_subclasses_of` map ty's `TypeHierarchyClass.selection_range`
    /// back to a graph node in O(1).
    pub(crate) class_by_selection: HashMap<(File, (u32, u32)), usize>,
    /// `file -> module node idx`. Lets `find_main_blocks` reach the
    /// file's module node without a linear scan over `builder.nodes`.
    pub(crate) module_nodes_by_file: HashMap<File, usize>,
    /// `(file, target_range_key) -> node idx`. Sister of
    /// ``class_by_selection`` but for every top-level decl ingest minted
    /// (function / class / variable / import). Lets ``find_decorated_decls``
    /// and the dispatch-app queries map an AST node's target range to a
    /// graph node in O(1) instead of scanning the full ``global_index``.
    pub(crate) decl_by_name_range: HashMap<(File, (u32, u32)), usize>,
    /// `decl_fqname -> [node idx]`. Lets ``find_declarations`` answer
    /// in O(parts) instead of O(parts × all_nodes). Multiple entries
    /// per fqname arise from try/except rebinds and conditional
    /// re-imports.
    pub(crate) decl_by_fqname: HashMap<String, Vec<usize>>,
    /// `module_fqname -> node idx`. Lets ``find_module`` answer in
    /// O(1) instead of scanning all nodes.
    pub(crate) module_by_fqname: HashMap<String, usize>,
}

/// Run the three build phases (ingest → hierarchy+imports → references)
/// and return every index the plugin queries need.
/// Indicatif bars for the three per-file phases of ``build_project_graph``.
///
/// When ``show_progress`` is true, draws to stderr with one bar per phase.
/// indicatif auto-downgrades to a hidden target on non-TTY stderr, so the
/// CLI can always pass ``True`` without checking ``isatty`` itself.
///
/// When ``show_progress`` is false, every bar uses ``ProgressDrawTarget::hidden()``
/// so ``inc(1)`` / ``finish`` are cheap no-ops — no allocation of a render
/// thread, no stderr writes.
pub(crate) struct ProgressBars {
    pub(crate) ingest: ProgressBar,
    pub(crate) imports: ProgressBar,
    pub(crate) references: ProgressBar,
}

impl ProgressBars {
    fn new(show_progress: bool, total_files: u64) -> Self {
        let multi = MultiProgress::with_draw_target(if show_progress {
            ProgressDrawTarget::stderr()
        } else {
            ProgressDrawTarget::hidden()
        });
        let style = ProgressStyle::with_template(
            "  {prefix:<10} [{bar:30.cyan/blue}] {pos:>6}/{len:<6} {msg}",
        )
        .expect("static template parses")
        .progress_chars("=> ");
        let mk = |prefix: &'static str| {
            let bar = multi.add(ProgressBar::new(total_files));
            bar.set_style(style.clone());
            bar.set_prefix(prefix);
            bar
        };
        Self {
            ingest: mk("ingest"),
            imports: mk("imports"),
            references: mk("refs"),
        }
    }

    /// Single bar for the plugin pass — sits on its own draw target since
    /// it's created after ``build_project_graph`` returns (the file-pass
    /// bars are already finished by then).
    fn plugin_bar(show_progress: bool, total_plugins: u64) -> ProgressBar {
        let bar = ProgressBar::with_draw_target(
            Some(total_plugins),
            if show_progress {
                ProgressDrawTarget::stderr()
            } else {
                ProgressDrawTarget::hidden()
            },
        );
        bar.set_style(
            ProgressStyle::with_template(
                "  {prefix:<10} [{bar:30.cyan/blue}] {pos:>6}/{len:<6} {msg}",
            )
            .expect("static template parses")
            .progress_chars("=> "),
        );
        bar.set_prefix("plugins");
        bar
    }
}

pub(crate) fn build_project_graph(
    py: Python<'_>,
    db: &ProjectDatabase,
    show_progress: bool,
) -> PyResult<BuildOutputs> {
    let timing = std::env::var_os("DEAD_CST_TIMING").is_some();
    let mut builder = GraphBuilder::new();
    let mut global_index: DeclIndex = HashMap::new();
    let mut module_nodes: HashMap<File, usize> = HashMap::new();
    let mut alias_imports: HashMap<usize, ImportSpec> = HashMap::new();
    let mut live_decls: LiveDeclIndex = HashMap::new();
    let mut globals_by_name: GlobalsByName = HashMap::new();
    let mut star_reexports: StarReexports = HashMap::new();
    let mut class_by_selection: HashMap<(File, (u32, u32)), usize> = HashMap::new();
    let mut decl_by_name_range: HashMap<(File, (u32, u32)), usize> = HashMap::new();

    let t0 = std::time::Instant::now();
    let project_files: Vec<File> = (&db.project().files(db)).into_iter().collect();
    let mut path_to_file: HashMap<String, File> = HashMap::with_capacity(project_files.len());
    for &file in &project_files {
        path_to_file.insert(file_path_string(db, file), file);
    }
    // Peer ``.pyi`` files: register the ``.pyi -> .py twin`` mapping
    // so import resolution can fall back when the stub lookup misses
    // and so ``file_default_flags`` can tell peer .pyi (decls reach
    // their runtime via the fallback) from stub-only .pyi (decls
    // need an artificial ENTRYPOINT to stay alive — native extension
    // stubs and protobuf-style _pb2.pyi shapes).
    let py_files_by_stem: HashMap<String, File> = project_files
        .iter()
        .filter_map(|&f| {
            file_path_string(db, f)
                .strip_suffix(".py")
                .map(|stem| (stem.to_string(), f))
        })
        .collect();
    for &f in &project_files {
        let path = file_path_string(db, f);
        if let Some(stem) = path.strip_suffix(".pyi") {
            if let Some(&py_twin) = py_files_by_stem.get(stem) {
                builder.peer_pyi_to_py.insert(f, py_twin);
            }
        }
    }
    let t_enum = t0.elapsed();
    let t1 = std::time::Instant::now();
    // Two-pass ingest so the per-decl ``.pyi`` stub flagging in
    // ``ingest_decls`` has the .py twin's ``globals_by_name`` entries
    // to probe. Pass 1 = everything that isn't a .pyi; pass 2 = .pyi.
    // The split doesn't change the graph for non-peer files; it's
    // ordering for the peer-stub flag-check only.
    let progress = ProgressBars::new(show_progress, project_files.len() as u64);
    for file in &project_files {
        if file_path_string(db, *file).ends_with(".pyi") {
            continue;
        }
        ingest_decls(
            py,
            db,
            *file,
            &mut builder,
            &mut global_index,
            &mut module_nodes,
            &mut alias_imports,
            &mut live_decls,
            &mut globals_by_name,
            &mut star_reexports,
            &mut class_by_selection,
            &mut decl_by_name_range,
        )?;
        progress.ingest.inc(1);
    }
    for file in &project_files {
        if !file_path_string(db, *file).ends_with(".pyi") {
            continue;
        }
        ingest_decls(
            py,
            db,
            *file,
            &mut builder,
            &mut global_index,
            &mut module_nodes,
            &mut alias_imports,
            &mut live_decls,
            &mut globals_by_name,
            &mut star_reexports,
            &mut class_by_selection,
            &mut decl_by_name_range,
        )?;
        progress.ingest.inc(1);
    }
    progress.ingest.finish_and_clear();
    // Per-decl ``pyi_decl -> py_decl`` edges for each peer .pyi whose
    // matching .py defines the same simple name. The edge documents
    // the stub-runtime relationship in the graph: consumers that ty
    // resolved through the stub get reachability into the runtime
    // decl via ``alias -> pyi_decl -> py_decl`` rather than stopping
    // at the stub. Stub-only decls (no matching .py decl) are
    // separately kept alive by the per-decl ENTRYPOINT flag set in
    // ``ingest_decls``.
    let peer_stubs: Vec<(File, File)> = builder
        .peer_pyi_to_py
        .iter()
        .map(|(&pyi, &py)| (pyi, py))
        .collect();
    for (pyi_file, py_twin) in peer_stubs {
        let pyi_decls: Vec<(String, usize)> = globals_by_name
            .iter()
            .filter(|((file, _), _)| *file == pyi_file)
            .flat_map(|((_, name), idxs)| idxs.iter().map(move |&idx| (name.clone(), idx)))
            .collect();
        for (name, pyi_idx) in pyi_decls {
            if let Some(py_idxs) = globals_by_name.get(&(py_twin, name)) {
                for &py_idx in py_idxs {
                    builder.add_edge(pyi_idx, py_idx, 0);
                }
            }
        }
    }
    let t_phase1 = t1.elapsed();
    // Walk every site-packages search path's ``*.dist-info/`` to build
    // the file -> canonical-dist-name map. Cheap (one read_dir + a
    // couple of read_to_string per installed dist), runs once per
    // ``materialize`` call. Both Phase 2 (alias minting) and Phase 3
    // (use-site emit_upstream) need the same map.
    let dist_lookup = build_dist_lookup(db);
    let t2 = std::time::Instant::now();
    for file in &project_files {
        emit_module_hierarchy(db, *file, &module_nodes, &mut builder);
        emit_import_edges(
            py,
            db,
            *file,
            &mut builder,
            &mut global_index,
            &mut module_nodes,
            &globals_by_name,
            &star_reexports,
            &dist_lookup,
        )?;
        progress.imports.inc(1);
    }
    progress.imports.finish_and_clear();
    let t_phase2 = t2.elapsed();
    let t3 = std::time::Instant::now();
    for file in &project_files {
        emit_reference_edges(
            db,
            *file,
            &global_index,
            &module_nodes,
            &alias_imports,
            &live_decls,
            &dist_lookup,
            &mut builder,
        );
        progress.references.inc(1);
    }
    progress.references.finish_and_clear();
    let t_phase3 = t3.elapsed();
    let t4 = std::time::Instant::now();
    let (decl_by_fqname, module_by_fqname) = build_fqname_indices(py, &builder);
    let t_fqname = t4.elapsed();
    if timing {
        eprintln!(
            "[dead-cst-timing] files={} nodes={} edges={} enum={:?} phase1={:?} phase2={:?} phase3={:?} fqname={:?} total={:?}",
            project_files.len(),
            builder.nodes.len(),
            builder.edges.len(),
            t_enum,
            t_phase1,
            t_phase2,
            t_phase3,
            t_fqname,
            t0.elapsed(),
        );
    }
    Ok(BuildOutputs {
        builder,
        project_files,
        global_index,
        path_to_file,
        class_by_selection,
        module_nodes_by_file: module_nodes,
        decl_by_name_range,
        decl_by_fqname,
        module_by_fqname,
    })
}

/// Pre-build the fqname -> idx maps used by ``find_declarations`` and
/// ``find_module``. One pass over interned nodes; module entries are
/// 1:1 (one module node per fqname) while decl entries can have
/// multiple binders for the same fqname (try/except rebind etc.).
pub(crate) fn build_fqname_indices(
    py: Python<'_>,
    builder: &GraphBuilder,
) -> (HashMap<String, Vec<usize>>, HashMap<String, usize>) {
    let mut decls: HashMap<String, Vec<usize>> = HashMap::new();
    let mut modules: HashMap<String, usize> = HashMap::new();
    for (idx, node_py) in builder.nodes.iter().enumerate() {
        let node = node_py.borrow(py);
        match node.kind {
            "module" => {
                modules.insert(node.fqname.clone(), idx);
            }
            "function" | "class" | "variable" | "import" => {
                decls.entry(node.fqname.clone()).or_default().push(idx);
            }
            _ => {}
        }
    }
    (decls, modules)
}

/// Plugin-aware project graph builder.
///
/// Python instantiates a `ProjectContext`, registers Python plugins via
/// `add_plugin`, then calls `materialize()`. `materialize` runs the
/// project-wide build in rust, then for each registered plugin calls
/// `plugin.run(ctx)` back into Python with `ctx` set to the same
/// `ProjectContext` instance. Plugins yield `GraphOp` values
/// (``AddNode`` / ``AddEdge`` / ``AddEntrypoint``) that we apply to
/// the graph; the rust `find_*` methods listed below answer queries
/// against the graph in-progress.
///
/// Queries are answered from ty's semantic index: subclass closure goes
/// through `type_hierarchy_subtypes`, method-defines walks each class's
/// `DefinitionKind::Class`, module dunders scan global-scope variable
/// nodes, and comment patterns walk the parser's `Tokens` stream.
#[pyclass(unsendable)]
pub(crate) struct ProjectContext {
    pub(crate) db: ProjectDatabase,
    /// Absolute path of the project root, echoed back to Python via the
    /// :attr:`project_root` getter. Plugins use it to compute paths
    /// relative to the project (e.g. ``ExplicitEntrypointPlugin`` matching
    /// path specs).
    pub(crate) root: String,
    pub(crate) plugins: Vec<PyObject>,
    /// Populated by `materialize` before plugins run. `None` outside a
    /// materialize call — `apply_graph_op` / queries assume it's
    /// `Some` and error if a plugin (incorrectly) caches the ctx and
    /// uses it after materialize returns.
    pub(crate) outputs: RefCell<Option<BuildOutputs>>,
    /// Compiled regexes keyed by the source pattern. Plugins call
    /// :meth:`decls_matching_name` / :meth:`find_comment_patterns`
    /// repeatedly across files with the same pattern, so caching keeps
    /// us off the regex compiler in the hot path.
    pub(crate) regex_cache: RefCell<HashMap<String, regex::Regex>>,
    /// When true, ``materialize`` and ``build_project_graph`` draw
    /// indicatif progress bars to stderr. Off by default (library use);
    /// the CLI's ``dead-cst analyze`` flips it on. indicatif itself
    /// downgrades to a hidden draw target when stderr isn't a TTY, so
    /// passing ``True`` is safe in CI / pipes.
    pub(crate) show_progress: bool,
}

#[pymethods]
impl ProjectContext {
    #[new]
    #[pyo3(signature = (
        root,
        *,
        src_roots = None,
        extra_paths = None,
        python_env = None,
        python_version = None,
        typeshed = None,
        show_progress = false,
    ))]
    pub(crate) fn new(
        root: &str,
        src_roots: Option<Vec<String>>,
        extra_paths: Option<Vec<String>>,
        python_env: Option<&str>,
        python_version: Option<&str>,
        typeshed: Option<&str>,
        show_progress: bool,
    ) -> PyResult<Self> {
        let db = make_db(
            root,
            src_roots,
            extra_paths,
            python_env,
            python_version,
            typeshed,
        )?;
        Ok(Self {
            db,
            root: root.to_string(),
            plugins: Vec::new(),
            outputs: RefCell::new(None),
            regex_cache: RefCell::new(HashMap::new()),
            show_progress,
        })
    }

    /// Absolute project root passed at construction.
    #[getter]
    pub(crate) fn project_root(&self) -> &str {
        &self.root
    }

    /// Register a Python plugin. Order of registration is order of
    /// invocation during `materialize`.
    pub(crate) fn add_plugin(&mut self, plugin: PyObject) {
        self.plugins.push(plugin);
    }

    /// Open a chainable query builder against this context.
    ///
    /// Equivalent to the top-level :func:`query` function; both return
    /// a :class:`QueryBuilder` that can chain into
    /// :class:`DecoratorQuery` / :class:`ConstructionQuery` /
    /// :class:`CallQuery`. See the result-type docstrings for the
    /// predicate vocabulary.
    pub(crate) fn query(slf: Py<Self>, _py: Python<'_>) -> QueryBuilder {
        QueryBuilder { ctx: slf }
    }

    /// Build the project-wide graph, run each plugin's `run(ctx)`,
    /// then snapshot the final state.
    ///
    /// Borrows are released between phases so plugin `run` methods can
    /// re-enter queries through the same ctx without aliasing violations.
    pub(crate) fn materialize(slf: Py<Self>, py: Python<'_>) -> PyResult<NativeGraph> {
        let show_progress = slf.borrow(py).show_progress;
        {
            let this = slf.borrow(py);
            let outputs = build_project_graph(py, &this.db, show_progress)?;
            *this.outputs.borrow_mut() = Some(outputs);
        }

        let plugins: Vec<PyObject> = slf
            .borrow(py)
            .plugins
            .iter()
            .map(|p| p.clone_ref(py))
            .collect();
        let plugin_bar = ProgressBars::plugin_bar(show_progress, plugins.len() as u64);
        for plugin in &plugins {
            // ``plugin.run(ctx)`` yields ``GraphOp`` values; we apply
            // each as it comes off the iterator. The plugin can run
            // queries against ``ctx`` mid-iteration since each
            // ``apply_graph_op`` call releases its borrows before
            // returning control to the generator. ``None`` (a regular
            // function that ran to completion without yielding) is
            // allowed for plugins with nothing to do.
            let plugin_name: String = plugin
                .bind(py)
                .getattr("name")
                .ok()
                .and_then(|n| n.extract().ok())
                .unwrap_or_else(|| "<unnamed>".to_string());
            plugin_bar.set_message(plugin_name);
            let result = plugin.bind(py).call_method1("run", (slf.clone_ref(py),))?;
            if !result.is_none() {
                for item in result.iter()? {
                    let op = item?;
                    apply_graph_op(&slf, py, &op)?;
                }
            }
            plugin_bar.inc(1);
        }
        plugin_bar.finish_and_clear();

        // Keep ``outputs`` alive past materialize so post-materialize
        // queries (``descendants`` / ``ancestors`` / ``reachable``) still
        // see the graph. Snapshot a fresh ``NativeGraph`` from the
        // builder's interned node + edge vecs; the originals stay put.
        let this = slf.borrow(py);
        let outputs_ref = this.outputs.borrow();
        let outputs = outputs_ref
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ProjectContext lost its outputs"))?;
        Ok(NativeGraph {
            nodes: outputs
                .builder
                .nodes
                .iter()
                .map(|n| n.clone_ref(py))
                .collect(),
            edges: outputs.builder.edges.clone(),
        })
    }
}

// ----- Queries (rust-only, exposed to Python via the chainable QueryBuilder) -

impl ProjectContext {
    /// Return every top-level variable node whose name matches `__xxx__`.
    ///
    /// Pure scan over already-interned nodes — no ty re-query needed —
    /// because the visitor's decl pass already minted one node per
    /// global-scope variable binding.
    pub(crate) fn find_module_dunders(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module_dunders"))?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if node.kind != "variable" {
                continue;
            }
            if is_dunder_name(&node.fqname) {
                out.push(node_py.clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Return every interned node whose path or fqname matches at
    /// least one of the supplied specs.
    ///
    /// Folds the per-node match loop down into a single rust pass for
    /// callers (like `ExplicitEntrypointPlugin`) that would otherwise
    /// pay an FFI hop + `pathlib.Path` allocation per node. Specs are
    /// pre-classified Python-side:
    ///
    /// * `regexes` — pattern strings applied to the path *relative to
    ///   `project_root`* (or to the absolute path when the node lives
    ///   outside `project_root`). Each pattern is anchored at the
    ///   start of input, mirroring Python's `re.Pattern.match()`.
    /// * `str_specs` — exact equality match against the relative path
    ///   OR the node's fqname.
    /// * `abs_paths` — exact equality match against the node's
    ///   absolute path.
    pub(crate) fn find_nodes_matching_specs(
        &self,
        py: Python<'_>,
        project_root: &str,
        regexes: Vec<String>,
        str_specs: Vec<String>,
        abs_paths: Vec<String>,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let compiled: Vec<regex::Regex> = regexes
            .iter()
            .map(|p| {
                // `^`-anchor for `re.match` semantics. Idempotent if
                // the caller already supplied a leading `^`.
                let anchored = if p.starts_with('^') {
                    p.clone()
                } else {
                    format!("^{p}")
                };
                regex::Regex::new(&anchored)
                    .map_err(|e| PyValueError::new_err(format!("invalid regex {p:?}: {e}")))
            })
            .collect::<PyResult<_>>()?;
        let str_set: std::collections::HashSet<&str> =
            str_specs.iter().map(String::as_str).collect();
        let abs_set: std::collections::HashSet<&str> =
            abs_paths.iter().map(String::as_str).collect();

        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_nodes_matching_specs"))?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            let path = node.path.as_str();
            if abs_set.contains(path) {
                out.push(node_py.clone_ref(py));
                continue;
            }
            // Mirror Python's ``path.relative_to(root)`` ⇒ ``str(...)``
            // with the ``except ValueError: rel = node.path`` fallback.
            let rel = path
                .strip_prefix(project_root)
                .map(|s| s.trim_start_matches(['/', '\\']))
                .unwrap_or(path);
            if str_set.contains(rel) || str_set.contains(node.fqname.as_str()) {
                out.push(node_py.clone_ref(py));
                continue;
            }
            if compiled.iter().any(|r| r.is_match(rel)) {
                out.push(node_py.clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Return every import-kind node whose upstream `module` matches.
    ///
    /// Covers both `import <module_name>` and
    /// `from <module_name> import ...` styles — both bind import-kind
    /// nodes whose `Import.module` is the absolute dotted name. Star
    /// reexports synthesized from `from <module_name> import *` are
    /// also included.
    pub(crate) fn find_imports_of(
        &self,
        py: Python<'_>,
        module_name: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_imports_of"))?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if node.kind != "import" {
                continue;
            }
            let Some(import_py) = node.imports.as_ref() else {
                continue;
            };
            if import_py.borrow(py).module == module_name {
                out.push(node_py.clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Return every declaration (function / class / variable / import)
    /// whose fully qualified name matches ``fqname``, walking back
    /// through dotted segments to find the enclosing top-level decl
    /// when the exact name doesn't match.
    ///
    /// ``pkg.lib.Cls.method`` returns ``pkg.lib.Cls`` because methods
    /// aren't represented as their own graph nodes — same rule the
    /// libcst :func:`find_declarations` follows. Modules are never
    /// returned; use :meth:`find_module` for that.
    pub(crate) fn find_declarations(
        &self,
        py: Python<'_>,
        fqname: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_declarations"))?;
        // Try exact match first, then strip trailing segments.
        let mut prefix = fqname;
        loop {
            if let Some(idxs) = outputs.decl_by_fqname.get(prefix) {
                return Ok(idxs
                    .iter()
                    .map(|&i| outputs.builder.nodes[i].clone_ref(py))
                    .collect());
            }
            match prefix.rsplit_once('.') {
                Some((parent, _)) => prefix = parent,
                None => return Ok(Vec::new()),
            }
        }
    }

    /// Return the module node for the given dotted fqname, if it
    /// exists in the project graph.
    pub(crate) fn find_module(
        &self,
        py: Python<'_>,
        fqname: &str,
    ) -> PyResult<Option<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module"))?;
        Ok(outputs
            .module_by_fqname
            .get(fqname)
            .map(|&idx| outputs.builder.nodes[idx].clone_ref(py)))
    }
}

#[pymethods]
impl ProjectContext {
    /// Return the module node owning ``path``, if any. O(1) — backed
    /// by the same ``module_nodes_by_file`` index `find_main_blocks`
    /// uses, so plugins don't have to scan ``nodes()`` per call.
    pub(crate) fn module_for(
        &self,
        py: Python<'_>,
        path: &str,
    ) -> PyResult<Option<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("module_for"))?;
        let Some(&file) = outputs.path_to_file.get(path) else {
            return Ok(None);
        };
        let Some(&idx) = outputs.module_nodes_by_file.get(&file) else {
            return Ok(None);
        };
        Ok(Some(outputs.builder.nodes[idx].clone_ref(py)))
    }

    /// Resolve a dotted FQN to either a declaration or a module node.
    ///
    /// Tries an exact decl match first, then an exact module match,
    /// then walks back through dotted segments looking for an enclosing
    /// decl (``pkg.lib.Cls.method`` resolves to ``pkg.lib.Cls`` because
    /// methods don't get their own graph nodes). Returns ``None`` when
    /// the fqname can't be found anywhere — never raises.
    pub(crate) fn resolve(&self, py: Python<'_>, fqname: &str) -> PyResult<Option<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("resolve"))?;
        let mut prefix = fqname;
        loop {
            if let Some(idxs) = outputs.decl_by_fqname.get(prefix) {
                if let Some(&idx) = idxs.first() {
                    return Ok(Some(outputs.builder.nodes[idx].clone_ref(py)));
                }
            }
            if let Some(&idx) = outputs.module_by_fqname.get(prefix) {
                return Ok(Some(outputs.builder.nodes[idx].clone_ref(py)));
            }
            match prefix.rsplit_once('.') {
                Some((parent, _)) => prefix = parent,
                None => return Ok(None),
            }
        }
    }

    /// Return the module node + every transitive decl whose fqname
    /// lives under ``module_fqn``.
    ///
    /// Models ``importlib.import_module(module_fqn)``: the module's
    /// whole top-level surface plus everything its submodules expose.
    /// Empty list when ``module_fqn`` doesn't resolve to a project
    /// module.
    pub(crate) fn module_surface(
        &self,
        py: Python<'_>,
        module_fqn: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("module_surface"))?;
        let Some(&module_idx) = outputs.module_by_fqname.get(module_fqn) else {
            return Ok(Vec::new());
        };
        let mut out = vec![outputs.builder.nodes[module_idx].clone_ref(py)];
        let prefix = format!("{module_fqn}.");
        for (fqname, &idx) in &outputs.module_by_fqname {
            if fqname.starts_with(&prefix) {
                out.push(outputs.builder.nodes[idx].clone_ref(py));
            }
        }
        for (fqname, idxs) in &outputs.decl_by_fqname {
            if fqname.starts_with(&prefix) {
                for &idx in idxs {
                    out.push(outputs.builder.nodes[idx].clone_ref(py));
                }
            }
        }
        Ok(out)
    }
}

impl ProjectContext {
    /// Return ``module_fqn``'s immediate top-level decls — every
    /// function / class / variable / import bound at its module scope.
    ///
    /// Models ``from module_fqn import *``: only the names that
    /// statement would bind into the importing scope. Unlike
    /// :meth:`module_surface`, submodules and their decls are
    /// excluded — a `from p.functions import *` doesn't pull in
    /// `p.functions.sub.x`. Empty list when ``module_fqn`` doesn't
    /// resolve to a project module.
    pub(crate) fn find_module_top_level_decls(
        &self,
        py: Python<'_>,
        module_fqn: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module_top_level_decls"))?;
        if !outputs.module_by_fqname.contains_key(module_fqn) {
            return Ok(Vec::new());
        }
        let prefix = format!("{module_fqn}.");
        let mut out = Vec::new();
        for (fqname, idxs) in &outputs.decl_by_fqname {
            let Some(rest) = fqname.strip_prefix(&prefix) else {
                continue;
            };
            // Skip transitive decls (`pkg.mod.sub.x` under `pkg.mod`).
            if rest.contains('.') {
                continue;
            }
            for &idx in idxs {
                out.push(outputs.builder.nodes[idx].clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Return the decls listed in ``module_fqn``'s ``__all__``, or
    /// ``None`` when the module doesn't declare ``__all__``.
    ///
    /// Composed on top of :meth:`find_literal_list_entries`: read the
    /// string entries from ``__all__``'s RHS, then resolve each name
    /// against ``module_fqn``'s scope via :attr:`decl_by_fqname`.
    /// Names that don't resolve in the module's global scope are
    /// skipped silently (``__all__ = ["missing"]`` is a runtime error
    /// at import time, not a static dep).
    ///
    /// The distinction between "no ``__all__``" (``None``) and "empty
    /// / unresolvable ``__all__``" (``Some([])``) matters: callers
    /// that want CPython's ``from X import *`` semantics should fall
    /// back to the non-underscore decl list only in the ``None``
    /// case.
    pub(crate) fn find_module_dunder_all_exports(
        &self,
        py: Python<'_>,
        module_fqn: &str,
    ) -> PyResult<Option<Vec<Py<SymbolNode>>>> {
        let all_fqn = format!("{module_fqn}.__all__");
        let Some(entries) = self.find_literal_list_entries(&all_fqn)? else {
            return Ok(None);
        };
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module_dunder_all_exports"))?;
        let mut out: Vec<Py<SymbolNode>> = Vec::new();
        for entry in entries {
            let entry_fqn = format!("{module_fqn}.{entry}");
            if let Some(idxs) = outputs.decl_by_fqname.get(&entry_fqn) {
                for &idx in idxs {
                    out.push(outputs.builder.nodes[idx].clone_ref(py));
                }
            }
        }
        Ok(Some(out))
    }

    /// Read the literal-list value of a top-level variable assignment
    /// (``X = ["a", "b"]`` / ``X: tuple[str, ...] = ("a", "b")``) and
    /// return the entries as owned strings.
    ///
    /// Returns ``None`` when the variable isn't found, when its
    /// assignment value isn't a list / tuple of string literals, or
    /// when any element is a non-literal (e.g. ``[*BASE, "c"]``).
    /// Each declaration of the variable that satisfies the pattern
    /// contributes its entries; multiple decls (e.g. conditional
    /// rebinding) get concatenated in declaration order.
    ///
    /// This is the targeted read the ``LiteralListPlugin`` uses to
    /// stay independent of the visitor's ``__all__``-only string-list
    /// edge emission.
    pub(crate) fn find_literal_list_entries(&self, var_fqn: &str) -> PyResult<Option<Vec<String>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_literal_list_entries"))?;
        let Some(idxs) = outputs.decl_by_fqname.get(var_fqn) else {
            return Ok(None);
        };
        let bare_name = var_fqn.rsplit('.').next().unwrap_or("");
        if bare_name.is_empty() {
            return Ok(None);
        }
        let mut out: Vec<String> = Vec::new();
        let mut found_any = false;
        for &idx in idxs {
            let path =
                pyo3::Python::with_gil(|py| outputs.builder.nodes[idx].borrow(py).path.clone());
            let Some(&file) = outputs.path_to_file.get(&path) else {
                continue;
            };
            let source = source_text(&self.db, file);
            let parsed = parsed_module(&self.db, file).load(&self.db);
            for stmt in &parsed.syntax().body {
                let Some((target_range, value)) = top_level_assign_to_name(stmt) else {
                    continue;
                };
                let start: usize = target_range.start().to_usize();
                let end: usize = target_range.end().to_usize();
                if &source[start..end] != bare_name {
                    continue;
                }
                let Some(entries) = string_literal_list(value) else {
                    continue;
                };
                found_any = true;
                out.extend(entries.into_iter().map(String::from));
            }
        }
        if found_any {
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }
}

#[pymethods]
impl ProjectContext {
    /// Every node whose ``path`` starts with the given prefix.
    pub(crate) fn decls_under(
        &self,
        py: Python<'_>,
        path_prefix: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("decls_under"))?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .filter(|n| n.borrow(py).path.starts_with(path_prefix))
            .map(|n| n.clone_ref(py))
            .collect())
    }

    /// Every node whose ``path`` contains ``substring`` anywhere.
    /// Useful for path-pattern plugins (``alembic/versions/``, ``.ignore.py``).
    pub(crate) fn decls_matching(
        &self,
        py: Python<'_>,
        substring: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("decls_matching"))?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .filter(|n| n.borrow(py).path.contains(substring))
            .map(|n| n.clone_ref(py))
            .collect())
    }

    /// Every top-level decl whose simple name matches ``regex``.
    /// Fills the gap the screenshot's API doesn't cover — needed by
    /// :class:`ModuleDundersPlugin` (``__xxx__`` names),
    /// :class:`PytestPlugin` (``test_*`` / ``Test*``), etc.
    pub(crate) fn decls_matching_name(
        &self,
        py: Python<'_>,
        pattern: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let regex = self.compile_regex(pattern)?;
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("decls_matching_name"))?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if !matches!(
                node.kind,
                "function" | "class" | "variable" | "import" | "type_alias"
            ) {
                continue;
            }
            let simple = node.fqname.rsplit('.').next().unwrap_or("");
            if regex.is_match(simple) {
                out.push(node_py.clone_ref(py));
            }
        }
        Ok(out)
    }

    /// Forward closure: every node reachable from ``root`` by following
    /// graph edges. ``skip_flags`` filters out edges whose flag mask
    /// matches (pass ``EdgeFlags.DEAD_BRANCH.value`` to compute strict
    /// reachability excluding dead branches).
    #[pyo3(signature = (root, *, skip_flags = 0))]
    pub(crate) fn descendants(
        &self,
        py: Python<'_>,
        root: &SymbolNode,
        skip_flags: u32,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("descendants"))?;
        let root_idx = lookup_idx(&outputs.builder, root, "root")?;
        Ok(
            bfs(&outputs.builder, [root_idx], Direction::Forward, skip_flags)
                .into_iter()
                .map(|i| outputs.builder.nodes[i].clone_ref(py))
                .collect(),
        )
    }

    /// Reverse closure: every node that can reach ``decl`` by following
    /// graph edges. Used for ``why-alive`` and blast-radius scoping.
    #[pyo3(signature = (decl, *, skip_flags = 0))]
    pub(crate) fn ancestors(
        &self,
        py: Python<'_>,
        decl: &SymbolNode,
        skip_flags: u32,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("ancestors"))?;
        let idx = lookup_idx(&outputs.builder, decl, "decl")?;
        Ok(bfs(&outputs.builder, [idx], Direction::Reverse, skip_flags)
            .into_iter()
            .map(|i| outputs.builder.nodes[i].clone_ref(py))
            .collect())
    }

    /// Forward closure from every node carrying any bit in
    /// ``seed_flags`` (defaults to ``NODE_FLAG_ENTRYPOINT`` for the
    /// classic "alive from entrypoints" question). The set of dead
    /// decls is the complement against ``nodes()``.
    #[pyo3(signature = (*, skip_flags = 0, seed_flags = NODE_FLAG_ENTRYPOINT))]
    pub(crate) fn reachable(
        &self,
        py: Python<'_>,
        skip_flags: u32,
        seed_flags: u32,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("reachable"))?;
        let seeds = outputs
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, n)| (n.borrow(py).flags & seed_flags != 0).then_some(idx));
        Ok(bfs(&outputs.builder, seeds, Direction::Forward, skip_flags)
            .into_iter()
            .map(|i| outputs.builder.nodes[i].clone_ref(py))
            .collect())
    }
}

impl ProjectContext {
    /// Return ``(module_node, [decls inside the block])`` for every
    /// file with a top-level ``if __name__ == "__main__":`` block.
    ///
    /// The decls list contains the file's class / function / variable
    /// / import nodes whose source position falls inside the block's
    /// range — same shape ``MainBlockPlugin``'s libcst path computes
    /// from the visitor's payload.
    pub(crate) fn find_main_blocks(&self, py: Python<'_>) -> PyResult<Vec<MainBlock>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_main_blocks"))?;
        let mut out: Vec<MainBlock> = Vec::new();
        for (&file, &module_idx) in &outputs.module_nodes_by_file {
            // Prefilter: ``if __name__ == "__main__":`` always has the
            // literal string ``__main__`` in source. Skip the parse
            // for files that don't even mention it.
            let source = source_text(&self.db, file);
            if !source.contains("__main__") {
                continue;
            }
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let Some(block_range) = find_main_block_range(&parsed) else {
                continue;
            };
            // Collect decls whose target_range falls within block_range.
            let mut decls: Vec<Py<SymbolNode>> = Vec::new();
            for ((entry_file, _place_id, (start, end)), idx) in &outputs.global_index {
                if *entry_file != file {
                    continue;
                }
                let block_start = block_range.start().to_u32();
                let block_end = block_range.end().to_u32();
                if *start >= block_start && *end <= block_end {
                    decls.push(outputs.builder.nodes[*idx].clone_ref(py));
                }
            }
            out.push((outputs.builder.nodes[module_idx].clone_ref(py), decls));
        }
        Ok(out)
    }

    /// Return every class that defines a method with the given name.
    ///
    /// Return every top-level function decorated with ``@<decorator_module>.<name>``
    /// or ``@<name>`` for any ``name`` in ``decorator_names``.
    ///
    /// Both ``@<name>`` (bare) and ``@<name>(...)`` (called) forms
    /// match — the function call is unwrapped before the pattern is
    /// checked. Identity for the attribute prefix is literal
    /// (``@pytest.fixture`` matches; ``@p.fixture`` with
    /// ``import pytest as p`` does not). Bare-name decorators
    /// (``@fixture``) match purely by attribute name regardless of
    /// what the local ``fixture`` refers to — this mirrors the libcst
    /// plugin helpers used by ``PytestPlugin`` etc., which intentionally
    /// keep a loose pattern match rather than trying to chase decorator
    /// imports through ty's resolver.
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_decorated_decls(
        &self,
        py: Python<'_>,
        decorator_module: &str,
        decorator_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, CallArgs)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_decorated_decls"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let names: HashSet<&str> = decorator_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = decorator_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let project_files = &outputs.project_files;
        // Text prefilter inside the parallel walk mirrors the LSP's
        // find_references — skip files that don't even mention any
        // of the decorator names. The rayon::scope here parallelizes
        // the per-file parse + walk across project files; we release
        // the GIL for the duration and re-acquire only to materialize
        // Py<SymbolNode> handles from the collected indices.
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let names_ref = &names;
        let needles_ref: &[&str] = &needle_strs;
        let pairs: Vec<(usize, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let imports = collect_module_imports_local(&parsed, decorator_module, names_ref);
                if imports.is_empty() {
                    return Vec::new();
                }
                let file_imports = collect_all_imports_local(&parsed);
                let mut local = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Stmt::FunctionDef(func) = stmt else {
                        continue;
                    };
                    let Some(call_form) =
                        decorators_match_imports(&func.decorator_list, &imports, names_ref)
                    else {
                        continue;
                    };
                    let key = (file, range_key(func.name.range()));
                    if let Some(&idx) = decl_by_name_range.get(&key) {
                        let call_args = call_form
                            .map(|c| extract_call_args_kwargs(c, &file_imports, decl_by_fqname))
                            .unwrap_or_default();
                        local.push((idx, call_args));
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(idx, call_args)| (outputs.builder.nodes[idx].clone_ref(py), call_args))
            .collect())
    }

    /// Find top-level ``<var> = <Ctor>(...)`` constructions where
    /// ``Ctor`` is imported from ``module`` and is one of ``ctor_names``.
    ///
    /// Recognized shapes (mirroring the libcst plugin helpers):
    /// * ``from <module> import <Ctor>; X = Ctor(...)``
    /// * ``from <module> import <Ctor> as A; X = A(...)``
    /// * ``import <module>; X = <module>.Ctor(...)``
    /// * ``import <module> as m; X = m.Ctor(...)``
    /// * ``X: T = Ctor(...)`` annotated form
    ///
    /// Returns ``[(var_node, ctor_name)]``; ``ctor_name`` is the
    /// upstream constructor's bare name (``"Flask"`` even when imported
    /// as ``F``).
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_instance_constructions(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_instance_constructions"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let allowed: HashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = ctor_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let allowed_ref = &allowed;
        let needles_ref: &[&str] = &needle_strs;
        let pairs: Vec<(usize, String)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let imports = collect_module_imports_local(&parsed, module, allowed_ref);
                if imports.is_empty() {
                    return Vec::new();
                }
                let mut local: Vec<(usize, String)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let (target_range, value) = match top_level_assign_to_name(stmt) {
                        Some(pair) => pair,
                        None => continue,
                    };
                    let Expr::Call(call) = value else { continue };
                    if let Some(matched) = matched_call_target(call, &imports, module, allowed_ref)
                    {
                        let key = (file, range_key(target_range));
                        if let Some(&idx) = decl_by_name_range.get(&key) {
                            local.push((idx, matched));
                        }
                    }
                }
                local
            })
        });
        let out: Vec<(Py<SymbolNode>, String)> = pairs
            .into_iter()
            .map(|(idx, name)| (outputs.builder.nodes[idx].clone_ref(py), name))
            .collect();
        Ok(out)
    }

    /// Find top-level functions decorated with ``@<owner>.<attr>(...)``
    /// where ``attr`` is in ``decorator_attrs``.
    ///
    /// Returns ``[(owner_name, function_node)]``. ``owner_name`` is the
    /// raw textual prefix of the decorator (``"app"`` for ``@app.route``),
    /// not resolved to a graph node — the caller decides which owners
    /// correspond to real framework instances. Multiple decorators on
    /// the same function emit multiple entries.
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_handler_decorators(
        &self,
        py: Python<'_>,
        decorator_attrs: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(String, Py<SymbolNode>, CallArgs)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_handler_decorators"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let attrs: HashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let attrs_ref = &attrs;
        let needles_ref: &[&str] = &needle_strs;
        let triples: Vec<(String, usize, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(String, usize, CallArgs)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Stmt::FunctionDef(func) = stmt else {
                        continue;
                    };
                    let mut seen_owners: HashSet<String> = HashSet::new();
                    for dec in &func.decorator_list {
                        let (root_expr, call_form): (&Expr, Option<&ruff_python_ast::ExprCall>) =
                            match &dec.expression {
                                Expr::Call(call) => (&*call.func, Some(call)),
                                other => (other, None),
                            };
                        let Expr::Attribute(attr) = root_expr else {
                            continue;
                        };
                        if !attrs_ref.contains(attr.attr.as_str()) {
                            continue;
                        }
                        let Expr::Name(owner) = attr.value.as_ref() else {
                            continue;
                        };
                        let owner_name = owner.id.as_str().to_string();
                        if !seen_owners.insert(owner_name.clone()) {
                            continue;
                        }
                        let key = (file, range_key(func.name.range()));
                        if let Some(&idx) = decl_by_name_range.get(&key) {
                            let call_args = call_form
                                .map(|c| extract_call_args_kwargs(c, &file_imports, decl_by_fqname))
                                .unwrap_or_default();
                            local.push((owner_name, idx, call_args));
                        }
                    }
                }
                local
            })
        });
        Ok(triples
            .into_iter()
            .map(|(name, idx, call_args)| {
                (name, outputs.builder.nodes[idx].clone_ref(py), call_args)
            })
            .collect())
    }

    /// Like ``find_handler_decorators`` but matches the two-level form
    /// ``@<owner>.<via_attr>.<attr>(...)`` (e.g. ``@bot.tree.command()``
    /// for discord.py's slash commands). Returns the same
    /// ``[(owner_name, function_node)]`` shape, where ``owner_name`` is
    /// the leftmost ``Name`` in the decorator chain.
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_handler_decorators_via(
        &self,
        py: Python<'_>,
        via_attr: &str,
        decorator_attrs: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(String, Py<SymbolNode>, CallArgs)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_handler_decorators_via"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let attrs: HashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let attrs_ref = &attrs;
        let triples: Vec<(String, usize, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                // ``via_attr`` is the more selective needle ("tree" for
                // discord.py slash commands) than the attr set.
                let source = source_text(db, file);
                if !_contains_identifier(&source, via_attr) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(String, usize, CallArgs)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Stmt::FunctionDef(func) = stmt else {
                        continue;
                    };
                    let mut seen_owners: HashSet<String> = HashSet::new();
                    for dec in &func.decorator_list {
                        let (root_expr, call_form): (&Expr, Option<&ruff_python_ast::ExprCall>) =
                            match &dec.expression {
                                Expr::Call(call) => (&*call.func, Some(call)),
                                other => (other, None),
                            };
                        let Expr::Attribute(outer) = root_expr else {
                            continue;
                        };
                        if !attrs_ref.contains(outer.attr.as_str()) {
                            continue;
                        }
                        let Expr::Attribute(middle) = outer.value.as_ref() else {
                            continue;
                        };
                        if middle.attr.as_str() != via_attr {
                            continue;
                        }
                        let Expr::Name(owner) = middle.value.as_ref() else {
                            continue;
                        };
                        let owner_name = owner.id.as_str().to_string();
                        if !seen_owners.insert(owner_name.clone()) {
                            continue;
                        }
                        let key = (file, range_key(func.name.range()));
                        if let Some(&idx) = decl_by_name_range.get(&key) {
                            let call_args = call_form
                                .map(|c| extract_call_args_kwargs(c, &file_imports, decl_by_fqname))
                                .unwrap_or_default();
                            local.push((owner_name, idx, call_args));
                        }
                    }
                }
                local
            })
        });
        Ok(triples
            .into_iter()
            .map(|(name, idx, call_args)| {
                (name, outputs.builder.nodes[idx].clone_ref(py), call_args)
            })
            .collect())
    }

    /// Find calls of the form ``<expr>.<attr>(...)`` regardless of
    /// receiver, where the positional arg at ``arg_index`` is either a
    /// string literal **or** a list/tuple of string literals. Returns
    /// ``[(owning_decl, captured_string)]`` — one row per captured
    /// string, so ``load_extensions(["a", "b"])`` yields two rows.
    ///
    /// Unlike ``find_calls_on_var``, this matches any receiver shape:
    /// ``bot.load_extension(...)``, ``self.bot.load_extension(...)``,
    /// ``get_bot().load_extension(...)``, etc. Use this when the call
    /// pattern is keyed on the method name and the receiver is the
    /// plugin's concern (typically gated by a per-file import check).
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_calls_on_attr(
        &self,
        py: Python<'_>,
        attr: &str,
        arg_index: usize,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, String, CallArgs)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_on_attr"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let triples: Vec<(usize, String, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_identifier(&source, attr) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(usize, String, CallArgs)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Some(owner_idx) = owner_idx_for_stmt_with(
                        decl_by_name_range,
                        module_nodes_by_file,
                        file,
                        stmt,
                    ) else {
                        continue;
                    };
                    let mut finder = AttrCallFinder {
                        attr,
                        arg_index,
                        file_imports: &file_imports,
                        decl_by_fqname,
                        results: Vec::new(),
                    };
                    finder.visit_stmt(stmt);
                    for (arg, call_args) in finder.results {
                        local.push((owner_idx, arg, call_args));
                    }
                }
                local
            })
        });
        Ok(triples
            .into_iter()
            .map(|(idx, arg, call_args)| (outputs.builder.nodes[idx].clone_ref(py), arg, call_args))
            .collect())
    }

    ///
    /// Recursively walks each candidate's body looking for ``<Ctor>(...)``
    /// or ``<module>.<Ctor>(...)`` call expressions. Returns
    /// ``[(decl_node, [kind, ...])]`` where ``kind`` is the matched
    /// constructor's bare name; multiple kinds appear when a single
    /// factory constructs more than one (e.g. a function that returns a
    /// ``Flask`` after mounting several ``Blueprint``s).
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_factory_decls(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
    ) -> PyResult<Vec<(Py<SymbolNode>, Vec<String>)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_factory_decls"))?;
        let allowed: HashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = ctor_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let allowed_ref = &allowed;
        let needles_ref: &[&str] = &needle_strs;
        let pairs: Vec<(usize, Vec<String>)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, &None, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let imports = collect_module_imports_local(&parsed, module, allowed_ref);
                if imports.is_empty() {
                    return Vec::new();
                }
                let mut local: Vec<(usize, Vec<String>)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let (name_range, body): (TextRange, &[Stmt]) = match stmt {
                        Stmt::FunctionDef(f) => (f.name.range(), &f.body),
                        Stmt::ClassDef(c) => (c.name.range(), &c.body),
                        _ => continue,
                    };
                    let mut finder = FactoryCallFinder {
                        imports: &imports,
                        module,
                        allowed: allowed_ref,
                        kinds: HashSet::new(),
                    };
                    for inner in body {
                        finder.visit_stmt(inner);
                    }
                    if finder.kinds.is_empty() {
                        continue;
                    }
                    let key = (file, range_key(name_range));
                    if let Some(&idx) = decl_by_name_range.get(&key) {
                        let mut kinds_vec: Vec<String> = finder.kinds.into_iter().collect();
                        kinds_vec.sort();
                        local.push((idx, kinds_vec));
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(idx, kinds)| (outputs.builder.nodes[idx].clone_ref(py), kinds))
            .collect())
    }

    /// Find calls to a callable imported from ``module`` with the name
    /// ``name``. Returns ``(owning_decl, string_literal_arg)`` pairs
    /// where the call resolves through the file's local imports.
    ///
    /// The owning decl is the top-level ``FunctionDef`` / ``ClassDef``
    /// the call lives under (including its decorator subtree); calls
    /// at module scope attribute to the module node.
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_calls_to_imported(
        &self,
        py: Python<'_>,
        module: &str,
        name: &str,
        arg_index: usize,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, String, CallArgs)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_to_imported"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let allowed: HashSet<&str> = [name].into_iter().collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let allowed_ref = &allowed;
        let triples: Vec<(usize, String, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_identifier(&source, name) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let imports = collect_module_imports_local(&parsed, module, allowed_ref);
                if imports.is_empty() {
                    return Vec::new();
                }
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(usize, String, CallArgs)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Some(owner_idx) = owner_idx_for_stmt_with(
                        decl_by_name_range,
                        module_nodes_by_file,
                        file,
                        stmt,
                    ) else {
                        continue;
                    };
                    let mut finder = StringArgCallFinder {
                        predicate: |call: &ruff_python_ast::ExprCall| {
                            matched_call_target(call, &imports, module, allowed_ref).is_some()
                        },
                        arg_index,
                        file_imports: &file_imports,
                        decl_by_fqname,
                        results: Vec::new(),
                    };
                    finder.visit_stmt(stmt);
                    for (arg, call_args) in finder.results {
                        local.push((owner_idx, arg, call_args));
                    }
                }
                local
            })
        });
        Ok(triples
            .into_iter()
            .map(|(idx, arg, call_args)| (outputs.builder.nodes[idx].clone_ref(py), arg, call_args))
            .collect())
    }

    /// Find ``<owner>.<attr>(...)`` calls where ``owner`` is the textual
    /// prefix (no import resolution — covers pytest fixture conventions
    /// like ``mocker.patch`` / ``monkeypatch.setattr``).
    ///
    /// ``required_positional`` disambiguates fqname-form calls from
    /// object-form calls when the same method name is overloaded:
    /// ``monkeypatch.setattr("X.Y", v)`` has 2 positional args
    /// (fqname + value) while ``monkeypatch.setattr(obj, "name", v)``
    /// has 3. Pass ``None`` to accept any positional-arg count.
    ///
    /// Returns ``(owning_decl, string_literal_arg)`` pairs.
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_calls_on_var(
        &self,
        py: Python<'_>,
        owner: &str,
        attr: &str,
        arg_index: usize,
        required_positional: Option<usize>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, String, CallArgs)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_on_var"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let triples: Vec<(usize, String, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                // ``owner`` is typically the more selective needle
                // (e.g. ``mocker`` / ``monkeypatch`` show up in far
                // fewer files than common method names like ``patch``).
                let source = source_text(db, file);
                if !_contains_identifier(&source, owner) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(usize, String, CallArgs)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Some(owner_idx) = owner_idx_for_stmt_with(
                        decl_by_name_range,
                        module_nodes_by_file,
                        file,
                        stmt,
                    ) else {
                        continue;
                    };
                    let mut finder = StringArgCallFinder {
                        predicate: |call: &ruff_python_ast::ExprCall| {
                            call_callee_matches_var(call, owner, attr, required_positional)
                        },
                        arg_index,
                        file_imports: &file_imports,
                        decl_by_fqname,
                        results: Vec::new(),
                    };
                    finder.visit_stmt(stmt);
                    for (arg, call_args) in finder.results {
                        local.push((owner_idx, arg, call_args));
                    }
                }
                local
            })
        });
        Ok(triples
            .into_iter()
            .map(|(idx, arg, call_args)| (outputs.builder.nodes[idx].clone_ref(py), arg, call_args))
            .collect())
    }

    /// Walks each class's `DefinitionKind::Class` body for an
    /// `Stmt::FunctionDef` whose name matches. ty's `parsed_module` is
    /// Salsa-cached, so this is just a body scan per class.
    pub(crate) fn find_classes_defining_method(
        &self,
        py: Python<'_>,
        method_name: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_classes_defining_method"))?;
        let global_index = &outputs.global_index;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let indices: Vec<usize> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, &None, |db, file| {
                // Prefilter: if the file source doesn't even contain
                // the method name as an identifier, no class in it
                // can define a method by that name. Avoids the
                // per-file ``semantic_index`` + use-def walk on files
                // that can't contribute.
                let source = source_text(db, file);
                if !_contains_identifier(&source, method_name) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let index = semantic_index(db, file);
                let global = FileScopeId::global();
                let use_def_map = index.use_def_map(global);
                let mut local: Vec<usize> = Vec::new();
                for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
                    let DefinitionState::Defined(def) = state else {
                        continue;
                    };
                    if def.file(db) != file || def.file_scope(db) != global {
                        continue;
                    }
                    let kind = def.kind(db);
                    let Some(class_ref) = kind.as_class() else {
                        continue;
                    };
                    let class_def = class_ref.node(&parsed);
                    if !class_body_defines_method(class_def, method_name) {
                        continue;
                    }
                    let key = (file, def.place(db), range_key(kind.target_range(&parsed)));
                    if let Some(&idx) = global_index.get(&key) {
                        local.push(idx);
                    }
                }
                local
            })
        });
        Ok(indices
            .into_iter()
            .map(|idx| outputs.builder.nodes[idx].clone_ref(py))
            .collect())
    }

    /// Return every transitive subclass of the given class node.
    ///
    /// Direct subtypes come from ty's `type_hierarchy_subtypes`; we BFS
    /// to collect the transitive closure. Results that don't land in
    /// the project (stdlib / external classes) are dropped.
    pub(crate) fn find_subclasses_of(
        &self,
        py: Python<'_>,
        class_node: &SymbolNode,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        if class_node.kind != "class" {
            return Ok(Vec::new());
        }
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_subclasses_of"))?;

        let Some((seed_file, seed_range)) = locate_class_def(
            &self.db,
            &outputs.path_to_file,
            &class_node.path,
            class_node,
        ) else {
            return Ok(Vec::new());
        };
        let out_idx =
            find_subclass_indices_via_refs(&self.db, outputs, seed_file, seed_range, true);
        Ok(out_idx
            .into_iter()
            .map(|idx| outputs.builder.nodes[idx].clone_ref(py))
            .collect())
    }

    // ----- Convenience helpers used by the chainable QueryBuilder ----------

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_decorated(
        &self,
        py: Python<'_>,
        decorator_fqn: &str,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, CallArgs)>> {
        let Some((module, name)) = decorator_fqn.rsplit_once('.') else {
            return Err(PyValueError::new_err(format!(
                "expected a dotted decorator fqn (e.g. 'pytest.fixture'), got {decorator_fqn:?}"
            )));
        };
        self.find_decorated_decls(py, module, vec![name.to_string()], path_regex)
    }

    pub(crate) fn find_constructions(
        &self,
        py: Python<'_>,
        class_fqn: &str,
        include_subclasses: bool,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let Some((module, name)) = class_fqn.rsplit_once('.') else {
            return Err(PyValueError::new_err(format!(
                "expected a dotted class fqn, got {class_fqn:?}"
            )));
        };
        let mut ctors: Vec<String> = vec![name.to_string()];
        if include_subclasses {
            for sub in self.find_subclasses(py, class_fqn, true)? {
                let simple = sub
                    .borrow(py)
                    .fqname
                    .rsplit('.')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if !simple.is_empty() && !ctors.contains(&simple) {
                    ctors.push(simple);
                }
            }
        }
        let pairs = self.find_instance_constructions(py, module, ctors, path_regex)?;
        Ok(pairs.into_iter().map(|(node, _)| node).collect())
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_decorations_on(
        &self,
        py: Python<'_>,
        instance: &SymbolNode,
        method_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, CallArgs)>> {
        let instance_simple = instance.fqname.rsplit('.').next().unwrap_or("").to_string();
        let handlers = self.find_handler_decorators(py, method_names, path_regex)?;
        let mut out = Vec::new();
        for (owner_name, handler, call_args) in handlers {
            if owner_name != instance_simple {
                continue;
            }
            if handler.borrow(py).path != instance.path {
                continue;
            }
            out.push((handler, call_args));
        }
        Ok(out)
    }

    /// Subclasses of the class addressed by ``base_fqn``.
    ///
    /// Works for both project classes (where the fqn resolves to a
    /// graph node) and external classes (``unittest.TestCase``,
    /// ``pydantic.BaseModel``) via ty's module resolver +
    /// ``type_hierarchy_subtypes``. ``transitive=True`` (default)
    /// walks the full subclass closure; ``transitive=False`` returns
    /// only direct subclasses.
    pub(crate) fn find_subclasses(
        &self,
        py: Python<'_>,
        base_fqn: &str,
        transitive: bool,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_subclasses"))?;
        let Some((seed_file, seed_range)) = locate_class_seed(&self.db, outputs, py, base_fqn)
        else {
            return Ok(Vec::new());
        };
        let out_idx =
            find_subclass_indices_via_refs(&self.db, outputs, seed_file, seed_range, transitive);
        Ok(out_idx
            .into_iter()
            .map(|idx| outputs.builder.nodes[idx].clone_ref(py))
            .collect())
    }

    /// Return `(decl_node, comment_text)` for every comment in the
    /// project that matches `pattern` (a regex), paired with the next
    /// declaration that follows it in the same file.
    ///
    /// Comments are scanned from the parser's `Tokens` stream (no
    /// re-lexing); regex matching is full-text against the comment
    /// content (leading `#` included).
    pub(crate) fn find_comment_patterns(
        &self,
        py: Python<'_>,
        pattern: &str,
    ) -> PyResult<Vec<(Py<SymbolNode>, String)>> {
        let regex = self.compile_regex(pattern)?;
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_comment_patterns"))?;
        let mut out = Vec::new();
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let source = source_text(&self.db, file);
            // Lazy — files with no matching comments skip the decl scan.
            let mut file_decls: Option<Vec<(u32, usize)>> = None;
            for token in parsed.tokens() {
                if token.kind() != TokenKind::Comment {
                    continue;
                }
                let range = token.range();
                let text = &source[range];
                if !regex.is_match(text) {
                    continue;
                }
                let decls =
                    file_decls.get_or_insert_with(|| file_decl_sites(file, &outputs.global_index));
                let comment_end = range.end().to_u32();
                let i = decls.partition_point(|(start, _)| *start < comment_end);
                let Some(&(_, decl_idx)) = decls.get(i) else {
                    continue;
                };
                out.push((
                    outputs.builder.nodes[decl_idx].clone_ref(py),
                    text.to_string(),
                ));
            }
        }
        Ok(out)
    }
}

#[pymethods]
impl ProjectContext {
    // ----- Read-only accessors -------------------------------------------

    /// Live nodes in the in-progress graph. Cheap, no copy.
    pub(crate) fn nodes(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs.as_ref().ok_or_else(|| not_materialized("nodes"))?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.clone_ref(py))
            .collect())
    }

    /// Live edges as `(src_idx, dst_idx, flags)` triples.
    pub(crate) fn edges(&self) -> PyResult<Vec<(usize, usize, u32)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs.as_ref().ok_or_else(|| not_materialized("edges"))?;
        Ok(outputs.builder.edges.clone())
    }
}

impl ProjectContext {
    /// Compile `pattern` once and reuse on subsequent calls. The cache
    /// is bounded by the (small) number of distinct patterns a plugin
    /// run uses, so unbounded growth isn't a concern in practice.
    pub(crate) fn compile_regex(&self, pattern: &str) -> PyResult<regex::Regex> {
        if let Some(cached) = self.regex_cache.borrow().get(pattern) {
            return Ok(cached.clone());
        }
        let regex = regex::Regex::new(pattern)
            .map_err(|e| PyValueError::new_err(format!("invalid regex {pattern:?}: {e}")))?;
        self.regex_cache
            .borrow_mut()
            .insert(pattern.to_string(), regex.clone());
        Ok(regex)
    }
}

pub(crate) fn make_db(
    root: &str,
    src_roots: Option<Vec<String>>,
    extra_paths: Option<Vec<String>>,
    python_env: Option<&str>,
    python_version: Option<&str>,
    typeshed: Option<&str>,
) -> PyResult<ProjectDatabase> {
    let root = SystemPathBuf::from(root);
    let env = EnvironmentOptions {
        root: src_roots.map(|paths| paths.into_iter().map(rel_path).collect()),
        extra_paths: extra_paths.map(|paths| paths.into_iter().map(rel_path).collect()),
        python: python_env.map(rel_path),
        python_version: python_version
            .map(|v| {
                SupportedPythonVersion::from_str(v).map_err(|e| {
                    PyValueError::new_err(format!("invalid python_version {v:?}: {e}"))
                })
            })
            .transpose()?
            .map(RangedValue::cli),
        typeshed: typeshed.map(rel_path),
        ..EnvironmentOptions::default()
    };
    let options = Options {
        environment: Some(env),
        ..Options::default()
    };
    let metadata = ProjectMetadata::from_options(options, root.clone(), None, &UseDefaultStrategy)
        .map_err(|e| PyValueError::new_err(format!("invalid configuration: {e:?}")))?;
    let cwd =
        std::env::current_dir().map_err(|e| PyOSError::new_err(format!("cwd unavailable: {e}")))?;
    let cwd = SystemPathBuf::from_path_buf(cwd).map_err(|_| {
        PyValueError::new_err("current working directory is not a valid absolute UTF-8 path")
    })?;
    let system = OsSystem::new(cwd);
    Ok(ProjectDatabase::use_defaults(metadata, system))
}
