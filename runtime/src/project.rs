//! `Project`, `BuildOutputs`, `ProgressBars`, `ProjectContext`, and the
//! `build_project_graph` pipeline entrypoint. This module owns the
//! Salsa-backed analysis context that the rest of the crate operates on.

use std::str::FromStr;
use std::sync::Arc;

use parking_lot::lock_api::RawRwLockRecursive;
use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard};

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
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
use ty_project::watch::{ChangeEvent as TyChangeEvent, ChangedKind, CreatedKind, DeletedKind};
use ty_project::{Db as ProjectDb, ProjectDatabase, ProjectMetadata};
use ty_python_core::definition::DefinitionState;
use ty_python_core::program::UseDefaultStrategy;
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;

use crate::builder::{
    apply_prepared_batch, bfs, lookup_idx, not_materialized, synthetic_node, CollectedOps,
    Direction, GraphBuilder, GraphNode, PreparedOp,
};
use crate::file_payload::{file_to_edges, file_to_nodes, FileEdges, NodeKind, NodeRef};
use crate::file_ref_edges::{file_to_ref_edges, FileRefEdges};
use crate::graph::{intern_kind, DeclIndex, NativeGraph, SymbolNode};
use crate::helpers::{
    call_callee_matches_var, class_body_defines_method, collect_all_imports_local,
    collect_modules_imports_local, decorators_match_imports, extract_call_args_kwargs,
    file_path_string, find_external_seed_direct_subclasses_par, find_main_block_range,
    find_subclass_indices_via_refs_with_queue, is_dunder_name, locate_class_def, locate_class_seed,
    matched_call_target_any, owner_idx_for_stmt_with, range_key, rel_path,
    top_level_assign_to_name, unwrap_subscripted_callee, AttrCallFinder, CallArgs,
    FactoryCallFinder, StringArgCallFinder, NODE_FLAG_ENTRYPOINT,
};
use crate::ingest::{emit_visitor_warning, file_package_name, string_literal_list};
use crate::progress::{
    ProgressCounters, ProgressHandle, ProgressSnapshot, PHASE_ASSEMBLE, PHASE_ENUM, PHASE_FQNAME,
    PHASE_PLUGINS, PHASE_POPULATE,
};
use crate::query::{
    _compile_path_regex, _contains_any_identifier, _contains_identifier, par_scan_files,
    QueryBuilder,
};
use rustc_hash::{FxHashMap, FxHashSet};

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
        let env = build_env_options(src_roots, extra_paths, python_env, python_version, typeshed)?;
        let db = make_db(root, env)?;
        Ok(Self { db })
    }

    /// Build the project-wide symbol graph (single-pass, no plugins).
    pub(crate) fn build(&mut self, py: Python<'_>) -> PyResult<NativeGraph> {
        let counters = Arc::new(ProgressCounters::new());
        let outputs = build_project_graph(py, &mut self.db, false, None, &counters, &[])?;
        let nodes = outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.to_symbol(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(NativeGraph {
            nodes,
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
    pub(crate) path_to_file: FxHashMap<String, File>,
    /// `(file, class_target_range_key) -> class node idx`. Lets
    /// `find_subclasses_of` map ty's `TypeHierarchyClass.selection_range`
    /// back to a graph node in O(1).
    pub(crate) class_by_selection: FxHashMap<(File, (u32, u32)), usize>,
    /// `file -> module node idx`. Lets `find_main_blocks` reach the
    /// file's module node without a linear scan over `builder.nodes`.
    pub(crate) module_nodes_by_file: FxHashMap<File, usize>,
    /// `(file, target_range_key) -> node idx`. Sister of
    /// ``class_by_selection`` but for every top-level decl ingest minted
    /// (function / class / variable / import). Lets ``find_decorated_decls``
    /// and the dispatch-app queries map an AST node's target range to a
    /// graph node in O(1) instead of scanning the full ``global_index``.
    pub(crate) decl_by_name_range: FxHashMap<(File, (u32, u32)), usize>,
    /// `decl_fqname -> [node idx]`. Lets ``find_declarations`` answer
    /// in O(parts) instead of O(parts × all_nodes). Multiple entries
    /// per fqname arise from try/except rebinds and conditional
    /// re-imports.
    pub(crate) decl_by_fqname: FxHashMap<String, Vec<usize>>,
    /// `module_fqname -> node idx`. Lets ``find_module`` answer in
    /// O(1) instead of scanning all nodes.
    pub(crate) module_by_fqname: FxHashMap<String, usize>,
    /// `Import.module -> [import node idx]`. Lets ``find_imports_of``
    /// answer in O(1) lookup + O(matches) walk instead of scanning all
    /// nodes — and a cheap ``imports_of_exists`` short-circuit for
    /// plugins that just need a per-module presence check.
    pub(crate) imports_by_module: FxHashMap<String, Vec<usize>>,
    /// `parent_fqname -> [child node idx]`. Children are the indices of
    /// nodes whose fqname's immediate parent (`fqname.rsplit_once('.').0`)
    /// equals the key. Includes both module and decl nodes; the kind
    /// can be read from ``builder.nodes[idx].kind`` when filtering.
    /// Top-level decls (no `.` in fqname) are bucketed under the empty
    /// string ``""``.
    ///
    /// Lets ``module_surface`` BFS the fqname tree (one O(matches)
    /// hop per parent) instead of linear-scanning ``decl_by_fqname`` /
    /// ``module_by_fqname`` with a ``starts_with`` filter, and lets
    /// ``find_module_top_level_decls`` answer with a single
    /// ``HashMap::get``.
    pub(crate) children_by_parent: FxHashMap<String, Vec<usize>>,
    /// Project-wide class hierarchy keyed by parent class graph idx.
    /// `class_idx -> [direct subclass class_idx]`. Derived in
    /// `assemble_graph` from each file's per-file `class_bases`
    /// payload (see [`crate::file_payload::FileNodes::class_bases`]).
    ///
    /// Lets ``UnittestPlugin`` / ``InitSubclassPlugin`` do an O(N) BFS
    /// over this map instead of calling ``find_references`` per class.
    /// External seeds (e.g. ``unittest.TestCase``) still need
    /// ``find_subclass_indices_via_refs`` for the first hop into the
    /// project; once a project class is found, transitive walks use
    /// this map.
    pub(crate) children_by_node: FxHashMap<usize, Vec<usize>>,
    /// Project-wide class hierarchy keyed by parent fqname (project or
    /// external). `base_fqn -> [direct subclass class_idx]`. Built
    /// alongside ``children_by_node`` from the per-file ``class_bases``
    /// payload. Retained for the upcoming DSL ``subclasses().of_fqn()``
    /// O(1) fast path; not yet wired into the current path.
    #[allow(dead_code)]
    pub(crate) children_by_fqn: FxHashMap<String, Vec<usize>>,
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
    db: &mut ProjectDatabase,
    show_progress: bool,
    stack_size: Option<usize>,
    counters: &Arc<ProgressCounters>,
    per_file_plugin_ids: &[crate::native_plugins::PerFilePluginId],
) -> PyResult<BuildOutputs> {
    let timing = std::env::var_os("DEAD_CST_TIMING").is_some();

    counters.start_phase(PHASE_ENUM, None);
    let t0 = std::time::Instant::now();
    let project_files: Vec<File> = (&db.project().files(db)).into_iter().collect();
    counters.set_enum_total(project_files.len());
    let mut path_to_file: FxHashMap<String, File> =
        FxHashMap::with_capacity_and_hasher(project_files.len(), Default::default());
    for &file in &project_files {
        path_to_file.insert(file_path_string(db, file), file);
    }
    // Peer ``.pyi`` <-> ``.py`` twin map. Used at assembly time for
    // ``pyi_decl -> py_decl`` reachability edges + the stub-only
    // ENTRYPOINT flag fixup (.pyi decls without a runtime twin need
    // ENTRYPOINT so native-extension / protobuf-style stubs stay
    // alive even with no consumer reference).
    let py_files_by_stem: FxHashMap<String, File> = project_files
        .iter()
        .filter_map(|&f| {
            file_path_string(db, f)
                .strip_suffix(".py")
                .map(|stem| (stem.to_string(), f))
        })
        .collect();
    let mut peer_pyi_to_py: FxHashMap<File, File> = FxHashMap::default();
    for &f in &project_files {
        let path = file_path_string(db, f);
        if let Some(stem) = path.strip_suffix(".pyi") {
            if let Some(&py_twin) = py_files_by_stem.get(stem) {
                peer_pyi_to_py.insert(f, py_twin);
            }
        }
    }
    let t_enum = t0.elapsed();
    counters.finish_phase(PHASE_ENUM);

    let progress = ProgressBars::new(show_progress, project_files.len() as u64);
    counters.start_phase(PHASE_POPULATE, Some(project_files.len()));

    // (prewarm phase deleted — confirmed redundant after the
    // fan-out refactor. The populate phase below already
    // parallelises parsed_module + semantic_index loading inside
    // its own attach-per-thread scope; pre-warming them in a
    // separate parallel pass was a no-op at every corpus size we
    // measured. See PR #226 follow-up for the A/B data.)

    // Parallel pre-populate: run file_to_nodes, file_to_edges, and
    // file_to_ref_edges as #[salsa::tracked] queries across all
    // project files. Workers are pure-rust (GIL released via
    // py.allow_threads). Each file's three queries are attached on
    // whatever rayon worker thread picks them up — work-stealing
    // distributes load across cores; salsa-tracked machinery owns
    // the lock-free coordination between workers. The salsa cache
    // is populated for the serial assembly pass below; nothing here
    // directly mutates the graph builder.
    //
    // project_dist_lookup runs in parallel with the file work via
    // rayon::join. The old pipeline overlapped build_dist_lookup
    // with phase 1; we keep the same shape so the first per-file
    // worker that needs it (for PEP 503 canonical [external dist] X
    // classification) hits a warm salsa cache instead of paying
    // the ~100ms walk inline.
    let t_populate = std::time::Instant::now();
    {
        // Per-file work-stealing on rayon: pre-clone N salsa
        // snapshots into a bounded MPMC channel; each task pulls a
        // snapshot, processes ONE file, returns it. ProjectDatabase
        // is Send-but-not-Sync so a shared db can't cross task
        // boundaries — channel transfer is the closest match.
        //
        // When `stack_size` is `Some`, build a local rayon pool
        // with that stack size and run the work inside `install`.
        // Otherwise use the global rayon pool — rayon's default
        // stack (2 MiB) is sufficient for typical Python code;
        // callers with deeply-nested generated code call
        // `ProjectContext.set_stack_size` to opt in to a bigger
        // stack. Linux commits stack pages lazily, so the declared
        // size is virtual address space, not resident memory.
        let num_workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        let (db_tx, db_rx) = crossbeam_channel::bounded::<ProjectDatabase>(num_workers);
        for _ in 0..num_workers {
            db_tx.send(db.clone()).expect("channel open");
        }
        // dist_lookup runs inside the same pool (custom or global)
        // so its par_extend sub-tasks share that pool's workers,
        // not stray onto unrelated rayon usage. Serialized before
        // the rayon::scope below to avoid the "Cannot change
        // database mid-query" salsa panic — see the prior PR's
        // commit message for the hazard.
        let dist_db: ProjectDatabase = db.clone();
        let files_ref: &[File] = &project_files;
        let per_file_ids_ref: &[crate::native_plugins::PerFilePluginId] = per_file_plugin_ids;
        let counters_ref = Arc::clone(counters);
        let run_populate = move || {
            use salsa::Database as _;
            dist_db.attach(|local_db| {
                let _ = crate::file_payload::project_dist_lookup(local_db);
            });
            rayon::scope(|s| {
                for &file in files_ref {
                    let db_tx = db_tx.clone();
                    let db_rx = db_rx.clone();
                    let counters_inner = Arc::clone(&counters_ref);
                    s.spawn(move |_| {
                        let local_db = db_rx.recv().expect("snapshot available");
                        local_db.attach(|local_db| {
                            let _ = file_to_nodes(local_db, file);
                            let _ = file_to_edges(local_db, file);
                            let _ = file_to_ref_edges(local_db, file);
                            // Per-file native plugins: warm the salsa-cached
                            // `per_file_plugin_ops(file, id)` query on this
                            // worker so the serial assembly fold below is a
                            // pure cache read. `run_on_file` is GIL-free and
                            // touches only this file, so it composes with the
                            // GIL-released fan-out.
                            for &id in per_file_ids_ref {
                                let _ =
                                    crate::native_plugins::per_file_plugin_ops(local_db, file, id);
                            }
                        });
                        db_tx.send(local_db).expect("channel open");
                        counters_inner.populate_inc();
                    });
                }
            });
        };
        py.allow_threads(move || match stack_size {
            Some(n) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_workers)
                    .stack_size(n)
                    .build()
                    .expect("build rayon thread pool");
                pool.install(run_populate);
            }
            None => run_populate(),
        });
    }
    let t_populate_elapsed = t_populate.elapsed();
    counters.finish_phase(PHASE_POPULATE);
    progress.ingest.finish_and_clear();
    progress.imports.finish_and_clear();
    progress.references.finish_and_clear();

    counters.start_phase(PHASE_ASSEMBLE, Some(project_files.len()));

    // Serial assembly pass: walk files in deterministic order,
    // assign global graph node indices to each NodeRef, translate
    // edges, mint synthetic External nodes, build Py<SymbolNode>
    // wrappers, flush warnings. All salsa-tracked queries are
    // memoized from the parallel pre-populate above so this pass
    // is pure hashmap/index work plus the GIL-bound Py creation.
    let t_assemble = std::time::Instant::now();
    let assembled = assemble_graph(
        py,
        db,
        &project_files,
        &peer_pyi_to_py,
        per_file_plugin_ids,
        counters,
    )?;
    let t_assemble_elapsed = t_assemble.elapsed();
    counters.finish_phase(PHASE_ASSEMBLE);

    let mut builder = assembled.builder;
    let global_index = assembled.global_index;
    let module_nodes_by_file = assembled.module_nodes_by_file;
    let class_by_selection = assembled.class_by_selection;
    let decl_by_name_range = assembled.decl_by_name_range;
    let ref_to_global = assembled.ref_to_global;

    counters.start_phase(PHASE_FQNAME, Some(builder.nodes.len()));
    let t4 = std::time::Instant::now();
    let (decl_by_fqname, module_by_fqname, imports_by_module, children_by_parent) =
        build_fqname_indices(&builder, counters);
    let t_fqname = t4.elapsed();
    counters.finish_phase(PHASE_FQNAME);

    // Class hierarchy fan-in: fold every file's per-file `class_bases`
    // payload (see `file_payload::FileNodes::class_bases`) into two
    // project-wide indices — node-keyed (intra-project subclass
    // walks) and fqname-keyed (external seeds + the dispatch-plugin
    // hot path). One pass over the per-file payloads + one pass per
    // base; no AST re-walk.
    let t_hierarchy = std::time::Instant::now();
    let (children_by_node, children_by_fqn) = build_class_hierarchy_indices(
        db,
        &project_files,
        &class_by_selection,
        &ref_to_global,
        &decl_by_fqname,
        &builder.nodes,
    );
    let t_hierarchy_elapsed = t_hierarchy.elapsed();
    if timing {
        eprintln!(
            "[dead-cst-timing] files={} nodes={} edges={} enum={:?} populate={:?} assemble={:?} class_hier={:?} fqname={:?} total={:?} rss={}MB",
            project_files.len(),
            builder.nodes.len(),
            builder.edges.len(),
            t_enum,
            t_populate_elapsed,
            t_assemble_elapsed,
            t_hierarchy_elapsed,
            t_fqname,
            t0.elapsed(),
            current_rss_mb(),
        );
    }
    if let Some(mode) = std::env::var_os("DEAD_CST_DUMP_HEAP") {
        let dump = db.salsa_memory_dump();
        if mode == "full" {
            eprintln!("{}", dump.display_full());
        } else {
            eprintln!("{}", dump.display_short());
        }
    }
    builder.peer_pyi_to_py = peer_pyi_to_py;
    Ok(BuildOutputs {
        builder,
        project_files,
        global_index,
        path_to_file,
        class_by_selection,
        module_nodes_by_file,
        decl_by_name_range,
        decl_by_fqname,
        module_by_fqname,
        imports_by_module,
        children_by_node,
        children_by_fqn,
        children_by_parent,
    })
}

/// Read VmRSS from /proc/self/status. Returns 0 on non-Linux or on
/// parse failure. Used by the timing dump for a cheap "what's the
/// resident set after the build" signal without needing
/// /usr/bin/time or external sampling.
fn current_rss_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<usize>().ok())
        })
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

/// Output of the [`assemble_graph`] pass. The fields are exactly the
/// pieces of `BuildOutputs` that the assembly pass computes from the
/// salsa-tracked per-file payloads; the driver adds `project_files`,
/// `path_to_file`, and the fqname indices.
struct AssembledGraph<'db> {
    builder: GraphBuilder,
    global_index: DeclIndex,
    module_nodes_by_file: FxHashMap<File, usize>,
    class_by_selection: FxHashMap<(File, (u32, u32)), usize>,
    decl_by_name_range: FxHashMap<(File, (u32, u32)), usize>,
    /// PROTOTYPE: kept around so the class-hierarchy builder can map
    /// `NodeRef -> global idx`.
    ref_to_global: FxHashMap<NodeRef<'db>, usize>,
}

/// Serial pass that converts the salsa-tracked per-file payloads
/// (`file_to_nodes` / `file_to_edges` / `file_to_ref_edges`) into a
/// fully-populated `GraphBuilder`.
///
/// Three sub-passes:
///
/// 1. **Node mint** — walk `project_files` in order, intern each
///    file's payload `NodeData` into the builder. Build the
///    `NodeRef -> global_idx` map and the assorted secondary
///    indices (`global_index`, `module_nodes_by_file`,
///    `class_by_selection`, `decl_by_name_range`). Apply the
///    stub-only `ENTRYPOINT` flag fixup for `.pyi` decls whose
///    runtime twin doesn't expose the same name.
///
/// 2. **Edge translation** — walk each file's `FileEdges` and
///    `FileRefEdges`, translate every `NodeRef` to its `usize`
///    index via the map. Synthetic External nodes get lazily
///    minted as encountered. Edges whose endpoint isn't recognised
///    (e.g. a Definition in a non-project file that ty's resolver
///    reached but we didn't enumerate) silently drop.
///
/// 3. **Peer .pyi/.py edges** — for each peer pair, walk both
///    files' `exports_by_name` and emit `pyi_decl -> py_decl`
///    edges for any name present in both.
///
/// Warnings buffered in each `FileRefEdges.warnings` are flushed to
/// the `dead_cst._visitor` Python logger at the end — once per
/// warning, on the main thread, with the GIL we already hold.
fn assemble_graph<'db>(
    py: Python<'_>,
    db: &'db ProjectDatabase,
    project_files: &[File],
    peer_pyi_to_py: &FxHashMap<File, File>,
    per_file_plugin_ids: &[crate::native_plugins::PerFilePluginId],
    counters: &Arc<ProgressCounters>,
) -> PyResult<AssembledGraph<'db>> {
    // Pre-count total nodes across project_files. file_to_nodes is
    // salsa-memoized from the parallel populate phase, so this is a
    // ~ns probe per file. Use the exact count to size the hashmaps
    // that grow one-per-node (ref_to_global, global_index,
    // decl_by_name_range) and skip rehashing entirely. The Vec-backed
    // builder fields get pre-allocated too.
    let mut total_nodes: usize = 0;
    let mut total_decls: usize = 0;
    for &file in project_files {
        let nodes_len = file_to_nodes(db, file).nodes.len();
        total_nodes += nodes_len;
        // Index 0 of every file's nodes is the synthetic module node;
        // everything else is a decl. So decls = nodes - 1 per file.
        total_decls += nodes_len.saturating_sub(1);
    }

    let mut builder = GraphBuilder::with_capacity(total_nodes);
    let mut global_index: DeclIndex =
        FxHashMap::with_capacity_and_hasher(total_decls, Default::default());
    let mut module_nodes_by_file: FxHashMap<File, usize> =
        FxHashMap::with_capacity_and_hasher(project_files.len(), Default::default());
    // class_by_selection only sees class decls — typically a small
    // fraction of all decls, so size to a modest fraction (1/4) to
    // dodge initial growth without wasting space.
    let mut class_by_selection: FxHashMap<(File, (u32, u32)), usize> =
        FxHashMap::with_capacity_and_hasher(total_decls / 4 + 1, Default::default());
    let mut decl_by_name_range: FxHashMap<(File, (u32, u32)), usize> =
        FxHashMap::with_capacity_and_hasher(total_decls, Default::default());
    let mut ref_to_global: FxHashMap<NodeRef<'db>, usize> =
        FxHashMap::with_capacity_and_hasher(total_nodes, Default::default());
    let mut local_to_global: FxHashMap<(File, u32), usize> =
        FxHashMap::with_capacity_and_hasher(total_nodes, Default::default());
    let mut all_warnings: Vec<String> = Vec::new();

    // Pass 1: node mint.
    for &file in project_files {
        let payload = file_to_nodes(db, file);
        let parsed = parsed_module(db, file).load(db);

        // Stub-only ENTRYPOINT context: if this is a .pyi without a
        // .py twin, every decl needs ENTRYPOINT. If it has a twin,
        // only decls whose name isn't in the twin's exports need it.
        let path_str = file_path_string(db, file);
        let is_stub = path_str.ends_with(".pyi");
        let stub_twin: Option<File> = if is_stub {
            peer_pyi_to_py.get(&file).copied()
        } else {
            None
        };

        for (local_idx, &node_ref) in payload.refs.iter().enumerate() {
            let node_data = &payload.nodes[local_idx];

            // Apply stub-only ENTRYPOINT flag if needed.
            let mut flags = node_data.flags;
            if is_stub && matches!(node_ref, NodeRef::Def(_)) {
                let has_runtime = match stub_twin {
                    Some(py_file) => {
                        let py_payload = file_to_nodes(db, py_file);
                        let name = node_data.fqname.rsplit('.').next().unwrap_or("");
                        py_payload.exports_by_name.contains_key(name)
                    }
                    None => false,
                };
                if !has_runtime {
                    flags |= NODE_FLAG_ENTRYPOINT;
                }
            }

            let node = GraphNode {
                fqname: node_data.fqname.clone(),
                kind: intern_kind(node_data.kind.as_static_str())?,
                path: node_data.path.clone(),
                start_line: node_data.start_line,
                start_column: node_data.start_column,
                end_line: node_data.end_line,
                end_column: node_data.end_column,
                flags,
                imports: node_data.imports.clone(),
            };
            let global_idx = builder.intern_node(node);
            ref_to_global.insert(node_ref, global_idx);
            local_to_global.insert((file, local_idx as u32), global_idx);

            match node_ref {
                NodeRef::Module(_) => {
                    module_nodes_by_file.insert(file, global_idx);
                }
                NodeRef::Def(d) => {
                    let place_id = d.place(db);
                    let kind = d.kind(db);
                    let tr = kind.target_range(&parsed);
                    let rk = (tr.start().to_u32(), tr.end().to_u32());
                    global_index.insert((file, place_id, rk), global_idx);
                    if matches!(node_data.kind, NodeKind::Class) {
                        class_by_selection.insert((file, rk), global_idx);
                    }
                    decl_by_name_range.insert((file, rk), global_idx);
                }
                NodeRef::External(_) => {
                    // External NodeRefs never appear in FileNodes.refs;
                    // they're only emitted as edge endpoints. This
                    // branch is unreachable in practice but kept
                    // exhaustive so a future variant addition fails
                    // loudly instead of silently dropping.
                }
            }
        }
        counters.assemble_inc();
    }

    // Pass 2: edge translation.
    //
    // The pipeline is three sub-phases, designed so the heavy
    // FxHashMap-probe loop runs without the GIL on rayon workers:
    //
    //   2a. Serial fetch of every file's salsa-tracked `FileEdges` /
    //       `FileRefEdges`. Salsa returns `&'db` refs valid for the
    //       whole assemble call; we stash them in a `Vec`.
    //   2b. Serial pre-mint of every `NodeRef::External` endpoint
    //       referenced by any payload. This pulls the fqname out of
    //       salsa under the GIL and interns the synthetic node so
    //       the parallel phase only has to read `ref_to_global`.
    //   2c. `Python::allow_threads` + `par_iter().flat_map(...)` to
    //       translate every `(NodeRef, NodeRef, flags)` triple into
    //       `(usize, usize, u32)` via two `ref_to_global` probes.
    //       Drops endpoints whose owning file wasn't enumerated.
    //
    // After the parallel phase, sort + dedup the triple list and
    // bulk-insert via `extend_edges` (which still probes `edge_set`
    // so any pre-existing edges merge cleanly).
    //
    // Skipped endpoints reference Definitions in non-project files.
    //
    // The `DEAD_CST_PASS2_SERIAL=1` env var falls back to the
    // pre-parallel serial implementation. Kept around as a kill switch
    // and as the baseline for the perf-bench comparison.

    let t_pass2_start = std::time::Instant::now();
    let serial_pass2 = std::env::var_os("DEAD_CST_PASS2_SERIAL").is_some();

    // 2a: serial fetch. Hold borrows for the whole pass.
    let mut edge_payloads: Vec<&FileEdges<'db>> = Vec::with_capacity(project_files.len());
    let mut ref_edge_payloads: Vec<&FileRefEdges<'db>> = Vec::with_capacity(project_files.len());
    for &file in project_files {
        edge_payloads.push(file_to_edges(db, file));
        ref_edge_payloads.push(file_to_ref_edges(db, file));
    }

    if serial_pass2 {
        // Original serial path: lazy-mint Externals as encountered.
        for payload in &edge_payloads {
            for &(src, dst, flags) in &payload.edges {
                let Some(src_idx) =
                    serial_lookup_or_mint(db, &mut builder, &mut ref_to_global, src)?
                else {
                    continue;
                };
                let Some(dst_idx) =
                    serial_lookup_or_mint(db, &mut builder, &mut ref_to_global, dst)?
                else {
                    continue;
                };
                builder.add_edge(src_idx, dst_idx, flags);
            }
        }
        for payload in &ref_edge_payloads {
            for &(src, dst, flags) in &payload.edges {
                let Some(src_idx) =
                    serial_lookup_or_mint(db, &mut builder, &mut ref_to_global, src)?
                else {
                    continue;
                };
                let Some(dst_idx) =
                    serial_lookup_or_mint(db, &mut builder, &mut ref_to_global, dst)?
                else {
                    continue;
                };
                builder.add_edge(src_idx, dst_idx, flags);
            }
        }
    } else {
        // 2b: pre-mint synthetic External nodes. Walk both payload
        // streams once and intern every `External` endpoint we
        // haven't seen. Synthetics dedup by fqname project-wide, so
        // this is at most `O(distinct externals)` GIL-bound interns.
        let mut external_keys: FxHashSet<NodeRef<'db>> = FxHashSet::default();
        for payload in &edge_payloads {
            for &(src, dst, _) in &payload.edges {
                if matches!(src, NodeRef::External(_)) && !ref_to_global.contains_key(&src) {
                    external_keys.insert(src);
                }
                if matches!(dst, NodeRef::External(_)) && !ref_to_global.contains_key(&dst) {
                    external_keys.insert(dst);
                }
            }
        }
        for payload in &ref_edge_payloads {
            for &(src, dst, _) in &payload.edges {
                if matches!(src, NodeRef::External(_)) && !ref_to_global.contains_key(&src) {
                    external_keys.insert(src);
                }
                if matches!(dst, NodeRef::External(_)) && !ref_to_global.contains_key(&dst) {
                    external_keys.insert(dst);
                }
            }
        }
        for r in external_keys {
            if let NodeRef::External(key) = r {
                let fqname = key.fqname(db).clone();
                let idx = builder.intern_synthetic(fqname);
                ref_to_global.insert(r, idx);
            }
        }

        // 2c: parallel translation. `ref_to_global` is owned + read-
        // only from here; no GIL needed. We release the GIL with
        // `Python::allow_threads` so rayon workers don't contend with
        // anyone else holding it. The closure is pure-Rust HashMap
        // probes, no salsa DB access, no Py allocations.
        let ref_to_global_ref = &ref_to_global;
        let edge_payloads_ref = &edge_payloads;
        let ref_edge_payloads_ref = &ref_edge_payloads;
        let mut triples: Vec<(usize, usize, u32)> = py.allow_threads(|| {
            use rayon::prelude::*;
            let from_edges = edge_payloads_ref.par_iter().flat_map_iter(|payload| {
                payload.edges.iter().filter_map(|&(src, dst, flags)| {
                    let src_idx = *ref_to_global_ref.get(&src)?;
                    let dst_idx = *ref_to_global_ref.get(&dst)?;
                    Some((src_idx, dst_idx, flags))
                })
            });
            let from_refs = ref_edge_payloads_ref.par_iter().flat_map_iter(|payload| {
                payload.edges.iter().filter_map(|&(src, dst, flags)| {
                    let src_idx = *ref_to_global_ref.get(&src)?;
                    let dst_idx = *ref_to_global_ref.get(&dst)?;
                    Some((src_idx, dst_idx, flags))
                })
            });
            let mut t: Vec<(usize, usize, u32)> = from_edges.chain(from_refs).collect();
            t.par_sort_unstable();
            t.dedup();
            t
        });

        // Bulk-insert the translated triples. `extend_edges` keeps
        // the `edge_set` dedup so any prior edges merge correctly.
        builder.extend_edges(std::mem::take(&mut triples));
    }

    // Warnings are still serial — they live on `FileRefEdges` only.
    for payload in &ref_edge_payloads {
        all_warnings.extend(payload.warnings.iter().cloned());
    }

    if std::env::var_os("DEAD_CST_TIMING").is_some() {
        eprintln!(
            "[dead-cst-timing] pass2_mode={} pass2={:?}",
            if serial_pass2 { "serial" } else { "parallel" },
            t_pass2_start.elapsed()
        );
    }

    // Pass 3: peer .pyi/.py reachability edges.
    for (&pyi_file, &py_twin) in peer_pyi_to_py {
        let pyi_payload = file_to_nodes(db, pyi_file);
        let py_payload = file_to_nodes(db, py_twin);
        for (name, pyi_locals) in &pyi_payload.exports_by_name {
            let Some(py_locals) = py_payload.exports_by_name.get(name) else {
                continue;
            };
            for &pyi_local in pyi_locals {
                let pyi_ref = pyi_payload.refs[pyi_local as usize];
                let Some(&pyi_idx) = ref_to_global.get(&pyi_ref) else {
                    continue;
                };
                for &py_local in py_locals {
                    let py_ref = py_payload.refs[py_local as usize];
                    let Some(&py_idx) = ref_to_global.get(&py_ref) else {
                        continue;
                    };
                    builder.add_edge(pyi_idx, py_idx, 0);
                }
            }
        }
    }

    // Pass 4: per-file native plugin fold. Replay each per-file
    // plugin's salsa-cached `FileLocalOp`s (warmed in the parallel
    // fan-out) into the builder, translating file-local indices to
    // global ones via `local_to_global`. Runs here, after the real
    // graph is fully assembled, so plugin synthetic nodes get the
    // highest indices — the same ordering the old post-build apply
    // pass produced. Synthetic nodes still dedup by fqname, so a
    // later project-wide plugin emitting the same node merges.
    fold_per_file_plugin_ops(
        db,
        &mut builder,
        project_files,
        per_file_plugin_ids,
        &local_to_global,
    )?;

    // Flush warnings to Python logger from the main thread (we hold
    // the GIL here; workers don't).
    for msg in &all_warnings {
        emit_visitor_warning(py, msg);
    }

    Ok(AssembledGraph {
        builder,
        global_index,
        module_nodes_by_file,
        class_by_selection,
        decl_by_name_range,
        ref_to_global,
    })
}

/// Fold every registered per-file native plugin's file-local ops into
/// the assembled graph. Each op is applied with the same semantics as
/// the `apply_prepared` handlers for the [`PreparedOp`] variants:
///
/// * [`FileLocalOp::Node`] → intern a synthetic node (dedup by
///   fqname) and wire its `edges_from`/`edges_to` with flags 0.
/// * [`FileLocalOp::Edge`] → add the translated edge with its flags.
/// * [`FileLocalOp::Entrypoint`] → mint the `{marker}:{decl_fqname}`
///   synthetic with [`NODE_FLAG_ENTRYPOINT`] and edge it to the decl.
///
/// Endpoints are file-local indices into that file's `FileNodes.refs`;
/// `local_to_global` maps them to global node indices (built in pass
/// 1, so every well-formed endpoint resolves). A local idx with no
/// global entry is skipped — same lenient contract as the old path.
///
/// `per_file_plugin_ops(db, file, id)` is a no-op cache read here: the
/// parallel fan-out already warmed it on a worker thread. `run_on_file`
/// is GIL-free and intra-file, so nothing in this fold needs the salsa
/// DB beyond the cached lookup.
fn fold_per_file_plugin_ops(
    db: &ProjectDatabase,
    builder: &mut GraphBuilder,
    project_files: &[File],
    per_file_plugin_ids: &[crate::native_plugins::PerFilePluginId],
    local_to_global: &FxHashMap<(File, u32), usize>,
) -> PyResult<()> {
    if per_file_plugin_ids.is_empty() {
        return Ok(());
    }
    use crate::native_plugins::{per_file_plugin_ops, FileLocalOp};
    for &file in project_files {
        let path = file_path_string(db, file);
        for &id in per_file_plugin_ids {
            let file_ops = per_file_plugin_ops(db, file, id);
            if file_ops.is_empty() {
                continue;
            }
            let to_global = |local: u32| local_to_global.get(&(file, local)).copied();
            for op in file_ops {
                match op {
                    FileLocalOp::Node {
                        fqname,
                        kind,
                        flags,
                        edges_to_local_idx,
                        edges_from_local_idx,
                    } => {
                        let node_idx = builder.intern_node(synthetic_node(
                            fqname.clone(),
                            kind,
                            path.clone(),
                            *flags,
                        ));
                        for &local in edges_from_local_idx {
                            if let Some(src_idx) = to_global(local) {
                                builder.add_edge(src_idx, node_idx, 0);
                            }
                        }
                        for &local in edges_to_local_idx {
                            if let Some(dst_idx) = to_global(local) {
                                builder.add_edge(node_idx, dst_idx, 0);
                            }
                        }
                    }
                    FileLocalOp::Edge {
                        src_local_idx,
                        dst_local_idx,
                        flags,
                    } => {
                        if let (Some(src_idx), Some(dst_idx)) =
                            (to_global(*src_local_idx), to_global(*dst_local_idx))
                        {
                            builder.add_edge(src_idx, dst_idx, *flags);
                        }
                    }
                    FileLocalOp::Entrypoint {
                        decl_local_idx,
                        marker,
                    } => {
                        let Some(decl_idx) = to_global(*decl_local_idx) else {
                            continue;
                        };
                        let (decl_fqname, decl_path) = {
                            let node = &builder.nodes[decl_idx];
                            (node.fqname.clone(), node.path.clone())
                        };
                        let marker_fqname = format!("{marker}:{decl_fqname}");
                        let marker_idx = builder.intern_node(synthetic_node(
                            marker_fqname,
                            "synthetic",
                            decl_path,
                            NODE_FLAG_ENTRYPOINT,
                        ));
                        builder.add_edge(marker_idx, decl_idx, 0);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Collect the [`PerFilePluginId`](crate::native_plugins::PerFilePluginId)
/// of every registered per-file native plugin, in registration order.
/// These are folded into the graph during the build (parallel warm in
/// the fan-out, serial replay in [`assemble_graph`]) rather than in the
/// post-build plugin pass. Project-wide plugins (including project-wide
/// external dylibs) and non-native Python plugins are skipped here —
/// they still run in [`collect_prepared_plugin_ops`].
fn extract_per_file_plugin_ids(
    py: Python<'_>,
    plugins: &[PyObject],
) -> Vec<crate::native_plugins::PerFilePluginId> {
    use crate::native_plugins::{NativePlugin, NativePluginKind, PerFilePluginId};
    let mut ids = Vec::new();
    for p in plugins {
        let Ok(native) = p.bind(py).downcast::<NativePlugin>() else {
            continue;
        };
        match &native.borrow().kind {
            NativePluginKind::PerFile(id) => ids.push(*id),
            NativePluginKind::External {
                per_file_id: Some(eid),
                ..
            } => ids.push(PerFilePluginId::External(*eid)),
            _ => {}
        }
    }
    ids
}

/// Fold every file's per-file `class_bases` payload (see
/// [`crate::file_payload::FileNodes::class_bases`]) into two
/// project-wide indices used by the subclass queries:
///
/// * `children_by_node` — parent class graph idx → direct subclass
///   graph idxs. Drives intra-project BFS for both project and
///   external seeds (once the first hop lands a project class).
/// * `children_by_fqn` — base fqname → direct subclass graph idxs.
///   Lets external seeds (`unittest.TestCase`, `flask.Flask`) be
///   answered without an AST re-walk and without ty's framework
///   loader. The same map also serves project bases via the import
///   path — `class C(Foo): ...` with `from m import Foo` lands as
///   `ImportedFqn("m.Foo")`, which matches the project's own
///   `m.Foo` class node when fqname lookup hits `decl_by_fqname`.
///
/// Resolution rules per `ResolvedBase` variant:
///
/// * `LocalSameFileClass(local_idx)` — translate via the file's
///   `refs[local_idx]` + `ref_to_global` to a parent class graph idx
///   and emit a `children_by_node` entry.
/// * `ImportedFqn(fqn)` — index in `children_by_fqn`. Additionally,
///   when the fqn resolves in `decl_by_fqname` to a project class
///   node, emit a `children_by_node` entry too (so a project base
///   imported via `from m import C` participates in node-keyed
///   walks).
/// * `Attribute { module_fqn, attr_name }` — probe `module_fqname`
///   via `module_by_fqname` (project-side) and resolve the attr in
///   that module's `exports_by_name`. Project hits emit
///   `children_by_node` entries; the fqname form
///   `{module_fqn}.{attr_name}` also lands in `children_by_fqn` so
///   external `M.N` references are answerable by string match.
/// * `Unresolvable` — dropped.
fn build_class_hierarchy_indices<'db>(
    db: &'db ProjectDatabase,
    project_files: &[File],
    class_by_selection: &FxHashMap<(File, (u32, u32)), usize>,
    ref_to_global: &FxHashMap<NodeRef<'db>, usize>,
    decl_by_fqname: &FxHashMap<String, Vec<usize>>,
    nodes: &[GraphNode],
) -> (FxHashMap<usize, Vec<usize>>, FxHashMap<String, Vec<usize>>) {
    use crate::file_payload::ResolvedBase;
    let mut by_node: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut by_fqn: FxHashMap<String, Vec<usize>> = FxHashMap::default();

    for &file in project_files {
        let payload = file_to_nodes(db, file);
        for (cls_rk, bases) in &payload.class_bases {
            let Some(&child_idx) = class_by_selection.get(&(file, *cls_rk)) else {
                continue;
            };
            for base in bases {
                match base {
                    ResolvedBase::LocalSameFileClass(local_idx) => {
                        let r = payload.refs[*local_idx as usize];
                        if let Some(&parent_idx) = ref_to_global.get(&r) {
                            if nodes[parent_idx].kind == "class" {
                                by_node.entry(parent_idx).or_default().push(child_idx);
                            }
                        }
                    }
                    ResolvedBase::ImportedFqn(fqn) => {
                        by_fqn.entry(fqn.clone()).or_default().push(child_idx);
                        if let Some(idxs) = decl_by_fqname.get(fqn) {
                            for &parent_idx in idxs {
                                if nodes[parent_idx].kind == "class" {
                                    by_node.entry(parent_idx).or_default().push(child_idx);
                                }
                            }
                        }
                    }
                    ResolvedBase::Attribute {
                        module_fqn,
                        attr_name,
                    } => {
                        // Index by the synthesised fqname for
                        // external bases + dispatch-plugin matches.
                        let composed = format!("{module_fqn}.{attr_name}");
                        by_fqn.entry(composed.clone()).or_default().push(child_idx);
                        // Project-side: resolve `module_fqn.attr_name`
                        // via `decl_by_fqname` (handles the common
                        // case where the target module's class is in
                        // the project) or via the target module's
                        // `exports_by_name`.
                        if let Some(idxs) = decl_by_fqname.get(&composed) {
                            for &parent_idx in idxs {
                                if nodes[parent_idx].kind == "class" {
                                    by_node.entry(parent_idx).or_default().push(child_idx);
                                }
                            }
                        }
                    }
                    ResolvedBase::Unresolvable => {}
                }
            }
        }
    }

    for v in by_node.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    for v in by_fqn.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    (by_node, by_fqn)
}

/// Serial pre-parallel Pass-2 helper: lookup-or-mint a `NodeRef` into
/// its global graph index. Only used when `DEAD_CST_PASS2_SERIAL=1`
/// is set (kept for A/B comparison and as a kill switch).
fn serial_lookup_or_mint<'db>(
    db: &'db ProjectDatabase,
    builder: &mut GraphBuilder,
    ref_to_global: &mut FxHashMap<NodeRef<'db>, usize>,
    r: NodeRef<'db>,
) -> PyResult<Option<usize>> {
    if let Some(&idx) = ref_to_global.get(&r) {
        return Ok(Some(idx));
    }
    match r {
        NodeRef::External(key) => {
            let fqname = key.fqname(db).clone();
            let idx = builder.intern_synthetic(fqname);
            ref_to_global.insert(r, idx);
            Ok(Some(idx))
        }
        _ => Ok(None),
    }
}

/// Pre-build the fqname -> idx maps used by ``find_declarations``,
/// ``find_module``, ``find_imports_of``, and ``module_surface``. One
/// pass over interned nodes; module entries are 1:1 (one module node
/// per fqname) while decl entries (and per-upstream-module import
/// entries) can have multiple binders for the same key — try/except
/// rebinds, conditional re-imports, and multiple ``from X import Y, Z``
/// aliases all bind into the same upstream module.
///
/// ``children_by_parent`` is the fqname-tree index used by
/// :meth:`ProjectContext::module_surface` and
/// :meth:`ProjectContext::find_module_top_level_decls`: for every
/// indexed node (module or decl), the entry is bucketed under the
/// fqname's immediate parent (``rsplit_once('.').0``), with top-level
/// names keyed under the empty string.
#[allow(clippy::type_complexity)]
pub(crate) fn build_fqname_indices(
    builder: &GraphBuilder,
    counters: &Arc<ProgressCounters>,
) -> (
    FxHashMap<String, Vec<usize>>,
    FxHashMap<String, usize>,
    FxHashMap<String, Vec<usize>>,
    FxHashMap<String, Vec<usize>>,
) {
    let mut decls: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    let mut modules: FxHashMap<String, usize> = FxHashMap::default();
    let mut imports_by_module: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    let mut children_by_parent: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    for (idx, node) in builder.nodes.iter().enumerate() {
        counters.fqname_inc();
        // `OVERLOAD`-flagged decls (the `@typing.overload`-decorated
        // stubs of an in-file overload group) are excluded from the
        // fqname trie so cross-module `from mod import f` resolves to
        // the impl only. Reachability still anchors them via the
        // explicit `impl → stub` edge emitted in `file_to_edges`.
        if node.flags & crate::graph::NodeFlags::OVERLOAD != 0 {
            continue;
        }
        let mut index_child = false;
        match node.kind {
            "module" => {
                modules.insert(node.fqname.clone(), idx);
                index_child = true;
            }
            "function" | "class" | "variable" => {
                decls.entry(node.fqname.clone()).or_default().push(idx);
                index_child = true;
            }
            "import" => {
                decls.entry(node.fqname.clone()).or_default().push(idx);
                if let Some(import) = node.imports.as_ref() {
                    imports_by_module
                        .entry(import.module.clone())
                        .or_default()
                        .push(idx);
                }
                index_child = true;
            }
            _ => {}
        }
        if index_child {
            let parent = node
                .fqname
                .rsplit_once('.')
                .map(|(p, _)| p.to_string())
                .unwrap_or_default();
            children_by_parent.entry(parent).or_default().push(idx);
        }
    }
    (decls, modules, imports_by_module, children_by_parent)
}

/// Plugin-aware project graph builder.
///
/// Python instantiates a `ProjectContext`, registers
/// :class:`NativePlugin`s via `add_plugin`, then calls `materialize()`.
/// `materialize` runs the project-wide build in rust, then runs each
/// registered plugin's native impl against the same `ProjectContext`.
/// The impl emits [`PreparedOp`] values that we apply to the graph; the
/// rust `find_*` methods listed below answer queries against the graph
/// in-progress.
///
/// Queries are answered from ty's semantic index: subclass closure goes
/// through `type_hierarchy_subtypes`, method-defines walks each class's
/// `DefinitionKind::Class`, module dunders scan global-scope variable
/// nodes, and comment patterns walk the parser's `Tokens` stream.
/// Variant discriminant for [`ChangeEvent`]. Mirrors the high-level
/// kinds of [`ty_project::watch::ChangeEvent`] that
/// [`ProjectContext::apply_changes`] forwards to the salsa db; the
/// per-variant `kind` enum on the upstream event (FileContent /
/// FileMetadata / Any) is collapsed to `Any` because that's the right
/// default for the user-facing constructors and lets ty do its own
/// stat-based classification.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ChangeEventKind {
    Changed,
    Created,
    Deleted,
    Rescan,
}

/// A file-system change event, passed to
/// [`ProjectContext::apply_changes`] to incrementally invalidate the
/// salsa db. Construct via the classmethods
/// :py:meth:`changed` / :py:meth:`created` / :py:meth:`deleted` /
/// :py:meth:`rescan`. Returned by
/// [`ProjectContext::detect_changes`] for auto-detection flows.
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct ChangeEvent {
    pub(crate) kind: ChangeEventKind,
    /// Absolute or relative path the event refers to. ``None`` for
    /// :py:meth:`rescan` (which has no path).
    pub(crate) path: Option<SystemPathBuf>,
}

impl ChangeEvent {
    /// Lower this event to the upstream :type:`ty_project::watch::ChangeEvent`
    /// representation `apply_changes` consumes. Relative paths are
    /// absolutized against the db's current directory (handled
    /// downstream by `File::sync_path` and friends).
    pub(crate) fn to_ty_event(&self) -> TyChangeEvent {
        match (self.kind, &self.path) {
            (ChangeEventKind::Changed, Some(path)) => TyChangeEvent::Changed {
                path: path.clone(),
                kind: ChangedKind::Any,
            },
            (ChangeEventKind::Created, Some(path)) => TyChangeEvent::Created {
                path: path.clone(),
                kind: CreatedKind::Any,
            },
            (ChangeEventKind::Deleted, Some(path)) => TyChangeEvent::Deleted {
                path: path.clone(),
                kind: DeletedKind::Any,
            },
            (ChangeEventKind::Rescan, _) => TyChangeEvent::Rescan,
            // Path is required for non-Rescan variants. The Python
            // constructors enforce this; if we somehow reach here
            // without one, fall back to a Rescan which is the safe
            // sledgehammer.
            (_, None) => TyChangeEvent::Rescan,
        }
    }
}

#[pymethods]
impl ChangeEvent {
    /// File at ``path`` was modified (content or metadata).
    ///
    /// Forwarded to ``ty_project`` as a ``Changed{kind: Any}`` event;
    /// ty's apply pass does the actual mtime / size comparison via
    /// ``File::sync_path_only`` and only bumps the salsa revision if
    /// something genuinely differs.
    #[classmethod]
    pub(crate) fn changed(_cls: &Bound<'_, PyType>, path: String) -> Self {
        Self {
            kind: ChangeEventKind::Changed,
            path: Some(SystemPathBuf::from(path)),
        }
    }

    /// File or directory at ``path`` was created.
    ///
    /// Forwarded as ``Created{kind: Any}``; ty stats the path to
    /// decide whether to register a single file or to walk a
    /// directory's contents.
    #[classmethod]
    pub(crate) fn created(_cls: &Bound<'_, PyType>, path: String) -> Self {
        Self {
            kind: ChangeEventKind::Created,
            path: Some(SystemPathBuf::from(path)),
        }
    }

    /// File or directory at ``path`` was deleted.
    ///
    /// Forwarded as ``Deleted{kind: Any}``; ty checks whether the
    /// path was previously a file (drop one entry) or a directory
    /// (drop recursively).
    #[classmethod]
    pub(crate) fn deleted(_cls: &Bound<'_, PyType>, path: String) -> Self {
        Self {
            kind: ChangeEventKind::Deleted,
            path: Some(SystemPathBuf::from(path)),
        }
    }

    /// Full-project rescan sentinel. Triggers ``Files::sync_all`` +
    /// project file re-walk + metadata rediscovery in the apply pass.
    /// Use when you don't know which paths changed (or want to be
    /// safe after a long quiet period).
    #[classmethod]
    pub(crate) fn rescan(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            kind: ChangeEventKind::Rescan,
            path: None,
        }
    }

    /// The path this event refers to, or ``None`` for ``rescan()``.
    #[getter]
    pub(crate) fn path(&self) -> Option<String> {
        self.path.as_ref().map(|p| p.as_str().to_string())
    }

    /// One of ``"changed"`` / ``"created"`` / ``"deleted"`` /
    /// ``"rescan"`` — useful for assertions and pretty-printing.
    #[getter]
    pub(crate) fn kind(&self) -> &'static str {
        match self.kind {
            ChangeEventKind::Changed => "changed",
            ChangeEventKind::Created => "created",
            ChangeEventKind::Deleted => "deleted",
            ChangeEventKind::Rescan => "rescan",
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        match &self.path {
            Some(p) => format!("ChangeEvent.{}({:?})", self.kind(), p.as_str()),
            None => format!("ChangeEvent.{}()", self.kind()),
        }
    }
}

#[pyclass]
pub(crate) struct ProjectContext {
    pub(crate) db: ProjectDatabase,
    /// Absolute path of the project root, echoed back to Python via the
    /// :attr:`project_root` getter. Plugins use it to compute paths
    /// relative to the project.
    pub(crate) root: String,
    pub(crate) plugins: Vec<PyObject>,
    /// Populated by `materialize` before plugins run. `None` outside a
    /// materialize call — the apply pass / queries assume it's
    /// `Some` and error if a plugin (incorrectly) caches the ctx and
    /// uses it after materialize returns.
    ///
    /// Wrapped in [`parking_lot::RwLock`] so plugins can run
    /// concurrently from a Python `ThreadPoolExecutor` — queries take a
    /// read guard, the end-of-pass apply takes a write guard.
    /// ``parking_lot`` over ``std::sync`` because the former exposes
    /// ``RwLockReadGuard::map`` for the ``Option`` projection that
    /// :meth:`materialized` performs.
    ///
    /// Held inside an [`Arc`] so the apply path
    /// (:func:`builder::apply_prepared_batch`) can clone the handle
    /// **before** dropping the GIL and then acquire the write lock
    /// without holding the GIL. Acquiring the write lock while
    /// still holding the GIL would deadlock against a reader stuck
    /// on ``take_gil`` (the reader can't drop its read guard until
    /// it re-attaches).
    pub(crate) outputs: Arc<RwLock<Option<BuildOutputs>>>,
    /// Compiled regexes keyed by the source pattern. Plugins call
    /// :meth:`decls_matching_name` / :meth:`find_comment_patterns`
    /// repeatedly across files with the same pattern, so caching keeps
    /// us off the regex compiler in the hot path. Mutex (rather than
    /// RwLock) because the cache write path is cheap and the hot path
    /// is a single hashmap lookup either way.
    pub(crate) regex_cache: Mutex<FxHashMap<String, regex::Regex>>,
    /// When true, ``materialize`` and ``build_project_graph`` draw
    /// indicatif progress bars to stderr.
    pub(crate) show_progress: bool,
    /// GIL-free atomic counters shared between the build pipeline
    /// and a Python-side polling thread (see
    /// :func:`crate::progress::ProgressCounters`). Created once per
    /// ``ProjectContext`` and reset between successive ``materialize``
    /// calls so callers can re-use a context. The Python polling
    /// thread reads snapshots via :meth:`read_progress_snapshot`.
    pub(crate) progress: Arc<ProgressCounters>,
    /// Optional override for the rayon worker stack size (bytes)
    /// used by the populate phase. `None` (the default) uses the
    /// global rayon pool, which honours rayon's own size (2 MiB
    /// unless `RAYON_STACK_SIZE` / `RUST_MIN_STACK` are set in the
    /// process env). `Some(n)` builds a local pool with stack_size
    /// `n`. Set via [`Self::set_stack_size`] for projects with
    /// deeply-nested generated code (protobuf modules, long
    /// star-reexport chains) that overflow the default.
    pub(crate) stack_size: Option<usize>,
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
        let env = build_env_options(src_roots, extra_paths, python_env, python_version, typeshed)?;
        let db = make_db(root, env)?;
        Ok(Self {
            db,
            root: root.to_string(),
            plugins: Vec::new(),
            outputs: Arc::new(RwLock::new(None)),
            regex_cache: Mutex::new(FxHashMap::default()),
            show_progress,
            stack_size: None,
            progress: Arc::new(ProgressCounters::new()),
        })
    }

    /// Absolute project root passed at construction.
    #[getter]
    pub(crate) fn project_root(&self) -> &str {
        &self.root
    }

    /// Override the rayon worker stack size (bytes) used by the
    /// populate phase. Call BEFORE `materialize`; later calls don't
    /// affect a build already in progress. By default no override
    /// is set — populate runs on rayon's global pool with rayon's
    /// own default stack (2 MiB). Set this on projects with
    /// deeply-nested generated code (protobuf modules, ML-generated
    /// ASTs, big literal dicts) that overflow the default. Passing
    /// 0 is invalid and raises ValueError.
    pub(crate) fn set_stack_size(&mut self, bytes: usize) -> PyResult<()> {
        if bytes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stack_size must be > 0",
            ));
        }
        self.stack_size = Some(bytes);
        Ok(())
    }

    /// Current rayon worker stack size override in bytes, or `None`
    /// if no override is set (in which case the populate phase uses
    /// rayon's global pool with its own default stack).
    #[getter]
    pub(crate) fn stack_size(&self) -> Option<usize> {
        self.stack_size
    }

    /// Register a Python plugin. Order of registration is order of
    /// invocation during `materialize`.
    pub(crate) fn add_plugin(&mut self, plugin: PyObject) {
        self.plugins.push(plugin);
    }

    /// Drop every plugin registered via :meth:`add_plugin`.
    ///
    /// :class:`dead_cst.Analysis` calls this at the top of its build
    /// driver so a re-materialize doesn't double-register plugins on
    /// the rust-serial path (where ``add_plugin`` is invoked once per
    /// :meth:`materialize` call). Idempotent.
    pub(crate) fn clear_plugins(&mut self) {
        self.plugins.clear();
    }

    /// Apply a batch of file-system change events to the salsa db.
    ///
    /// Forwards to :meth:`ty_project::ProjectDatabase::apply_changes`,
    /// which handles every event variant correctly: ``Changed`` bumps
    /// the file's revision iff its mtime / size differ; ``Created``
    /// adds the path to the project file set (so brand-new files
    /// become visible on the next ``files()`` enumeration);
    /// ``Deleted`` removes the file from the set; ``Rescan`` triggers
    /// a full ``Files::sync_all`` + project re-walk. Project
    /// configuration files (``pyproject.toml``, ignore files,
    /// ``VERSIONS``) are detected automatically and trigger a project
    /// reload.
    ///
    /// Call before re-running :meth:`materialize` (or
    /// :meth:`build_only`) to incrementally rebuild after a source
    /// edit. Salsa's per-file cache for files whose events didn't
    /// invalidate them stays warm.
    ///
    /// :class:`dead_cst.Analysis.re_materialize` calls this in
    /// combination with :meth:`detect_changes` to autodetect the
    /// dirty set; callers with explicit knowledge of what changed
    /// (e.g. an LSP integration) can build :class:`ChangeEvent` lists
    /// directly via the classmethods on :class:`ChangeEvent`.
    pub(crate) fn apply_changes(&mut self, events: Vec<Py<ChangeEvent>>, py: Python<'_>) {
        let ty_events: Vec<TyChangeEvent> =
            events.iter().map(|e| e.borrow(py).to_ty_event()).collect();
        self.db.apply_changes(&ty_events, None);
    }

    /// Return a list of :class:`ChangeEvent`\\s that, when passed to
    /// :meth:`apply_changes`, brings the salsa db in sync with the
    /// current on-disk state of the project.
    ///
    /// Currently emits a single ``Rescan`` event, which under the
    /// hood runs:
    ///
    /// * ``Files::sync_all(db)`` — stats every salsa-known file; bumps
    ///   each file's revision only if its mtime / size changed. Per-file
    ///   salsa caches for unchanged files survive the call.
    /// * ``Project::reload_files`` — re-walks the project tree to
    ///   discover newly created files and drop deleted ones.
    /// * Project metadata rediscovery — re-reads ``pyproject.toml`` /
    ///   ignore files so config changes take effect.
    ///
    /// This is the "simple and correct" implementation; a future
    /// optimization could compare salsa-cached metadata to disk and
    /// emit precise ``Changed`` / ``Created`` / ``Deleted`` events to
    /// skip the metadata rediscovery and tree re-walk in the
    /// "everything unchanged" case.
    pub(crate) fn detect_changes(&self, py: Python<'_>) -> PyResult<Vec<Py<ChangeEvent>>> {
        let ev = Py::new(
            py,
            ChangeEvent {
                kind: ChangeEventKind::Rescan,
                path: None,
            },
        )?;
        Ok(vec![ev])
    }

    /// Reset the progress counter state to a fresh instance so a
    /// subsequent :meth:`materialize` / :meth:`build_only` call starts
    /// from zero. Without this, the polling thread on the next run
    /// would observe ``finished=true`` from the prior run and exit
    /// immediately. Called by :meth:`dead_cst.Analysis.re_materialize`
    /// before driving a re-build on the same context.
    pub(crate) fn reset_progress(&mut self) {
        self.progress = Arc::new(ProgressCounters::new());
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
    /// Plugin execution may run serially in this method (legacy path,
    /// also forced by `DEAD_CST_PLUGINS_SERIAL=1` in Python) or be
    /// driven concurrently from Python via a
    /// :class:`concurrent.futures.ThreadPoolExecutor` calling
    /// :meth:`build_only` + :meth:`run_plugin_collect` per worker
    /// followed by :meth:`apply_ops_batched`.
    ///
    /// Plugins operate on a **frozen graph**: every plugin's
    /// ``run(ctx)`` sees the same base-graph state, the yielded ops
    /// accumulate in registration / yield order, and a single
    /// end-of-pass write-lock window folds the lot into the graph.
    /// A plugin's own emissions are invisible to its own subsequent
    /// queries (and to other plugins' queries) during ``run``.
    pub(crate) fn materialize(slf: Py<Self>, py: Python<'_>) -> PyResult<NativeGraph> {
        let show_progress = slf.borrow(py).show_progress;
        let stack_size = slf.borrow(py).stack_size;
        let counters = Arc::clone(&slf.borrow(py).progress);
        let build_result = {
            let mut this = slf.borrow_mut(py);
            let per_file_ids = extract_per_file_plugin_ids(py, &this.plugins);
            build_project_graph(
                py,
                &mut this.db,
                show_progress,
                stack_size,
                &counters,
                &per_file_ids,
            )
        };
        let outputs = match build_result {
            Ok(o) => o,
            Err(e) => {
                counters.mark_finished();
                return Err(e);
            }
        };
        *slf.borrow(py).outputs.write() = Some(outputs);

        // Serial plugin pass (legacy / fallback path). Concurrent
        // execution is driven from Python — see
        // ``Analysis.materialize_all``. We collect ops per plugin
        // then apply once at the end, matching the parallel path's
        // frozen-graph contract.
        let plugins: Vec<PyObject> = slf
            .borrow(py)
            .plugins
            .iter()
            .map(|p| p.clone_ref(py))
            .collect();
        let plugin_bar = ProgressBars::plugin_bar(show_progress, plugins.len() as u64);
        let plugin_names: Vec<String> = plugins
            .iter()
            .map(|p| {
                p.bind(py)
                    .get_type()
                    .getattr("__qualname__")
                    .ok()
                    .and_then(|n| n.extract().ok())
                    .unwrap_or_else(|| "<unnamed>".to_string())
            })
            .collect();
        counters.init_plugin_slots(plugin_names.clone());
        counters.start_phase(PHASE_PLUGINS, Some(plugins.len()));
        let plugin_result = (|| -> PyResult<()> {
            let mut prepared: Vec<PreparedOp> = Vec::new();
            for (idx, plugin) in plugins.iter().enumerate() {
                plugin_bar.set_message(plugin_names[idx].clone());
                counters.plugin_started(idx);
                let res = collect_prepared_plugin_ops(py, &slf, plugin, &mut prepared);
                counters.plugin_finished(idx);
                res?;
                plugin_bar.inc(1);
                counters.plugins_inc();
            }
            apply_prepared_batch(&slf, py, prepared)?;
            Ok(())
        })();
        plugin_bar.finish_and_clear();
        counters.finish_phase(PHASE_PLUGINS);
        counters.mark_finished();
        plugin_result?;

        // Snapshot a fresh ``NativeGraph`` from the builder's
        // interned node + edge vecs; the originals stay put.
        let this = slf.borrow(py);
        let outputs_ref = this.outputs.read();
        let outputs = outputs_ref
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ProjectContext lost its outputs"))?;
        let nodes = outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.to_symbol(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(NativeGraph {
            nodes,
            edges: outputs.builder.edges.clone(),
        })
    }

    /// Run a single plugin's ``run(ctx)`` callback and apply every
    /// yielded op to the graph in one end-of-plugin write-lock
    /// window. The plugin's own emissions are **not** visible to
    /// queries issued from its own body — the graph is frozen for
    /// the duration of ``run``.
    ///
    /// Useful when callers want one plugin's emissions landed before
    /// the next plugin runs; the Python concurrent driver prefers
    /// :meth:`run_plugin_collect` + :meth:`apply_ops_batched` so
    /// every plugin in a pool sees the same frozen base graph.
    pub(crate) fn run_plugin(slf: Py<Self>, py: Python<'_>, plugin: PyObject) -> PyResult<()> {
        let mut prepared: Vec<PreparedOp> = Vec::new();
        collect_prepared_plugin_ops(py, &slf, &plugin, &mut prepared)?;
        apply_prepared_batch(&slf, py, prepared)
    }

    /// Run a plugin's ``run(ctx)``, collect every yielded op into a
    /// :class:`CollectedOps` handle, and return it without touching
    /// the graph. The handle is fed back to
    /// :meth:`apply_ops_batched` (one or more handles per call) so
    /// the apply pass runs in a single write-lock window.
    ///
    /// Python's :class:`concurrent.futures.ThreadPoolExecutor` drives
    /// this concurrently across plugins so query time spent in
    /// GIL-releasing rust paths (``find_decorated_decls`` etc.)
    /// overlaps. The graph is read-only during the collect window —
    /// every plugin sees the same base graph.
    pub(crate) fn run_plugin_collect(
        slf: Py<Self>,
        py: Python<'_>,
        plugin: PyObject,
    ) -> PyResult<CollectedOps> {
        let mut prepared: Vec<PreparedOp> = Vec::new();
        collect_prepared_plugin_ops(py, &slf, &plugin, &mut prepared)?;
        Ok(CollectedOps::new(prepared))
    }

    /// Apply a flat list of :class:`CollectedOps` handles to the
    /// graph in registration / yield order under one write-lock
    /// window. Each handle is drained on consumption; calling
    /// :meth:`apply_ops_batched` twice on the same handle raises
    /// :class:`ValueError`.
    pub(crate) fn apply_ops_batched(
        slf: Py<Self>,
        py: Python<'_>,
        ops: Vec<Py<CollectedOps>>,
    ) -> PyResult<()> {
        let mut flat: Vec<PreparedOp> = Vec::new();
        for handle in &ops {
            let mut taken = handle.borrow(py).take()?;
            flat.append(&mut taken);
        }
        apply_prepared_batch(&slf, py, flat)
    }

    /// Snapshot the current graph as a [`NativeGraph`]. Used after
    /// concurrent plugin execution to return the final graph without
    /// going through [`Self::materialize`] (which would re-run the
    /// build).
    pub(crate) fn snapshot_graph(&self, py: Python<'_>) -> PyResult<NativeGraph> {
        let outputs_ref = self.outputs.read();
        let outputs = outputs_ref
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("ProjectContext lost its outputs"))?;
        let nodes = outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.to_symbol(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(NativeGraph {
            nodes,
            edges: outputs.builder.edges.clone(),
        })
    }

    /// Run the materialize **build pass only** — populate
    /// ``outputs`` from a project-wide scan, without running any
    /// plugins. Python then drives plugins via a ``ThreadPoolExecutor``
    /// (one :meth:`run_plugin` call per worker) and finishes with
    /// :meth:`snapshot_graph` to return the final graph to Python.
    pub(crate) fn build_only(slf: Py<Self>, py: Python<'_>) -> PyResult<()> {
        let show_progress = slf.borrow(py).show_progress;
        let stack_size = slf.borrow(py).stack_size;
        let counters = Arc::clone(&slf.borrow(py).progress);
        let mut this = slf.borrow_mut(py);
        let per_file_ids = extract_per_file_plugin_ids(py, &this.plugins);
        match build_project_graph(
            py,
            &mut this.db,
            show_progress,
            stack_size,
            &counters,
            &per_file_ids,
        ) {
            Ok(outputs) => {
                *this.outputs.write() = Some(outputs);
                Ok(())
            }
            Err(e) => {
                counters.mark_finished();
                Err(e)
            }
        }
    }

    /// Atomic snapshot of the build-progress counters as a Python
    /// dict. Called from the Python polling thread at ~100 ms cadence
    /// to drive structured progress events. Field semantics live on
    /// :class:`crate::progress::ProgressCounters`.
    ///
    /// Reading is GIL-bound + non-blocking — every counter is a
    /// relaxed atomic load. Calling this before
    /// :meth:`materialize` / :meth:`build_only` returns the
    /// zero-initialised state (`phase=0`, `finished=False`).
    ///
    /// **Note on long-running ``materialize`` calls**: this method
    /// takes ``&self``, which acquires a pyo3 borrow on the
    /// ``ProjectContext``. ``materialize`` holds ``borrow_mut`` for
    /// the full build, so a concurrent reader thread will see
    /// "Already mutably borrowed" until the build releases. For
    /// race-free polling, prefer :meth:`progress_handle` (returns a
    /// shared Arc-backed handle that's read without re-borrowing
    /// the context).
    pub(crate) fn read_progress_snapshot<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let snap: ProgressSnapshot = self.progress.snapshot();
        snap.to_pydict(py)
    }

    /// Hand out a borrow-free handle over the progress counters.
    /// The handle can be polled concurrently with a long-running
    /// :meth:`materialize` call — it doesn't go through pyo3's
    /// per-context borrow flag (which materialize holds mutably for
    /// the full build).
    pub(crate) fn progress_handle(&self) -> ProgressHandle {
        ProgressHandle {
            counters: Arc::clone(&self.progress),
        }
    }

    /// Mark progress as finished. Called from Python after a build
    /// error so the polling thread exits its loop and stops firing
    /// events. Idempotent.
    pub(crate) fn mark_progress_finished(&self) {
        self.progress.mark_finished();
    }

    /// Bump the per-plugin counter. Called from Python when the
    /// :class:`concurrent.futures.ThreadPoolExecutor` plugin pass
    /// finishes a plugin (the rust side can't observe Python-side
    /// completion ordering without re-entering the GIL).
    pub(crate) fn progress_plugin_done(&self) {
        self.progress.plugins_inc();
    }

    /// Stamp the plugins phase as started + record its total + name
    /// the registered plugins. ``names`` is the plugin list in
    /// registration order (``type(plugin).__qualname__`` per entry);
    /// indices passed to :meth:`progress_plugin_started` /
    /// :meth:`progress_plugin_finished` match.
    pub(crate) fn progress_plugins_start(&self, names: Vec<String>) {
        let total = names.len();
        self.progress.init_plugin_slots(names);
        self.progress.start_phase(PHASE_PLUGINS, Some(total));
    }

    /// Stamp the indexed plugin's start time. Called on worker-thread
    /// entry. Idempotent on out-of-range indices — the
    /// [`crate::progress::ProgressCounters`] no-ops a missing slot.
    pub(crate) fn progress_plugin_started(&self, idx: usize) {
        self.progress.plugin_started(idx);
    }

    /// Stamp the indexed plugin's finish time. Called on worker-thread
    /// exit (both the success and exception paths — the Python driver
    /// wraps the call in ``try/finally``).
    pub(crate) fn progress_plugin_finished(&self, idx: usize) {
        self.progress.plugin_finished(idx);
    }

    /// Stamp the plugins phase as finished + mark the whole pipeline
    /// finished. Called from Python once every plugin future has
    /// resolved (or one raised).
    pub(crate) fn progress_plugins_finish(&self) {
        self.progress.finish_phase(PHASE_PLUGINS);
        self.progress.mark_finished();
    }
}

/// Run a registered ``NativePlugin`` and collect every emitted op into
/// ``sink`` as a [`PreparedOp`]. The graph is read-only for the
/// duration — the plugin's own emissions never make it into
/// ``outputs`` until the apply pass runs.
///
/// Used by both the serial fallback in :meth:`ProjectContext::materialize`
/// and the per-plugin :meth:`ProjectContext::run_plugin_collect` worker
/// driven from Python. Only ``NativePlugin`` is supported — the legacy
/// Python plugin protocol (``plugin.run(ctx)`` yielding ``GraphOp``s)
/// has been removed.
fn collect_prepared_plugin_ops(
    py: Python<'_>,
    ctx: &Py<ProjectContext>,
    plugin: &PyObject,
    sink: &mut Vec<PreparedOp>,
) -> PyResult<()> {
    // ``NativePlugin`` instances run rust directly, filling ``sink``
    // with [`PreparedOp`] variants — no Python ``.run(ctx)`` call and
    // no per-yield ``GraphOp`` allocation.
    let plugin_bound = plugin.bind(py);
    if let Ok(native) = plugin_bound.downcast::<crate::native_plugins::NativePlugin>() {
        let ctx_ref = ctx.borrow(py);
        match &native.borrow().kind {
            // Project-wide impl: one ``run`` against the whole graph.
            crate::native_plugins::NativePluginKind::ProjectWide(inner) => {
                return inner.run(&ctx_ref, sink);
            }
            // Per-file impls (builtin + external-with-per-file dispatch)
            // are folded into the graph during the build itself —
            // their salsa-cached ops are warmed in the parallel fan-out
            // and replayed in `assemble_graph`'s per-file fold. By the
            // time this post-build plugin pass runs they're already
            // applied; collecting them here would double-apply. No-op.
            crate::native_plugins::NativePluginKind::PerFile(_)
            | crate::native_plugins::NativePluginKind::External {
                per_file_id: Some(_),
                ..
            } => {
                return Ok(());
            }
            // Project-wide external dylib plugin: run it once against a
            // restricted, public ``PluginCtx`` view and fold its ops in.
            crate::native_plugins::NativePluginKind::External { plugin, .. } => {
                let pctx = crate::native_plugins::plugin_api::PluginCtx::new(&ctx_ref);
                let mut ops = crate::native_plugins::plugin_api::PluginOps::new();
                plugin.run(&pctx, &mut ops);
                sink.extend(ops.into_inner());
                return Ok(());
            }
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "expected a dead_cst._native.NativePlugin, got {:?}; the Python \
         plugin protocol (plugin.run(ctx) yielding GraphOps) has been removed",
        plugin_bound.get_type().name()?,
    )))
}

/// Acquire a read guard on ``lock`` while releasing the GIL during
/// the (potentially-parking) wait. Used by hot-path readers that
/// would otherwise dead-lock with a concurrent
/// :func:`builder::apply_prepared_batch` writer: the writer drops the GIL
/// before contending for the write lock, but to call ``Py::new`` /
/// ``intern_node`` it has to re-attach the GIL once it owns the
/// guard. A reader that parks on ``read()`` *while holding the GIL*
/// keeps the writer locked out of the GIL re-attach and the system
/// hangs.
///
/// We use the recursive read path
/// (:meth:`parking_lot::RwLock::read_recursive_unchecked`) so a
/// queued writer doesn't starve us indefinitely (parking_lot's
/// default ``read`` is writer-priority, which leads to a different
/// deadlock pattern when the writer is itself parked on the GIL
/// before it can release the queued-writer flag). The "recursive"
/// name is slightly misleading — we're not actually re-entering;
/// we just want the writer-priority bypass.
fn acquire_read_releasing_gil(
    lock: &RwLock<Option<BuildOutputs>>,
) -> RwLockReadGuard<'_, Option<BuildOutputs>> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            // SAFETY: ``lock_shared_recursive`` is the raw lock
            // primitive that bypasses writer-priority. The matching
            // ``make_read_guard_unchecked`` reconstructs the public
            // guard.
            unsafe { lock.raw() }.lock_shared_recursive();
        });
    });
    // SAFETY: we just took the read lock above.
    unsafe { lock.make_read_guard_unchecked() }
}

impl ProjectContext {
    /// Borrow the active `BuildOutputs` or raise the standard
    /// "not materialized" error. Threads the `op` label into the
    /// error message so the caller name appears in the traceback.
    ///
    /// The returned guard keeps the [`RwLock`] read borrow alive for
    /// the lifetime of the receiver, so callers can hold it across
    /// an entire query body without re-borrowing.
    ///
    /// Concurrent plugin invocations (from
    /// :class:`concurrent.futures.ThreadPoolExecutor`) all take read
    /// guards, so they don't serialize on this lock; only the apply
    /// pass that mutates the graph takes the write side.
    ///
    /// **GIL discipline**: when the read lock is contended (a write
    /// is pending), :func:`parking_lot::RwLock::read` parks. Any
    /// thread parking while still holding the GIL deadlocks against
    /// concurrent writers running in the apply path — the writer
    /// holds the write lock and wants to re-acquire the GIL to call
    /// ``Py::new`` / ``intern_node``, but can't because we have the
    /// GIL parked on the read-acquire. This helper accepts ``py``
    /// and drops the GIL during the acquire so writers can make
    /// progress, then re-attaches after we own the read guard.
    pub(crate) fn materialized(
        &self,
        op: &str,
    ) -> PyResult<MappedRwLockReadGuard<'_, BuildOutputs>> {
        // ``parking_lot::RwLock`` is uncontended by default — the
        // fast path takes a single CAS and never parks. We only
        // need the GIL-release dance when there's an actual writer
        // waiting; ``try_read`` checks without parking.
        let guard = match self.outputs.try_read() {
            Some(g) => g,
            None => acquire_read_releasing_gil(&self.outputs),
        };
        if guard.is_none() {
            return Err(not_materialized(op));
        }
        Ok(RwLockReadGuard::map(guard, |o| o.as_ref().unwrap()))
    }
}

// ----- Point-lookup queries (exposed to Python directly on ProjectContext) -

#[pymethods]
impl ProjectContext {
    /// Return every top-level binding whose name matches `__xxx__`.
    ///
    /// Covers module-scope variables (``__all__``, ``__version__``, …)
    /// and module-scope PEP 562 dunder functions (``__getattr__``,
    /// ``__dir__``) — both are observable to import / attribute-access
    /// machinery and must be kept alive even when no source reference
    /// points at them. Plugins use the DSL form:
    /// ``native.query(ctx).modules().with_dunders().indices()``.
    pub(crate) fn find_module_dunders_indices(&self) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_module_dunders_indices")?;
        let mut out = Vec::new();
        for (idx, node) in outputs.builder.nodes.iter().enumerate() {
            if !matches!(node.kind, "variable" | "function") {
                continue;
            }
            if is_dunder_name(&node.fqname) {
                out.push(idx);
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
    pub(crate) fn find_nodes_matching_specs_indices(
        &self,
        project_root: &str,
        regexes: Vec<String>,
        str_specs: Vec<String>,
        abs_paths: Vec<String>,
    ) -> PyResult<Vec<usize>> {
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
        let str_set: FxHashSet<&str> = str_specs.iter().map(String::as_str).collect();
        let abs_set: FxHashSet<&str> = abs_paths.iter().map(String::as_str).collect();

        let outputs = self.materialized("find_nodes_matching_specs_indices")?;
        let mut out: Vec<usize> = Vec::new();
        for (idx, node) in outputs.builder.nodes.iter().enumerate() {
            let path = node.path.as_str();
            if abs_set.contains(path) {
                out.push(idx);
                continue;
            }
            let rel = path
                .strip_prefix(project_root)
                .map(|s| s.trim_start_matches(['/', '\\']))
                .unwrap_or(path);
            if str_set.contains(rel) || str_set.contains(node.fqname.as_str()) {
                out.push(idx);
                continue;
            }
            if compiled.iter().any(|r| r.is_match(rel)) {
                out.push(idx);
            }
        }
        Ok(out)
    }
}

// ``find_imports_of`` / ``_indices`` / ``imports_of_count`` /
// ``find_imports_of_indices`` / ``imports_of_count`` / ``has_imports_of``
// back :class:`ImportQuery` and its ``.count()`` / ``.exists()``
// shortcuts. NOT exposed to Python — plugin authors use the DSL:
// ``native.query(ctx).imports().of(module).indices()`` / ``.count()`` /
// ``.exists()``.
impl ProjectContext {
    /// Every import-kind node whose upstream `module` matches, as
    /// positional indices into ``ctx.nodes()``.
    ///
    /// Covers both `import <module_name>` and
    /// `from <module_name> import ...` styles — both bind import-kind
    /// nodes whose `Import.module` is the absolute dotted name. Star
    /// reexports synthesized from `from <module_name> import *` are
    /// also included.
    pub(crate) fn find_imports_of_indices(&self, module_name: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_imports_of_indices")?;
        Ok(outputs
            .imports_by_module
            .get(module_name)
            .cloned()
            .unwrap_or_default())
    }

    /// O(1) count of how many project import nodes target
    /// ``module_name`` — pre-built index lookup, no Py allocation.
    pub(crate) fn imports_of_count(&self, module_name: &str) -> usize {
        // Fast path: uncontended read avoids GIL juggling. Slow
        // path matches the GIL-drop discipline in
        // :meth:`materialized` — see that method's docstring for
        // the deadlock rationale.
        let read_guard = match self.outputs.try_read() {
            Some(g) => g,
            None => acquire_read_releasing_gil(&self.outputs),
        };
        read_guard
            .as_ref()
            .and_then(|o| o.imports_by_module.get(module_name).map(Vec::len))
            .unwrap_or(0)
    }

    /// Cheap "does any project file import ``module_name``?" check.
    ///
    /// Plugins use this as an early-exit guard so they don't pay the
    /// cost of resolving the framework module out of the venv when the
    /// project doesn't use it. O(1) — single hashmap probe against
    /// ``imports_by_module``, no node iteration, no PyObject clone.
    pub(crate) fn has_imports_of(&self, module_name: &str) -> PyResult<bool> {
        let outputs = self.materialized("has_imports_of")?;
        Ok(outputs
            .imports_by_module
            .get(module_name)
            .is_some_and(|v| !v.is_empty()))
    }
}

#[pymethods]
impl ProjectContext {
    /// Return every declaration (function / class / variable / import)
    /// whose fully qualified name matches ``fqname``, walking back
    /// through dotted segments to find the enclosing top-level decl
    /// when the exact name doesn't match.
    ///
    /// ``pkg.lib.Cls.method`` returns ``pkg.lib.Cls`` because methods
    /// aren't represented as their own graph nodes — same rule the
    /// libcst :func:`find_declarations` follows. Modules are never
    /// returned; use :meth:`find_module` for that.
    /// Indices for every project decl whose fqname matches (walk-back
    /// included). Plugins use the DSL:
    /// ``native.query(ctx).declarations().with_fqname(fqn).indices()``.
    pub(crate) fn find_declarations_indices(&self, fqname: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_declarations_indices")?;
        Ok(find_declarations_indices_in(&outputs, fqname))
    }

    /// O(1) module-by-fqname lookup. Plugins use the DSL:
    /// ``native.query(ctx).modules().with_fqn(fqn).first_idx()``.
    pub(crate) fn find_module_idx(&self, fqname: &str) -> PyResult<Option<usize>> {
        let outputs = self.materialized("find_module_idx")?;
        Ok(outputs.module_by_fqname.get(fqname).copied())
    }
}

#[pymethods]
impl ProjectContext {
    /// O(1) path-to-module lookup. Plugins use the DSL:
    /// ``native.query(ctx).modules().with_path(path).first_idx()``.
    pub(crate) fn module_for_indices(&self, path: &str) -> PyResult<Option<usize>> {
        let outputs = self.materialized("module_for_indices")?;
        Ok(module_for_idx_in(&outputs, path))
    }

    /// Bulk form of :meth:`module_for_indices`. One materialize check
    /// + one ``path_to_file`` / ``module_nodes_by_file`` lookup per
    /// path; missing paths map to ``None``. Lets plugins that
    /// ``module_for(path)`` once per ref row collapse N FFI hops into
    /// one.
    pub(crate) fn modules_for_paths(&self, paths: Vec<String>) -> PyResult<Vec<Option<usize>>> {
        let outputs = self.materialized("modules_for_paths")?;
        Ok(paths
            .iter()
            .map(|p| module_for_idx_in(&outputs, p))
            .collect())
    }

    /// Resolve a dotted FQN to either a declaration or a module node.
    ///
    /// Tries an exact decl match first, then an exact module match,
    /// then walks back through dotted segments looking for an enclosing
    /// decl (``pkg.lib.Cls.method`` resolves to ``pkg.lib.Cls`` because
    /// methods don't get their own graph nodes). Returns ``None`` when
    /// the fqname can't be found anywhere — never raises.
    pub(crate) fn resolve_idx(&self, fqname: &str) -> PyResult<Option<usize>> {
        let outputs = self.materialized("resolve_idx")?;
        Ok(resolve_idx_in(&outputs, fqname))
    }

    /// Return the module node + every transitive decl whose fqname
    /// lives under ``module_fqn``.
    ///
    /// Models ``importlib.import_module(module_fqn)``: the module's
    /// whole top-level surface plus everything its submodules expose.
    /// Empty list when ``module_fqn`` doesn't resolve to a project
    /// module.
    ///
    /// Walks the ``children_by_parent`` fqname-tree index breadth-first
    /// starting from ``module_fqn``. Only module nodes are recursed
    /// into — a decl ``pkg.foo.MyClass`` is included as a child of
    /// ``pkg.foo`` but its synthetic sub-fqnames (``pkg.foo.MyClass.method``)
    /// are not surfaced; nested defs aren't graph nodes anyway, and the
    /// "module surface" contract is per-module.
    /// BFS over the fqname tree from ``module_fqn``: the module idx
    /// + every transitive descendant idx. Plugins use the DSL:
    /// ``native.query(ctx).modules().with_fqn(fqn).surface().indices()``.
    pub(crate) fn module_surface_indices(&self, module_fqn: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("module_surface_indices")?;
        Ok(module_surface_indices_in(&outputs, module_fqn))
    }

    /// Bulk form: resolve every fqname in ``module_fqns`` in a single
    /// scan. Returns a dict keyed by input fqname; missing modules
    /// map to empty lists. No DSL equivalent — the bulk shape is the
    /// reason this exists (one materialize check + one scan instead
    /// of N).
    pub(crate) fn module_surfaces_indices(
        &self,
        module_fqns: Vec<String>,
    ) -> PyResult<FxHashMap<String, Vec<usize>>> {
        let outputs = self.materialized("module_surfaces_indices")?;
        Ok(module_surfaces_indices_in(&outputs, &module_fqns))
    }
}

#[pymethods]
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
    /// One-level lookup via the ``children_by_parent`` fqname tree;
    /// excludes module children. Plugins use the DSL:
    /// ``native.query(ctx).modules().with_fqn(fqn).top_level().indices()``.
    pub(crate) fn find_module_top_level_decls_indices(
        &self,
        module_fqn: &str,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_module_top_level_decls_indices")?;
        if !outputs.module_by_fqname.contains_key(module_fqn) {
            return Ok(Vec::new());
        }
        // One-level lookup via the ``children_by_parent`` fqname tree.
        // Filter out module children so `from p.functions import *`
        // doesn't surface ``p.functions.sub`` (matching the docstring's
        // "submodules and their decls are excluded" contract).
        let mut out: Vec<usize> = Vec::new();
        if let Some(children) = outputs.children_by_parent.get(module_fqn) {
            for &idx in children {
                if outputs.builder.nodes[idx].kind == "module" {
                    continue;
                }
                out.push(idx);
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
    /// Read the entries from ``{module_fqn}.__all__``'s string-literal
    /// RHS and resolve each name in the module's scope. ``None`` means
    /// "no ``__all__``"; ``Some([])`` means "empty / unresolvable".
    /// Plugins use the DSL:
    /// ``native.query(ctx).modules().with_fqn(fqn).dunder_all()``.
    pub(crate) fn find_module_dunder_all_exports_indices(
        &self,
        module_fqn: &str,
    ) -> PyResult<Option<Vec<usize>>> {
        let all_fqn = format!("{module_fqn}.__all__");
        let Some(entries) = self.find_literal_list_entries(&all_fqn)? else {
            return Ok(None);
        };
        let outputs = self.materialized("find_module_dunder_all_exports_indices")?;
        let mut out: Vec<usize> = Vec::new();
        for entry in entries {
            let entry_fqn = format!("{module_fqn}.{entry}");
            if let Some(idxs) = outputs.decl_by_fqname.get(&entry_fqn) {
                out.extend(idxs.iter().copied());
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
        let outputs = self.materialized("find_literal_list_entries")?;
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
            let path = outputs.builder.nodes[idx].path.clone();
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
    /// Every node whose ``path`` starts with ``path_prefix``. Plugins
    /// use the DSL: ``native.query(ctx).decls().with_path_prefix(p)``.
    pub(crate) fn decls_under_indices(&self, path_prefix: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("decls_under_indices")?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter(|(_i, n)| n.path.starts_with(path_prefix))
            .map(|(i, _n)| i)
            .collect())
    }

    /// Every node whose ``path`` contains ``substring`` anywhere.
    /// Plugins use the DSL:
    /// ``native.query(ctx).decls().with_path_contains(s)``.
    pub(crate) fn decls_matching_indices(&self, substring: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("decls_matching_indices")?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter(|(_i, n)| n.path.contains(substring))
            .map(|(i, _n)| i)
            .collect())
    }

    /// Every top-level decl (function / class / variable / import /
    /// type_alias) whose simple name matches ``pattern``. Plugins use
    /// the DSL — compose
    /// ``decls().with_simple_name_regex(p).with_kinds([...])``.
    pub(crate) fn decls_matching_name_indices(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let regex = self.compile_regex(pattern)?;
        let outputs = self.materialized("decls_matching_name_indices")?;
        let mut out: Vec<usize> = Vec::new();
        for (idx, node) in outputs.builder.nodes.iter().enumerate() {
            if !matches!(
                node.kind,
                "function" | "class" | "variable" | "import" | "type_alias"
            ) {
                continue;
            }
            let simple = node.fqname.rsplit('.').next().unwrap_or("");
            if regex.is_match(simple) {
                out.push(idx);
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
        let outputs = self.materialized("descendants")?;
        let root_idx = lookup_idx(&outputs.builder, root, "root")?;
        bfs(&outputs.builder, [root_idx], Direction::Forward, skip_flags)
            .into_iter()
            .map(|i| outputs.builder.nodes[i].to_symbol(py))
            .collect()
    }

    /// Idx-keyed variant of :meth:`descendants`. Takes a positional
    /// index into ``ctx.nodes()`` and returns descendant indices
    /// rather than allocating ``Py<SymbolNode>`` clones. Raises
    /// :class:`IndexError` when ``root_idx`` is out of range.
    #[pyo3(signature = (root_idx, *, skip_flags = 0))]
    pub(crate) fn descendants_indices(
        &self,
        root_idx: usize,
        skip_flags: u32,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("descendants_indices")?;
        let len = outputs.builder.nodes.len();
        if root_idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "descendants_indices: root_idx {root_idx} out of range (len={len})"
            )));
        }
        Ok(
            bfs(&outputs.builder, [root_idx], Direction::Forward, skip_flags)
                .into_iter()
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
        let outputs = self.materialized("ancestors")?;
        let idx = lookup_idx(&outputs.builder, decl, "decl")?;
        bfs(&outputs.builder, [idx], Direction::Reverse, skip_flags)
            .into_iter()
            .map(|i| outputs.builder.nodes[i].to_symbol(py))
            .collect()
    }

    /// Idx-keyed variant of :meth:`ancestors`. Takes a positional
    /// index into ``ctx.nodes()`` and returns ancestor indices rather
    /// than allocating ``Py<SymbolNode>`` clones. Raises
    /// :class:`IndexError` when ``decl_idx`` is out of range.
    #[pyo3(signature = (decl_idx, *, skip_flags = 0))]
    pub(crate) fn ancestors_indices(
        &self,
        decl_idx: usize,
        skip_flags: u32,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("ancestors_indices")?;
        let len = outputs.builder.nodes.len();
        if decl_idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "ancestors_indices: decl_idx {decl_idx} out of range (len={len})"
            )));
        }
        Ok(
            bfs(&outputs.builder, [decl_idx], Direction::Reverse, skip_flags)
                .into_iter()
                .collect(),
        )
    }

    /// One-hop reverse step: every node with an edge directly into
    /// ``idx``. ``skip_flags`` filters edges by intersecting flag
    /// mask — same semantics as :meth:`ancestors_indices`. Dedups by
    /// source index. Plugins use the DSL:
    /// ``native.query(ctx).from_idx(idx).direct_predecessors()``.
    #[pyo3(signature = (idx, *, skip_flags = 0))]
    pub(crate) fn direct_predecessors_idx(
        &self,
        idx: usize,
        skip_flags: u32,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("direct_predecessors_idx")?;
        let len = outputs.builder.nodes.len();
        if idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "direct_predecessors_idx: idx {idx} out of range (len={len})"
            )));
        }
        Ok(direct_predecessors_idxs_in(&outputs, idx, skip_flags))
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
        let outputs = self.materialized("reachable")?;
        let seeds = outputs
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, n)| (n.flags & seed_flags != 0).then_some(idx));
        bfs(&outputs.builder, seeds, Direction::Forward, skip_flags)
            .into_iter()
            .map(|i| outputs.builder.nodes[i].to_symbol(py))
            .collect()
    }
}

#[pymethods]
impl ProjectContext {
    /// Return ``(module_idx, [decl_idx])`` pairs into ``ctx.nodes()``
    /// for every file with a top-level ``if __name__ == "__main__":``
    /// block. The decls list contains the file's class / function /
    /// variable / import nodes whose source position falls inside the
    /// block's range. Plugins use the DSL:
    /// ``native.query(ctx).main_blocks().index_pairs()``.
    pub(crate) fn find_main_blocks_indices(&self) -> PyResult<Vec<(usize, Vec<usize>)>> {
        let outputs = self.materialized("find_main_blocks_indices")?;

        // First pass: identify files that actually contain a top-level
        // ``if __name__ == "__main__":`` block. The cheap ``__main__``
        // substring check filters >99% of modules out before we touch
        // the parsed AST. Only matched files contribute to the per-file
        // decl bucketing below, so the bucket build itself is bounded
        // by O(decls_in_matched_files) — typically a tiny fraction of
        // the full ``global_index``.
        let mut hits: Vec<(File, usize, (u32, u32))> = Vec::new();
        for (&file, &module_idx) in &outputs.module_nodes_by_file {
            let source = source_text(&self.db, file);
            if !source.contains("__main__") {
                continue;
            }
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let Some(block_range) = find_main_block_range(&parsed) else {
                continue;
            };
            hits.push((
                file,
                module_idx,
                (block_range.start().to_u32(), block_range.end().to_u32()),
            ));
        }
        if hits.is_empty() {
            return Ok(Vec::new());
        }

        // Build a per-file bucket of (start, end, node_idx) for just the
        // matched files. Previously we linearly scanned the entire
        // ``global_index`` for *every* matched file, giving us
        // O(F_with_main × N_global_decls) — death on large projects.
        // One sweep here turns the inner loop into a bucket lookup.
        let matched_files: FxHashSet<File> = hits.iter().map(|(f, _, _)| *f).collect();
        let mut decls_by_file: FxHashMap<File, Vec<(u32, u32, usize)>> = FxHashMap::default();
        for ((entry_file, _place_id, (start, end)), idx) in &outputs.global_index {
            if matched_files.contains(entry_file) {
                decls_by_file
                    .entry(*entry_file)
                    .or_default()
                    .push((*start, *end, *idx));
            }
        }

        let mut out: Vec<(usize, Vec<usize>)> = Vec::with_capacity(hits.len());
        for (file, module_idx, (block_start, block_end)) in hits {
            let mut decl_idxs: Vec<usize> = Vec::new();
            if let Some(file_decls) = decls_by_file.get(&file) {
                for &(start, end, idx) in file_decls {
                    if start >= block_start && end <= block_end {
                        decl_idxs.push(idx);
                    }
                }
            }
            out.push((module_idx, decl_idxs));
        }
        Ok(out)
    }
}

impl ProjectContext {
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
        decorator_modules: &[String],
        decorator_names: Vec<String>,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, CallArgs)>> {
        let outputs = self.materialized("find_decorated_decls")?;
        let path_re = _compile_path_regex(path_regex)?;
        let names: FxHashSet<&str> = decorator_names.iter().map(String::as_str).collect();
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
        let modules_ref: &[String] = decorator_modules;
        let pairs: Vec<(usize, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_package = file_package_name(db, file);
                let imports = collect_modules_imports_local(
                    &parsed,
                    modules_ref,
                    names_ref,
                    file_package.as_deref(),
                );
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
                        let call_args = if extract_args {
                            call_form
                                .map(|c| extract_call_args_kwargs(c, &file_imports, decl_by_fqname))
                                .unwrap_or_default()
                        } else {
                            CallArgs::default()
                        };
                        local.push((idx, call_args));
                    }
                }
                local
            })
        });
        drop(outputs);
        Ok(pairs)
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
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_instance_constructions(
        &self,
        py: Python<'_>,
        modules: &[String],
        ctor_names: Vec<String>,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, String, CallArgs)>> {
        let outputs = self.materialized("find_instance_constructions")?;
        let path_re = _compile_path_regex(path_regex)?;
        let allowed: FxHashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = ctor_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let allowed_ref = &allowed;
        let needles_ref: &[&str] = &needle_strs;
        let modules_ref: &[String] = modules;
        let pairs: Vec<(usize, String, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_package = file_package_name(db, file);
                let imports = collect_modules_imports_local(
                    &parsed,
                    modules_ref,
                    allowed_ref,
                    file_package.as_deref(),
                );
                if imports.is_empty() {
                    return Vec::new();
                }
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(usize, String, CallArgs)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let (target_range, value) = match top_level_assign_to_name(stmt) {
                        Some(pair) => pair,
                        None => continue,
                    };
                    let Expr::Call(call) = value else { continue };
                    if let Some(matched) =
                        matched_call_target_any(call, &imports, modules_ref, allowed_ref)
                    {
                        let key = (file, range_key(target_range));
                        if let Some(&idx) = decl_by_name_range.get(&key) {
                            let call_args = if extract_args {
                                extract_call_args_kwargs(call, &file_imports, decl_by_fqname)
                            } else {
                                CallArgs::default()
                            };
                            local.push((idx, matched, call_args));
                        }
                    }
                }
                local
            })
        });
        drop(outputs);
        Ok(pairs)
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
        extract_args: bool,
    ) -> PyResult<Vec<(String, usize, CallArgs)>> {
        let outputs = self.materialized("find_handler_decorators")?;
        let path_re = _compile_path_regex(path_regex)?;
        let attrs: FxHashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
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
                    let mut seen_owners: FxHashSet<String> = FxHashSet::default();
                    for dec in &func.decorator_list {
                        let (root_expr, call_form): (&Expr, Option<&ruff_python_ast::ExprCall>) =
                            match &dec.expression {
                                Expr::Call(call) => (&*call.func, Some(call)),
                                other => (other, None),
                            };
                        // ``@app.route[T]()`` — unwrap the leading
                        // subscript so the attribute walk matches the
                        // bare ``@app.route(...)`` shape.
                        let root_expr = unwrap_subscripted_callee(root_expr);
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
                            let call_args = if extract_args {
                                call_form
                                    .map(|c| {
                                        extract_call_args_kwargs(c, &file_imports, decl_by_fqname)
                                    })
                                    .unwrap_or_default()
                            } else {
                                CallArgs::default()
                            };
                            local.push((owner_name, idx, call_args));
                        }
                    }
                }
                local
            })
        });
        drop(outputs);
        Ok(triples)
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
        extract_args: bool,
    ) -> PyResult<Vec<(String, usize, CallArgs)>> {
        let outputs = self.materialized("find_handler_decorators_via")?;
        let path_re = _compile_path_regex(path_regex)?;
        let attrs: FxHashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
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
                    let mut seen_owners: FxHashSet<String> = FxHashSet::default();
                    for dec in &func.decorator_list {
                        let (root_expr, call_form): (&Expr, Option<&ruff_python_ast::ExprCall>) =
                            match &dec.expression {
                                Expr::Call(call) => (&*call.func, Some(call)),
                                other => (other, None),
                            };
                        let root_expr = unwrap_subscripted_callee(root_expr);
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
                            let call_args = if extract_args {
                                call_form
                                    .map(|c| {
                                        extract_call_args_kwargs(c, &file_imports, decl_by_fqname)
                                    })
                                    .unwrap_or_default()
                            } else {
                                CallArgs::default()
                            };
                            local.push((owner_name, idx, call_args));
                        }
                    }
                }
                local
            })
        });
        drop(outputs);
        Ok(triples)
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
        extract_args: bool,
    ) -> PyResult<Vec<(usize, String, CallArgs)>> {
        let outputs = self.materialized("find_calls_on_attr")?;
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
                        extract_args,
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
        drop(outputs);
        Ok(triples)
    }
}

impl ProjectContext {
    /// Top-level functions / classes whose body constructs one of
    /// ``ctor_names`` imported from any of ``modules``.
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
        modules: &[String],
        ctor_names: Vec<String>,
    ) -> PyResult<Vec<(usize, Vec<String>)>> {
        let outputs = self.materialized("find_factory_decls")?;
        let allowed: FxHashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = ctor_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let allowed_ref = &allowed;
        let needles_ref: &[&str] = &needle_strs;
        let modules_ref: &[String] = modules;
        let pairs: Vec<(usize, Vec<String>)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, &None, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_package = file_package_name(db, file);
                let imports = collect_modules_imports_local(
                    &parsed,
                    modules_ref,
                    allowed_ref,
                    file_package.as_deref(),
                );
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
                        modules: modules_ref,
                        allowed: allowed_ref,
                        kinds: FxHashSet::default(),
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
        drop(outputs);
        Ok(pairs)
    }
}

impl ProjectContext {
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
        modules: &[String],
        name: &str,
        arg_index: usize,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, String, CallArgs)>> {
        let outputs = self.materialized("find_calls_to_imported")?;
        let path_re = _compile_path_regex(path_regex)?;
        let allowed: FxHashSet<&str> = [name].into_iter().collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let allowed_ref = &allowed;
        let modules_ref: &[String] = modules;
        let triples: Vec<(usize, String, CallArgs)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_identifier(&source, name) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let file_package = file_package_name(db, file);
                let imports = collect_modules_imports_local(
                    &parsed,
                    modules_ref,
                    allowed_ref,
                    file_package.as_deref(),
                );
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
                            matched_call_target_any(call, &imports, modules_ref, allowed_ref)
                                .is_some()
                        },
                        arg_index,
                        file_imports: &file_imports,
                        decl_by_fqname,
                        extract_args,
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
        drop(outputs);
        Ok(triples)
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
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(crate) fn find_calls_on_var(
        &self,
        py: Python<'_>,
        owner: &str,
        attr: &str,
        arg_index: usize,
        required_positional: Option<usize>,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, String, CallArgs)>> {
        let outputs = self.materialized("find_calls_on_var")?;
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
                        extract_args,
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
        drop(outputs);
        Ok(triples)
    }
}

// ``find_classes_defining_method_indices`` is the rust-side helper
// backing :class:`ClassQuery`. It's NOT exposed to Python — plugin
// authors use the DSL:
// ``native.query(ctx).classes().defining_method(name).indices()``.
impl ProjectContext {
    /// Every class that defines a method with the given name, as
    /// positional indices into ``ctx.nodes()``.
    ///
    /// Walks each class's `DefinitionKind::Class` body for an
    /// `Stmt::FunctionDef` whose name matches. ty's `parsed_module` is
    /// Salsa-cached, so this is just a body scan per class.
    pub(crate) fn find_classes_defining_method_indices(
        &self,
        py: Python<'_>,
        method_name: &str,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_classes_defining_method_indices")?;
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
        drop(outputs);
        Ok(indices)
    }
}

// ``find_subclasses_of_idx`` backs :class:`SubclassQuery.of_idx`.
// NOT exposed to Python — plugin authors use the DSL:
// ``native.query(ctx).subclasses().of_idx(idx).indices()`` or
// ``of_fqn(fqn)``.
impl ProjectContext {
    /// Subclasses of the class at positional index ``class_idx`` into
    /// ``ctx.nodes()``. Bounds-checks the index and raises
    /// :class:`IndexError` when out of range; returns an empty list
    /// when the seed isn't a class node.
    pub(crate) fn find_subclasses_of_idx(&self, class_idx: usize) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_subclasses_of_idx")?;
        let len = outputs.builder.nodes.len();
        if class_idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "find_subclasses_of_idx: class_idx {class_idx} out of range (len={len})"
            )));
        }
        // Snapshot the node fields the inner helper reads — same shape
        // it gets when called from the node-form ``find_subclasses_of``.
        let class_node = &outputs.builder.nodes[class_idx];
        if class_node.kind != "class" {
            return Ok(Vec::new());
        }
        if let Some((seed_file, seed_range)) = locate_class_def(
            &self.db,
            &outputs.path_to_file,
            &class_node.path,
            class_node,
        ) {
            let rk = (seed_range.start().to_u32(), seed_range.end().to_u32());
            if let Some(&seed_idx) = outputs.class_by_selection.get(&(seed_file, rk)) {
                return Ok(transitive_subclasses_via_index(
                    seed_idx,
                    &outputs.children_by_node,
                ));
            }
        }
        Ok(Vec::new())
    }
}

/// Shared one-hop reverse-adjacency walk used by
/// :meth:`ProjectContext::direct_predecessors` and
/// :meth:`direct_predecessors_idx`.
fn direct_predecessors_idxs_in(outputs: &BuildOutputs, idx: usize, skip_flags: u32) -> Vec<usize> {
    let mut seen: FxHashSet<usize> = FxHashSet::default();
    let mut out: Vec<usize> = Vec::new();
    for &(src, flags) in &outputs.builder.reverse_adj[idx] {
        if flags & skip_flags != 0 {
            continue;
        }
        if seen.insert(src) {
            out.push(src);
        }
    }
    out
}

/// Shared BFS used by :meth:`ProjectContext::module_surface` and
/// :meth:`module_surface_indices` — the module index, then every
/// transitive child node (modules recursed into; non-module decls
/// surfaced but not chased for sub-fqnames).
fn module_surface_indices_in(outputs: &BuildOutputs, module_fqn: &str) -> Vec<usize> {
    let Some(&module_idx) = outputs.module_by_fqname.get(module_fqn) else {
        return Vec::new();
    };
    let mut out: Vec<usize> = vec![module_idx];
    let mut queue: std::collections::VecDeque<String> =
        std::collections::VecDeque::from([module_fqn.to_string()]);
    while let Some(parent) = queue.pop_front() {
        let Some(children) = outputs.children_by_parent.get(parent.as_str()) else {
            continue;
        };
        for &child_idx in children {
            let child = &outputs.builder.nodes[child_idx];
            let is_module = child.kind == "module";
            let child_fqname = if is_module {
                Some(child.fqname.clone())
            } else {
                None
            };
            out.push(child_idx);
            if let Some(fqn) = child_fqname {
                queue.push_back(fqn);
            }
        }
    }
    out
}

/// Shared bulk-resolution loop used by
/// :meth:`ProjectContext::module_surfaces` and
/// :meth:`module_surfaces_indices`: dedupe the input fqnames, seed
/// each requested bucket with its module idx (if any), then sweep
/// ``module_by_fqname`` + ``decl_by_fqname`` once and drop matches
/// into every bucket whose prefix lights up.
fn module_surfaces_indices_in(
    outputs: &BuildOutputs,
    module_fqns: &[String],
) -> FxHashMap<String, Vec<usize>> {
    let mut buckets: FxHashMap<String, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(module_fqns.len(), Default::default());
    let mut prefixes: Vec<(String, String)> = Vec::with_capacity(module_fqns.len());
    for fqn in module_fqns {
        if buckets.contains_key(fqn) {
            continue;
        }
        let Some(&module_idx) = outputs.module_by_fqname.get(fqn) else {
            buckets.insert(fqn.clone(), Vec::new());
            continue;
        };
        buckets.insert(fqn.clone(), vec![module_idx]);
        prefixes.push((fqn.clone(), format!("{fqn}.")));
    }
    if prefixes.is_empty() {
        return buckets;
    }
    // Single sweep — K small (≤ inputs), so the nested ``starts_with``
    // is fine. Mirrors the original ``module_surfaces`` body, just
    // staying in idx-space.
    for (fqname, &idx) in &outputs.module_by_fqname {
        for (key, prefix) in &prefixes {
            if fqname.starts_with(prefix) {
                buckets.get_mut(key).expect("seeded above").push(idx);
            }
        }
    }
    for (fqname, idxs) in &outputs.decl_by_fqname {
        for (key, prefix) in &prefixes {
            if fqname.starts_with(prefix) {
                let bucket = buckets.get_mut(key).expect("seeded above");
                bucket.extend(idxs.iter().copied());
            }
        }
    }
    buckets
}

/// Shared walk-back lookup for :meth:`ProjectContext::resolve` and
/// :meth:`resolve_idx`. Tries an exact decl match first, then an
/// exact module match, then strips dotted segments. Returns the
/// idx of the first hit or ``None``.
fn resolve_idx_in(outputs: &BuildOutputs, fqname: &str) -> Option<usize> {
    let mut prefix = fqname;
    loop {
        if let Some(idxs) = outputs.decl_by_fqname.get(prefix) {
            if let Some(&idx) = idxs.first() {
                return Some(idx);
            }
        }
        if let Some(&idx) = outputs.module_by_fqname.get(prefix) {
            return Some(idx);
        }
        match prefix.rsplit_once('.') {
            Some((parent, _)) => prefix = parent,
            None => return None,
        }
    }
}

/// `path -> module-node idx`, shared by :meth:`ProjectContext::module_for`,
/// :meth:`module_for_indices`, and :meth:`modules_for_paths`. Returns
/// ``None`` when the path doesn't name a project module.
fn module_for_idx_in(outputs: &BuildOutputs, path: &str) -> Option<usize> {
    let &file = outputs.path_to_file.get(path)?;
    outputs.module_nodes_by_file.get(&file).copied()
}

/// Walk-back lookup shared by :meth:`ProjectContext::find_declarations`
/// and :meth:`ProjectContext::find_declarations_indices` — try an exact
/// match first, then strip trailing dotted segments until either a
/// decl bucket lands or no parent remains.
fn find_declarations_indices_in(outputs: &BuildOutputs, fqname: &str) -> Vec<usize> {
    let mut prefix = fqname;
    loop {
        if let Some(idxs) = outputs.decl_by_fqname.get(prefix) {
            return idxs.clone();
        }
        match prefix.rsplit_once('.') {
            Some((parent, _)) => prefix = parent,
            None => return Vec::new(),
        }
    }
}

/// BFS from `seed_idx` through `children_by_node`. Returns the
/// transitive closure (excluding the seed itself).
fn transitive_subclasses_via_index(
    seed_idx: usize,
    children_by_node: &FxHashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let mut seen: FxHashSet<usize> = FxHashSet::default();
    let mut out: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = vec![seed_idx];
    while let Some(cur) = stack.pop() {
        let Some(kids) = children_by_node.get(&cur) else {
            continue;
        };
        for &k in kids {
            if seen.insert(k) {
                out.push(k);
                stack.push(k);
            }
        }
    }
    out
}

impl ProjectContext {
    // ----- Convenience helpers used by the chainable QueryBuilder ----------

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_decorated(
        &self,
        py: Python<'_>,
        decorator_fqn: &str,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, CallArgs)>> {
        let Some((module, name)) = decorator_fqn.rsplit_once('.') else {
            return Err(PyValueError::new_err(format!(
                "expected a dotted decorator fqn (e.g. 'pytest.fixture'), got {decorator_fqn:?}"
            )));
        };
        let modules = [module.to_string()];
        self.find_decorated_decls(
            py,
            &modules,
            vec![name.to_string()],
            path_regex,
            extract_args,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_constructions(
        &self,
        py: Python<'_>,
        class_fqn: &str,
        include_subclasses: bool,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, CallArgs)>> {
        let Some((module, name)) = class_fqn.rsplit_once('.') else {
            return Err(PyValueError::new_err(format!(
                "expected a dotted class fqn, got {class_fqn:?}"
            )));
        };
        let mut ctors: Vec<String> = vec![name.to_string()];
        if include_subclasses {
            let sub_idxs = self.find_subclasses_indices(py, class_fqn, true)?;
            let outputs = self.materialized("find_constructions.subclasses")?;
            for idx in sub_idxs {
                let simple = outputs.builder.nodes[idx]
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
        let modules = [module.to_string()];
        let triples =
            self.find_instance_constructions(py, &modules, ctors, path_regex, extract_args)?;
        Ok(triples
            .into_iter()
            .map(|(idx, _name, call_args)| (idx, call_args))
            .collect())
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_decorations_on(
        &self,
        py: Python<'_>,
        instance: &GraphNode,
        method_names: Vec<String>,
        path_regex: Option<&str>,
        extract_args: bool,
    ) -> PyResult<Vec<(usize, CallArgs)>> {
        let instance_simple = instance.fqname.rsplit('.').next().unwrap_or("").to_string();
        let handlers = self.find_handler_decorators(py, method_names, path_regex, extract_args)?;
        let outputs = self.materialized("find_decorations_on")?;
        let mut out: Vec<(usize, CallArgs)> = Vec::new();
        for (owner_name, handler_idx, call_args) in handlers {
            if owner_name != instance_simple {
                continue;
            }
            if outputs.builder.nodes[handler_idx].path != instance.path {
                continue;
            }
            out.push((handler_idx, call_args));
        }
        Ok(out)
    }
}

// ``find_subclasses_indices`` backs :class:`SubclassQuery`.
// NOT exposed to Python — plugin authors use the DSL:
// ``native.query(ctx).subclasses().of_fqn(fqn).indices()``.
impl ProjectContext {
    /// Subclasses of the class addressed by ``base_fqn``, as positional
    /// indices into ``ctx.nodes()``.
    ///
    /// Works for both project classes (where the fqn resolves to a
    /// graph node) and external classes (``unittest.TestCase``,
    /// ``pydantic.BaseModel``) via ty's module resolver +
    /// ``type_hierarchy_subtypes``. ``transitive=true`` walks the full
    /// subclass closure; ``transitive=false`` returns only direct
    /// subclasses.
    pub(crate) fn find_subclasses_indices(
        &self,
        py: Python<'_>,
        base_fqn: &str,
        transitive: bool,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_subclasses_indices")?;
        let Some((seed_file, seed_range)) = locate_class_seed(&self.db, &outputs, base_fqn) else {
            return Ok(Vec::new());
        };
        // Project seeds: use the in-memory class-hierarchy index for
        // an O(deg) BFS. Built once at the end of ``assemble_graph``
        // (see ``build_class_hierarchy_indices``); short-circuits every
        // intra-project subclass query.
        let rk = (seed_range.start().to_u32(), seed_range.end().to_u32());
        if let Some(&seed_idx) = outputs.class_by_selection.get(&(seed_file, rk)) {
            let out_idx = if transitive {
                transitive_subclasses_via_index(seed_idx, &outputs.children_by_node)
            } else {
                outputs
                    .children_by_node
                    .get(&seed_idx)
                    .cloned()
                    .unwrap_or_default()
            };
            return Ok(out_idx);
        }

        // External seeds (seed file lives outside the project, e.g.
        // ``typer.Typer``'s definition in the typer package): the
        // first ``ty_ide::find_references`` hop dominates and is what
        // every dispatch-app plugin pays for. Pre-collect the direct
        // first-hop subclasses via a parallel per-file AST scan over
        // project files (text-prefiltered on the seed's bare name),
        // and skip ``find_references`` entirely when no project file
        // even imports the seed module.
        //
        // The ``Python::allow_threads`` wrap releases the GIL so the
        // internal ``rayon::scope`` inside ``find_references`` (and
        // the ``par_scan_files`` helper used for the fast path) can
        // run without GIL contention, matching the pattern used by
        // ``find_decorated_decls``.
        let class_by_selection = &outputs.class_by_selection;
        let children_by_node = &outputs.children_by_node;
        let project_files: &[File] = &outputs.project_files;
        // ``base_fqn`` splits into ``(seed_module, seed_simple_name)``;
        // both halves are required for the fast path.
        let seed_split: Option<(String, String)> = base_fqn
            .rsplit_once('.')
            .map(|(m, n)| (m.to_string(), n.to_string()));
        // ``ProjectDatabase`` is ``!Sync`` (salsa's per-thread query
        // stack lives in a ``RefCell``), so we can't move ``&self.db``
        // into the ``allow_threads`` closure directly. Hand out a
        // ``Box<dyn ProjectDb>`` snapshot — the ``ProjectDb`` trait
        // object carries ``Send``, and ``find_references`` already
        // takes ``&dyn Db``.
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let direct: Vec<usize> = py.allow_threads(move || {
            let db: &dyn ProjectDb = &*db_handle;
            // Parallel AST-scan + import-presence sweep over project
            // files serves two roles:
            //
            //   1. **Fast-path direct subclasses.** Identifies the
            //      common ``from <module> import <Name>; class
            //      X(<Name>):`` shape without paying ty's expensive
            //      ``find_references`` setup. Those classes are
            //      pre-recorded as hits and seeded into the BFS.
            //
            //   2. **"Re-export possible?" presence check.** If no
            //      project file imports ``<seed_module>`` in any
            //      form, there's no chain through which a re-export
            //      could surface — we can skip the find_references
            //      walk on the external seed entirely. This is the
            //      main optimization for plugins like
            //      ``DispatchAppPlugin(typer.Typer)`` /
            //      ``DispatchAppPlugin(fastapi.FastAPI)`` where the
            //      project usually has zero subclasses of the
            //      framework class.
            type FileRangeList = Vec<(File, TextRange)>;
            let (initial_queue, prefound): (FileRangeList, FileRangeList) =
                if let Some((module, name)) = seed_split.as_ref() {
                    let res =
                        find_external_seed_direct_subclasses_par(db, project_files, module, name);
                    let initial = if res.any_project_file_imports_seed_module {
                        vec![(seed_file, seed_range)]
                    } else {
                        Vec::new()
                    };
                    (initial, res.direct_classes)
                } else {
                    (vec![(seed_file, seed_range)], Vec::new())
                };
            find_subclass_indices_via_refs_with_queue(
                db,
                class_by_selection,
                initial_queue,
                &prefound,
                /*transitive=*/ false,
            )
        });
        // BFS the in-memory class-hierarchy index from each direct hit
        // for the transitive walk — cheaper than recursing back through
        // ``find_references``.
        let mut out_idx: FxHashSet<usize> = direct.iter().copied().collect();
        if transitive {
            for &d in &direct {
                for idx in transitive_subclasses_via_index(d, children_by_node) {
                    out_idx.insert(idx);
                }
            }
        }
        Ok(out_idx.into_iter().collect())
    }
}

#[pymethods]
impl ProjectContext {
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
        let outputs = self.materialized("find_comment_patterns")?;
        let mut out = Vec::new();
        // Per-file bucketed view of `global_index`, materialized lazily on
        // the first matching comment anywhere in the project. The naive
        // approach filtered `global_index` once per matched file — O(F × N).
        // Bucketing once flips that to O(N) + O(1) per matched file.
        let mut decls_by_file: Option<FxHashMap<File, Vec<(u32, usize)>>> = None;
        for &file in &outputs.project_files {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            let source = source_text(&self.db, file);
            for token in parsed.tokens() {
                if token.kind() != TokenKind::Comment {
                    continue;
                }
                let range = token.range();
                let text = &source[range];
                if !regex.is_match(text) {
                    continue;
                }
                let bucket = decls_by_file.get_or_insert_with(|| {
                    let mut m: FxHashMap<File, Vec<(u32, usize)>> = FxHashMap::default();
                    for ((f, _, (start, _)), &idx) in &outputs.global_index {
                        m.entry(*f).or_default().push((*start, idx));
                    }
                    for sites in m.values_mut() {
                        sites.sort_by_key(|(s, _)| *s);
                    }
                    m
                });
                let Some(decls) = bucket.get(&file) else {
                    continue;
                };
                let comment_end = range.end().to_u32();
                let i = decls.partition_point(|(start, _)| *start < comment_end);
                let Some(&(_, decl_idx)) = decls.get(i) else {
                    continue;
                };
                out.push((
                    outputs.builder.nodes[decl_idx].to_symbol(py)?,
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
        let outputs = self.materialized("nodes")?;
        outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.to_symbol(py))
            .collect()
    }

    /// Live edges as `(src_idx, dst_idx, flags)` triples.
    pub(crate) fn edges(&self) -> PyResult<Vec<(usize, usize, u32)>> {
        Ok(self.materialized("edges")?.builder.edges.clone())
    }

    /// Idx-only sibling of :meth:`reachable`. Same semantics: forward
    /// closure from every node carrying any bit in ``seed_flags``,
    /// filtering edges by ``skip_flags``. Returns positional indices
    /// into ``ctx.nodes()`` rather than materialising
    /// ``Py<SymbolNode>`` clones.
    ///
    /// Use when you only need set membership / counting on the
    /// reached set — pair with :meth:`nodes_at` to revive specific
    /// nodes on demand.
    #[pyo3(signature = (*, skip_flags = 0, seed_flags = NODE_FLAG_ENTRYPOINT))]
    pub(crate) fn reachable_indices(
        &self,
        skip_flags: u32,
        seed_flags: u32,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("reachable_indices")?;
        let seeds = outputs
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, n)| (n.flags & seed_flags != 0).then_some(idx));
        Ok(bfs(&outputs.builder, seeds, Direction::Forward, skip_flags)
            .into_iter()
            .collect())
    }

    /// Flat-form predicate filter returning positional indices into
    /// ``ctx.nodes()``. Mirrors the :class:`DeclQuery` predicate
    /// vocabulary (``kind`` / ``kinds`` / ``filename`` /
    /// ``filenames`` / ``simple_name`` / ``simple_names`` / ``paths``
    /// / ``path_regex`` / ``flags`` / ``flags_any`` /
    /// ``fqname_prefix``) but skips the builder construction —
    /// useful when you only have one filter step and just need a
    /// ``list[int]`` of indices.
    ///
    /// Every parameter is keyword-only and optional; unset arguments
    /// don't filter. ``kind`` and ``kinds`` (and similarly ``filename``
    /// / ``filenames``, ``simple_name`` / ``simple_names``) are merged
    /// — pass either form. All set predicates AND together.
    #[pyo3(signature = (
        *,
        kind = None,
        kinds = None,
        filename = None,
        filenames = None,
        simple_name = None,
        simple_names = None,
        paths = None,
        path_regex = None,
        flags = None,
        flags_any = None,
        fqname_prefix = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn indices_where(
        &self,
        kind: Option<String>,
        kinds: Option<Vec<String>>,
        filename: Option<String>,
        filenames: Option<Vec<String>>,
        simple_name: Option<String>,
        simple_names: Option<Vec<String>>,
        paths: Option<Vec<String>>,
        path_regex: Option<String>,
        flags: Option<u32>,
        flags_any: Option<u32>,
        fqname_prefix: Option<String>,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("indices_where")?;
        // Merge singular + plural forms once up front.
        let kinds_vec: Option<Vec<String>> = match (kind, kinds) {
            (None, None) => None,
            (Some(s), None) => Some(vec![s]),
            (None, Some(v)) => Some(v),
            (Some(s), Some(mut v)) => {
                v.push(s);
                Some(v)
            }
        };
        let filenames_vec: Option<Vec<String>> = match (filename, filenames) {
            (None, None) => None,
            (Some(s), None) => Some(vec![s]),
            (None, Some(v)) => Some(v),
            (Some(s), Some(mut v)) => {
                v.push(s);
                Some(v)
            }
        };
        let simple_vec: Option<Vec<String>> = match (simple_name, simple_names) {
            (None, None) => None,
            (Some(s), None) => Some(vec![s]),
            (None, Some(v)) => Some(v),
            (Some(s), Some(mut v)) => {
                v.push(s);
                Some(v)
            }
        };

        let kinds_set: Option<rustc_hash::FxHashSet<&str>> = kinds_vec
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let filenames_set: Option<rustc_hash::FxHashSet<&str>> = filenames_vec
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let simple_set: Option<rustc_hash::FxHashSet<&str>> = simple_vec
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let paths_set: Option<rustc_hash::FxHashSet<&str>> = paths
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());

        let re_compiled: Option<regex::Regex> = match path_regex.as_deref() {
            None => None,
            Some(s) => Some(
                regex::Regex::new(s)
                    .map_err(|e| PyValueError::new_err(format!("invalid path regex {s:?}: {e}")))?,
            ),
        };

        // ``flags`` (all-set) and ``flags_any`` (any-bit) — at most one
        // should be passed; if both are given, AND them (require all
        // bits in ``flags`` AND any bit in ``flags_any``).
        let mut out: Vec<usize> = Vec::new();
        for (idx, node) in outputs.builder.nodes.iter().enumerate() {
            if let Some(k) = &kinds_set {
                if !k.contains(node.kind) {
                    continue;
                }
            }
            if let Some(p) = &paths_set {
                if !p.contains(node.path.as_str()) {
                    continue;
                }
            }
            if let Some(f) = &filenames_set {
                let path = node.path.as_str();
                let basename = path
                    .rsplit_once(std::path::MAIN_SEPARATOR)
                    .map(|(_, name)| name)
                    .unwrap_or(path);
                if !f.contains(basename) {
                    continue;
                }
            }
            if let Some(s) = &simple_set {
                let fq = node.fqname.as_str();
                let simple = fq.rsplit_once('.').map(|(_, n)| n).unwrap_or(fq);
                if !s.contains(simple) {
                    continue;
                }
            }
            if let Some(mask) = flags {
                if node.flags & mask != mask {
                    continue;
                }
            }
            if let Some(mask) = flags_any {
                if node.flags & mask == 0 {
                    continue;
                }
            }
            if let Some(prefix) = &fqname_prefix {
                if !node.fqname.starts_with(prefix.as_str()) {
                    continue;
                }
            }
            if let Some(re) = &re_compiled {
                if !re.is_match(node.path.as_str()) {
                    continue;
                }
            }
            out.push(idx);
        }
        Ok(out)
    }

    /// Inverse of the ``.indices()`` terminals: materialize specific
    /// nodes by their positional indices into ``ctx.nodes()``.
    /// Validates bounds and raises :class:`IndexError` when any index
    /// is out of range.
    pub(crate) fn nodes_at(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.materialized("nodes_at")?;
        let len = outputs.builder.nodes.len();
        let mut out: Vec<Py<SymbolNode>> = Vec::with_capacity(indices.len());
        for idx in indices {
            if idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "node index {idx} out of range (len={len})"
                )));
            }
            out.push(outputs.builder.nodes[idx].to_symbol(py)?);
        }
        Ok(out)
    }

    /// Batched snapshot of ``(kind, path, fqname, flags)`` for each
    /// index in ``indices``. One FFI hop instead of N per-attribute
    /// ``borrow`` round-trips — lets plugins that filter / partition by
    /// these four fields stay GIL-free in the inner loop.
    ///
    /// When a plugin only needs ``path`` (the common "bucket by file"
    /// case), prefer :meth:`node_paths` — it skips the per-row
    /// ``kind`` / ``fqname`` / ``flags`` clones that ``node_attrs`` would
    /// allocate but the plugin would throw away.
    ///
    /// Validates bounds the same way :meth:`nodes_at` does and raises
    /// :class:`IndexError` when any index is out of range.
    pub(crate) fn node_attrs(
        &self,
        indices: Vec<usize>,
    ) -> PyResult<Vec<crate::helpers::NodeAttrs>> {
        let outputs = self.materialized("node_attrs")?;
        let len = outputs.builder.nodes.len();
        let mut out: Vec<crate::helpers::NodeAttrs> = Vec::with_capacity(indices.len());
        for idx in indices {
            if idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "node index {idx} out of range (len={len})"
                )));
            }
            // The builder stores pure-rust ``GraphNode``s, so reading a
            // node's fields needs neither the GIL nor a ``Py`` borrow.
            let node = &outputs.builder.nodes[idx];
            out.push(crate::helpers::NodeAttrs {
                kind: node.kind.to_string(),
                path: node.path.clone(),
                fqname: node.fqname.clone(),
                flags: node.flags,
            });
        }
        Ok(out)
    }

    /// Batched ``path``-only snapshot for each index in ``indices``.
    /// Same FFI shape as :meth:`node_attrs` but allocates only one
    /// ``str`` per row instead of ``(kind, path, fqname, flags)`` —
    /// roughly 3× fewer Python allocations on the common "bucket by
    /// file" path that doesn't need the other three fields.
    ///
    /// Validates bounds and raises :class:`IndexError` when any index
    /// is out of range. Mirrors the contract of :meth:`node_attrs`.
    pub(crate) fn node_paths(&self, indices: Vec<usize>) -> PyResult<Vec<String>> {
        let outputs = self.materialized("node_paths")?;
        let len = outputs.builder.nodes.len();
        let mut out: Vec<String> = Vec::with_capacity(indices.len());
        for idx in indices {
            if idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "node index {idx} out of range (len={len})"
                )));
            }
            // Pure-rust ``GraphNode`` read — no GIL, no ``Py`` borrow.
            // See :meth:`node_attrs` for the rationale.
            out.push(outputs.builder.nodes[idx].path.clone());
        }
        Ok(out)
    }

    /// Batched ``parameter_names`` snapshot for each function-kind
    /// index in ``indices``. Non-function nodes (and indices that
    /// don't resolve to a top-level ``FunctionDef`` in the AST)
    /// surface an empty list at the same position.
    ///
    /// Walks the parsed AST once per distinct path; matches each
    /// top-level ``FunctionDef`` against ``decl_by_name_range`` via
    /// its name's byte range. Used by ``PytestPlugin`` to discover
    /// fixture-dependency edges (``test_foo(my_fixture)`` →
    /// ``test_foo → my_fixture``) — same shape as
    /// :meth:`node_attrs` / :meth:`node_paths`: one FFI hop per
    /// batch, validated bounds.
    ///
    /// Parameter shape: returns the union of positional-only,
    /// positional-or-keyword, and keyword-only parameter names in
    /// declaration order. ``*args`` and ``**kwargs`` are skipped
    /// (pytest never resolves them as fixture references).
    pub(crate) fn function_parameters(
        &self,
        _py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        let outputs = self.materialized("function_parameters")?;
        let nodes = &outputs.builder.nodes;
        let len = nodes.len();
        for &idx in &indices {
            if idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "node index {idx} out of range (len={len})"
                )));
            }
        }
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        // Build the wanted set + an idx → (file, name_range) inverse
        // by scanning ``decl_by_name_range`` once. The decl-by-name
        // index already has the byte range of every function decl
        // (the name range, e.g. ``test_foo`` in ``def test_foo``);
        // we use that as the rendezvous key against the AST below.
        let wanted: FxHashSet<usize> = indices.iter().copied().collect();
        let mut idx_to_loc: FxHashMap<usize, (File, (u32, u32))> =
            FxHashMap::with_capacity_and_hasher(wanted.len(), Default::default());
        for (&(file, rk), &idx) in &outputs.decl_by_name_range {
            if wanted.contains(&idx) {
                idx_to_loc.insert(idx, (file, rk));
            }
        }

        // Group wanted ranges by file for a single AST walk per
        // distinct path.
        let mut by_file: FxHashMap<File, FxHashMap<(u32, u32), usize>> = FxHashMap::default();
        for (&idx, &(file, rk)) in &idx_to_loc {
            by_file.entry(file).or_default().insert(rk, idx);
        }

        let mut params_for_idx: FxHashMap<usize, Vec<String>> =
            FxHashMap::with_capacity_and_hasher(wanted.len(), Default::default());
        for (&file, rk_to_idx) in &by_file {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            for stmt in &parsed.syntax().body {
                let Stmt::FunctionDef(func) = stmt else {
                    continue;
                };
                let rk = crate::helpers::range_key(func.name.range());
                let Some(&idx) = rk_to_idx.get(&rk) else {
                    continue;
                };
                let params = &func.parameters;
                let mut names: Vec<String> = Vec::with_capacity(
                    params.posonlyargs.len() + params.args.len() + params.kwonlyargs.len(),
                );
                for p in &params.posonlyargs {
                    names.push(p.parameter.name.id.to_string());
                }
                for p in &params.args {
                    names.push(p.parameter.name.id.to_string());
                }
                for p in &params.kwonlyargs {
                    names.push(p.parameter.name.id.to_string());
                }
                params_for_idx.insert(idx, names);
            }
        }

        Ok(indices
            .into_iter()
            .map(|idx| params_for_idx.remove(&idx).unwrap_or_default())
            .collect())
    }

    /// Batched class-method parameter-name snapshot for each
    /// class-kind index. For each input class, walks its body's
    /// top-level ``FunctionDef`` statements and returns the union of
    /// their parameter names (positional-only + positional-or-keyword
    /// + keyword-only), deduped in first-seen order, with ``self`` and
    /// ``cls`` excluded.
    ///
    /// Indices that don't resolve to a top-level ``ClassDef`` in the
    /// AST surface an empty list at the same position. ``*args`` and
    /// ``**kwargs`` are skipped (pytest never resolves them as
    /// fixture references).
    ///
    /// Used by ``PytestPlugin`` to wire ``class → fixture`` edges for
    /// ``Test*`` classes whose method signatures mention fixtures.
    /// We don't represent class methods as their own graph nodes, so
    /// the class itself is the rendezvous point for any fixture any
    /// method uses.
    pub(crate) fn class_method_parameters(
        &self,
        _py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        let outputs = self.materialized("class_method_parameters")?;
        let nodes = &outputs.builder.nodes;
        let len = nodes.len();
        for &idx in &indices {
            if idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "node index {idx} out of range (len={len})"
                )));
            }
        }
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        // Same rendezvous strategy as ``function_parameters``, but
        // keyed off ``class_by_selection`` (class name-range → idx)
        // instead of ``decl_by_name_range``.
        let wanted: FxHashSet<usize> = indices.iter().copied().collect();
        let mut idx_to_loc: FxHashMap<usize, (File, (u32, u32))> =
            FxHashMap::with_capacity_and_hasher(wanted.len(), Default::default());
        for (&(file, rk), &idx) in &outputs.class_by_selection {
            if wanted.contains(&idx) {
                idx_to_loc.insert(idx, (file, rk));
            }
        }

        let mut by_file: FxHashMap<File, FxHashMap<(u32, u32), usize>> = FxHashMap::default();
        for (&idx, &(file, rk)) in &idx_to_loc {
            by_file.entry(file).or_default().insert(rk, idx);
        }

        let mut params_for_idx: FxHashMap<usize, Vec<String>> =
            FxHashMap::with_capacity_and_hasher(wanted.len(), Default::default());
        for (&file, rk_to_idx) in &by_file {
            let parsed = parsed_module(&self.db, file).load(&self.db);
            for stmt in &parsed.syntax().body {
                let Stmt::ClassDef(cls) = stmt else {
                    continue;
                };
                let rk = crate::helpers::range_key(cls.name.range());
                let Some(&idx) = rk_to_idx.get(&rk) else {
                    continue;
                };
                let mut seen: FxHashSet<String> = FxHashSet::default();
                let mut names: Vec<String> = Vec::new();
                for body_stmt in &cls.body {
                    let Stmt::FunctionDef(method) = body_stmt else {
                        continue;
                    };
                    let params = &method.parameters;
                    for p in params
                        .posonlyargs
                        .iter()
                        .chain(params.args.iter())
                        .chain(params.kwonlyargs.iter())
                    {
                        let n = p.parameter.name.id.as_str();
                        if n == "self" || n == "cls" {
                            continue;
                        }
                        if seen.insert(n.to_string()) {
                            names.push(n.to_string());
                        }
                    }
                }
                params_for_idx.insert(idx, names);
            }
        }

        Ok(indices
            .into_iter()
            .map(|idx| params_for_idx.remove(&idx).unwrap_or_default())
            .collect())
    }
}

impl ProjectContext {
    /// Compile `pattern` once and reuse on subsequent calls. The cache
    /// is bounded by the (small) number of distinct patterns a plugin
    /// run uses, so unbounded growth isn't a concern in practice.
    pub(crate) fn compile_regex(&self, pattern: &str) -> PyResult<regex::Regex> {
        if let Some(cached) = self.regex_cache.lock().get(pattern) {
            return Ok(cached.clone());
        }
        let regex = regex::Regex::new(pattern)
            .map_err(|e| PyValueError::new_err(format!("invalid regex {pattern:?}: {e}")))?;
        self.regex_cache
            .lock()
            .insert(pattern.to_string(), regex.clone());
        Ok(regex)
    }
}

pub(crate) fn build_env_options(
    src_roots: Option<Vec<String>>,
    extra_paths: Option<Vec<String>>,
    python_env: Option<&str>,
    python_version: Option<&str>,
    typeshed: Option<&str>,
) -> PyResult<EnvironmentOptions> {
    Ok(EnvironmentOptions {
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
    })
}

pub(crate) fn make_db(root: &str, env: EnvironmentOptions) -> PyResult<ProjectDatabase> {
    let root = SystemPathBuf::from(root);
    let options = Options {
        environment: Some(env),
        ..Options::default()
    };
    let metadata = ProjectMetadata::from_options(options, root, None, &UseDefaultStrategy)
        .map_err(|e| PyValueError::new_err(format!("invalid configuration: {e:?}")))?;
    let cwd =
        std::env::current_dir().map_err(|e| PyOSError::new_err(format!("cwd unavailable: {e}")))?;
    let cwd = SystemPathBuf::from_path_buf(cwd).map_err(|_| {
        PyValueError::new_err("current working directory is not a valid absolute UTF-8 path")
    })?;
    let system = OsSystem::new(cwd);
    Ok(ProjectDatabase::use_defaults(metadata, system))
}
