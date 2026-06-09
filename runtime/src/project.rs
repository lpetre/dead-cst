//! `Project`, `BuildOutputs`, `ProgressBars`, `ProjectContext`, and the
//! `build_project_graph` pipeline entrypoint. This module owns the
//! Salsa-backed analysis context that the rest of the crate operates on.

use std::str::FromStr;
use std::sync::Arc;

use parking_lot::lock_api::RawRwLockRecursive;
use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard};

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::system::{OsSystem, SystemPathBuf};
use ruff_text_size::TextRange;
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::RangedValue;
use ty_project::watch::{ChangeEvent as TyChangeEvent, ChangedKind, CreatedKind, DeletedKind};
use ty_project::{Db as ProjectDb, ProjectDatabase, ProjectMetadata};
use ty_python_core::program::UseDefaultStrategy;

use crate::builder::{
    apply_prepared_batch, bfs, not_materialized, Direction, GraphBuilder, GraphNode, PartEdges,
    PreparedOp,
};
use crate::file_extraction::{
    file_extraction, imports_local_from_facts, match_callee_chain, match_callee_descriptor,
    CallSiteFact, FileExtraction,
};
use crate::file_payload::{
    file_to_nodes, parent_module_file, resolution_surface_fp, ClassBaseSpec, FileNodes, NodeKind,
    NodeRef,
};
use crate::file_ref_edges::{file_to_refspecs, FileRefSpecs};
use crate::flag_registry::FlagRegistry;
use crate::graph::{intern_kind, NativeGraph, SymbolNode};
use crate::helpers::{
    file_path_string, is_dunder_name, locate_class_seed, range_key, rel_path, resolve_member_def,
    CallArgs, ReloadLog, MODULE_ALIAS_MARKER, NODE_FLAG_ENTRYPOINT, NODE_FLAG_UNRESOLVED,
};
use crate::ingest::emit_visitor_warning;
use crate::progress::{
    ProgressCounters, ProgressHandle, ProgressSnapshot, PHASE_ASSEMBLE, PHASE_ENUM, PHASE_FQNAME,
    PHASE_PLUGINS, PHASE_POPULATE,
};
use crate::query::_path_re_matches;
use crate::refspec::{
    resolve_dynamic, resolve_member, DynamicRef, MemberRef, MemberRole, ResolvedNode, Target,
    Touched,
};
use crate::topic_registry::TopicRegistry;
use compact_str::CompactString;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

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
        let outputs = build_project_graph(
            py,
            &mut self.db,
            false,
            None,
            &counters,
            None,
            GraphBuilder::with_capacity(0),
            ResolveCache::default(),
            None,
        )?;
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
    /// `File → position in the sorted `project_files` list` — the
    /// O(1) hop from a `File`-keyed identity to its node block
    /// (`builder.file_blocks[ordinal]`). With it, per-`DeclKey` /
    /// per-`NodeRef` global maps are unnecessary: every lookup goes
    /// `ordinal → block base + payload-local index`.
    pub(crate) ordinal_of: FxHashMap<File, usize>,
    /// `file_path_string(file) -> File` so seed lookups don't have to
    /// linear-scan `project_files`. Populated alongside ingest.
    pub(crate) path_to_file: FxHashMap<String, File>,
    /// `(file, class_target_range_key) -> class node idx`. Lets
    /// `find_subclasses_of` map ty's `TypeHierarchyClass.selection_range`
    /// back to a graph node in O(1).
    pub(crate) class_by_selection: FxHashMap<(File, (u32, u32)), usize>,
    /// Inverse of `class_by_selection`: `class node idx -> (file, name
    /// TextRange)`. Lets `locate_class_seed` recover a project class's
    /// name-range seed directly — the same `(file, range)` key
    /// `class_by_selection` is built on — without re-parsing the file to
    /// re-find the def by line.
    pub(crate) class_selection_by_idx: FxHashMap<usize, (File, TextRange)>,
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
    /// External seeds (e.g. ``unittest.TestCase``) get their first hop
    /// into the project from the cached ``external_base_children`` index;
    /// once a project class is found, transitive walks use this map.
    pub(crate) children_by_node: FxHashMap<usize, Vec<usize>>,
    /// Resolved *external* base class (a base whose definition lives
    /// outside the project, e.g. ``unittest.TestCase`` in typeshed) ->
    /// direct subclass class graph idxs. Derived in the same `class_bases`
    /// fan-in as `children_by_node`, but keyed by the external class's
    /// resolved decl ``(File, name_range)`` rather than a project node idx.
    ///
    /// The key is the resolved decl's name-token range, produced by
    /// [`crate::helpers::resolve_member_def`] (ty's module resolver plus the
    /// same use-def decomposition the store side runs, which follows re-export
    /// and assignment-alias chains) — the same resolver the external-seed
    /// branch of `find_subclasses_indices_core` runs on its query input (via
    /// `locate_class_seed`). Funneling both the observed bases (via
    /// [`crate::helpers::resolve_base_spec`]) and the query through
    /// `resolve_member_def` collapses sibling spellings (`unittest.TestCase`
    /// vs `unittest.case.TestCase`) and renamed re-exports to a single key
    /// by construction, so the query is an O(1) lookup with no scan.
    pub(crate) external_base_children: FxHashMap<(File, (u32, u32)), Vec<usize>>,
    /// Per-file plugin facts, keyed by topic **name** (the salsa-stable key the
    /// per-file side emits under — see [`crate::native_plugins::FileLocalOp`]).
    /// Collected from every file's `per_file_plugin_ops` in `assemble_graph`,
    /// with each fact's optional decl already translated to a global node idx.
    /// The project-wide plugin pass reads it through
    /// [`crate::native_plugins::plugin_api::PluginCtx::facts_for_topic`]
    /// (handle → name → this map); Python reads it via
    /// `ProjectContext.facts_for_topic`.
    pub(crate) topic_facts: FxHashMap<String, Vec<crate::native_plugins::plugin_api::Fact>>,
    /// On-demand reload tracking for the post-plugin-pass eviction
    /// sweep — see [`crate::helpers::ReloadLog`].
    pub(crate) reload_log: ReloadLog,
    /// Cross-build resolution memo — see [`ResolveCache`]. Taken out
    /// of the previous build's outputs by `materialize` and handed to
    /// the next assemble, which reuses every entry whose read set is
    /// untouched and stores the updated cache here.
    pub(crate) resolve_cache: ResolveCache,
    /// `(resolved, reused)` cross-file resolution counts from the most
    /// recent assemble — the incremental-reuse observability hook
    /// (`ProjectContext._last_resolve_counts`, the timing line, tests).
    pub(crate) resolve_stats: (usize, usize),
}

/// Cross-build memo for the assemble pass's cross-file resolutions —
/// the one piece of builder output that is *not* re-derived from
/// scratch on every materialize. Everything global (node table, edge
/// translation, fqname / class-hierarchy indices, the plugin pass) is
/// refolded from the per-file parts on every build, so nothing can be
/// stale; this cache only short-circuits the `resolve_member` /
/// `resolve_dynamic` / `resolve_member_def` calls whose *inputs*
/// provably didn't change.
///
/// Results are stored in **symbolic** terms ([`ResolvedNode`]: `File`
/// / `DeclKey`, never dense graph indices), so reuse needs no target
/// tracking at all — every build re-translates the memo against its
/// own fresh `ref_to_global`.
///
/// Reuse is gated three ways (any failure ⇒ resolve everything, i.e.
/// exactly today's full build):
///
/// 1. **Event scope.** The batch applied since the last build must be
///    content-only: every event a `Changed` naming a known project
///    file ([`ProjectContext::pending_scope`]). `Created` / `Deleted`
///    / `Rescan` can flip module → file resolution project-wide,
///    which read sets don't (and shouldn't) capture.
/// 2. **Stable file list.** The sorted project file list must be
///    identical to the previous build's (so per-ordinal rows and
///    anchor-directory ids line up).
/// 3. **Per-entry read sets.** A memo entry re-resolves iff a file in
///    its recorded [`Touched`] read set is *effectively changed*:
///    named by the event batch, or its salsa-tracked
///    [`resolution_surface_fp`] differs from the previous build's.
///    The fingerprint is what makes the read sets transitive without
///    a hand-built dependency walk — `from X import *` importers'
///    payloads are derived from X's exports, and salsa recomputes
///    their fingerprints when X changes.
///
/// Entries whose refs disappear from the project keep their slots
/// (ids must stay stable for the retained per-file `spec_ids`); the
/// garbage is bounded by churn and clears with the next full resolve.
///
/// This is a deliberate, sanctioned exception to the crate's "no
/// per-file cache" rule: resolution is anchor-directory-memoized
/// cross-file work that salsa cannot track for us (it must run on
/// plain snapshots, off the tracked-query graph).
#[derive(Default)]
pub(crate) struct ResolveCache {
    /// False until the first build completes (or after an out-of-scope
    /// batch invalidates the cache).
    pub(crate) valid: bool,
    /// The sorted project file list the cache was built against.
    pub(crate) files: Vec<File>,
    /// Per file ordinal: [`resolution_surface_fp`] at build time.
    pub(crate) surface_fp: Vec<u64>,
    /// Per file ordinal: the dense resolution id of each spec row,
    /// parallel to `file_to_refspecs(file).specs` (`u32::MAX` = a
    /// `Local` row with no cross-file resolution).
    pub(crate) spec_ids: Vec<Box<[u32]>>,
    /// `(ref, anchor-directory id) → member_results index`.
    pub(crate) member_keys: FxHashMap<(MemberRef, u32), u32>,
    /// Id-indexed symbolic results + each resolution's read set.
    pub(crate) member_results: Vec<(crate::refspec::ResolvedMember, Touched)>,
    /// Id-indexed: whether any current spec row references the entry.
    /// Stale entries (their last referencing row vanished) keep their
    /// slot — ids must stay stable for the retained per-file
    /// `spec_ids` — but are excluded from re-resolution, external
    /// pre-mint, and slot translation, so they can never leak into
    /// the output; a row referencing one again re-resolves it first.
    pub(crate) member_live: Vec<bool>,
    pub(crate) dynamic_keys: FxHashMap<(DynamicRef, u32), u32>,
    pub(crate) dynamic_results: Vec<(SmallVec<[ResolvedNode; 4]>, Touched)>,
    pub(crate) dynamic_live: Vec<bool>,
    /// Class-base resolution memo (shared with the class-hierarchy
    /// fan-in) plus each entry's read set.
    pub(crate) class_base_memo: ClassBaseMemo,
    pub(crate) class_touched: FxHashMap<(CompactString, CompactString, u32), Touched>,
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_project_graph(
    py: Python<'_>,
    db: &mut ProjectDatabase,
    show_progress: bool,
    stack_size: Option<usize>,
    counters: &Arc<ProgressCounters>,
    per_file_set_id: Option<u32>,
    carry: GraphBuilder,
    cache: ResolveCache,
    scope: Option<Vec<File>>,
) -> PyResult<BuildOutputs> {
    let timing = std::env::var_os("DEAD_CST_TIMING").is_some();

    counters.start_phase(PHASE_ENUM, None);
    let t0 = std::time::Instant::now();
    let mut project_files: Vec<File> = (&db.project().files(db)).into_iter().collect();
    // Canonical, deterministic file order. ty's `files()` set has no
    // stable iteration order, so without this the global node indices
    // assigned in assembly (and everything keyed on them) would vary
    // run-to-run. Paths are unique per `File`, so sorting on the path
    // string is a total order; `sort_by_cached_key` computes each
    // (allocating) path string exactly once.
    project_files.sort_by_cached_key(|&f| file_path_string(db, f));
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

    // Incremental gate + effectively-dirty file set, derived *before*
    // the populate phase so the fan-out (query warm + AST/index
    // eviction — the salsa sweep) can run over exactly the touched
    // set instead of every project file. `Some(dirty)` requires: a
    // content-only event scope, a valid resolve cache, an identical
    // file list, a coherent carried builder, and the compaction
    // trigger not firing (tombstones ≤ live nodes). Dirty = event
    // scope ∪ files whose salsa-tracked resolution-surface
    // fingerprint moved (computing the fingerprints is itself the
    // parallel sweep that re-derives star importers' payloads) ∪
    // `.pyi` stubs of dirty twins (their stub-only ENTRYPOINT flags
    // read the twin's exports).
    let ordinal_of: FxHashMap<File, usize> = project_files
        .iter()
        .enumerate()
        .map(|(i, &f)| (f, i))
        .collect();
    let mut new_fps: Option<Vec<u64>> = None;
    let dirty_files: Option<Vec<bool>> = match &scope {
        Some(scope_files)
            if cache.valid
                && cache.files.as_slice() == project_files.as_slice()
                && carry.file_blocks.len() == project_files.len()
                && carry.part_edges.len() == project_files.len()
                && carry.tombstoned.len() * 2 <= carry.nodes.len() =>
        {
            let pool = DbPool::new(db);
            let fps: Vec<u64> = py.allow_threads(|| {
                run_job(&pool, &project_files, |ldb, &file| {
                    resolution_surface_fp(ldb, file)
                })
            });
            let mut dirty: Vec<bool> = (0..project_files.len())
                .map(|pos| fps[pos] != cache.surface_fp[pos])
                .collect();
            let mut in_scope = true;
            for f in scope_files {
                match ordinal_of.get(f) {
                    Some(&pos) => dirty[pos] = true,
                    // A scoped file that's no longer in the project
                    // contradicts the stable-file-list gate — bail to
                    // the full path.
                    None => {
                        in_scope = false;
                        break;
                    }
                }
            }
            for (&pyi, &twin) in &peer_pyi_to_py {
                if dirty[ordinal_of[&twin]] {
                    dirty[ordinal_of[&pyi]] = true;
                }
            }
            new_fps = Some(fps);
            in_scope.then_some(dirty)
        }
        _ => None,
    };

    // The populate fan-out's worklist: the touched set on an
    // incremental build, every project file otherwise.
    let populate_files: Vec<File> = match &dirty_files {
        Some(dirty) => project_files
            .iter()
            .enumerate()
            .filter(|&(pos, _)| dirty[pos])
            .map(|(_, &f)| f)
            .collect(),
        None => project_files.clone(),
    };

    let progress = ProgressBars::new(show_progress, populate_files.len() as u64);
    counters.start_phase(PHASE_POPULATE, Some(populate_files.len()));

    // (prewarm phase deleted — confirmed redundant after the
    // fan-out refactor. The populate phase below already
    // parallelises parsed_module + semantic_index loading inside
    // its own attach-per-thread scope; pre-warming them in a
    // separate parallel pass was a no-op at every corpus size we
    // measured. See PR #226 follow-up for the A/B data.)

    // Parallel pre-populate: run file_to_nodes and file_to_refspecs
    // as #[salsa::tracked] queries across all
    // project files. Workers are pure-rust (GIL released via
    // py.allow_threads). Each file's queries are attached on
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
        let files_ref: &[File] = &populate_files;
        let counters_ref = Arc::clone(counters);
        let run_populate = move || {
            use salsa::Database as _;
            dist_db.attach(|local_db| {
                let _ = crate::file_payload::project_dist_lookup(local_db);
            });
            // Stride the spawn order with a block transpose so the tasks
            // in flight at any instant come from different packages. The
            // path sort clusters a package's files contiguously, so a
            // naive in-order fan-out runs sibling files concurrently and
            // they contend on the shared upstream salsa queries a package
            // resolves into (its `__init__`, common imports). Walking
            // `lane * cols + col` column-major over a `lanes x cols` grid
            // decorrelates neighbours: consecutive spawns sit `cols` apart
            // in sorted order, i.e. in different packages. This only
            // permutes processing order; each task still records its count
            // at its true sorted `file_idx`, so offsets stay deterministic.
            let lanes = num_workers.max(1);
            let cols = files_ref.len().div_ceil(lanes);
            rayon::scope(|s| {
                for col in 0..cols {
                    for lane in 0..lanes {
                        let file_idx = lane * cols + col;
                        if file_idx >= files_ref.len() {
                            continue;
                        }
                        let file = files_ref[file_idx];
                        let db_tx = db_tx.clone();
                        let db_rx = db_rx.clone();
                        let counters_inner = Arc::clone(&counters_ref);
                        s.spawn(move |_| {
                            let local_db = db_rx.recv().expect("snapshot available");
                            local_db.attach(|local_db| {
                                let _ = file_to_nodes(local_db, file);
                                let _ = file_to_refspecs(local_db, file);
                                // Resolution-surface fingerprint: the
                                // resolve cache's per-file change gate.
                                // Warmed here so the assemble pass's
                                // dirty-set derivation is memo reads.
                                let _ = resolution_surface_fp(local_db, file);
                                // Owned per-file facts for the project-wide
                                // plugin queries. Warmed here, while the AST
                                // is live, so the queries below run as cache
                                // reads.
                                let _ = file_extraction(local_db, file);
                                // Per-file native plugins: warm the salsa-cached
                                // `per_file_plugin_ops(file, set_id)` query on
                                // this worker so the assembly fold below is a
                                // pure cache read. One query runs the whole
                                // plugin set (keyed on `set_id`), so this is O(1)
                                // in plugin count, not one warm per plugin.
                                // `run_on_file` is GIL-free and touches only this
                                // file, so it composes with the GIL-released
                                // fan-out.
                                if let Some(set_id) = per_file_set_id {
                                    let _ = crate::native_plugins::per_file_plugin_ops(
                                        local_db, file, set_id,
                                    );
                                }
                                // Every per-file AST consumer for this file has
                                // now run and memoized its result, so drop the
                                // parsed AST *and* the semantic index's per-scope
                                // analysis data (place tables, use-def maps, AST
                                // ids) rather than let them sit resident through
                                // assembly and the project-wide plugin pass —
                                // that's the memory win, and the index data is
                                // the larger share of it. Assembly, fqname, and
                                // the class hierarchy read only the cached
                                // payloads (`class_bases`, `external_base_children`,
                                // `children_by_node`), never the AST or index, so
                                // the all-files-resident peak never forms. Subclass
                                // resolution can still pull an individual file's
                                // AST/index back on demand to relocate a class
                                // seed — `SemanticIndex::load` lazily rebuilds in
                                // ingredient-reuse mode — and that rare reload is
                                // re-cleared right after the plugin pass.
                                ty_python_core::semantic_index(local_db, file).clear();
                                parsed_module(local_db, file).clear();
                            });
                            db_tx.send(local_db).expect("channel open");
                            counters_inner.populate_inc();
                        });
                    }
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

    // Assembly pass: assign global graph node indices to each
    // NodeRef (offset-derived, so deterministic regardless of
    // worker scheduling), translate edges, mint synthetic External
    // nodes, flush warnings. All salsa-tracked queries are
    // memoized from the parallel pre-populate above so this pass
    // is pure hashmap/index work.
    let t_assemble = std::time::Instant::now();
    // Records every file the assembly/plugin passes reload on demand
    // after the populate-phase eviction; the post-plugin-pass sweep
    // re-clears exactly that set. Created here so assembly's
    // class-base resolution shares the log with the plugin pass (it
    // moves into `BuildOutputs` below).
    let reload_log = ReloadLog::default();
    // Full-path fingerprints, computed after populate (warm payload
    // memos — pure hash work, no AST loads); the incremental path
    // computed them up front as the dirty-set input.
    let new_fps: Vec<u64> = match new_fps {
        Some(fps) => fps,
        None => {
            let pool = DbPool::new(db);
            py.allow_threads(|| {
                run_job(&pool, &project_files, |ldb, &file| {
                    resolution_surface_fp(ldb, file)
                })
            })
        }
    };
    let assembled = assemble_graph(
        py,
        db,
        &project_files,
        &peer_pyi_to_py,
        per_file_set_id,
        counters,
        &reload_log,
        carry,
        cache,
        dirty_files,
        new_fps,
        ordinal_of,
    )?;
    let t_assemble_elapsed = t_assemble.elapsed();
    counters.finish_phase(PHASE_ASSEMBLE);

    let mut builder = assembled.builder;
    let ordinal_of = assembled.ordinal_of;
    let module_nodes_by_file = assembled.module_nodes_by_file;
    let class_by_selection = assembled.class_by_selection;
    let decl_by_name_range = assembled.decl_by_name_range;
    let topic_facts = assembled.topic_facts;
    let resolve_cache = assembled.resolve_cache;
    let resolve_stats = assembled.resolve_stats;
    let assemble_dir_ids = assembled.dir_ids;

    counters.start_phase(PHASE_FQNAME, Some(builder.nodes.len()));
    let t4 = std::time::Instant::now();
    let (decl_by_fqname, module_by_fqname, imports_by_module, children_by_parent) =
        py.allow_threads(|| build_fqname_indices(&builder, counters));
    let t_fqname = t4.elapsed();
    counters.finish_phase(PHASE_FQNAME);

    // Class hierarchy fan-in: fold every file's per-file `class_bases`
    // payload (see `file_payload::FileNodes::class_bases`) into the
    // node-keyed project-wide subclass index. One pass over the
    // per-file payloads + one pass per base; no AST re-walk.
    let t_hierarchy = std::time::Instant::now();
    let ClassHierarchyIndices {
        children_by_node,
        external_base_children,
    } = build_class_hierarchy_indices(
        db,
        &project_files,
        &class_by_selection,
        &resolve_cache.class_base_memo,
        &assemble_dir_ids,
    );
    let t_hierarchy_elapsed = t_hierarchy.elapsed();
    if timing {
        eprintln!(
            "[dead-cst-timing] files={} nodes={} edges={} resolved={} reused={} enum={:?} populate={:?} assemble={:?} class_hier={:?} fqname={:?} total={:?} rss={}MB",
            project_files.len(),
            builder.nodes.len(),
            builder.edges.len(),
            resolve_stats.0,
            resolve_stats.1,
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
    // Invert `class_by_selection` once so per-class name-range lookups
    // (subclass seed resolution) are O(1) cache reads instead of an AST
    // re-parse. It's a bijection — one class node per `(file, range)` —
    // so no value is clobbered.
    let class_selection_by_idx: FxHashMap<usize, (File, TextRange)> = class_by_selection
        .iter()
        .map(|(&(file, (start, end)), &idx)| {
            (idx, (file, TextRange::new(start.into(), end.into())))
        })
        .collect();

    builder.peer_pyi_to_py = peer_pyi_to_py;
    Ok(BuildOutputs {
        builder,
        project_files,
        ordinal_of,
        path_to_file,
        class_by_selection,
        class_selection_by_idx,
        module_nodes_by_file,
        decl_by_name_range,
        decl_by_fqname,
        module_by_fqname,
        imports_by_module,
        children_by_node,
        external_base_children,
        children_by_parent,
        topic_facts,
        reload_log,
        resolve_cache,
        resolve_stats,
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
struct AssembledGraph {
    builder: GraphBuilder,
    ordinal_of: FxHashMap<File, usize>,
    module_nodes_by_file: FxHashMap<File, usize>,
    class_by_selection: FxHashMap<(File, (u32, u32)), usize>,
    decl_by_name_range: FxHashMap<(File, (u32, u32)), usize>,
    topic_facts: FxHashMap<String, Vec<crate::native_plugins::plugin_api::Fact>>,
    /// The updated cross-build resolution memo (its class-base memo +
    /// the per-file anchor-dir ids also feed the post-assembly
    /// class-hierarchy scatter).
    resolve_cache: ResolveCache,
    /// `(resolved, reused)` cross-file resolution counts.
    resolve_stats: (usize, usize),
    dir_ids: Vec<u32>,
}

/// Fan-in pass that converts the salsa-tracked per-file payloads
/// (`file_to_nodes` / `file_to_refspecs`) into a fully-populated
/// `GraphBuilder` of **per-file parts**.
///
/// The builder output is per-file: each file owns a contiguous node
/// block in the dense id space (`file_blocks` / `node_file` are the
/// O(1) indirection) and a translated out-edge list
/// (`GraphBuilder::part_edges`). Both persist across builds — `carry`
/// is the previous build's builder. Every **derived** structure (the
/// global edge vec + adjacency, `ref_to_global`, the decl/fqname
/// indices, topic facts, the class hierarchy) is refolded from the
/// parts on every build, so nothing global is ever patched in place
/// and nothing can be stale.
///
/// Two paths share all the machinery:
///
/// * **Full** (first build, out-of-scope events, changed file list, or
///   the tombstone-compaction trigger): a fresh builder, every file
///   minted and translated — exactly the historical pipeline, plus the
///   parts as a by-product.
/// * **Incremental** (`dirty_files = Some`): clean files keep their
///   node block and part verbatim. Dirty files (event scope ∪
///   fingerprint-moved ∪ `.pyi` stubs of dirty twins) are re-minted —
///   their old block is tombstoned and a fresh block appended at the
///   tail, so dense ids of untouched files never remap. Files whose
///   *cross-file answers* changed (a spec row whose memo entry
///   re-resolved, or a cached triple targeting a tombstoned block) are
///   re-translated only: nodes kept, part edges rebuilt.
///
/// Pass shape (mirrors the historical numbering):
///
/// 1. **Node mint** — bases assigned (prefix sums on the full path;
///    tail appends on the incremental path), parallel fill of the
///    minted blocks, concurrent refold of the `(File, …)`-keyed decl
///    indices over *all* files, overlapped with the resolve arm
///    (incremental via [`ResolveCache`]).
/// 2. **Translation** — memo slots to dense ids, `UNRESOLVED`
///    re-stamps and per-part edge translation for the re-translate
///    set, then the global edge refold (concat parts → sort →
///    [`GraphBuilder::init_edges_bulk`]).
///
/// Warnings buffered in each `FileRefSpecs.warnings` are flushed to
/// the `dead_cst._visitor` Python logger at the end — once per
/// warning, on the main thread, with the GIL we already hold.
#[allow(clippy::too_many_arguments)]
fn assemble_graph<'db>(
    py: Python<'_>,
    db: &'db ProjectDatabase,
    project_files: &[File],
    peer_pyi_to_py: &FxHashMap<File, File>,
    per_file_set_id: Option<u32>,
    counters: &Arc<ProgressCounters>,
    reload_log: &ReloadLog,
    carry: GraphBuilder,
    cache: ResolveCache,
    dirty_files: Option<Vec<bool>>,
    new_fps: Vec<u64>,
    ordinal_of: FxHashMap<File, usize>,
) -> PyResult<AssembledGraph> {
    let t_pass1_start = std::time::Instant::now();
    let n_files = project_files.len();

    // 1a: serial prefetch + base assignment.
    struct FileMint<'db> {
        file: File,
        /// This file's nodes occupy the contiguous global-index block
        /// `[base, base + payload.nodes.len())`; each local ref lands
        /// at `base + local_idx`.
        base: usize,
        payload: &'db FileNodes,
        path_str: String,
        is_stub: bool,
        /// Stub-only ENTRYPOINT context: if this is a .pyi without a
        /// .py twin (`None`), every decl needs ENTRYPOINT. If it has a
        /// twin, only decls whose name isn't in the twin's exports
        /// need it.
        twin_payload: Option<&'db FileNodes>,
        /// The `.py` twin itself, for the peer-edge translation
        /// (`ordinal_of[twin] → twin block base`).
        twin_file: Option<File>,
        /// Whether this file's node block is (re-)minted this build.
        /// Clean incremental files keep their block and skip the fill.
        fill: bool,
    }
    let mut mints: Vec<FileMint<'db>> = Vec::with_capacity(n_files);
    let mut builder;
    // Old-block ids tombstoned *this* build — the dst-scan that pulls
    // `Module`-target consumers into the re-translate set probes it.
    let mut newly_tombstoned: FxHashSet<u32> = FxHashSet::default();
    match &dirty_files {
        None => {
            // Full path: fresh builder, prefix-sum bases, prefilled
            // payload region (parallel fill writes disjoint slices).
            drop(carry);
            let mut total_nodes: usize = 0;
            for &file in project_files.iter() {
                let path_str = file_path_string(db, file);
                let is_stub = path_str.ends_with(".pyi");
                let twin_payload = if is_stub {
                    peer_pyi_to_py
                        .get(&file)
                        .map(|&py_file| file_to_nodes(db, py_file))
                } else {
                    None
                };
                let payload = file_to_nodes(db, file);
                mints.push(FileMint {
                    file,
                    base: total_nodes,
                    payload,
                    path_str,
                    is_stub,
                    twin_payload,
                    twin_file: peer_pyi_to_py.get(&file).copied(),
                    fill: true,
                });
                total_nodes += payload.nodes.len();
            }
            builder = GraphBuilder::with_capacity(total_nodes);
            builder.prefill_payload_region(total_nodes);
            builder.node_file = vec![crate::builder::NO_FILE; total_nodes];
            builder.part_edges = (0..n_files).map(|_| PartEdges::default()).collect();
            for (ordinal, mint) in mints.iter().enumerate() {
                let n = mint.payload.nodes.len();
                builder.file_blocks.push((mint.base as u32, n as u32));
                builder.node_file[mint.base..mint.base + n].fill(ordinal as u32);
            }
        }
        Some(dirty) => {
            // Incremental path: keep the carried builder. Restore the
            // base flags first — that erases the previous plugin
            // pass's ORs (the only post-base mutation) with no undo
            // bookkeeping. Dirty files tombstone their old block and
            // append a fresh one at the tail (ascending ordinal order,
            // so multi-file batches are deterministic).
            builder = carry;
            debug_assert_eq!(builder.base_flags.len(), builder.nodes.len());
            for (idx, &fl) in builder.base_flags.iter().enumerate() {
                builder.nodes[idx].flags = fl;
            }
            let mut append = builder.nodes.len();
            for (pos, &file) in project_files.iter().enumerate() {
                let path_str = file_path_string(db, file);
                let is_stub = path_str.ends_with(".pyi");
                let twin_payload = if is_stub {
                    peer_pyi_to_py
                        .get(&file)
                        .map(|&py_file| file_to_nodes(db, py_file))
                } else {
                    None
                };
                let payload = file_to_nodes(db, file);
                let base = if dirty[pos] {
                    let (ob, ol) = builder.file_blocks[pos];
                    for idx in ob..ob + ol {
                        newly_tombstoned.insert(idx);
                    }
                    builder.tombstone_block(ob, ol);
                    let base = append;
                    append += payload.nodes.len();
                    base
                } else {
                    builder.file_blocks[pos].0 as usize
                };
                mints.push(FileMint {
                    file,
                    base,
                    payload,
                    path_str,
                    is_stub,
                    twin_payload,
                    twin_file: peer_pyi_to_py.get(&file).copied(),
                    fill: dirty[pos],
                });
            }
            // Extend the persistent per-node storage for the appended
            // region; the parallel fill below overwrites every new
            // slot, and the end-of-assemble snapshot rewrites
            // `base_flags` wholesale.
            builder.nodes.resize_with(append, GraphNode::default);
            builder.node_file.resize(append, crate::builder::NO_FILE);
            builder.base_flags.resize(append, 0);
            for (pos, mint) in mints.iter().enumerate() {
                if mint.fill {
                    let n = mint.payload.nodes.len();
                    builder.file_blocks[pos] = (mint.base as u32, n as u32);
                    builder.node_file[mint.base..mint.base + n].fill(pos as u32);
                }
            }
        }
    }
    let total_nodes = builder.nodes.len();
    // Upper bound for the one-entry-per-decl index maps (the full
    // path's exact total; an over-estimate on the incremental path,
    // which is fine for capacity reservation).
    let total_decls = total_nodes.saturating_sub(n_files);

    let mut module_nodes_by_file: FxHashMap<File, usize> =
        FxHashMap::with_capacity_and_hasher(n_files, Default::default());
    // class_by_selection only sees class decls — typically a small
    // fraction of all decls, so size to a modest fraction (1/4) to
    // dodge initial growth without wasting space.
    let mut class_by_selection: FxHashMap<(File, (u32, u32)), usize> =
        FxHashMap::with_capacity_and_hasher(total_decls / 4 + 1, Default::default());
    let mut decl_by_name_range: FxHashMap<(File, (u32, u32)), usize> =
        FxHashMap::with_capacity_and_hasher(total_decls, Default::default());
    let mut all_warnings: Vec<String> = Vec::new();

    // Pass 0: serial-only resolution prep — fetch the spec payloads
    // and per-file plugin ops (salsa cache reads that borrow from the
    // main db handle, so they can't move off this thread), derive the
    // per-file anchor-directory ids, and pre-clone the snapshot pool.
    // Everything row-shaped (the unique-key interning, the class-base
    // gather) runs inside pass 1's resolve arm instead, overlapped
    // with the node fill.
    let t_pass0_start = std::time::Instant::now();
    let mut spec_payloads: Vec<&FileRefSpecs> = Vec::with_capacity(n_files);
    for &file in project_files {
        spec_payloads.push(file_to_refspecs(db, file));
    }
    // Per-file plugin ops: one salsa query per file runs the whole plugin
    // set (keyed on `set_id`) and returns every plugin's ops concatenated in
    // registration order. Prefetched here (a cache read — warmed in populate)
    // so the per-part edge translation below can run db-free on rayon workers.
    let plugin_ops: Vec<&'db [crate::native_plugins::FileLocalOp]> = match per_file_set_id {
        Some(set_id) => project_files
            .iter()
            .map(|&file| crate::native_plugins::per_file_plugin_ops(db, file, set_id).as_slice())
            .collect(),
        None => vec![&[][..]; n_files],
    };

    // Per-file anchor-directory ids. ty's `resolve_module` is anchor-
    // sensitive only through its desperate-resolution fallback, which
    // depends on the importing file's ancestor directories — so files
    // sharing a directory share resolutions. `mints` is path-sorted,
    // so the first file seen for a directory is deterministic; that
    // file is the directory's representative resolution anchor.
    let mut dir_to_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut dir_anchors: Vec<File> = Vec::new();
    let mut dir_ids: Vec<u32> = Vec::with_capacity(n_files);
    for mint in &mints {
        let dir = match mint.path_str.rsplit_once(['/', '\\']) {
            Some((dir, _)) => dir,
            None => "",
        };
        let id = *dir_to_id.entry(dir).or_insert_with(|| {
            dir_anchors.push(mint.file);
            (dir_anchors.len() - 1) as u32
        });
        dir_ids.push(id);
    }

    // Snapshot pool for the resolve fan-out, cloned here on the main
    // thread (a salsa clone must not race a worker mid-query — the
    // populate fan-out documents the same hazard). A recycled pool
    // rather than per-chunk clones because the chunk counts aren't
    // known until the gather runs (on a worker, overlapped with the
    // node fill).
    let pool = DbPool::new(db);

    if std::env::var_os("DEAD_CST_TIMING").is_some() {
        eprintln!("[dead-cst-timing] pass0={:?}", t_pass0_start.elapsed());
    }

    // Disjoint per-file `&mut` windows over the fill region (the whole
    // payload region on the full path; the appended tail on the
    // incremental path). Fill-minted blocks are contiguous and
    // ascending there, so successive `split_at_mut` calls partition it
    // exactly (debug-asserted).
    let fill_region_start = match &dirty_files {
        None => 0,
        Some(_) => mints
            .iter()
            .filter(|m| m.fill)
            .map(|m| m.base)
            .min()
            .unwrap_or(total_nodes),
    };
    let mut fill_mints: Vec<&FileMint<'db>> = Vec::new();
    let mut file_slices: Vec<&mut [GraphNode]> = Vec::new();
    {
        let mut rest: &mut [GraphNode] = &mut builder.nodes[fill_region_start..];
        let mut consumed = fill_region_start;
        for mint in mints.iter().filter(|m| m.fill) {
            debug_assert_eq!(consumed, mint.base);
            let (head, tail) = rest.split_at_mut(mint.payload.nodes.len());
            fill_mints.push(mint);
            file_slices.push(head);
            consumed += mint.payload.nodes.len();
            rest = tail;
        }
    }

    /// Everything the resolve arm hands back across the pass-1 join.
    /// All `'static` row data — no payload borrows escape the arm. The
    /// id-indexed result vecs are *complete*: on an incremental build
    /// they are the previous build's vecs with the dirty / new entries
    /// overwritten in place, so every downstream pass (external mint,
    /// memo-slot translation, stamping) is identical for the full and
    /// incremental paths. `member_redo` / `dynamic_redo` mark the
    /// entries (re-)resolved this build — the consumer detection that
    /// drives per-part re-translation reads them.
    struct ResolveOutputs {
        member_keys: FxHashMap<(MemberRef, u32), u32>,
        member_results: Vec<(crate::refspec::ResolvedMember, Touched)>,
        member_redo: Vec<bool>,
        member_live: Vec<bool>,
        dynamic_keys: FxHashMap<(DynamicRef, u32), u32>,
        dynamic_results: Vec<(SmallVec<[ResolvedNode; 4]>, Touched)>,
        dynamic_redo: Vec<bool>,
        dynamic_live: Vec<bool>,
        parent_results: Vec<Option<File>>,
        spec_ids: Vec<Box<[u32]>>,
        class_base_memo: ClassBaseMemo,
        class_touched: FxHashMap<(CompactString, CompactString, u32), Touched>,
        /// `(resolved, reused)` across the member / dynamic / class
        /// resolution families.
        stats: (usize, usize),
        gather_time: std::time::Duration,
        resolve_time: std::time::Duration,
    }

    // The resolve arm: the row-shaped gather plus every cross-file
    // resolution family — member refs, dynamic refs, the parent-module
    // sweep, class bases — run concurrently with the node fill on the
    // same rayon pool. Touches only salsa snapshots (via the pool),
    // never the builder, so it composes with pass 1's mutable folds.
    let spec_payloads_ref = &spec_payloads;
    let dir_ids_ref = &dir_ids;
    let dir_anchors_ref = &dir_anchors;
    let mints_for_arm = &mints;
    let dirty_ref = &dirty_files;
    let ordinal_of_ref = &ordinal_of;
    let resolve_arm = move || -> ResolveOutputs {
        // Gather: intern every `Member` / `Dynamic` spec row, exactly
        // once, into a dense id (`spec_ids[pos][i]` parallels
        // `spec_payloads[pos].specs`), so the downstream passes —
        // unresolved stamping, edge translation — index a Vec instead
        // of re-hashing string-keyed rows. At tens-of-millions-of-rows
        // scale the repeated row hashing dominated assemble time.
        // Encounter order is deterministic (the file list and each
        // payload's specs are sorted); External mint order does not
        // depend on it — the External pre-mint sorts by (fqname, path)
        // itself. Hot imports repeated across a directory (or
        // project-wide, one entry per directory) resolve once, not
        // once per use site.
        //
        // Incremental builds (`dirty = Some`) skip the gather for
        // clean files entirely — their cached id rows are reused — and
        // hash only the dirty files' rows, interning against the
        // *retained* key maps so existing memo entries keep their ids.
        let t_gather = std::time::Instant::now();
        const LOCAL_SPEC: u32 = u32::MAX;
        let ResolveCache {
            mut member_keys,
            mut member_results,
            member_live: prev_member_live,
            mut dynamic_keys,
            mut dynamic_results,
            dynamic_live: prev_dynamic_live,
            mut class_base_memo,
            mut class_touched,
            spec_ids: mut cached_spec_ids,
            ..
        } = cache;

        let mut spec_ids: Vec<Box<[u32]>> = Vec::with_capacity(spec_payloads_ref.len());
        // Ids needing a (re-)resolve this build, with the owned key +
        // anchor the resolver wants. Indexed scatter: results land at
        // `member_results[id]`, so work order is irrelevant.
        let mut member_work: Vec<(u32, MemberRef, File)> = Vec::new();
        let mut dynamic_work: Vec<(u32, DynamicRef, File)> = Vec::new();
        let mut member_redo: Vec<bool>;
        let mut dynamic_redo: Vec<bool>;
        // This build's liveness: an entry is live iff some current
        // spec row references it. Stale entries are never re-resolved,
        // never pre-minted, never translated — they cannot leak into
        // the output, and a row referencing one again (the file
        // re-adds the import) re-resolves it first via the
        // `!prev_live` resurrection rule below.
        let mut member_live: Vec<bool>;
        let mut dynamic_live: Vec<bool>;

        match dirty_ref {
            None => {
                // Full resolve: drop any retained state and run the
                // sharded concurrent gather. Workers intern each
                // row's (ref, dir) key against a per-worker cache
                // first — hot refs (the same import used across the
                // project) hit it after their first touch — falling
                // back to one of 256 mutexed shards (picked by key
                // hash) that assign a provisional (shard, local) id
                // and keep the borrowed unique keys in encounter
                // order. A serial prefix-sum over shard sizes makes
                // the ids dense, the owned work/key rows are cloned
                // per shard in parallel, and a parallel remap
                // rewrites the per-file rows to final ids. Ids are
                // therefore scheduling-dependent: nothing downstream
                // reads id *order* (results/redo/live are id-indexed
                // scatter targets; edges derive from slot content,
                // not slot index), and the parity suite pins the
                // emitted graph against a fresh build.
                member_keys.clear();
                member_results.clear();
                dynamic_keys.clear();
                dynamic_results.clear();
                class_base_memo.clear();
                class_touched.clear();
                const SHARDS: usize = 256;
                struct Shard<'a, K> {
                    map: FxHashMap<(&'a K, u32), u32>,
                    items: Vec<(&'a K, u32)>,
                }
                /// Provisional id: shard index in the low 8 bits,
                /// shard-local ordinal above (capped at 2^24 entries
                /// per shard, so it can never alias `LOCAL_SPEC` —
                /// and `Local` rows are never remapped anyway).
                fn intern<'a, K: std::hash::Hash + Eq>(
                    shards: &[Mutex<Shard<'a, K>>],
                    cache: &mut FxHashMap<(&'a K, u32), u32>,
                    key: (&'a K, u32),
                ) -> u32 {
                    if let Some(&enc) = cache.get(&key) {
                        return enc;
                    }
                    use std::hash::{Hash, Hasher};
                    let mut hasher = rustc_hash::FxHasher::default();
                    key.hash(&mut hasher);
                    let sh = (hasher.finish() as usize) & (shards.len() - 1);
                    let mut guard = shards[sh].lock();
                    let st = &mut *guard;
                    let local = match st.map.entry(key) {
                        std::collections::hash_map::Entry::Occupied(o) => *o.get(),
                        std::collections::hash_map::Entry::Vacant(v) => {
                            let l = st.items.len() as u32;
                            st.items.push(key);
                            *v.insert(l)
                        }
                    };
                    drop(guard);
                    assert!(local < (1 << 24), "resolve gather shard overflow");
                    let enc = (local << 8) | sh as u32;
                    // Bound the per-worker cache: most refs are
                    // file-local one-offs that never repay caching,
                    // and unbounded growth costs ~tens of MB *per
                    // worker* on many-core machines. The hot refs a
                    // clear evicts re-warm on their next touch.
                    if cache.len() >= 1 << 16 {
                        cache.clear();
                    }
                    cache.insert(key, enc);
                    enc
                }
                fn new_shards<'a, K>() -> Vec<Mutex<Shard<'a, K>>> {
                    (0..SHARDS)
                        .map(|_| {
                            Mutex::new(Shard {
                                map: FxHashMap::default(),
                                items: Vec::new(),
                            })
                        })
                        .collect()
                }
                let mshards: Vec<Mutex<Shard<'_, MemberRef>>> = new_shards();
                let dshards: Vec<Mutex<Shard<'_, DynamicRef>>> = new_shards();
                use rayon::prelude::*;
                let mut rows: Vec<Vec<u32>> = spec_payloads_ref
                    .par_iter()
                    .enumerate()
                    .map_init(
                        || (FxHashMap::default(), FxHashMap::default()),
                        |(mcache, dcache), (pos, payload)| {
                            let dir = dir_ids_ref[pos];
                            payload
                                .specs
                                .iter()
                                .map(|spec| match &spec.target {
                                    Target::Local(_) => LOCAL_SPEC,
                                    Target::Member(m) => intern(&mshards, mcache, (m, dir)),
                                    Target::Dynamic(d) => intern(&dshards, dcache, (d, dir)),
                                })
                                .collect()
                        },
                    )
                    .collect();
                // Dense bases per shard, then clone the owned work /
                // key rows per shard in parallel (the clones are the
                // bulk of the serial cost otherwise); the map inserts
                // stay serial.
                let mshards: Vec<Shard<'_, MemberRef>> =
                    mshards.into_iter().map(|m| m.into_inner()).collect();
                let dshards: Vec<Shard<'_, DynamicRef>> =
                    dshards.into_iter().map(|d| d.into_inner()).collect();
                let mut mbase = [0u32; SHARDS];
                let mut total = 0u32;
                for (sh, st) in mshards.iter().enumerate() {
                    mbase[sh] = total;
                    total += st.items.len() as u32;
                }
                let mut dbase = [0u32; SHARDS];
                let mut dtotal = 0u32;
                for (sh, st) in dshards.iter().enumerate() {
                    dbase[sh] = dtotal;
                    dtotal += st.items.len() as u32;
                }
                type Parts<K> = Vec<(Vec<(u32, K, File)>, Vec<((K, u32), u32)>)>;
                let mparts: Parts<MemberRef> = mshards
                    .par_iter()
                    .enumerate()
                    .map(|(sh, st)| {
                        let mut work = Vec::with_capacity(st.items.len());
                        let mut keys = Vec::with_capacity(st.items.len());
                        for (local, &(m, dir)) in st.items.iter().enumerate() {
                            let id = mbase[sh] + local as u32;
                            work.push((id, m.clone(), dir_anchors_ref[dir as usize]));
                            keys.push(((m.clone(), dir), id));
                        }
                        (work, keys)
                    })
                    .collect();
                let dparts: Parts<DynamicRef> = dshards
                    .par_iter()
                    .enumerate()
                    .map(|(sh, st)| {
                        let mut work = Vec::with_capacity(st.items.len());
                        let mut keys = Vec::with_capacity(st.items.len());
                        for (local, &(d, dir)) in st.items.iter().enumerate() {
                            let id = dbase[sh] + local as u32;
                            work.push((id, d.clone(), dir_anchors_ref[dir as usize]));
                            keys.push(((d.clone(), dir), id));
                        }
                        (work, keys)
                    })
                    .collect();
                member_work.reserve(total as usize);
                member_keys.reserve(total as usize);
                for (work, keys) in mparts {
                    member_work.extend(work);
                    for (k, id) in keys {
                        member_keys.insert(k, id);
                    }
                }
                dynamic_work.reserve(dtotal as usize);
                dynamic_keys.reserve(dtotal as usize);
                for (work, keys) in dparts {
                    dynamic_work.extend(work);
                    for (k, id) in keys {
                        dynamic_keys.insert(k, id);
                    }
                }
                // Remap the per-file rows from provisional to dense
                // ids (the row's variant picks the base table).
                rows.par_iter_mut().enumerate().for_each(|(pos, ids)| {
                    let payload = spec_payloads_ref[pos];
                    for (slot, spec) in ids.iter_mut().zip(payload.specs.iter()) {
                        match &spec.target {
                            Target::Local(_) => {}
                            Target::Member(_) => {
                                *slot = mbase[(*slot & 0xFF) as usize] + (*slot >> 8);
                            }
                            Target::Dynamic(_) => {
                                *slot = dbase[(*slot & 0xFF) as usize] + (*slot >> 8);
                            }
                        }
                    }
                });
                spec_ids.extend(rows.into_iter().map(|v| v.into_boxed_slice()));
                member_results.resize_with(member_work.len(), Default::default);
                dynamic_results.resize_with(dynamic_work.len(), Default::default);
                member_redo = vec![true; member_work.len()];
                dynamic_redo = vec![true; dynamic_work.len()];
                member_live = vec![true; member_work.len()];
                dynamic_live = vec![true; dynamic_work.len()];
            }
            Some(dirty) => {
                // Incremental: a retained entry re-resolves iff its
                // read set touches an effectively-dirty file (files
                // outside the project can't change under the
                // content-only scope gate and count as clean).
                let file_dirty = |f: &File| ordinal_of_ref.get(f).is_some_and(|&pos| dirty[pos]);
                let prev_member_len = member_results.len();
                let prev_dynamic_len = dynamic_results.len();
                member_live = vec![false; prev_member_len];
                dynamic_live = vec![false; prev_dynamic_len];
                // Row gather first, so this build's liveness is known
                // before any redo decision: clean files reuse their
                // cached id rows verbatim (and only mark them live);
                // dirty files re-hash their rows, interning against
                // the retained key maps. A per-build borrowed probe
                // map keeps the owned-key probes (which clone) to one
                // per unique (ref, dir). Brand-new unique refs append
                // a placeholder slot and go straight onto the work
                // list.
                // Parallel liveness sweep over the clean files'
                // cached rows (read-only — the serial loop below only
                // moves the arrays): per-chunk bitmaps OR-reduced,
                // then folded into the live vecs.
                {
                    use rayon::prelude::*;
                    let m_words = prev_member_len.div_ceil(64);
                    let d_words = prev_dynamic_len.div_ceil(64);
                    let cached_ref = &cached_spec_ids;
                    let (m_bits, d_bits) = (0..spec_payloads_ref.len())
                        .into_par_iter()
                        .with_min_len(64)
                        .fold(
                            || (vec![0u64; m_words], vec![0u64; d_words]),
                            |mut acc, pos| {
                                if !dirty[pos] {
                                    let ids = &cached_ref[pos];
                                    // Guard the zip: a clean file's retained row
                                    // must match this build's spec count (clean ⇒
                                    // content unchanged ⇒ identical refspecs memo).
                                    // A mismatch means the dirtiness gate let a
                                    // changed file through — fail loudly instead
                                    // of silently truncating the sweep.
                                    assert_eq!(
                                        ids.len(),
                                        spec_payloads_ref[pos].specs.len(),
                                        "clean-file spec row out of sync at pos {pos}"
                                    );
                                    for (spec, &id) in
                                        spec_payloads_ref[pos].specs.iter().zip(ids.iter())
                                    {
                                        match &spec.target {
                                            Target::Local(_) => {}
                                            Target::Member(_) => {
                                                acc.0[(id as usize) / 64] |= 1u64 << (id % 64);
                                            }
                                            Target::Dynamic(_) => {
                                                acc.1[(id as usize) / 64] |= 1u64 << (id % 64);
                                            }
                                        }
                                    }
                                }
                                acc
                            },
                        )
                        .reduce(
                            || (vec![0u64; m_words], vec![0u64; d_words]),
                            |mut a, b| {
                                for (x, y) in a.0.iter_mut().zip(&b.0) {
                                    *x |= y;
                                }
                                for (x, y) in a.1.iter_mut().zip(&b.1) {
                                    *x |= y;
                                }
                                a
                            },
                        );
                    for (id, live) in member_live.iter_mut().enumerate() {
                        if m_bits[id / 64] & (1u64 << (id % 64)) != 0 {
                            *live = true;
                        }
                    }
                    for (id, live) in dynamic_live.iter_mut().enumerate() {
                        if d_bits[id / 64] & (1u64 << (id % 64)) != 0 {
                            *live = true;
                        }
                    }
                }
                let mut seen_member: FxHashMap<(&MemberRef, u32), u32> = FxHashMap::default();
                let mut seen_dynamic: FxHashMap<(&DynamicRef, u32), u32> = FxHashMap::default();
                for (pos, payload) in spec_payloads_ref.iter().enumerate() {
                    if !dirty[pos] {
                        let ids = std::mem::take(&mut cached_spec_ids[pos]);
                        // Same guard at the adoption point: every later
                        // consumer (redo detection, slot demand, edge
                        // emission) zips or indexes this row against
                        // `payload.specs`, so a length drift here would
                        // corrupt the graph silently.
                        assert_eq!(
                            ids.len(),
                            payload.specs.len(),
                            "clean-file spec row out of sync at pos {pos}"
                        );
                        spec_ids.push(ids);
                        continue;
                    }
                    let dir = dir_ids_ref[pos];
                    let anchor = dir_anchors_ref[dir as usize];
                    let mut ids: Vec<u32> = Vec::with_capacity(payload.specs.len());
                    for spec in payload.specs.iter() {
                        let id = match &spec.target {
                            Target::Local(_) => LOCAL_SPEC,
                            Target::Member(m) => {
                                let id =
                                    *seen_member.entry((m, dir)).or_insert_with(
                                        || match member_keys.get(&(m.clone(), dir)) {
                                            Some(&id) => id,
                                            None => {
                                                let id = member_results.len() as u32;
                                                member_keys.insert((m.clone(), dir), id);
                                                member_results.push(Default::default());
                                                member_live.push(false);
                                                member_work.push((id, m.clone(), anchor));
                                                id
                                            }
                                        },
                                    );
                                member_live[id as usize] = true;
                                id
                            }
                            Target::Dynamic(d) => {
                                let id = *seen_dynamic.entry((d, dir)).or_insert_with(|| {
                                    match dynamic_keys.get(&(d.clone(), dir)) {
                                        Some(&id) => id,
                                        None => {
                                            let id = dynamic_results.len() as u32;
                                            dynamic_keys.insert((d.clone(), dir), id);
                                            dynamic_results.push(Default::default());
                                            dynamic_live.push(false);
                                            dynamic_work.push((id, d.clone(), anchor));
                                            id
                                        }
                                    }
                                });
                                dynamic_live[id as usize] = true;
                                id
                            }
                        };
                        ids.push(id);
                    }
                    spec_ids.push(ids.into_boxed_slice());
                }
                // Redo decision for *retained* ids (new ids are already
                // on the work list): a live entry re-resolves iff its
                // read set touches a dirty file, or it was stale last
                // build (resurrection — its cached answer was allowed
                // to rot while nothing referenced it). Stale entries
                // are never re-resolved: they can't reach the output.
                member_redo = vec![false; member_results.len()];
                for id in 0..prev_member_len {
                    if member_live[id]
                        && (!prev_member_live.get(id).copied().unwrap_or(false)
                            || member_results[id].1 .0.iter().any(&file_dirty))
                    {
                        member_redo[id] = true;
                    }
                }
                dynamic_redo = vec![false; dynamic_results.len()];
                for id in 0..prev_dynamic_len {
                    if dynamic_live[id]
                        && (!prev_dynamic_live.get(id).copied().unwrap_or(false)
                            || dynamic_results[id].1 .0.iter().any(&file_dirty))
                    {
                        dynamic_redo[id] = true;
                    }
                }
                member_redo[prev_member_len..].fill(true);
                dynamic_redo[prev_dynamic_len..].fill(true);
                // Recover the (key, anchor) for redo'd retained ids
                // with one scan over the retained key maps (the only
                // id → key mapping kept).
                for ((m, dir), &id) in &member_keys {
                    if (id as usize) < prev_member_len && member_redo[id as usize] {
                        member_work.push((id, m.clone(), dir_anchors_ref[*dir as usize]));
                    }
                }
                for ((d, dir), &id) in &dynamic_keys {
                    if (id as usize) < prev_dynamic_len && dynamic_redo[id as usize] {
                        dynamic_work.push((id, d.clone(), dir_anchors_ref[*dir as usize]));
                    }
                }
            }
        }

        // Class-base member gather: every build scans all payloads'
        // class_bases (cheap — per class, not per use site) and
        // resolves the keys that are missing from the memo or whose
        // read set touches a dirty file.
        let mut class_work: Vec<(CompactString, CompactString, u32, File)> = Vec::new();
        let mut class_reused: usize = 0;
        {
            let mut seen_class: FxHashSet<(&str, &str, u32)> = FxHashSet::default();
            let file_dirty = |f: &File| match dirty_ref {
                Some(dirty) => ordinal_of_ref.get(f).is_some_and(|&pos| dirty[pos]),
                None => true,
            };
            for (pos, mint) in mints_for_arm.iter().enumerate() {
                let dir = dir_ids_ref[pos];
                let anchor = dir_anchors_ref[dir as usize];
                for (_cls_rk, bases) in &mint.payload.class_bases {
                    for base in bases.iter() {
                        let ClassBaseSpec::ModuleMember { module, name } = base else {
                            continue;
                        };
                        if !seen_class.insert((module.as_str(), name.as_str(), dir)) {
                            continue;
                        }
                        let key = (module.clone(), name.clone(), dir);
                        let redo = match class_touched.get(&key) {
                            Some(touched) => touched.0.iter().any(&file_dirty),
                            None => true,
                        };
                        if redo {
                            class_work.push((key.0, key.1, dir, anchor));
                        } else {
                            class_reused += 1;
                        }
                    }
                }
            }
        }
        let gather_time = t_gather.elapsed();

        let t_resolve = std::time::Instant::now();
        let resolved = member_work.len() + dynamic_work.len() + class_work.len();
        // Live-and-clean entry count: every work item is live, so
        // subtracting the work lists from the live populations leaves
        // exactly the reused entries (stale slots count for nothing).
        let reused = (member_live.iter().filter(|&&l| l).count() - member_work.len())
            + (dynamic_live.iter().filter(|&&l| l).count() - dynamic_work.len())
            + class_reused;
        let member_done: Vec<(u32, crate::refspec::ResolvedMember, Touched)> =
            run_job(&pool, &member_work, |ldb, (id, m, anchor)| {
                let mut touched = Touched::default();
                let res = resolve_member(ldb, *anchor, m, &mut touched);
                (*id, res, touched)
            });
        for (id, res, touched) in member_done {
            member_results[id as usize] = (res, touched);
        }
        let dynamic_done: Vec<(u32, SmallVec<[ResolvedNode; 4]>, Touched)> =
            run_job(&pool, &dynamic_work, |ldb, (id, d, anchor)| {
                let mut touched = Touched::default();
                let res = resolve_dynamic(ldb, *anchor, d, &mut touched);
                (*id, res, touched)
            });
        for (id, res, touched) in dynamic_done {
            dynamic_results[id as usize] = (res, touched);
        }
        let parent_results: Vec<Option<File>> = run_job(&pool, project_files, |ldb, &file| {
            parent_module_file(ldb, file)
        });
        let class_done: Vec<(Option<(File, TextRange)>, Touched)> =
            run_job(&pool, &class_work, |ldb, (module, name, _dir, anchor)| {
                let mut touched = Touched::default();
                let res =
                    resolve_member_def(ldb, module, name, *anchor, 0, reload_log, &mut touched);
                (res, touched)
            });
        for ((res, touched), (module, name, dir, _anchor)) in
            class_done.into_iter().zip(&class_work)
        {
            let key = (module.clone(), name.clone(), *dir);
            class_base_memo.insert(key.clone(), res);
            class_touched.insert(key, touched);
        }

        ResolveOutputs {
            member_keys,
            member_results,
            member_redo,
            member_live,
            dynamic_keys,
            dynamic_results,
            dynamic_redo,
            dynamic_live,
            parent_results,
            spec_ids,
            class_base_memo,
            class_touched,
            stats: (resolved, reused),
            gather_time,
            resolve_time: t_resolve.elapsed(),
        }
    };

    // 1b: parallel node fill (fill-minted blocks only) + concurrent
    // index refolds over ALL files — the `(File, …)`-keyed decl
    // indices are derived state, rebuilt from the per-file payload
    // refs every pass.
    let mints_ref = &mints;
    let counters_ref: &ProgressCounters = counters;
    let module_nodes_mut = &mut module_nodes_by_file;
    let class_by_selection_mut = &mut class_by_selection;
    let decl_by_name_range_mut = &mut decl_by_name_range;
    let fill_mints_ref = fill_mints;
    let ((fill_result, ()), resolve_outputs) = py.allow_threads(move || {
        use rayon::prelude::*;
        rayon::join(
            || {
                rayon::join(
                    move || -> PyResult<()> {
                        fill_mints_ref
                            .into_par_iter()
                            .zip_eq(file_slices)
                            .try_for_each(|(mint, slots)| {
                                for (local_idx, node_data) in mint.payload.nodes.iter().enumerate()
                                {
                                    let node_ref = mint.payload.refs[local_idx];

                                    // Apply stub-only ENTRYPOINT flag if needed.
                                    let mut flags = node_data.flags;
                                    if mint.is_stub && matches!(node_ref, NodeRef::Def(_)) {
                                        let has_runtime = match mint.twin_payload {
                                            Some(py_payload) => {
                                                let name = node_data
                                                    .fqname
                                                    .rsplit('.')
                                                    .next()
                                                    .unwrap_or("");
                                                py_payload.exports_by_name.contains_key(name)
                                            }
                                            None => false,
                                        };
                                        if !has_runtime {
                                            flags |= NODE_FLAG_ENTRYPOINT;
                                        }
                                    }

                                    let node = GraphNode {
                                        fqname: node_data.fqname.to_string(),
                                        kind: intern_kind(node_data.kind.as_static_str())?,
                                        // Every node in this file's payload shares
                                        // the file, so its path is the per-file
                                        // `path_str` — re-derived once per file
                                        // instead of stored per `NodeData`.
                                        path: mint.path_str.clone(),
                                        start_line: node_data.start_line,
                                        start_column: node_data.start_column,
                                        end_line: node_data.end_line,
                                        end_column: node_data.end_column,
                                        flags,
                                        is_overload: node_data.is_overload,
                                        imports: node_data.imports.clone(),
                                    };
                                    slots[local_idx] = node;
                                }
                                counters_ref.assemble_inc();
                                Ok(())
                            })
                    },
                    move || {
                        // One serial insert loop over `Copy` keys —
                        // cheap enough to hide behind the node fill /
                        // resolve arm. (There is no project-wide
                        // NodeRef → idx map, and no per-DeclKey
                        // `global_index` either: symbolic identities
                        // carry their `File`, so every lookup goes
                        // `ordinal_of[file] → block base +
                        // payload-local index`. The position-distinct
                        // invariant — `CLAUDE.md` principle 3 — is
                        // enforced per file by the payload's own
                        // `ref_to_local`, and the `File` inside every
                        // key rules out cross-file collisions by
                        // construction.)
                        for mint in mints_ref {
                            for (local_idx, &node_ref) in mint.payload.refs.iter().enumerate() {
                                let global_idx = mint.base + local_idx;
                                match node_ref {
                                    NodeRef::Module(_) => {
                                        module_nodes_mut.insert(mint.file, global_idx);
                                    }
                                    NodeRef::Def(_) => {
                                        let node_data = &mint.payload.nodes[local_idx];
                                        let rk = node_data.name_range;
                                        if matches!(node_data.kind, NodeKind::Class) {
                                            class_by_selection_mut
                                                .insert((mint.file, rk), global_idx);
                                        }
                                        decl_by_name_range_mut.insert((mint.file, rk), global_idx);
                                    }
                                    NodeRef::StarStmt(_, _) => {
                                        // Star-statement node: not a decl,
                                        // contributes nothing to the decl
                                        // indices.
                                    }
                                }
                            }
                        }
                    },
                )
            },
            resolve_arm,
        )
    });

    // Payload nodes are NOT interned into `node_index`: their identity
    // is `(file, local)` by construction (position-distinct per
    // `CLAUDE.md` principle 3 — `ref_to_global`'s fold debug-asserts
    // it), nothing ever resolves a payload node by content key, and at
    // graph scale the per-node `NodeKey` (a cloned fqname + position)
    // was pure overhead. `node_index` now holds only synthetic nodes
    // (`intern_external` / future plugin mints), whose content key is
    // their identity.
    fill_result?;

    let ResolveOutputs {
        member_keys,
        member_results,
        member_redo,
        member_live,
        dynamic_keys,
        dynamic_results,
        dynamic_redo,
        dynamic_live,
        parent_results,
        spec_ids,
        class_base_memo,
        class_touched,
        stats: resolve_stats,
        gather_time,
        resolve_time,
    } = resolve_outputs;

    if std::env::var_os("DEAD_CST_TIMING").is_some() {
        // pass1 is the wall time of the whole join: node mint + index
        // folds on one arm, the gather + resolves on the other.
        // gather/resolve are the resolve arm's internal split.
        eprintln!(
            "[dead-cst-timing] pass1={:?} (resolve arm: gather={:?} resolve={:?})",
            t_pass1_start.elapsed(),
            gather_time,
            resolve_time,
        );
    }

    // Pass 2: edge translation. All cross-file resolution already ran
    // in pass 1's resolve arm (overlapped with the node fill); this
    // pass folds the results into idx-level memo slots and rebuilds
    // exactly the re-translate set's parts. Dropped triples reference
    // Definitions in non-project files.

    let t_pass2_start = std::time::Instant::now();

    // 2c: serial fold into idx-level memos.
    //
    // External nodes are pre-minted in (fqname, path) order across every
    // resolution output — the same deterministic order the old pipeline
    // used — so duplicate-fqname externals keep the lexicographically
    // smallest path and land at the same graph indices. (On an
    // incremental build the intern is a re-probe of the persistent
    // external map, so existing anchors keep their ids and only
    // genuinely new externals append.)
    {
        let mut externals: Vec<(&str, String, File)> = Vec::new();
        let member_nodes = member_results
            .iter()
            .zip(&member_live)
            .filter(|(_, &live)| live)
            .flat_map(|((res, _), _)| res.targets.iter());
        let dynamic_nodes = dynamic_results
            .iter()
            .zip(&dynamic_live)
            .filter(|(_, &live)| live)
            .flat_map(|((res, _), _)| res.iter());
        for node in member_nodes.chain(dynamic_nodes) {
            if let ResolvedNode::External { fqname, file } = node {
                externals.push((fqname.as_str(), file_path_string(db, *file), *file));
            }
        }
        externals.sort_unstable();
        let mut live_externals: FxHashSet<usize> = FxHashSet::default();
        for (fqname, path, file) in externals {
            live_externals.insert(builder.intern_external(fqname.to_string(), path, file));
        }
        // External GC: an anchor no live resolution targets must not
        // survive into the output (a fresh build wouldn't mint it).
        // Tombstone the slot but keep the `external_nodes` map entry,
        // so a later re-import resurrects the same stable dense id
        // (`intern_external` un-tombstones on a map hit).
        let stale: Vec<usize> = builder
            .external_nodes
            .values()
            .copied()
            .filter(|idx| {
                !live_externals.contains(idx) && !builder.tombstoned.contains(&(*idx as u32))
            })
            .collect();
        for idx in stale {
            builder.tombstoned.insert(idx as u32);
            builder.nodes[idx] = GraphNode::default();
        }
    }

    let t_detect = std::time::Instant::now();
    // Re-translate set: the files whose part must be rebuilt this
    // build. Dirty (re-minted) files always; plus files whose
    // cross-file answers changed — a spec row whose memo entry
    // re-resolved this build (read-set consumers), or a cached triple
    // targeting a block tombstoned this build (`Module`-target edges
    // whose resolution read nothing — e.g. plain `import a` — and the
    // structural submodule → parent edges land here). The scan reads
    // every clean file's rows + cached triples, so it fans out on
    // rayon (read-only, GIL released).
    let retranslate: Vec<bool> =
        match &dirty_files {
            None => vec![true; n_files],
            Some(dirty) => {
                let spec_payloads_ref = &spec_payloads;
                let spec_ids_ref = &spec_ids;
                let member_redo_ref = &member_redo;
                let dynamic_redo_ref = &dynamic_redo;
                let newly_tombstoned_ref = &newly_tombstoned;
                let part_edges_ref = &builder.part_edges;
                py.allow_threads(move || {
                    use rayon::prelude::*;
                    (0..n_files)
                        .into_par_iter()
                        .with_min_len(64)
                        .map(|pos| {
                            if dirty[pos] {
                                return true;
                            }
                            let ids = &spec_ids_ref[pos];
                            spec_payloads_ref[pos].specs.iter().zip(ids.iter()).any(
                                |(spec, &id)| match &spec.target {
                                    Target::Local(_) => false,
                                    Target::Member(_) => member_redo_ref[id as usize],
                                    Target::Dynamic(_) => dynamic_redo_ref[id as usize],
                                },
                            ) || (!newly_tombstoned_ref.is_empty()
                                && part_edges_ref[pos]
                                    .iter_all()
                                    .any(|&(_, d, _)| newly_tombstoned_ref.contains(&d)))
                        })
                        .collect()
                })
            }
        };

    let t_detect = t_detect.elapsed();
    let t_slots = std::time::Instant::now();
    // Slot demand: the memo-slot translations are only ever read by
    // the re-translate set's rows (spec-edge emission + UNRESOLVED
    // stamping), so translate exactly the ids those rows reference —
    // on a one-file edit that's a few dozen entries instead of the
    // whole memo; a full build needs every live entry.
    let (needed_member, needed_dynamic): (Vec<bool>, Vec<bool>) = match &dirty_files {
        None => (member_live.clone(), dynamic_live.clone()),
        Some(_) => {
            let mut nm = vec![false; member_results.len()];
            let mut nd = vec![false; dynamic_results.len()];
            for pos in 0..n_files {
                if !retranslate[pos] {
                    continue;
                }
                for (spec, &id) in spec_payloads[pos].specs.iter().zip(spec_ids[pos].iter()) {
                    match &spec.target {
                        Target::Local(_) => {}
                        Target::Member(_) => nm[id as usize] = true,
                        Target::Dynamic(_) => nd[id as usize] = true,
                    }
                }
            }
            (nm, nd)
        }
    };

    // Memo-slot translation: symbolic targets to this build's dense
    // ids. Every target carries its `File`, so translation is
    // `ordinal_of[file] → block base + payload.ref_to_local[ref]` —
    // no project-wide NodeRef map, no db, no builder mutation (the
    // external pre-mint above already interned every live anchor), so
    // the fan-out is GIL-free and embarrassingly parallel. Un-needed
    // ids keep an empty placeholder to hold the id space aligned.
    let external_nodes_ref = &builder.external_nodes;
    let mints_for_slots = &mints;
    let ordinal_for_slots = &ordinal_of;
    let member_results_ref = &member_results;
    let dynamic_results_ref = &dynamic_results;
    let needed_member_ref = &needed_member;
    let needed_dynamic_ref = &needed_dynamic;
    let (member_slots, dynamic_slots): (Vec<MemberSlot>, Vec<SmallVec<[usize; 4]>>) = py
        .allow_threads(move || {
            use rayon::prelude::*;
            let translate = |node: &ResolvedNode| -> Option<usize> {
                match node {
                    ResolvedNode::Module(f) => {
                        // Every payload's index 0 is its synthetic
                        // module node.
                        ordinal_for_slots
                            .get(f)
                            .map(|&pos| mints_for_slots[pos].base)
                    }
                    ResolvedNode::Def(k) => {
                        let &pos = ordinal_for_slots.get(&k.0)?;
                        let mint = &mints_for_slots[pos];
                        mint.payload
                            .ref_to_local
                            .get(&NodeRef::Def(*k))
                            .map(|&l| mint.base + l as usize)
                    }
                    ResolvedNode::StarStmt(f, rk) => {
                        let &pos = ordinal_for_slots.get(f)?;
                        let mint = &mints_for_slots[pos];
                        mint.payload
                            .ref_to_local
                            .get(&NodeRef::StarStmt(*f, *rk))
                            .map(|&l| mint.base + l as usize)
                    }
                    ResolvedNode::External { fqname, .. } => {
                        external_nodes_ref.get(fqname.as_str()).copied()
                    }
                }
            };
            let member_slots: Vec<MemberSlot> = member_results_ref
                .par_iter()
                .enumerate()
                .with_min_len(2048)
                .map(|(id, (res, _touched))| {
                    if !needed_member_ref[id] {
                        return MemberSlot::default();
                    }
                    let mut idxs: SmallVec<[usize; 4]> = SmallVec::new();
                    for node in &res.targets {
                        if let Some(idx) = translate(node) {
                            idxs.push(idx);
                        }
                    }
                    MemberSlot {
                        idxs,
                        unresolved: res.unresolved,
                        start_file: res.start_file,
                    }
                })
                .collect();
            let dynamic_slots: Vec<SmallVec<[usize; 4]>> = dynamic_results_ref
                .par_iter()
                .enumerate()
                .with_min_len(2048)
                .map(|(id, (nodes, _touched))| {
                    if !needed_dynamic_ref[id] {
                        return SmallVec::new();
                    }
                    let mut idxs: SmallVec<[usize; 4]> = SmallVec::new();
                    for node in nodes {
                        if let Some(idx) = translate(node) {
                            idxs.push(idx);
                        }
                    }
                    idxs
                })
                .collect();
            (member_slots, dynamic_slots)
        });

    let t_slots = t_slots.elapsed();
    // Re-stamp `NodeFlags::UNRESOLVED` on every import alias in the
    // re-translate set whose upstream module didn't resolve — the flag
    // rides the alias node directly. Only `Binding`-role refs are
    // eligible (one flag per alias; a use through it resolves or drops
    // silently). Clean files' stamps survive in their restored base
    // flags; consumers clear-and-set so a flipped resolution outcome
    // lands either way.
    for (pos, payload) in spec_payloads.iter().enumerate() {
        if !retranslate[pos] {
            continue;
        }
        let base = mints[pos].base;
        let ids = &spec_ids[pos];
        for (i, spec) in payload.specs.iter().enumerate() {
            let Target::Member(m) = &spec.target else {
                continue;
            };
            if !matches!(m.role, MemberRole::Binding) {
                continue;
            }
            let node = &mut builder.nodes[base + spec.src as usize];
            if member_slots[ids[i] as usize].unresolved {
                node.flags |= NODE_FLAG_UNRESOLVED;
            } else {
                node.flags &= !NODE_FLAG_UNRESOLVED;
            }
        }
    }

    // Per-file plugin flag + fact ops. Edge ops fold into the part
    // translation below; flags apply to freshly minted blocks only
    // (clean files carry theirs in the restored base flags); facts are
    // derived state, rebuilt from every file's ops each pass.
    //
    // The per-fact work — `file_path_string`, the topic/value `to_string`s,
    // the `Fact` alloc — is fanned out across the snapshot pool, one job per
    // file that emitted any op; the cheap, order-sensitive reduce (OR the
    // flags onto disjoint blocks, push facts into `topic_facts`) stays serial
    // in `pos` order so the output is byte-identical to the serial loop.
    let mut topic_facts: FxHashMap<String, Vec<crate::native_plugins::plugin_api::Fact>> =
        FxHashMap::default();
    if per_file_set_id.is_some() {
        use crate::native_plugins::plugin_api::Fact;
        use crate::native_plugins::FileLocalOp;
        type Applied = (Vec<(usize, u32)>, Vec<(String, Fact)>);
        let op_positions: Vec<usize> = (0..n_files)
            .filter(|&pos| !plugin_ops[pos].is_empty())
            .collect();
        // Fresh snapshot pool: the assemble-wide `pool` was moved into the
        // resolve arm. Cheap (a handful of snapshot clones).
        let plugin_pool = DbPool::new(db);
        let mints_ref = &mints;
        let plugin_ops_ref = &plugin_ops;
        let applied: Vec<Applied> = py.allow_threads(|| {
            run_job(&plugin_pool, &op_positions, |ldb, &pos| {
                let ops = plugin_ops_ref[pos];
                let mint = &mints_ref[pos];
                let len = mint.payload.nodes.len();
                let to_global =
                    |local: u32| ((local as usize) < len).then_some(mint.base + local as usize);
                let mut flag_writes: Vec<(usize, u32)> = Vec::new();
                let mut facts: Vec<(String, Fact)> = Vec::new();
                let mut file_path: Option<String> = None;
                for op in ops {
                    match op {
                        FileLocalOp::Edge { .. } => {}
                        FileLocalOp::Entrypoint { decl_local_idx } => {
                            if mint.fill {
                                if let Some(idx) = to_global(*decl_local_idx) {
                                    flag_writes.push((idx, NODE_FLAG_ENTRYPOINT));
                                }
                            }
                        }
                        FileLocalOp::FlagDecl {
                            decl_local_idx,
                            flags,
                        } => {
                            if mint.fill {
                                if let Some(idx) = to_global(*decl_local_idx) {
                                    flag_writes.push((idx, *flags));
                                }
                            }
                        }
                        FileLocalOp::Fact {
                            topic,
                            decl_local_idx,
                            value,
                        } => {
                            // A fact that pinned a decl is dropped when that
                            // decl doesn't resolve to a global node — same
                            // lenient contract as edge/flag ops.
                            let decl_idx = match decl_local_idx {
                                Some(local) => match to_global(*local) {
                                    Some(global) => Some(global),
                                    None => continue,
                                },
                                None => None,
                            };
                            let path = file_path
                                .get_or_insert_with(|| file_path_string(ldb, mint.file))
                                .clone();
                            facts.push((
                                topic.to_string(),
                                Fact {
                                    path,
                                    decl_idx,
                                    value: value.to_string(),
                                },
                            ));
                        }
                    }
                }
                (flag_writes, facts)
            })
        });
        // Serial reduce in `pos` order (op_positions is ascending), so the
        // fact order — and thus the build — matches the serial loop exactly.
        for (flag_writes, facts) in applied {
            for (idx, bits) in flag_writes {
                builder.nodes[idx].flags |= bits;
            }
            for (topic, fact) in facts {
                topic_facts.entry(topic).or_default().push(fact);
            }
        }
    }

    // 2d: parallel per-part translation for the re-translate set. Each
    // part collects its spec edges, structural edges (decl → module
    // anchors, overload impl → stub anchors, the submodule →
    // parent-package edge), peer .pyi → .py edges (when this part is a
    // stub with a project twin), and its per-file plugin edge ops —
    // then sorts + dedups. Everything read is owned or prefetched, so
    // the fan-out is db-free and GIL-free.
    let member_slots_ref = &member_slots;
    let dynamic_slots_ref = &dynamic_slots;
    let spec_payloads_ref = &spec_payloads;
    let spec_ids_ref = &spec_ids;
    let mints_ref2 = &mints;
    let parent_results_ref = &parent_results;
    let module_nodes_ref = &module_nodes_by_file;
    let ordinal_of_ref2 = &ordinal_of;
    let plugin_ops_ref = &plugin_ops;
    let retranslate_pos: Vec<usize> = (0..n_files).filter(|&pos| retranslate[pos]).collect();
    let t_parts = std::time::Instant::now();
    /// One re-translated part: `(file ordinal, sectioned triples)`.
    type NewPart = (usize, PartEdges);
    let new_parts: Vec<NewPart> = py.allow_threads(move || {
        use rayon::prelude::*;
        retranslate_pos
            .par_iter()
            .map(|&pos| {
                let mint = &mints_ref2[pos];
                let base = mint.base;
                let file = project_files[pos];
                let payload = spec_payloads_ref[pos];
                let ids = &spec_ids_ref[pos];
                let mut part = PartEdges::default();
                part.spec.reserve(payload.specs.len() + 8);
                for (i, spec) in payload.specs.iter().enumerate() {
                    for (s, d, fl) in spec_edges(
                        spec,
                        ids[i],
                        base,
                        file,
                        member_slots_ref,
                        dynamic_slots_ref,
                    ) {
                        part.spec.push((s as u32, d as u32, fl));
                    }
                }
                // Structural edges — fully derivable from the node
                // payloads, deliberately not part of the RefSpec
                // payload. Same section as the spec rows: the old
                // pipeline sorted them into one bulk init together.
                for local in 1..mint.payload.nodes.len() {
                    part.spec.push(((base + local) as u32, base as u32, 0));
                }
                for &(impl_local, stub_local) in mint.payload.overload_anchors.iter() {
                    part.spec.push((
                        (base + impl_local as usize) as u32,
                        (base + stub_local as usize) as u32,
                        0,
                    ));
                }
                if let Some(parent_file) = parent_results_ref[pos] {
                    if let Some(&parent_idx) = module_nodes_ref.get(&parent_file) {
                        part.spec.push((base as u32, parent_idx as u32, 0));
                    }
                }
                part.spec.sort_unstable();
                part.spec.dedup();
                // Peer .pyi → .py reachability edges: for each name
                // both sides export, edge every stub decl to every
                // runtime decl. Own section — the old pipeline's
                // pass 3 extended these after the bulk init.
                if mint.is_stub {
                    if let (Some(py_payload), Some(twin_file)) = (mint.twin_payload, mint.twin_file)
                    {
                        // A per-file local index *is* the offset into
                        // the twin's block — no map probe needed.
                        let twin_base = ordinal_of_ref2
                            .get(&twin_file)
                            .map(|&pos| mints_ref2[pos].base);
                        if let Some(twin_base) = twin_base {
                            for (name, pyi_locals) in &mint.payload.exports_by_name {
                                let Some(py_locals) = py_payload.exports_by_name.get(name) else {
                                    continue;
                                };
                                for &pyi_local in pyi_locals {
                                    let pyi_idx = base + pyi_local as usize;
                                    for &py_local in py_locals {
                                        let py_idx = twin_base + py_local as usize;
                                        part.peer.push((pyi_idx as u32, py_idx as u32, 0));
                                    }
                                }
                            }
                            part.peer.sort_unstable();
                            part.peer.dedup();
                        }
                    }
                }
                // Per-file plugin edge ops (file-local endpoints). Own
                // section, kept in op order and *not* deduplicated —
                // the refold replays them through `add_edge` exactly
                // like the old pipeline's pass 4 did.
                {
                    use crate::native_plugins::FileLocalOp;
                    let len = mint.payload.nodes.len();
                    for op in plugin_ops_ref[pos] {
                        if let FileLocalOp::Edge {
                            src_local_idx,
                            dst_local_idx,
                            flags,
                        } = op
                        {
                            if (*src_local_idx as usize) < len && (*dst_local_idx as usize) < len {
                                part.plugin.push((
                                    (base + *src_local_idx as usize) as u32,
                                    (base + *dst_local_idx as usize) as u32,
                                    *flags,
                                ));
                            }
                        }
                    }
                }
                (pos, part)
            })
            .collect()
    });
    for (pos, part) in new_parts {
        builder.part_edges[pos] = part;
    }

    let t_parts = t_parts.elapsed();
    let t_refold = std::time::Instant::now();
    // Global edge refold, replaying the historical assemble's phases
    // so a full build's edge order is **byte-identical** to the old
    // pipeline (and an incremental build's order follows the same
    // construction over its sparse id space):
    //
    // 1. every part's spec section → one globally-sorted bulk init
    //    (cross-part duplicates are impossible — a triple's src pins
    //    it to one part — so per-part dedup equals global dedup);
    // 2. every part's peer section → sort + dedup + `extend_edges`
    //    (the old pass 3);
    // 3. every part's plugin section → `add_edge` replay in ordinal +
    //    op order (the old pass 4; `add_edge` supplies the dedup).
    {
        builder.reset_derived_edges();
        // Every spec-section triple's src lies inside the owning
        // file's node block (spec rows, decl → module anchors,
        // overload anchors, the submodule → parent edge — all
        // src-anchored), and each section is sorted + deduplicated.
        // Concatenating sections in block-base order therefore yields
        // the globally-sorted, duplicate-free triple list directly —
        // the 5M-row parallel sort and dedup the refold used to pay
        // are gone (`init_edges_bulk` debug-asserts sortedness, so
        // the dev/test builds verify the invariant on every run).
        let mut order: Vec<usize> = (0..n_files).collect();
        order.sort_unstable_by_key(|&pos| builder.file_blocks[pos].0);
        let total: usize = builder.part_edges.iter().map(|p| p.spec.len()).sum();
        let mut triples: Vec<(usize, usize, u8)> = Vec::with_capacity(total);
        for &pos in &order {
            triples.extend(
                builder.part_edges[pos]
                    .spec
                    .iter()
                    .map(|&(s, d, f)| (s as usize, d as usize, f)),
            );
        }
        py.allow_threads(|| {
            builder.init_edges_bulk(triples);
        });

        let mut peer_triples: Vec<(usize, usize, u8)> = Vec::new();
        for part in &builder.part_edges {
            peer_triples.extend(
                part.peer
                    .iter()
                    .map(|&(s, d, f)| (s as usize, d as usize, f)),
            );
        }
        peer_triples.sort_unstable();
        peer_triples.dedup();
        builder.extend_edges(peer_triples);

        let plugin_triples: Vec<(usize, usize, u8)> = builder
            .part_edges
            .iter()
            .flat_map(|p| {
                p.plugin
                    .iter()
                    .map(|&(s, d, f)| (s as usize, d as usize, f))
            })
            .collect();
        for (src, dst, flags) in plugin_triples {
            builder.add_edge(src, dst, flags);
        }
    }

    // Base-flag snapshot: the post-assemble, pre-plugin-pass state the
    // next build restores from (erasing the plugin pass's ORs).
    builder.base_flags = builder.nodes.iter().map(|n| n.flags).collect();

    // Warnings are still serial — they live on `FileRefSpecs` only.
    // Re-emitted for every file each build (matching the historical
    // full-rebuild behavior).
    for payload in &spec_payloads {
        all_warnings.extend(payload.warnings.iter().cloned());
    }

    if std::env::var_os("DEAD_CST_TIMING").is_some() {
        eprintln!(
            "[dead-cst-timing] pass2={:?} (detect={:?} slots={:?} parts={:?} refold={:?}) retranslated={} tombstoned={}",
            t_pass2_start.elapsed(),
            t_detect,
            t_slots,
            t_parts,
            t_refold.elapsed(),
            retranslate.iter().filter(|&&r| r).count(),
            builder.tombstoned.len(),
        );
    }

    // Flush warnings to Python logger from the main thread (we hold
    // the GIL here; workers don't).
    for msg in &all_warnings {
        emit_visitor_warning(py, msg);
    }

    Ok(AssembledGraph {
        builder,
        ordinal_of,
        module_nodes_by_file,
        class_by_selection,
        decl_by_name_range,
        topic_facts,
        resolve_cache: ResolveCache {
            valid: true,
            files: project_files.to_vec(),
            surface_fp: new_fps,
            spec_ids,
            member_keys,
            member_results,
            member_live,
            dynamic_keys,
            dynamic_results,
            dynamic_live,
            class_base_memo,
            class_touched,
        },
        resolve_stats,
        dir_ids,
    })
}

/// One memoized cross-file resolution, translated to dense graph
/// indices for this build. Rebuilt from the symbolic
/// [`ResolveCache`] memo every pass (`assemble_graph` 2c), so the
/// translation can never go stale across re-minted blocks.
#[derive(Default)]
struct MemberSlot {
    idxs: SmallVec<[usize; 4]>,
    unresolved: bool,
    start_file: Option<File>,
}

/// Translate one spec row into its edge triples, given the row's dense
/// resolution id and the owning file's node-block base. The single
/// translation used for every part — full builds and incremental
/// re-translations can't drift.
///
/// Semantics carried here (see the inline comments): `Local` rows skip
/// self-loops; `Binding` member rows are suppressed entirely when the
/// upstream resolves to the importing file itself (mirroring the old
/// `target_file == file` skip — which must apply per importing file,
/// not per memo entry) but *keep* circular-star-reexport self-loops;
/// `Use` member rows and dynamic rows filter `dst == src`.
fn spec_edges(
    spec: &crate::refspec::RefSpec,
    slot_id: u32,
    base: usize,
    file: File,
    member_slots: &[MemberSlot],
    dynamic_slots: &[SmallVec<[usize; 4]>],
) -> SmallVec<[(usize, usize, u8); 4]> {
    let src_idx = base + spec.src as usize;
    let flags = spec.flags;
    let mut out: SmallVec<[(usize, usize, u8); 4]> = SmallVec::new();
    match &spec.target {
        Target::Local(dst_local) => {
            if *dst_local != spec.src {
                out.push((src_idx, base + *dst_local as usize, flags));
            }
        }
        Target::Member(m) => {
            let slot = &member_slots[slot_id as usize];
            // Self-import suppression: a Binding ref whose upstream
            // resolves to the importing file itself emits nothing —
            // the whole emission, decl edges included.
            match m.role {
                MemberRole::Binding => {
                    // Old file_to_edges had no self filter: a circular
                    // star re-export can land the chain walk back on
                    // the alias itself, and that self-loop is kept.
                    if slot.start_file != Some(file) {
                        for &idx in &slot.idxs {
                            out.push((src_idx, idx, flags));
                        }
                    }
                }
                MemberRole::Use => {
                    // Use side mirrors the old emit_edge `dst != owner`
                    // skip.
                    for &idx in &slot.idxs {
                        if idx != src_idx {
                            out.push((src_idx, idx, flags));
                        }
                    }
                }
            }
        }
        Target::Dynamic(_) => {
            for &idx in &dynamic_slots[slot_id as usize] {
                if idx != src_idx {
                    out.push((src_idx, idx, flags));
                }
            }
        }
    }
    out
}

/// Fixed pool of salsa snapshots for the resolve fan-out, cloned on
/// the main thread up front (a salsa clone must not race a worker
/// mid-query — the populate fan-out documents the same hazard) and
/// recycled through a bounded channel. A pool rather than per-chunk
/// clones because the resolve arm's chunk counts aren't known until
/// its gather runs, on a rayon worker.
///
/// Sized to the executing rayon pool, so at most `num_workers` chunk
/// tasks run at once and a `recv` can never starve: every checked-out
/// snapshot belongs to a task that is actively running on some thread
/// and will send it back.
struct DbPool {
    tx: crossbeam_channel::Sender<ProjectDatabase>,
    rx: crossbeam_channel::Receiver<ProjectDatabase>,
    num_workers: usize,
}

impl DbPool {
    fn new(db: &ProjectDatabase) -> Self {
        let num_workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        let (tx, rx) = crossbeam_channel::bounded::<ProjectDatabase>(num_workers);
        for _ in 0..num_workers {
            tx.send(db.clone()).expect("channel open");
        }
        DbPool {
            tx,
            rx,
            num_workers,
        }
    }
}

/// Run `f` over `work` on the current rayon pool, drawing snapshots
/// from the pool per chunk. Chunked 16× finer than the worker count:
/// resolution cost is skewed (one package's imports cluster in
/// encounter order), so fine-grained chunks let work-stealing even out
/// the tail, and the small families (class bases, parent modules)
/// still spread across every core. Results come back in `work` order,
/// so deterministic inputs give deterministic outputs regardless of
/// scheduling. Pure rust — composes inside `py.allow_threads` / a
/// rayon join arm.
fn run_job<T: Sync, R: Send>(
    pool: &DbPool,
    work: &[T],
    f: impl Fn(&ProjectDatabase, &T) -> R + Send + Sync,
) -> Vec<R> {
    use rayon::prelude::*;
    if work.is_empty() {
        return Vec::new();
    }
    let chunk_size = work.len().div_ceil(pool.num_workers * 16).max(1);
    let nested: Vec<Vec<R>> = work
        .par_chunks(chunk_size)
        .map(|chunk| {
            let local_db = pool.rx.recv().expect("snapshot available");
            use salsa::Database as _;
            let out = local_db.attach(|ldb| chunk.iter().map(|item| f(ldb, item)).collect());
            pool.tx.send(local_db).expect("channel open");
            out
        })
        .collect();
    nested.into_iter().flatten().collect()
}

/// Collect the [`PerFilePluginId`](crate::native_plugins::PerFilePluginId)
/// of every registered per-file native plugin, in registration order.
/// These are folded into the graph during the build (parallel warm in
/// the fan-out, serial replay in [`assemble_graph`]) rather than in the
/// post-build plugin pass. Project-wide plugins (including project-wide
/// external dylibs) and non-native Python plugins are skipped here —
/// they still run in [`collect_prepared_plugin_ops`].
/// Collect the registered per-file plugins (in registration order) and intern
/// the ordered id list to a single process-stable `set_id`. `None` when no
/// per-file plugins are registered (the build skips the per-file pass). The
/// whole set then rides one salsa query per file keyed on this `set_id` — see
/// [`crate::native_plugins::per_file_plugin_ops`].
fn extract_per_file_plugin_set(py: Python<'_>, plugins: &[PyObject]) -> Option<u32> {
    use crate::native_plugins::{NativePlugin, NativePluginKind, PerFilePluginId};
    let mut ids: Vec<PerFilePluginId> = Vec::new();
    let mut seen: FxHashSet<PerFilePluginId> = FxHashSet::default();
    // Dedup by id, preserving first-occurrence (registration) order. Two
    // plugins with the same id — e.g. identical `ServerConfig` configs that
    // intern to one `Configured` id, or two `main_block()`s — collapse to one
    // run, exactly as the old `(file, id)` salsa key deduped them.
    let mut push = |id: PerFilePluginId, ids: &mut Vec<PerFilePluginId>| {
        if seen.insert(id) {
            ids.push(id);
        }
    };
    for p in plugins {
        let Ok(native) = p.bind(py).downcast::<NativePlugin>() else {
            continue;
        };
        match &native.borrow().kind {
            NativePluginKind::PerFile(id) => push(*id, &mut ids),
            NativePluginKind::External {
                per_file_id: Some(eid),
                ..
            } => push(PerFilePluginId::External(*eid), &mut ids),
            _ => {}
        }
    }
    (!ids.is_empty()).then(|| crate::native_plugins::register_per_file_set(ids))
}

/// Owned `(module, name, anchor-dir)` → resolved class-base memo.
/// Built by `assemble_graph` (the resolves run in pass 1's resolve
/// arm, overlapped with the node fill) and consumed by the
/// scatter-only [`build_class_hierarchy_indices`].
pub(crate) type ClassBaseMemo =
    FxHashMap<(CompactString, CompactString, u32), Option<(File, TextRange)>>;

/// Project-wide class-hierarchy indices produced in a single pass over
/// the per-file `class_bases` payloads by [`build_class_hierarchy_indices`].
pub(crate) struct ClassHierarchyIndices {
    /// Parent class graph idx → direct subclass graph idxs (project
    /// bases only). See [`BuildOutputs::children_by_node`].
    pub(crate) children_by_node: FxHashMap<usize, Vec<usize>>,
    /// Resolved external base class decl ``(File, name_range)`` → direct
    /// subclass graph idxs. See [`BuildOutputs::external_base_children`].
    pub(crate) external_base_children: FxHashMap<(File, (u32, u32)), Vec<usize>>,
}

/// Fold every file's per-file `class_bases` payload (see
/// [`crate::file_payload::FileNodes::class_bases`]) into the project-wide
/// subclass indices:
///
/// * `children_by_node` — parent class graph idx → direct subclass
///   graph idxs. Drives intra-project BFS for both project and
///   external seeds (once the first hop lands a project class).
/// * `external_base_children` — external base class decl `(File,
///   name_range)` (a base whose definition lives outside the project,
///   e.g. `unittest.TestCase` in typeshed) → direct subclass graph idxs.
///   Lets the external-seed query recover the first project hop from
///   cached payloads instead of re-parsing every file.
///
/// Each base in a payload is a lightweight
/// [`ClassBaseSpec`](crate::file_payload::ClassBaseSpec) — a
/// `LocalClass(range)` for a same-file class or a `ModuleMember{module,
/// name}` symbolic reference — deliberately *not* resolved at store time
/// so the per-file payload stays salsa-invalidation-local (see
/// `build_class_bases`). The `ModuleMember` resolutions (through
/// [`crate::helpers::resolve_member_def`] — ty's module resolver +
/// re-export / assignment-alias following) already ran inside
/// `assemble_graph`'s resolve arm, memoized per `(module, name, anchor
/// directory)` and overlapped with the node fill; this fan-in is a
/// scatter over the handed-back [`ClassBaseMemo`]. Dispatch is a single
/// hashmap probe per resolved base:
///
/// * the key hits `class_by_selection` → it's a project class; record a
///   `children_by_node` edge to that parent (every value in the map is a
///   class node, so no kind check is needed).
/// * the key misses → it's an external base; record the child under
///   `external_base_children` keyed by that same `(File, name_range)`.
///
/// Because assembly and the query side both funnel `ModuleMember` rows
/// through the same `resolve_member_def`, every spelling of one base
/// (direct, aliased, re-exported, sibling-module) collapses to the same
/// `(File, name_range)` key by construction — the query side
/// ([`crate::helpers::locate_class_seed`]) lands on the identical key.
fn build_class_hierarchy_indices(
    db: &ProjectDatabase,
    project_files: &[File],
    class_by_selection: &FxHashMap<(File, (u32, u32)), usize>,
    class_base_memo: &ClassBaseMemo,
    dir_ids: &[u32],
) -> ClassHierarchyIndices {
    let mut by_node: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut external_base_children: FxHashMap<(File, (u32, u32)), Vec<usize>> =
        FxHashMap::default();

    for (pos, &file) in project_files.iter().enumerate() {
        let payload = file_to_nodes(db, file);
        let dir = dir_ids[pos];
        for (cls_rk, bases) in &payload.class_bases {
            let Some(&child_idx) = class_by_selection.get(&(file, *cls_rk)) else {
                continue;
            };
            for base in bases.iter() {
                let resolved = match base {
                    // `LocalClass` resolves inline — the spec range IS
                    // the `(File, name_range)` key.
                    ClassBaseSpec::LocalClass((start, end)) => {
                        Some((file, TextRange::new((*start).into(), (*end).into())))
                    }
                    ClassBaseSpec::ModuleMember { module, name } => class_base_memo
                        .get(&(module.clone(), name.clone(), dir))
                        .copied()
                        .flatten(),
                };
                let Some((base_file, base_range)) = resolved else {
                    continue;
                };
                let base_rk = range_key(base_range);
                match class_by_selection.get(&(base_file, base_rk)) {
                    Some(&parent_idx) => by_node.entry(parent_idx).or_default().push(child_idx),
                    None => external_base_children
                        .entry((base_file, base_rk))
                        .or_default()
                        .push(child_idx),
                }
            }
        }
    }

    for v in by_node.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    for v in external_base_children.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    ClassHierarchyIndices {
        children_by_node: by_node,
        external_base_children,
    }
}

/// Pre-build the fqname -> idx maps used by ``find_declarations``,
/// ``find_module``, ``find_imports_of``, and ``module_surface``. One
/// map-reduce pass over interned nodes (callers should release the
/// GIL); module entries are 1:1 (one module node per fqname) while
/// decl entries (and per-upstream-module import entries) can have
/// multiple binders for the same key — try/except rebinds, conditional
/// re-imports, and multiple ``from X import Y, Z`` aliases all bind
/// into the same upstream module.
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
    use rayon::prelude::*;
    use std::collections::hash_map::Entry;

    /// Per-range accumulator for the rayon `fold`: the four maps, built
    /// over one contiguous slice of `builder.nodes`.
    #[derive(Default)]
    struct FqnameAcc {
        decls: FxHashMap<String, Vec<usize>>,
        modules: FxHashMap<String, usize>,
        imports_by_module: FxHashMap<String, Vec<usize>>,
        children_by_parent: FxHashMap<String, Vec<usize>>,
    }

    /// Merge a right-range multi-map into the left one, appending the
    /// right's index lists after the left's. Entries move (no String
    /// re-clone); an empty left steals the right map outright so
    /// reducing against the identity accumulator is free.
    fn merge_multi(into: &mut FxHashMap<String, Vec<usize>>, from: FxHashMap<String, Vec<usize>>) {
        if into.is_empty() {
            *into = from;
            return;
        }
        into.reserve(from.len());
        for (k, v) in from {
            match into.entry(k) {
                Entry::Occupied(mut e) => e.get_mut().extend(v),
                Entry::Vacant(e) => {
                    e.insert(v);
                }
            }
        }
    }

    counters.fqname_add(builder.nodes.len());
    // Map-reduce over the node table. The fold builds per-range maps
    // (where the String clones + hashing happen, on rayon workers);
    // the reduce merges adjacent ranges left-to-right. rayon's reduce
    // tree combines accumulators in sequence order — `op(left, right)`
    // over adjacent ranges, never reordered — so appending right after
    // left reproduces the serial loop's idx order in every Vec, and
    // `modules.extend` (right wins) reproduces its last-write-wins.
    // Output is deterministic and identical to the serial fold.
    let acc = builder
        .nodes
        .par_iter()
        .enumerate()
        .with_min_len(1024)
        .fold(FqnameAcc::default, |mut acc, (idx, node)| {
            // `@typing.overload`-decorated stubs of an in-file overload group
            // are excluded from the fqname trie so cross-module `from mod
            // import f` resolves to the impl only. Reachability still anchors
            // them via the explicit `impl → stub` anchor edge the
            // assembly pass synthesizes.
            if node.is_overload {
                return acc;
            }
            let mut index_child = false;
            match node.kind {
                "module" => {
                    acc.modules.insert(node.fqname.clone(), idx);
                    index_child = true;
                }
                "function" | "class" | "variable" => {
                    acc.decls.entry(node.fqname.clone()).or_default().push(idx);
                    index_child = true;
                }
                "import" => {
                    acc.decls.entry(node.fqname.clone()).or_default().push(idx);
                    if let Some(import) = node.imports.as_ref() {
                        acc.imports_by_module
                            .entry(import.module.to_string())
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
                acc.children_by_parent.entry(parent).or_default().push(idx);
            }
            acc
        })
        .reduce(FqnameAcc::default, |mut a, b| {
            merge_multi(&mut a.decls, b.decls);
            if a.modules.is_empty() {
                a.modules = b.modules;
            } else {
                a.modules.extend(b.modules);
            }
            merge_multi(&mut a.imports_by_module, b.imports_by_module);
            merge_multi(&mut a.children_by_parent, b.children_by_parent);
            a
        });
    (
        acc.decls,
        acc.modules,
        acc.imports_by_module,
        acc.children_by_parent,
    )
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
    /// :meth:`decls_matching_name`
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
    /// Node-flag registry: engine built-ins plus every flag the registered
    /// plugins declared. Seeded with `with_node_builtins()` at construction
    /// (so a never-materialized ctx still decodes the engine flags) and
    /// overwritten by `materialize` with the post-declaration registry. Read
    /// by the `node_flag` / `default_seed_mask` / `node_flag_registry`
    /// getters and serialized into the graph file.
    pub(crate) node_flag_registry: FlagRegistry,
    /// Edge-flag registry — the edge-space twin of
    /// [`Self::node_flag_registry`] (`u8`-width).
    pub(crate) edge_flag_registry: FlagRegistry,
    /// Topic registry: every topic the registered plugins declared, in
    /// registration order. Empty at construction (topics are plugin-only —
    /// no engine built-ins) and overwritten by `materialize` with the
    /// post-declaration registry. Read by the `topic_registry` /
    /// `facts_for_topic` getters; not serialized (facts are an in-memory
    /// build-time channel, not graph-file content).
    pub(crate) topic_registry: TopicRegistry,
    /// Accumulated change scope since the last completed
    /// ``materialize``: ``Some(files)`` while every applied event has
    /// been a content-only ``Changed`` naming a known project file (the
    /// precondition for [`ResolveCache`] reuse); ``None`` once any
    /// out-of-scope event (``Created`` / ``Deleted`` / ``Rescan``, or an
    /// unknown path) arrives, which makes the next build resolve
    /// everything. Reset to ``Some(vec![])`` after each successful
    /// build; merged (never narrowed) by every ``apply_changes`` call in
    /// between.
    pub(crate) pending_scope: Option<Vec<File>>,
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
            node_flag_registry: FlagRegistry::with_node_builtins(),
            edge_flag_registry: FlagRegistry::with_edge_builtins(),
            topic_registry: TopicRegistry::new(),
            pending_scope: Some(Vec::new()),
        })
    }

    /// Absolute project root passed at construction.
    #[getter]
    pub(crate) fn project_root(&self) -> &str {
        &self.root
    }

    /// Node-flag registry entries `(name, bit, seed, default_on, description)`,
    /// bit-sorted: engine built-ins plus every flag the registered plugins
    /// declared (populated by :meth:`materialize`; engine-only before then).
    /// Threaded into :func:`write_graph` so the graph file records the table.
    /// A method (not a getter) to match the duck-typed `nodes()` / `edges()`
    /// the `write_graph` wrapper calls.
    pub(crate) fn node_flag_registry(&self) -> Vec<(String, u64, bool, bool, String)> {
        flag_registry_tuples(&self.node_flag_registry)
    }

    /// Edge-flag registry entries — the edge-space twin of
    /// :meth:`node_flag_registry`.
    pub(crate) fn edge_flag_registry(&self) -> Vec<(String, u64, bool, bool, String)> {
        flag_registry_tuples(&self.edge_flag_registry)
    }

    /// Default reachability seed mask: the OR of every node-flag bit whose
    /// flag both seeds reachability and is on by default (e.g.
    /// ``engine/entrypoint``, and ``test/testcase`` when a test plugin is
    /// registered). The Python layer uses it as the default ``seed_flags``.
    pub(crate) fn default_seed_mask(&self) -> u32 {
        self.node_flag_registry.default_seed_mask() as u32
    }

    /// The bit a node flag named ``name`` (``owner/name``) occupies, or
    /// ``None`` if undeclared. Engine flags resolve even before materialize;
    /// plugin flags resolve after.
    pub(crate) fn node_flag(&self, name: &str) -> Option<u32> {
        self.node_flag_registry.get(name).map(|bit| bit as u32)
    }

    /// The bit an edge flag named ``name`` occupies (edge-space twin of
    /// :meth:`node_flag`), or ``None`` if undeclared.
    pub(crate) fn edge_flag(&self, name: &str) -> Option<u8> {
        self.edge_flag_registry.get(name).map(|bit| bit as u8)
    }

    /// Topic registry entries ``(name, handle, description)`` in handle
    /// (registration) order: every topic the registered plugins declared
    /// (populated by :meth:`materialize`; empty before then). Topics are a
    /// plugin-only channel, so there are no engine built-ins.
    pub(crate) fn topic_registry(&self) -> Vec<(String, u32, String)> {
        self.topic_registry
            .entries()
            .into_iter()
            .map(|(handle, spec)| (spec.name, handle, spec.description))
            .collect()
    }

    /// Every fact published under topic ``name`` across the project, as
    /// ``(path, decl_idx, value)`` tuples — ``decl_idx`` is the global node
    /// index a fact was pinned to (or ``None``). Empty for an unknown topic or
    /// one nothing published under. Errors only if called before
    /// :meth:`materialize`.
    pub(crate) fn facts_for_topic(
        &self,
        name: &str,
    ) -> PyResult<Vec<(String, Option<usize>, String)>> {
        let outputs = self.materialized("facts_for_topic")?;
        Ok(outputs
            .topic_facts
            .get(name)
            .map(|facts| {
                facts
                    .iter()
                    .map(|f| (f.path.clone(), f.decl_idx, f.value.clone()))
                    .collect()
            })
            .unwrap_or_default())
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
    /// Call before re-running :meth:`materialize` to incrementally
    /// rebuild after a source
    /// edit. Salsa's per-file cache for files whose events didn't
    /// invalidate them stays warm.
    ///
    /// :class:`dead_cst.Analysis.re_materialize` calls this in
    /// combination with :meth:`detect_changes` to autodetect the
    /// dirty set; callers with explicit knowledge of what changed
    /// (e.g. an LSP integration) can build :class:`ChangeEvent` lists
    /// directly via the classmethods on :class:`ChangeEvent`.
    /// Returns ``True`` when the events changed any analysis input —
    /// i.e. the salsa revision advanced. ty's ``apply_changes`` only
    /// bumps file revisions when mtime / size actually changed (and
    /// only re-registers metadata when it differs), so a no-op event
    /// batch — including a ``Rescan`` over an unchanged tree — leaves
    /// the revision untouched and returns ``False``, letting
    /// ``re_materialize`` skip the rebuild entirely.
    pub(crate) fn apply_changes(&mut self, events: Vec<Py<ChangeEvent>>, py: Python<'_>) -> bool {
        use salsa::plumbing::ZalsaDatabase as _;
        // Fold the batch into the pending incremental scope. A batch
        // is in scope iff every event is a content-only `Changed`
        // naming a file the previous build knows; anything else makes
        // the next materialize resolve everything (module → file
        // resolution can flip under file-set / config changes, which
        // per-resolution read sets deliberately don't model).
        let batch_scope: Option<Vec<File>> = {
            let guard = self.outputs.read();
            guard.as_ref().and_then(|outputs| {
                events
                    .iter()
                    .map(|e| {
                        let ev = e.borrow(py);
                        if !matches!(ev.kind, ChangeEventKind::Changed) {
                            return None;
                        }
                        ev.path
                            .as_ref()
                            .and_then(|p| outputs.path_to_file.get(&p.to_string()).copied())
                    })
                    .collect::<Option<Vec<File>>>()
            })
        };
        self.pending_scope = match (self.pending_scope.take(), batch_scope) {
            (Some(mut acc), Some(batch)) => {
                acc.extend(batch);
                Some(acc)
            }
            _ => None,
        };
        let ty_events: Vec<TyChangeEvent> =
            events.iter().map(|e| e.borrow(py).to_ty_event()).collect();
        let before = self.db.zalsa().current_revision();
        self.db.apply_changes(&ty_events, None);
        self.db.zalsa().current_revision() != before
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
    /// subsequent :meth:`materialize` call starts
    /// from zero. Without this, the polling thread on the next run
    /// would observe ``finished=true`` from the prior run and exit
    /// immediately. Called by :meth:`dead_cst.Analysis.re_materialize`
    /// before driving a re-build on the same context.
    pub(crate) fn reset_progress(&mut self) {
        self.progress = Arc::new(ProgressCounters::new());
    }

    /// ``(resolved, reused)`` cross-file resolution counts from the
    /// most recent build — the observability hook for the
    /// [`ResolveCache`]'s incremental reuse (a full build reports
    /// ``reused == 0``; an incremental rebuild after a small content
    /// edit should report ``resolved`` close to the edit's blast
    /// radius). Diagnostic only — not part of the supported surface.
    pub(crate) fn _last_resolve_counts(&self) -> PyResult<(usize, usize)> {
        Ok(self.materialized("_last_resolve_counts")?.resolve_stats)
    }

    /// Sorted dense node indices tombstoned by incremental re-mints:
    /// slots whose file block was replaced by a later
    /// ``re_materialize``. The slots stay in place (live indices never
    /// remap — cached positional indices into unchanged files stay
    /// valid) but are dead: blanked node data, zero flags, no edges,
    /// excluded from every query. Empty after a full build, which
    /// compacts the id space (and runs automatically once tombstones
    /// outnumber live nodes).
    pub(crate) fn tombstoned_indices(&self) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("tombstoned_indices")?;
        let mut v: Vec<usize> = outputs
            .builder
            .tombstoned
            .iter()
            .map(|&i| i as usize)
            .collect();
        v.sort_unstable();
        Ok(v)
    }

    /// Build the project-wide graph, run every registered plugin, then
    /// snapshot the final state.
    ///
    /// The plugin pass fans out across a GIL-free ``rayon`` scope —
    /// one task per project-wide plugin, each owning its own ``Send``
    /// [`FrozenView`] (a cheap salsa db clone + a shared read borrow of
    /// the frozen ``BuildOutputs``). This is the rust mirror of the
    /// per-file fan-out; it replaces the former Python
    /// :class:`concurrent.futures.ThreadPoolExecutor` driver.
    ///
    /// Plugins operate on a **frozen graph**: every plugin's ``run``
    /// sees the same base-graph state, the emitted ops accumulate in
    /// registration order, and a single end-of-pass write-lock window
    /// folds the lot into the graph. A plugin's own emissions are
    /// invisible to its own queries (and to other plugins' queries)
    /// during ``run``.
    pub(crate) fn materialize(slf: Py<Self>, py: Python<'_>) -> PyResult<()> {
        let show_progress = slf.borrow(py).show_progress;
        let stack_size = slf.borrow(py).stack_size;
        let counters = Arc::clone(&slf.borrow(py).progress);
        let build_result = {
            let mut this = slf.borrow_mut(py);
            let per_file_set_id = extract_per_file_plugin_set(py, &this.plugins);
            // Hand the previous build's resolve cache + the
            // accumulated change scope to the new build. The cache
            // moves out (the old outputs keep an invalid Default);
            // taking the scope leaves `None`, so a failed build can't
            // under-scope the retry — the post-build reset below only
            // runs on success.
            let scope = this.pending_scope.take();
            let (carry, cache) = {
                let mut guard = this.outputs.write();
                match guard.as_mut() {
                    Some(o) => (
                        std::mem::replace(&mut o.builder, GraphBuilder::with_capacity(0)),
                        std::mem::take(&mut o.resolve_cache),
                    ),
                    None => (GraphBuilder::with_capacity(0), ResolveCache::default()),
                }
            };
            build_project_graph(
                py,
                &mut this.db,
                show_progress,
                stack_size,
                &counters,
                per_file_set_id,
                carry,
                cache,
                scope,
            )
        };
        let outputs = match build_result {
            Ok(o) => o,
            Err(e) => {
                counters.mark_finished();
                return Err(e);
            }
        };
        // Swap in the freshly built outputs and offload the *previous*
        // ``BuildOutputs`` to a detached thread for dropping. It's ``None``
        // on the first build and ``Some`` on every ``re_materialize``; the
        // old graph (nodes, edges, adjacency, the fqname/base indices) is a
        // large web of allocations whose dealloc would otherwise run on this
        // GIL-holding thread and stall the caller. The swap itself is a
        // pointer move under the write lock; the drop happens after the lock
        // is released. ``BuildOutputs`` is ``Send + 'static`` (it already
        // rides the ``Arc<RwLock<…>>`` here and the rayon plugin scope below),
        // so the move is sound. If the OS can't spawn the thread, the unrun
        // closure is dropped and ``old`` deallocs inline — correct, just not
        // offloaded.
        let previous = slf.borrow(py).outputs.write().replace(outputs);
        if let Some(old) = previous {
            let _ = std::thread::Builder::new()
                .name("dead-cst-drop-outputs".into())
                .spawn(move || drop(old));
        }
        // The build completed: the new outputs (and their resolve
        // cache) reflect every change applied so far, so the next
        // incremental scope starts empty.
        slf.borrow_mut(py).pending_scope = Some(Vec::new());

        // Project-wide plugin pass — fan out across a GIL-free
        // ``rayon`` scope, one task per plugin, mirroring the per-file
        // query fan-out. Each worker
        // owns a cheap salsa clone of the db wrapped in a
        // [`FrozenView`], so every plugin query runs with the GIL
        // released; the frozen ``BuildOutputs`` is shared read-only
        // across workers. Per-file plugins already folded their ops in
        // during the build, so their jobs are no-ops here. Collected
        // ops fold into the graph in one end-of-pass apply, in
        // registration order — the same frozen-graph contract the old
        // Python ``ThreadPoolExecutor`` driver honoured.
        let (jobs, plugin_names, node_reg, edge_reg, topic_reg) =
            match crate::native_plugins::extract_plugin_jobs(py, &slf.borrow(py).plugins) {
                Ok(extracted) => extracted,
                Err(e) => {
                    counters.mark_finished();
                    return Err(e);
                }
            };
        let n_plugins = jobs.len();
        let plugin_bar = ProgressBars::plugin_bar(show_progress, n_plugins as u64);
        let timing_names = plugin_names.clone();
        let t_plugin_pass = std::time::Instant::now();
        counters.init_plugin_slots(plugin_names);
        counters.start_phase(PHASE_PLUGINS, Some(n_plugins));
        let num_workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);

        // Clone the salsa db + project root + outputs handle out of the
        // pyclass *before* releasing the GIL — no ``PyRef`` may cross
        // the ``allow_threads`` boundary. Each worker re-clones the db
        // (a cheap salsa snapshot) and shares ``&BuildOutputs``.
        let base_db = slf.borrow(py).db.clone();
        let root = slf.borrow(py).root.clone();
        let outputs_lock = Arc::clone(&slf.borrow(py).outputs);

        // Collect ``(idx, result)`` pairs into a ``std::sync::Mutex``
        // (fully qualified — ``project.rs`` imports ``parking_lot::Mutex``
        // for the pyclass guards). ``&Mutex<Vec<…>>`` is ``Send`` because
        // ``Mutex`` is ``Sync``, so workers push through a shared borrow;
        // registration order is restored by sorting on ``idx`` after.
        let results: std::sync::Mutex<Vec<(usize, PyResult<Vec<PreparedOp>>)>> =
            std::sync::Mutex::new(Vec::with_capacity(n_plugins));
        let plugin_walls: std::sync::Mutex<Vec<(usize, std::time::Duration)>> =
            std::sync::Mutex::new(Vec::with_capacity(n_plugins));
        {
            let counters_ref = &*counters;
            let bar_ref = &plugin_bar;
            let root_ref: &str = &root;
            let jobs_ref = &jobs;
            let results_ref = &results;
            let plugin_walls_ref = &plugin_walls;
            // Lend the frozen registries into each worker's `FrozenView`.
            // `&FlagRegistry` is `Send` (the registry is `Sync`), so the refs
            // ride the `allow_threads` boundary the way `root_ref` does; the
            // owned registries stay on this stack and move into the pyclass
            // after the pass.
            let node_reg_ref = &node_reg;
            let edge_reg_ref = &edge_reg;
            // The topic registry rides the same way — `&TopicRegistry` is
            // `Send` (the registry is `Sync`); plugins resolve topic handles
            // and read collected facts off it inside the GIL-free pass.
            let topic_reg_ref = &topic_reg;
            // ``base_db`` + ``outputs_lock`` move *into* the closure: a
            // ``&ProjectDatabase`` is ``!Send`` (salsa's ``ZalsaLocal`` is
            // ``!Sync``), so — like every GIL-free per-file scan here — we
            // move an owned db in and ``.clone()`` a cheap snapshot per worker.
            py.allow_threads(move || {
                let guard = outputs_lock.read();
                let outputs_ref: &BuildOutputs =
                    guard.as_ref().expect("materialize stored outputs above");
                // `move` so it owns `base_db` (`ProjectDatabase` is `Send`,
                // a borrow would not be) and thus satisfies `install`'s
                // `Send` bound — same shape as `run_populate`.
                let run_plugins = move || {
                    rayon::scope(move |s| {
                        for (idx, job) in jobs_ref.iter().enumerate() {
                            let db_t = base_db.clone();
                            s.spawn(move |_| {
                                counters_ref.plugin_started(idx);
                                let view = FrozenView::new(
                                    db_t,
                                    outputs_ref,
                                    root_ref,
                                    node_reg_ref,
                                    edge_reg_ref,
                                    topic_reg_ref,
                                );
                                let mut sink: Vec<PreparedOp> = Vec::new();
                                let t_job = std::time::Instant::now();
                                let res = job.run(&view, &mut sink).map(|()| sink);
                                plugin_walls_ref
                                    .lock()
                                    .unwrap()
                                    .push((idx, t_job.elapsed()));
                                results_ref.lock().unwrap().push((idx, res));
                                counters_ref.plugin_finished(idx);
                                counters_ref.plugins_inc();
                                bar_ref.inc(1);
                            });
                        }
                    });
                };
                // Honour `set_stack_size` here too: subclass-resolution
                // plugins recurse over the class hierarchy, so the deep
                // stack lives in this pass, not only the populate phase.
                match stack_size {
                    Some(n) => {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(num_workers)
                            .stack_size(n)
                            .build()
                            .expect("build rayon thread pool");
                        pool.install(run_plugins);
                    }
                    None => run_plugins(),
                }
            });
        }
        plugin_bar.finish_and_clear();

        // Fold every plugin's ops into the graph in registration order
        // (restored by sorting on the worker-recorded idx). First error
        // wins and aborts the apply — matching the serial path's "fail the
        // whole batch" semantics (earlier plugins' ops are dropped on error).
        let mut collected = results.into_inner().expect("plugin results mutex poisoned");
        collected.sort_by_key(|(idx, _)| *idx);
        let t_apply = std::time::Instant::now();
        let plugin_result = (|| -> PyResult<()> {
            let mut prepared: Vec<PreparedOp> = Vec::new();
            for (_, res) in collected {
                prepared.extend(res?);
            }
            apply_prepared_batch(&slf, py, prepared)
        })();
        if std::env::var_os("DEAD_CST_TIMING").is_some() && n_plugins > 0 {
            let mut walls = plugin_walls
                .into_inner()
                .expect("plugin walls mutex poisoned");
            walls.sort_by_key(|(idx, _)| *idx);
            let per: Vec<String> = walls
                .iter()
                .map(|(idx, d)| format!("{}={:?}", timing_names[*idx], d))
                .collect();
            eprintln!(
                "[dead-cst-timing] plugins={:?} (apply={:?}) [{}]",
                t_plugin_pass.elapsed(),
                t_apply.elapsed(),
                per.join(" ")
            );
        }
        counters.finish_phase(PHASE_PLUGINS);
        counters.mark_finished();
        plugin_result?;

        // Persist the post-declaration registries on the pyclass so the
        // `node_flag` / `default_seed_mask` / `*_flag_registry` getters and
        // the graph-file writer see the plugin-allocated bits. The parallel
        // pass's borrows have ended (the `allow_threads` block above returned),
        // so the owned registries are free to move in.
        {
            let mut this = slf.borrow_mut(py);
            this.node_flag_registry = node_reg;
            this.edge_flag_registry = edge_reg;
            this.topic_registry = topic_reg;
        }

        // The assembly and plugin passes can still reload individual files
        // on demand: class-base resolution and subclass resolution's
        // `locate_class_seed` resolve members through ty
        // (`resolve_member_def` → the module resolver + use-def chain),
        // which loads the target module's `parsed_module` and rebuilds its
        // semantic index's per-scope data, repopulating those files' salsa
        // slots. Every such reload funnels through
        // `resolve_member_in_file`, which records the file in
        // `outputs.reload_log` — so the sweep re-clears exactly the touched
        // set (typically a handful of base-defining files) instead of
        // probing every project file's slots, and the post-build resident
        // set stays at the lean post-populate level.
        {
            let this = slf.borrow(py);
            let outputs_ref = this.outputs.read();
            if let Some(outputs) = outputs_ref.as_ref() {
                for file in outputs.reload_log.drain() {
                    ty_python_core::semantic_index(&this.db, file).clear();
                    parsed_module(&this.db, file).clear();
                }
            }
        }

        // No snapshot: the live graph is queried through the context
        // (`nodes()` / `edges()` / the index-returning queries).
        // Materializing a `NativeGraph` here cost one `Py<SymbolNode>`
        // per node on every (re)build — many GB of Python objects at
        // scale — and the only caller discarded it.
        Ok(())
    }

    /// Atomic snapshot of the build-progress counters as a Python
    /// dict. Called from the Python polling thread at ~100 ms cadence
    /// to drive structured progress events. Field semantics live on
    /// :class:`crate::progress::ProgressCounters`.
    ///
    /// Reading is GIL-bound + non-blocking — every counter is a
    /// relaxed atomic load. Calling this before
    /// :meth:`materialize` returns the
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
    /// points at them.
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
        let outputs = self.materialized("find_nodes_matching_specs_indices")?;
        Ok(find_nodes_matching_specs_indices_in(
            &outputs,
            project_root,
            &compiled,
            &str_specs,
            &abs_paths,
        ))
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
    /// included).
    pub(crate) fn find_declarations_indices(&self, fqname: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_declarations_indices")?;
        Ok(find_declarations_indices_in(&outputs, fqname))
    }

    /// O(1) module-by-fqname lookup.
    pub(crate) fn find_module_idx(&self, fqname: &str) -> PyResult<Option<usize>> {
        let outputs = self.materialized("find_module_idx")?;
        Ok(find_module_idx_in(&outputs, fqname))
    }
}

#[pymethods]
impl ProjectContext {
    /// O(1) path-to-module lookup.
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
    /// + every transitive descendant idx.
    pub(crate) fn module_surface_indices(&self, module_fqn: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("module_surface_indices")?;
        Ok(module_surface_indices_in(&outputs, module_fqn))
    }

    /// Bulk form: resolve every fqname in ``module_fqns`` in a single
    /// scan. Returns a dict keyed by input fqname; missing modules
    /// map to empty lists — the bulk shape is the reason this exists
    /// (one materialize check + one scan instead of N).
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
    /// excludes module children.
    pub(crate) fn find_module_top_level_decls_indices(
        &self,
        module_fqn: &str,
    ) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("find_module_top_level_decls_indices")?;
        Ok(find_module_top_level_decls_indices_in(&outputs, module_fqn))
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
    pub(crate) fn find_module_dunder_all_exports_indices(
        &self,
        module_fqn: &str,
    ) -> PyResult<Option<Vec<usize>>> {
        let outputs = self.materialized("find_module_dunder_all_exports_indices")?;
        Ok(find_module_dunder_all_exports_indices_in(
            &self.db, &outputs, module_fqn,
        ))
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
        Ok(find_literal_list_entries_in(&self.db, &outputs, var_fqn))
    }
}

#[pymethods]
impl ProjectContext {
    /// Every node whose ``path`` starts with ``path_prefix``.
    pub(crate) fn decls_under_indices(&self, path_prefix: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("decls_under_indices")?;
        Ok(decls_under_indices_in(&outputs, path_prefix))
    }

    /// Every node whose ``path`` contains ``substring`` anywhere.
    pub(crate) fn decls_matching_indices(&self, substring: &str) -> PyResult<Vec<usize>> {
        let outputs = self.materialized("decls_matching_indices")?;
        Ok(decls_matching_indices_in(&outputs, substring))
    }

    /// Every top-level decl (function / class / variable / import /
    /// type_alias) whose simple name matches ``pattern``.
    pub(crate) fn decls_matching_name_indices(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let regex = self.compile_regex(pattern)?;
        let outputs = self.materialized("decls_matching_name_indices")?;
        Ok(decls_matching_name_indices_in(&outputs, &regex))
    }

    /// Forward closure: every node reachable from ``root_idx`` by
    /// following graph edges. Takes a positional index into
    /// ``ctx.nodes()`` and returns descendant indices. ``skip_flags``
    /// filters out edges whose flag mask matches (pass
    /// ``EdgeFlags.DEAD_BRANCH.value`` for strict reachability excluding
    /// dead branches). Raises :class:`IndexError` when ``root_idx`` is
    /// out of range.
    #[pyo3(signature = (root_idx, *, skip_flags = 0))]
    pub(crate) fn descendants_indices(
        &self,
        root_idx: usize,
        skip_flags: u8,
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

    /// Reverse closure: every node that can reach ``decl_idx`` by
    /// following graph edges. Takes a positional index into
    /// ``ctx.nodes()`` and returns ancestor indices. Used for
    /// ``why-alive`` and blast-radius scoping. ``skip_flags`` works the
    /// same as in :meth:`descendants_indices`. Raises :class:`IndexError`
    /// when ``decl_idx`` is out of range.
    #[pyo3(signature = (decl_idx, *, skip_flags = 0))]
    pub(crate) fn ancestors_indices(
        &self,
        decl_idx: usize,
        skip_flags: u8,
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
    /// source index.
    #[pyo3(signature = (idx, *, skip_flags = 0))]
    pub(crate) fn direct_predecessors_idx(
        &self,
        idx: usize,
        skip_flags: u8,
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
        skip_flags: u8,
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
    /// block's range.
    pub(crate) fn find_main_blocks_indices(&self) -> PyResult<Vec<(usize, Vec<usize>)>> {
        let outputs = self.materialized("find_main_blocks_indices")?;
        Ok(find_main_blocks_indices_in(&self.db, &outputs))
    }
}

/// Shared one-hop reverse-adjacency walk used by
/// :meth:`ProjectContext::direct_predecessors` and
/// :meth:`direct_predecessors_idx`.
fn direct_predecessors_idxs_in(outputs: &BuildOutputs, idx: usize, skip_flags: u8) -> Vec<usize> {
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

/// Shared O(1) probe behind :meth:`ProjectContext::has_imports_of`.
fn has_imports_of_in(outputs: &BuildOutputs, module_name: &str) -> bool {
    if outputs
        .imports_by_module
        .get(module_name)
        .is_some_and(|v| !v.is_empty())
    {
        return true;
    }
    // A submodule import (`from unittest.case import …`) counts as importing
    // the package (`unittest`), per the documented contract. Match on the
    // dot boundary so `flask_login` doesn't satisfy a probe for `flask`.
    let prefix = format!("{module_name}.");
    outputs
        .imports_by_module
        .iter()
        .any(|(k, v)| !v.is_empty() && k.starts_with(prefix.as_str()))
}

/// Shared import-by-module lookup behind
/// :meth:`ProjectContext::find_imports_of_indices`.
fn find_imports_of_indices_in(outputs: &BuildOutputs, module_name: &str) -> Vec<usize> {
    // Exact module plus any submodule (dot boundary), per the documented
    // contract — `from unittest.case import …` imports the `unittest` tree.
    let prefix = format!("{module_name}.");
    let mut out: Vec<usize> = Vec::new();
    for (k, idxs) in &outputs.imports_by_module {
        if k.as_str() == module_name || k.starts_with(prefix.as_str()) {
            out.extend(idxs.iter().copied());
        }
    }
    out
}

/// Shared O(1) module-by-fqname lookup behind
/// :meth:`ProjectContext::find_module_idx`.
fn find_module_idx_in(outputs: &BuildOutputs, fqname: &str) -> Option<usize> {
    outputs.module_by_fqname.get(fqname).copied()
}

/// Shared one-level surface lookup behind
/// :meth:`ProjectContext::find_module_top_level_decls_indices`: the
/// module's immediate non-module children via the ``children_by_parent``
/// fqname tree.
fn find_module_top_level_decls_indices_in(outputs: &BuildOutputs, module_fqn: &str) -> Vec<usize> {
    if !outputs.module_by_fqname.contains_key(module_fqn) {
        return Vec::new();
    }
    let mut out: Vec<usize> = Vec::new();
    if let Some(children) = outputs.children_by_parent.get(module_fqn) {
        for &idx in children {
            if outputs.builder.nodes[idx].kind == "module" {
                continue;
            }
            out.push(idx);
        }
    }
    out
}

/// Shared path-prefix scan behind
/// :meth:`ProjectContext::decls_under_indices`.
fn decls_under_indices_in(outputs: &BuildOutputs, path_prefix: &str) -> Vec<usize> {
    outputs
        .builder
        .nodes
        .iter()
        .enumerate()
        .filter(|(_i, n)| n.path.starts_with(path_prefix))
        .map(|(i, _n)| i)
        .collect()
}

/// Shared path-substring scan behind
/// :meth:`ProjectContext::decls_matching_indices`.
fn decls_matching_indices_in(outputs: &BuildOutputs, substring: &str) -> Vec<usize> {
    outputs
        .builder
        .nodes
        .iter()
        .enumerate()
        .filter(|(_i, n)| n.path.contains(substring))
        .map(|(i, _n)| i)
        .collect()
}

/// Shared simple-name regex scan behind
/// :meth:`ProjectContext::decls_matching_name_indices`. The regex is
/// compiled by the caller (the pymethod surfaces a `ValueError`; the
/// `FrozenView` facade treats a bad pattern as "no matches").
fn decls_matching_name_indices_in(outputs: &BuildOutputs, regex: &regex::Regex) -> Vec<usize> {
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
    out
}

/// Shared spec-matcher behind
/// :meth:`ProjectContext::find_nodes_matching_specs_indices`. Regexes
/// are pre-compiled (and `^`-anchored) by the caller.
fn find_nodes_matching_specs_indices_in(
    outputs: &BuildOutputs,
    project_root: &str,
    compiled: &[regex::Regex],
    str_specs: &[String],
    abs_paths: &[String],
) -> Vec<usize> {
    let str_set: FxHashSet<&str> = str_specs.iter().map(String::as_str).collect();
    let abs_set: FxHashSet<&str> = abs_paths.iter().map(String::as_str).collect();
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
    out
}

/// Shared multi-predicate node scan behind
/// :meth:`ProjectContext::indices_where`. Singular/plural args are
/// pre-merged and the path regex pre-compiled by the caller.
#[allow(clippy::too_many_arguments)]
fn indices_where_in(
    outputs: &BuildOutputs,
    kinds_vec: Option<Vec<String>>,
    filenames_vec: Option<Vec<String>>,
    simple_vec: Option<Vec<String>>,
    paths: Option<Vec<String>>,
    re_compiled: Option<regex::Regex>,
    flags: Option<u32>,
    flags_any: Option<u32>,
    fqname_prefix: Option<String>,
) -> Vec<usize> {
    let kinds_set: Option<FxHashSet<&str>> = kinds_vec
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let filenames_set: Option<FxHashSet<&str>> = filenames_vec
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let simple_set: Option<FxHashSet<&str>> = simple_vec
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let paths_set: Option<FxHashSet<&str>> = paths
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());

    let mut out: Vec<usize> = Vec::new();
    for (idx, node) in outputs.builder.nodes.iter().enumerate() {
        // Incremental tombstones keep their (blanked) slot in place so
        // live indices never remap; an unfiltered scan must still
        // never surface one.
        if outputs.builder.tombstoned.contains(&(idx as u32)) {
            continue;
        }
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
    out
}

/// Merge the singular + plural forms of an `indices_where` filter into a
/// single optional list (shared by the pymethod and the `FrozenView`
/// facade so both agree on the merge semantics).
fn merge_singular_plural(
    singular: Option<String>,
    plural: Option<Vec<String>>,
) -> Option<Vec<String>> {
    match (singular, plural) {
        (None, None) => None,
        (Some(s), None) => Some(vec![s]),
        (None, Some(v)) => Some(v),
        (Some(s), Some(mut v)) => {
            v.push(s);
            Some(v)
        }
    }
}

/// Shared bounds-checked `(kind, path, fqname, flags)` snapshot behind
/// :meth:`ProjectContext::node_attrs`. `None` => some index was out of
/// range (the caller maps that to `IndexError`).
fn node_attrs_in(
    outputs: &BuildOutputs,
    indices: &[usize],
) -> Option<Vec<crate::helpers::NodeAttrs>> {
    let len = outputs.builder.nodes.len();
    let mut out: Vec<crate::helpers::NodeAttrs> = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= len {
            return None;
        }
        let node = &outputs.builder.nodes[idx];
        out.push(crate::helpers::NodeAttrs {
            kind: node.kind.to_string(),
            path: node.path.clone(),
            fqname: node.fqname.clone(),
            flags: node.flags,
        });
    }
    Some(out)
}

/// Shared bounds-checked path-only snapshot behind
/// :meth:`ProjectContext::node_paths`. `None` => out-of-range index.
fn node_paths_in(outputs: &BuildOutputs, indices: &[usize]) -> Option<Vec<String>> {
    let len = outputs.builder.nodes.len();
    let mut out: Vec<String> = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= len {
            return None;
        }
        out.push(outputs.builder.nodes[idx].path.clone());
    }
    Some(out)
}

/// Shared db-serial read behind
/// :meth:`ProjectContext::find_literal_list_entries`.
fn find_literal_list_entries_in(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    var_fqn: &str,
) -> Option<Vec<String>> {
    let idxs = outputs.decl_by_fqname.get(var_fqn)?;
    let bare_name = var_fqn.rsplit('.').next().unwrap_or("");
    if bare_name.is_empty() {
        return None;
    }
    let mut out: Vec<String> = Vec::new();
    let mut found_any = false;
    for &idx in idxs {
        let path = outputs.builder.nodes[idx].path.clone();
        let Some(&file) = outputs.path_to_file.get(&path) else {
            continue;
        };
        for (name, entries) in &file_extraction(db, file).literal_list_rows {
            if name != bare_name {
                continue;
            }
            found_any = true;
            out.extend(entries.iter().map(|e| e.to_string()));
        }
    }
    found_any.then_some(out)
}

/// Shared db-serial read behind
/// :meth:`ProjectContext::find_module_dunder_all_exports_indices`.
fn find_module_dunder_all_exports_indices_in(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    module_fqn: &str,
) -> Option<Vec<usize>> {
    let all_fqn = format!("{module_fqn}.__all__");
    let entries = find_literal_list_entries_in(db, outputs, &all_fqn)?;
    let mut out: Vec<usize> = Vec::new();
    for entry in entries {
        let entry_fqn = format!("{module_fqn}.{entry}");
        if let Some(idxs) = outputs.decl_by_fqname.get(&entry_fqn) {
            out.extend(idxs.iter().copied());
        }
    }
    Some(out)
}

/// Shared db-serial scan behind
/// :meth:`ProjectContext::find_main_blocks_indices`.
fn find_main_blocks_indices_in(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
) -> Vec<(usize, Vec<usize>)> {
    let mut hits: Vec<(File, usize, (u32, u32))> = Vec::new();
    for (&file, &module_idx) in &outputs.module_nodes_by_file {
        let Some(range) = file_extraction(db, file).main_block_range else {
            continue;
        };
        hits.push((file, module_idx, range));
    }
    if hits.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<(usize, Vec<usize>)> = Vec::with_capacity(hits.len());
    for (file, module_idx, (block_start, block_end)) in hits {
        // Per-file decl walk through the payload: the `(file, local)`
        // identity is the block offset, so no global decl map is
        // needed — and the result is in deterministic local order
        // (the old `global_index` scan iterated a hashmap).
        let mut decl_idxs: Vec<usize> = Vec::new();
        if let Some(&pos) = outputs.ordinal_of.get(&file) {
            let base = outputs.builder.file_blocks[pos].0 as usize;
            let payload = file_to_nodes(db, file);
            for (local, node_ref) in payload.refs.iter().enumerate() {
                if matches!(node_ref, NodeRef::Def(_)) {
                    let (start, end) = payload.nodes[local].name_range;
                    if start >= block_start && end <= block_end {
                        decl_idxs.push(base + local);
                    }
                }
            }
        }
        out.push((module_idx, decl_idxs));
    }
    out
}

/// Shared db-serial read behind
/// :meth:`ProjectContext::function_parameters`.
fn function_parameters_in(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    indices: &[usize],
) -> Option<Vec<Vec<String>>> {
    let len = outputs.builder.nodes.len();
    for &idx in indices {
        if idx >= len {
            return None;
        }
    }
    if indices.is_empty() {
        return Some(Vec::new());
    }
    let wanted: FxHashSet<usize> = indices.iter().copied().collect();
    let mut idx_to_loc: FxHashMap<usize, (File, (u32, u32))> =
        FxHashMap::with_capacity_and_hasher(wanted.len(), Default::default());
    for (&(file, rk), &idx) in &outputs.decl_by_name_range {
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
        for (rk, names) in &file_extraction(db, file).function_params {
            if let Some(&idx) = rk_to_idx.get(rk) {
                params_for_idx.insert(idx, names.iter().map(|n| n.to_string()).collect());
            }
        }
    }
    Some(
        indices
            .iter()
            .map(|idx| params_for_idx.remove(idx).unwrap_or_default())
            .collect(),
    )
}

/// Shared db-serial read behind
/// :meth:`ProjectContext::class_method_parameters`.
fn class_method_parameters_in(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    indices: &[usize],
) -> Option<Vec<Vec<String>>> {
    let len = outputs.builder.nodes.len();
    for &idx in indices {
        if idx >= len {
            return None;
        }
    }
    if indices.is_empty() {
        return Some(Vec::new());
    }
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
        for (rk, names) in &file_extraction(db, file).class_method_params {
            if let Some(&idx) = rk_to_idx.get(rk) {
                params_for_idx.insert(idx, names.iter().map(|n| n.to_string()).collect());
            }
        }
    }
    Some(
        indices
            .iter()
            .map(|idx| params_for_idx.remove(idx).unwrap_or_default())
            .collect(),
    )
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

// ============================================================
// Off-GIL query cores. Each takes an owned `db` snapshot (or a
// `&BuildOutputs` for pure-read queries) and returns plain Rust
// values, so both the GIL-managing `ProjectContext` pymethods and
// the GIL-free `FrozenView` plugin receiver can delegate to the
// same logic. The `ProjectContext` side wraps the call in
// `py.allow_threads`; `FrozenView` calls it directly (it already
// runs inside the project-wide plugin `rayon::scope`).
// ============================================================

#[allow(clippy::type_complexity)]
fn find_decorated_decls_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    decorator_modules: &[String],
    decorator_names: &[String],
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(usize, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let names: FxHashSet<&str> = decorator_names.iter().map(String::as_str).collect();
    let mut out: Vec<(usize, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        let imports = imports_local_from_facts(&facts.import_facts, decorator_modules, &names);
        if imports.is_empty() {
            continue;
        }
        for (rk, descriptors) in &facts.decorator_rows {
            let Some(&idx) = outputs.decl_by_name_range.get(&(file, *rk)) else {
                continue;
            };
            // First matching decorator wins (mirrors `decorators_match_imports`).
            for desc in descriptors {
                let matched = match desc.attrs.as_slice() {
                    // `@name` / `@name(...)` bound via `from <module> import name`.
                    [] => imports
                        .get(&desc.root_name)
                        .is_some_and(|target| names.contains(target.as_str())),
                    // `@alias.attr` / `@alias.attr(...)` where `alias` is the
                    // module (`import <module> [as alias]`).
                    [attr] => {
                        imports
                            .get(&desc.root_name)
                            .map(compact_str::CompactString::as_str)
                            == Some(MODULE_ALIAS_MARKER)
                            && names.contains(attr.as_str())
                    }
                    _ => false,
                };
                if matched {
                    let call_args = if extract_args {
                        desc.kwargs.clone()
                    } else {
                        CallArgs::default()
                    };
                    out.push((idx, call_args));
                    break;
                }
            }
        }
    }
    out
}

#[allow(clippy::type_complexity)]
fn find_instance_constructions_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    modules: &[String],
    ctor_names: &[String],
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(usize, String, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let allowed: FxHashSet<&str> = ctor_names.iter().map(String::as_str).collect();
    let mut out: Vec<(usize, String, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        if facts.construction_rows.is_empty() {
            continue;
        }
        let imports = imports_local_from_facts(&facts.import_facts, modules, &allowed);
        if imports.is_empty() {
            continue;
        }
        for (rk, desc) in &facts.construction_rows {
            let Some(matched) = match_callee_descriptor(desc, &imports, modules, &allowed) else {
                continue;
            };
            let Some(&idx) = outputs.decl_by_name_range.get(&(file, *rk)) else {
                continue;
            };
            let call_args = if extract_args {
                desc.kwargs.clone()
            } else {
                CallArgs::default()
            };
            out.push((idx, matched.into_string(), call_args));
        }
    }
    out
}

#[allow(clippy::type_complexity)]
fn find_handler_decorators_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    decorator_attrs: &[String],
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(String, usize, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let attrs: FxHashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
    let mut out: Vec<(String, usize, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        for (rk, descriptors) in &facts.decorator_rows {
            let Some(&idx) = outputs.decl_by_name_range.get(&(file, *rk)) else {
                continue;
            };
            // One row per distinct owner with a matching `<owner>.<attr>`
            // decorator on this function (dedup in source order).
            let mut seen_owners: FxHashSet<&str> = FxHashSet::default();
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
                let call_args = if extract_args {
                    desc.kwargs.clone()
                } else {
                    CallArgs::default()
                };
                out.push((desc.root_name.to_string(), idx, call_args));
            }
        }
    }
    out
}

#[allow(clippy::type_complexity)]
fn find_handler_decorators_via_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    via_attr: &str,
    decorator_attrs: &[String],
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(String, usize, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let attrs: FxHashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
    let mut out: Vec<(String, usize, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        for (rk, descriptors) in &facts.decorator_rows {
            let Some(&idx) = outputs.decl_by_name_range.get(&(file, *rk)) else {
                continue;
            };
            // One row per distinct owner with a matching
            // `<owner>.<via_attr>.<attr>` decorator (dedup in source order).
            let mut seen_owners: FxHashSet<&str> = FxHashSet::default();
            for desc in descriptors {
                let [via, outer] = desc.attrs.as_slice() else {
                    continue;
                };
                if via.as_str() != via_attr || !attrs.contains(outer.as_str()) {
                    continue;
                }
                if !seen_owners.insert(desc.root_name.as_str()) {
                    continue;
                }
                let call_args = if extract_args {
                    desc.kwargs.clone()
                } else {
                    CallArgs::default()
                };
                out.push((desc.root_name.to_string(), idx, call_args));
            }
        }
    }
    out
}

/// Fan a file's [`CallSiteFact`]s in to their owning node index, the way
/// `owner_idx_for_stmt_with` did over the live AST: each `def` / `class`
/// owns the calls in its subtree (keyed by name-range, falling back to the
/// module node), and every other top-level statement attributes its calls
/// to the module node. Skips a bucket whose owner can't be resolved (no
/// decl index and no module node), matching the old `else { continue }`.
fn for_each_call_site(
    facts: &FileExtraction,
    outputs: &BuildOutputs,
    file: File,
    mut f: impl FnMut(usize, &CallSiteFact),
) {
    let module_idx = outputs.module_nodes_by_file.get(&file).copied();
    for (rk, sites) in &facts.call_sites_by_decl {
        let Some(owner_idx) = outputs
            .decl_by_name_range
            .get(&(file, *rk))
            .copied()
            .or(module_idx)
        else {
            continue;
        };
        for site in sites {
            f(owner_idx, site);
        }
    }
    if let Some(module_idx) = module_idx {
        for site in &facts.module_call_sites {
            f(module_idx, site);
        }
    }
}

#[allow(clippy::type_complexity)]
fn find_calls_on_attr_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    attr: &str,
    arg_index: usize,
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(usize, String, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let mut out: Vec<(usize, String, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        for_each_call_site(facts, outputs, file, |owner_idx, site| {
            // `find_calls_on_attr` matches `<any-receiver>.<attr>(...)`, so it
            // keys off the recorded `callee_attr` (set for any receiver shape).
            if site.callee_attr.as_deref() != Some(attr) {
                return;
            }
            let hits = site.string_or_collection(arg_index);
            if hits.is_empty() {
                return;
            }
            let call_args = if extract_args {
                site.kwargs.clone()
            } else {
                CallArgs::default()
            };
            for s in hits {
                out.push((owner_idx, s.to_string(), call_args.clone()));
            }
        });
    }
    out
}

#[allow(clippy::type_complexity)]
fn find_factory_decls_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    modules: &[String],
    ctor_names: &[String],
) -> Vec<(usize, Vec<String>)> {
    let db: &dyn ProjectDb = &*db;
    let allowed: FxHashSet<&str> = ctor_names.iter().map(String::as_str).collect();
    let mut out: Vec<(usize, Vec<String>)> = Vec::new();
    for &file in &outputs.project_files {
        let facts = file_extraction(db, file);
        if facts.factory_rows.is_empty() {
            continue;
        }
        let imports = imports_local_from_facts(&facts.import_facts, modules, &allowed);
        if imports.is_empty() {
            continue;
        }
        for (rk, descriptors) in &facts.factory_rows {
            let Some(&idx) = outputs.decl_by_name_range.get(&(file, *rk)) else {
                continue;
            };
            let mut kinds: FxHashSet<String> = FxHashSet::default();
            for desc in descriptors {
                if let Some(name) = match_callee_descriptor(desc, &imports, modules, &allowed) {
                    kinds.insert(name.into_string());
                }
            }
            if kinds.is_empty() {
                continue;
            }
            let mut kinds_vec: Vec<String> = kinds.into_iter().collect();
            kinds_vec.sort();
            out.push((idx, kinds_vec));
        }
    }
    out
}

#[allow(clippy::type_complexity)]
fn find_calls_to_imported_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    modules: &[String],
    name: &str,
    arg_index: usize,
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(usize, String, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let allowed: FxHashSet<&str> = [name].into_iter().collect();
    let mut out: Vec<(usize, String, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        // The imported name has to be bound in this file for any call to
        // resolve; an empty map subsumes the old `_contains_identifier` skip.
        let imports = imports_local_from_facts(&facts.import_facts, modules, &allowed);
        if imports.is_empty() {
            continue;
        }
        for_each_call_site(facts, outputs, file, |owner_idx, site| {
            let Some(chain) = &site.callee else {
                return;
            };
            if match_callee_chain(&chain.root_name, &chain.attrs, &imports, modules, &allowed)
                .is_none()
            {
                return;
            }
            let Some(arg) = site.nth_positional_string(arg_index) else {
                return;
            };
            let call_args = if extract_args {
                site.kwargs.clone()
            } else {
                CallArgs::default()
            };
            out.push((owner_idx, arg.to_string(), call_args));
        });
    }
    out
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn find_calls_on_var_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    owner: &str,
    attr: &str,
    arg_index: usize,
    required_positional: Option<usize>,
    extract_args: bool,
    path_re: &Option<regex::Regex>,
) -> Vec<(usize, String, CallArgs)> {
    let db: &dyn ProjectDb = &*db;
    let mut out: Vec<(usize, String, CallArgs)> = Vec::new();
    for &file in &outputs.project_files {
        if !_path_re_matches(path_re, db, file) {
            continue;
        }
        let facts = file_extraction(db, file);
        for_each_call_site(facts, outputs, file, |owner_idx, site| {
            // `call_callee_matches_var`: `<owner>.<attr>(...)` with a bare-name
            // receiver — i.e. a `Name`-rooted chain of exactly `[attr]`.
            let Some(chain) = &site.callee else {
                return;
            };
            let [only] = chain.attrs.as_slice() else {
                return;
            };
            if chain.root_name.as_str() != owner || only.as_str() != attr {
                return;
            }
            if let Some(expected) = required_positional {
                if site.positional_len != expected {
                    return;
                }
            }
            let Some(arg) = site.nth_positional_string(arg_index) else {
                return;
            };
            let call_args = if extract_args {
                site.kwargs.clone()
            } else {
                CallArgs::default()
            };
            out.push((owner_idx, arg.to_string(), call_args));
        });
    }
    out
}

fn find_classes_defining_method_indices_core(
    db: Box<dyn ProjectDb>,
    outputs: &BuildOutputs,
    method_name: &str,
) -> Vec<usize> {
    let db: &dyn ProjectDb = &*db;
    let mut out: Vec<usize> = Vec::new();
    for &file in &outputs.project_files {
        for (rk, methods) in &file_extraction(db, file).class_method_defs {
            if methods.iter().any(|m| m == method_name) {
                if let Some(&idx) = outputs.class_by_selection.get(&(file, *rk)) {
                    out.push(idx);
                }
            }
        }
    }
    out
}

/// Subclasses of the class addressed by ``base_fqn`` (project or
/// external seed). Takes an owned ``ProjectDatabase`` so it can resolve
/// the seed serially (``locate_class_seed`` needs a concrete db). A
/// project seed walks the cached ``children_by_node`` index directly; an
/// external seed reads its direct project subclasses from the cached
/// ``external_base_children`` index (no re-parse — re-export / alias
/// chains resolve to the same decl through `resolve_member_def`, so
/// they're already folded into that index). See
/// ``FrozenView::find_subclasses_indices`` for the fast/slow-path
/// rationale.
fn find_subclasses_indices_core(
    db: ProjectDatabase,
    outputs: &BuildOutputs,
    base_fqn: &str,
    transitive: bool,
) -> Vec<usize> {
    let Some((seed_file, seed_range)) = locate_class_seed(&db, outputs, base_fqn) else {
        return Vec::new();
    };
    let rk = (seed_range.start().to_u32(), seed_range.end().to_u32());
    if let Some(&seed_idx) = outputs.class_by_selection.get(&(seed_file, rk)) {
        return if transitive {
            transitive_subclasses_via_index(seed_idx, &outputs.children_by_node)
        } else {
            outputs
                .children_by_node
                .get(&seed_idx)
                .cloned()
                .unwrap_or_default()
        };
    }
    let children_by_node = &outputs.children_by_node;
    // External seed: the class lives outside the project. Its direct
    // project subclasses are cached in `external_base_children` (folded
    // from the per-file `class_bases` payloads — no re-parse). A base bound
    // through one or more project re-exports or module-level aliases
    // resolves to the same external decl through `resolve_member_def`
    // (which follows re-export and alias chains), so those subclasses land
    // in `external_base_children` too.
    //
    // Sibling fold by construction: the same external class can be spelled
    // several ways (`unittest.TestCase` vs `unittest.case.TestCase`), but
    // assemble ran every observed base through `resolve_member_def`, keying
    // by the resolved decl's `(file, name-range)`. `locate_class_seed`
    // resolved this query's input through the same resolver, so
    // `(seed_file, rk)` is that same key: sibling spellings and renamed
    // re-exports already collapsed into one entry. A single O(1) lookup,
    // no scan.
    let direct = outputs
        .external_base_children
        .get(&(seed_file, rk))
        .cloned()
        .unwrap_or_default();
    if !transitive {
        return direct;
    }
    let mut out_idx: FxHashSet<usize> = direct.iter().copied().collect();
    for &d in &direct {
        out_idx.extend(transitive_subclasses_via_index(d, children_by_node));
    }
    out_idx.into_iter().collect()
}

/// Subclasses of the class at positional index ``class_idx``. Returns
/// ``None`` when ``class_idx`` is out of range (callers map that to an
/// ``IndexError``); ``Some(vec)`` otherwise (empty when the node isn't
/// a class or has no subclasses).
fn find_subclasses_of_idx_in(outputs: &BuildOutputs, class_idx: usize) -> Option<Vec<usize>> {
    let len = outputs.builder.nodes.len();
    if class_idx >= len {
        return None;
    }
    if outputs.builder.nodes[class_idx].kind != "class" {
        return Some(Vec::new());
    }
    // `class_idx` is its own hierarchy seed: assemble records every
    // class node in `class_by_selection`, and `children_by_node` is
    // keyed by those same global indices. The former locate-by-line
    // round trip (re-parse → name range → `class_by_selection`) always
    // resolved straight back to `class_idx`, so walk the cached
    // hierarchy from it directly.
    Some(transitive_subclasses_via_index(
        class_idx,
        &outputs.children_by_node,
    ))
}

// ============================================================
// `FrozenView` — the GIL-free, `Send` plugin receiver.
//
// A project-wide native plugin's `run` is handed a `&FrozenView`
// instead of a `&ProjectContext`. `ProjectContext` is a `#[pyclass]`
// whose salsa `db` is `!Sync`, so it can neither cross a thread
// boundary by reference nor be shared across the project-wide plugin
// fan-out. `FrozenView` sidesteps both: it owns a cheap
// `ProjectDatabase` clone (`Send`) and borrows the shared, `Sync`
// `BuildOutputs`. Each rayon task in the fan-out builds its own
// `FrozenView` and runs one plugin against it with the GIL released.
//
// Every method here is a thin delegation to the same `*_in` / `*_core`
// free function the matching `ProjectContext` pymethod uses, so the
// two surfaces can never drift. The db-serial reads go through
// `&self.db`; the parallel file-walks `dyn_clone` it per walk; the
// subclass walk clones it once (it resolves the seed serially first).
// ============================================================

/// Flatten a [`FlagRegistry`] into the bit-sorted `(name, bit, seed,
/// default_on, description)` tuples the Python getters and `write_graph`
/// consume.
fn flag_registry_tuples(reg: &FlagRegistry) -> Vec<(String, u64, bool, bool, String)> {
    reg.entries()
        .into_iter()
        .map(|(bit, spec)| (spec.name, bit, spec.seed, spec.default_on, spec.description))
        .collect()
}

/// Read-only, owned-db snapshot of a materialized project, handed to a
/// project-wide native plugin's `run`. `Send` but `!Sync`: each
/// fan-out task owns its view; views are never shared across threads.
pub(crate) struct FrozenView<'a> {
    pub(crate) db: ProjectDatabase,
    pub(crate) outputs: &'a BuildOutputs,
    pub(crate) root: &'a str,
    /// Frozen node/edge flag registries (engine built-ins + plugin
    /// declarations), lent read-only for the parallel plugin pass so a
    /// plugin can resolve a declared flag name to its bit via
    /// [`crate::native_plugins::plugin_api::PluginCtx::node_flag`] /
    /// `edge_flag`.
    pub(crate) node_flags: &'a FlagRegistry,
    pub(crate) edge_flags: &'a FlagRegistry,
    /// Frozen topic registry, lent the same way as the flag registries so a
    /// plugin resolves a declared topic name to its handle via
    /// [`crate::native_plugins::plugin_api::PluginCtx::topic`] and reads the
    /// collected facts via `facts_for_topic`.
    pub(crate) topics: &'a TopicRegistry,
}

impl<'a> FrozenView<'a> {
    pub(crate) fn new(
        db: ProjectDatabase,
        outputs: &'a BuildOutputs,
        root: &'a str,
        node_flags: &'a FlagRegistry,
        edge_flags: &'a FlagRegistry,
        topics: &'a TopicRegistry,
    ) -> Self {
        Self {
            db,
            outputs,
            root,
            node_flags,
            edge_flags,
            topics,
        }
    }

    // --- data accessors -------------------------------------------------

    pub(crate) fn project_root(&self) -> &str {
        self.root
    }

    pub(crate) fn node_count(&self) -> usize {
        self.outputs.builder.nodes.len()
    }

    // --- pure-read structural lookups -----------------------------------

    pub(crate) fn has_imports_of(&self, module_name: &str) -> bool {
        has_imports_of_in(self.outputs, module_name)
    }

    pub(crate) fn find_imports_of_indices(&self, module_name: &str) -> Vec<usize> {
        find_imports_of_indices_in(self.outputs, module_name)
    }

    pub(crate) fn find_module_idx(&self, fqname: &str) -> Option<usize> {
        find_module_idx_in(self.outputs, fqname)
    }

    pub(crate) fn find_declarations_indices(&self, fqname: &str) -> Vec<usize> {
        find_declarations_indices_in(self.outputs, fqname)
    }

    pub(crate) fn module_for_idx(&self, path: &str) -> Option<usize> {
        module_for_idx_in(self.outputs, path)
    }

    pub(crate) fn modules_for_paths(&self, paths: &[String]) -> Vec<Option<usize>> {
        paths
            .iter()
            .map(|p| module_for_idx_in(self.outputs, p))
            .collect()
    }

    pub(crate) fn resolve_idx(&self, fqname: &str) -> Option<usize> {
        resolve_idx_in(self.outputs, fqname)
    }

    pub(crate) fn module_surface_indices(&self, module_fqn: &str) -> Vec<usize> {
        module_surface_indices_in(self.outputs, module_fqn)
    }

    pub(crate) fn module_surfaces_indices(
        &self,
        module_fqns: &[String],
    ) -> FxHashMap<String, Vec<usize>> {
        module_surfaces_indices_in(self.outputs, module_fqns)
    }

    pub(crate) fn find_module_top_level_decls_indices(&self, module_fqn: &str) -> Vec<usize> {
        find_module_top_level_decls_indices_in(self.outputs, module_fqn)
    }

    pub(crate) fn decls_under_indices(&self, path_prefix: &str) -> Vec<usize> {
        decls_under_indices_in(self.outputs, path_prefix)
    }

    /// A pattern that doesn't compile yields no matches (the airlock
    /// convention, matching `PluginCtx::decls_matching_name`).
    pub(crate) fn decls_matching_name_indices(&self, pattern: &str) -> Vec<usize> {
        match regex::Regex::new(pattern) {
            Ok(re) => decls_matching_name_indices_in(self.outputs, &re),
            Err(_) => Vec::new(),
        }
    }

    pub(crate) fn find_nodes_matching_specs_indices(
        &self,
        regexes: &[String],
        str_specs: &[String],
        abs_paths: &[String],
    ) -> Vec<usize> {
        // `^`-anchor for `re.match` semantics (mirrors the pymethod);
        // patterns that don't compile are dropped.
        let compiled: Vec<regex::Regex> = regexes
            .iter()
            .filter_map(|p| {
                let anchored = if p.starts_with('^') {
                    p.clone()
                } else {
                    format!("^{p}")
                };
                regex::Regex::new(&anchored).ok()
            })
            .collect();
        find_nodes_matching_specs_indices_in(
            self.outputs,
            self.root,
            &compiled,
            str_specs,
            abs_paths,
        )
    }

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
    ) -> Vec<usize> {
        let re_compiled = path_regex
            .as_deref()
            .and_then(|s| regex::Regex::new(s).ok());
        indices_where_in(
            self.outputs,
            merge_singular_plural(kind, kinds),
            merge_singular_plural(filename, filenames),
            merge_singular_plural(simple_name, simple_names),
            paths,
            re_compiled,
            flags,
            flags_any,
            fqname_prefix,
        )
    }

    // --- reachability (shared `bfs`) ------------------------------------

    pub(crate) fn descendants_indices(
        &self,
        root_idx: usize,
        skip_flags: u8,
    ) -> Option<Vec<usize>> {
        if root_idx >= self.outputs.builder.nodes.len() {
            return None;
        }
        Some(
            bfs(
                &self.outputs.builder,
                [root_idx],
                Direction::Forward,
                skip_flags,
            )
            .into_iter()
            .collect(),
        )
    }

    pub(crate) fn ancestors_indices(&self, decl_idx: usize, skip_flags: u8) -> Option<Vec<usize>> {
        if decl_idx >= self.outputs.builder.nodes.len() {
            return None;
        }
        Some(
            bfs(
                &self.outputs.builder,
                [decl_idx],
                Direction::Reverse,
                skip_flags,
            )
            .into_iter()
            .collect(),
        )
    }

    pub(crate) fn direct_predecessors_idx(&self, idx: usize, skip_flags: u8) -> Option<Vec<usize>> {
        if idx >= self.outputs.builder.nodes.len() {
            return None;
        }
        Some(direct_predecessors_idxs_in(self.outputs, idx, skip_flags))
    }

    // --- db-serial reads ------------------------------------------------

    pub(crate) fn find_main_blocks_indices(&self) -> Vec<(usize, Vec<usize>)> {
        find_main_blocks_indices_in(&self.db, self.outputs)
    }

    pub(crate) fn find_literal_list_entries(&self, var_fqn: &str) -> Option<Vec<String>> {
        find_literal_list_entries_in(&self.db, self.outputs, var_fqn)
    }

    pub(crate) fn find_module_dunder_all_exports_indices(
        &self,
        module_fqn: &str,
    ) -> Option<Vec<usize>> {
        find_module_dunder_all_exports_indices_in(&self.db, self.outputs, module_fqn)
    }

    /// Parameter names per function index, or `None` on out-of-range.
    pub(crate) fn function_parameters(&self, indices: &[usize]) -> Option<Vec<Vec<String>>> {
        function_parameters_in(&self.db, self.outputs, indices)
    }

    /// Union of method parameter names per class index (excluding
    /// `self` / `cls`), or `None` on out-of-range.
    pub(crate) fn class_method_parameters(&self, indices: &[usize]) -> Option<Vec<Vec<String>>> {
        class_method_parameters_in(&self.db, self.outputs, indices)
    }

    // --- subclasses -----------------------------------------------------

    /// Subclasses of the class at `class_idx`, or `None` on out-of-range
    /// (an empty vec means "not a class node").
    pub(crate) fn find_subclasses_of_idx(&self, class_idx: usize) -> Option<Vec<usize>> {
        find_subclasses_of_idx_in(self.outputs, class_idx)
    }

    pub(crate) fn find_subclasses_indices(&self, base_fqn: &str, transitive: bool) -> Vec<usize> {
        find_subclasses_indices_core(self.db.clone(), self.outputs, base_fqn, transitive)
    }

    // --- parallel file-walk matchers ------------------------------------
    //
    // Each `dyn_clone`s the owned db for the inner rayon scope.
    // Nested inside the project-wide plugin scope, this is
    // safe: rayon scopes compose. These are GIL-free: no path filtering,
    // no `PyResult` — the curated plugin surface is the only caller and
    // it never path-scopes, so the matchers walk every project file.

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_decorated_decls(
        &self,
        decorator_modules: &[String],
        decorator_names: &[String],
        extract_args: bool,
    ) -> Vec<(usize, CallArgs)> {
        find_decorated_decls_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            decorator_modules,
            decorator_names,
            extract_args,
            &None,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_instance_constructions(
        &self,
        modules: &[String],
        ctor_names: &[String],
        extract_args: bool,
    ) -> Vec<(usize, String, CallArgs)> {
        find_instance_constructions_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            modules,
            ctor_names,
            extract_args,
            &None,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_handler_decorators(
        &self,
        decorator_attrs: &[String],
        extract_args: bool,
    ) -> Vec<(String, usize, CallArgs)> {
        find_handler_decorators_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            decorator_attrs,
            extract_args,
            &None,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_handler_decorators_via(
        &self,
        via_attr: &str,
        decorator_attrs: &[String],
        extract_args: bool,
    ) -> Vec<(String, usize, CallArgs)> {
        find_handler_decorators_via_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            via_attr,
            decorator_attrs,
            extract_args,
            &None,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_calls_on_attr(
        &self,
        attr: &str,
        arg_index: usize,
        extract_args: bool,
    ) -> Vec<(usize, String, CallArgs)> {
        find_calls_on_attr_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            attr,
            arg_index,
            extract_args,
            &None,
        )
    }

    pub(crate) fn find_factory_decls(
        &self,
        modules: &[String],
        ctor_names: &[String],
    ) -> Vec<(usize, Vec<String>)> {
        find_factory_decls_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            modules,
            ctor_names,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_calls_to_imported(
        &self,
        modules: &[String],
        name: &str,
        arg_index: usize,
        extract_args: bool,
    ) -> Vec<(usize, String, CallArgs)> {
        find_calls_to_imported_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            modules,
            name,
            arg_index,
            extract_args,
            &None,
        )
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_calls_on_var(
        &self,
        owner: &str,
        attr: &str,
        arg_index: usize,
        required_positional: Option<usize>,
        extract_args: bool,
    ) -> Vec<(usize, String, CallArgs)> {
        find_calls_on_var_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            owner,
            attr,
            arg_index,
            required_positional,
            extract_args,
            &None,
        )
    }

    pub(crate) fn find_classes_defining_method_indices(&self, method_name: &str) -> Vec<usize> {
        find_classes_defining_method_indices_core(
            ProjectDb::dyn_clone(&self.db),
            self.outputs,
            method_name,
        )
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
    pub(crate) fn edges(&self) -> PyResult<Vec<(usize, usize, u8)>> {
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
        skip_flags: u8,
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
    /// ``ctx.nodes()``. Accepts the predicate vocabulary (``kind`` /
    /// ``kinds`` / ``filename`` / ``filenames`` / ``simple_name`` /
    /// ``simple_names`` / ``paths`` / ``path_regex`` / ``flags`` /
    /// ``flags_any`` / ``fqname_prefix``) directly — no builder, just
    /// the one filter step yielding a ``list[int]`` of indices.
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
        let re_compiled: Option<regex::Regex> = match path_regex.as_deref() {
            None => None,
            Some(s) => Some(
                regex::Regex::new(s)
                    .map_err(|e| PyValueError::new_err(format!("invalid path regex {s:?}: {e}")))?,
            ),
        };
        Ok(indices_where_in(
            &outputs,
            merge_singular_plural(kind, kinds),
            merge_singular_plural(filename, filenames),
            merge_singular_plural(simple_name, simple_names),
            paths,
            re_compiled,
            flags,
            flags_any,
            fqname_prefix,
        ))
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
        node_attrs_in(&outputs, &indices).ok_or_else(|| {
            let len = outputs.builder.nodes.len();
            pyo3::exceptions::PyIndexError::new_err(format!("node index out of range (len={len})"))
        })
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
        node_paths_in(&outputs, &indices).ok_or_else(|| {
            let len = outputs.builder.nodes.len();
            pyo3::exceptions::PyIndexError::new_err(format!("node index out of range (len={len})"))
        })
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
        function_parameters_in(&self.db, &outputs, &indices).ok_or_else(|| {
            let len = outputs.builder.nodes.len();
            pyo3::exceptions::PyIndexError::new_err(format!("node index out of range (len={len})"))
        })
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
        class_method_parameters_in(&self.db, &outputs, &indices).ok_or_else(|| {
            let len = outputs.builder.nodes.len();
            pyo3::exceptions::PyIndexError::new_err(format!("node index out of range (len={len})"))
        })
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

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the project-level index builders. The
    //! salsa/pyo3-backed pipeline is exercised end-to-end by the
    //! python suite; here we cover `build_fqname_indices`' map-reduce
    //! against a serial reference over hand-built `GraphNode` tables.
    use super::*;
    use crate::file_payload::ImportPayload;

    fn node(fqname: &str, kind: &'static str) -> GraphNode {
        GraphNode {
            fqname: fqname.to_string(),
            kind,
            ..GraphNode::default()
        }
    }

    /// The pre-map-reduce serial loop, kept verbatim as the reference.
    #[allow(clippy::type_complexity)]
    fn serial_reference(
        builder: &GraphBuilder,
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
            if node.is_overload {
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
                            .entry(import.module.to_string())
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

    fn assert_matches_reference(builder: &GraphBuilder) {
        let counters = Arc::new(ProgressCounters::new());
        let expected = serial_reference(builder);
        let actual = build_fqname_indices(builder, &counters);
        // Vec orders matter: multi-binder decl lists and
        // children_by_parent buckets are index-ordered in the serial
        // loop, and the ordered reduce must reproduce that exactly.
        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
        assert_eq!(actual.2, expected.2);
        assert_eq!(actual.3, expected.3);
    }

    #[test]
    fn fqname_indices_match_serial_reference_small() {
        let mut b = GraphBuilder::with_capacity(0);
        b.nodes.push(node("pkg.mod", "module"));
        b.nodes.push(node("pkg.mod.f", "function"));
        b.nodes.push(node("pkg.mod.C", "class"));
        b.nodes.push(node("pkg.mod.x", "variable"));
        let mut imp = node("pkg.mod.os", "import");
        imp.imports = Some(ImportPayload {
            module: "os".into(),
            decl: None,
            star: false,
        });
        b.nodes.push(imp);
        // Shadowed decl: same fqname, second binder — multi-entry list.
        b.nodes.push(node("pkg.mod.f", "function"));
        // Overload stubs are skipped entirely.
        let mut ov = node("pkg.mod.g", "function");
        ov.is_overload = true;
        b.nodes.push(ov);
        // External nodes don't index.
        b.nodes.push(node("[external dist] numpy", "external"));
        // Top-level name buckets under the empty-string parent.
        b.nodes.push(node("toplevel", "module"));
        assert_matches_reference(&b);
    }

    #[test]
    fn fqname_indices_match_serial_reference_large() {
        // Big enough that rayon actually splits (with_min_len = 1024),
        // with heavy fqname collisions so the ordered reduce's
        // append-after semantics are really exercised: every Vec must
        // come back in ascending idx order.
        let mut b = GraphBuilder::with_capacity(0);
        for i in 0..6000 {
            match i % 5 {
                0 => b.nodes.push(node(&format!("pkg.m{}", i % 7), "module")),
                1 => b
                    .nodes
                    .push(node(&format!("pkg.m{}.f", i % 11), "function")),
                2 => b.nodes.push(node(&format!("pkg.m{}.C", i % 13), "class")),
                3 => {
                    let mut imp = node(&format!("pkg.m{}.dep", i % 11), "import");
                    imp.imports = Some(ImportPayload {
                        module: compact_str::format_compact!("dep{}", i % 3),
                        decl: Some("thing".into()),
                        star: false,
                    });
                    b.nodes.push(imp);
                }
                _ => b
                    .nodes
                    .push(node(&format!("pkg.m{}.x", i % 17), "variable")),
            }
        }
        let (decls, ..) = build_fqname_indices(&b, &Arc::new(ProgressCounters::new()));
        for list in decls.values() {
            assert!(list.windows(2).all(|w| w[0] < w[1]), "idx order scrambled");
        }
        assert_matches_reference(&b);
    }
}
