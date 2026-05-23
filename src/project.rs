//! `Project`, `BuildOutputs`, `ProgressBars`, `ProjectContext`, and the
//! `build_project_graph` pipeline entrypoint. This module owns the
//! Salsa-backed analysis context that the rest of the crate operates on.

use std::cell::{Ref, RefCell};
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
use crate::file_payload::{file_to_edges, file_to_nodes, NodeKind, NodeRef};
use crate::file_ref_edges::file_to_ref_edges;
use crate::graph::{intern_kind, DeclIndex, Import, MainBlock, NativeGraph, SymbolNode};
use crate::helpers::{
    call_callee_matches_var, class_body_defines_method, collect_all_imports_local,
    collect_module_imports_local, decorators_match_imports, extract_call_args_kwargs,
    file_decl_sites, file_path_string, find_main_block_range, find_subclass_indices_via_refs,
    is_dunder_name, locate_class_def, locate_class_seed, matched_call_target,
    owner_idx_for_stmt_with, range_key, rel_path, top_level_assign_to_name, AttrCallFinder,
    CallArgs, FactoryCallFinder, StringArgCallFinder, NODE_FLAG_ENTRYPOINT,
};
use crate::ingest::{emit_visitor_warning, string_literal_list};
use crate::query::{
    _compile_path_regex, _contains_any_identifier, _contains_identifier, par_scan_files,
    QueryBuilder,
};
use rustc_hash::FxHashMap;
use std::sync::OnceLock as StdOnceLock;

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
        let outputs = build_project_graph(py, &mut self.db, false)?;
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
    /// `Import.module -> [import node idx]`. Lets ``find_imports_of``
    /// answer in O(1) lookup + O(matches) walk instead of scanning all
    /// nodes — and a cheap ``imports_of_exists`` short-circuit for
    /// plugins that just need a per-module presence check.
    pub(crate) imports_by_module: HashMap<String, Vec<usize>>,
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
) -> PyResult<BuildOutputs> {
    let timing = std::env::var_os("DEAD_CST_TIMING").is_some();

    let t0 = std::time::Instant::now();
    let project_files: Vec<File> = (&db.project().files(db)).into_iter().collect();
    let mut path_to_file: HashMap<String, File> = HashMap::with_capacity(project_files.len());
    for &file in &project_files {
        path_to_file.insert(file_path_string(db, file), file);
    }
    // Peer ``.pyi`` <-> ``.py`` twin map. Used at assembly time for
    // ``pyi_decl -> py_decl`` reachability edges + the stub-only
    // ENTRYPOINT flag fixup (.pyi decls without a runtime twin need
    // ENTRYPOINT so native-extension / protobuf-style stubs stay
    // alive even with no consumer reference).
    let py_files_by_stem: HashMap<String, File> = project_files
        .iter()
        .filter_map(|&f| {
            file_path_string(db, f)
                .strip_suffix(".py")
                .map(|stem| (stem.to_string(), f))
        })
        .collect();
    let mut peer_pyi_to_py: HashMap<File, File> = HashMap::new();
    for &f in &project_files {
        let path = file_path_string(db, f);
        if let Some(stem) = path.strip_suffix(".pyi") {
            if let Some(&py_twin) = py_files_by_stem.get(stem) {
                peer_pyi_to_py.insert(f, py_twin);
            }
        }
    }
    let t_enum = t0.elapsed();

    let progress = ProgressBars::new(show_progress, project_files.len() as u64);

    // Pre-warm: in parallel, force salsa's ``parsed_module`` and
    // ``semantic_index`` queries to fill for every project file.
    // The body is a ``#[salsa::tracked]`` function, so salsa owns
    // the parallel coordination (lock-free fast paths, not the
    // parking_lot contention plain-Rust callers hit). After this,
    // phase 1's serial ``ingest_decls`` loop hits warm caches.
    //
    // DEAD_CST_SKIP_PREWARM=1 disables this so the per-file populate
    // workers below fault parsed_module + semantic_index in
    // themselves. Used for the "does prewarm still buy us anything?"
    // A/B experiment; the populate phase is already parallel, so the
    // hypothesis is that prewarm is now redundant.
    let t_warm = std::time::Instant::now();
    let skip_prewarm = std::env::var_os("DEAD_CST_SKIP_PREWARM").is_some();
    if !skip_prewarm {
        let parent_db: ProjectDatabase = db.clone();
        let files_ref: &[File] = &project_files;
        let num_workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        let chunk_size = files_ref.len().div_ceil(num_workers).max(1);
        py.allow_threads(move || {
            std::thread::scope(|s| {
                for chunk in files_ref.chunks(chunk_size) {
                    let local_db = parent_db.clone();
                    s.spawn(move || {
                        use salsa::Database as _;
                        local_db.attach(|local_db| {
                            for &file in chunk {
                                crate::ingest::prewarm_file(local_db, file);
                            }
                        });
                    });
                }
            });
        });
    }
    let t_prewarm = t_warm.elapsed();

    // Parallel pre-populate: run file_to_nodes, file_to_edges, and
    // file_to_ref_edges as #[salsa::tracked] queries across all
    // project files. Workers are pure-rust (GIL released via
    // py.allow_threads). Each chunk attaches a cloned db so salsa's
    // tracked-fn machinery can do lock-free coordination — same
    // pattern as prewarm above. The salsa cache is populated for the
    // serial assembly pass below; nothing here directly mutates the
    // graph builder.
    //
    // project_dist_lookup is also populated here on its own thread.
    // The old pipeline overlapped build_dist_lookup with phase 1; we
    // do the same so the first per-file worker that needs it (for
    // PEP 503 canonical [external dist] X classification) hits a
    // warm salsa cache instead of paying the ~100ms walk inline.
    let t_populate = std::time::Instant::now();
    {
        let parent_db: ProjectDatabase = db.clone();
        let dist_db: ProjectDatabase = db.clone();
        let files_ref: &[File] = &project_files;
        let num_workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        let chunk_size = files_ref.len().div_ceil(num_workers).max(1);
        py.allow_threads(move || {
            std::thread::scope(|s| {
                // dist_lookup on its own thread — overlapped with
                // the file_to_* work below.
                s.spawn(move || {
                    use salsa::Database as _;
                    dist_db.attach(|local_db| {
                        let _ = crate::file_payload::project_dist_lookup(local_db);
                    });
                });
                for chunk in files_ref.chunks(chunk_size) {
                    let local_db = parent_db.clone();
                    s.spawn(move || {
                        use salsa::Database as _;
                        local_db.attach(|local_db| {
                            for &file in chunk {
                                let _ = file_to_nodes(local_db, file);
                                let _ = file_to_edges(local_db, file);
                                let _ = file_to_ref_edges(local_db, file);
                            }
                        });
                    });
                }
            });
        });
    }
    let t_populate_elapsed = t_populate.elapsed();
    progress.ingest.finish_and_clear();
    progress.imports.finish_and_clear();
    progress.references.finish_and_clear();

    // Serial assembly pass: walk files in deterministic order,
    // assign global graph node indices to each NodeRef, translate
    // edges, mint synthetic External nodes, build Py<SymbolNode>
    // wrappers, flush warnings. All salsa-tracked queries are
    // memoized from the parallel pre-populate above so this pass
    // is pure hashmap/index work plus the GIL-bound Py creation.
    let t_assemble = std::time::Instant::now();
    let assembled = assemble_graph(py, db, &project_files, &peer_pyi_to_py)?;
    let t_assemble_elapsed = t_assemble.elapsed();

    let mut builder = assembled.builder;
    let global_index = assembled.global_index;
    let module_nodes_by_file = assembled.module_nodes_by_file;
    let class_by_selection = assembled.class_by_selection;
    let decl_by_name_range = assembled.decl_by_name_range;

    let t4 = std::time::Instant::now();
    let (decl_by_fqname, module_by_fqname, imports_by_module) = build_fqname_indices(py, &builder);
    let t_fqname = t4.elapsed();
    if timing {
        eprintln!(
            "[dead-cst-timing] files={} nodes={} edges={} enum={:?} prewarm={:?} populate={:?} assemble={:?} fqname={:?} total={:?} rss={}MB",
            project_files.len(),
            builder.nodes.len(),
            builder.edges.len(),
            t_enum,
            t_prewarm,
            t_populate_elapsed,
            t_assemble_elapsed,
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
    global_index: DeclIndex,
    module_nodes_by_file: HashMap<File, usize>,
    class_by_selection: HashMap<(File, (u32, u32)), usize>,
    decl_by_name_range: HashMap<(File, (u32, u32)), usize>,
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
    peer_pyi_to_py: &HashMap<File, File>,
) -> PyResult<AssembledGraph> {
    let mut builder = GraphBuilder::new();
    let mut global_index: DeclIndex = HashMap::new();
    let mut module_nodes_by_file: HashMap<File, usize> = HashMap::new();
    let mut class_by_selection: HashMap<(File, (u32, u32)), usize> = HashMap::new();
    let mut decl_by_name_range: HashMap<(File, (u32, u32)), usize> = HashMap::new();
    let mut ref_to_global: FxHashMap<NodeRef<'db>, usize> = FxHashMap::default();
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

            let imports = node_data
                .imports
                .as_ref()
                .map(|ip| {
                    Py::new(
                        py,
                        Import {
                            module: ip.module.clone(),
                            decl: ip.decl.clone(),
                            star: ip.star,
                        },
                    )
                })
                .transpose()?;

            let symbol = SymbolNode {
                fqname: node_data.fqname.clone(),
                kind: intern_kind(node_data.kind.as_static_str())?,
                path: node_data.path.clone(),
                start_line: node_data.start_line,
                start_column: node_data.start_column,
                end_line: node_data.end_line,
                end_column: node_data.end_column,
                flags,
                imports,
                cached_hash: StdOnceLock::new(),
            };
            let global_idx = builder.intern_node(py, symbol)?;
            ref_to_global.insert(node_ref, global_idx);

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
    }

    // Pass 2: edge translation. Skip edges with unrecognized
    // endpoints — those reference Definitions in non-project files.
    for &file in project_files {
        let edges_payload = file_to_edges(db, file);
        for &(src, dst, flags) in &edges_payload.edges {
            let Some(src_idx) = lookup_or_mint_ref(py, db, &mut builder, &mut ref_to_global, src)?
            else {
                continue;
            };
            let Some(dst_idx) = lookup_or_mint_ref(py, db, &mut builder, &mut ref_to_global, dst)?
            else {
                continue;
            };
            builder.add_edge(src_idx, dst_idx, flags);
        }

        let ref_edges_payload = file_to_ref_edges(db, file);
        for &(src, dst, flags) in &ref_edges_payload.edges {
            let Some(src_idx) = lookup_or_mint_ref(py, db, &mut builder, &mut ref_to_global, src)?
            else {
                continue;
            };
            let Some(dst_idx) = lookup_or_mint_ref(py, db, &mut builder, &mut ref_to_global, dst)?
            else {
                continue;
            };
            builder.add_edge(src_idx, dst_idx, flags);
        }
        all_warnings.extend(ref_edges_payload.warnings.iter().cloned());
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
    })
}

/// Translate a `NodeRef` into its global graph index. For `Def` /
/// `Module` already-minted nodes, this is a hashmap probe. For
/// `External`, lazily mint the synthetic node via the builder.
/// Returns `Ok(None)` when the NodeRef is a `Def`/`Module` whose
/// owning file wasn't enumerated (cross-project edge endpoint that
/// ty resolved past the project boundary).
fn lookup_or_mint_ref<'db>(
    py: Python<'_>,
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
            let idx = builder.intern_synthetic(py, fqname)?;
            ref_to_global.insert(r, idx);
            Ok(Some(idx))
        }
        _ => Ok(None),
    }
}

/// Pre-build the fqname -> idx maps used by ``find_declarations``,
/// ``find_module``, and ``find_imports_of``. One pass over interned
/// nodes; module entries are 1:1 (one module node per fqname) while
/// decl entries (and per-upstream-module import entries) can have
/// multiple binders for the same key — try/except rebinds,
/// conditional re-imports, and multiple ``from X import Y, Z`` aliases
/// all bind into the same upstream module.
#[allow(clippy::type_complexity)]
pub(crate) fn build_fqname_indices(
    py: Python<'_>,
    builder: &GraphBuilder,
) -> (
    HashMap<String, Vec<usize>>,
    HashMap<String, usize>,
    HashMap<String, Vec<usize>>,
) {
    let mut decls: HashMap<String, Vec<usize>> = HashMap::new();
    let mut modules: HashMap<String, usize> = HashMap::new();
    let mut imports_by_module: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, node_py) in builder.nodes.iter().enumerate() {
        let node = node_py.borrow(py);
        match node.kind {
            "module" => {
                modules.insert(node.fqname.clone(), idx);
            }
            "function" | "class" | "variable" => {
                decls.entry(node.fqname.clone()).or_default().push(idx);
            }
            "import" => {
                decls.entry(node.fqname.clone()).or_default().push(idx);
                if let Some(import_py) = node.imports.as_ref() {
                    imports_by_module
                        .entry(import_py.borrow(py).module.clone())
                        .or_default()
                        .push(idx);
                }
            }
            _ => {}
        }
    }
    (decls, modules, imports_by_module)
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
    /// relative to the project.
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
    /// indicatif progress bars to stderr.
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
        let env = build_env_options(src_roots, extra_paths, python_env, python_version, typeshed)?;
        let db = make_db(root, env)?;
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
            let mut this = slf.borrow_mut(py);
            let outputs = build_project_graph(py, &mut this.db, show_progress)?;
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
            // Progress-bar label: use the plugin's class qualname.
            // Falls back to ``<unnamed>`` if anything goes wrong (e.g.
            // a plain function passed in by mistake).
            let plugin_name: String = plugin
                .bind(py)
                .get_type()
                .getattr("__qualname__")
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

impl ProjectContext {
    /// Borrow the active `BuildOutputs` or raise the standard
    /// "not materialized" error. Threads the `op` label into the
    /// error message so the caller name appears in the traceback.
    ///
    /// The returned `Ref` keeps the `RefCell` borrow alive for the
    /// lifetime of the receiver, so callers can hold it across an
    /// entire query body without re-borrowing.
    pub(crate) fn materialized(&self, op: &str) -> PyResult<Ref<'_, BuildOutputs>> {
        Ref::filter_map(self.outputs.borrow(), Option::as_ref).map_err(|_| not_materialized(op))
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
    pub(crate) fn find_module_dunders(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.materialized("find_module_dunders")?;
        let mut out = Vec::new();
        for node_py in &outputs.builder.nodes {
            let node = node_py.borrow(py);
            if !matches!(node.kind, "variable" | "function") {
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

        let outputs = self.materialized("find_nodes_matching_specs")?;
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
        let outputs = self.materialized("find_imports_of")?;
        // O(1) lookup against the pre-built ``imports_by_module``
        // index — no scan over all interned nodes. Empty when nothing
        // imports the module.
        let Some(idxs) = outputs.imports_by_module.get(module_name) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(idxs.len());
        for &idx in idxs {
            out.push(outputs.builder.nodes[idx].clone_ref(py));
        }
        Ok(out)
    }

    /// O(1) count of how many project import nodes target
    /// ``module_name`` — pre-built index lookup, no Py allocation.
    pub(crate) fn imports_of_count(&self, module_name: &str) -> usize {
        self.outputs
            .borrow()
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
        let outputs = self.materialized("find_declarations")?;
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
        let outputs = self.materialized("find_module")?;
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
        let outputs = self.materialized("module_for")?;
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
        let outputs = self.materialized("resolve")?;
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
        let outputs = self.materialized("module_surface")?;
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
    pub(crate) fn find_module_top_level_decls(
        &self,
        py: Python<'_>,
        module_fqn: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.materialized("find_module_top_level_decls")?;
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
        let outputs = self.materialized("find_module_dunder_all_exports")?;
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
        let outputs = self.materialized("decls_under")?;
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
        let outputs = self.materialized("decls_matching")?;
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
        let outputs = self.materialized("decls_matching_name")?;
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
        let outputs = self.materialized("descendants")?;
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
        let outputs = self.materialized("ancestors")?;
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
        let outputs = self.materialized("reachable")?;
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

#[pymethods]
impl ProjectContext {
    /// Return ``(module_node, [decls inside the block])`` for every
    /// file with a top-level ``if __name__ == "__main__":`` block.
    ///
    /// The decls list contains the file's class / function / variable
    /// / import nodes whose source position falls inside the block's
    /// range — same shape ``MainBlockPlugin``'s libcst path computes
    /// from the visitor's payload.
    pub(crate) fn find_main_blocks(&self, py: Python<'_>) -> PyResult<Vec<MainBlock>> {
        let outputs = self.materialized("find_main_blocks")?;
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
        decorator_module: &str,
        decorator_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, CallArgs)>> {
        let outputs = self.materialized("find_decorated_decls")?;
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
    #[allow(clippy::type_complexity)]
    pub(crate) fn find_instance_constructions(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, String, CallArgs)>> {
        let outputs = self.materialized("find_instance_constructions")?;
        let path_re = _compile_path_regex(path_regex)?;
        let allowed: HashSet<&str> = ctor_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = ctor_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let decl_by_fqname = &outputs.decl_by_fqname;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let allowed_ref = &allowed;
        let needles_ref: &[&str] = &needle_strs;
        let pairs: Vec<(usize, String, CallArgs)> = py.allow_threads(move || {
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
                let file_imports = collect_all_imports_local(&parsed);
                let mut local: Vec<(usize, String, CallArgs)> = Vec::new();
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
                            let call_args =
                                extract_call_args_kwargs(call, &file_imports, decl_by_fqname);
                            local.push((idx, matched, call_args));
                        }
                    }
                }
                local
            })
        });
        let out: Vec<(Py<SymbolNode>, String, CallArgs)> = pairs
            .into_iter()
            .map(|(idx, name, call_args)| {
                (outputs.builder.nodes[idx].clone_ref(py), name, call_args)
            })
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
        let outputs = self.materialized("find_handler_decorators")?;
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
        let outputs = self.materialized("find_handler_decorators_via")?;
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
}

#[pymethods]
impl ProjectContext {
    /// Top-level functions / classes whose body constructs one of
    /// ``ctor_names`` imported from ``module``.
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
        let outputs = self.materialized("find_factory_decls")?;
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
        module: &str,
        name: &str,
        arg_index: usize,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, String, CallArgs)>> {
        let outputs = self.materialized("find_calls_to_imported")?;
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
}

#[pymethods]
impl ProjectContext {
    /// Every class that defines a method with the given name.
    ///
    /// Walks each class's `DefinitionKind::Class` body for an
    /// `Stmt::FunctionDef` whose name matches. ty's `parsed_module` is
    /// Salsa-cached, so this is just a body scan per class.
    pub(crate) fn find_classes_defining_method(
        &self,
        py: Python<'_>,
        method_name: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.materialized("find_classes_defining_method")?;
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
        let outputs = self.materialized("find_subclasses_of")?;

        let Some((seed_file, seed_range)) = locate_class_def(
            &self.db,
            &outputs.path_to_file,
            &class_node.path,
            class_node,
        ) else {
            return Ok(Vec::new());
        };
        let out_idx =
            find_subclass_indices_via_refs(&self.db, &outputs, seed_file, seed_range, true);
        Ok(out_idx
            .into_iter()
            .map(|idx| outputs.builder.nodes[idx].clone_ref(py))
            .collect())
    }
}

impl ProjectContext {
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

    #[allow(clippy::type_complexity)]
    pub(crate) fn find_constructions(
        &self,
        py: Python<'_>,
        class_fqn: &str,
        include_subclasses: bool,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<SymbolNode>, CallArgs)>> {
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
        let triples = self.find_instance_constructions(py, module, ctors, path_regex)?;
        Ok(triples
            .into_iter()
            .map(|(node, _name, call_args)| (node, call_args))
            .collect())
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
}

#[pymethods]
impl ProjectContext {
    /// Subclasses of the class addressed by ``base_fqn``.
    ///
    /// Works for both project classes (where the fqn resolves to a
    /// graph node) and external classes (``unittest.TestCase``,
    /// ``pydantic.BaseModel``) via ty's module resolver +
    /// ``type_hierarchy_subtypes``. ``transitive=True`` (default)
    /// walks the full subclass closure; ``transitive=False`` returns
    /// only direct subclasses.
    #[pyo3(signature = (base_fqn, *, transitive = true))]
    pub(crate) fn find_subclasses(
        &self,
        py: Python<'_>,
        base_fqn: &str,
        transitive: bool,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        let outputs = self.materialized("find_subclasses")?;
        let Some((seed_file, seed_range)) = locate_class_seed(&self.db, &outputs, py, base_fqn)
        else {
            return Ok(Vec::new());
        };
        let out_idx =
            find_subclass_indices_via_refs(&self.db, &outputs, seed_file, seed_range, transitive);
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
        let outputs = self.materialized("find_comment_patterns")?;
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
        let outputs = self.materialized("nodes")?;
        Ok(outputs
            .builder
            .nodes
            .iter()
            .map(|n| n.clone_ref(py))
            .collect())
    }

    /// Live edges as `(src_idx, dst_idx, flags)` triples.
    pub(crate) fn edges(&self) -> PyResult<Vec<(usize, usize, u32)>> {
        Ok(self.materialized("edges")?.builder.edges.clone())
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
