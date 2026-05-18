//! ty-backed native graph builder for dead-cst.
//!
//! Architecture is governed by the crate's `CLAUDE.md`: ty does every
//! piece of Python semantics, ruff is only used where ty hasn't surfaced
//! the structure we need, and there is **no per-file cache** (ty's
//! Salsa db is the cache).
//!
//! The pipeline is one method (`Project.build()`) returning one
//! project-wide `NativeGraph`:
//!
//! 1. **Phase 1 — decls**. For every project file, iterate every
//!    binding in the file's global scope via
//!    `UseDefMap::all_definitions_with_usage`, minting a node per
//!    binding (including each name brought in by `from foo import *`).
//!    Each node lands in a global `(File, target_range) → node_idx`
//!    index so cross-file edges can find it later.
//! 2. **Phase 2 — chain**. For every module node, emit the submodule
//!    edge to its parent. For every import-kind binding, resolve the
//!    upstream target via `ty_module_resolver::resolve_module` and
//!    emit `alias_node → upstream_node`; lazily mint a module-only
//!    node for any target outside the project (stdlib / site-packages).
//! 3. **Phase 3 — references**. For every Definition that owns an
//!    expression (function body, class body, assignment value,
//!    annotation), walk the contained `Name`s and resolve each to its
//!    reaching def via `visible_ancestor_scopes` +
//!    `end_of_scope_symbol_bindings` (Principle 2 — the local alias
//!    is the target, not the upstream definition). Module-level
//!    non-definition statements attribute to the module.

#![allow(clippy::useless_conversion)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyClass;
use ruff_db::files::{File, FilePath};
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_db::source::{line_index, source_text};
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{Expr, ExprName, Stmt, StmtClassDef};
use ruff_source_file::LineIndex;
use ruff_text_size::{Ranged, TextRange};
use ty_module_resolver::{
    file_to_module, resolve_module, search_paths, ModuleName, ModuleResolveMode,
};
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::{RangedValue, RelativePathBuf};
use ty_project::{Db as ProjectDb, ProjectDatabase, ProjectMetadata};
use ty_python_core::ast_ids::HasScopedUseId;
use ty_python_core::definition::{DefinitionKind, DefinitionState, TargetKind};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::program::UseDefaultStrategy;
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;
use ty_python_core::SemanticIndex;
use ty_python_semantic::SemanticModel;

// ---------------------------------------------------------------------------
// Public Python data classes
// ---------------------------------------------------------------------------

/// Raw record of one cross-file import reference, attached to a
/// `kind="import"` node. Mirrors `dead_cst.graph.Import`.
///
/// `module` is the import's *absolute* dotted target (relative dots
/// are resolved by ty before this field is populated). `decl` is the
/// from-style imported name (`None` for plain `import` and for the
/// per-name nodes minted from `from X import *`). `star` flags the
/// implicit-from-star case.
#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct Import {
    module: String,
    decl: Option<String>,
    star: bool,
}

#[pymethods]
impl Import {
    #[new]
    #[pyo3(signature = (module, decl = None, star = false))]
    fn new(module: String, decl: Option<String>, star: bool) -> Self {
        Self { module, decl, star }
    }

    fn __repr__(&self) -> String {
        format!(
            "Import(module={:?}, decl={:?}, star={})",
            self.module, self.decl, self.star,
        )
    }
}

/// A single node in a `NativeGraph`.
///
/// `imports` is populated for `kind="import"` nodes only (one per
/// alias, plus one per name brought in by `from X import *`). All
/// other kinds carry `None`.
#[pyclass(get_all, frozen)]
struct NativeNode {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    flags: u32,
    imports: Option<Py<Import>>,
}

#[pymethods]
impl NativeNode {
    fn __repr__(&self) -> String {
        format!(
            "NativeNode(fqname={:?}, kind={:?}, path={:?}, start=({}, {}), end=({}, {}), flags={})",
            self.fqname,
            self.kind,
            self.path,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column,
            self.flags,
        )
    }
}

/// One project-wide graph contribution, packed for one FFI hop.
#[pyclass(get_all, frozen)]
struct NativeGraph {
    nodes: Vec<Py<NativeNode>>,
    edges: Vec<(usize, usize, u32)>,
}

#[pymethods]
impl NativeGraph {
    fn __repr__(&self) -> String {
        format!(
            "NativeGraph(nodes={}, edges={})",
            self.nodes.len(),
            self.edges.len(),
        )
    }
}

/// Key into the cross-file decl lookup.
///
/// `target_range` alone is ambiguous for star imports — every name
/// brought in by one `from foo import *` shares the same `*` alias
/// range. Including the bound `place_id` disambiguates star bindings
/// while still distinguishing two `def f` redefinitions by their
/// distinct target ranges.
type DeclKey = (File, ScopedPlaceId, (u32, u32));

type DeclIndex = HashMap<DeclKey, usize>;

/// Cloneable snapshot of an alias's import payload.
///
/// Lives in `alias_imports` (alias_node_idx -> spec) so the reference
/// collector can emit *parallel reachability edges* through an alias
/// without holding a PyO3 reference. Mirrors `Import`'s three fields.
#[derive(Clone, Debug)]
struct ImportSpec {
    module: String,
    decl: Option<String>,
    star: bool,
}

/// (upstream_file, local_decl_name) -> upstream decl node idx.
///
/// Populated during `ingest_decls` for every non-import, non-module
/// node. The value is a `Vec` so branch-bound names (try/except,
/// if/else where both branches assign) keep every live binding;
/// sequentially-rebound names collapse to the latest via the
/// post-pass that populates this from ty's
/// `end_of_scope_symbol_bindings`. Mirrors how the libcst pipeline's
/// trie excludes `SHADOWED` decls but keeps multi-branch ones.
type LiveDeclIndex = HashMap<(File, String), Vec<usize>>;

/// (file, name) -> idx of *any* live module-scope binding, decl or
/// import alias. Last-write-wins like `LiveDeclIndex`.
///
/// Used by `resolve_from_imported` so it can shortcut ty's full
/// `definitions_for_imported_symbol` walk (which recursively chases
/// alias chains across files) into a single hashmap probe.
///
/// The value is a `Vec` so that branch-bound names (try/except, if/else
/// where both branches assign) keep every live binding instead of
/// the last-write only. A cross-module `from lib import f` then
/// resolves to every reaching def of `f` in `lib`, matching the
/// libcst pipeline's `SHADOWED`-excluding trie merge over multiple
/// bindings.
type GlobalsByName = HashMap<(File, String), Vec<usize>>;

/// (file, name) -> upstream module name when `name` in `file` is bound
/// by `from <upstream> import *`. Populated alongside `GlobalsByName`
/// during Phase 1.
///
/// Lets `resolve_from_imported` walk a star-reexport chain
/// `A → B → C` and emit `consumer → C.g` directly when `from A import g`
/// resolves through the chain, matching the libcst pipeline's
/// fixed-point trie merge. Cycle-safe via a `seen` set in the caller.
type StarReexports = HashMap<(File, String), String>;

/// Return type of `ProjectContext.find_main_blocks`: one entry per
/// file with a top-level ``if __name__ == "__main__":`` block, paired
/// with the decls that fall inside it.
type MainBlock = (Py<NativeNode>, Vec<Py<NativeNode>>);

/// Outcome of resolving a `Name` use to its reaching definition.
///
/// `Alias` is the module-scope path: the use has a local graph node
/// (an import alias or a top-level decl) that takes the in-edge.
/// `NestedImport` is the function-/class-scope path: ty saw an import
/// binding in a non-global scope, so no graph node was minted, and
/// the use's parallel upstream edges flow from the enclosing top-level
/// owner instead.
enum Resolution {
    Alias(usize),
    NestedImport {
        spec: ImportSpec,
        bound_name: String,
    },
}

// ---------------------------------------------------------------------------
// GraphBuilder + node interning
// ---------------------------------------------------------------------------

/// Positional identity for a node.
///
/// Per `CLAUDE.md` principle 3, `flags` is deliberately *not* part of
/// the key: two nodes for the same `(fqname, kind, path, position)` are
/// the same node regardless of which path computed their flags.
#[derive(Hash, Eq, PartialEq, Clone)]
struct NodeKey {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

struct GraphBuilder {
    nodes: Vec<Py<NativeNode>>,
    node_index: HashMap<NodeKey, usize>,
    edges: Vec<(usize, usize, u32)>,
    edge_set: HashSet<(usize, usize, u32)>,
    /// Per-node forward / reverse adjacency lists kept in sync with
    /// ``edges``. ``bfs`` reads these so traversals are O(deg(i)) per
    /// pop instead of O(|edges|).
    forward_adj: Vec<Vec<(usize, u32)>>,
    reverse_adj: Vec<Vec<(usize, u32)>>,
    /// `{ pyi_file -> py_twin_file }` for peer ``.pyi`` files whose
    /// ``.py`` twin is also in the project. Both files get ingested
    /// independently (the rust path differs from libcst here — we
    /// trust ty's per-source-type understanding). The map drives two
    /// behaviors that close the liveness gap when ty's module
    /// resolver prefers the stub:
    ///
    /// * ``resolve_from_imported`` falls back to the .py twin's
    ///   namespace when the .pyi lookup misses, so the consumer's
    ///   ``alias -> upstream decl`` parallel edge lands on the
    ///   runtime decl instead of being dropped.
    /// * ``file_default_flags`` distinguishes peer ``.pyi`` (no
    ///   default flag; liveness flows via the fallback) from
    ///   stub-only ``.pyi`` (no .py twin -> flagged ``ENTRYPOINT``
    ///   so native-extension / protobuf-style stubs stay alive
    ///   artificially even when no consumer references them).
    peer_pyi_to_py: HashMap<File, File>,
    /// `{ synthetic_fqname -> node idx }` for ``[external dist] X`` and
    /// ``[unresolved] X`` synthetics. Synthetics are deduplicated by
    /// fqname project-wide: every site that imports ``rustworkx``
    /// resolves to the same ``[external dist] rustworkx`` node, so
    /// reachability and the codemod's "this import has no
    /// dependents" query both work on a single anchor.
    synthetic_nodes: HashMap<String, usize>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            edges: Vec::new(),
            edge_set: HashSet::new(),
            forward_adj: Vec::new(),
            reverse_adj: Vec::new(),
            peer_pyi_to_py: HashMap::new(),
            synthetic_nodes: HashMap::new(),
        }
    }

    fn intern_node(&mut self, py: Python<'_>, node: NativeNode) -> PyResult<usize> {
        let key = node_key_of(&node);
        if let Some(&idx) = self.node_index.get(&key) {
            return Ok(idx);
        }
        let idx = self.nodes.len();
        self.nodes.push(Py::new(py, node)?);
        self.node_index.insert(key, idx);
        self.forward_adj.push(Vec::new());
        self.reverse_adj.push(Vec::new());
        Ok(idx)
    }

    /// Get (or mint) the deduplicated synthetic node with the given
    /// fully qualified name. Synthetics anchor edges to imports the
    /// project doesn't own — stdlib stays silent, ``[external dist] X``
    /// covers third-party site-packages, ``[unresolved] X`` covers
    /// genuinely-missing top-level names. Path is empty (synthetics
    /// don't correspond to a file in the project tree) and position
    /// is the (0, 0) sentinel.
    fn intern_synthetic(&mut self, py: Python<'_>, fqname: String) -> PyResult<usize> {
        if let Some(&idx) = self.synthetic_nodes.get(&fqname) {
            return Ok(idx);
        }
        let idx = self.intern_node(
            py,
            NativeNode {
                fqname: fqname.clone(),
                kind: "synthetic",
                path: String::new(),
                start_line: 0,
                start_column: 0,
                end_line: 0,
                end_column: 0,
                flags: 0,
                imports: None,
            },
        )?;
        self.synthetic_nodes.insert(fqname, idx);
        Ok(idx)
    }

    fn add_edge(&mut self, src: usize, dst: usize, flags: u32) {
        let triple = (src, dst, flags);
        if self.edge_set.insert(triple) {
            self.edges.push(triple);
            self.forward_adj[src].push((dst, flags));
            self.reverse_adj[dst].push((src, flags));
        }
    }
}

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

/// A ty-backed analysis project with explicitly-injected configuration.
#[pyclass(unsendable)]
struct Project {
    db: ProjectDatabase,
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
    fn new(
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
    fn build(&self, py: Python<'_>) -> PyResult<NativeGraph> {
        let outputs = build_project_graph(py, &self.db)?;
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
struct BuildOutputs {
    builder: GraphBuilder,
    project_files: Vec<File>,
    global_index: DeclIndex,
    /// `file_path_string(file) -> File` so seed lookups don't have to
    /// linear-scan `project_files`. Populated alongside ingest.
    path_to_file: HashMap<String, File>,
    /// `(file, class_target_range_key) -> class node idx`. Lets
    /// `find_subclasses_of` map ty's `TypeHierarchyClass.selection_range`
    /// back to a graph node in O(1).
    class_by_selection: HashMap<(File, (u32, u32)), usize>,
    /// `file -> module node idx`. Lets `find_main_blocks` reach the
    /// file's module node without a linear scan over `builder.nodes`.
    module_nodes_by_file: HashMap<File, usize>,
    /// `(file, target_range_key) -> node idx`. Sister of
    /// ``class_by_selection`` but for every top-level decl ingest minted
    /// (function / class / variable / import). Lets ``find_decorated_decls``
    /// and the dispatch-app queries map an AST node's target range to a
    /// graph node in O(1) instead of scanning the full ``global_index``.
    decl_by_name_range: HashMap<(File, (u32, u32)), usize>,
    /// `decl_fqname -> [node idx]`. Lets ``find_declarations`` answer
    /// in O(parts) instead of O(parts × all_nodes). Multiple entries
    /// per fqname arise from try/except rebinds and conditional
    /// re-imports.
    decl_by_fqname: HashMap<String, Vec<usize>>,
    /// `module_fqname -> node idx`. Lets ``find_module`` answer in
    /// O(1) instead of scanning all nodes.
    module_by_fqname: HashMap<String, usize>,
}

/// Run the three build phases (ingest → hierarchy+imports → references)
/// and return every index the plugin queries need.
fn build_project_graph(py: Python<'_>, db: &ProjectDatabase) -> PyResult<BuildOutputs> {
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
    }
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
    }
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
    }
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
fn build_fqname_indices(
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

// ---------------------------------------------------------------------------
// ProjectContext — plugin protocol entry point
// ---------------------------------------------------------------------------

/// Add an edge between two interned nodes.
///
/// ``flags`` carries ``DEAD_BRANCH`` / future edge classifications.
/// Plugins yield this from ``run(ctx)`` instead of mutating the graph
/// directly so the apply pass is a single atomic step on the rust side.
#[pyclass(frozen, get_all)]
struct AddEdge {
    src: Py<NativeNode>,
    dst: Py<NativeNode>,
    flags: u32,
}

#[pymethods]
impl AddEdge {
    #[new]
    #[pyo3(signature = (src, dst, *, flags = 0))]
    fn new(src: Py<NativeNode>, dst: Py<NativeNode>, flags: u32) -> Self {
        Self { src, dst, flags }
    }
}

/// Mark ``decl`` as an entrypoint.
///
/// ``marker`` is a self-documenting label (``"<celery-worker>"``,
/// ``"<external-execution>:alembic"``, ...) shown in ``why-alive`` to
/// explain *why* the decl is alive without minting a synthetic graph
/// node for the reason.
#[pyclass(frozen, get_all)]
struct AddEntrypoint {
    decl: Py<NativeNode>,
    marker: String,
}

#[pymethods]
impl AddEntrypoint {
    #[new]
    #[pyo3(signature = (decl, *, marker))]
    fn new(decl: Py<NativeNode>, marker: String) -> Self {
        Self { decl, marker }
    }
}

/// Mint a synthetic intermediate node.
///
/// ``edges_from`` / ``edges_to`` wire the new node atomically — every
/// element of ``edges_from`` becomes a ``source -> this`` edge, every
/// element of ``edges_to`` a ``this -> target`` edge — so a plugin
/// doesn't need a separate handle to reference the freshly-minted node
/// from subsequent ops. Set ``flags = NodeFlags.ENTRYPOINT`` to make
/// the node a seed (``AddEntrypoint`` is the single-target sugar).
#[pyclass(frozen, get_all)]
struct AddNode {
    fqname: String,
    kind: &'static str,
    path: String,
    flags: u32,
    edges_from: Vec<Py<NativeNode>>,
    edges_to: Vec<Py<NativeNode>>,
}

#[pymethods]
impl AddNode {
    #[new]
    #[pyo3(signature = (
        fqname,
        *,
        path,
        kind = "synthetic",
        flags = 0,
        edges_from = Vec::new(),
        edges_to = Vec::new(),
    ))]
    fn new(
        fqname: String,
        path: String,
        kind: &str,
        flags: u32,
        edges_from: Vec<Py<NativeNode>>,
        edges_to: Vec<Py<NativeNode>>,
    ) -> PyResult<Self> {
        Ok(Self {
            fqname,
            kind: static_kind_str(kind)?,
            path,
            flags,
            edges_from,
            edges_to,
        })
    }
}

// ---------------------------------------------------------------------------
// Builder query API: result types
// ---------------------------------------------------------------------------

/// One decorator application on a top-level function or class.
///
/// Field nullability follows the query shape that produced the ref:
/// * ``where_module + where_name`` populates ``decorated`` only.
/// * ``where_owner_attr`` fills ``decorator_owner`` (the textual
///   ``@<owner>.<attr>`` prefix).
/// * ``where_owner_attr_via`` additionally fills ``decorator_via``
///   with the middle attribute name.
#[pyclass(frozen, get_all)]
struct DecoratorRef {
    decorated: Py<NativeNode>,
    decorator_name: Option<String>,
    decorator_owner: Option<String>,
    decorator_via: Option<String>,
}

#[pymethods]
impl DecoratorRef {
    /// File path of the decorated decl. Read off ``decorated.path`` —
    /// surfaced as a top-level attribute for ergonomics in path-keyed
    /// dispatch.
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.decorated.borrow(py).path.clone()
    }
}

/// One ``<var> = <Ctor>(...)`` construction at module scope.
///
/// ``class_name`` is the upstream constructor's bare name
/// (``"Flask"`` even when imported as ``F``).
#[pyclass(frozen, get_all)]
struct ConstructionRef {
    var: Py<NativeNode>,
    class_name: String,
}

#[pymethods]
impl ConstructionRef {
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.var.borrow(py).path.clone()
    }
}

/// One matched call site. ``string_arg`` is the literal at the
/// positional index passed to :meth:`CallQuery.string_arg_at`.
#[pyclass(frozen, get_all)]
struct CallRef {
    owner: Py<NativeNode>,
    string_arg: String,
}

#[pymethods]
impl CallRef {
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.owner.borrow(py).path.clone()
    }
}

// ---------------------------------------------------------------------------
// Builder query API: builders
// ---------------------------------------------------------------------------

/// Entry point for the chainable query API. Returned by
/// :meth:`ProjectContext.query`. Pick a stream type:
/// :meth:`decorators`, :meth:`constructions`, or :meth:`calls`.
#[pyclass(unsendable)]
struct QueryBuilder {
    ctx: Py<ProjectContext>,
}

#[pymethods]
impl QueryBuilder {
    fn decorators(&self, py: Python<'_>) -> DecoratorQuery {
        DecoratorQuery::new(self.ctx.clone_ref(py))
    }
    fn constructions(&self, py: Python<'_>) -> ConstructionQuery {
        ConstructionQuery::new(self.ctx.clone_ref(py))
    }
    fn calls(&self, py: Python<'_>) -> CallQuery {
        CallQuery::new(self.ctx.clone_ref(py))
    }
}

fn _extract_str_or_list(py: Python<'_>, obj: PyObject) -> PyResult<Vec<String>> {
    let bound = obj.bind(py);
    if let Ok(s) = bound.extract::<String>() {
        Ok(vec![s])
    } else {
        bound.extract::<Vec<String>>()
    }
}

/// Compile an optional path regex once for a query's file-iteration
/// loop. Centralized so every ``find_*`` method that takes a
/// ``path_regex`` parameter shares the same error reporting.
fn _compile_path_regex(re_str: Option<&str>) -> PyResult<Option<regex::Regex>> {
    match re_str {
        None => Ok(None),
        Some(s) => regex::Regex::new(s)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("invalid path regex {s:?}: {e}"))),
    }
}

/// Per-file predicate-fusion check. ``true`` when the file should be
/// processed (no regex, or regex matches its absolute path).
fn _path_re_matches(re: &Option<regex::Regex>, db: &dyn ProjectDb, file: File) -> bool {
    match re {
        None => true,
        Some(re) => re.is_match(&file_path_string(db, file)),
    }
}

/// Cheap text prefilter for identifier references before AST/semantic
/// validation. Mirrors `ty_ide::references::contains_identifier`
/// (vendor/ruff/crates/ty_ide/src/references.rs:198) so per-file
/// queries can skip the parse + walk when the file source doesn't
/// even mention the target identifier.
///
/// Matches an ASCII approximation of `\b<name>\b`: every occurrence
/// of `needle` in `source` whose surrounding bytes aren't identifier
/// continuations. Used by every decorator / construction / call /
/// method query that walks ``project_files``; saves the parse on the
/// (typically large) majority of files that don't reference the
/// query's target name at all.
fn _contains_identifier(source: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = source.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut start = 0;
    while let Some(rel) = source[start..].find(needle) {
        let pos = start + rel;
        let after = pos + needle_bytes.len();
        let boundary_before = pos == 0 || !_is_ident_continue(bytes[pos - 1]);
        let boundary_after = bytes
            .get(after)
            .is_none_or(|byte| !_is_ident_continue(*byte));
        if boundary_before && boundary_after {
            return true;
        }
        start = pos + 1;
    }
    false
}

fn _is_ident_continue(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// Multi-needle variant of [`_contains_identifier`]: returns ``true``
/// as soon as any one of ``needles`` is found. Used by queries that
/// take a list of names (decorator name set, ctor name set, …) so the
/// per-file prefilter passes when the file mentions any of them.
fn _contains_any_identifier(source: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| _contains_identifier(source, n))
}

/// Generic parallel per-file walk. ``per_file`` runs on a Salsa
/// snapshot of ``db`` (one ``Db::dyn_clone`` per worker, mirroring
/// the ty_ide find_references pattern at
/// ``vendor/ruff/crates/ty_ide/src/references.rs:107-130``) and
/// returns a ``Vec<T>`` of opaque per-file results.
///
/// Caller is responsible for releasing the GIL with
/// :meth:`pyo3::Python::allow_threads` — the closure passed in must
/// be ``Send + Sync`` and ``T`` must be ``Send``. Materializing
/// ``Py<NativeNode>`` values (which are GIL-bound) belongs in the
/// caller AFTER ``allow_threads`` returns.
fn par_scan_files<T, F>(
    db: Box<dyn ProjectDb>,
    files: &[File],
    path_re: &Option<regex::Regex>,
    per_file: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(&dyn ProjectDb, File) -> Vec<T> + Send + Sync,
{
    let result = std::sync::Mutex::new(Vec::<T>::new());
    let per_file_ref = &per_file;
    let result_ref = &result;
    // `move` captures `db: Box<dyn ProjectDb>` by value — `dyn Db`
    // has a `Send` supertrait via `salsa::Database`, so the box is
    // Send, but `&dyn Db` is NOT Send (the trait isn't Sync), which
    // is why the box can't be borrowed across the rayon scope.
    rayon::scope(move |s| {
        for &file in files {
            if !_path_re_matches(path_re, &*db, file) {
                continue;
            }
            let db_t: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&*db);
            s.spawn(move |_| {
                let local = per_file_ref(&*db_t, file);
                if !local.is_empty() {
                    result_ref.lock().unwrap().extend(local);
                }
            });
        }
    });
    result.into_inner().unwrap_or_default()
}

fn _to_iter(py: Python<'_>, items: Vec<Py<impl PyClass>>) -> PyResult<PyObject> {
    let list = pyo3::types::PyList::new_bound(py, items);
    let iter_obj = list.call_method0("__iter__")?;
    Ok(iter_obj.unbind())
}

/// Find decorated top-level functions / classes. Pick exactly one of:
/// * ``where_module(m).where_name(n)`` — ``@m.x`` / ``@x`` where ``x``
///   is imported from ``m``.
/// * ``where_callee(fqn)`` — fqn-form ``@<fqn>``.
/// * ``where_owner_attr(attrs)`` — ``@<owner>.<attr>(...)``;
///   ``decorator_owner`` carries the textual prefix.
/// * ``where_owner_attr_via(via, attrs)`` —
///   ``@<owner>.<via>.<attr>(...)`` two-level chain.
/// * ``in_decl(node).where_name(names)`` — ``@<node>.<name>``
///   same-file instance-method decorators.
#[pyclass(unsendable)]
struct DecoratorQuery {
    ctx: Py<ProjectContext>,
    module: Option<String>,
    callee_fqn: Option<String>,
    names: Option<Vec<String>>,
    owner_attrs: Option<Vec<String>>,
    via_attr: Option<String>,
    in_decl: Option<Py<NativeNode>>,
    path_regex: Option<String>,
}

impl DecoratorQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            module: None,
            callee_fqn: None,
            names: None,
            owner_attrs: None,
            via_attr: None,
            in_decl: None,
            path_regex: None,
        }
    }
}

#[pymethods]
impl DecoratorQuery {
    fn where_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
    }
    fn where_callee<'py>(mut slf: PyRefMut<'py, Self>, fqn: String) -> PyRefMut<'py, Self> {
        slf.callee_fqn = Some(fqn);
        slf
    }
    fn where_name<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.names = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    fn where_owner_attr<'py>(
        mut slf: PyRefMut<'py, Self>,
        attrs: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.owner_attrs = Some(_extract_str_or_list(py, attrs)?);
        Ok(slf)
    }
    fn where_owner_attr_via<'py>(
        mut slf: PyRefMut<'py, Self>,
        via: String,
        attrs: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.via_attr = Some(via);
        slf.owner_attrs = Some(_extract_str_or_list(py, attrs)?);
        Ok(slf)
    }
    fn in_decl<'py>(mut slf: PyRefMut<'py, Self>, node: Py<NativeNode>) -> PyRefMut<'py, Self> {
        slf.in_decl = Some(node);
        slf
    }
    fn where_path<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<DecoratorRef>>> {
        let ctx = self.ctx.borrow(py);
        let path_regex = self.path_regex.as_deref();
        let mut refs: Vec<Py<DecoratorRef>> = Vec::new();
        if let Some(owner_attrs) = &self.owner_attrs {
            let pairs = if let Some(via) = &self.via_attr {
                ctx.find_handler_decorators_via(py, via, owner_attrs.clone(), path_regex)?
            } else {
                ctx.find_handler_decorators(py, owner_attrs.clone(), path_regex)?
            };
            for (owner_name, decorated) in pairs {
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated,
                        decorator_name: None,
                        decorator_owner: Some(owner_name),
                        decorator_via: self.via_attr.clone(),
                    },
                )?);
            }
        } else if let Some(in_decl_node) = &self.in_decl {
            let names = self.names.as_ref().ok_or_else(|| {
                PyValueError::new_err("DecoratorQuery.in_decl(...) requires .where_name(...)")
            })?;
            let in_decl_ref = in_decl_node.borrow(py);
            let decls = ctx.find_decorations_on(py, &in_decl_ref, names.clone(), path_regex)?;
            let owner_simple = in_decl_ref
                .fqname
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_string();
            drop(in_decl_ref);
            for d in decls {
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated: d,
                        decorator_name: None,
                        decorator_owner: Some(owner_simple.clone()),
                        decorator_via: None,
                    },
                )?);
            }
        } else if let Some(fqn) = &self.callee_fqn {
            let decls = ctx.find_decorated(py, fqn, path_regex)?;
            for d in decls {
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated: d,
                        decorator_name: None,
                        decorator_owner: None,
                        decorator_via: None,
                    },
                )?);
            }
        } else if let (Some(module), Some(names)) = (&self.module, &self.names) {
            let decls = ctx.find_decorated_decls(py, module, names.clone(), path_regex)?;
            for d in decls {
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated: d,
                        decorator_name: None,
                        decorator_owner: None,
                        decorator_via: None,
                    },
                )?);
            }
        } else {
            return Err(PyValueError::new_err(
                "DecoratorQuery requires one of: where_callee(...); \
                 where_module(...) + where_name(...); where_owner_attr(...); \
                 where_owner_attr_via(via, attrs); or in_decl(node) + where_name(...)",
            ));
        }
        Ok(refs)
    }

    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<DecoratorRef>>> {
        Ok(self.collect(py)?.into_iter().next())
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

/// Find module-scope ``<var> = <Ctor>(...)`` sites. Pick exactly one
/// of ``where_module + where_name`` or
/// ``where_class(fqn, include_subclasses=...)``.
#[pyclass(unsendable)]
struct ConstructionQuery {
    ctx: Py<ProjectContext>,
    module: Option<String>,
    names: Option<Vec<String>>,
    class_fqn: Option<String>,
    include_subclasses: bool,
    path_regex: Option<String>,
}

impl ConstructionQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            module: None,
            names: None,
            class_fqn: None,
            include_subclasses: false,
            path_regex: None,
        }
    }
}

#[pymethods]
impl ConstructionQuery {
    fn where_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
    }
    fn where_name<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.names = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    #[pyo3(signature = (fqn, *, include_subclasses = false))]
    fn where_class<'py>(
        mut slf: PyRefMut<'py, Self>,
        fqn: String,
        include_subclasses: bool,
    ) -> PyRefMut<'py, Self> {
        slf.class_fqn = Some(fqn);
        slf.include_subclasses = include_subclasses;
        slf
    }
    fn where_path<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<ConstructionRef>>> {
        let ctx = self.ctx.borrow(py);
        let path_regex = self.path_regex.as_deref();
        let mut refs: Vec<Py<ConstructionRef>> = Vec::new();
        if let Some(fqn) = &self.class_fqn {
            let decls = ctx.find_constructions(py, fqn, self.include_subclasses, path_regex)?;
            let cls_name = fqn.rsplit('.').next().unwrap_or("").to_string();
            for d in decls {
                refs.push(Py::new(
                    py,
                    ConstructionRef {
                        var: d,
                        class_name: cls_name.clone(),
                    },
                )?);
            }
        } else if let (Some(module), Some(names)) = (&self.module, &self.names) {
            let pairs = ctx.find_instance_constructions(py, module, names.clone(), path_regex)?;
            for (var, name) in pairs {
                refs.push(Py::new(
                    py,
                    ConstructionRef {
                        var,
                        class_name: name,
                    },
                )?);
            }
        } else {
            return Err(PyValueError::new_err(
                "ConstructionQuery requires either where_class(...) \
                 or where_module(...) + where_name(...)",
            ));
        }
        Ok(refs)
    }

    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<ConstructionRef>>> {
        Ok(self.collect(py)?.into_iter().next())
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

/// Find call sites whose positional string-literal at the configured
/// index is captured. :meth:`string_arg_at` is required. Pick one of:
/// * ``where_module(m).where_name(n)`` — call to ``n`` imported from
///   ``m``.
/// * ``where_owner(o).where_attr(a)`` — ``<o>.<a>(...)`` literal
///   receiver match.
/// * ``where_attr(a)`` — ``<expr>.<a>(...)`` any receiver.
#[pyclass(unsendable)]
struct CallQuery {
    ctx: Py<ProjectContext>,
    module: Option<String>,
    name: Option<String>,
    owner: Option<String>,
    attr: Option<String>,
    arg_index: Option<usize>,
    required_positional: Option<usize>,
    path_regex: Option<String>,
}

impl CallQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            module: None,
            name: None,
            owner: None,
            attr: None,
            arg_index: None,
            required_positional: None,
            path_regex: None,
        }
    }
}

#[pymethods]
impl CallQuery {
    fn where_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
    }
    fn where_name<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.name = Some(name);
        slf
    }
    fn where_owner<'py>(mut slf: PyRefMut<'py, Self>, owner: String) -> PyRefMut<'py, Self> {
        slf.owner = Some(owner);
        slf
    }
    fn where_attr<'py>(mut slf: PyRefMut<'py, Self>, attr: String) -> PyRefMut<'py, Self> {
        slf.attr = Some(attr);
        slf
    }
    fn string_arg_at<'py>(mut slf: PyRefMut<'py, Self>, index: usize) -> PyRefMut<'py, Self> {
        slf.arg_index = Some(index);
        slf
    }
    #[pyo3(signature = (n=None))]
    fn where_required_positional<'py>(
        mut slf: PyRefMut<'py, Self>,
        n: Option<usize>,
    ) -> PyRefMut<'py, Self> {
        slf.required_positional = n;
        slf
    }
    fn where_path<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<CallRef>>> {
        let ctx = self.ctx.borrow(py);
        let arg_index = self
            .arg_index
            .ok_or_else(|| PyValueError::new_err("CallQuery: .string_arg_at(index) is required"))?;
        let path_regex = self.path_regex.as_deref();
        let pairs = if let (Some(module), Some(name)) = (&self.module, &self.name) {
            ctx.find_calls_to_imported(py, module, name, arg_index, path_regex)?
        } else if let (Some(owner), Some(attr)) = (&self.owner, &self.attr) {
            ctx.find_calls_on_var(
                py,
                owner,
                attr,
                arg_index,
                self.required_positional,
                path_regex,
            )?
        } else if let Some(attr) = &self.attr {
            ctx.find_calls_on_attr(py, attr, arg_index, path_regex)?
        } else {
            return Err(PyValueError::new_err(
                "CallQuery requires one of: where_module(...) + where_name(...); \
                 where_owner(...) + where_attr(...); or where_attr(...)",
            ));
        };
        let mut refs: Vec<Py<CallRef>> = Vec::new();
        for (owner_node, s) in pairs {
            refs.push(Py::new(
                py,
                CallRef {
                    owner: owner_node,
                    string_arg: s,
                },
            )?);
        }
        Ok(refs)
    }

    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<CallRef>>> {
        Ok(self.collect(py)?.into_iter().next())
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
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
struct ProjectContext {
    db: ProjectDatabase,
    /// Absolute path of the project root, echoed back to Python via the
    /// :attr:`project_root` getter. Plugins use it to compute paths
    /// relative to the project (e.g. ``ExplicitEntrypointPlugin`` matching
    /// path specs).
    root: String,
    plugins: Vec<PyObject>,
    /// Populated by `materialize` before plugins run. `None` outside a
    /// materialize call — `apply_graph_op` / queries assume it's
    /// `Some` and error if a plugin (incorrectly) caches the ctx and
    /// uses it after materialize returns.
    outputs: RefCell<Option<BuildOutputs>>,
    /// Compiled regexes keyed by the source pattern. Plugins call
    /// :meth:`decls_matching_name` / :meth:`find_comment_patterns`
    /// repeatedly across files with the same pattern, so caching keeps
    /// us off the regex compiler in the hot path.
    regex_cache: RefCell<HashMap<String, regex::Regex>>,
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
    ))]
    fn new(
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
        Ok(Self {
            db,
            root: root.to_string(),
            plugins: Vec::new(),
            outputs: RefCell::new(None),
            regex_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Absolute project root passed at construction.
    #[getter]
    fn project_root(&self) -> &str {
        &self.root
    }

    /// Register a Python plugin. Order of registration is order of
    /// invocation during `materialize`.
    fn add_plugin(&mut self, plugin: PyObject) {
        self.plugins.push(plugin);
    }

    /// Open a chainable query builder against this context.
    ///
    /// Equivalent to the top-level :func:`query` function; both return
    /// a :class:`QueryBuilder` that can chain into
    /// :class:`DecoratorQuery` / :class:`ConstructionQuery` /
    /// :class:`CallQuery`. See the result-type docstrings for the
    /// predicate vocabulary.
    fn query(slf: Py<Self>, _py: Python<'_>) -> QueryBuilder {
        QueryBuilder { ctx: slf }
    }

    /// Build the project-wide graph, run each plugin's `run(ctx)`,
    /// then snapshot the final state.
    ///
    /// Borrows are released between phases so plugin `run` methods can
    /// re-enter queries through the same ctx without aliasing violations.
    fn materialize(slf: Py<Self>, py: Python<'_>) -> PyResult<NativeGraph> {
        {
            let this = slf.borrow(py);
            let outputs = build_project_graph(py, &this.db)?;
            *this.outputs.borrow_mut() = Some(outputs);
        }

        let plugins: Vec<PyObject> = slf
            .borrow(py)
            .plugins
            .iter()
            .map(|p| p.clone_ref(py))
            .collect();
        for plugin in &plugins {
            // ``plugin.run(ctx)`` yields ``GraphOp`` values; we apply
            // each as it comes off the iterator. The plugin can run
            // queries against ``ctx`` mid-iteration since each
            // ``apply_graph_op`` call releases its borrows before
            // returning control to the generator. ``None`` (a regular
            // function that ran to completion without yielding) is
            // allowed for plugins with nothing to do.
            let result = plugin.bind(py).call_method1("run", (slf.clone_ref(py),))?;
            if !result.is_none() {
                for item in result.iter()? {
                    let op = item?;
                    apply_graph_op(&slf, py, &op)?;
                }
            }
        }

        let outputs =
            slf.borrow(py).outputs.borrow_mut().take().ok_or_else(|| {
                PyRuntimeError::new_err("ProjectContext was already materialized")
            })?;
        Ok(NativeGraph {
            nodes: outputs.builder.nodes,
            edges: outputs.builder.edges,
        })
    }

    // ----- Queries (rust-resident, results pass back to Python) -----------

    /// Return every top-level variable node whose name matches `__xxx__`.
    ///
    /// Pure scan over already-interned nodes — no ty re-query needed —
    /// because the visitor's decl pass already minted one node per
    /// global-scope variable binding.
    fn find_module_dunders(&self, py: Python<'_>) -> PyResult<Vec<Py<NativeNode>>> {
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

    /// Return every import-kind node whose upstream `module` matches.
    ///
    /// Covers both `import <module_name>` and
    /// `from <module_name> import ...` styles — both bind import-kind
    /// nodes whose `Import.module` is the absolute dotted name. Star
    /// reexports synthesized from `from <module_name> import *` are
    /// also included.
    fn find_imports_of(&self, py: Python<'_>, module_name: &str) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn find_declarations(&self, py: Python<'_>, fqname: &str) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn find_module(&self, py: Python<'_>, fqname: &str) -> PyResult<Option<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module"))?;
        Ok(outputs
            .module_by_fqname
            .get(fqname)
            .map(|&idx| outputs.builder.nodes[idx].clone_ref(py)))
    }

    /// Return the module node owning ``path``, if any. O(1) — backed
    /// by the same ``module_nodes_by_file`` index `find_main_blocks`
    /// uses, so plugins don't have to scan ``nodes()`` per call.
    fn module_for(&self, py: Python<'_>, path: &str) -> PyResult<Option<Py<NativeNode>>> {
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
    fn resolve(&self, py: Python<'_>, fqname: &str) -> PyResult<Option<Py<NativeNode>>> {
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
    fn module_surface(&self, py: Python<'_>, module_fqn: &str) -> PyResult<Vec<Py<NativeNode>>> {
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

    /// Return ``module_fqn``'s immediate top-level decls — every
    /// function / class / variable / import bound at its module scope.
    ///
    /// Models ``from module_fqn import *``: only the names that
    /// statement would bind into the importing scope. Unlike
    /// :meth:`module_surface`, submodules and their decls are
    /// excluded — a `from p.functions import *` doesn't pull in
    /// `p.functions.sub.x`. Empty list when ``module_fqn`` doesn't
    /// resolve to a project module.
    fn find_module_top_level_decls(
        &self,
        py: Python<'_>,
        module_fqn: &str,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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
    /// The visitor's ``emit_dunder_all_edges`` already wires
    /// ``__all__`` → each string-listed decl as a regular edge; this
    /// query walks those successor edges, filters out the default
    /// ``decl -> parent_module`` edge, and returns the rest. The
    /// distinction between "no ``__all__``" (``None``) and "empty
    /// ``__all__``" (``Some([])``) matters: callers that want
    /// CPython's ``from X import *`` semantics should fall back to
    /// the non-underscore decl list only in the ``None`` case.
    fn find_module_dunder_all_exports(
        &self,
        py: Python<'_>,
        module_fqn: &str,
    ) -> PyResult<Option<Vec<Py<NativeNode>>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_module_dunder_all_exports"))?;
        let all_fqn = format!("{module_fqn}.__all__");
        let Some(idxs) = outputs.decl_by_fqname.get(&all_fqn) else {
            return Ok(None);
        };
        let module_idx = outputs.module_by_fqname.get(module_fqn).copied();
        let mut out: Vec<Py<NativeNode>> = Vec::new();
        for &all_idx in idxs {
            for &(dst, _flags) in &outputs.builder.forward_adj[all_idx] {
                if Some(dst) == module_idx {
                    // Skip the default ``decl -> parent_module`` edge.
                    continue;
                }
                out.push(outputs.builder.nodes[dst].clone_ref(py));
            }
        }
        Ok(Some(out))
    }

    /// Every node whose ``path`` starts with the given prefix.
    fn decls_under(&self, py: Python<'_>, path_prefix: &str) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn decls_matching(&self, py: Python<'_>, substring: &str) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn decls_matching_name(&self, py: Python<'_>, pattern: &str) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn descendants(
        &self,
        py: Python<'_>,
        root: &NativeNode,
        skip_flags: u32,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn ancestors(
        &self,
        py: Python<'_>,
        decl: &NativeNode,
        skip_flags: u32,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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

    /// Forward closure from every entrypoint-flagged node. The set of
    /// dead decls is the complement against ``nodes()``.
    #[pyo3(signature = (*, skip_flags = 0))]
    fn reachable(&self, py: Python<'_>, skip_flags: u32) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("reachable"))?;
        let seeds = outputs
            .builder
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, n)| (n.borrow(py).flags & NODE_FLAG_ENTRYPOINT != 0).then_some(idx));
        Ok(bfs(&outputs.builder, seeds, Direction::Forward, skip_flags)
            .into_iter()
            .map(|i| outputs.builder.nodes[i].clone_ref(py))
            .collect())
    }

    /// Return ``(module_node, [decls inside the block])`` for every
    /// file with a top-level ``if __name__ == "__main__":`` block.
    ///
    /// The decls list contains the file's class / function / variable
    /// / import nodes whose source position falls inside the block's
    /// range — same shape ``MainBlockPlugin``'s libcst path computes
    /// from the visitor's payload.
    fn find_main_blocks(&self, py: Python<'_>) -> PyResult<Vec<MainBlock>> {
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
            let mut decls: Vec<Py<NativeNode>> = Vec::new();
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
    #[pyo3(signature = (decorator_module, decorator_names, *, path_regex = None))]
    fn find_decorated_decls(
        &self,
        py: Python<'_>,
        decorator_module: &str,
        decorator_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<Py<NativeNode>>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_decorated_decls"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let names: HashSet<&str> = decorator_names.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = decorator_names.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let project_files = &outputs.project_files;
        // Text prefilter inside the parallel walk mirrors the LSP's
        // find_references — skip files that don't even mention any
        // of the decorator names. The rayon::scope here parallelizes
        // the per-file parse + walk across project files; we release
        // the GIL for the duration and re-acquire only to materialize
        // Py<NativeNode> handles from the collected indices.
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let names_ref = &names;
        let needles_ref: &[&str] = &needle_strs;
        let indices: Vec<usize> = py.allow_threads(move || {
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
                let mut local = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Stmt::FunctionDef(func) = stmt else {
                        continue;
                    };
                    if !decorators_match_imports(&func.decorator_list, &imports, names_ref) {
                        continue;
                    }
                    let key = (file, range_key(func.name.range()));
                    if let Some(&idx) = decl_by_name_range.get(&key) {
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
    #[pyo3(signature = (module, ctor_names, *, path_regex = None))]
    fn find_instance_constructions(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
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
        let out: Vec<(Py<NativeNode>, String)> = pairs
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
    #[pyo3(signature = (decorator_attrs, *, path_regex = None))]
    fn find_handler_decorators(
        &self,
        py: Python<'_>,
        decorator_attrs: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(String, Py<NativeNode>)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_handler_decorators"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let attrs: HashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let needle_strs: Vec<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let attrs_ref = &attrs;
        let needles_ref: &[&str] = &needle_strs;
        let pairs: Vec<(String, usize)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_any_identifier(&source, needles_ref) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let mut local: Vec<(String, usize)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Stmt::FunctionDef(func) = stmt else {
                        continue;
                    };
                    let mut seen_owners: HashSet<String> = HashSet::new();
                    for dec in &func.decorator_list {
                        let mut expr = &dec.expression;
                        if let Expr::Call(call) = expr {
                            expr = &call.func;
                        }
                        let Expr::Attribute(attr) = expr else {
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
                            local.push((owner_name, idx));
                        }
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(name, idx)| (name, outputs.builder.nodes[idx].clone_ref(py)))
            .collect())
    }

    /// Like ``find_handler_decorators`` but matches the two-level form
    /// ``@<owner>.<via_attr>.<attr>(...)`` (e.g. ``@bot.tree.command()``
    /// for discord.py's slash commands). Returns the same
    /// ``[(owner_name, function_node)]`` shape, where ``owner_name`` is
    /// the leftmost ``Name`` in the decorator chain.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (via_attr, decorator_attrs, *, path_regex = None))]
    fn find_handler_decorators_via(
        &self,
        py: Python<'_>,
        via_attr: &str,
        decorator_attrs: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(String, Py<NativeNode>)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_handler_decorators_via"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let attrs: HashSet<&str> = decorator_attrs.iter().map(String::as_str).collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let attrs_ref = &attrs;
        let pairs: Vec<(String, usize)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                // ``via_attr`` is the more selective needle ("tree" for
                // discord.py slash commands) than the attr set.
                let source = source_text(db, file);
                if !_contains_identifier(&source, via_attr) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let mut local: Vec<(String, usize)> = Vec::new();
                for stmt in &parsed.syntax().body {
                    let Stmt::FunctionDef(func) = stmt else {
                        continue;
                    };
                    let mut seen_owners: HashSet<String> = HashSet::new();
                    for dec in &func.decorator_list {
                        let mut expr = &dec.expression;
                        if let Expr::Call(call) = expr {
                            expr = &call.func;
                        }
                        let Expr::Attribute(outer) = expr else {
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
                            local.push((owner_name, idx));
                        }
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(name, idx)| (name, outputs.builder.nodes[idx].clone_ref(py)))
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
    #[pyo3(signature = (attr, arg_index, *, path_regex = None))]
    fn find_calls_on_attr(
        &self,
        py: Python<'_>,
        attr: &str,
        arg_index: usize,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_on_attr"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let decl_by_name_range = &outputs.decl_by_name_range;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let pairs: Vec<(usize, String)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                let source = source_text(db, file);
                if !_contains_identifier(&source, attr) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let mut local: Vec<(usize, String)> = Vec::new();
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
                        results: Vec::new(),
                    };
                    finder.visit_stmt(stmt);
                    for arg in finder.results {
                        local.push((owner_idx, arg));
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(idx, arg)| (outputs.builder.nodes[idx].clone_ref(py), arg))
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
    fn find_factory_decls(
        &self,
        py: Python<'_>,
        module: &str,
        ctor_names: Vec<String>,
    ) -> PyResult<Vec<(Py<NativeNode>, Vec<String>)>> {
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
    #[pyo3(signature = (module, name, arg_index, *, path_regex = None))]
    fn find_calls_to_imported(
        &self,
        py: Python<'_>,
        module: &str,
        name: &str,
        arg_index: usize,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_to_imported"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let allowed: HashSet<&str> = [name].into_iter().collect();
        let decl_by_name_range = &outputs.decl_by_name_range;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let allowed_ref = &allowed;
        let pairs: Vec<(usize, String)> = py.allow_threads(move || {
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
                let mut local: Vec<(usize, String)> = Vec::new();
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
                        results: Vec::new(),
                    };
                    finder.visit_stmt(stmt);
                    for arg in finder.results {
                        local.push((owner_idx, arg));
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(idx, arg)| (outputs.builder.nodes[idx].clone_ref(py), arg))
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
    #[pyo3(signature = (owner, attr, arg_index, *, required_positional = None, path_regex = None))]
    fn find_calls_on_var(
        &self,
        py: Python<'_>,
        owner: &str,
        attr: &str,
        arg_index: usize,
        required_positional: Option<usize>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs
            .as_ref()
            .ok_or_else(|| not_materialized("find_calls_on_var"))?;
        let path_re = _compile_path_regex(path_regex)?;
        let decl_by_name_range = &outputs.decl_by_name_range;
        let module_nodes_by_file = &outputs.module_nodes_by_file;
        let project_files: &[File] = &outputs.project_files;
        let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&self.db);
        let path_re_ref = &path_re;
        let pairs: Vec<(usize, String)> = py.allow_threads(move || {
            par_scan_files(db_handle, project_files, path_re_ref, |db, file| {
                // ``owner`` is typically the more selective needle
                // (e.g. ``mocker`` / ``monkeypatch`` show up in far
                // fewer files than common method names like ``patch``).
                let source = source_text(db, file);
                if !_contains_identifier(&source, owner) {
                    return Vec::new();
                }
                let parsed = parsed_module(db, file).load(db);
                let mut local: Vec<(usize, String)> = Vec::new();
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
                        results: Vec::new(),
                    };
                    finder.visit_stmt(stmt);
                    for arg in finder.results {
                        local.push((owner_idx, arg));
                    }
                }
                local
            })
        });
        Ok(pairs
            .into_iter()
            .map(|(idx, arg)| (outputs.builder.nodes[idx].clone_ref(py), arg))
            .collect())
    }

    /// Walks each class's `DefinitionKind::Class` body for an
    /// `Stmt::FunctionDef` whose name matches. ty's `parsed_module` is
    /// Salsa-cached, so this is just a body scan per class.
    fn find_classes_defining_method(
        &self,
        py: Python<'_>,
        method_name: &str,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn find_subclasses_of(
        &self,
        py: Python<'_>,
        class_node: &NativeNode,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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

    // ----- Stage 2: ty-backed semantic queries ---------------------------

    /// Decls decorated by ``@<decorator_fqn>`` or ``@<decorator_fqn>(...)``.
    ///
    /// Resolves through the file's local imports — aliased / dotted /
    /// module-prefixed forms all match. ``decorator_fqn`` is the
    /// upstream callable's absolute fqn (``celery.shared_task``,
    /// ``pytest.fixture``). For instance-method decorators
    /// (``@app.route(...)`` where ``app`` is a ``flask.Flask``) use
    /// :meth:`find_decorations_on` instead.
    #[pyo3(signature = (decorator_fqn, *, path_regex = None))]
    fn find_decorated(
        &self,
        py: Python<'_>,
        decorator_fqn: &str,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<Py<NativeNode>>> {
        let Some((module, name)) = decorator_fqn.rsplit_once('.') else {
            return Err(PyValueError::new_err(format!(
                "expected a dotted decorator fqn (e.g. 'pytest.fixture'), got {decorator_fqn:?}"
            )));
        };
        self.find_decorated_decls(py, module, vec![name.to_string()], path_regex)
    }

    /// Module-level variables assigned an instance of ``class_fqn``.
    ///
    /// e.g. ``find_constructions("flask.Flask")`` → every ``app =
    /// Flask(...)`` variable node. ``include_subclasses=True`` also
    /// matches direct constructions of any class that subclasses
    /// ``class_fqn`` (works for both project subclasses and external
    /// ones via ty's type hierarchy).
    #[pyo3(signature = (class_fqn, *, include_subclasses = false, path_regex = None))]
    fn find_constructions(
        &self,
        py: Python<'_>,
        class_fqn: &str,
        include_subclasses: bool,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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

    /// Decls decorated by ``@<instance>.<method>(...)`` for ``method``
    /// in ``method_names``, where ``<instance>`` resolves to the given
    /// decl in the same file.
    ///
    /// Cross-file owners (where ``app = imported_factory()`` and
    /// ``@app.route`` is in a different file) aren't matched — same
    /// limitation the rust dispatch-app path has today.
    #[pyo3(signature = (instance, method_names, *, path_regex = None))]
    fn find_decorations_on(
        &self,
        py: Python<'_>,
        instance: &NativeNode,
        method_names: Vec<String>,
        path_regex: Option<&str>,
    ) -> PyResult<Vec<Py<NativeNode>>> {
        let instance_simple = instance.fqname.rsplit('.').next().unwrap_or("").to_string();
        let handlers = self.find_handler_decorators(py, method_names, path_regex)?;
        let mut out = Vec::new();
        for (owner_name, handler) in handlers {
            if owner_name != instance_simple {
                continue;
            }
            if handler.borrow(py).path != instance.path {
                continue;
            }
            out.push(handler);
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
    #[pyo3(signature = (base_fqn, *, transitive = true))]
    fn find_subclasses(
        &self,
        py: Python<'_>,
        base_fqn: &str,
        transitive: bool,
    ) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn find_comment_patterns(
        &self,
        py: Python<'_>,
        pattern: &str,
    ) -> PyResult<Vec<(Py<NativeNode>, String)>> {
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

    // ----- Read-only accessors -------------------------------------------

    /// Live nodes in the in-progress graph. Cheap, no copy.
    fn nodes(&self, py: Python<'_>) -> PyResult<Vec<Py<NativeNode>>> {
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
    fn edges(&self) -> PyResult<Vec<(usize, usize, u32)>> {
        let outputs = self.outputs.borrow();
        let outputs = outputs.as_ref().ok_or_else(|| not_materialized("edges"))?;
        Ok(outputs.builder.edges.clone())
    }
}

impl ProjectContext {
    /// Compile `pattern` once and reuse on subsequent calls. The cache
    /// is bounded by the (small) number of distinct patterns a plugin
    /// run uses, so unbounded growth isn't a concern in practice.
    fn compile_regex(&self, pattern: &str) -> PyResult<regex::Regex> {
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

// ---------------------------------------------------------------------------
// ProjectContext support helpers
// ---------------------------------------------------------------------------

fn make_db(
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

fn not_materialized(op: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ProjectContext.{op}() called outside an active materialize() — \
         did you call it from a plugin's run() method?"
    ))
}

/// Project a `NativeNode` onto the `NodeKey` used for intern-table
/// identity. Clones the two `String` fields (`fqname`, `path`); the
/// rest are `Copy`.
fn node_key_of(node: &NativeNode) -> NodeKey {
    NodeKey {
        fqname: node.fqname.clone(),
        kind: node.kind,
        path: node.path.clone(),
        start_line: node.start_line,
        start_column: node.start_column,
        end_line: node.end_line,
        end_column: node.end_column,
    }
}

/// Direction passed to :func:`bfs` — forward follows ``src -> dst``
/// edges; reverse follows the inverse for ``ancestors``-style queries.
#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Reverse,
}

/// Generic BFS over the build graph. ``skip_flags`` filters edges whose
/// flag mask intersects (any bit) — pass ``0`` to follow every edge.
/// Returns the set of reached node indices including ``start``.
fn bfs(
    builder: &GraphBuilder,
    seeds: impl IntoIterator<Item = usize>,
    direction: Direction,
    skip_flags: u32,
) -> HashSet<usize> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = seeds.into_iter().collect();
    while let Some(i) = stack.pop() {
        if !visited.insert(i) {
            continue;
        }
        let adj = match direction {
            Direction::Forward => &builder.forward_adj[i],
            Direction::Reverse => &builder.reverse_adj[i],
        };
        for &(next, flags) in adj {
            if flags & skip_flags != 0 {
                continue;
            }
            if !visited.contains(&next) {
                stack.push(next);
            }
        }
    }
    visited
}

fn apply_graph_op(ctx: &Py<ProjectContext>, py: Python<'_>, op: &Bound<'_, PyAny>) -> PyResult<()> {
    let this = ctx.borrow(py);
    let mut outputs = this.outputs.borrow_mut();
    let outputs = outputs
        .as_mut()
        .ok_or_else(|| not_materialized("apply_graph_op"))?;

    if let Ok(add_edge) = op.extract::<PyRef<AddEdge>>() {
        let src_idx = lookup_idx(&outputs.builder, &add_edge.src.borrow(py), "src")?;
        let dst_idx = lookup_idx(&outputs.builder, &add_edge.dst.borrow(py), "dst")?;
        outputs.builder.add_edge(src_idx, dst_idx, add_edge.flags);
        return Ok(());
    }
    if let Ok(add_ep) = op.extract::<PyRef<AddEntrypoint>>() {
        let decl = add_ep.decl.borrow(py);
        let decl_idx = lookup_idx(&outputs.builder, &decl, "decl")?;
        let marker_fqname = format!("{}:{}", add_ep.marker, decl.fqname);
        let path = decl.path.clone();
        drop(decl);
        let marker_idx = outputs.builder.intern_node(
            py,
            NativeNode {
                fqname: marker_fqname,
                kind: "synthetic",
                path,
                start_line: 0,
                start_column: 0,
                end_line: 0,
                end_column: 0,
                flags: NODE_FLAG_ENTRYPOINT,
                imports: None,
            },
        )?;
        outputs.builder.add_edge(marker_idx, decl_idx, 0);
        return Ok(());
    }
    if let Ok(add_node) = op.extract::<PyRef<AddNode>>() {
        let node_idx = outputs.builder.intern_node(
            py,
            NativeNode {
                fqname: add_node.fqname.clone(),
                kind: add_node.kind,
                path: add_node.path.clone(),
                start_line: 0,
                start_column: 0,
                end_line: 0,
                end_column: 0,
                flags: add_node.flags,
                imports: None,
            },
        )?;
        for src in &add_node.edges_from {
            let src_idx = lookup_idx(&outputs.builder, &src.borrow(py), "edges_from")?;
            outputs.builder.add_edge(src_idx, node_idx, 0);
        }
        for dst in &add_node.edges_to {
            let dst_idx = lookup_idx(&outputs.builder, &dst.borrow(py), "edges_to")?;
            outputs.builder.add_edge(node_idx, dst_idx, 0);
        }
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "expected a GraphOp (AddEdge / AddEntrypoint / AddNode), got {:?}",
        op.get_type().name()?,
    )))
}

/// Resolve a `NativeNode` reference to its builder-side index for an
/// edge endpoint. Surfaces a precise `ValueError` (with `side`) when
/// the node was never interned in this context.
fn lookup_idx(builder: &GraphBuilder, node: &NativeNode, side: &str) -> PyResult<usize> {
    builder
        .node_index
        .get(&node_key_of(node))
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "add_edge: {side} node {:?} is not interned in this ProjectContext",
                node.fqname
            ))
        })
}

/// Map a plugin-supplied `kind` string to one of the stable `&'static
/// str` kinds NativeNode carries.
fn static_kind_str(kind: &str) -> PyResult<&'static str> {
    Ok(match kind {
        "synthetic" => "synthetic",
        "module" => "module",
        "import" => "import",
        "function" => "function",
        "class" => "class",
        "variable" => "variable",
        "type_alias" => "type_alias",
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown node kind {other:?} — expected one of synthetic, module, import, \
                 function, class, variable, type_alias"
            )))
        }
    })
}

fn is_dunder_name(fqname: &str) -> bool {
    let name = fqname.rsplit('.').next().unwrap_or("");
    name.len() > 4 && name.starts_with("__") && name.ends_with("__")
}

fn class_body_defines_method(class_def: &StmtClassDef, method_name: &str) -> bool {
    class_def.body.iter().any(|stmt| match stmt {
        Stmt::FunctionDef(f) => f.name.as_str() == method_name,
        _ => false,
    })
}

/// Return ``true`` if any decorator in ``decorators`` matches
/// ``@<module>.<name>`` (literal module name) or ``@<name>`` for
/// any ``name`` in ``names``. Trailing ``(...)`` is unwrapped before
/// the pattern check.
/// Resolves decorator references through the file's local imports map
/// (built by :func:`collect_module_imports_local`) — mirrors the libcst
/// helpers' ``matched_attr_call`` shape. Recognized forms:
///
/// * ``@<name>(...)`` / ``@<name>`` where ``imports[name]`` is in ``names``
///   (covers ``from module import name`` and aliased variants).
/// * ``@<alias>.<attr>(...)`` / ``@<alias>.<attr>`` where
///   ``imports[alias] == "<module>"`` and ``attr`` is in ``names``
///   (covers ``import module`` and ``import module as alias``).
fn decorators_match_imports(
    decorators: &[ruff_python_ast::Decorator],
    imports: &HashMap<String, String>,
    names: &HashSet<&str>,
) -> bool {
    for dec in decorators {
        let mut expr = &dec.expression;
        if let Expr::Call(call) = expr {
            expr = &call.func;
        }
        match expr {
            Expr::Name(n) => {
                if let Some(target) = imports.get(n.id.as_str()) {
                    if names.contains(target.as_str()) {
                        return true;
                    }
                }
            }
            Expr::Attribute(attr) => {
                if let Expr::Name(prefix) = attr.value.as_ref() {
                    if imports.get(prefix.id.as_str()).map(String::as_str) == Some("<module>")
                        && names.contains(attr.attr.as_str())
                    {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn iter_top_level_classes(parsed: &ParsedModuleRef) -> impl Iterator<Item = &StmtClassDef> {
    parsed.syntax().body.iter().filter_map(|stmt| match stmt {
        Stmt::ClassDef(cls) => Some(cls),
        _ => None,
    })
}

/// Locate a class's File + name TextRange from its NativeNode positions.
///
/// We don't store ty `Definition<'db>` references across plugin calls
/// (the `'db` lifetime is tied to the active borrow), so this re-walks
/// Locate a class's File + name TextRange from its NativeNode positions.
///
/// We don't store ty `Definition<'db>` references across plugin calls
/// (the `'db` lifetime is tied to the active borrow), so this re-walks
/// the matching file's top-level classes for one whose name lands on
/// the node's start line. Match-by-line (not line+column) because
/// Function / Class / TypeAlias node columns are snapped to the line's
/// indent — not the bound name's column — to align with libcst, and
/// two top-level classes can't share a source line.
fn locate_class_def(
    db: &ProjectDatabase,
    path_to_file: &HashMap<String, File>,
    path: &str,
    class_node: &NativeNode,
) -> Option<(File, TextRange)> {
    let &file = path_to_file.get(path)?;
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    for cls in iter_top_level_classes(&parsed) {
        let name_range = cls.name.range();
        let (sl, _, _, _) = position(&line_index, &source, name_range);
        if sl == class_node.start_line {
            return Some((file, name_range));
        }
    }
    None
}

/// Find a top-level `StmtClassDef` whose name range equals `selection_range`.
/// Return the top-level ``StmtClassDef`` (if any) that uses
/// ``ref_range`` as one of its direct base-list arguments.
///
/// Used by the find_references-based subclass walk: each
/// :func:`ty_ide::find_references` hit gives us a ``(File, range)``
/// for a use of the seed class; if that range falls inside a top-level
/// class's ``arguments.args`` list, the surrounding class is a direct
/// subclass.
///
/// Note: this is a syntactic match (range containment), so a use of
/// ``TestCase`` inside a parameterized generic base
/// (``class X(SomeGeneric[TestCase]):``) will falsely flag ``X`` as a
/// subclass of ``TestCase``. ty's :func:`type_hierarchy_subtypes`
/// avoids that via real type inference; we trade that accuracy for
/// the prefilter+rayon scan ty_ide's find_references provides.
fn class_base_arg_owner(parsed: &ParsedModuleRef, ref_range: TextRange) -> Option<&StmtClassDef> {
    for stmt in &parsed.syntax().body {
        let Stmt::ClassDef(class_def) = stmt else {
            continue;
        };
        let Some(arguments) = &class_def.arguments else {
            continue;
        };
        for arg in &arguments.args {
            if arg.range().contains_range(ref_range) {
                return Some(class_def);
            }
        }
    }
    None
}

/// If ``ref_range`` covers the "original name" of an aliased
/// ``from M import Name as Alias`` (or ``import M as Alias``), return
/// the alias's local-binding name range so the BFS can follow the
/// re-binding to its uses.
///
/// ty_ide's :func:`find_references` is designed for IDE rename
/// semantics: it does NOT follow aliased imports across the
/// re-binding (renaming ``TestCase`` should not rename uses of
/// ``TC`` from ``from unittest import TestCase as TC``). For our
/// subclass walk we DO want the alias's uses — they're the
/// subclasses we'd otherwise miss — so we seed the BFS with the
/// alias's local name range and call find_references on that.
fn aliased_import_local_name_range(
    parsed: &ParsedModuleRef,
    ref_range: TextRange,
) -> Option<TextRange> {
    for stmt in &parsed.syntax().body {
        let aliases: &[ruff_python_ast::Alias] = match stmt {
            Stmt::ImportFrom(import_from) => &import_from.names,
            Stmt::Import(import) => &import.names,
            _ => continue,
        };
        for alias in aliases {
            let Some(asname) = &alias.asname else {
                continue;
            };
            if alias.name.range() == ref_range {
                return Some(asname.range());
            }
        }
    }
    None
}

/// Walk transitive subclasses of the seed class via
/// :func:`ty_ide::find_references`, which carries an
/// identifier-aware text prefilter and per-file rayon parallelism for
/// free. Each find_references hit is filtered to "syntactically in a
/// class base list"; matched classes seed the next round when
/// ``transitive`` is set.
///
/// Returns ``Vec<usize>`` of indices into
/// ``BuildOutputs::builder.nodes`` (only project classes — typeshed
/// / external matches are dropped because they don't have a
/// :attr:`class_by_selection` entry).
fn find_subclass_indices_via_refs(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    seed_file: File,
    seed_name_range: TextRange,
    transitive: bool,
) -> Vec<usize> {
    let mut out_idx: HashSet<usize> = HashSet::new();
    let mut visited_seeds: HashSet<(File, (u32, u32))> = HashSet::new();
    let mut queue: Vec<(File, TextRange)> = vec![(seed_file, seed_name_range)];

    while let Some((cur_file, cur_range)) = queue.pop() {
        if !visited_seeds.insert((cur_file, range_key(cur_range))) {
            continue;
        }
        // Cursor offset inside the identifier (start can sit on the
        // token boundary and ty_ide's tokens.at_offset has edge-cases
        // there). Anywhere inside the identifier resolves the same
        // goto target.
        let offset =
            cur_range.start() + ruff_text_size::TextSize::from(u32::from(cur_range.len()) / 2);
        // Pass include_declaration=true: skip-declaration mode filters
        // out re-binding declarations like `from M import Name as Alias`
        // alongside the seed itself. We need that import-line entry to
        // detect the alias and recurse, so we take the full list and
        // drop the seed in class_base_arg_owner / via the visited_seeds
        // dedup.
        let Some(refs) = ty_ide::find_references(db, cur_file, offset, true) else {
            continue;
        };
        for r in refs {
            let r_file = r.file();
            let r_range = r.range();
            let parsed = parsed_module(db, r_file).load(db);
            if let Some(class_def) = class_base_arg_owner(&parsed, r_range) {
                let class_name_range = class_def.name.range();
                let key = (r_file, range_key(class_name_range));
                let Some(&idx) = outputs.class_by_selection.get(&key) else {
                    continue;
                };
                if out_idx.insert(idx) && transitive {
                    queue.push((r_file, class_name_range));
                }
            } else if let Some(alias_range) = aliased_import_local_name_range(&parsed, r_range) {
                queue.push((r_file, alias_range));
            }
        }
    }

    out_idx.into_iter().collect()
}

/// Find a top-level ``StmtClassDef`` by its bound name.
fn class_def_named<'a>(parsed: &'a ParsedModuleRef, name: &str) -> Option<&'a StmtClassDef> {
    iter_top_level_classes(parsed).find(|cls| cls.name.as_str() == name)
}

/// Resolve a dotted class fqn to ``(File, name_range)`` so the caller
/// can fetch the corresponding ``Type`` via ``class_def_at`` +
/// ``inferred_type``. Handles both project classes (looked up via
/// ``decl_by_fqname``) and external classes (ty's ``resolve_module``
/// + AST scan by name in the resolved module).
fn locate_class_seed(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    py: Python<'_>,
    fqn: &str,
) -> Option<(File, TextRange)> {
    // Project class: cheap path through the existing indices.
    if let Some(idxs) = outputs.decl_by_fqname.get(fqn) {
        for &idx in idxs {
            let node = outputs.builder.nodes[idx].borrow(py);
            if node.kind != "class" {
                continue;
            }
            let path = node.path.clone();
            if let Some(seed) = locate_class_def(db, &outputs.path_to_file, &path, &node) {
                return Some(seed);
            }
        }
    }
    // External class: resolve the module via ty, then follow any
    // re-export chains until we find the actual class def.
    let (module_str, class_name) = fqn.rsplit_once('.')?;
    let module_name = ModuleName::new(module_str)?;
    let anchor = *outputs.project_files.first()?;
    let module = resolve_module(db, anchor, &module_name)?;
    let module_file = module.file(db)?;
    let mut visited: HashSet<File> = HashSet::new();
    follow_class_through_module(db, module_file, class_name, &mut visited)
}

/// Walk through a module looking for ``class_name``. Handles three
/// re-export shapes: a direct ``class class_name: ...`` definition,
/// an explicit ``from .other import class_name [as class_name]``
/// re-export, and ``from .other import *`` (which exposes everything
/// the source module defines at the top level). Bounded by a
/// visited-file set so cycles can't loop forever.
fn follow_class_through_module(
    db: &ProjectDatabase,
    start_file: File,
    class_name: &str,
    visited: &mut HashSet<File>,
) -> Option<(File, TextRange)> {
    if !visited.insert(start_file) {
        return None;
    }
    let parsed = parsed_module(db, start_file).load(db);
    if let Some(cls) = class_def_named(&parsed, class_name) {
        return Some((start_file, cls.name.range()));
    }
    for stmt in &parsed.syntax().body {
        let Stmt::ImportFrom(im) = stmt else {
            continue;
        };
        let Ok(module_name) = ModuleName::from_import_statement(db, start_file, im) else {
            continue;
        };
        let Some(resolved) = resolve_module(db, start_file, &module_name) else {
            continue;
        };
        let Some(source_file) = resolved.file(db) else {
            continue;
        };
        for alias in &im.names {
            let imported_name = alias.name.as_str();
            if imported_name == "*" {
                if let Some(seed) =
                    follow_class_through_module(db, source_file, class_name, visited)
                {
                    return Some(seed);
                }
                continue;
            }
            let local_name = alias
                .asname
                .as_ref()
                .map(|n| n.as_str())
                .unwrap_or(imported_name);
            if local_name != class_name {
                continue;
            }
            if let Some(seed) = follow_class_through_module(db, source_file, imported_name, visited)
            {
                return Some(seed);
            }
        }
    }
    None
}

/// Locate the top-level ``if __name__ == "__main__":`` block in a
/// parsed module and return its source range. Both orderings
/// (``__name__ == "__main__"`` and ``"__main__" == __name__``) are
/// recognized; the ``elif`` / ``else`` branches are not matched.
fn find_main_block_range(parsed: &ParsedModuleRef) -> Option<TextRange> {
    for stmt in &parsed.syntax().body {
        let Stmt::If(if_stmt) = stmt else {
            continue;
        };
        if is_name_eq_main(&if_stmt.test) {
            return Some(if_stmt.range);
        }
    }
    None
}

/// Match ``__name__ == "__main__"`` or its reverse.
fn is_name_eq_main(expr: &Expr) -> bool {
    let Expr::Compare(cmp) = expr else {
        return false;
    };
    if cmp.ops.len() != 1 {
        return false;
    }
    if !matches!(cmp.ops[0], ruff_python_ast::CmpOp::Eq) {
        return false;
    }
    let comparators = &cmp.comparators;
    if comparators.len() != 1 {
        return false;
    }
    let left = &*cmp.left;
    let right = &comparators[0];
    (is_name(left, "__name__") && is_string_literal(right, "__main__"))
        || (is_string_literal(left, "__main__") && is_name(right, "__name__"))
}

fn is_name(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Name(n) if n.id.as_str() == value)
}

fn is_string_literal(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::StringLiteral(s) if s.value.to_str() == value)
}

/// Build the file-local imports map ``{local_name: target}`` for
/// names imported from ``module``. ``target`` is the upstream
/// constructor / decl name (e.g. ``"Flask"`` when bound via
/// ``from flask import Flask``) or the sentinel ``"<module>"``
/// when bound via ``import flask`` / ``import flask as f``.
///
/// Only entries whose target is in ``allowed`` survive — keeps the
/// map small and lets call-site matchers do a cheap second check.
fn collect_module_imports_local(
    parsed: &ParsedModuleRef,
    module: &str,
    allowed: &HashSet<&str>,
) -> HashMap<String, String> {
    // Submodule binding: ``from <parent> import <last_seg>`` makes
    // ``last_seg`` a local alias for the queried module (e.g.
    // ``from unittest import mock`` for module ``unittest.mock``).
    let parent_last = module.rsplit_once('.');
    let mut out = HashMap::new();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                let Some(mod_name) = &im.module else { continue };
                if mod_name.as_str() == module {
                    for alias in &im.names {
                        let target = alias.name.as_str();
                        if !allowed.contains(target) {
                            continue;
                        }
                        let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                        out.insert(local.to_string(), target.to_string());
                    }
                } else if let Some((parent, last)) = parent_last {
                    if mod_name.as_str() == parent {
                        for alias in &im.names {
                            if alias.name.as_str() != last {
                                continue;
                            }
                            let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(last);
                            out.insert(local.to_string(), "<module>".to_string());
                        }
                    }
                }
            }
            Stmt::Import(im) => {
                for alias in &im.names {
                    if alias.name.as_str() != module {
                        continue;
                    }
                    let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(module);
                    out.insert(local.to_string(), "<module>".to_string());
                }
            }
            _ => {}
        }
    }
    out
}

/// Match an expression against a callable-from-module pattern. Three
/// shapes resolve through ``imports`` to the configured ``module``:
///
/// * ``<imported_local>(...)`` — local was bound via ``from <module>
///   import <name>``;
/// * ``<module_alias>.<allowed_name>(...)`` — alias bound via
///   ``import <module> [as <alias>]``;
/// * ``<m>.<n>.…<allowed_name>(...)`` — literal dotted access of a
///   multi-segment ``<module>`` (e.g. ``import unittest.mock;
///   unittest.mock.patch(...)``).
///
/// Returns the matched upstream name (``"Flask"``) on hit, else ``None``.
fn matched_call_target(
    call: &ruff_python_ast::ExprCall,
    imports: &HashMap<String, String>,
    module: &str,
    allowed: &HashSet<&str>,
) -> Option<String> {
    match call.func.as_ref() {
        Expr::Name(name) => {
            let target = imports.get(name.id.as_str())?;
            allowed.contains(target.as_str()).then(|| target.clone())
        }
        Expr::Attribute(attr) => {
            let attr_name = attr.attr.as_str();
            if !allowed.contains(attr_name) {
                return None;
            }
            match attr.value.as_ref() {
                Expr::Name(prefix) => (imports.get(prefix.id.as_str()).map(String::as_str)
                    == Some("<module>"))
                .then(|| attr_name.to_string()),
                _ => {
                    let (root, segs) = collapse_attribute_chain(attr.value.as_ref())?;
                    let mut dotted = String::with_capacity(module.len());
                    dotted.push_str(root.id.as_str());
                    for seg in &segs {
                        dotted.push('.');
                        dotted.push_str(seg);
                    }
                    (dotted == module).then(|| attr_name.to_string())
                }
            }
        }
        _ => None,
    }
}

/// Top-level ``X = <expr>`` / ``X: T = <expr>`` where the target is a
/// bare ``Name``. Returns ``(name_range, &expr)`` so the caller can
/// look up ``X``'s node and inspect the RHS.
fn top_level_assign_to_name(stmt: &Stmt) -> Option<(TextRange, &Expr)> {
    match stmt {
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return None;
            }
            let Expr::Name(name) = &assign.targets[0] else {
                return None;
            };
            Some((name.range, assign.value.as_ref()))
        }
        Stmt::AnnAssign(assign) => {
            let value = assign.value.as_deref()?;
            let Expr::Name(name) = assign.target.as_ref() else {
                return None;
            };
            Some((name.range, value))
        }
        _ => None,
    }
}

/// Recursive visitor: walk a function / class body collecting the set
/// of constructor names called anywhere inside it.
struct FactoryCallFinder<'a> {
    imports: &'a HashMap<String, String>,
    module: &'a str,
    allowed: &'a HashSet<&'a str>,
    kinds: HashSet<String>,
}

impl<'ast, 'a> Visitor<'ast> for FactoryCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Some(name) = matched_call_target(call, self.imports, self.module, self.allowed) {
                self.kinds.insert(name);
            }
        }
        walk_expr(self, expr);
    }
}

/// Owner-resolution helper shared by the call-finder queries.
/// Top-level ``FunctionDef`` / ``ClassDef`` own calls inside their
/// subtree (decorators included via the walk); other top-level stmts
/// attribute their calls to the module node.
/// Look up the owning decl index for a top-level statement. Takes
/// the two hashmaps directly (rather than ``&BuildOutputs``) because
/// ``BuildOutputs`` carries ``Vec<Py<NativeNode>>`` and is therefore
/// ``!Sync`` — these maps are ``Sync`` on their own, which lets the
/// callers borrow them across rayon thread boundaries.
fn owner_idx_for_stmt_with(
    decl_by_name_range: &HashMap<(File, (u32, u32)), usize>,
    module_nodes_by_file: &HashMap<File, usize>,
    file: File,
    stmt: &Stmt,
) -> Option<usize> {
    let module_idx = module_nodes_by_file.get(&file).copied();
    let name_range = match stmt {
        Stmt::FunctionDef(f) => f.name.range(),
        Stmt::ClassDef(c) => c.name.range(),
        _ => return module_idx,
    };
    decl_by_name_range
        .get(&(file, range_key(name_range)))
        .copied()
        .or(module_idx)
}

/// Extract the string-literal value of a call's ``args[arg_index]``
/// positional argument. ``None`` when out of range or not a single
/// string literal — concatenated / f-string / b-string forms don't
/// project to a static fqname and are deliberately rejected.
fn nth_positional_string(call: &ruff_python_ast::ExprCall, arg_index: usize) -> Option<String> {
    match call.arguments.args.get(arg_index)? {
        Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
        _ => None,
    }
}

/// Match ``<owner>.<attr>(...)`` where ``<owner>`` is a bare ``Name``
/// equal to the given owner string and ``<attr>`` matches. No import
/// resolution — meant for pytest fixture conventions (``mocker``,
/// ``monkeypatch``) whose names come from function parameters.
fn call_callee_matches_var(
    call: &ruff_python_ast::ExprCall,
    owner: &str,
    attr: &str,
    required_positional: Option<usize>,
) -> bool {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    if attribute.attr.as_str() != attr {
        return false;
    }
    let Expr::Name(name) = attribute.value.as_ref() else {
        return false;
    };
    if name.id.as_str() != owner {
        return false;
    }
    if let Some(expected) = required_positional {
        if call.arguments.args.len() != expected {
            return false;
        }
    }
    true
}

/// Visit every Call expression in a subtree, push the string-literal
/// arg at ``arg_index`` whenever ``predicate(call)`` returns ``true``.
/// Backs both call-finder queries.
struct StringArgCallFinder<F>
where
    F: FnMut(&ruff_python_ast::ExprCall) -> bool,
{
    predicate: F,
    arg_index: usize,
    results: Vec<String>,
}

impl<'ast, F> Visitor<'ast> for StringArgCallFinder<F>
where
    F: FnMut(&ruff_python_ast::ExprCall) -> bool,
{
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if (self.predicate)(call) {
                if let Some(value) = nth_positional_string(call, self.arg_index) {
                    self.results.push(value);
                }
            }
        }
        walk_expr(self, expr);
    }
}

/// Visit every Call expression in a subtree, capture string-literal
/// args at ``arg_index`` for calls whose callee is ``<expr>.<attr>(...)``
/// regardless of receiver shape. The captured arg can be a single
/// string literal **or** a list/tuple of string literals (the latter
/// produces multiple results for one call). Backs ``find_calls_on_attr``.
struct AttrCallFinder<'a> {
    attr: &'a str,
    arg_index: usize,
    results: Vec<String>,
}

impl<'ast, 'a> Visitor<'ast> for AttrCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Expr::Attribute(attribute) = call.func.as_ref() {
                if attribute.attr.as_str() == self.attr {
                    if let Some(arg) = call.arguments.args.get(self.arg_index) {
                        match arg {
                            Expr::StringLiteral(s) => {
                                self.results.push(s.value.to_str().to_string());
                            }
                            Expr::List(list) => {
                                for elt in &list.elts {
                                    if let Expr::StringLiteral(s) = elt {
                                        self.results.push(s.value.to_str().to_string());
                                    }
                                }
                            }
                            Expr::Tuple(tup) => {
                                for elt in &tup.elts {
                                    if let Expr::StringLiteral(s) = elt {
                                        self.results.push(s.value.to_str().to_string());
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        walk_expr(self, expr);
    }
}

/// Per-file sorted list of `(target_start_offset, node_idx)` for the
/// file's top-level decls. Sorted ascending so callers can binary-search
/// for the next decl after a comment.
///
/// Reads straight from `global_index` — every key in there already
/// carries the decl's `(start, end)` range tuple, and `ingest_decls`
/// populated it from the same `all_definitions_with_usage()` walk.
fn file_decl_sites(file: File, global_index: &DeclIndex) -> Vec<(u32, usize)> {
    let mut out: Vec<(u32, usize)> = global_index
        .iter()
        .filter(|((f, _, _), _)| *f == file)
        .map(|((_, _, (start, _)), idx)| (*start, *idx))
        .collect();
    out.sort_by_key(|(start, _)| *start);
    out
}

// ---------------------------------------------------------------------------
// Phase 1: decl enumeration via ty's SemanticIndex
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn ingest_decls(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
    alias_imports: &mut HashMap<usize, ImportSpec>,
    live_decls: &mut LiveDeclIndex,
    globals_by_name: &mut GlobalsByName,
    star_reexports: &mut StarReexports,
    class_by_selection: &mut HashMap<(File, (u32, u32)), usize>,
    decl_by_name_range: &mut HashMap<(File, (u32, u32)), usize>,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = module_fqname_for_file(db, file);
    let default_flags = file_default_flags(db, file);
    // Per-decl stub flagging context. For .pyi files we OR in
    // ``NODE_FLAG_ENTRYPOINT`` for any decl whose name has no
    // matching runtime decl in the .py twin (or has no twin at all
    // — native-extension and protobuf-style stubs). Decls that DO
    // have a runtime counterpart stay un-flagged; reachability flows
    // through them via the stub-runtime edge emitted in
    // ``emit_stub_runtime_edges`` after both files have ingested.
    let is_stub = path_str.ends_with(".pyi");
    let stub_py_twin: Option<File> = if is_stub {
        builder.peer_pyi_to_py.get(&file).copied()
    } else {
        None
    };
    let (file_pinned_by_noqa, per_line_noqa_pins) =
        scan_noqa_directives(&parsed, &source, &line_index);

    let (msl, msc, mel, mec) = position(&line_index, &source, parsed.syntax().range);
    let module_idx = builder.intern_node(
        py,
        NativeNode {
            fqname: module_fqname.clone(),
            kind: "module",
            path: path_str.clone(),
            start_line: msl,
            start_column: msc,
            end_line: mel,
            end_column: mec,
            flags: default_flags,
            imports: None,
        },
    )?;
    module_nodes.insert(file, module_idx);

    // Iterate every binding (including shadowed siblings) — Principle 3.
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);
    // ty emits one `StarImport` definition per imported name, but
    // every per-name def from the same `from X import *` statement
    // shares the `*` token's range — so the `*<src>` local name we
    // synthesize is identical across all of them. Cache by that
    // shared range to avoid re-resolving `from_module_string` and
    // re-allocating the format string N times for a star statement
    // that brings in N names.
    let mut star_local_name_cache: HashMap<TextRange, String> = HashMap::new();

    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }

        let kind = def.kind(db);
        let Some(node_kind) = decl_kind_str(kind) else {
            continue;
        };

        // Bound place must be a simple symbol — Member places (e.g.
        // `x.y = ...` style attribute defs) aren't top-level decls in
        // our model.
        let place_id = def.place(db);
        let PlaceExprRef::Symbol(symbol) = place_table.place(place_id) else {
            continue;
        };
        // The per-name binding (e.g. `f`, `g`) ty sees. We keep this
        // for the `globals_by_name` / `star_reexports` maps that
        // Phase 2's cross-module chain walk uses to chase a
        // `from A import g` through A's `from B import *`.
        let per_name = symbol.name().as_str().to_string();

        // `from X import *` — ty produces one `StarImport` definition
        // per name brought in, but the graph holds *one* node per
        // statement: the import statement itself is the local thing
        // that should be kept alive by use sites, and ty's name
        // resolution still routes each use to its specific upstream.
        // The libcst per-name synthetic alias was a workaround for
        // libcst's inability to resolve uses through star imports —
        // we don't need it here.
        //
        // Collapse by giving every per-name `StarImport` from the
        // same statement a shared `*<source>` local name; combined
        // with the shared `target_range` (the `*` token), they all
        // intern to the same `NodeKey` and therefore the same
        // `node_idx`. The per-name lookup maps (`globals_by_name`,
        // `star_reexports`) keep using each per-name as their key,
        // all pointing at this single node — so a downstream
        // `from A import g` still finds the star alias by name and
        // can chase through it to B.
        let target_range = kind.target_range(&parsed);
        let local_name = match kind {
            DefinitionKind::StarImport(k) => star_local_name_cache
                .entry(target_range)
                .or_insert_with(|| {
                    let src = from_module_string(db, file, k.import(&parsed));
                    format!("*{src}")
                })
                .clone(),
            _ => per_name.clone(),
        };

        let (mut sl, mut sc, el, ec) = position(&line_index, &source, target_range);

        // For Function / Class / TypeAlias, ty's `target_range` is the
        // bound *name* (e.g. `f` in `def f(): ...`). The libcst
        // pipeline reports the position of the introducing keyword
        // (`def` / `class` / `type`) — which is the first non-space
        // character on the same line as the name (decorators sit on
        // earlier lines and don't count). Align to libcst by snapping
        // the start column to the line's indent.
        if matches!(
            kind,
            DefinitionKind::Function(_) | DefinitionKind::Class(_) | DefinitionKind::TypeAlias(_)
        ) {
            let name_start = line_index.line_column(target_range.start(), &source);
            let line_text = &source[line_index.line_range(name_start.line, &source)];
            let indent = line_text
                .bytes()
                .take_while(|b| matches!(*b, b' ' | b'\t'))
                .count();
            sl = name_start.line.get();
            sc = indent;
        }

        let import_spec = if node_kind == "import" {
            Some(import_payload_for(kind, db, file, &parsed))
        } else {
            None
        };
        let imports = if let Some(spec) = &import_spec {
            Some(Py::new(
                py,
                Import {
                    module: spec.module.clone(),
                    decl: spec.decl.clone(),
                    star: spec.star,
                },
            )?)
        } else {
            None
        };

        // Pin imports preserved by a `# noqa[: …F401…]` (per-alias)
        // or by a file-level `# ruff: noqa` / `# flake8: noqa` so
        // reachability keeps them alive — matching ruff's own
        // semantics for explicitly-preserved unused-import lines.
        // We tag both `ENTRYPOINT` (the live-set seed) and `NOQA`
        // (so the blast-radius query can subtract noqa-only liveness).
        let mut flags: u32 = default_flags;
        if node_kind == "import" && (file_pinned_by_noqa || per_line_noqa_pins.contains(&sl)) {
            flags |= NODE_FLAGS_NOQA_PIN;
        }
        if is_stub {
            let has_runtime = stub_py_twin
                .map(|py| globals_by_name.contains_key(&(py, local_name.clone())))
                .unwrap_or(false);
            if !has_runtime {
                flags |= NODE_FLAG_ENTRYPOINT;
            }
        }

        let node_idx = builder.intern_node(
            py,
            NativeNode {
                fqname: format!("{module_fqname}.{local_name}"),
                kind: node_kind,
                path: path_str.clone(),
                start_line: sl,
                start_column: sc,
                end_line: el,
                end_column: ec,
                flags,
                imports,
            },
        )?;
        builder.add_edge(node_idx, module_idx, 0);
        global_index.insert((file, place_id, range_key(target_range)), node_idx);
        if node_kind == "class" {
            class_by_selection.insert((file, range_key(target_range)), node_idx);
        }
        // Last-write-wins matches ``live_decls``: a single (file, range)
        // can host multiple bindings (try/except rebind, star imports),
        // and the query callers care about the end-of-scope live binding.
        decl_by_name_range.insert((file, range_key(target_range)), node_idx);

        // Lookup maps key by the *per-name* (the actual bound symbol
        // each ty `StarImport` corresponds to), not the node's
        // `local_name` — the star node's collapsed `*<src>` fqname
        // would otherwise miss `from A import g` chains that probe
        // for the per-name `"g"`. For non-star imports `per_name`
        // and `local_name` are identical so this is a no-op there.
        let name_key = (file, per_name);
        if let Some(spec) = import_spec {
            // Star-reexport synthetics need their upstream tracked so
            // Phase 2 can walk a `from A import g` resolution through
            // an `A.g` star-reexport alias all the way to its real
            // def. Non-star imports clear the entry so a shadowing
            // import correctly disables chain walking.
            if spec.star {
                star_reexports.insert(name_key.clone(), spec.module.clone());
            } else {
                star_reexports.remove(&name_key);
            }
            alias_imports.insert(node_idx, spec);
        } else if node_kind != "module" {
            // Star-reexport gets killed by any later real decl; the
            // multi-binding `live_decls` is populated by the post-pass
            // below from ty's end-of-scope view, which already encodes
            // sequential-rebind / branch-bind semantics correctly.
            star_reexports.remove(&name_key);
        }
    }

    // Post-pass: populate `globals_by_name` (every binding kind) and
    // `live_decls` (real decls only) from ty's end-of-scope live
    // bindings per symbol. We can't do this inside the loop above
    // because that loop iterates `all_definitions_with_usage`, which
    // includes *every* binding — including dead ones superseded by a
    // later sequential rebind (`def f; def f` keeps only the second
    // as the live one). The end-of-scope query already encodes ty's
    // flow analysis: it preserves both branches of `if/else` and
    // `try/except` when each is a real bind, and collapses sequential
    // rebinds to the latest. Walking it once here keeps cross-module
    // `from lib import f` resolution and `emit_upstream`'s decl probe
    // multi-binding-aware without having to re-derive shadowing rules
    // at lookup time.
    for (symbol_id, bindings) in use_def_map.all_end_of_scope_symbol_bindings() {
        let PlaceExprRef::Symbol(sym) = place_table.place(ScopedPlaceId::Symbol(symbol_id)) else {
            continue;
        };
        let name = sym.name().as_str().to_string();
        let mut live: Vec<usize> = Vec::new();
        let mut live_real_decls: Vec<usize> = Vec::new();
        for binding in bindings {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != file || def.file_scope(db) != global {
                continue;
            }
            let kind = def.kind(db);
            let key = (file, def.place(db), range_key(kind.target_range(&parsed)));
            if let Some(&idx) = global_index.get(&key) {
                live.push(idx);
                // ``live_decls`` mirrors what's reachable as a decl-like
                // target in the module's namespace — an import alias is
                // still a decl from the consumer's standpoint, so it
                // belongs here too. Filtering on ``!kind.is_import()``
                // would skip ``mod -> lib.f@2:18`` when ``lib.f`` is
                // ``from a import f`` (Principle 2's parallel-upstream
                // edge from the use site to the decl in ``lib``).
                live_real_decls.push(idx);
            }
        }
        let key = (file, name);
        if !live.is_empty() {
            globals_by_name.insert(key.clone(), live);
        }
        if !live_real_decls.is_empty() {
            live_decls.insert(key, live_real_decls);
        }
    }

    Ok(())
}

fn decl_kind_str(kind: &DefinitionKind<'_>) -> Option<&'static str> {
    if kind.is_import() {
        return Some("import");
    }
    Some(match kind {
        DefinitionKind::Function(_) => "function",
        DefinitionKind::Class(_) => "class",
        DefinitionKind::TypeAlias(_) => "type_alias",
        // Walrus's bound name (PEP 572) leaks to the enclosing scope
        // — at module level that's a top-level decl. The other
        // binding-statement targets (`for`, `with ... as`, `except as`,
        // structural-`match` captures) are local loop / context /
        // pattern bindings that the libcst pipeline does *not* model
        // as top-level nodes; skip them so we don't mint phantom
        // module-scope decls.
        DefinitionKind::Assignment(_)
        | DefinitionKind::AnnotatedAssignment(_)
        | DefinitionKind::NamedExpression(_) => "variable",
        _ => return None,
    })
}

fn import_payload_for<'db>(
    kind: &DefinitionKind<'db>,
    db: &'db dyn ty_python_semantic::Db,
    file: File,
    parsed: &ParsedModuleRef,
) -> ImportSpec {
    match kind {
        DefinitionKind::Import(k) => {
            let alias = k.alias(parsed);
            ImportSpec {
                module: alias.name.id.as_str().to_string(),
                decl: None,
                star: false,
            }
        }
        DefinitionKind::ImportFrom(k) => {
            let alias = k.alias(parsed);
            ImportSpec {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(alias.name.id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::ImportFromSubmodule(k) => {
            // Bound name is one of the dotted submodule segments. The
            // module string is the parent of that segment; ``decl`` is
            // the segment itself, mirroring the libcst convention for
            // ``from a.b import c`` where ``c`` is a submodule.
            ImportSpec {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(k.module(parsed).id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::StarImport(k) => ImportSpec {
            module: from_module_string(db, file, k.import(parsed)),
            decl: None,
            star: true,
        },
        _ => unreachable!("import_payload_for called with non-import kind"),
    }
}

/// Resolved absolute module name for a `from ... import ...` clause.
///
/// Returns the empty string when ty's `from_import_statement` fails to
/// resolve (invalid syntax or too many leading dots) — downstream
/// classification can treat that as an unresolved target.
fn from_module_string(
    db: &dyn ty_python_semantic::Db,
    file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
) -> String {
    ModuleName::from_import_statement(db, file, stmt)
        .map(|n| n.as_str().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Phase 2: module-hierarchy + cross-file alias edges
// ---------------------------------------------------------------------------

fn emit_module_hierarchy(
    db: &ProjectDatabase,
    file: File,
    module_nodes: &HashMap<File, usize>,
    builder: &mut GraphBuilder,
) {
    if let Some((self_idx, parent_idx)) = parent_module_edge(db, file, module_nodes) {
        builder.add_edge(self_idx, parent_idx, 0);
    }
}

fn parent_module_edge(
    db: &ProjectDatabase,
    file: File,
    module_nodes: &HashMap<File, usize>,
) -> Option<(usize, usize)> {
    let parent_name = file_to_module(db, file)?.name(db).parent()?;
    let parent_file = resolve_module(db, file, &parent_name)?.file(db)?;
    let self_idx = *module_nodes.get(&file)?;
    let parent_idx = *module_nodes.get(&parent_file)?;
    Some((self_idx, parent_idx))
}

#[allow(clippy::too_many_arguments)]
fn emit_import_edges(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    global_index: &mut DeclIndex,
    module_nodes: &mut HashMap<File, usize>,
    globals_by_name: &GlobalsByName,
    star_reexports: &StarReexports,
    dist_lookup: &DistLookup,
) -> PyResult<()> {
    let parsed = parsed_module(db, file).load(db);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);

    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let kind = def.kind(db);
        let place_id = def.place(db);
        let PlaceExprRef::Symbol(_) = place_table.place(place_id) else {
            continue;
        };

        let alias_idx =
            match global_index.get(&(file, place_id, range_key(kind.target_range(&parsed)))) {
                Some(&idx) => idx,
                None => continue,
            };

        let from_stmt = match kind {
            DefinitionKind::ImportFrom(k) => Some(k.import(&parsed)),
            DefinitionKind::ImportFromSubmodule(k) => Some(k.import(&parsed)),
            DefinitionKind::StarImport(k) => Some(k.import(&parsed)),
            _ => None,
        };

        let targets: Vec<usize> = match kind {
            DefinitionKind::Import(k) => resolve_import_target(
                py,
                db,
                k.alias(&parsed).name.id.as_str(),
                file,
                builder,
                module_nodes,
                dist_lookup,
            )?
            .into_iter()
            .collect(),
            DefinitionKind::ImportFrom(k) => resolve_from_imported(
                py,
                db,
                file,
                k.import(&parsed),
                k.alias(&parsed).name.id.as_str(),
                builder,
                module_nodes,
                globals_by_name,
                star_reexports,
                dist_lookup,
            )?,
            DefinitionKind::ImportFromSubmodule(k) => resolve_from_imported(
                py,
                db,
                file,
                k.import(&parsed),
                k.module(&parsed).id.as_str(),
                builder,
                module_nodes,
                globals_by_name,
                star_reexports,
                dist_lookup,
            )?,
            // `from X import *` — no per-name fan-out edge. ty's
            // `definitions_for_imported_symbol` would resolve each
            // per-name `StarImport` to its specific upstream decl,
            // but we collapse the per-name aliases to one node per
            // statement (see `ingest_decls`) and don't want N edges
            // pointing at the same alias. The from-style parallel
            // path below still emits `alias → upstream module` once,
            // which is exactly what carries reachability under the
            // new model — uses of star-bound names emit their own
            // parallel `use → upstream module / upstream decl` edges
            // via `emit_upstream` and ty's name resolution.
            DefinitionKind::StarImport(_) => Vec::new(),
            _ => continue,
        };
        let module_idx_set: HashSet<usize> = module_nodes.values().copied().collect();
        let all_targets_are_modules =
            !targets.is_empty() && targets.iter().all(|idx| module_idx_set.contains(idx));
        for target_idx in &targets {
            builder.add_edge(alias_idx, *target_idx, 0);
        }

        // Parallel reachability edge: when `from X import Y` resolved
        // to a *decl* (Y in module X), also link the alias to the
        // upstream module X so reachability can see the file's
        // module-level dependency. Skip when *every* resolved target
        // is itself a module (the `from X import sub` case where
        // `sub` is a submodule of X) — the submodule's parent-module
        // edge already keeps X alive. When at least one target is a
        // decl, the upstream-module edge is still useful for the
        // others' reachability.
        if let Some(stmt) = from_stmt {
            if !all_targets_are_modules {
                if let Some(upstream_module_idx) =
                    from_import_module_node(py, db, file, stmt, builder, module_nodes, dist_lookup)?
                {
                    if !targets.contains(&upstream_module_idx) && upstream_module_idx != alias_idx {
                        builder.add_edge(alias_idx, upstream_module_idx, 0);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Resolve a dotted module name to its target module node.
///
/// For `import a.b.c`, the alias binds the local name "a" to the
/// deepest module (`a.b.c`) — that's what this returns. Submodule
/// hierarchy edges are only emitted for project modules via
/// [`emit_module_hierarchy`]; external chains are not modeled today.
/// Classification of an import target into one of four graph shapes.
///
/// Mirrors ``dead_cst.resolvers._imports.default_resolve_import``'s
/// stdlib / external-dist / external-file / unresolved buckets from
/// the libcst pipeline. The rust path only needs three of those
/// outcomes — ``[stdlib]`` is silent (no node minted), ``FirstParty``
/// mints a real ``"module"`` node, and ``External`` / ``Unresolved``
/// mint deduplicated ``"synthetic"`` nodes. The ``[external file]``
/// libcst distinction (editable installs) collapses into ``External``
/// here; ty's public ``SearchPath`` predicates don't separate them.
enum ImportTarget {
    Stdlib,
    FirstParty(File),
    /// A site-packages file owned by an installed distribution's
    /// ``RECORD``. Carries the PEP 503-canonical dist name (e.g.
    /// ``"pillow"`` for an ``import PIL``).
    ExternalDist(String),
    /// A site-packages file (or editable / extra search-path file)
    /// not claimed by any installed distribution's ``RECORD``.
    /// Carries the top-level module name. Mirrors libcst's
    /// ``[external file]`` bucket for editable installs and orphan
    /// files in ``site-packages``.
    ExternalFile(String),
    Unresolved(String),
}

/// ``abs_file_path -> PEP 503-canonical dist name``. Populated once
/// per ``materialize`` call by walking every ``*.dist-info/`` directory
/// under each ``is_site_packages()`` search path that ty knows about
/// and indexing the files listed in the distribution's ``RECORD``.
///
/// Mirrors ``dead_cst.resolvers._imports.distribution_lookup`` from
/// the libcst pipeline but reads dist-info directly off the
/// filesystem — no Python callback, so the same lookup works in
/// pyo3 contexts where issuing ``importlib.metadata.distributions()``
/// would round-trip through the runtime.
type DistLookup = HashMap<PathBuf, String>;

/// PEP 503 normalization. Replaces every run of ``[-_.]`` with a
/// single ``-`` and lowercases the rest. Equivalent to Python's
/// ``re.sub(r"[-_.]+", "-", name).lower()`` from libcst's
/// ``_canonical_dist_name``.
fn pep503_canonicalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for ch in name.chars() {
        if matches!(ch, '-' | '_' | '.') {
            if !prev_sep {
                out.push('-');
                prev_sep = true;
            }
        } else {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        }
    }
    out
}

/// Build the dist-file lookup by walking ``*.dist-info/`` under every
/// site-packages search path ty's resolver is configured with.
///
/// For each dist-info directory:
///
/// 1. Read ``METADATA`` for the ``Name:`` header (the project's PyPI
///    name pre-canonicalization). Stop at the blank line that
///    separates headers from the long description.
/// 2. Apply PEP 503 normalization so casing / separators don't matter
///    at lookup time.
/// 3. Read ``RECORD`` for the comma-separated ``path,hash,size``
///    entries, take the path (relative to the site-packages dir),
///    join + canonicalize against the dist-info parent, and map each
///    resolved absolute path to the canonical name.
///
/// Failures (missing METADATA, malformed RECORD, unreadable site-
/// packages) are silently skipped — the file just won't appear in
/// the lookup and the caller falls through to ``[external file]``.
fn build_dist_lookup(db: &dyn ty_python_semantic::Db) -> DistLookup {
    let mut out: DistLookup = HashMap::new();
    for sp in search_paths(db, ModuleResolveMode::StubsAllowed) {
        if !sp.is_site_packages() {
            continue;
        }
        let Some(system_path) = sp.as_system_path() else {
            continue;
        };
        let sp_root = std::fs::canonicalize(system_path.as_str())
            .unwrap_or_else(|_| PathBuf::from(system_path.as_str()));
        let Ok(entries) = std::fs::read_dir(&sp_root) else {
            continue;
        };
        for entry in entries.flatten() {
            let dist_info = entry.path();
            if dist_info.extension().is_none_or(|e| e != "dist-info") {
                continue;
            }
            // METADATA — find the ``Name:`` header.
            let Ok(metadata) = std::fs::read_to_string(dist_info.join("METADATA")) else {
                continue;
            };
            let mut canonical = None;
            for line in metadata.lines() {
                if line.is_empty() {
                    break;
                }
                if let Some(value) = line.strip_prefix("Name:") {
                    canonical = Some(pep503_canonicalize(value.trim()));
                    break;
                }
            }
            let Some(canonical) = canonical else {
                continue;
            };
            // RECORD — index every owned file's absolute path.
            let Ok(record) = std::fs::read_to_string(dist_info.join("RECORD")) else {
                continue;
            };
            for line in record.lines() {
                let Some((rel, _)) = line.split_once(',') else {
                    continue;
                };
                let joined = sp_root.join(rel);
                let abs = std::fs::canonicalize(&joined).unwrap_or(joined);
                out.insert(abs, canonical.clone());
            }
        }
    }
    out
}

/// Classify a dotted module name relative to the file that imports it.
///
/// * Resolved + stdlib → ``Stdlib``
/// * Resolved + first-party + file present → ``FirstParty(file)``
/// * Resolved + non-first-party, file in some dist's ``RECORD`` →
///   ``ExternalDist(canonical-dist-name)``
/// * Resolved + non-first-party, file not in any ``RECORD`` →
///   ``ExternalFile(top-level-module-name)``
/// * Resolved + namespace package (no file) → ``Unresolved(name)``
/// * Unresolved → ``Unresolved(top-level-module-name)``, with a fix-up
///   for dotted-child stdlib misses (``collections.abc`` shouldn't
///   surface as ``[unresolved] collections`` when ``collections`` itself
///   is stdlib but ty's resolver returned None for the child).
fn classify_import_target(
    db: &dyn ty_python_semantic::Db,
    importing_file: File,
    module_name: &ModuleName,
    dist_lookup: &DistLookup,
) -> ImportTarget {
    let dotted = module_name.as_str();
    let top_level = dotted.split('.').next().unwrap_or(dotted);
    let Some(module) = resolve_module(db, importing_file, module_name) else {
        // Dotted-child miss: if the top-level resolves as stdlib,
        // inherit (matches libcst's parent-fallback behavior for
        // ``collections.abc``-shaped names).
        if dotted != top_level {
            if let Some(top_mn) = ModuleName::new(top_level) {
                if resolve_module(db, importing_file, &top_mn)
                    .and_then(|m| m.search_path(db))
                    .is_some_and(|sp| sp.is_standard_library())
                {
                    return ImportTarget::Stdlib;
                }
            }
        }
        return ImportTarget::Unresolved(top_level.to_string());
    };
    let search_path = module.search_path(db);
    if search_path.is_some_and(|sp| sp.is_standard_library()) {
        return ImportTarget::Stdlib;
    }
    if search_path.is_some_and(|sp| sp.is_first_party()) {
        return match module.file(db) {
            Some(f) => ImportTarget::FirstParty(f),
            None => ImportTarget::Unresolved(top_level.to_string()),
        };
    }
    // Non-first-party, non-stdlib: site-packages, editable, extra,
    // or namespace package. Probe the dist-RECORD lookup for the
    // resolved file's canonical name; fall back to ``[external file]``
    // when the file isn't owned by any installed distribution.
    let Some(file) = module.file(db) else {
        return ImportTarget::Unresolved(top_level.to_string());
    };
    let path_str = match file.path(db) {
        FilePath::System(p) => p.to_string(),
        _ => return ImportTarget::ExternalFile(top_level.to_string()),
    };
    let path = std::fs::canonicalize(&path_str).unwrap_or_else(|_| PathBuf::from(&path_str));
    if let Some(canonical) = dist_lookup.get(&path) {
        ImportTarget::ExternalDist(canonical.clone())
    } else {
        ImportTarget::ExternalFile(top_level.to_string())
    }
}

/// Mint (or look up) the graph node a target classification should
/// resolve to. ``Stdlib`` returns ``None`` (silent drop); the other
/// four return a node index.
fn target_to_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    target: ImportTarget,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<Option<usize>> {
    match target {
        ImportTarget::Stdlib => Ok(None),
        ImportTarget::FirstParty(file) => {
            Ok(Some(mint_module_node(py, db, file, builder, module_nodes)?))
        }
        ImportTarget::ExternalDist(name) => Ok(Some(
            builder.intern_synthetic(py, format!("[external dist] {name}"))?,
        )),
        ImportTarget::ExternalFile(name) => Ok(Some(
            builder.intern_synthetic(py, format!("[external file] {name}"))?,
        )),
        ImportTarget::Unresolved(top) => Ok(Some(
            builder.intern_synthetic(py, format!("[unresolved] {top}"))?,
        )),
    }
}

fn resolve_import_target(
    py: Python<'_>,
    db: &ProjectDatabase,
    dotted: &str,
    importing_file: File,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    dist_lookup: &DistLookup,
) -> PyResult<Option<usize>> {
    let Some(module_name) = ModuleName::new(dotted) else {
        return Ok(None);
    };
    let target = classify_import_target(db, importing_file, &module_name, dist_lookup);
    target_to_node(py, db, target, builder, module_nodes)
}

/// Resolve `from <stmt> import <symbol>` to its upstream target node.
///
/// Mirrors CPython's `_handle_fromlist` (`Lib/importlib/_bootstrap.py`):
/// check the package's namespace first, fall back to a submodule only
/// when nothing's bound. Concretely:
///
/// 1. **Namespace lookup** — walks `globals_by_name` for the target
///    module, through any star-reexport chain (`A → B → C`), until it
///    hits a non-reexport binding (a decl or non-star import alias).
///    This is CPython's `hasattr(module, name)` step: if a name is
///    bound in `p/__init__.py` (whether by `q = 42` or by
///    `from . import q`), that binding wins — *even* if a submodule
///    `p/q.py` also exists. The shadow case
///    (`from other import *; def g():` in mod) also lands here via
///    last-write-wins `globals_by_name`.
/// 2. **Submodule fallback** — `<module>.<symbol>` resolves as a
///    module (`from p import q` where `p/q.py` exists and nothing is
///    bound to `q` in `p/__init__.py`). This is CPython's
///    `__import__(f"{p}.{name}")` branch.
/// 3. **Module fallback** — alias still gets an out-edge to the
///    upstream module so reachability propagates.
///
/// Why namespace-first (the CPython order) matters: for `from p
/// import q` where `p/__init__.py` does `q = 42` *and* `p/q.py`
/// exists, CPython binds `q` to the int — the submodule never
/// executes. A submodule-first analyzer would wrongly keep `p/q.py`
/// alive and miss the real binding. The reorder fixes this case and
/// agrees with CPython semantics on every other case (including
/// `from . import q` aliases that bind the submodule to the package
/// namespace explicitly).
///
/// Deliberately does NOT use ty's `definitions_for_imported_symbol`,
/// which recursively chases alias chains across files. Per Principle 2
/// every alias is its own graph node with an outgoing edge, so the
/// transitive walk is already encoded in the graph — replicating it
/// here cost ~100µs per from-import (94% of Phase 2 on flux0 workspace)
/// for no extra reachability information.
#[allow(clippy::too_many_arguments)]
fn resolve_from_imported(
    py: Python<'_>,
    db: &ProjectDatabase,
    importing_file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
    symbol_name: &str,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    globals_by_name: &GlobalsByName,
    star_reexports: &StarReexports,
    dist_lookup: &DistLookup,
) -> PyResult<Vec<usize>> {
    let Ok(module_name) = ModuleName::from_import_statement(db, importing_file, stmt) else {
        return Ok(Vec::new());
    };

    let target = classify_import_target(db, importing_file, &module_name, dist_lookup);
    // External / unresolved targets short-circuit at the module level —
    // we don't have file-level namespace info, so the alias edges to
    // the ``[external dist] X`` / ``[external file] X`` / ``[unresolved] X``
    // synthetic. Stdlib drops silently.
    let target_file = match target {
        ImportTarget::Stdlib => return Ok(Vec::new()),
        ImportTarget::FirstParty(f) => f,
        ImportTarget::ExternalDist(_)
        | ImportTarget::ExternalFile(_)
        | ImportTarget::Unresolved(_) => {
            return Ok(target_to_node(py, db, target, builder, module_nodes)?
                .into_iter()
                .collect());
        }
    };

    // 1. Probe the first-party target's namespace — CPython's
    //    ``hasattr(module, name)``. May return multiple bindings when
    //    the name is branch-bound (try/except, if/else with both
    //    branches assigning).
    let chain = walk_globals_chain(
        db,
        target_file,
        symbol_name,
        globals_by_name,
        star_reexports,
    );
    if !chain.is_empty() {
        return Ok(chain);
    }

    // 2. Namespace miss — fall back to importing `<module>.<symbol>`
    //    as a submodule. CPython's `__import__(f"{p}.{name}")`.
    if let Some(submodule_name) = ModuleName::new(symbol_name) {
        let mut combined = module_name.clone();
        combined.extend(&submodule_name);
        let sub_target = classify_import_target(db, importing_file, &combined, dist_lookup);
        if !matches!(sub_target, ImportTarget::Unresolved(_)) {
            // Stdlib silent-drop / first-party submodule / external
            // submodule all land here. ``Unresolved`` falls through to
            // step 3 so we link to the original module rather than
            // minting ``[unresolved] symbol`` for what's really just
            // "X has no attribute symbol".
            return Ok(target_to_node(py, db, sub_target, builder, module_nodes)?
                .into_iter()
                .collect());
        }
    }

    // 3. Fallback: link the alias to the upstream module node.
    Ok(vec![mint_module_node(
        py,
        db,
        target_file,
        builder,
        module_nodes,
    )?])
}

/// Walk `(file, name)` through star-reexport chains. Returns every
/// non-star-reexport binding reachable from `(target_file, symbol_name)`
/// — a decl, a non-star import alias, or several of either when the
/// name has multiple live bindings (try/except, if/else where both
/// branches assign).
///
/// `from A import g` where A has `from B import *` lands on A's star
/// alias for `g`; we resolve B → file, look up `g` there, and recurse.
/// Stops on a decl, on a non-star import, on a missed lookup (yields
/// nothing past it), or on a cycle (revisit of an already-seen key).
fn walk_globals_chain(
    db: &ProjectDatabase,
    target_file: File,
    symbol_name: &str,
    globals_by_name: &GlobalsByName,
    star_reexports: &StarReexports,
) -> Vec<usize> {
    let mut seen: HashSet<(File, String)> = HashSet::new();
    let mut out: Vec<usize> = Vec::new();
    let mut stack: Vec<(File, String)> = vec![(target_file, symbol_name.to_string())];
    while let Some(key) = stack.pop() {
        if !seen.insert(key.clone()) {
            continue;
        }
        let Some(idxs) = globals_by_name.get(&key) else {
            continue;
        };
        // If `key` is a `from <upstream> import *` reexport, step
        // into the upstream file's same-name lookup. Star reexports
        // carry the name unchanged. Otherwise the binding(s) are the
        // terminal answer.
        if let Some(upstream_module) = star_reexports.get(&key) {
            if let Some(mn) = ModuleName::new(upstream_module) {
                if let Some(upstream) = resolve_module(db, key.0, &mn) {
                    if let Some(upstream_file) = upstream.file(db) {
                        stack.push((upstream_file, key.1.clone()));
                        // The star-alias node itself isn't a useful
                        // target — uses should land on the upstream
                        // decl. Skip emitting it.
                        continue;
                    }
                }
            }
        }
        for &idx in idxs {
            out.push(idx);
        }
    }
    out
}

/// Resolve the upstream module of a `from <stmt> import ...` and
/// return (or mint) its module node.
///
/// Returns `Ok(None)` when ty's `from_import_statement` cannot resolve
/// the target (invalid syntax, too many leading dots, missing file).
fn from_import_module_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    importing_file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
    dist_lookup: &DistLookup,
) -> PyResult<Option<usize>> {
    let Ok(module_name) = ModuleName::from_import_statement(db, importing_file, stmt) else {
        return Ok(None);
    };
    let target = classify_import_target(db, importing_file, &module_name, dist_lookup);
    target_to_node(py, db, target, builder, module_nodes)
}

fn mint_module_node(
    py: Python<'_>,
    db: &ProjectDatabase,
    file: File,
    builder: &mut GraphBuilder,
    module_nodes: &mut HashMap<File, usize>,
) -> PyResult<usize> {
    if let Some(&idx) = module_nodes.get(&file) {
        return Ok(idx);
    }
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let (sl, sc, el, ec) = position(&line_index, &source, parsed.syntax().range);
    let fqname = module_fqname_for_file(db, file);
    let path_str = file_path_string(db, file);
    let flags = file_default_flags(db, file);
    let idx = builder.intern_node(
        py,
        NativeNode {
            fqname,
            kind: "module",
            path: path_str,
            start_line: sl,
            start_column: sc,
            end_line: el,
            end_column: ec,
            flags,
            imports: None,
        },
    )?;
    module_nodes.insert(file, idx);
    Ok(idx)
}

// ---------------------------------------------------------------------------
// Phase 3: same-file Name→decl reference edges
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_reference_edges(
    db: &ProjectDatabase,
    file: File,
    global_index: &DeclIndex,
    module_nodes: &HashMap<File, usize>,
    alias_imports: &HashMap<usize, ImportSpec>,
    live_decls: &LiveDeclIndex,
    dist_lookup: &DistLookup,
    builder: &mut GraphBuilder,
) {
    let Some(&module_idx) = module_nodes.get(&file) else {
        return;
    };
    let parsed = parsed_module(db, file).load(db);
    let dead_ranges = detect_dead_ranges(&parsed);
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let use_def_map = index.use_def_map(global);
    let model = SemanticModel::new(db, file);
    // Move ``synthetic_nodes`` out of the builder for the duration of
    // this pass: the per-statement walks need a long-lived immutable
    // borrow while ``coll.flush(builder)`` takes ``&mut builder``.
    // The synthetic map is populated by ``emit_import_edges`` ahead of
    // this phase and isn't mutated here, so swap-out / swap-in is safe.
    let synthetic_nodes = std::mem::take(&mut builder.synthetic_nodes);

    // (a) Definitions that own an expression / body — attribute their
    //     contained Names to the owning decl.
    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let kind = def.kind(db);
        let target_range = kind.target_range(&parsed);
        let Some(&owner_idx) = global_index.get(&(file, def.place(db), range_key(target_range)))
        else {
            continue;
        };

        let mut coll = RefCollector::new(
            owner_idx,
            &model,
            file,
            &parsed,
            index,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            &synthetic_nodes,
            dist_lookup,
            &dead_ranges,
        );
        walk_owned(kind, &parsed, &mut coll);
        coll.flush(builder);
    }

    // (b) Module-level statements that don't carry a Definition (and
    //     so didn't get covered by (a)) attribute to the module node.
    for stmt in &parsed.syntax().body {
        if stmt_creates_top_level_definition(stmt) {
            continue;
        }
        let mut coll = RefCollector::new(
            module_idx,
            &model,
            file,
            &parsed,
            index,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            &synthetic_nodes,
            dist_lookup,
            &dead_ranges,
        );
        coll.visit_stmt(stmt);
        coll.flush(builder);
    }
    builder.synthetic_nodes = synthetic_nodes;
}

/// True iff this top-level statement is a binding form whose Names
/// have already been attributed by the per-definition walk in (a).
///
/// Compound non-scope statements (``if`` / ``while`` / ``for`` / ...)
/// return ``false`` here: their *bodies* contain definitions that (a)
/// covers, but their *test/iter/etc. expressions* belong to the module
/// and (b) needs to walk them.
fn stmt_creates_top_level_definition(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::FunctionDef(_)
            | Stmt::ClassDef(_)
            | Stmt::Assign(_)
            | Stmt::AnnAssign(_)
            | Stmt::TypeAlias(_)
            | Stmt::Import(_)
            | Stmt::ImportFrom(_)
    )
}

/// Walk every value-bearing AST node a Definition owns.
///
/// Functions and classes own their body statements; assignments own
/// the RHS expression; annotated assignments own annotation + value;
/// `for x in iter:` owns the iterable; `with X as y:` owns the
/// context expression; walrus owns its value; type aliases own their
/// value expression. Other Definition kinds (imports, parameters, …)
/// own no walk-worthy expression.
fn walk_owned(kind: &DefinitionKind<'_>, parsed: &ParsedModuleRef, v: &mut RefCollector<'_, '_>) {
    match kind {
        DefinitionKind::Function(func) => {
            let node = func.node(parsed);
            // Header parts evaluate at the *definition* site (module
            // scope for top-level defs), not inside the body — leave
            // `nested_context` false so a stray `import X` in a
            // decorator expression doesn't get re-attributed as a
            // body-local nested import.
            walk_decorators(&node.decorator_list, v);
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            walk_parameters(&node.parameters, v);
            if let Some(returns) = &node.returns {
                v.visit_expr(returns);
            }
            v.nested_context = true;
            for s in &node.body {
                v.visit_stmt(s);
            }
            v.nested_context = false;
        }
        DefinitionKind::Class(cls) => {
            let node = cls.node(parsed);
            walk_decorators(&node.decorator_list, v);
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            if let Some(args) = &node.arguments {
                for base in &args.args {
                    v.visit_expr(base);
                }
                for kw in &args.keywords {
                    v.visit_expr(&kw.value);
                }
            }
            v.nested_context = true;
            for s in &node.body {
                v.visit_stmt(s);
            }
            v.nested_context = false;
        }
        DefinitionKind::Assignment(a) => {
            let value = a.value(parsed);
            if target_is_dunder_all(a.target(parsed)) {
                emit_dunder_all_edges(v, value);
            } else if let TargetKind::Sequence(_, unpack) = a.target_kind() {
                // ``c, d = a, b`` produces one Definition per LHS
                // name, each with ``value`` set to the whole RHS
                // ``(a, b)``. Walking the full RHS for both ``c`` and
                // ``d`` over-approximates (``c -> b``, ``d -> a``);
                // when both sides are flat sequences of matching
                // arity, pair index-by-index instead.
                let db = v.model.db();
                let lhs = unpack.target(db, parsed);
                if let Some(paired) = paired_unpack_rhs(lhs, a.target(parsed), value) {
                    v.visit_expr(paired);
                } else {
                    v.visit_expr(value);
                }
            } else {
                v.visit_expr(value);
            }
        }
        DefinitionKind::AnnotatedAssignment(a) => {
            v.visit_expr(a.annotation(parsed));
            if let Some(val) = a.value(parsed) {
                if target_is_dunder_all(a.target(parsed)) {
                    emit_dunder_all_edges(v, val);
                } else {
                    v.visit_expr(val);
                }
            }
        }
        DefinitionKind::NamedExpression(named) => v.visit_expr(named.node(parsed).value.as_ref()),
        DefinitionKind::TypeAlias(alias) => {
            let node = alias.node(parsed);
            if let Some(type_params) = &node.type_params {
                walk_type_params(type_params, v);
            }
            v.visit_expr(node.value.as_ref());
        }
        // For / WithItem / ExceptHandler / MatchPattern bindings are
        // not modeled as top-level decls (see `decl_kind_str`), so
        // their definitions never appear in `global_index` and
        // `walk_owned` never runs for them. Their value-bearing
        // sub-expressions (loop iterables, context managers, etc.)
        // are walked instead from the module-level non-definition
        // pass, where `Stmt::For` / `Stmt::With` / `Stmt::Try` get
        // their normal `walk_stmt` recursion.
        _ => {}
    }
}

fn walk_decorators(decorators: &[ruff_python_ast::Decorator], v: &mut RefCollector<'_, '_>) {
    for d in decorators {
        v.visit_expr(&d.expression);
    }
}

fn walk_parameters(parameters: &ruff_python_ast::Parameters, v: &mut RefCollector<'_, '_>) {
    for p in parameters
        .posonlyargs
        .iter()
        .chain(&parameters.args)
        .chain(&parameters.kwonlyargs)
    {
        if let Some(annotation) = &p.parameter.annotation {
            v.visit_expr(annotation);
        }
        if let Some(default) = &p.default {
            v.visit_expr(default);
        }
    }
    if let Some(vararg) = &parameters.vararg {
        if let Some(annotation) = &vararg.annotation {
            v.visit_expr(annotation);
        }
    }
    if let Some(kwarg) = &parameters.kwarg {
        if let Some(annotation) = &kwarg.annotation {
            v.visit_expr(annotation);
        }
    }
}

fn walk_type_params(type_params: &ruff_python_ast::TypeParams, v: &mut RefCollector<'_, '_>) {
    for tp in &type_params.type_params {
        match tp {
            ruff_python_ast::TypeParam::TypeVar(t) => {
                if let Some(bound) = &t.bound {
                    v.visit_expr(bound);
                }
                if let Some(default) = &t.default {
                    v.visit_expr(default);
                }
            }
            ruff_python_ast::TypeParam::TypeVarTuple(t) => {
                if let Some(default) = &t.default {
                    v.visit_expr(default);
                }
            }
            ruff_python_ast::TypeParam::ParamSpec(t) => {
                if let Some(default) = &t.default {
                    v.visit_expr(default);
                }
            }
        }
    }
}

/// Pair a tuple-unpack target with the RHS element at the same
/// position, so each LHS name only collects edges from the value it
/// actually receives.
///
/// Returns the paired RHS element when:
/// * the full LHS (the parent unpack target) is a flat `Tuple` /
///   `List` literal with no starred element, and
/// * the RHS is a flat `Tuple` / `List` literal with the same arity,
///   and
/// * the bare `target` name appears as one of the LHS elements.
///
/// Returns `None` otherwise so the caller falls back to walking the
/// whole RHS (the safe over-approximation for non-literal RHS values
/// like `c, d = f()` and starred-target patterns).
fn paired_unpack_rhs<'ast>(lhs: &Expr, target: &Expr, rhs: &'ast Expr) -> Option<&'ast Expr> {
    let lhs_elts = match lhs {
        Expr::Tuple(t) => &t.elts,
        Expr::List(l) => &l.elts,
        _ => return None,
    };
    let rhs_elts = match rhs {
        Expr::Tuple(t) => &t.elts,
        Expr::List(l) => &l.elts,
        _ => return None,
    };
    if lhs_elts.len() != rhs_elts.len() {
        return None;
    }
    if lhs_elts.iter().any(|e| matches!(e, Expr::Starred(_))) {
        return None;
    }
    let target_range = target.range();
    let pos = lhs_elts.iter().position(|e| e.range() == target_range)?;
    Some(&rhs_elts[pos])
}

/// True iff `target` is the bare `Name("__all__")` LHS of an
/// assignment (`__all__ = ...` or `__all__: list[str] = ...`).
///
/// Subscript / attribute / tuple-unpack targets (e.g.
/// `__all__[0] = "f"`, `mod.__all__ = [...]`) are excluded — those
/// don't redefine the binding in a way the libcst pipeline picks up,
/// and treating them as `__all__` would emit edges from unrelated
/// nodes.
fn target_is_dunder_all(target: &Expr) -> bool {
    matches!(target, Expr::Name(n) if n.id.as_str() == "__all__")
}

/// Walk the value of an `__all__` assignment and emit one edge from
/// the owner (the `__all__` variable node) to each module-scope
/// binding whose name appears in the list/tuple.
///
/// Only string-literal elements are followed; computed entries (e.g.
/// `__all__ = [*BASE, "extra"]`, `__all__ = list(...)`) are silently
/// skipped — matching the libcst pipeline, which folds `__all__` only
/// when it's assigned a list or tuple of string literals. Names that
/// don't resolve in the file's global scope are skipped without a
/// warning (`__all__ = ["missing"]` is a runtime error at import
/// time, not a static dep).
fn emit_dunder_all_edges(v: &mut RefCollector<'_, '_>, value: &Expr) {
    let elements = match value {
        Expr::List(l) => &l.elts,
        Expr::Tuple(t) => &t.elts,
        _ => return,
    };
    for elem in elements {
        if let Expr::StringLiteral(s) = elem {
            if let Some(idx) = v.lookup_module_scope_name(s.value.to_str()) {
                v.emit_edge(idx);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference collector
// ---------------------------------------------------------------------------

/// Walks an expression / body and records every Name reference,
/// attributing to a single owner decl.
///
/// Per Principle 2, every use of an imported name emits an edge to
/// the local alias *and* parallel reachability edges to whatever the
/// alias resolves to upstream. Bare-Name uses (`f()`) emit edges to
/// the upstream module/decl directly. Attribute chains on aliased
/// modules (`foo.bar.f()`) are walked segment by segment, emitting
/// edges to each module / decl reached. Shadow handling falls out of
/// ty's flow-sensitive use-def chain (Principle 3).
struct RefCollector<'a, 'db> {
    owner: usize,
    model: &'a SemanticModel<'db>,
    file: File,
    parsed: &'a ParsedModuleRef,
    /// Cached `&SemanticIndex` for `self.file`. Avoids re-doing the
    /// Salsa `semantic_index(db, file)` lookup per Name reference —
    /// every call to `find_local_bindings` / `lookup_module_scope_name`
    /// used to issue one. ~10k saved lookups on a 100-file workspace.
    index: &'a SemanticIndex<'db>,
    global_index: &'a DeclIndex,
    module_nodes: &'a HashMap<File, usize>,
    alias_imports: &'a HashMap<usize, ImportSpec>,
    live_decls: &'a LiveDeclIndex,
    /// Ranges of statically-dead source regions in the current file
    /// (`if False:` bodies, statements after a `return`/`raise`/`break`/
    /// `continue`/`assert <falsy>`, etc.). A use whose source range
    /// is `contains_range`-covered by any of these gets
    /// `EdgeFlags::DEAD_BRANCH` stamped on every edge it emits.
    dead_ranges: &'a [TextRange],
    /// `{ synthetic_fqname -> node idx }` mirrored from
    /// ``GraphBuilder::synthetic_nodes`` so the collector can route
    /// upstream edges through pre-minted ``[external dist] X``,
    /// ``[external file] X``, and ``[unresolved] X`` synthetics
    /// without holding the mutable builder. Populated by
    /// ``emit_import_edges`` before the collector pass runs.
    synthetic_nodes: &'a HashMap<String, usize>,
    /// ``abs_file_path -> PEP 503-canonical dist name`` built once
    /// per ``materialize`` call. The collector re-classifies
    /// ``emit_upstream``'s loading target to compute the same
    /// synthetic fqname ``emit_import_edges`` used at mint time.
    dist_lookup: &'a DistLookup,
    /// Edges accumulated for this collector pass. The value is the
    /// AND of the flags contributed by each reference that produced
    /// this `(src, dst)` pair — so a `(src, dst)` reachable from both
    /// a live and a dead reference loses `DEAD_BRANCH` (the live ref
    /// wins), matching the libcst pipeline's parallel-edge semantics.
    edges: HashMap<(usize, usize), u32>,
    /// `true` while walking a function- or class-body subtree. In
    /// that context, nested `Stmt::Import` / `Stmt::ImportFrom`
    /// statements emit parallel upstream edges from `owner` (no
    /// alias node is minted, since the binding lives in a non-global
    /// scope). At module level the flag stays false, so we don't
    /// double-emit for imports that `emit_import_edges` already
    /// processed via their proper alias nodes.
    nested_context: bool,
    /// Flags stamped on each edge emitted by the current reference
    /// (set by `emit_name_use` / nested-import handlers based on the
    /// reference's source position; cleared afterward).
    current_flags: u32,
}

impl<'a, 'db> RefCollector<'a, 'db> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        owner: usize,
        model: &'a SemanticModel<'db>,
        file: File,
        parsed: &'a ParsedModuleRef,
        index: &'a SemanticIndex<'db>,
        global_index: &'a DeclIndex,
        module_nodes: &'a HashMap<File, usize>,
        alias_imports: &'a HashMap<usize, ImportSpec>,
        live_decls: &'a LiveDeclIndex,
        synthetic_nodes: &'a HashMap<String, usize>,
        dist_lookup: &'a DistLookup,
        dead_ranges: &'a [TextRange],
    ) -> Self {
        Self {
            owner,
            model,
            file,
            parsed,
            index,
            global_index,
            module_nodes,
            alias_imports,
            live_decls,
            synthetic_nodes,
            dist_lookup,
            dead_ranges,
            edges: HashMap::new(),
            nested_context: false,
            current_flags: 0,
        }
    }

    fn flush(self, builder: &mut GraphBuilder) {
        for ((src, dst), flags) in self.edges {
            builder.add_edge(src, dst, flags);
        }
    }

    fn emit_edge(&mut self, dst: usize) {
        if dst != self.owner {
            let flags = self.current_flags;
            self.edges
                .entry((self.owner, dst))
                .and_modify(|f| *f &= flags)
                .or_insert(flags);
        }
    }

    /// Returns `EDGE_FLAG_DEAD_BRANCH` if `range` is contained in any
    /// dead-region recorded for this file, else `0`.
    fn flags_for_range(&self, range: TextRange) -> u32 {
        if self.dead_ranges.iter().any(|r| r.contains_range(range)) {
            EDGE_FLAG_DEAD_BRANCH
        } else {
            0
        }
    }

    /// Look up a name in the current file's global scope and return
    /// its end-of-scope live binding's graph node.
    ///
    /// Used by the `__all__` walk: each string literal listed there
    /// should resolve to a top-level decl (or import alias) bound in
    /// the same file. Names that don't resolve are silently skipped —
    /// `__all__ = ["missing"]` is a runtime error at import time but
    /// doesn't influence static dep tracking.
    fn lookup_module_scope_name(&self, name: &str) -> Option<usize> {
        let db = self.model.db();
        let global = FileScopeId::global();
        let place_table = self.index.place_table(global);
        let symbol_id = place_table.symbol_id(name)?;
        let use_def_map = self.index.use_def_map(global);
        for binding in use_def_map.end_of_scope_symbol_bindings(symbol_id) {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != self.file {
                continue;
            }
            let key = (
                self.file,
                def.place(db),
                range_key(def.kind(db).target_range(self.parsed)),
            );
            if let Some(&idx) = self.global_index.get(&key) {
                return Some(idx);
            }
        }
        None
    }

    /// Walk the use's scope chain looking for the reaching definitions
    /// of `name`. Returns one entry per reaching def — typically a
    /// single binding, but `try`/`except` branches that both bind the
    /// name leave multiple defs reaching the use and every one gets
    /// edges (Principle 3).
    ///
    /// The use's own scope is queried with
    /// :meth:`UseDefMap::bindings_at_use`, which is position-sensitive
    /// — for `a = 1; a = a + 1`, the RHS `a` resolves to the line-1
    /// def even though the line-2 def has already been registered in
    /// the scope's place table. For free variables (used in a scope
    /// that doesn't bind the name), walks outward through
    /// `visible_ancestor_scopes` and falls back to that scope's
    /// end-of-scope bindings.
    ///
    /// Resolutions split on whether the binding has a graph node:
    /// module-scope decls and import aliases give `Alias(idx)`;
    /// imports nested in a function/class scope (no graph node minted)
    /// give `NestedImport { spec, bound_name }` so the caller can fan
    /// out parallel upstream edges from the enclosing top-level owner.
    ///
    /// Deliberately does NOT use `definitions_for_name`: that walks
    /// past `from X import y` to the upstream definition in `X`,
    /// flattening the local alias edge Principle 2 requires.
    fn find_local_bindings(&self, name: &ExprName) -> Vec<Resolution> {
        let db = self.model.db();
        let Some(file_scope) = self.model.scope(name.into()) else {
            return Vec::new();
        };
        let mut first = true;
        for (scope_id, _scope) in self.index.visible_ancestor_scopes(file_scope) {
            let place_table = self.index.place_table(scope_id);
            let Some(symbol_id) = place_table.symbol_id(name.id.as_str()) else {
                first = false;
                continue;
            };
            let use_def_map = self.index.use_def_map(scope_id);
            // Position-sensitive query for the use's own scope; fall
            // back to end-of-scope bindings for enclosing scopes (where
            // the use isn't recorded under any specific position).
            let bindings = if first {
                let use_id = name.scoped_use_id(db, scope_id.to_scope_id(db, self.file));
                use_def_map.bindings_at_use(use_id)
            } else {
                use_def_map.end_of_scope_symbol_bindings(symbol_id)
            };
            let mut saw_binding = false;
            let mut results: Vec<Resolution> = Vec::new();
            for binding in bindings {
                let Some(def) = binding.binding.definition() else {
                    continue;
                };
                if def.file(db) != self.file {
                    continue;
                }
                saw_binding = true;
                let kind = def.kind(db);
                let place_id = def.place(db);
                let key = (
                    self.file,
                    place_id,
                    range_key(kind.target_range(self.parsed)),
                );
                if let Some(&idx) = self.global_index.get(&key) {
                    results.push(Resolution::Alias(idx));
                    continue;
                }
                if kind.is_import() {
                    let PlaceExprRef::Symbol(sym) = place_table.place(place_id) else {
                        continue;
                    };
                    let bound_name = sym.name().as_str().to_string();
                    let spec = import_payload_for(kind, db, self.file, self.parsed);
                    results.push(Resolution::NestedImport { spec, bound_name });
                }
            }
            if saw_binding {
                return results;
            }
            first = false;
        }
        Vec::new()
    }

    /// Emit edges implied by a use of `name`.
    ///
    /// `extra_chain` is the list of attribute segments past the bare
    /// name (`[]` for a bare-name use, `["bar", "f"]` for
    /// `name.bar.f`).
    ///
    /// For module-scope bindings: emit `owner → alias` (codemod
    /// invariant), then parallel upstream reachability edges from
    /// `alias_imports[idx]` if the binding is an import. For nested
    /// imports there's no alias node — go straight to the parallel
    /// edges, using the spec ty handed us. When ty reports multiple
    /// reaching defs (if/else branches, try/except, …), we emit for
    /// each one.
    fn emit_name_use(&mut self, name: &ExprName, extra_chain: &[&str]) {
        // `Name`s in non-`Load` context are binding sites, not uses:
        // the `x` in `for x in data`, `with y as x`, `except E as x`,
        // and the LHS of `x = ...` / `del x`. Skipping them keeps the
        // graph free of spurious target → binding-source edges (and
        // mirrors the libcst pipeline, which only emits edges for
        // reads).
        if !matches!(name.ctx, ruff_python_ast::ExprContext::Load) {
            return;
        }
        self.current_flags = self.flags_for_range(name.range());
        for resolution in self.find_local_bindings(name) {
            match resolution {
                Resolution::Alias(alias_idx) => {
                    self.emit_edge(alias_idx);
                    if let Some(spec) = self.alias_imports.get(&alias_idx).cloned() {
                        self.emit_upstream(&spec, name.id.as_str(), extra_chain);
                    }
                }
                Resolution::NestedImport { spec, bound_name } => {
                    self.emit_upstream(&spec, &bound_name, extra_chain);
                }
            }
        }
        self.current_flags = 0;
    }

    fn emit_upstream(&mut self, spec: &ImportSpec, bound_name: &str, extra_chain: &[&str]) {
        if spec.module.is_empty() {
            return;
        }

        // Decide the "loading target" — the absolute dotted module
        // the alias makes available — plus any extra prefix segments
        // the import statement walked deeper than the bound name.
        //
        // `import M[.Y.Z]` no-asname: bound name matches `M` (first
        //   segment). The runtime object of `M` is just the package
        //   `M`, but the import statement *loaded* `M.Y.Z`. The
        //   user's attribute chain typically re-traverses `.Y.Z`
        //   before walking beyond; we peel that prefix off so the
        //   walk continues from the deepest loaded module.
        // `import M[.Y.Z] as A`: bound name is `A`; loading target
        //   is `M.Y.Z` and there's no prefix to strip.
        // `from P import D[as A]`: try `P.D` as submodule; if it
        //   resolves the alias represents that submodule, else it
        //   binds the decl `D` in `P`.
        // `from P import *` (per-name synthetic): bound name is the
        //   name brought in; alias binds either `P.<name>` (submodule)
        //   or decl `<name>` in `P`.
        let db = self.model.db();
        let module_first_seg = spec.module.split('.').next().unwrap_or("").to_string();

        let mut adjusted_chain: Vec<&str> = extra_chain.to_vec();
        let loading_target: String;
        let mut decl_tail: Option<String> = None;

        if spec.star {
            // Star reexport: alias's `module` is the source package and
            // `bound_name` is one of the names it exports. Resolve as
            // either submodule `module.bound_name` or decl `bound_name`
            // in `module`.
            let candidate = format!("{}.{}", spec.module, bound_name);
            if module_name_resolves(&candidate, self.file, db) {
                loading_target = candidate;
            } else {
                loading_target = spec.module.clone();
                decl_tail = Some(bound_name.to_string());
            }
        } else {
            match &spec.decl {
                Some(decl) => {
                    let candidate = format!("{}.{}", spec.module, decl);
                    if module_name_resolves(&candidate, self.file, db) {
                        loading_target = candidate;
                    } else {
                        loading_target = spec.module.clone();
                        decl_tail = Some(decl.clone());
                    }
                }
                None => {
                    let no_asname = bound_name == module_first_seg;
                    if no_asname && spec.module != module_first_seg {
                        // `import M.Y.Z` no-asname: peel the loading
                        // prefix off the chain before walking past.
                        let loading_extras: Vec<&str> = spec.module.split('.').skip(1).collect();
                        let n = loading_extras.len();
                        let prefix_matches = adjusted_chain.len() >= n
                            && adjusted_chain
                                .iter()
                                .take(n)
                                .zip(&loading_extras)
                                .all(|(a, b)| *a == *b);
                        if prefix_matches {
                            adjusted_chain.drain(..n);
                            loading_target = spec.module.clone();
                        } else {
                            // User reached for something off the bare
                            // `M` module (`import M.Y.Z; M.other`).
                            // Walk from `M`, not `M.Y.Z`.
                            loading_target = module_first_seg;
                        }
                    } else {
                        loading_target = spec.module.clone();
                    }
                }
            }
        }

        // Classify the loading target. Stdlib drops silently;
        // external / unresolved fan out to the pre-minted synthetic
        // and stop (no submodule chain walk through them).
        let Some(start_mn) = ModuleName::new(&loading_target) else {
            return;
        };
        let target = classify_import_target(db, self.file, &start_mn, self.dist_lookup);
        let start_file = match target {
            ImportTarget::Stdlib => return,
            ImportTarget::FirstParty(f) => f,
            ImportTarget::ExternalDist(name) => {
                if let Some(&idx) = self.synthetic_nodes.get(&format!("[external dist] {name}")) {
                    self.emit_edge(idx);
                }
                return;
            }
            ImportTarget::ExternalFile(name) => {
                if let Some(&idx) = self.synthetic_nodes.get(&format!("[external file] {name}")) {
                    self.emit_edge(idx);
                }
                return;
            }
            ImportTarget::Unresolved(top) => {
                if let Some(&idx) = self.synthetic_nodes.get(&format!("[unresolved] {top}")) {
                    self.emit_edge(idx);
                }
                return;
            }
        };

        // Decl-style alias: emit edges to the upstream module and the
        // decl inside it, then stop. Attribute access past a decl is
        // field access on the decl's value, which we don't model.
        if let Some(decl_name) = decl_tail {
            if let Some(&idx) = self.module_nodes.get(&start_file) {
                self.emit_edge(idx);
            }
            if let Some(idxs) = self.live_decls.get(&(start_file, decl_name)) {
                for &idx in idxs {
                    self.emit_edge(idx);
                }
            }
            return;
        }

        // Module-style alias: walk the chain submodule-by-submodule
        // and record the *deepest* module reached plus any decl the
        // chain ends on. We emit exactly one module edge (the deepest)
        // and at most one decl edge — mirroring the libcst stitcher's
        // canonicalization rule that pushes decl parts into the module
        // as long as they resolve as submodules.
        let mut current_file = start_file;
        let mut current_path = loading_target.clone();
        let mut terminal_decl_idxs: Vec<usize> = Vec::new();
        for seg in &adjusted_chain {
            let candidate = format!("{current_path}.{seg}");
            let submodule_file = ModuleName::new(&candidate)
                .and_then(|mn| resolve_module(db, self.file, &mn))
                .and_then(|m| m.file(db));
            if let Some(sub_file) = submodule_file {
                current_file = sub_file;
                current_path = candidate;
                continue;
            }
            terminal_decl_idxs = self
                .live_decls
                .get(&(current_file, (*seg).to_string()))
                .cloned()
                .unwrap_or_default();
            break;
        }

        if let Some(&idx) = self.module_nodes.get(&current_file) {
            self.emit_edge(idx);
        }
        for idx in terminal_decl_idxs {
            self.emit_edge(idx);
        }
    }

    /// Handle `import X[.Y.Z][ as A]` inside a function/class body.
    ///
    /// No alias node is minted — the binding lives in a non-global
    /// scope. We emit parallel upstream edges directly from `self.owner`
    /// for each alias in the statement, simulating a `synthetic chain`
    /// that matches the loading prefix so `emit_upstream` walks all
    /// the way to the deepest loaded module rather than stopping at
    /// the bound name's first segment (the bare-name use shortcut
    /// that's correct for use sites but wrong for the import
    /// statement itself).
    fn emit_nested_import(&mut self, stmt: &ruff_python_ast::StmtImport) {
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let dotted = alias.name.id.as_str();
            let first_seg = dotted.split('.').next().unwrap_or(dotted);
            let (bound_name, synthetic_chain): (&str, Vec<&str>) = match &alias.asname {
                Some(asname) => (asname.id.as_str(), Vec::new()),
                None => (first_seg, dotted.split('.').skip(1).collect()),
            };
            let spec = ImportSpec {
                module: dotted.to_string(),
                decl: None,
                star: false,
            };
            self.emit_upstream(&spec, bound_name, &synthetic_chain);
        }
        self.current_flags = 0;
    }

    /// Handle `from X import ...` inside a function/class body.
    ///
    /// Resolves the from-clause via ty (so relative imports get
    /// their level dots converted to an absolute name), then walks
    /// every alias. `*` imports fan out to every non-underscore decl
    /// in the upstream module via `emit_nested_star`; explicit names
    /// (with or without `as`) emit upstream edges through the same
    /// `emit_upstream` path module-scope from-imports use.
    fn emit_nested_import_from(&mut self, stmt: &ruff_python_ast::StmtImportFrom) {
        let db = self.model.db();
        let Ok(module_name) = ModuleName::from_import_statement(db, self.file, stmt) else {
            return;
        };
        let module_str = module_name.as_str().to_string();
        self.current_flags = self.flags_for_range(stmt.range);
        for alias in &stmt.names {
            let name = alias.name.id.as_str();
            if name == "*" {
                self.emit_nested_star(&module_name);
                continue;
            }
            let bound_name = match &alias.asname {
                Some(asname) => asname.id.as_str(),
                None => name,
            };
            let spec = ImportSpec {
                module: module_str.clone(),
                decl: Some(name.to_string()),
                star: false,
            };
            self.emit_upstream(&spec, bound_name, &[]);
        }
        self.current_flags = 0;
    }

    /// Fan a nested `from X import *` out to every non-underscore
    /// decl in the upstream module, plus the module itself.
    ///
    /// Module-scope star imports use ty's `StarImport` definitions
    /// (one per resolved name) and route through the per-alias-node
    /// path; nested star imports have no per-name graph nodes to
    /// mint, so we go straight to the upstream `live_decls`. The
    /// underscore filter approximates the libcst pipeline's star
    /// expansion, which respects PEP 8's "names starting with `_`
    /// are not star-exported" rule (and `__all__` when present —
    /// not modeled here yet, but no current test exercises that).
    fn emit_nested_star(&mut self, module_name: &ModuleName) {
        let db = self.model.db();
        let Some(module) = resolve_module(db, self.file, module_name) else {
            return;
        };
        let Some(target_file) = module.file(db) else {
            return;
        };
        if let Some(&idx) = self.module_nodes.get(&target_file) {
            self.emit_edge(idx);
        }
        let targets: Vec<usize> = self
            .live_decls
            .iter()
            .filter(|((file, name), _)| *file == target_file && !name.starts_with('_'))
            .flat_map(|(_, idxs)| idxs.iter().copied())
            .collect();
        for idx in targets {
            self.emit_edge(idx);
        }
    }

    /// Handle a call expression that may be a dynamic-import shape:
    /// `__import__('name', …)` or `importlib.import_module('name', …)`.
    ///
    /// Returns `true` if the call was recognized (recognized but
    /// rejected — e.g. non-literal first argument — still returns
    /// `true`, since the visitor shouldn't fall through to walk the
    /// arguments looking for normal name references). Returns `false`
    /// when the call doesn't match either shape.
    fn try_emit_dynamic_import(&mut self, call: &ruff_python_ast::ExprCall) -> bool {
        let Some(kind) = detect_dynamic_call(&call.func) else {
            return false;
        };
        let py = unsafe { Python::assume_gil_acquired() };
        match parse_dynamic_args(kind, call) {
            DynamicParseResult::Ok {
                name,
                fromlist,
                explicit_package,
                explicit_level,
            } => {
                let db = self.model.db();
                let file_pkg = file_package_name(db, self.file);
                let pkg = explicit_package.or(file_pkg.as_deref());
                match resolve_dynamic_target(kind, &name, explicit_level, pkg) {
                    Ok(target) => self.emit_dynamic_edges(&target, &fromlist),
                    Err(message) => emit_visitor_warning(py, &message),
                }
                true
            }
            DynamicParseResult::Warn(message) => {
                emit_visitor_warning(py, &message);
                true
            }
            DynamicParseResult::NotApplicable => false,
        }
    }

    /// Emit `owner → target` (and `owner → target.entry` for each
    /// fromlist entry that resolves) for a dynamic import, each
    /// tagged with `EDGE_FLAG_DYNAMIC_IMPORT`. The visitor stays
    /// minimal: one edge per literal symbol the call mentioned. A
    /// contrib plugin can read the flag and fan out further.
    fn emit_dynamic_edges(&mut self, target: &str, fromlist: &[&str]) {
        let db = self.model.db();
        let saved = self.current_flags;
        self.current_flags |= EDGE_FLAG_DYNAMIC_IMPORT;

        // Edge to the literal name's module — but only when there's
        // no fromlist, since with a fromlist the base module is just
        // a stepping stone to the named entries.
        if fromlist.is_empty() {
            self.emit_resolved_module(target);
        } else {
            // With a non-empty fromlist Python still loads the base
            // module (`__import__('p', fromlist=[…])` returns `p`),
            // so emit the base edge and then resolve each entry as
            // either a submodule or a global-scope decl in that
            // module.
            self.emit_resolved_module(target);
            for entry in fromlist {
                if entry.is_empty() {
                    continue;
                }
                let candidate = format!("{target}.{entry}");
                if module_name_resolves(&candidate, self.file, db) {
                    self.emit_resolved_module(&candidate);
                    continue;
                }
                let target_file = ModuleName::new(target)
                    .and_then(|n| resolve_module(db, self.file, &n))
                    .and_then(|m| m.file(db));
                if let Some(target_file) = target_file {
                    if let Some(idxs) = self.live_decls.get(&(target_file, (*entry).to_string())) {
                        for &decl_idx in idxs {
                            self.emit_edge(decl_idx);
                        }
                    }
                }
                // Entries that don't resolve as either submodule or
                // decl are dropped silently — the libcst pipeline
                // does the same.
            }
        }

        self.current_flags = saved;
    }

    fn emit_resolved_module(&mut self, dotted: &str) {
        let db = self.model.db();
        let Some(mn) = ModuleName::new(dotted) else {
            return;
        };
        let Some(module) = resolve_module(db, self.file, &mn) else {
            return;
        };
        if module
            .search_path(db)
            .is_some_and(|sp| sp.is_standard_library())
        {
            return;
        }
        let Some(target_file) = module.file(db) else {
            return;
        };
        if let Some(&idx) = self.module_nodes.get(&target_file) {
            self.emit_edge(idx);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DynamicKind {
    DunderImport,
    ImportlibImportModule,
}

impl DynamicKind {
    fn label(self) -> &'static str {
        match self {
            DynamicKind::DunderImport => "__import__(...)",
            DynamicKind::ImportlibImportModule => "importlib.import_module(...)",
        }
    }
}

/// Match `call.func` against the two dynamic-import call shapes we
/// recognize: a bare `Name("__import__")` or
/// `Attribute(Name("importlib"), "import_module")`. The `importlib`
/// receiver is matched textually — a local `class importlib` would
/// be a false positive, but that pattern is vanishingly rare in
/// practice and the libcst pipeline does the same syntactic match.
fn detect_dynamic_call(func: &Expr) -> Option<DynamicKind> {
    match func {
        Expr::Name(n) if n.id.as_str() == "__import__" => Some(DynamicKind::DunderImport),
        Expr::Attribute(attr) => {
            if let Expr::Name(receiver) = &*attr.value {
                if receiver.id.as_str() == "importlib" && attr.attr.as_str() == "import_module" {
                    return Some(DynamicKind::ImportlibImportModule);
                }
            }
            None
        }
        _ => None,
    }
}

enum DynamicParseResult<'a> {
    /// Successfully parsed.
    Ok {
        name: String,
        fromlist: Vec<&'a str>,
        explicit_package: Option<&'a str>,
        explicit_level: Option<u32>,
    },
    /// Recognized as a dynamic import but rejected with this warning
    /// message. The visitor still skips the call's argument walk —
    /// the call site itself doesn't carry normal name references
    /// once we've identified it as a dynamic import.
    Warn(String),
    /// Not a dynamic import (e.g. the first argument is missing
    /// entirely, so it isn't really one of these call shapes). Caller
    /// should fall through to a normal expression walk.
    NotApplicable,
}

fn parse_dynamic_args<'a>(
    kind: DynamicKind,
    call: &'a ruff_python_ast::ExprCall,
) -> DynamicParseResult<'a> {
    let Some(first_arg) = first_positional(call) else {
        return DynamicParseResult::NotApplicable;
    };
    let Some(name) = string_literal(first_arg) else {
        return DynamicParseResult::Warn(format!(
            "Skipping dynamic import '{}': name is not a string literal",
            kind.label()
        ));
    };

    match kind {
        DynamicKind::DunderImport => {
            // `__import__(name, globals=None, locals=None, fromlist=[], level=0)`
            let fromlist_expr = positional_or_kwarg(call, 3, "fromlist");
            let level_expr = positional_or_kwarg(call, 4, "level");
            let fromlist = match fromlist_expr {
                None => Vec::new(),
                Some(expr) => match string_literal_list(expr) {
                    Some(list) => list,
                    None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': fromlist is not a literal \
                             list/tuple of strings (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            let explicit_level = match level_expr {
                None => None,
                Some(expr) => match int_literal(expr) {
                    Some(n) if n >= 0 => Some(n as u32),
                    Some(_) | None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': level is not an int literal \
                             (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            DynamicParseResult::Ok {
                name: name.to_string(),
                fromlist,
                explicit_package: None,
                explicit_level,
            }
        }
        DynamicKind::ImportlibImportModule => {
            // `importlib.import_module(name, package=None)`
            let pkg_expr = positional_or_kwarg(call, 1, "package");
            let explicit_package = match pkg_expr {
                None => None,
                Some(expr) => match string_literal(expr) {
                    Some(s) => Some(s),
                    None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': package is not a string literal \
                             (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            DynamicParseResult::Ok {
                name: name.to_string(),
                fromlist: Vec::new(),
                explicit_package,
                explicit_level: None,
            }
        }
    }
}

fn first_positional(call: &ruff_python_ast::ExprCall) -> Option<&Expr> {
    call.arguments.args.first()
}

fn positional_or_kwarg<'a>(
    call: &'a ruff_python_ast::ExprCall,
    idx: usize,
    kwarg_name: &str,
) -> Option<&'a Expr> {
    if let Some(expr) = call.arguments.args.get(idx) {
        return Some(expr);
    }
    call.arguments
        .keywords
        .iter()
        .find(|k| k.arg.as_ref().is_some_and(|id| id.as_str() == kwarg_name))
        .map(|k| &k.value)
}

fn string_literal(expr: &Expr) -> Option<&str> {
    if let Expr::StringLiteral(s) = expr {
        Some(s.value.to_str())
    } else {
        None
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    let Expr::NumberLiteral(n) = expr else {
        return None;
    };
    if let ruff_python_ast::Number::Int(i) = &n.value {
        i.as_i64()
    } else {
        None
    }
}

fn string_literal_list(expr: &Expr) -> Option<Vec<&str>> {
    let elements: &[Expr] = match expr {
        Expr::List(l) => &l.elts,
        Expr::Tuple(t) => &t.elts,
        _ => return None,
    };
    let mut out = Vec::with_capacity(elements.len());
    for elem in elements {
        out.push(string_literal(elem)?);
    }
    Some(out)
}

/// Resolve a dynamic-import `name` against its level / package
/// context to an absolute module name, or return an error message
/// suitable for logging.
///
/// Per CPython semantics:
/// * `__import__`: leading dots in `name` are invalid (level is the
///   `level=` keyword/positional arg); `level=N` means "start from
///   the current package and go up N-1 levels."
/// * `importlib.import_module`: leading dots in `name` are the level
///   (count them); `level=1` means current package, `level=2` parent,
///   etc. `package=` is optional and defaults to the caller's
///   package — that's what `pkg` is here.
fn resolve_dynamic_target(
    kind: DynamicKind,
    name: &str,
    explicit_level: Option<u32>,
    pkg: Option<&str>,
) -> Result<String, String> {
    let (name_no_dots, level): (&str, u32) = match kind {
        DynamicKind::DunderImport => {
            if name.starts_with('.') {
                return Err(format!(
                    "Skipping dynamic import '{}': leading dots are invalid for \
                     __import__ (name: '{name}')",
                    kind.label()
                ));
            }
            (name, explicit_level.unwrap_or(0))
        }
        DynamicKind::ImportlibImportModule => {
            let dots = name.chars().take_while(|c| *c == '.').count();
            (&name[dots..], dots as u32)
        }
    };

    if level == 0 {
        return Ok(name_no_dots.to_string());
    }
    let Some(pkg) = pkg else {
        return Err(format!(
            "Skipping dynamic import '{}': relative import '{name}' has no package context",
            kind.label()
        ));
    };
    let segments: Vec<&str> = pkg.split('.').filter(|s| !s.is_empty()).collect();
    let levels_up = (level - 1) as usize;
    if levels_up > segments.len() {
        return Err(format!(
            "Skipping dynamic import '{}': relative import '{name}' goes beyond top-level \
             package '{pkg}'",
            kind.label()
        ));
    }
    let base = &segments[..segments.len() - levels_up];
    let mut parts: Vec<&str> = base.to_vec();
    if !name_no_dots.is_empty() {
        parts.push(name_no_dots);
    }
    Ok(parts.join("."))
}

/// Package name (i.e. enclosing package) of `file`. For
/// `pkg/__init__.py` this is `"pkg"`; for `pkg/sub.py` this is
/// `"pkg"`; for a top-level `mod.py` this is `None`.
fn file_package_name(db: &dyn ty_python_semantic::Db, file: File) -> Option<String> {
    let module = file_to_module(db, file)?;
    let name = module.name(db);
    let path_str = match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    };
    let is_init = path_str.ends_with("/__init__.py") || path_str.ends_with("\\__init__.py");
    if is_init {
        Some(name.as_str().to_string())
    } else {
        name.parent().map(|n| n.as_str().to_string())
    }
}

/// Send a warning to the `dead_cst._visitor` logger so the
/// `visitor_warnings` test fixture (pytest caplog scoped to that
/// logger) can observe it.
fn emit_visitor_warning(py: Python<'_>, message: &str) {
    let _ = (|| -> PyResult<()> {
        let logging = py.import_bound("logging")?;
        let logger = logging.call_method1("getLogger", ("dead_cst._visitor",))?;
        logger.call_method1("warning", (message,))?;
        Ok(())
    })();
}

/// Peel an attribute chain back to its `Name` root.
///
/// Returns `Some((root, segments))` when `expr` is `Name`, `Name.s1`,
/// `Name.s1.s2`, ... — i.e. one or more attribute accesses on a bare
/// name. Returns `None` when the chain bottoms out at anything else
/// (a call result, subscript, attribute of attribute of a non-Name, …).
fn collapse_attribute_chain(expr: &Expr) -> Option<(&ExprName, Vec<&str>)> {
    let mut segments: Vec<&str> = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Attribute(attr) => {
                segments.push(attr.attr.as_str());
                current = &attr.value;
            }
            Expr::Name(n) => {
                segments.reverse();
                return Some((n, segments));
            }
            _ => return None,
        }
    }
}

/// True iff `dotted` resolves to *some* module (project, stdlib, or
/// third-party) as seen from `anchor`. Used to disambiguate
/// "submodule" vs "decl in module" for `from X import Y`.
fn module_name_resolves(dotted: &str, anchor: File, db: &dyn ty_python_semantic::Db) -> bool {
    ModuleName::new(dotted)
        .and_then(|n| resolve_module(db, anchor, &n))
        .is_some()
}

impl<'ast, 'db> Visitor<'ast> for RefCollector<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Name(n) = expr {
            self.emit_name_use(n, &[]);
            return;
        }
        if let Expr::Named(named) = expr {
            // Walrus `(y := expr)` at module scope has its own
            // ``DefinitionKind::NamedExpression`` entry in ty's global
            // scope. ``walk_owned`` walks the inner expression and
            // attributes uses to `y`; walking it again here would
            // double-attribute every reference (once to the walrus
            // target, once to whatever owns the enclosing expression).
            // Skip into the walrus and leave its body to its own walk.
            //
            // Inside function / class bodies (``nested_context``), the
            // walrus's Definition lives in the nested scope and isn't
            // covered by ``ingest_top_level``'s per-def loop, so we
            // still need to walk the inner value here — attributing to
            // the enclosing top-level decl.
            if !self.nested_context {
                return;
            }
            self.visit_expr(&named.value);
            return;
        }
        if matches!(expr, Expr::Attribute(_)) {
            if let Some((root, segments)) = collapse_attribute_chain(expr) {
                self.emit_name_use(root, &segments);
                // Don't recurse into the chain — every Name in it has
                // been handled and an attribute access has no other
                // walk-worthy children.
                return;
            }
        }
        if let Expr::Call(call) = expr {
            if self.try_emit_dynamic_import(call) {
                // For `importlib.import_module(...)`, attribute the
                // `importlib` receiver to the call's owner so the
                // alias keeps a use edge — without it the
                // module-level call would leave `p.x.importlib`
                // orphaned from the call site. The chain segment
                // (`import_module`) isn't a walkable target, so we
                // pass an empty extra-chain rather than walking the
                // attribute via `collapse_attribute_chain`.
                if let Expr::Attribute(attr) = &*call.func {
                    if let Expr::Name(receiver) = &*attr.value {
                        self.emit_name_use(receiver, &[]);
                    }
                }
                // Walk arguments for any nested non-string Names
                // (e.g. `__import__(name, fromlist=names)` where
                // `name` and `names` should still emit normal use
                // edges that attribute the *receiver* of those
                // values to the owner).
                for arg in &call.arguments.args {
                    self.visit_expr(arg);
                }
                for kw in &call.arguments.keywords {
                    self.visit_expr(&kw.value);
                }
                return;
            }
        }
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        if self.nested_context {
            match stmt {
                Stmt::Import(s) => {
                    self.emit_nested_import(s);
                    return;
                }
                Stmt::ImportFrom(s) => {
                    self.emit_nested_import_from(s);
                    return;
                }
                _ => {}
            }
        } else if stmt_creates_top_level_definition(stmt) {
            // At module-level walks (or per-def walks of value
            // expressions, where Stmt nodes wouldn't appear anyway),
            // nested top-level-definition statements have already been
            // processed by the per-def pass with their own owner.
            // Skipping them here prevents double-attribution when a
            // compound statement (`if` / `for` / `try` / `with` /
            // `match`) at module scope contains a definition in its
            // body — e.g. `if True: f = g` would otherwise emit
            // `mod -> g` *and* the proper `mod.f -> g`.
            return;
        }
        walk_stmt(self, stmt);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rel_path<P: AsRef<str>>(path: P) -> RelativePathBuf {
    RelativePathBuf::cli(SystemPath::new(path.as_ref()))
}

// ---------------------------------------------------------------------------
// Noqa parsing
// ---------------------------------------------------------------------------

/// Per-source-type default flags for ``.ipynb``: notebook decls are
/// always alive (cells run top-to-bottom, not imported) and the
/// codemod must skip them (it can't rewrite the cell JSON envelope).
const NODE_FLAGS_NOTEBOOK_DEFAULT: u32 = NodeFlags::ENTRYPOINT | NodeFlags::NOTEBOOK;

/// Bits stamped on every import alias pinned by a noqa directive — both
/// `ENTRYPOINT` (so reachability keeps it alive) and `NOQA` (so the
/// `kept_alive_by_flags_only(NOQA)` blast-radius query can find it).
const NODE_FLAGS_NOQA_PIN: u32 = NodeFlags::ENTRYPOINT | NodeFlags::NOQA;

/// Internal aliases for the pyclass classattrs used by the call sites
/// scattered through this file. Read as bare constants rather than
/// `NodeFlags::ENTRYPOINT` (which would force every reader to chase the
/// `pyclass` macro to decide whether it's a runtime lookup or a const).
const NODE_FLAG_ENTRYPOINT: u32 = NodeFlags::ENTRYPOINT;
const EDGE_FLAG_DEAD_BRANCH: u32 = EdgeFlags::DEAD_BRANCH;
const EDGE_FLAG_DYNAMIC_IMPORT: u32 = EdgeFlags::DYNAMIC_IMPORT;

/// Bit values stamped into [`NativeNode::flags`]. Mirrors
/// `dead_cst.graph.NodeFlags` exactly so plugin code can mix
/// rust-emitted and libcst-emitted nodes.
///
/// Exposed as Python class attributes — the classattr values are plain
/// `int`s so `NodeFlags.ENTRYPOINT | NodeFlags.NOQA` works the same as
/// the libcst `IntFlag` (just without the `.name` / `.value` surface).
#[pyclass(frozen)]
struct NodeFlags;

#[pymethods]
impl NodeFlags {
    #[classattr]
    const NONE: u32 = 0;
    /// Decl rebound by a later assignment in the same file. Kept in the
    /// graph (with its parent-module edge) but excluded from the
    /// cross-module lookup so consumers of an exported name route to
    /// the live binding.
    #[classattr]
    const SHADOWED: u32 = 1;
    /// Reachability seed. BFS for "what's live" starts from every node
    /// carrying this bit.
    #[classattr]
    const ENTRYPOINT: u32 = 2;
    /// `typing.overload` stub (or any same-name decl whose lifetime is
    /// anchored to a matching impl). Excluded from the lookup trie like
    /// `SHADOWED`; kept alive by an explicit `impl -> overload` edge.
    #[classattr]
    const OVERLOAD: u32 = 4;
    /// Tags an entrypoint as test-only (pytest / unittest fixtures and
    /// test methods). Layered on top of `ENTRYPOINT` so the
    /// `kept_alive_by_flags_only(TESTCASE)` query can ask "what's only
    /// alive because of tests".
    #[classattr]
    const TESTCASE: u32 = 8;
    /// Tags an entrypoint as preserved by an explicit user noqa
    /// directive (bare `# noqa`, `# noqa: F401`, multi-rule
    /// `# noqa: E501, F401`, or the file-level `# ruff: noqa` /
    /// `# flake8: noqa`).
    #[classattr]
    const NOQA: u32 = 16;
    /// Every node sourced from a Jupyter `.ipynb` file. Combined with
    /// `ENTRYPOINT` via `NODE_FLAGS_NOTEBOOK_DEFAULT` because cells run
    /// top-to-bottom rather than being imported, and the codemod skips
    /// notebook nodes (it can't rewrite the cell JSON envelope).
    #[classattr]
    const NOTEBOOK: u32 = 32;
    /// Every node sourced from a file under the package's `exported`
    /// glob. Used by the cross-package merge to filter to entries the
    /// owning package opts into exposing.
    #[classattr]
    const EXPORTED: u32 = 64;
    /// Import decl synthesized from `from X import *` — one per name
    /// the star statement brought in. Set so the cross-module trie can
    /// distinguish "real" import aliases from per-name star fan-out.
    #[classattr]
    const STAR_REEXPORT: u32 = 128;
}

/// Bit values stamped into the third tuple slot of each `NativeGraph`
/// edge. Mirrors `dead_cst.graph.EdgeFlags`.
#[pyclass(frozen)]
struct EdgeFlags;

#[pymethods]
impl EdgeFlags {
    #[classattr]
    const NONE: u32 = 0;
    /// Reference originated inside a statically-dead region (the body of
    /// `if False:`, the else of `if True:`, after an unconditional
    /// `return` / `raise` / `break` / `continue`, …). Metadata only —
    /// the edge still participates in default reachability; pass to
    /// `descendants(..., skip_flags=EdgeFlags.DEAD_BRANCH)` to compute
    /// the kept-alive-only-by-dead-branches set.
    #[classattr]
    const DEAD_BRANCH: u32 = 1;
    /// Edge emitted from a runtime-import call (`__import__('X')` /
    /// `importlib.import_module('X')`). Lets plugins read which edges
    /// the visitor produced from dynamic-import shapes and choose to
    /// fan out / specialize.
    #[classattr]
    const DYNAMIC_IMPORT: u32 = 2;
}

// ---------------------------------------------------------------------------
// Dead-region detection
// ---------------------------------------------------------------------------

/// Collect the source ranges of every statically-dead statement in
/// `parsed` — bodies of `if False` / `else` of `if True` / `while False`
/// / `while True else` branches, statements after an unconditional
/// terminator (`return` / `raise` / `break` / `continue` /
/// `assert <falsy>`), and the suites of compound terminators (an
/// `if/elif/else` whose every branch terminates, a `try` whose body
/// and every handler terminate, a `with` whose body terminates).
///
/// The result is a flat `Vec<TextRange>`; callers use
/// `TextRange::contains_range` to test whether a use site sits
/// inside any of them.
/// Per-scope name → known-truthiness map. Populated by
/// `build_scope_table` from the scope's literal assignments,
/// annotated assignments, and walrus expressions. Function and class
/// bodies inherit and override their enclosing scope's table.
type NameTable = HashMap<String, bool>;

fn detect_dead_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut dead = Vec::new();
    let empty: NameTable = HashMap::new();
    let module_table = build_scope_table(&parsed.syntax().body, &empty);
    walk_suite_for_dead(&parsed.syntax().body, &module_table, &mut dead);
    dead
}

fn walk_suite_for_dead(stmts: &[Stmt], table: &NameTable, dead: &mut Vec<TextRange>) {
    let mut killed = false;
    for stmt in stmts {
        if killed {
            // Past an in-suite terminator. Mark this stmt's range as
            // dead and don't recurse — descendant uses are covered by
            // the outer range.
            dead.push(stmt.range());
            continue;
        }
        walk_compound_for_dead(stmt, table, dead);
        if stmt_is_terminator(stmt, table) {
            killed = true;
        }
    }
}

fn walk_compound_for_dead(stmt: &Stmt, table: &NameTable, dead: &mut Vec<TextRange>) {
    match stmt {
        Stmt::If(if_stmt) => {
            let mut taken = false;
            match evaluate_truthiness(&if_stmt.test, table) {
                Some(true) => {
                    walk_suite_for_dead(&if_stmt.body, table, dead);
                    taken = true;
                }
                Some(false) => {
                    for s in &if_stmt.body {
                        dead.push(s.range());
                    }
                }
                None => {
                    walk_suite_for_dead(&if_stmt.body, table, dead);
                }
            }
            for clause in &if_stmt.elif_else_clauses {
                if taken {
                    dead.push(clause.range);
                    continue;
                }
                match &clause.test {
                    Some(test) => match evaluate_truthiness(test, table) {
                        Some(true) => {
                            walk_suite_for_dead(&clause.body, table, dead);
                            taken = true;
                        }
                        Some(false) => {
                            for s in &clause.body {
                                dead.push(s.range());
                            }
                        }
                        None => walk_suite_for_dead(&clause.body, table, dead),
                    },
                    None => walk_suite_for_dead(&clause.body, table, dead),
                }
            }
        }
        Stmt::While(w) => match evaluate_truthiness(&w.test, table) {
            Some(false) => {
                for s in &w.body {
                    dead.push(s.range());
                }
                walk_suite_for_dead(&w.orelse, table, dead);
            }
            Some(true) => {
                walk_suite_for_dead(&w.body, table, dead);
                for s in &w.orelse {
                    dead.push(s.range());
                }
            }
            None => {
                walk_suite_for_dead(&w.body, table, dead);
                walk_suite_for_dead(&w.orelse, table, dead);
            }
        },
        Stmt::For(f) => {
            walk_suite_for_dead(&f.body, table, dead);
            walk_suite_for_dead(&f.orelse, table, dead);
        }
        Stmt::With(w) => walk_suite_for_dead(&w.body, table, dead),
        Stmt::Try(t) => {
            walk_suite_for_dead(&t.body, table, dead);
            for handler in &t.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
                walk_suite_for_dead(&h.body, table, dead);
            }
            walk_suite_for_dead(&t.orelse, table, dead);
            walk_suite_for_dead(&t.finalbody, table, dead);
        }
        Stmt::FunctionDef(f) => {
            let nested = build_scope_table(&f.body, table);
            walk_suite_for_dead(&f.body, &nested, dead);
        }
        Stmt::ClassDef(c) => {
            let nested = build_scope_table(&c.body, table);
            walk_suite_for_dead(&c.body, &nested, dead);
        }
        Stmt::Match(m) => {
            for case in &m.cases {
                walk_suite_for_dead(&case.body, table, dead);
            }
        }
        _ => {}
    }
}

/// Does `stmt` unconditionally hand off control out of the enclosing
/// suite? Statements *after* a terminator in the same suite are dead.
///
/// Bare terminators are `return` / `raise` / `break` / `continue`
/// plus `assert <falsy>`. Compound forms (`if/elif/else`, `try`,
/// `with`, `match`) are terminators when every reachable path inside
/// them ends in one.
fn stmt_is_terminator(stmt: &Stmt, table: &NameTable) -> bool {
    match stmt {
        Stmt::Return(_) | Stmt::Raise(_) | Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::Assert(a) => evaluate_truthiness(&a.test, table) == Some(false),
        Stmt::If(if_stmt) => {
            // A constant-truthy `if` is unconditional: it's a
            // terminator iff its body terminates, with or without
            // any else (`if True: return; …` kills the rest of
            // the enclosing suite).
            if evaluate_truthiness(&if_stmt.test, table) == Some(true) {
                return suite_terminates(&if_stmt.body, table);
            }
            // Otherwise we need an else for completeness — without
            // one, the `if` (or every `elif`) might not fire and the
            // suite continues normally.
            let has_else = if_stmt
                .elif_else_clauses
                .last()
                .is_some_and(|c| c.test.is_none());
            if !has_else {
                return false;
            }
            let test = evaluate_truthiness(&if_stmt.test, table);
            let body_term = suite_terminates(&if_stmt.body, table) || test == Some(false);
            let clauses_term = if_stmt
                .elif_else_clauses
                .iter()
                .all(|c| suite_terminates(&c.body, table));
            body_term && clauses_term
        }
        Stmt::With(w) => suite_terminates(&w.body, table),
        Stmt::Try(t) => {
            if suite_terminates(&t.finalbody, table) {
                return true;
            }
            suite_terminates(&t.body, table)
                && t.handlers.iter().all(|h| {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(eh) = h;
                    suite_terminates(&eh.body, table)
                })
        }
        _ => false,
    }
}

fn suite_terminates(stmts: &[Stmt], table: &NameTable) -> bool {
    stmts.iter().any(|s| stmt_is_terminator(s, table))
}

/// Constant-fold the truthiness of `expr`. Returns `None` when the
/// expression doesn't reduce to a known value — names not in `table`,
/// calls, attribute access, comparisons, etc.
///
/// `BoolOp` (`and` / `or`) short-circuits the way Python does:
/// `True or anything` is `True` even when "anything" is unknown.
/// `not` flips a known truth value through `UnaryOp`.
fn evaluate_truthiness(expr: &Expr, table: &NameTable) -> Option<bool> {
    match expr {
        Expr::BooleanLiteral(b) => Some(b.value),
        Expr::NoneLiteral(_) => Some(false),
        Expr::EllipsisLiteral(_) => Some(true),
        Expr::NumberLiteral(n) => match &n.value {
            ruff_python_ast::Number::Int(i) => i.as_i64().map(|x| x != 0),
            ruff_python_ast::Number::Float(f) => Some(*f != 0.0),
            ruff_python_ast::Number::Complex { real, imag } => Some(*real != 0.0 || *imag != 0.0),
        },
        Expr::StringLiteral(s) => Some(!s.value.is_empty()),
        Expr::BytesLiteral(b) => Some(!b.value.is_empty()),
        Expr::List(l) => Some(!l.elts.is_empty()),
        Expr::Tuple(t) => Some(!t.elts.is_empty()),
        Expr::Set(s) => Some(!s.elts.is_empty()),
        Expr::Dict(d) => Some(!d.items.is_empty()),
        Expr::Name(n) => table.get(n.id.as_str()).copied(),
        Expr::Named(named) => evaluate_truthiness(&named.value, table),
        Expr::BoolOp(b) => match b.op {
            ruff_python_ast::BoolOp::Or => {
                let mut any_unknown = false;
                for v in &b.values {
                    match evaluate_truthiness(v, table) {
                        Some(true) => return Some(true),
                        Some(false) => continue,
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(false)
                }
            }
            ruff_python_ast::BoolOp::And => {
                let mut any_unknown = false;
                for v in &b.values {
                    match evaluate_truthiness(v, table) {
                        Some(false) => return Some(false),
                        Some(true) => continue,
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(true)
                }
            }
        },
        Expr::UnaryOp(u) => {
            if matches!(u.op, ruff_python_ast::UnaryOp::Not) {
                evaluate_truthiness(&u.operand, table).map(|v| !v)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build the scope's name→truthiness table.
///
/// Collects all unconditional assignments at the scope's top level
/// (`X = literal`, `X: T = literal`) plus every walrus binding the
/// scope's expressions contain (PEP 572 leaks walrus bindings to the
/// enclosing function/module). Iterates to fixed point so that
/// chained constants — `foo = False; bar = foo or False; baz = bar`
/// — fold through the chain.
///
/// Conservative on conflicts: a name with multiple disagreeing
/// bindings (`X = True` and `X = False` both at top level), or any
/// binding whose RHS we can't fold, drops out of the table. That
/// matches the libcst pipeline's "only fold when every binding
/// agrees" rule and is what keeps the `does_not_fold_*` tests honest.
fn build_scope_table(stmts: &[Stmt], enclosing: &NameTable) -> NameTable {
    let mut table = enclosing.clone();
    let mut bindings: HashMap<String, Vec<&Expr>> = HashMap::new();
    collect_scope_bindings(stmts, &mut bindings);

    let mut changed = true;
    while changed {
        changed = false;
        for (name, exprs) in &bindings {
            let values: Vec<Option<bool>> = exprs
                .iter()
                .map(|e| evaluate_truthiness(e, &table))
                .collect();
            let resolved = if let Some(first) = values.first().copied() {
                if values.iter().all(|v| *v == first) {
                    first
                } else {
                    None
                }
            } else {
                None
            };
            match (resolved, table.get(name).copied()) {
                (Some(v), prev) if prev != Some(v) => {
                    table.insert(name.clone(), v);
                    changed = true;
                }
                (None, Some(_)) => {
                    // Previously folded, now conflicting / unknown:
                    // drop it. Don't fall back to the enclosing
                    // scope's value because a same-name binding in
                    // this scope shadows it.
                    table.remove(name);
                    changed = true;
                }
                _ => {}
            }
        }
    }
    table
}

/// Top-level `name → rhs-expression` bindings for the scope, plus
/// any walrus bindings the scope's *expressions* contain (PEP 572
/// leaks walruses to the enclosing function/module). Does NOT
/// recurse into nested function/class bodies — those are separate
/// scopes with their own tables.
fn collect_scope_bindings<'a>(stmts: &'a [Stmt], out: &mut HashMap<String, Vec<&'a Expr>>) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign(a) => {
                if let [Expr::Name(target)] = a.targets.as_slice() {
                    out.entry(target.id.as_str().to_string())
                        .or_default()
                        .push(&a.value);
                }
                collect_walrus_in_expr(&a.value, out);
            }
            Stmt::AnnAssign(a) => {
                if let (Expr::Name(target), Some(value)) = (&*a.target, &a.value) {
                    out.entry(target.id.as_str().to_string())
                        .or_default()
                        .push(value);
                }
                if let Some(value) = &a.value {
                    collect_walrus_in_expr(value, out);
                }
                collect_walrus_in_expr(&a.annotation, out);
            }
            Stmt::Expr(e) => {
                collect_walrus_in_expr(&e.value, out);
            }
            Stmt::If(if_stmt) => {
                collect_walrus_in_expr(&if_stmt.test, out);
            }
            Stmt::While(w) => {
                collect_walrus_in_expr(&w.test, out);
            }
            Stmt::Return(r) => {
                if let Some(value) = &r.value {
                    collect_walrus_in_expr(value, out);
                }
            }
            // Nested function/class bodies are separate scopes —
            // don't drag their bindings into this table.
            Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
            _ => {}
        }
    }
}

/// Scan `expr` for walrus (`:=`) targets that bind names in the
/// enclosing scope.
fn collect_walrus_in_expr<'a>(expr: &'a Expr, out: &mut HashMap<String, Vec<&'a Expr>>) {
    match expr {
        Expr::Named(named) => {
            if let Expr::Name(target) = &*named.target {
                out.entry(target.id.as_str().to_string())
                    .or_default()
                    .push(&named.value);
            }
            collect_walrus_in_expr(&named.value, out);
        }
        Expr::BoolOp(b) => {
            for v in &b.values {
                collect_walrus_in_expr(v, out);
            }
        }
        Expr::UnaryOp(u) => collect_walrus_in_expr(&u.operand, out),
        Expr::BinOp(b) => {
            collect_walrus_in_expr(&b.left, out);
            collect_walrus_in_expr(&b.right, out);
        }
        Expr::Compare(c) => {
            collect_walrus_in_expr(&c.left, out);
            for v in &c.comparators {
                collect_walrus_in_expr(v, out);
            }
        }
        _ => {}
    }
}

/// Result of parsing the `noqa[: codes…]` tail of a comment.
///
/// `Bare` means the directive carried no code list and so suppresses
/// every rule (including F401); `F401Present` means F401 is in the
/// comma-separated rule list; `OtherOnly` means a rule list was given
/// but F401 was absent. The first two pin the import alive; the third
/// doesn't.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoqaKind {
    Bare,
    F401Present,
    OtherOnly,
}

impl NoqaKind {
    fn pins_f401(self) -> bool {
        matches!(self, NoqaKind::Bare | NoqaKind::F401Present)
    }
}

/// Parse a `noqa[: codes…]` tail from `content`. Returns `None` if
/// `content` doesn't start with `noqa` (case-insensitive) after
/// optional whitespace.
fn parse_noqa_tail(content: &str) -> Option<NoqaKind> {
    let trimmed = content.trim_start();
    // `str::get` short-circuits when byte 4 isn't a char boundary —
    // e.g. a comment that starts with the 3-byte `─` box-drawing char
    // (common in section banners). Direct slicing would panic.
    let prefix = trimmed.get(..4)?;
    if !prefix.eq_ignore_ascii_case("noqa") {
        return None;
    }
    let after_noqa = &trimmed[4..];
    let after_colon = after_noqa.trim_start();
    let Some(rest) = after_colon.strip_prefix(':') else {
        // Bare `noqa` with no code list.
        return Some(NoqaKind::Bare);
    };
    let rules = rest.trim();
    if rules.is_empty() {
        return Some(NoqaKind::Bare);
    }
    let has_f401 = rules
        .split(',')
        .any(|code| code.trim().eq_ignore_ascii_case("F401"));
    if has_f401 {
        Some(NoqaKind::F401Present)
    } else {
        Some(NoqaKind::OtherOnly)
    }
}

/// Returns `true` when `comment_body` (the text *after* the leading
/// `#`) is a per-line `# noqa` directive that silences F401.
fn is_per_line_pin(comment_body: &str) -> bool {
    parse_noqa_tail(comment_body)
        .map(NoqaKind::pins_f401)
        .unwrap_or(false)
}

/// Returns `true` when `comment_body` is a *file-level*
/// `# ruff: noqa` / `# flake8: noqa` directive that silences F401.
///
/// The `ruff:` / `flake8:` prefix is matched case-sensitively (per
/// ruff's documented behavior); the `noqa` keyword itself is
/// case-insensitive.
fn is_file_pin(comment_body: &str) -> bool {
    let trimmed = comment_body.trim_start();
    let after_prefix = trimmed
        .strip_prefix("ruff:")
        .or_else(|| trimmed.strip_prefix("flake8:"));
    let Some(after_prefix) = after_prefix else {
        return false;
    };
    parse_noqa_tail(after_prefix)
        .map(NoqaKind::pins_f401)
        .unwrap_or(false)
}

/// Scan every Comment token in `parsed` and partition the file's
/// noqa directives into `(file_pinned, per_line_pins)`.
///
/// `file_pinned` is true when *any* comment in the file is a
/// file-level `# ruff: noqa` / `# flake8: noqa` that silences F401 —
/// ruff scans the whole source, not just the header.
/// `per_line_pins` collects the (1-indexed) line numbers carrying a
/// `# noqa[: …F401…]` per-line directive; an import alias on one of
/// those lines is pinned individually.
fn scan_noqa_directives(
    parsed: &ParsedModuleRef,
    source: &str,
    line_index: &LineIndex,
) -> (bool, HashSet<usize>) {
    let mut file_pinned = false;
    let mut per_line_pins: HashSet<usize> = HashSet::new();
    for token in parsed.tokens().iter() {
        if token.kind() != TokenKind::Comment {
            continue;
        }
        let range = token.range();
        let text = &source[range];
        let Some(body) = text.strip_prefix('#') else {
            continue;
        };
        if is_file_pin(body) {
            file_pinned = true;
        }
        if is_per_line_pin(body) {
            let line = line_index.line_column(range.start(), source).line.get();
            per_line_pins.insert(line);
        }
    }
    (file_pinned, per_line_pins)
}

fn position(index: &LineIndex, source: &str, range: TextRange) -> (usize, usize, usize, usize) {
    let start = index.line_column(range.start(), source);
    let end = index.line_column(range.end(), source);
    (
        start.line.get(),
        start.column.get() - 1,
        end.line.get(),
        end.column.get() - 1,
    )
}

fn range_key(range: TextRange) -> (u32, u32) {
    (range.start().to_u32(), range.end().to_u32())
}

fn file_path_string(db: &dyn ProjectDb, file: File) -> String {
    match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    }
}

/// Per-source-type default flag mask ORed into every node minted from
/// this file. Only ``.ipynb`` opts in today — cells are inherently
/// entrypoints (run top-to-bottom, not imported) and the codemod must
/// skip the JSON envelope. ``.pyi`` decls get per-decl flagging in
/// ``ingest_decls`` (the flag depends on whether a matching ``.py``
/// runtime decl exists), so the file-level helper returns ``0`` for
/// them.
fn file_default_flags(db: &dyn ProjectDb, file: File) -> u32 {
    if file_path_string(db, file).ends_with(".ipynb") {
        NODE_FLAGS_NOTEBOOK_DEFAULT
    } else {
        0
    }
}

fn module_fqname_for_file(db: &dyn ProjectDb, file: File) -> String {
    if let Some(module) = file_to_module(db, file) {
        return module.name(db).as_str().to_string();
    }
    // Fallback for files ty doesn't classify (e.g. ``gunicorn.conf.py`` at
    // the project root — there's no ``__init__.py`` and the stem contains
    // a dot, so ty's module resolver returns ``None``). Mirror the libcst
    // pipeline's ``strip .py / replace / with .`` derivation so plugins
    // like ``ServerConfigPlugin`` can address the file by its conventional
    // dotted name.
    let path_str = file_path_string(db, file);
    let root_str = db.project().root(db).as_str();
    let rel = path_str
        .strip_prefix(root_str)
        .unwrap_or(&path_str)
        .trim_start_matches('/')
        .trim_start_matches('\\');
    let stem = rel
        .strip_suffix(".pyi")
        .or_else(|| rel.strip_suffix(".py"))
        .unwrap_or(rel);
    stem.replace(['/', '\\'], ".")
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// Module-level alias for ``ctx.query()`` — exists for the ergonomic
/// ``from dead_cst_ty_native import query; query(ctx).decorators()...``
/// idiom that the plugins rely on.
#[pyfunction]
fn query(slf: Py<ProjectContext>, _py: Python<'_>) -> QueryBuilder {
    QueryBuilder { ctx: slf }
}

#[pymodule]
fn dead_cst_ty_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Import>()?;
    m.add_class::<NativeNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<Project>()?;
    m.add_class::<ProjectContext>()?;
    m.add_class::<AddEdge>()?;
    m.add_class::<AddEntrypoint>()?;
    m.add_class::<AddNode>()?;
    m.add_class::<NodeFlags>()?;
    m.add_class::<EdgeFlags>()?;
    m.add_class::<DecoratorRef>()?;
    m.add_class::<ConstructionRef>()?;
    m.add_class::<CallRef>()?;
    m.add_class::<QueryBuilder>()?;
    m.add_class::<DecoratorQuery>()?;
    m.add_class::<ConstructionQuery>()?;
    m.add_class::<CallQuery>()?;
    m.add_function(wrap_pyfunction!(query, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_noqa_tail_multibyte_prefix_does_not_panic() {
        // Regression: byte-slicing `trimmed[..4]` panicked when the
        // comment started with a multi-byte UTF-8 char (e.g. the
        // 3-byte `─` box-drawing char in section banners).
        assert_eq!(parse_noqa_tail("── Top-level model (dataclass) ──"), None,);
        assert_eq!(parse_noqa_tail("─"), None);
        assert_eq!(parse_noqa_tail("héllo"), None);
        assert_eq!(parse_noqa_tail("🙂🙂"), None);
    }

    #[test]
    fn parse_noqa_tail_short_content_returns_none() {
        assert_eq!(parse_noqa_tail(""), None);
        assert_eq!(parse_noqa_tail("no"), None);
        assert_eq!(parse_noqa_tail("noq"), None);
    }

    #[test]
    fn parse_noqa_tail_recognizes_bare_directive() {
        assert_eq!(parse_noqa_tail(" noqa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("NOQA"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("noqa: "), Some(NoqaKind::Bare));
    }

    #[test]
    fn parse_noqa_tail_recognizes_f401() {
        assert_eq!(parse_noqa_tail("noqa: F401"), Some(NoqaKind::F401Present));
        assert_eq!(
            parse_noqa_tail("noqa: E501, F401"),
            Some(NoqaKind::F401Present),
        );
        assert_eq!(parse_noqa_tail("noqa: E501"), Some(NoqaKind::OtherOnly));
    }
}
