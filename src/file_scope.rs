//! Per-file plugin infrastructure: `FileScope` query handle, per-file
//! graph-op pyclasses, salsa-tracked plugin runner, and the assemble-
//! time fan-in that folds per-file output into the global graph.
//!
//! ## Why pre-assemble?
//!
//! Per-file plugins run during the parallel populate phase, BEFORE
//! the global graph is assembled. There are no global indices yet —
//! the plugin sees only file-local `u32` handles into the file's
//! [`crate::file_payload::FileNodes`] payload. The output of every
//! file's plugin run is salsa-tracked ([`file_to_plugin_ops`]) so
//! editing one file only re-runs plugins for THAT file. The assemble
//! pass walks every file's cached ops and translates the file-local
//! indices to global indices via the same `ref_to_global` map that
//! folds in `file_to_edges` output.
//!
//! ## What plugins see
//!
//! [`FileScope`] is pre-computed once per file before any plugin runs:
//! main-block position, dunder decl indices, and a `module -> [import
//! idx]` map. Plugins consume these via small Python methods (returning
//! plain `int` lists), then yield [`PerFileEdge`] / [`PerFileEntrypoint`]
//! / [`PerFileNode`] ops that reference the same file-local indices.
//! That keeps the Python surface narrow — a per-file plugin literally
//! cannot reach cross-file state.
//!
//! ## Plugin registry
//!
//! The salsa-tracked function can't be parameterised on Python plugin
//! instances (PyObject isn't `Hash + Eq`), so per-file plugins are
//! installed into a process-global registry by [`install_per_file_plugins`]
//! at the top of `materialize()` and cleared by
//! [`clear_per_file_plugins`] at the end. Per the design discussion:
//! plugin code is static program code — the same plugins are present
//! on every materialize() call — so this trades clean composability
//! for a much simpler salsa cache key (just `(file,)`).

use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use rustc_hash::FxHashMap;
use ty_project::Db as ProjectDb;

use crate::builder::{synthetic_node, GraphBuilder};
use crate::file_payload::{file_to_nodes, FileNodes, NodeKind, NodeRef};
use crate::graph::intern_kind;
use crate::helpers::{
    file_path_string, find_main_block_range, is_dunder_name, NODE_FLAG_ENTRYPOINT,
};

// ---------------------------------------------------------------------------
// FileScope: per-file query handle handed to per-file plugins.
// ---------------------------------------------------------------------------

/// Per-file query handle. Fully pre-computed before plugins run, so
/// every method is a pure read on owned data — no `db` access, no
/// lifetime trickery, no GIL juggling beyond the call itself.
///
/// Indices returned by `FileScope` methods are local positions into
/// the file's [`crate::file_payload::FileNodes`] payload. The
/// assemble-time fan-in translates them to global graph indices via
/// the file's `refs[i] -> NodeRef -> global idx` chain.
#[pyclass(module = "dead_cst._native", frozen)]
pub(crate) struct FileScope {
    pub(crate) path_str: String,
    pub(crate) module_fqname: String,
    /// `Some((module_local=0, decl_local_idxs))` if the file has a
    /// top-level `if __name__ == "__main__":` block.
    pub(crate) main_block: Option<(u32, Vec<u32>)>,
    /// Indices of every top-level variable/function decl whose
    /// fqname is a Python dunder name (`__all__`, `__getattr__`,
    /// `__version__`, ...).
    pub(crate) dunder_decls: Vec<u32>,
    /// `module_name -> [import-node local idx]` for every import in
    /// this file. Single hashmap probe gets all `from <m> import ...`
    /// or `import <m>` aliases bound in the file.
    pub(crate) imports_by_module: FxHashMap<String, Vec<u32>>,
}

#[pymethods]
impl FileScope {
    /// Absolute path of the file this scope wraps.
    #[getter]
    fn path(&self) -> &str {
        &self.path_str
    }

    /// Fully-qualified dotted module name of this file.
    #[getter]
    fn module_fqname(&self) -> &str {
        &self.module_fqname
    }

    /// Local index of the synthetic module node in the file's
    /// payload. Always 0 by [`crate::file_payload::file_to_nodes`]
    /// convention — exposed as a property for self-documenting plugin
    /// code.
    #[getter]
    fn module_idx(&self) -> u32 {
        0
    }

    /// `(module_idx, decl_idxs)` for the file's
    /// `if __name__ == "__main__":` block, or `None` when the file
    /// has no such block. `decl_idxs` are the top-level decls whose
    /// binding range falls inside the block range.
    fn main_block(&self) -> Option<(u32, Vec<u32>)> {
        self.main_block.as_ref().map(|(m, d)| (*m, d.clone()))
    }

    /// Local indices of every top-level variable / function decl in
    /// this file whose fqname is a Python dunder name (`__all__`,
    /// `__getattr__`, `__version__`, ...).
    fn dunder_decls(&self) -> Vec<u32> {
        self.dunder_decls.clone()
    }

    /// Local indices of every import node in this file whose
    /// upstream module matches `module_name`. Covers both
    /// `import <m>` and `from <m> import ...` styles.
    fn imports_of(&self, module_name: &str) -> Vec<u32> {
        self.imports_by_module
            .get(module_name)
            .cloned()
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Python-facing per-file op pyclasses.
//
// Same shapes as `AddEdgeByIdx` / `AddEntrypointByIdx` / `AddNodeByIdx`
// in `builder.rs`, but indices are file-local. The fan-in pass
// translates each to its global-indexed counterpart at assemble time.
// ---------------------------------------------------------------------------

/// Add an edge between two file-local nodes.
#[pyclass(module = "dead_cst._native", frozen, get_all, name = "PerFileEdge")]
pub(crate) struct PerFileEdge {
    pub(crate) src: u32,
    pub(crate) dst: u32,
    pub(crate) flags: u32,
}

#[pymethods]
impl PerFileEdge {
    #[new]
    #[pyo3(signature = (src, dst, *, flags = 0))]
    fn new(src: u32, dst: u32, flags: u32) -> Self {
        Self { src, dst, flags }
    }
}

/// Mark a file-local node as an entrypoint with a self-documenting
/// marker (`"<__main__>"`, `"<dunder>"`, ...).
#[pyclass(
    module = "dead_cst._native",
    frozen,
    get_all,
    name = "PerFileEntrypoint"
)]
pub(crate) struct PerFileEntrypoint {
    pub(crate) target: u32,
    pub(crate) marker: String,
}

#[pymethods]
impl PerFileEntrypoint {
    #[new]
    #[pyo3(signature = (target, *, marker))]
    fn new(target: u32, marker: String) -> Self {
        Self { target, marker }
    }
}

/// Mint a synthetic node and wire it to file-local nodes.
/// `edges_from` are `source -> this`; `edges_to` are `this -> target`.
#[pyclass(module = "dead_cst._native", frozen, get_all, name = "PerFileNode")]
pub(crate) struct PerFileNode {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
    pub(crate) flags: u32,
    pub(crate) edges_from: Vec<u32>,
    pub(crate) edges_to: Vec<u32>,
}

#[pymethods]
impl PerFileNode {
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
        edges_from: Vec<u32>,
        edges_to: Vec<u32>,
    ) -> PyResult<Self> {
        Ok(Self {
            fqname,
            kind: intern_kind(kind)?,
            path,
            flags,
            edges_from,
            edges_to,
        })
    }
}

// ---------------------------------------------------------------------------
// Salsa-internal per-file op + carrier struct.
// ---------------------------------------------------------------------------

/// Rust-internal per-file op. Stored as the salsa-tracked return of
/// [`file_to_plugin_ops`] so derives include `salsa::Update +
/// GetSize`. The conversion from Python `PerFile{Edge,Entrypoint,Node}`
/// pyclasses to `PerFileOp` happens at the GIL boundary inside
/// [`file_to_plugin_ops`].
#[derive(Debug, Clone, Eq, PartialEq, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) enum PerFileOp {
    Edge {
        src: u32,
        dst: u32,
        flags: u32,
    },
    Entrypoint {
        target: u32,
        marker: String,
    },
    Node {
        fqname: String,
        kind: String,
        path: String,
        flags: u32,
        edges_from: Vec<u32>,
        edges_to: Vec<u32>,
    },
}

/// Salsa-tracked carrier — every per-file plugin's ops for one file,
/// in registration order.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FilePluginOps {
    pub(crate) ops: Vec<PerFileOp>,
}

// ---------------------------------------------------------------------------
// Plugin registry (process-global).
//
// PyObject isn't `Hash + Eq`, so we can't pass plugin instances as
// salsa-tracked args. Instead, `materialize()` installs the per-file
// plugin list into this registry before populate runs and clears it
// after. The tracked function reads the registry; salsa keys cells
// purely by `(file,)` and treats plugin code as static program code.
// ---------------------------------------------------------------------------

/// Plugins kept alive for the duration of one `materialize()` call.
pub(crate) struct PerFilePluginRegistry {
    pub(crate) plugins: Vec<PyObject>,
}

static PER_FILE_PLUGINS: RwLock<Option<Arc<PerFilePluginRegistry>>> = RwLock::new(None);

/// Install the per-file plugin list. Called once per `materialize()`
/// call before the populate phase fans out. Replaces any previously
/// installed registry.
pub(crate) fn install_per_file_plugins(plugins: Vec<PyObject>) {
    *PER_FILE_PLUGINS.write() = Some(Arc::new(PerFilePluginRegistry { plugins }));
}

/// Drop the per-file plugin list at the end of `materialize()`.
pub(crate) fn clear_per_file_plugins() {
    *PER_FILE_PLUGINS.write() = None;
}

/// RAII guard around the process-global plugin registry.
///
/// `materialize()` and `build_only()` install the per-file plugin
/// list at the top of the call and need to clear it on every exit
/// path — happy path, build error, plugin panic. The guard installs
/// in `new` and clears in `Drop`, so unwinding (from a `?` or a
/// rust panic in [`file_to_plugin_ops`]) restores the registry to
/// the empty state before the next caller observes it.
pub(crate) struct PerFilePluginGuard;

impl PerFilePluginGuard {
    pub(crate) fn install(plugins: Vec<PyObject>) -> Self {
        install_per_file_plugins(plugins);
        PerFilePluginGuard
    }
}

impl Drop for PerFilePluginGuard {
    fn drop(&mut self) {
        clear_per_file_plugins();
    }
}

fn current_per_file_plugins() -> Option<Arc<PerFilePluginRegistry>> {
    PER_FILE_PLUGINS.read().clone()
}

// ---------------------------------------------------------------------------
// FileScope construction from the file's salsa payload.
// ---------------------------------------------------------------------------

/// Build a [`FileScope`] from the file's [`crate::file_payload::FileNodes`]
/// payload plus a one-shot AST walk for the main-block range. All work
/// is bounded by O(decls_in_file); the AST parse is salsa-memoised.
fn build_file_scope(db: &dyn ProjectDb, file: File) -> FileScope {
    let payload = file_to_nodes(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = payload
        .nodes
        .first()
        .map(|n| n.fqname.clone())
        .unwrap_or_default();

    // dunder decls: top-level variable / function nodes whose fqname
    // ends in a Python dunder name.
    let mut dunder_decls: Vec<u32> = Vec::new();
    for (i, node) in payload.nodes.iter().enumerate() {
        if !matches!(node.kind, NodeKind::Variable | NodeKind::Function) {
            continue;
        }
        if is_dunder_name(&node.fqname) {
            dunder_decls.push(i as u32);
        }
    }

    // imports_by_module: every import node bucketed by its upstream
    // module. Star reexports inherit the source module like normal
    // imports — `Import.module` already carries the upstream name.
    let mut imports_by_module: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    for (i, node) in payload.nodes.iter().enumerate() {
        if !matches!(node.kind, NodeKind::Import) {
            continue;
        }
        let Some(imp) = node.imports.as_ref() else {
            continue;
        };
        imports_by_module
            .entry(imp.module.clone())
            .or_default()
            .push(i as u32);
    }

    // Main block: cheap substring check on the source first (>99% of
    // modules don't contain ``__main__``), then a one-shot AST walk
    // on the salsa-cached parse.
    let main_block = compute_main_block(db, file, payload);

    FileScope {
        path_str,
        module_fqname,
        main_block,
        dunder_decls,
        imports_by_module,
    }
}

fn compute_main_block<'db>(
    db: &'db dyn ProjectDb,
    file: File,
    payload: &crate::file_payload::FileNodes<'db>,
) -> Option<(u32, Vec<u32>)> {
    let source = source_text(db, file);
    if !source.contains("__main__") {
        return None;
    }
    let parsed = parsed_module(db, file).load(db);
    let block_range = find_main_block_range(&parsed)?;
    let bs = block_range.start();
    let be = block_range.end();
    // refs[0] is `NodeRef::Module(file)`; real decls start at refs[1].
    // Each `NodeRef::Def(definition)` carries the upstream `Definition`,
    // whose `kind().target_range()` is the byte range — no AST re-walk
    // needed beyond the (already salsa-cached) `parsed_module` load.
    let mut decls: Vec<u32> = Vec::new();
    for (i, r) in payload.refs.iter().enumerate().skip(1) {
        let crate::file_payload::NodeRef::Def(def) = r else {
            continue;
        };
        let target_range = def.kind(db).target_range(&parsed);
        if target_range.start() >= bs && target_range.end() <= be {
            decls.push(i as u32);
        }
    }
    Some((0, decls))
}

// ---------------------------------------------------------------------------
// Salsa-tracked entry: file_to_plugin_ops.
// ---------------------------------------------------------------------------

/// Per-file plugin output, salsa-tracked. Runs every registered
/// per-file plugin against the file (via Python under the GIL), folds
/// the yielded ops into one carrier, and returns it.
///
/// Salsa caches the output keyed by `(file,)`. The plugin list isn't
/// part of the key — per the design discussion, plugin code is
/// treated as static program code (the same plugins on every build).
/// Editing one file invalidates only that file's cell because
/// `file_to_nodes(file)` is the only tracked dependency.
///
/// First-build cost: GIL-serialised across rayon workers because the
/// plugin call is Python code. Cache hits skip the GIL entirely.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_plugin_ops(db: &dyn ProjectDb, file: File) -> FilePluginOps {
    let Some(registry) = current_per_file_plugins() else {
        return FilePluginOps { ops: Vec::new() };
    };
    if registry.plugins.is_empty() {
        return FilePluginOps { ops: Vec::new() };
    }

    // Touching `file_to_nodes(file)` here registers it as a salsa
    // dependency; when a file's payload changes, our cell invalidates.
    // `build_file_scope` already calls it, but be explicit.
    let _ = file_to_nodes(db, file);

    let scope = build_file_scope(db, file);
    let path_for_err = scope.path_str.clone();

    let mut all_ops: Vec<PerFileOp> = Vec::new();
    let result: PyResult<()> = Python::with_gil(|py| {
        let scope_py = Py::new(py, scope)?;
        for plugin in &registry.plugins {
            let bound = plugin.bind(py);
            let result = bound.call_method1("run_per_file", (scope_py.clone_ref(py),))?;
            if result.is_none() {
                continue;
            }
            for item in result.iter()? {
                let op = item?;
                all_ops.push(convert_per_file_op(py, &op)?);
            }
        }
        Ok(())
    });

    if let Err(err) = result {
        // Salsa-tracked functions can't return Result without making
        // the error type salsa::Update. Per the design, per-file
        // plugins are expected to be straightforward enough that
        // errors mean a bug — surface them as a panic carrying the
        // file path for diagnosis. Whole-project plugins remain on
        // the existing `run_plugin_collect` path with full PyResult
        // propagation.
        panic!("per-file plugin raised on {path_for_err}: {err}");
    }

    FilePluginOps { ops: all_ops }
}

/// Convert a yielded `PerFile{Edge,Entrypoint,Node}` pyclass into the
/// pure-rust `PerFileOp` salsa cache stores.
fn convert_per_file_op(py: Python<'_>, op: &Bound<'_, PyAny>) -> PyResult<PerFileOp> {
    if let Ok(e) = op.extract::<PyRef<PerFileEdge>>() {
        return Ok(PerFileOp::Edge {
            src: e.src,
            dst: e.dst,
            flags: e.flags,
        });
    }
    if let Ok(ep) = op.extract::<PyRef<PerFileEntrypoint>>() {
        return Ok(PerFileOp::Entrypoint {
            target: ep.target,
            marker: ep.marker.clone(),
        });
    }
    if let Ok(n) = op.extract::<PyRef<PerFileNode>>() {
        return Ok(PerFileOp::Node {
            fqname: n.fqname.clone(),
            kind: n.kind.to_string(),
            path: n.path.clone(),
            flags: n.flags,
            edges_from: n.edges_from.clone(),
            edges_to: n.edges_to.clone(),
        });
    }
    let _ = py;
    Err(PyValueError::new_err(format!(
        "expected a per-file op (PerFileEdge / PerFileEntrypoint / PerFileNode), got {:?}",
        op.get_type().name()?,
    )))
}

// ---------------------------------------------------------------------------
// Assemble-time fan-in: file-local indices -> global graph indices.
// ---------------------------------------------------------------------------

/// Translate one file-local index into a global graph index via the
/// file's `refs[local] -> NodeRef -> ref_to_global` chain.
fn local_to_global<'db>(
    payload: &FileNodes<'db>,
    ref_to_global: &FxHashMap<NodeRef<'db>, usize>,
    local: u32,
) -> PyResult<usize> {
    let idx = local as usize;
    let node_ref = payload.refs.get(idx).ok_or_else(|| {
        PyValueError::new_err(format!(
            "per-file op references local idx {idx} out of range \
             (file has {} refs)",
            payload.refs.len()
        ))
    })?;
    ref_to_global.get(node_ref).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "per-file op references local idx {idx} but its NodeRef \
             did not assemble into a global node — was the file's \
             `file_to_nodes` payload stale at fan-in time?"
        ))
    })
}

/// Fold every file's salsa-cached [`file_to_plugin_ops`] payload into
/// the global graph. Called from `build_project_graph` once
/// `ref_to_global` is available and before `build_fqname_indices`.
///
/// Cheap when no per-file plugins are registered — every file's
/// payload is empty and the outer loop short-circuits per file. With
/// plugins registered, each op is one or two `local_to_global`
/// lookups plus an `add_edge` / `intern_node`.
pub(crate) fn fan_in_per_file_plugin_ops<'db>(
    py: Python<'_>,
    db: &'db dyn ProjectDb,
    project_files: &[File],
    builder: &mut GraphBuilder,
    ref_to_global: &FxHashMap<NodeRef<'db>, usize>,
) -> PyResult<()> {
    for &file in project_files {
        let ops = file_to_plugin_ops(db, file);
        if ops.ops.is_empty() {
            continue;
        }
        let payload = file_to_nodes(db, file);
        for op in &ops.ops {
            match op {
                PerFileOp::Edge { src, dst, flags } => {
                    let src_g = local_to_global(payload, ref_to_global, *src)?;
                    let dst_g = local_to_global(payload, ref_to_global, *dst)?;
                    builder.add_edge(src_g, dst_g, *flags);
                }
                PerFileOp::Entrypoint { target, marker } => {
                    let target_g = local_to_global(payload, ref_to_global, *target)?;
                    let (decl_fqname, decl_path) = {
                        let node = builder.nodes[target_g].borrow(py);
                        (node.fqname.clone(), node.path.clone())
                    };
                    let marker_fqname = format!("{marker}:{decl_fqname}");
                    let marker_idx = builder.intern_node(
                        py,
                        synthetic_node(marker_fqname, "synthetic", decl_path, NODE_FLAG_ENTRYPOINT),
                    )?;
                    builder.add_edge(marker_idx, target_g, 0);
                }
                PerFileOp::Node {
                    fqname,
                    kind,
                    path,
                    flags,
                    edges_from,
                    edges_to,
                } => {
                    // Resolve every endpoint key BEFORE minting the
                    // node so a bad local index doesn't leave an
                    // unconnected synthetic in the graph (same
                    // pre-validation as `AddNode` apply in builder.rs).
                    let mut from_g: Vec<usize> = Vec::with_capacity(edges_from.len());
                    for &local in edges_from {
                        from_g.push(local_to_global(payload, ref_to_global, local)?);
                    }
                    let mut to_g: Vec<usize> = Vec::with_capacity(edges_to.len());
                    for &local in edges_to {
                        to_g.push(local_to_global(payload, ref_to_global, local)?);
                    }
                    let kind_static = intern_kind(kind)?;
                    let node_idx = builder.intern_node(
                        py,
                        synthetic_node(fqname.clone(), kind_static, path.clone(), *flags),
                    )?;
                    for src in from_g {
                        builder.add_edge(src, node_idx, 0);
                    }
                    for dst in to_g {
                        builder.add_edge(node_idx, dst, 0);
                    }
                }
            }
        }
    }
    Ok(())
}
