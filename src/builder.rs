//! Graph-construction primitives: `NodeKey` (positional identity for an
//! interned node), `GraphBuilder` (the in-progress graph), the
//! plugin-facing graph ops (`AddNode`/`AddEdge`/`AddEntrypoint`), the
//! generic BFS walker, and the `apply_graph_op` apply pass.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::File;

use crate::graph::{intern_kind, SymbolNode};
use crate::helpers::NODE_FLAG_ENTRYPOINT;
use crate::project::ProjectContext;

/// Positional identity for a node.
///
/// Per `CLAUDE.md` principle 3, `flags` is deliberately *not* part of
/// the key: two nodes for the same `(fqname, kind, path, position)` are
/// the same node regardless of which path computed their flags.
#[derive(Hash, Eq, PartialEq, Clone)]
pub(crate) struct NodeKey {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
}

pub(crate) struct GraphBuilder {
    pub(crate) nodes: Vec<Py<SymbolNode>>,
    pub(crate) node_index: HashMap<NodeKey, usize>,
    pub(crate) edges: Vec<(usize, usize, u32)>,
    pub(crate) edge_set: HashSet<(usize, usize, u32)>,
    /// Per-node forward / reverse adjacency lists kept in sync with
    /// ``edges``. ``bfs`` reads these so traversals are O(deg(i)) per
    /// pop instead of O(|edges|).
    pub(crate) forward_adj: Vec<Vec<(usize, u32)>>,
    pub(crate) reverse_adj: Vec<Vec<(usize, u32)>>,
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
    pub(crate) peer_pyi_to_py: HashMap<File, File>,
    /// `{ synthetic_fqname -> node idx }` for ``[external dist] X`` and
    /// ``[unresolved] X`` synthetics. Synthetics are deduplicated by
    /// fqname project-wide: every site that imports ``rustworkx``
    /// resolves to the same ``[external dist] rustworkx`` node, so
    /// reachability and the codemod's "this import has no
    /// dependents" query both work on a single anchor.
    pub(crate) synthetic_nodes: HashMap<String, usize>,
}

impl GraphBuilder {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn intern_node(&mut self, py: Python<'_>, node: SymbolNode) -> PyResult<usize> {
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
    pub(crate) fn intern_synthetic(&mut self, py: Python<'_>, fqname: String) -> PyResult<usize> {
        if let Some(&idx) = self.synthetic_nodes.get(&fqname) {
            return Ok(idx);
        }
        let idx = self.intern_node(
            py,
            synthetic_node(fqname.clone(), "synthetic", String::new(), 0),
        )?;
        self.synthetic_nodes.insert(fqname, idx);
        Ok(idx)
    }

    pub(crate) fn add_edge(&mut self, src: usize, dst: usize, flags: u32) {
        let triple = (src, dst, flags);
        if self.edge_set.insert(triple) {
            self.edges.push(triple);
            self.forward_adj[src].push((dst, flags));
            self.reverse_adj[dst].push((src, flags));
        }
    }
}

/// Add an edge between two interned nodes.
///
/// ``flags`` carries ``DEAD_BRANCH`` / future edge classifications.
/// Plugins yield this from ``run(ctx)`` instead of mutating the graph
/// directly so the apply pass is a single atomic step on the rust side.
#[pyclass(frozen, get_all)]
pub(crate) struct AddEdge {
    pub(crate) src: Py<SymbolNode>,
    pub(crate) dst: Py<SymbolNode>,
    pub(crate) flags: u32,
}

#[pymethods]
impl AddEdge {
    #[new]
    #[pyo3(signature = (src, dst, *, flags = 0))]
    fn new(src: Py<SymbolNode>, dst: Py<SymbolNode>, flags: u32) -> Self {
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
pub(crate) struct AddEntrypoint {
    pub(crate) decl: Py<SymbolNode>,
    pub(crate) marker: String,
}

#[pymethods]
impl AddEntrypoint {
    #[new]
    #[pyo3(signature = (decl, *, marker))]
    fn new(decl: Py<SymbolNode>, marker: String) -> Self {
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
pub(crate) struct AddNode {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
    pub(crate) flags: u32,
    pub(crate) edges_from: Vec<Py<SymbolNode>>,
    pub(crate) edges_to: Vec<Py<SymbolNode>>,
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
        edges_from: Vec<Py<SymbolNode>>,
        edges_to: Vec<Py<SymbolNode>>,
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

pub(crate) fn not_materialized(op: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ProjectContext.{op}() called outside an active materialize() — \
         did you call it from a plugin's run() method?"
    ))
}

/// Project a `SymbolNode` onto the `NodeKey` used for intern-table
/// identity. Clones the two `String` fields (`fqname`, `path`); the
/// rest are `Copy`.
pub(crate) fn node_key_of(node: &SymbolNode) -> NodeKey {
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
pub(crate) enum Direction {
    Forward,
    Reverse,
}

/// Generic BFS over the build graph. ``skip_flags`` filters edges whose
/// flag mask intersects (any bit) — pass ``0`` to follow every edge.
/// Returns the set of reached node indices including ``start``.
pub(crate) fn bfs(
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

/// Construct a position-less `SymbolNode` (start/end zeroed, `imports`
/// absent) for ops minted at plugin-apply time. The four
/// caller-supplied fields are the only ones that vary across the
/// synthetic / entrypoint / `AddNode` shapes.
pub(crate) fn synthetic_node(
    fqname: String,
    kind: &'static str,
    path: String,
    flags: u32,
) -> SymbolNode {
    SymbolNode {
        fqname,
        kind,
        path,
        start_line: 0,
        start_column: 0,
        end_line: 0,
        end_column: 0,
        flags,
        imports: None,
        cached_hash: OnceLock::new(),
    }
}

pub(crate) fn apply_graph_op(
    ctx: &Py<ProjectContext>,
    py: Python<'_>,
    op: &Bound<'_, PyAny>,
) -> PyResult<()> {
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
            synthetic_node(marker_fqname, "synthetic", path, NODE_FLAG_ENTRYPOINT),
        )?;
        outputs.builder.add_edge(marker_idx, decl_idx, 0);
        return Ok(());
    }
    if let Ok(add_node) = op.extract::<PyRef<AddNode>>() {
        let node_idx = outputs.builder.intern_node(
            py,
            synthetic_node(
                add_node.fqname.clone(),
                add_node.kind,
                add_node.path.clone(),
                add_node.flags,
            ),
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

/// Resolve a `SymbolNode` reference to its builder-side index for an
/// edge endpoint. Surfaces a precise `ValueError` (with `side`) when
/// the node was never interned in this context.
pub(crate) fn lookup_idx(builder: &GraphBuilder, node: &SymbolNode, side: &str) -> PyResult<usize> {
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
