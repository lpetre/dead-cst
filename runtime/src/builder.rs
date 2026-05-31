//! Graph-construction primitives: `NodeKey` (positional identity for an
//! interned node), `GraphBuilder` (the in-progress graph), the
//! plugin-facing graph ops (`AddNode`/`AddEdge`/`AddEntrypoint`), the
//! generic BFS walker, and the batched `apply_prepared_batch`
//! apply pass.

use std::sync::OnceLock;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::File;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::file_payload::ImportPayload;
use crate::graph::{intern_kind, Import, SymbolNode};
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

/// Pure-rust mirror of [`SymbolNode`]'s data fields.
///
/// The builder stores these instead of `Py<SymbolNode>` so node
/// interning costs no GIL acquisition or Python heap allocation — the
/// hot path during a build mints hundreds of thousands of nodes, the
/// vast majority of which are never handed back to Python. A
/// `Py<SymbolNode>` is materialized lazily by [`GraphNode::to_symbol`]
/// only for the nodes a Python caller actually surfaces (`nodes()`,
/// `nodes_at()`, the descendants / ancestors / reachable snapshots).
/// Internal queries read the fields here directly with no borrow.
pub(crate) struct GraphNode {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
    pub(crate) flags: u32,
    pub(crate) imports: Option<ImportPayload>,
}

impl GraphNode {
    /// Positional identity for the intern table. Mirrors
    /// [`node_key_of`] (which projects a `SymbolNode`) but reads the
    /// pure-rust fields directly.
    pub(crate) fn key(&self) -> NodeKey {
        NodeKey {
            fqname: self.fqname.clone(),
            kind: self.kind,
            path: self.path.clone(),
            start_line: self.start_line,
            start_column: self.start_column,
            end_line: self.end_line,
            end_column: self.end_column,
        }
    }

    /// Mint the `Py<SymbolNode>` for this node. Called lazily, only at
    /// the boundary where a node is actually handed to Python.
    pub(crate) fn to_symbol(&self, py: Python<'_>) -> PyResult<Py<SymbolNode>> {
        let imports = match &self.imports {
            Some(ip) => Some(Py::new(
                py,
                Import {
                    module: ip.module.clone(),
                    decl: ip.decl.clone(),
                    star: ip.star,
                },
            )?),
            None => None,
        };
        Py::new(
            py,
            SymbolNode {
                fqname: self.fqname.clone(),
                kind: self.kind,
                path: self.path.clone(),
                start_line: self.start_line,
                start_column: self.start_column,
                end_line: self.end_line,
                end_column: self.end_column,
                flags: self.flags,
                imports,
                cached_hash: OnceLock::new(),
            },
        )
    }
}

pub(crate) struct GraphBuilder {
    pub(crate) nodes: Vec<GraphNode>,
    pub(crate) node_index: FxHashMap<NodeKey, usize>,
    pub(crate) edges: Vec<(usize, usize, u32)>,
    pub(crate) edge_set: FxHashSet<(usize, usize, u32)>,
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
    pub(crate) peer_pyi_to_py: FxHashMap<File, File>,
    /// `{ synthetic_fqname -> node idx }` for ``[external dist] X`` and
    /// ``[unresolved] X`` synthetics. Synthetics are deduplicated by
    /// fqname project-wide: every site that imports ``rustworkx``
    /// resolves to the same ``[external dist] rustworkx`` node, so
    /// reachability and the codemod's "this import has no
    /// dependents" query both work on a single anchor.
    pub(crate) synthetic_nodes: FxHashMap<String, usize>,
}

impl GraphBuilder {
    /// Reserves capacity for `expected_nodes`-many interned nodes
    /// up-front. The assemble pass cheaply pre-counts the total node
    /// population via the salsa-memoized `file_to_nodes` payloads and
    /// passes the sum here, saving the rehash chain the four-doubling
    /// growth path of an unsized hashmap pays. Pass `0` when no
    /// estimate is available (e.g. from tests).
    pub(crate) fn with_capacity(expected_nodes: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(expected_nodes),
            node_index: FxHashMap::with_capacity_and_hasher(expected_nodes, Default::default()),
            edges: Vec::new(),
            edge_set: FxHashSet::default(),
            forward_adj: Vec::with_capacity(expected_nodes),
            reverse_adj: Vec::with_capacity(expected_nodes),
            peer_pyi_to_py: FxHashMap::default(),
            synthetic_nodes: FxHashMap::default(),
        }
    }

    pub(crate) fn intern_node(&mut self, node: GraphNode) -> usize {
        let key = node.key();
        if let Some(&idx) = self.node_index.get(&key) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.node_index.insert(key, idx);
        self.forward_adj.push(Vec::new());
        self.reverse_adj.push(Vec::new());
        idx
    }

    /// Get (or mint) the deduplicated synthetic node with the given
    /// fully qualified name. Synthetics anchor edges to imports the
    /// project doesn't own — stdlib stays silent, ``[external dist] X``
    /// covers third-party site-packages, ``[unresolved] X`` covers
    /// genuinely-missing top-level names. Path is empty (synthetics
    /// don't correspond to a file in the project tree) and position
    /// is the (0, 0) sentinel.
    pub(crate) fn intern_synthetic(&mut self, fqname: String) -> usize {
        if let Some(&idx) = self.synthetic_nodes.get(&fqname) {
            return idx;
        }
        let idx = self.intern_node(synthetic_node(
            fqname.clone(),
            "synthetic",
            String::new(),
            0,
        ));
        self.synthetic_nodes.insert(fqname, idx);
        idx
    }

    pub(crate) fn add_edge(&mut self, src: usize, dst: usize, flags: u32) {
        let triple = (src, dst, flags);
        if self.edge_set.insert(triple) {
            self.edges.push(triple);
            self.forward_adj[src].push((dst, flags));
            self.reverse_adj[dst].push((src, flags));
        }
    }

    /// Bulk-insert a batch of pre-sorted, pre-deduplicated edge triples.
    ///
    /// The caller is responsible for ordering / dedup'ing `triples`
    /// (e.g. via `sort_unstable` + `dedup`); this method still probes
    /// `edge_set` per triple to merge against any edges already present
    /// (so a second `extend_edges` call won't double-insert). Compared
    /// with looping `add_edge`, the win is amortising the per-edge
    /// branch / hash overhead and pre-reserving capacity on `edges`
    /// and the per-node adjacency vectors.
    pub(crate) fn extend_edges(&mut self, triples: Vec<(usize, usize, u32)>) {
        if triples.is_empty() {
            return;
        }
        self.edge_set.reserve(triples.len());
        self.edges.reserve(triples.len());
        for triple in triples {
            if self.edge_set.insert(triple) {
                let (src, dst, flags) = triple;
                self.edges.push(triple);
                self.forward_adj[src].push((dst, flags));
                self.reverse_adj[dst].push((src, flags));
            }
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

/// Index-keyed variant of :class:`AddEdge`. Accepts positional
/// indices into ``ctx.nodes()`` instead of ``SymbolNode`` references.
///
/// Lets plugins that already work in index space (e.g. paired with
/// :meth:`DeclQuery.indices` or :meth:`ProjectContext.indices_where`)
/// emit edges without ever round-tripping through ``Py<SymbolNode>``.
/// The apply pass treats it identically to :class:`AddEdge` once the
/// indices land in :meth:`GraphBuilder::add_edge`.
///
/// Raises :class:`IndexError` at apply time when either endpoint is
/// out of range for the current graph snapshot.
#[pyclass(frozen, get_all)]
pub(crate) struct AddEdgeByIdx {
    pub(crate) src_idx: usize,
    pub(crate) dst_idx: usize,
    pub(crate) flags: u32,
}

#[pymethods]
impl AddEdgeByIdx {
    #[new]
    #[pyo3(signature = (src_idx, dst_idx, *, flags = 0))]
    fn new(src_idx: usize, dst_idx: usize, flags: u32) -> Self {
        Self {
            src_idx,
            dst_idx,
            flags,
        }
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

/// Index-keyed variant of :class:`AddEntrypoint`. Takes a positional
/// index into ``ctx.nodes()`` instead of a ``SymbolNode`` reference;
/// the apply pass derefs the decl on the rust side to read its
/// ``fqname`` / ``path`` for the marker, so plugins working in
/// idx-space don't pay the ``Py<SymbolNode>`` allocation just to flag
/// a seed.
///
/// Raises :class:`IndexError` at apply time when ``decl_idx`` is out
/// of range.
#[pyclass(frozen, get_all)]
pub(crate) struct AddEntrypointByIdx {
    pub(crate) decl_idx: usize,
    pub(crate) marker: String,
}

#[pymethods]
impl AddEntrypointByIdx {
    #[new]
    #[pyo3(signature = (decl_idx, *, marker))]
    fn new(decl_idx: usize, marker: String) -> Self {
        Self { decl_idx, marker }
    }
}

/// Mint a synthetic intermediate node.
///
/// ``edges_from`` / ``edges_to`` wire the new node atomically — every
/// element of ``edges_from`` becomes a ``source -> this`` edge, every
/// element of ``edges_to`` a ``this -> target`` edge — so a plugin
/// doesn't need a separate handle to reference the freshly-minted node
/// from subsequent ops. Set ``flags = NodeFlags.ENTRYPOINT`` to make
/// the node a seed (``AddEntrypoint`` / ``AddEntrypointByIdx`` are the
/// single-target sugar).
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

/// Index-keyed variant of :class:`AddNode`. Wires the freshly-minted
/// synthetic node with positional indices into ``ctx.nodes()`` instead
/// of ``SymbolNode`` references for ``edges_from`` / ``edges_to``.
///
/// Pairs with the ``.indices()`` query terminals and
/// :meth:`ProjectContext.indices_where` so plugins that work in index
/// space don't have to round-trip through ``Py<SymbolNode>`` just to
/// wire their synthetic markers. The apply pass treats it identically
/// to :class:`AddNode` once the indices land in the builder.
///
/// Raises :class:`IndexError` at apply time when any endpoint is out
/// of range for the current graph snapshot (pre-intern, so the new
/// node is not created if any endpoint check fails).
#[pyclass(frozen, get_all)]
pub(crate) struct AddNodeByIdx {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
    pub(crate) flags: u32,
    pub(crate) edges_from_idx: Vec<usize>,
    pub(crate) edges_to_idx: Vec<usize>,
}

#[pymethods]
impl AddNodeByIdx {
    #[new]
    #[pyo3(signature = (
        fqname,
        *,
        path,
        kind = "synthetic",
        flags = 0,
        edges_from_idx = Vec::new(),
        edges_to_idx = Vec::new(),
    ))]
    fn new(
        fqname: String,
        path: String,
        kind: &str,
        flags: u32,
        edges_from_idx: Vec<usize>,
        edges_to_idx: Vec<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            fqname,
            kind: intern_kind(kind)?,
            path,
            flags,
            edges_from_idx,
            edges_to_idx,
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
) -> FxHashSet<usize> {
    let mut visited: FxHashSet<usize> = FxHashSet::default();
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

/// Construct a position-less [`GraphNode`] (start/end zeroed, `imports`
/// absent) for ops minted at plugin-apply time. The four
/// caller-supplied fields are the only ones that vary across the
/// synthetic / entrypoint / `AddNode` shapes.
pub(crate) fn synthetic_node(
    fqname: String,
    kind: &'static str,
    path: String,
    flags: u32,
) -> GraphNode {
    GraphNode {
        fqname,
        kind,
        path,
        start_line: 0,
        start_column: 0,
        end_line: 0,
        end_column: 0,
        flags,
        imports: None,
    }
}

/// A [`GraphOp`] prepared for application: every Python-owned field
/// has been hoisted into pure-rust storage so that
/// [`apply_prepared_batch`] can drop the GIL before contending for
/// the write lock on ``outputs``. Edge endpoints are represented as
/// [`NodeKey`]s (the positional identity used by the builder's
/// index) plus the source ``decl.fqname`` for entrypoint markers.
/// See [`apply_prepared_batch`] for the concurrency rationale.
pub(crate) enum PreparedOp {
    Edge {
        src: NodeKey,
        dst: NodeKey,
        flags: u32,
    },
    EdgeByIdx {
        src_idx: usize,
        dst_idx: usize,
        flags: u32,
    },
    Entrypoint {
        decl: NodeKey,
        decl_fqname: String,
        decl_path: String,
        marker: String,
    },
    EntrypointByIdx {
        decl_idx: usize,
        marker: String,
    },
    Node {
        fqname: String,
        kind: &'static str,
        path: String,
        flags: u32,
        edges_from: Vec<NodeKey>,
        edges_to: Vec<NodeKey>,
    },
    NodeByIdx {
        fqname: String,
        kind: &'static str,
        path: String,
        flags: u32,
        edges_from_idx: Vec<usize>,
        edges_to_idx: Vec<usize>,
    },
}

/// Opaque Python handle wrapping a plugin's pre-prepared op queue.
///
/// :meth:`ProjectContext.run_plugin_collect` mints one of these per
/// plugin (containing the plugin's full yield order in pure-rust
/// form); :meth:`ProjectContext.apply_ops_batched` takes a list of
/// these and folds them all into the graph under a single write-lock
/// window. Holding the prepared ops in an opaque pyclass lets the
/// Python driver shuttle them through a
/// :class:`concurrent.futures.ThreadPoolExecutor` without paying
/// per-op FFI roundtrips and without ``PreparedOp`` having to be a
/// pyclass itself.
///
/// Concurrency: the inner [`Mutex`] is never observed contended by
/// design — each `CollectedOps` is created on one thread, consumed
/// on another (the apply pass), and never aliased. The lock is here
/// so the pyclass can be ``Send``.
#[pyclass]
pub(crate) struct CollectedOps {
    pub(crate) ops: parking_lot::Mutex<Option<Vec<PreparedOp>>>,
}

impl CollectedOps {
    pub(crate) fn new(ops: Vec<PreparedOp>) -> Self {
        Self {
            ops: parking_lot::Mutex::new(Some(ops)),
        }
    }

    /// Drain the inner ops. Returns an error if already drained
    /// (calling :meth:`apply_ops_batched` twice on the same handle).
    pub(crate) fn take(&self) -> PyResult<Vec<PreparedOp>> {
        self.ops.lock().take().ok_or_else(|| {
            PyValueError::new_err(
                "CollectedOps already drained: \
                 apply_ops_batched consumes the handle and it cannot be re-used.",
            )
        })
    }
}

/// Convert a single Python-owned :class:`GraphOp` into a [`PreparedOp`].
///
/// The "prepare" half of the old single-op apply pipeline. Splitting
/// this out lets the batched-apply pass extract every op's fields
/// under one GIL window, then drop the GIL once for the full apply
/// rather than ping-ponging the GIL / write lock per op.
pub(crate) fn prepare_graph_op(py: Python<'_>, op: &Bound<'_, PyAny>) -> PyResult<PreparedOp> {
    let prepared = if let Ok(add_edge) = op.extract::<PyRef<AddEdge>>() {
        PreparedOp::Edge {
            src: node_key_of(&add_edge.src.borrow(py)),
            dst: node_key_of(&add_edge.dst.borrow(py)),
            flags: add_edge.flags,
        }
    } else if let Ok(add_edge_idx) = op.extract::<PyRef<AddEdgeByIdx>>() {
        PreparedOp::EdgeByIdx {
            src_idx: add_edge_idx.src_idx,
            dst_idx: add_edge_idx.dst_idx,
            flags: add_edge_idx.flags,
        }
    } else if let Ok(add_ep) = op.extract::<PyRef<AddEntrypoint>>() {
        let decl_ref = add_ep.decl.borrow(py);
        PreparedOp::Entrypoint {
            decl: node_key_of(&decl_ref),
            decl_fqname: decl_ref.fqname.clone(),
            decl_path: decl_ref.path.clone(),
            marker: add_ep.marker.clone(),
        }
    } else if let Ok(add_ep_idx) = op.extract::<PyRef<AddEntrypointByIdx>>() {
        PreparedOp::EntrypointByIdx {
            decl_idx: add_ep_idx.decl_idx,
            marker: add_ep_idx.marker.clone(),
        }
    } else if let Ok(add_node) = op.extract::<PyRef<AddNode>>() {
        let edges_from: Vec<NodeKey> = add_node
            .edges_from
            .iter()
            .map(|n| node_key_of(&n.borrow(py)))
            .collect();
        let edges_to: Vec<NodeKey> = add_node
            .edges_to
            .iter()
            .map(|n| node_key_of(&n.borrow(py)))
            .collect();
        PreparedOp::Node {
            fqname: add_node.fqname.clone(),
            kind: add_node.kind,
            path: add_node.path.clone(),
            flags: add_node.flags,
            edges_from,
            edges_to,
        }
    } else if let Ok(add_node_idx) = op.extract::<PyRef<AddNodeByIdx>>() {
        PreparedOp::NodeByIdx {
            fqname: add_node_idx.fqname.clone(),
            kind: add_node_idx.kind,
            path: add_node_idx.path.clone(),
            flags: add_node_idx.flags,
            edges_from_idx: add_node_idx.edges_from_idx.clone(),
            edges_to_idx: add_node_idx.edges_to_idx.clone(),
        }
    } else {
        return Err(PyValueError::new_err(format!(
            "expected a GraphOp (AddEdge / AddEdgeByIdx / AddEntrypoint / \
             AddEntrypointByIdx / AddNode / AddNodeByIdx), got {:?}",
            op.get_type().name()?,
        )));
    };
    Ok(prepared)
}

/// Apply a batch of pre-prepared ops to the graph in a single
/// write-lock window. The GIL is released for the entire apply pass:
/// every op is pure-rust now — node interning stores [`GraphNode`]s
/// with no `Py<SymbolNode>` allocation — so the apply never needs the
/// GIL back.
///
/// Ops apply in input order; an error in any op fails the whole
/// batch with that op's error. Earlier ops in the batch still land
/// — graph state is not transactional.
pub(crate) fn apply_prepared_batch(
    ctx: &Py<ProjectContext>,
    py: Python<'_>,
    prepared: Vec<PreparedOp>,
) -> PyResult<()> {
    if prepared.is_empty() {
        return Ok(());
    }
    // The ``write()`` MUST happen with the GIL released. Otherwise a
    // concurrent reader returning from ``allow_threads`` and waiting
    // on ``take_gil`` would deadlock against us — we'd be holding the
    // GIL while blocked on ``wait_for_readers``, and the reader
    // can't drop its read guard without first re-attaching the GIL.
    //
    // We grab an ``Arc<RwLock<Option<BuildOutputs>>>`` from the
    // pyclass (which is the outputs handle itself) so we can move
    // it across the ``allow_threads`` boundary — ``Arc`` is Send,
    // ``PyRef`` is not.
    let outputs_lock = ctx.borrow(py).outputs.clone();
    py.allow_threads(move || -> PyResult<()> {
        let mut outputs = outputs_lock.write();
        let outputs = outputs
            .as_mut()
            .ok_or_else(|| not_materialized("apply_ops_batched"))?;
        for op in prepared {
            apply_prepared(outputs, op)?;
        }
        Ok(())
    })
}

fn apply_prepared(
    outputs: &mut crate::project::BuildOutputs,
    prepared: PreparedOp,
) -> PyResult<()> {
    match prepared {
        PreparedOp::Edge { src, dst, flags } => {
            let src_idx = lookup_idx_by_key(&outputs.builder, &src, "src")?;
            let dst_idx = lookup_idx_by_key(&outputs.builder, &dst, "dst")?;
            outputs.builder.add_edge(src_idx, dst_idx, flags);
            Ok(())
        }
        PreparedOp::EdgeByIdx {
            src_idx,
            dst_idx,
            flags,
        } => {
            let len = outputs.builder.nodes.len();
            check_idx_in_range(len, src_idx, "AddEdgeByIdx", "src_idx")?;
            check_idx_in_range(len, dst_idx, "AddEdgeByIdx", "dst_idx")?;
            outputs.builder.add_edge(src_idx, dst_idx, flags);
            Ok(())
        }
        PreparedOp::Entrypoint {
            decl,
            decl_fqname,
            decl_path,
            marker,
        } => {
            let decl_idx = lookup_idx_by_key(&outputs.builder, &decl, "decl")?;
            let marker_fqname = format!("{marker}:{decl_fqname}");
            let marker_idx = outputs.builder.intern_node(synthetic_node(
                marker_fqname,
                "synthetic",
                decl_path,
                NODE_FLAG_ENTRYPOINT,
            ));
            outputs.builder.add_edge(marker_idx, decl_idx, 0);
            Ok(())
        }
        PreparedOp::EntrypointByIdx { decl_idx, marker } => {
            let len = outputs.builder.nodes.len();
            check_idx_in_range(len, decl_idx, "AddEntrypointByIdx", "decl_idx")?;
            // Read fqname / path off the existing node — same shape
            // the ``AddEntrypoint`` arm gets from the prepare step.
            let (decl_fqname, decl_path) = {
                let node = &outputs.builder.nodes[decl_idx];
                (node.fqname.clone(), node.path.clone())
            };
            let marker_fqname = format!("{marker}:{decl_fqname}");
            let marker_idx = outputs.builder.intern_node(synthetic_node(
                marker_fqname,
                "synthetic",
                decl_path,
                NODE_FLAG_ENTRYPOINT,
            ));
            outputs.builder.add_edge(marker_idx, decl_idx, 0);
            Ok(())
        }
        PreparedOp::Node {
            fqname,
            kind,
            path,
            flags,
            edges_from,
            edges_to,
        } => {
            // Resolve endpoint keys to indices *before* minting the
            // node — a missing key then surfaces as a clean
            // ``ValueError`` without leaving an unconnected synthetic
            // behind in the graph. Same pre-validation discipline as
            // the ``NodeByIdx`` arm below.
            let from_idxs = resolve_edge_keys(&outputs.builder, &edges_from, "edges_from")?;
            let to_idxs = resolve_edge_keys(&outputs.builder, &edges_to, "edges_to")?;
            let node_idx = outputs
                .builder
                .intern_node(synthetic_node(fqname, kind, path, flags));
            wire_synthetic_edges(&mut outputs.builder, node_idx, &from_idxs, &to_idxs);
            Ok(())
        }
        PreparedOp::NodeByIdx {
            fqname,
            kind,
            path,
            flags,
            edges_from_idx,
            edges_to_idx,
        } => {
            // Bounds-check every endpoint *before* minting the new
            // node so that a bad index doesn't leave an unconnected
            // synthetic in the graph. The check uses the pre-intern
            // node count: callers cannot reference an idx that
            // doesn't yet exist (including the about-to-be-minted
            // node).
            let len = outputs.builder.nodes.len();
            check_idx_slice_in_range(len, &edges_from_idx, "AddNodeByIdx", "edges_from_idx")?;
            check_idx_slice_in_range(len, &edges_to_idx, "AddNodeByIdx", "edges_to_idx")?;
            let node_idx = outputs
                .builder
                .intern_node(synthetic_node(fqname, kind, path, flags));
            wire_synthetic_edges(
                &mut outputs.builder,
                node_idx,
                &edges_from_idx,
                &edges_to_idx,
            );
            Ok(())
        }
    }
}

/// Wire a freshly-minted synthetic node into the graph: every entry
/// of ``edges_from_idx`` becomes ``source -> node``, every entry of
/// ``edges_to_idx`` becomes ``node -> target``. Endpoints are already
/// validated indices.
fn wire_synthetic_edges(
    builder: &mut GraphBuilder,
    node_idx: usize,
    edges_from_idx: &[usize],
    edges_to_idx: &[usize],
) {
    for &src_idx in edges_from_idx {
        builder.add_edge(src_idx, node_idx, 0);
    }
    for &dst_idx in edges_to_idx {
        builder.add_edge(node_idx, dst_idx, 0);
    }
}

/// Resolve every endpoint key to its builder-side index, propagating
/// the first lookup failure as a ``ValueError`` with the matching
/// ``side`` label.
fn resolve_edge_keys(builder: &GraphBuilder, keys: &[NodeKey], side: &str) -> PyResult<Vec<usize>> {
    keys.iter()
        .map(|k| lookup_idx_by_key(builder, k, side))
        .collect()
}

/// Bounds-check a single index against the builder's current node
/// count, surfacing an ``IndexError`` matching the existing
/// ``AddEdgeByIdx`` style.
fn check_idx_in_range(len: usize, idx: usize, op: &str, side: &str) -> PyResult<()> {
    if idx >= len {
        return Err(PyIndexError::new_err(format!(
            "{op}: {side} {idx} out of range (len={len})"
        )));
    }
    Ok(())
}

/// Bounds-check every index in a slice. Returns on the first
/// out-of-range index with the matching ``IndexError``.
fn check_idx_slice_in_range(len: usize, idxs: &[usize], op: &str, side: &str) -> PyResult<()> {
    for &idx in idxs {
        check_idx_in_range(len, idx, op, side)?;
    }
    Ok(())
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

/// Variant of [`lookup_idx`] that takes a [`NodeKey`] directly so it
/// can be called from [`apply_prepared_batch`]'s GIL-free post-prepare
/// section — see [`PreparedOp`] for why the key is pre-extracted.
pub(crate) fn lookup_idx_by_key(
    builder: &GraphBuilder,
    key: &NodeKey,
    side: &str,
) -> PyResult<usize> {
    builder.node_index.get(key).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "add_edge: {side} node {:?} is not interned in this ProjectContext",
            key.fqname
        ))
    })
}

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the graph-building primitives.
    //!
    //! The interning surface mints `Py<SymbolNode>` and so requires
    //! pyo3-init / a GIL; those paths are exercised by the python suite
    //! end-to-end. Here we cover the data-only pieces:
    //!
    //! * `NodeKey` equality / hashing (flags excluded from identity).
    //! * `node_key_of` projection.
    //! * `synthetic_node` field defaults.
    //! * `bfs` traversal — direction switch + `skip_flags` filtering.
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Build a bare `SymbolNode` for projection tests. Avoids pyo3
    /// because the helper doesn't need to be reachable from Python.
    fn raw_symbol(
        fqname: &str,
        kind: &'static str,
        path: &str,
        start_line: usize,
        flags: u32,
    ) -> SymbolNode {
        SymbolNode {
            fqname: fqname.to_string(),
            kind,
            path: path.to_string(),
            start_line,
            start_column: 0,
            end_line: start_line,
            end_column: 0,
            flags,
            imports: None,
            cached_hash: OnceLock::new(),
        }
    }

    fn hash_of(key: &NodeKey) -> u64 {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    }

    // -- node_key_of / NodeKey -------------------------------------------

    #[test]
    fn node_key_of_projects_identity_fields() {
        let n = raw_symbol("pkg.mod.f", "function", "pkg/mod.py", 12, 0);
        let key = node_key_of(&n);
        assert_eq!(key.fqname, "pkg.mod.f");
        assert_eq!(key.kind, "function");
        assert_eq!(key.path, "pkg/mod.py");
        assert_eq!(key.start_line, 12);
        assert_eq!(key.end_line, 12);
    }

    #[test]
    fn node_key_excludes_flags_from_identity() {
        // Principle 3 from CLAUDE.md: two nodes for the same
        // (fqname, kind, path, position) are the same node regardless
        // of which path computed their flags.
        let a = node_key_of(&raw_symbol("m.f", "function", "m.py", 1, 0));
        let b = node_key_of(&raw_symbol("m.f", "function", "m.py", 1, 7));
        assert!(a == b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn node_key_distinguishes_by_position() {
        // Two `def f` at different lines must be distinct nodes
        // (shadowing semantics).
        let a = node_key_of(&raw_symbol("m.f", "function", "m.py", 1, 0));
        let b = node_key_of(&raw_symbol("m.f", "function", "m.py", 9, 0));
        assert!(a != b);
    }

    #[test]
    fn node_key_distinguishes_by_kind() {
        let a = node_key_of(&raw_symbol("m.X", "class", "m.py", 1, 0));
        let b = node_key_of(&raw_symbol("m.X", "variable", "m.py", 1, 0));
        assert!(a != b);
    }

    #[test]
    fn node_key_distinguishes_by_path() {
        let a = node_key_of(&raw_symbol("X", "class", "a/m.py", 1, 0));
        let b = node_key_of(&raw_symbol("X", "class", "b/m.py", 1, 0));
        assert!(a != b);
    }

    // -- synthetic_node ---------------------------------------------------

    #[test]
    fn synthetic_node_zeroes_positions_and_omits_imports() {
        let n = synthetic_node(
            "pkg".to_string(),
            "synthetic",
            "p.py".to_string(),
            NODE_FLAG_ENTRYPOINT,
        );
        assert_eq!(n.fqname, "pkg");
        assert_eq!(n.kind, "synthetic");
        assert_eq!(n.path, "p.py");
        assert_eq!(n.start_line, 0);
        assert_eq!(n.start_column, 0);
        assert_eq!(n.end_line, 0);
        assert_eq!(n.end_column, 0);
        assert_eq!(n.flags, NODE_FLAG_ENTRYPOINT);
        assert!(n.imports.is_none());
    }

    #[test]
    fn synthetic_node_accepts_arbitrary_kind() {
        // The signature takes a `&'static str` so callers pass already-interned values.
        let n = synthetic_node(String::from("m"), "module", String::new(), 0);
        assert_eq!(n.kind, "module");
        assert!(n.path.is_empty());
    }

    // -- GraphBuilder construction (no pyo3) -----------------------------

    #[test]
    fn graph_builder_with_capacity_starts_empty() {
        let b = GraphBuilder::with_capacity(0);
        assert!(b.nodes.is_empty());
        assert!(b.node_index.is_empty());
        assert!(b.edges.is_empty());
        assert!(b.edge_set.is_empty());
        assert!(b.forward_adj.is_empty());
        assert!(b.reverse_adj.is_empty());
        assert!(b.synthetic_nodes.is_empty());
        assert!(b.peer_pyi_to_py.is_empty());
    }

    // -- bfs --------------------------------------------------------------

    /// Build a `GraphBuilder` with `n` allocated adjacency slots so we
    /// can push edges by hand for `bfs` tests.
    fn empty_builder_with_n_slots(n: usize) -> GraphBuilder {
        let mut b = GraphBuilder::with_capacity(0);
        for _ in 0..n {
            b.forward_adj.push(Vec::new());
            b.reverse_adj.push(Vec::new());
        }
        b
    }

    /// Append a forward+reverse adjacency entry. The plain `Vec::push`
    /// path is enough because `bfs` only reads `forward_adj` /
    /// `reverse_adj` — it never inspects `edges` / `edge_set` / `nodes`.
    fn add_edge_manual(b: &mut GraphBuilder, src: usize, dst: usize, flags: u32) {
        b.forward_adj[src].push((dst, flags));
        b.reverse_adj[dst].push((src, flags));
    }

    #[test]
    fn bfs_forward_walks_descendants_from_single_seed() {
        // 0 -> 1 -> 2 ;  0 -> 3
        let mut b = empty_builder_with_n_slots(4);
        add_edge_manual(&mut b, 0, 1, 0);
        add_edge_manual(&mut b, 1, 2, 0);
        add_edge_manual(&mut b, 0, 3, 0);
        let reachable = bfs(&b, [0], Direction::Forward, 0);
        let expected: FxHashSet<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(reachable, expected);
    }

    #[test]
    fn bfs_reverse_walks_ancestors() {
        let mut b = empty_builder_with_n_slots(4);
        add_edge_manual(&mut b, 0, 1, 0);
        add_edge_manual(&mut b, 1, 2, 0);
        add_edge_manual(&mut b, 3, 2, 0);
        let ancestors = bfs(&b, [2], Direction::Reverse, 0);
        let expected: FxHashSet<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(ancestors, expected);
    }

    #[test]
    fn bfs_skip_flags_filters_edges_by_any_intersecting_bit() {
        // Dead-branch edge gates 0 -> 1.
        let mut b = empty_builder_with_n_slots(3);
        add_edge_manual(&mut b, 0, 1, 1 /* DEAD_BRANCH */);
        add_edge_manual(&mut b, 0, 2, 0);
        // With skip_flags = 1, the 0 -> 1 edge is filtered out.
        let reachable = bfs(&b, [0], Direction::Forward, 1);
        let expected: FxHashSet<usize> = [0, 2].into_iter().collect();
        assert_eq!(reachable, expected);

        // skip_flags = 0 keeps every edge.
        let reachable_all = bfs(&b, [0], Direction::Forward, 0);
        let expected_all: FxHashSet<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(reachable_all, expected_all);
    }

    #[test]
    fn bfs_skip_flags_intersects_any_bit() {
        // flags = 0b11; skipping any of the bits should drop the edge.
        let mut b = empty_builder_with_n_slots(2);
        add_edge_manual(&mut b, 0, 1, 0b11);
        for mask in [0b01, 0b10, 0b11, 0b100 | 0b01] {
            let r = bfs(&b, [0], Direction::Forward, mask);
            assert_eq!(
                r,
                [0].into_iter().collect::<FxHashSet<_>>(),
                "skip_flags=0b{mask:b} should drop the edge"
            );
        }
        // mask=0b100 doesn't intersect — edge survives.
        let r = bfs(&b, [0], Direction::Forward, 0b100);
        assert_eq!(r, [0, 1].into_iter().collect::<FxHashSet<_>>());
    }

    #[test]
    fn bfs_handles_cycles_without_looping() {
        let mut b = empty_builder_with_n_slots(3);
        add_edge_manual(&mut b, 0, 1, 0);
        add_edge_manual(&mut b, 1, 2, 0);
        add_edge_manual(&mut b, 2, 0, 0);
        let reachable = bfs(&b, [0], Direction::Forward, 0);
        let expected: FxHashSet<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(reachable, expected);
    }

    #[test]
    fn bfs_empty_seeds_returns_empty_set() {
        let b = empty_builder_with_n_slots(5);
        let reachable = bfs(&b, std::iter::empty::<usize>(), Direction::Forward, 0);
        assert!(reachable.is_empty());
    }

    #[test]
    fn bfs_isolated_seed_returns_just_itself() {
        let b = empty_builder_with_n_slots(3);
        let r = bfs(&b, [2], Direction::Forward, 0);
        assert_eq!(r, [2].into_iter().collect::<FxHashSet<_>>());
    }

    #[test]
    fn bfs_multiple_seeds_unions_reachable() {
        let mut b = empty_builder_with_n_slots(5);
        add_edge_manual(&mut b, 0, 1, 0);
        add_edge_manual(&mut b, 2, 3, 0);
        let r = bfs(&b, [0, 2], Direction::Forward, 0);
        let expected: FxHashSet<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(r, expected);
    }

    #[test]
    fn bfs_includes_seed_node_in_result() {
        let b = empty_builder_with_n_slots(1);
        let r = bfs(&b, [0], Direction::Forward, 0);
        assert!(r.contains(&0));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn bfs_reverse_skip_flags_also_filters() {
        // Sanity check: skip_flags applies to reverse BFS too.
        let mut b = empty_builder_with_n_slots(3);
        add_edge_manual(&mut b, 0, 2, 1);
        add_edge_manual(&mut b, 1, 2, 0);
        let r = bfs(&b, [2], Direction::Reverse, 1);
        let expected: FxHashSet<usize> = [1, 2].into_iter().collect();
        assert_eq!(r, expected);
    }
}
