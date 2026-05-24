//! Graph-construction primitives: `NodeKey` (positional identity for an
//! interned node), `GraphBuilder` (the in-progress graph), the
//! plugin-facing graph ops (`AddNode`/`AddEdge`/`AddEntrypoint`), the
//! generic BFS walker, and the `apply_graph_op` apply pass.

use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::File;
use rustc_hash::{FxHashMap, FxHashSet};

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

    /// Like ``intern_node`` but skips the get-before-insert dedup
    /// probe. Caller guarantees the node is fresh: ``assemble_graph``
    /// iterates each file's ``FileNodes`` payload (which is already
    /// deduped within the file) exactly once, so positional identity
    /// is unique by construction. Saves one hashmap probe per node
    /// (~5–10ns); over a 5k-node graph that adds up. Still inserts
    /// into ``node_index`` because the assemble's Pass 2/3 + plugin
    /// ops use ``lookup_idx`` to resolve a `SymbolNode` back to its
    /// index.
    pub(crate) fn append_node(&mut self, py: Python<'_>, node: SymbolNode) -> PyResult<usize> {
        let key = node_key_of(&node);
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
