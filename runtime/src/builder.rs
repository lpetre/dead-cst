//! Graph-construction primitives: `NodeKey` (positional identity for an
//! interned node), `GraphBuilder` (the in-progress graph), the
//! `PreparedOp` mutation vocabulary native plugins emit, the generic
//! BFS walker, and the batched `apply_prepared_batch` apply pass.

use std::sync::OnceLock;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::File;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::file_payload::ImportPayload;
use crate::graph::{Import, SymbolNode};
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
///
/// `Default` yields an empty placeholder used by
/// [`GraphBuilder::prefill_payload_region`]; every slot it creates is
/// overwritten by [`GraphBuilder::place_node`] before the graph is read.
#[derive(Default)]
pub(crate) struct GraphNode {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
    pub(crate) flags: u32,
    /// `@overload` stub marker carried from [`NodeData`]. Build-time only —
    /// drives the fqname-trie exclusion in `build_fqname_indices`; never
    /// reaches `SymbolNode`, the dcg file, or Python.
    pub(crate) is_overload: bool,
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
    pub(crate) edges: Vec<(usize, usize, u8)>,
    pub(crate) edge_set: FxHashSet<(usize, usize, u8)>,
    /// Per-node forward / reverse adjacency lists kept in sync with
    /// ``edges``. ``bfs`` reads these so traversals are O(deg(i)) per
    /// pop instead of O(|edges|).
    pub(crate) forward_adj: Vec<Vec<(usize, u8)>>,
    pub(crate) reverse_adj: Vec<Vec<(usize, u8)>>,
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
    /// `{ external_fqname -> node idx }` for the `kind="external"`
    /// anchors: ``[external dist] X``, ``[external file] X``, and
    /// ``[unresolved] X``. Deduplicated by fqname project-wide: every
    /// site that imports ``rustworkx`` resolves to the same
    /// ``[external dist] rustworkx`` node, so reachability and the
    /// codemod's "this import has no dependents" query both work on a
    /// single anchor.
    pub(crate) external_nodes: FxHashMap<String, usize>,
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
            external_nodes: FxHashMap::default(),
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

    /// Pre-size the payload region to `total_nodes` so the
    /// offset-driven assembly pass can write each file's nodes at a
    /// precomputed global index (`offset[file] + local_idx`) via
    /// [`place_node`] instead of relying on serial append order.
    /// `nodes` is filled with `GraphNode::default()` placeholders that
    /// `place_node` overwrites exactly once each — the per-file
    /// offsets partition `[0, total_nodes)`, so no placeholder
    /// survives into the finished graph. External nodes minted later
    /// via [`intern_external`] append past this region (index
    /// `total_nodes` onward).
    pub(crate) fn prefill_payload_region(&mut self, total_nodes: usize) {
        self.nodes.resize_with(total_nodes, GraphNode::default);
        self.forward_adj.resize_with(total_nodes, Vec::new);
        self.reverse_adj.resize_with(total_nodes, Vec::new);
        self.node_index.reserve(total_nodes);
    }

    /// Place a payload node at its precomputed global index. Unlike
    /// [`intern_node`] (append + dedup-by-key), the offset-driven
    /// assembly pass owns the index, so this writes `nodes[idx]`
    /// directly and records the key. Payload nodes are
    /// position-distinct (`CLAUDE.md` principle 3), so no two ever
    /// share a [`NodeKey`]; an assertion — kept on in release too,
    /// since a collision would silently overwrite a node and corrupt
    /// the index map — guards that invariant. Requires the region to
    /// be pre-sized via [`prefill_payload_region`].
    pub(crate) fn place_node(&mut self, idx: usize, node: GraphNode) {
        let prev = self.node_index.insert(node.key(), idx);
        assert!(
            prev.is_none(),
            "payload node key collision at index {idx}: assembly must mint each NodeKey once"
        );
        self.nodes[idx] = node;
    }

    /// Get (or mint) the deduplicated `kind="external"` node for a
    /// non-first-party import target the project doesn't own —
    /// ``[external dist] X`` covers third-party site-packages,
    /// ``[external file] X`` a resolved-but-out-of-tree file, and
    /// ``[unresolved] X`` a genuinely-missing top-level name. The
    /// resolved target's `path` (empty for ``[unresolved]``, which has
    /// no file) is carried so callers can exclude these nodes by path;
    /// position is the (0, 0) sentinel. Deduped by fqname project-wide;
    /// the first `path` seen for a given fqname wins, so several
    /// submodules of one dist collapse to a single node.
    pub(crate) fn intern_external(&mut self, fqname: String, path: String) -> usize {
        if let Some(&idx) = self.external_nodes.get(&fqname) {
            return idx;
        }
        let idx = self.intern_node(positionless_node(fqname.clone(), "external", path, 0));
        self.external_nodes.insert(fqname, idx);
        idx
    }

    pub(crate) fn add_edge(&mut self, src: usize, dst: usize, flags: u8) {
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
    pub(crate) fn extend_edges(&mut self, triples: Vec<(usize, usize, u8)>) {
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
    skip_flags: u8,
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
/// absent) for the `kind="external"` anchors minted by
/// [`GraphBuilder::intern_external`], which stand in for imports the
/// project doesn't own and have no source range of their own.
pub(crate) fn positionless_node(
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
        is_overload: false,
        imports: None,
    }
}

/// A graph mutation prepared for application: a native plugin emits
/// these directly so that [`apply_prepared_batch`] can drop the GIL
/// before contending for the write lock on ``outputs``. Endpoints are
/// positional indices into the current graph snapshot
/// (``ctx.nodes()``); see [`apply_prepared_batch`] for the concurrency
/// rationale.
pub(crate) enum PreparedOp {
    Edge {
        src_idx: usize,
        dst_idx: usize,
        flags: u8,
    },
    Entrypoint {
        decl_idx: usize,
    },
    /// OR `flags` onto an existing decl's node flags (no new node). Unlike
    /// [`PreparedOp::Entrypoint`] (which always ORs `ENTRYPOINT`), the bits
    /// are caller-supplied — used to stamp a registered node flag, e.g.
    /// `test/testcase`, directly on a decl the plugin discovered rather than
    /// routing it through a seed marker node.
    FlagDecl {
        decl_idx: usize,
        flags: u32,
    },
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
            .ok_or_else(|| not_materialized("apply_prepared_batch"))?;
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
        PreparedOp::Edge {
            src_idx,
            dst_idx,
            flags,
        } => {
            let len = outputs.builder.nodes.len();
            check_idx_in_range(len, src_idx, "PreparedOp::Edge", "src_idx")?;
            check_idx_in_range(len, dst_idx, "PreparedOp::Edge", "dst_idx")?;
            outputs.builder.add_edge(src_idx, dst_idx, flags);
            Ok(())
        }
        PreparedOp::Entrypoint { decl_idx } => {
            let len = outputs.builder.nodes.len();
            check_idx_in_range(len, decl_idx, "PreparedOp::Entrypoint", "decl_idx")?;
            // Flag the decl itself an entrypoint seed — no synthetic
            // marker node, since reachability seeds off the flag, not
            // an edge from a marker.
            outputs.builder.nodes[decl_idx].flags |= NODE_FLAG_ENTRYPOINT;
            Ok(())
        }
        PreparedOp::FlagDecl { decl_idx, flags } => {
            let len = outputs.builder.nodes.len();
            check_idx_in_range(len, decl_idx, "PreparedOp::FlagDecl", "decl_idx")?;
            outputs.builder.nodes[decl_idx].flags |= flags;
            Ok(())
        }
    }
}

/// Bounds-check a single index against the builder's current node
/// count, surfacing an ``IndexError`` with the op name and side.
fn check_idx_in_range(len: usize, idx: usize, op: &str, side: &str) -> PyResult<()> {
    if idx >= len {
        return Err(PyIndexError::new_err(format!(
            "{op}: {side} {idx} out of range (len={len})"
        )));
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
    //! * `positionless_node` field defaults.
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

    // -- positionless_node ------------------------------------------------

    #[test]
    fn positionless_node_zeroes_positions_and_omits_imports() {
        let n = positionless_node(
            "pkg".to_string(),
            "external",
            "p.py".to_string(),
            NODE_FLAG_ENTRYPOINT,
        );
        assert_eq!(n.fqname, "pkg");
        assert_eq!(n.kind, "external");
        assert_eq!(n.path, "p.py");
        assert_eq!(n.start_line, 0);
        assert_eq!(n.start_column, 0);
        assert_eq!(n.end_line, 0);
        assert_eq!(n.end_column, 0);
        assert_eq!(n.flags, NODE_FLAG_ENTRYPOINT);
        assert!(n.imports.is_none());
    }

    #[test]
    fn positionless_node_accepts_arbitrary_kind() {
        // The signature takes a `&'static str` so callers pass already-interned values.
        let n = positionless_node(String::from("m"), "module", String::new(), 0);
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
        assert!(b.external_nodes.is_empty());
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
    fn add_edge_manual(b: &mut GraphBuilder, src: usize, dst: usize, flags: u8) {
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
