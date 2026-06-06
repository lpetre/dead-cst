//! Graph-construction primitives: `NodeKey` (positional identity for an
//! interned node), `GraphBuilder` (the in-progress graph), the
//! `PreparedOp` mutation vocabulary native plugins emit, the generic
//! BFS walker, and the batched `apply_prepared_batch` apply pass.

use std::sync::OnceLock;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
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
/// the key: two nodes for the same `(fqname, kind, file, position)` are
/// the same node regardless of which path computed their flags. `file`
/// is a `Copy` salsa id (one per path), so it discriminates exactly as
/// the old path string did without the per-node `String` clone.
#[derive(Hash, Eq, PartialEq, Clone)]
pub(crate) struct NodeKey {
    pub(crate) fqname: String,
    pub(crate) kind: &'static str,
    pub(crate) file: File,
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
/// overwritten by the assemble pass's parallel node fill before the
/// graph is read.
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
    /// Positional identity for the intern table. The `file` is supplied
    /// by the caller (it owns the `File` at mint — the assembly loop and
    /// [`GraphBuilder::intern_external`]) so [`GraphNode`] need not carry
    /// one and can stay `Default`-constructible for the prefill region.
    pub(crate) fn key(&self, file: File) -> NodeKey {
        NodeKey {
            fqname: self.fqname.clone(),
            kind: self.kind,
            file,
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

    pub(crate) fn intern_node(&mut self, node: GraphNode, file: File) -> usize {
        let key = node.key(file);
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
    /// precomputed global index (`offset[file] + local_idx`) through
    /// disjoint per-file `&mut` slices instead of relying on serial
    /// append order. `nodes` is filled with `GraphNode::default()`
    /// placeholders that the assemble pass's parallel node fill
    /// overwrites exactly once each — the per-file offsets partition
    /// `[0, total_nodes)`, so no placeholder survives into the
    /// finished graph. External nodes minted later via
    /// [`intern_external`] append past this region (index
    /// `total_nodes` onward).
    pub(crate) fn prefill_payload_region(&mut self, total_nodes: usize) {
        self.nodes.resize_with(total_nodes, GraphNode::default);
        self.forward_adj.resize_with(total_nodes, Vec::new);
        self.reverse_adj.resize_with(total_nodes, Vec::new);
        self.node_index.reserve(total_nodes);
    }

    /// Get (or mint) the deduplicated `kind="external"` node for a
    /// resolved non-first-party import target the project doesn't own —
    /// ``[external dist] X`` covers third-party site-packages and
    /// ``[external file] X`` a resolved-but-out-of-tree file. (Genuinely
    /// unresolved imports never reach here: they keep their alias node,
    /// flagged `NodeFlags::UNRESOLVED`.) The resolved target's `path` is
    /// carried so callers can exclude these nodes by path, and its
    /// `file` keys the [`NodeKey`]; position is the (0, 0) sentinel.
    /// Deduped by fqname project-wide; the first `path`/`file` seen for a
    /// given fqname wins, so several submodules of one dist collapse to a
    /// single node.
    pub(crate) fn intern_external(&mut self, fqname: String, path: String, file: File) -> usize {
        if let Some(&idx) = self.external_nodes.get(&fqname) {
            return idx;
        }
        let idx = self.intern_node(positionless_node(fqname.clone(), "external", path, 0), file);
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

    /// Bulk-*initialize* the edge storage from a pre-sorted,
    /// pre-deduplicated triple list — the assemble pass's one big
    /// batch. Requires an edge-less builder (debug-asserted): with no
    /// pre-existing edges to merge against, every piece of the edge
    /// storage is derived from `triples` independently, on concurrent
    /// rayon tasks. Callers should release the GIL around this.
    ///
    /// * `edges` is the triple list itself (moved in, no copy).
    /// * `edge_set` is folded by a sibling task — still needed, since
    ///   later `add_edge` / `extend_edges` calls (peer-stub edges, the
    ///   plugin apply pass) dedup against it.
    /// * `forward_adj`: `triples` is sorted by `(src, dst, flags)`, so
    ///   each node's out-list is one contiguous run; [`scatter_adjacency`]
    ///   partitions the runs into worker chunks and writes each chunk's
    ///   disjoint `forward_adj[lo..hi]` window in parallel.
    /// * `reverse_adj`: the same scatter over a copy re-sorted by
    ///   `(dst, src, flags)`.
    ///
    /// The result — including every per-node adjacency order — is
    /// identical to looping the same triples through [`extend_edges`].
    pub(crate) fn init_edges_bulk(&mut self, triples: Vec<(usize, usize, u8)>) {
        debug_assert!(
            self.edges.is_empty() && self.edge_set.is_empty(),
            "init_edges_bulk requires an edge-less builder"
        );
        debug_assert!(
            triples.windows(2).all(|w| w[0] < w[1]),
            "init_edges_bulk requires sorted + deduplicated triples"
        );
        if triples.is_empty() {
            return;
        }
        let mut edge_set = std::mem::take(&mut self.edge_set);
        let forward = self.forward_adj.as_mut_slice();
        let reverse = self.reverse_adj.as_mut_slice();
        let triples_ref = &triples;
        rayon::join(
            || {
                edge_set.reserve(triples_ref.len());
                edge_set.extend(triples_ref.iter().copied());
            },
            || {
                rayon::join(
                    || {
                        scatter_adjacency(forward, triples_ref, |&(src, dst, flags)| {
                            (src, (dst, flags))
                        })
                    },
                    || {
                        use rayon::prelude::*;
                        let mut by_dst = triples_ref.clone();
                        by_dst.par_sort_unstable_by_key(|&(src, dst, flags)| (dst, src, flags));
                        scatter_adjacency(reverse, &by_dst, |&(src, dst, flags)| {
                            (dst, (src, flags))
                        });
                    },
                );
            },
        );
        self.edge_set = edge_set;
        self.edges = triples;
    }
}

/// Scatter `rows` — sorted by the key half of `key_entry`'s return —
/// into `adj`, writing each key's contiguous run as that node's
/// adjacency list (replacing the empty placeholder `Vec`, so each list
/// is allocated exactly once at its final size).
///
/// This is a partitioned scatter, not a reduction: rows are chunked at
/// run boundaries (~8 chunks per worker for load balance), so chunk
/// `i`'s keys are all strictly smaller than chunk `i + 1`'s and each
/// chunk owns a disjoint `adj[lo..hi]` window carved off with
/// `split_at_mut` — no locks, no merge pass.
#[allow(clippy::type_complexity)]
fn scatter_adjacency(
    adj: &mut [Vec<(usize, u8)>],
    rows: &[(usize, usize, u8)],
    key_entry: fn(&(usize, usize, u8)) -> (usize, (usize, u8)),
) {
    use rayon::prelude::*;
    if rows.is_empty() {
        return;
    }
    let n_chunks = (rayon::current_num_threads() * 8).max(1);
    let approx = rows.len().div_ceil(n_chunks);
    // Row ranges aligned to run boundaries: extend each chunk until the
    // key changes so no run straddles two chunks.
    let mut chunks: Vec<(usize, usize)> = Vec::with_capacity(n_chunks + 1);
    let mut start = 0;
    while start < rows.len() {
        let mut end = (start + approx).min(rows.len());
        let key = key_entry(&rows[end - 1]).0;
        while end < rows.len() && key_entry(&rows[end]).0 == key {
            end += 1;
        }
        chunks.push((start, end));
        start = end;
    }
    // Carve a disjoint `adj` window per chunk. Chunk keys are sorted
    // and chunk-exclusive, so splitting after each chunk's last key
    // hands every window to exactly one worker.
    let mut windows: Vec<(usize, &mut [Vec<(usize, u8)>], usize, usize)> =
        Vec::with_capacity(chunks.len());
    let mut rest = adj;
    let mut consumed = 0usize;
    for &(row_start, row_end) in &chunks {
        let last_key = key_entry(&rows[row_end - 1]).0;
        let (head, tail) = rest.split_at_mut(last_key + 1 - consumed);
        windows.push((consumed, head, row_start, row_end));
        consumed = last_key + 1;
        rest = tail;
    }
    windows
        .into_par_iter()
        .for_each(|(base, window, row_start, row_end)| {
            let mut i = row_start;
            while i < row_end {
                let key = key_entry(&rows[i]).0;
                let mut j = i + 1;
                while j < row_end && key_entry(&rows[j]).0 == key {
                    j += 1;
                }
                window[key - base] = rows[i..j].iter().map(|r| key_entry(r).1).collect();
                i = j;
            }
        });
}

pub(crate) fn not_materialized(op: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ProjectContext.{op}() called outside an active materialize() — \
         did you call it from a plugin's run() method?"
    ))
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

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the graph-building primitives.
    //!
    //! The interning surface mints `Py<SymbolNode>` and so requires
    //! pyo3-init / a GIL; those paths are exercised by the python suite
    //! end-to-end. Here we cover the data-only pieces:
    //!
    //! * `positionless_node` field defaults.
    //! * `bfs` traversal — direction switch + `skip_flags` filtering.
    //!
    //! `NodeKey` identity (positional, flags-excluded) is keyed on a
    //! salsa `File`, which needs a db to construct, so it's covered by
    //! the Python suite's shadowing / cross-file tests rather than here.
    use super::*;

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

    // -- init_edges_bulk ----------------------------------------------------

    /// Deterministic pseudo-random `(src, dst, flags)` triples over
    /// `n_nodes`, sorted + deduplicated — the exact precondition
    /// `init_edges_bulk` documents. Plain LCG; no rng dependency.
    fn make_sorted_triples(n_nodes: usize, n_edges: usize, seed: u64) -> Vec<(usize, usize, u8)> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut t: Vec<(usize, usize, u8)> = (0..n_edges)
            .map(|_| (next() % n_nodes, next() % n_nodes, (next() % 3) as u8))
            .collect();
        t.sort_unstable();
        t.dedup();
        t
    }

    /// Assert every edge-side field of two builders matches, including
    /// per-node adjacency order (the documented bit-for-bit contract).
    fn assert_same_edges(a: &GraphBuilder, b: &GraphBuilder) {
        assert_eq!(a.edges, b.edges);
        assert_eq!(a.edge_set, b.edge_set);
        assert_eq!(a.forward_adj, b.forward_adj);
        assert_eq!(a.reverse_adj, b.reverse_adj);
    }

    #[test]
    fn init_edges_bulk_matches_extend_edges() {
        for (n_nodes, n_edges, seed) in [(1, 5, 7), (10, 40, 1), (50, 600, 2), (500, 5000, 3)] {
            let triples = make_sorted_triples(n_nodes, n_edges, seed);
            let mut serial = empty_builder_with_n_slots(n_nodes);
            serial.extend_edges(triples.clone());
            let mut bulk = empty_builder_with_n_slots(n_nodes);
            bulk.init_edges_bulk(triples);
            assert_same_edges(&serial, &bulk);
        }
    }

    #[test]
    fn init_edges_bulk_empty_is_noop() {
        let mut b = empty_builder_with_n_slots(3);
        b.init_edges_bulk(Vec::new());
        assert!(b.edges.is_empty());
        assert!(b.edge_set.is_empty());
        assert!(b.forward_adj.iter().all(Vec::is_empty));
        assert!(b.reverse_adj.iter().all(Vec::is_empty));
    }

    #[test]
    fn init_edges_bulk_trailing_isolated_nodes_stay_empty() {
        // Edges touch only node 0; nodes 1..4 keep their empty lists
        // (the scatter's final `split_at_mut` window never reaches them).
        let mut b = empty_builder_with_n_slots(5);
        b.init_edges_bulk(vec![(0, 0, 0)]);
        assert_eq!(b.forward_adj[0], vec![(0, 0)]);
        assert!(b.forward_adj[1..].iter().all(Vec::is_empty));
        assert!(b.reverse_adj[1..].iter().all(Vec::is_empty));
    }

    #[test]
    fn init_edges_bulk_seeds_edge_set_for_later_extend() {
        // A later extend_edges (the pass-3 / plugin path) must dedup
        // against the bulk-initialized edge_set.
        let mut b = empty_builder_with_n_slots(3);
        b.init_edges_bulk(vec![(0, 1, 0), (1, 2, 0)]);
        b.extend_edges(vec![(0, 1, 0), (2, 0, 0)]);
        assert_eq!(b.edges, vec![(0, 1, 0), (1, 2, 0), (2, 0, 0)]);
        assert_eq!(b.forward_adj[0], vec![(1, 0)]);
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
