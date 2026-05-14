"""rustworkx-backed graph wrappers that expose the networkx API used by dead-cst.

The analyzer historically grew on top of :mod:`networkx`. ``rustworkx``
ships the same directed-multigraph data structure at a fraction of
the cost, but its native API is index-based rather than node-keyed.
The wrappers in this module keep the call sites unchanged: pass a
hashable :class:`SymbolNode` (or any hashable) and the wrapper looks
up the underlying integer index. Two classes are exposed --
:class:`MultiDiGraph` (allows parallel edges) and :class:`DiGraph`
(does not). They are intentionally small: only the methods the
analyzer, codemod, CLI, plugins, and tests actually call are
implemented.
"""

from __future__ import annotations

from typing import Any, Generic, Hashable, Iterable, Iterator, Mapping, Self, TypeVar, overload

import rustworkx as rx

NodeT = TypeVar("NodeT", bound=Hashable)


class _NodesView(Generic[NodeT]):
    """``g.nodes``: iterable + container view over node objects."""

    __slots__ = ("_graph",)

    def __init__(self, graph: _BaseGraph[NodeT]) -> None:
        self._graph = graph

    def __iter__(self) -> Iterator[NodeT]:
        return iter(self._graph._node_index)

    def __contains__(self, node: object) -> bool:
        return node in self._graph._node_index

    def __len__(self) -> int:
        return len(self._graph._node_index)


class _BaseGraph(Generic[NodeT]):
    """Shared implementation for :class:`MultiDiGraph` and :class:`DiGraph`.

    Node identity follows networkx semantics: any hashable may be a
    node, and re-adding the same node is a no-op. The wrapper maintains
    a ``dict[node, int]`` mapping to translate between user-facing
    nodes and the integer indices ``rustworkx.PyDiGraph`` uses.
    """

    __slots__ = ("_inner", "_node_index", "graph")

    _multigraph: bool = False

    def __init__(self) -> None:
        self._inner: rx.PyDiGraph = rx.PyDiGraph(multigraph=self._multigraph)
        self._node_index: dict[NodeT, int] = {}
        # Mirrors ``networkx.Graph.graph``: a free-form dict for
        # graph-level attributes the analyzer stashes alongside the
        # structural data (today: the per-file ``dead_suites`` map).
        self.graph: dict[str, Any] = {}

    # ---------- construction ----------

    def add_node(self, node: NodeT) -> int:
        idx = self._node_index.get(node)
        if idx is not None:
            return idx
        idx = self._inner.add_node(node)
        self._node_index[node] = idx
        return idx

    def add_nodes_from(self, nodes: Iterable[NodeT]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, u: NodeT, v: NodeT, **attr: Any) -> None:
        ui = self.add_node(u)
        vi = self.add_node(v)
        self._inner.add_edge(ui, vi, dict(attr) if attr else {})

    def add_edges_from(
        self,
        edges: Iterable[
            tuple[NodeT, NodeT]
            | tuple[NodeT, NodeT, Mapping[str, Any]]
            | tuple[NodeT, NodeT, Any, Mapping[str, Any]]
        ],
    ) -> None:
        """Add edges in any of networkx's accepted tuple shapes.

        Mirrors :meth:`networkx.MultiDiGraph.add_edges_from`: 2-tuples
        are ``(u, v)``; 3-tuples carry an attribute dict in the trailing
        slot (the multigraph "key" variant isn't used here, so anything
        non-dict in that slot is treated as edge attributes); 4-tuples
        are ``(u, v, key, data)`` where ``key`` is discarded -- the
        wrapper doesn't expose stable edge keys.
        """
        for edge in edges:
            if len(edge) == 2:
                u, v = edge  # type: ignore[misc]
                self.add_edge(u, v)
            elif len(edge) == 3:
                u, v, data = edge  # type: ignore[misc]
                if isinstance(data, Mapping):
                    self.add_edge(u, v, **data)
                else:
                    # Treat non-dict third element as edge "key" /
                    # ignored discriminator; preserve the bare edge.
                    self.add_edge(u, v)
            elif len(edge) == 4:
                u, v, _key, data = edge  # type: ignore[misc]
                if isinstance(data, Mapping):
                    self.add_edge(u, v, **data)
                else:
                    self.add_edge(u, v)
            else:  # pragma: no cover - defensive
                raise ValueError(f"unrecognized edge tuple: {edge!r}")

    # ---------- removal ----------

    def remove_edge(self, u: NodeT, v: NodeT) -> None:
        ui = self._node_index[u]
        vi = self._node_index[v]
        self._inner.remove_edge(ui, vi)

    def remove_node(self, node: NodeT) -> None:
        idx = self._node_index.pop(node)
        self._inner.remove_node(idx)

    # ---------- queries ----------

    def has_node(self, node: NodeT) -> bool:
        return node in self._node_index

    def has_edge(self, u: NodeT, v: NodeT) -> bool:
        ui = self._node_index.get(u)
        vi = self._node_index.get(v)
        if ui is None or vi is None:
            return False
        return self._inner.has_edge(ui, vi)

    def successors(self, node: NodeT) -> Iterator[NodeT]:
        idx = self._node_index[node]
        return iter(self._inner.successors(idx))

    def predecessors(self, node: NodeT) -> Iterator[NodeT]:
        idx = self._node_index[node]
        return iter(self._inner.predecessors(idx))

    def in_degree(self, node: NodeT) -> int:
        return self._inner.in_degree(self._node_index[node])

    def out_degree(self, node: NodeT) -> int:
        return self._inner.out_degree(self._node_index[node])

    @overload
    def out_edges(self, node: NodeT) -> Iterator[tuple[NodeT, NodeT]]: ...
    @overload
    def out_edges(
        self, node: NodeT, *, data: bool
    ) -> Iterator[tuple[NodeT, NodeT, dict[str, Any]]]: ...
    def out_edges(
        self, node: NodeT, *, data: bool = False
    ) -> Iterator[tuple[NodeT, NodeT] | tuple[NodeT, NodeT, dict[str, Any]]]:
        idx = self._node_index[node]
        for u_idx, v_idx, attrs in self._inner.out_edges(idx):
            u = self._inner[u_idx]
            v = self._inner[v_idx]
            if data:
                yield (u, v, attrs if attrs is not None else {})
            else:
                yield (u, v)

    @overload
    def in_edges(self, node: NodeT) -> Iterator[tuple[NodeT, NodeT]]: ...
    @overload
    def in_edges(
        self, node: NodeT, *, data: bool
    ) -> Iterator[tuple[NodeT, NodeT, dict[str, Any]]]: ...
    def in_edges(
        self, node: NodeT, *, data: bool = False
    ) -> Iterator[tuple[NodeT, NodeT] | tuple[NodeT, NodeT, dict[str, Any]]]:
        idx = self._node_index[node]
        for u_idx, v_idx, attrs in self._inner.in_edges(idx):
            u = self._inner[u_idx]
            v = self._inner[v_idx]
            if data:
                yield (u, v, attrs if attrs is not None else {})
            else:
                yield (u, v)

    @overload
    def edges(
        self, nbunch: NodeT | Iterable[NodeT] | None = None, *, keys: bool = False
    ) -> Iterator[tuple[NodeT, NodeT]]: ...
    @overload
    def edges(
        self,
        nbunch: NodeT | Iterable[NodeT] | None = None,
        *,
        data: bool,
        keys: bool = False,
    ) -> Iterator[tuple[NodeT, NodeT, dict[str, Any]]]: ...
    def edges(
        self,
        nbunch: NodeT | Iterable[NodeT] | None = None,
        *,
        data: bool = False,
        keys: bool = False,
    ) -> Iterator[tuple[NodeT, NodeT] | tuple[NodeT, NodeT, dict[str, Any]]]:
        """Iterate every ``(u, v[, data])`` edge in the graph.

        Matches the networkx call shapes the codebase uses:
        ``g.edges()`` walks the full edge list, while ``g.edges(node)``
        restricts the iteration to edges leaving ``node`` (same
        contract as :meth:`out_edges`). The ``keys`` argument is
        accepted for source compatibility -- we don't expose per-edge
        keys -- and is otherwise ignored.
        """
        del keys
        if nbunch is None:
            for u_idx, v_idx, attrs in self._inner.weighted_edge_list():
                u = self._inner[u_idx]
                v = self._inner[v_idx]
                if data:
                    yield (u, v, attrs if attrs is not None else {})
                else:
                    yield (u, v)
            return
        if isinstance(nbunch, (str, bytes)) or not hasattr(nbunch, "__iter__"):
            nodes: Iterable[NodeT] = (nbunch,)  # type: ignore[assignment]
        else:
            nodes = nbunch
        for node in nodes:
            yield from self.out_edges(node, data=data)

    def number_of_nodes(self) -> int:
        return self._inner.num_nodes()

    def number_of_edges(self) -> int:
        return self._inner.num_edges()

    # ---------- views ----------

    @property
    def nodes(self) -> _NodesView[NodeT]:
        return _NodesView(self)

    def __contains__(self, node: object) -> bool:
        return node in self._node_index

    def __len__(self) -> int:
        return len(self._node_index)

    def __iter__(self) -> Iterator[NodeT]:
        return iter(self._node_index)

    # ---------- copy / subgraph ----------

    def copy(self) -> Self:
        new = type(self)()
        for node in self._node_index:
            new.add_node(node)
        for u, v, attrs in self._inner.weighted_edge_list():
            new._inner.add_edge(
                new._node_index[self._inner[u]],
                new._node_index[self._inner[v]],
                dict(attrs) if isinstance(attrs, Mapping) else {},
            )
        new.graph = dict(self.graph)
        return new

    def subgraph(self, nodes: Iterable[NodeT]) -> Self:
        """Return a new graph containing ``nodes`` and the edges between them.

        Unlike :meth:`networkx.Graph.subgraph` (which returns a read-only
        view sharing storage with the original) this returns an
        independent graph. Tests and call sites that follow up with
        ``.copy()`` (a common networkx idiom) still work because
        :meth:`copy` is a no-op rename on top of an already-detached
        graph.
        """
        kept = [n for n in nodes if n in self._node_index]
        kept_set = set(kept)
        new = type(self)()
        for node in kept:
            new.add_node(node)
        for u_idx, v_idx, attrs in self._inner.weighted_edge_list():
            u = self._inner[u_idx]
            v = self._inner[v_idx]
            if u in kept_set and v in kept_set:
                new._inner.add_edge(
                    new._node_index[u],
                    new._node_index[v],
                    dict(attrs) if isinstance(attrs, Mapping) else {},
                )
        # ``networkx.subgraph`` propagates the parent's graph-level
        # attribute dict; mirror that so call sites that read
        # ``g.graph["dead_suites"]`` off a subgraph keep working.
        new.graph = dict(self.graph)
        return new


class MultiDiGraph(_BaseGraph[NodeT]):
    """Directed multigraph (parallel edges allowed)."""

    _multigraph: bool = True


class DiGraph(_BaseGraph[NodeT]):
    """Directed graph (at most one edge per ordered pair)."""

    _multigraph: bool = False


def relabel_nodes(
    graph: _BaseGraph[NodeT], mapping: Mapping[NodeT, NodeT], *, copy: bool = True
) -> _BaseGraph[NodeT]:
    """In-place or copying node relabel; mirrors :func:`networkx.relabel_nodes`.

    Each ``old -> new`` pair swaps the *data* stored at the underlying
    rustworkx index, so every incoming and outgoing edge is preserved
    without touching edge storage. With ``copy=False`` the operation is
    O(len(mapping)) and returns the same graph object; ``copy=True``
    first clones and then mutates, matching the networkx contract.

    Today's only caller is the ``mark_entrypoint`` test fixture, which
    re-binds a :class:`SymbolNode` to a copy carrying an extra flag bit
    after the graph has been built.
    """
    target = graph.copy() if copy else graph
    for old, new in mapping.items():
        idx = target._node_index.pop(old, None)
        if idx is None:
            continue
        target._inner[idx] = new
        target._node_index[new] = idx
    return target


__all__ = ["DiGraph", "MultiDiGraph", "relabel_nodes"]
