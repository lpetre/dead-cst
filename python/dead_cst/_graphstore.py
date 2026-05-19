"""Pure-Python adjacency-list symbol graph.

:class:`SymbolGraph` is a thin dict-of-lists graph: ``_out`` and
``_in`` map a node index to a list of ``(other_index, edge_flags)``
tuples, and a ``SymbolNode -> int`` map interns each node to a stable
integer.

The public API mirrors the rustworkx-backed predecessor for the
methods callers actually used (:meth:`successor_indices`,
:meth:`predecessor_indices`, :meth:`in_degree`, :meth:`out_edges`,
:meth:`has_edge`, :meth:`get_all_edge_data`, :meth:`edge_list`,
:meth:`weighted_edge_list`, :meth:`nodes`, plus indexing
``graph[i]``). :attr:`SymbolGraph.raw` is kept as an alias for
``self`` so legacy ``graph.raw.successor_indices(i)`` calls keep
working without churn through tests/conftest.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .graph import SymbolNode


class SymbolGraph:
    """Dict-of-lists adjacency graph keyed on :class:`SymbolNode`."""

    __slots__ = ("_idx", "_nodes", "_out", "_in")

    def __init__(self) -> None:
        # Index -> node payload (positional).
        self._nodes: list[SymbolNode] = []
        # Node -> index (dedup map).
        self._idx: dict[SymbolNode, int] = {}
        # Index -> list[(neighbor_idx, flags)].
        self._out: list[list[tuple[int, int]]] = []
        self._in: list[list[tuple[int, int]]] = []

    # ----- core mutation ----------------------------------------------------

    def add(self, node: SymbolNode) -> int:
        i = self._idx.get(node)
        if i is None:
            i = len(self._nodes)
            self._nodes.append(node)
            self._idx[node] = i
            self._out.append([])
            self._in.append([])
        return i

    def add_node(self, node: SymbolNode) -> int:
        """rustworkx-compatible alias for :meth:`add`."""
        return self.add(node)

    def add_edge(self, src: SymbolNode, dst: SymbolNode, flags: int = 0) -> None:
        # Auto-insert missing endpoints to match networkx behaviour
        # the callers relied on. Multigraph: duplicate edges allowed.
        si = self.add(src)
        di = self.add(dst)
        self._out[si].append((di, flags))
        self._in[di].append((si, flags))

    # ----- index bookkeeping ------------------------------------------------

    def index(self, node: SymbolNode) -> int:
        return self._idx[node]

    def node(self, idx: int) -> SymbolNode:
        return self._nodes[idx]

    def __getitem__(self, idx: int) -> SymbolNode:
        # rustworkx ``graph.raw[i]`` -> payload.
        return self._nodes[idx]

    # ----- iteration --------------------------------------------------------

    def __contains__(self, node: object) -> bool:
        return node in self._idx

    def __iter__(self) -> Iterator[SymbolNode]:
        return iter(self._idx)

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def nodes(self):
        # The codebase uses ``graph.nodes`` (no parens) for iteration
        # *and* ``graph.raw.nodes()`` (with parens) for the rustworkx
        # accessor. Return a small helper object that satisfies both.
        return _NodesView(self._nodes)

    def node_indices(self) -> list[int]:
        return list(range(len(self._nodes)))

    # ----- neighbor queries (rustworkx-compatible) --------------------------

    def successor_indices(self, i: int) -> list[int]:
        return [j for j, _ in self._out[i]]

    def predecessor_indices(self, i: int) -> list[int]:
        return [j for j, _ in self._in[i]]

    def out_edges(self, i: int) -> list[tuple[int, int, int]]:
        return [(i, j, flags) for j, flags in self._out[i]]

    def in_edges(self, i: int) -> list[tuple[int, int, int]]:
        return [(j, i, flags) for j, flags in self._in[i]]

    def out_degree(self, i: int) -> int:
        return len(self._out[i])

    def in_degree(self, i: int) -> int:
        return len(self._in[i])

    def has_edge(self, u: int, v: int) -> bool:
        return any(j == v for j, _ in self._out[u])

    def get_all_edge_data(self, u: int, v: int) -> list[int]:
        return [flags for j, flags in self._out[u] if j == v]

    def edge_list(self) -> list[tuple[int, int]]:
        return [(u, j) for u, neighbors in enumerate(self._out) for j, _ in neighbors]

    def weighted_edge_list(self) -> list[tuple[int, int, int]]:
        return [(u, j, flags) for u, neighbors in enumerate(self._out) for j, flags in neighbors]

    # ----- structural ops ---------------------------------------------------

    def subgraph(self, nodes: Iterable[SymbolNode]) -> "SymbolGraph":
        """Return a new :class:`SymbolGraph` induced by ``nodes``.

        Edges between two kept nodes are preserved (with payload);
        edges that touch a dropped node are skipped. Node indices are
        renumbered in the result.
        """
        keep_set = {n for n in nodes if n in self._idx}
        out = SymbolGraph()
        for n in self._nodes:
            if n in keep_set:
                out.add(n)
        for u, neighbors in enumerate(self._out):
            src = self._nodes[u]
            if src not in keep_set:
                continue
            for v, flags in neighbors:
                dst = self._nodes[v]
                if dst not in keep_set:
                    continue
                out.add_edge(src, dst, flags)
        return out

    # ----- back-compat shim -------------------------------------------------

    @property
    def raw(self) -> "SymbolGraph":
        """Legacy ``graph.raw.X`` accessor — same object as ``self``."""
        return self


class _NodesView:
    """``graph.nodes`` -> iterable; ``graph.nodes()`` -> the same list.

    Bridges the historic split where ``graph.nodes`` was a property
    yielding payloads but ``graph.raw.nodes()`` was a callable.
    """

    __slots__ = ("_payloads",)

    def __init__(self, payloads: list[SymbolNode]) -> None:
        self._payloads = payloads

    def __iter__(self) -> Iterator[SymbolNode]:
        return iter(self._payloads)

    def __len__(self) -> int:
        return len(self._payloads)

    def __call__(self) -> list[SymbolNode]:
        return list(self._payloads)

    def __contains__(self, node: object) -> bool:
        return node in self._payloads
