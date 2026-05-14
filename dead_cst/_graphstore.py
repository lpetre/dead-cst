"""Minimal :mod:`rustworkx` wrapper for the symbol graph.

:class:`SymbolGraph` holds a :class:`rustworkx.PyDiGraph` plus a
``SymbolNode -> int`` map. It exposes the index bookkeeping and trivial
``SymbolNode``-keyed traversal sugar (because that sugar *is* the reason
the index map exists). Anything beyond that -- edge-payload-aware
iteration, in-place edge removal, algorithm calls -- goes through
:attr:`SymbolGraph.raw` directly using rustworkx primitives.

The graph is always built as ``multigraph=True`` to match the
``networkx.MultiDiGraph`` behaviour the codebase relied on; cross-package
edge dedup is enforced by :func:`dead_cst.analyze._claim_edge`, not by
the graph itself.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import rustworkx as rx

from .graph import EdgeFlags, SymbolNode


class SymbolGraph:
    """``rustworkx.PyDiGraph`` + ``SymbolNode <-> int`` index map."""

    __slots__ = ("raw", "_idx")

    def __init__(self) -> None:
        self.raw: rx.PyDiGraph = rx.PyDiGraph(multigraph=True)
        self._idx: dict[SymbolNode, int] = {}

    def add(self, node: SymbolNode) -> int:
        i = self._idx.get(node)
        if i is None:
            i = self.raw.add_node(node)
            self._idx[node] = i
        return i

    def add_edge(self, src: SymbolNode, dst: SymbolNode, flags: EdgeFlags = EdgeFlags.NONE) -> None:
        # Auto-insert missing endpoints: matches networkx behavior the
        # callers relied on. ``resolve_edges`` emits edges to synthetic
        # external nodes that are never explicitly added.
        si = self.add(src)
        di = self.add(dst)
        self.raw.add_edge(si, di, flags)

    def index(self, node: SymbolNode) -> int:
        return self._idx[node]

    def node(self, idx: int) -> SymbolNode:
        return self.raw[idx]

    def relabel(self, old: SymbolNode, new: SymbolNode) -> None:
        """Swap a node's payload in place, keeping its index stable."""
        i = self._idx.pop(old)
        self.raw[i] = new
        self._idx[new] = i

    def subgraph(self, nodes: Iterable[SymbolNode]) -> SymbolGraph:
        """Return a new ``SymbolGraph`` induced by ``nodes``.

        Uses :meth:`rustworkx.PyDiGraph.subgraph` under the hood, which
        preserves edge payloads but renumbers node indices in the
        returned graph -- the wrapper's index map is rebuilt accordingly.
        """
        keep_idx = [self._idx[n] for n in nodes if n in self._idx]
        sub_raw = self.raw.subgraph(keep_idx)
        out = SymbolGraph.__new__(SymbolGraph)
        out.raw = sub_raw
        out._idx = {sub_raw[i]: i for i in sub_raw.node_indices()}
        return out

    def __contains__(self, node: object) -> bool:
        return node in self._idx

    def __iter__(self) -> Iterator[SymbolNode]:
        return iter(self._idx)

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def nodes(self) -> Iterator[SymbolNode]:
        return iter(self._idx)

    def successors(self, node: SymbolNode) -> Iterator[SymbolNode]:
        for succ in self.raw.successors(self._idx[node]):
            yield succ

    def predecessors(self, node: SymbolNode) -> Iterator[SymbolNode]:
        for pred in self.raw.predecessors(self._idx[node]):
            yield pred
