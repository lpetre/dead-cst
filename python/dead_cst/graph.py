"""Public data types of the symbol graph.

:class:`SymbolNode` is what every node in the project graph is.
:class:`Import` captures a cross-file reference (the dotted module
name plus optional decl). :class:`NodeFlags` and :class:`EdgeFlags`
mark structural attributes (``SHADOWED`` decls, ``DEAD_BRANCH`` edges,
explicit ``ENTRYPOINT``\\s).

All four are re-exported straight from the rust extension
(:mod:`dead_cst.native`) — there is no parallel Python copy anymore.

:func:`write_graph` and :func:`read_graph` persist a built graph to
disk. The on-disk format is one header + a bincode body holding the
node / edge lists and a small :class:`GraphMetadata` block. Plugins
deliberately do *not* round-trip — load a graph back and you get a
plain :class:`LoadedGraph` view, no ``ProjectContext``: a project that
needs plugin-emitted edges must rebuild rather than load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from dead_cst import native
from dead_cst.native import (
    EdgeFlags,
    GraphMetadata,
    Import,
    NodeFlags,
    SymbolNode,
)

#: Default seed mask for reachability queries. ORs together every
#: :class:`NodeFlags` bit that semantically "keeps a node alive":
#:
#: * :data:`NodeFlags.ENTRYPOINT` — explicit entrypoints (plugin-emitted
#:   seeds, ``-e`` CLI flag, ``[project.scripts]`` targets, ...).
#: * :data:`NodeFlags.TESTCASE` — pytest / unittest discoveries.
#: * :data:`NodeFlags.NOQA` — imports pinned by a ``# noqa: F401``
#:   directive.
#: * :data:`NodeFlags.NOTEBOOK` — notebook cells (run top-to-bottom,
#:   never imported, always alive).
#:
#: :meth:`Analysis.reachable` / :meth:`Analysis.dead` default to this
#: mask. Pass a subset to scope the question — e.g.
#: ``reachable(seed_flags=NodeFlags.ENTRYPOINT)`` asks "what would be
#: alive if the test suite, ``noqa`` pins, and notebooks didn't exist."
KEEPALIVE_DEFAULT: int = (
    NodeFlags.ENTRYPOINT | NodeFlags.TESTCASE | NodeFlags.NOQA | NodeFlags.NOTEBOOK
)


class LoadedGraph:
    """In-memory view of a graph loaded from disk.

    Mirrors the slice of :class:`native.ProjectContext` the CLI needs
    to answer reachability questions without a live rust context:
    :meth:`nodes`, :meth:`edges`, and :meth:`reachable`. The BFS is
    pure Python — graphs are loaded only when the caller passed
    ``--graph PATH`` instead of rebuilding, so spending one Python
    walk here keeps the ``ProjectContext`` off the critical path.
    """

    __slots__ = ("_nodes", "_edges", "_out", "_index")

    def __init__(
        self,
        nodes: Sequence[SymbolNode],
        edges: Sequence[tuple[int, int, int]],
    ) -> None:
        self._nodes: list[SymbolNode] = list(nodes)
        self._edges: list[tuple[int, int, int]] = list(edges)
        self._index: dict[SymbolNode, int] = {n: i for i, n in enumerate(self._nodes)}
        out: list[list[tuple[int, int]]] = [[] for _ in self._nodes]
        for src, dst, flags in self._edges:
            out[src].append((dst, flags))
        self._out: list[list[tuple[int, int]]] = out

    def nodes(self) -> list[SymbolNode]:
        return list(self._nodes)

    def edges(self) -> list[tuple[int, int, int]]:
        return list(self._edges)

    def reachable(
        self,
        *,
        seed_flags: int = KEEPALIVE_DEFAULT,
        skip_flags: int = 0,
    ) -> list[SymbolNode]:
        """Forward closure from every node whose ``flags & seed_flags``.

        Mirrors :meth:`native.ProjectContext.reachable` — same seed
        semantics, same ``skip_flags`` masking on edges — implemented
        as a Python BFS over the loaded adjacency list.
        """
        seeds = [i for i, n in enumerate(self._nodes) if n.flags & seed_flags]
        visited: set[int] = set()
        stack: list[int] = list(seeds)
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            for j, flags in self._out[i]:
                if skip_flags and (flags & skip_flags):
                    continue
                stack.append(j)
        return [self._nodes[i] for i in visited]


def _node_iterable(obj: object) -> list[SymbolNode]:
    nodes_fn = getattr(obj, "nodes", None)
    if nodes_fn is None:
        raise TypeError(
            f"write_graph: expected a ProjectContext or LoadedGraph, got {type(obj).__name__!r}"
        )
    return list(nodes_fn())


def _edge_iterable(obj: object) -> list[tuple[int, int, int]]:
    edges_fn = getattr(obj, "edges", None)
    if edges_fn is None:
        raise TypeError(
            f"write_graph: expected a ProjectContext or LoadedGraph, got {type(obj).__name__!r}"
        )
    return list(edges_fn())


def write_graph(
    path: Path | str,
    graph: native.ProjectContext | LoadedGraph,
    meta: Sequence[tuple[str, str]] = (),
) -> None:
    """Persist ``graph`` to ``path`` as a bincode-encoded file.

    ``graph`` is either a live :class:`native.ProjectContext` (the
    result of :meth:`Analysis.materialize_all`) or a previously-loaded
    :class:`LoadedGraph`. ``meta`` is a sequence of ``(key, value)``
    pairs stored verbatim in the file's metadata block — the CLI
    threads ``--meta key=value`` flags here. Plugins are not
    serialized; loading a graph back gives you a :class:`LoadedGraph`,
    so a project that needs plugin-emitted edges must rebuild rather
    than restore.
    """
    nodes = _node_iterable(graph)
    edges = _edge_iterable(graph)
    native.write_graph(str(path), nodes, edges, list(meta))


def read_graph(path: Path | str) -> tuple[LoadedGraph, GraphMetadata]:
    """Load a graph previously written by :func:`write_graph`.

    Raises ``ValueError`` when the file's magic bytes or format
    version don't match this build — rebuilding the graph is cheap,
    so the loader is intentionally strict instead of attempting
    in-place migrations.
    """
    native_graph, metadata = native.read_graph(str(path))
    return LoadedGraph(list(native_graph.nodes), list(native_graph.edges)), metadata


__all__ = [
    "KEEPALIVE_DEFAULT",
    "EdgeFlags",
    "GraphMetadata",
    "Import",
    "LoadedGraph",
    "NodeFlags",
    "SymbolNode",
    "read_graph",
    "write_graph",
]
