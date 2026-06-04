"""Public data types of the symbol graph.

:class:`SymbolNode` is what every node in the project graph is.
:class:`Import` captures a cross-file reference (the dotted module
name plus optional decl). :class:`NodeFlags` and :class:`EdgeFlags`
mark structural attributes (``NOQA``-pinned imports, ``DEAD_BRANCH``
edges, explicit ``ENTRYPOINT``\\s).

All four are re-exported straight from the rust extension
(:mod:`dead_cst._native`) — there is no parallel Python copy anymore.

:func:`write_graph` and :func:`read_graph` persist a built graph to
disk. The on-disk format is one header + a bincode body holding the
node / edge lists and a small :class:`GraphMetadata` block. Plugins
deliberately do *not* round-trip — load a graph back and you get a
plain :class:`LoadedGraph` view, no ``ProjectContext``: a project that
needs plugin-emitted edges must rebuild rather than load.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from dead_cst import _native as _native_mod
from dead_cst._native import (
    EdgeFlags,
    GraphMetadata,
    Import,
    NodeFlags,
    SymbolNode,
)

if TYPE_CHECKING:
    from dead_cst import _native as native

#: A flag-registry entry: ``(name, bit, seed, default_on, description)``.
FlagEntry = tuple[str, int, bool, bool, str]


def _seed_mask_from_registry(registry: Sequence[FlagEntry]) -> int:
    """OR of every node-flag bit whose flag both seeds reachability and is
    on by default — the registry-derived replacement for the old
    hand-maintained ``KEEPALIVE_DEFAULT`` constant."""
    mask = 0
    for _name, bit, seed, default_on, _description in registry:
        if seed and default_on:
            mask |= bit
    return mask


class LoadedGraph:
    """In-memory view of a graph loaded from disk.

    Mirrors the slice of :class:`native.ProjectContext` the CLI needs
    to answer reachability questions without a live rust context:
    :meth:`nodes`, :meth:`edges`, and :meth:`reachable`. The BFS is
    pure Python — graphs are loaded only when the caller passed
    ``--graph PATH`` instead of rebuilding, so spending one Python
    walk here keeps the ``ProjectContext`` off the critical path.
    """

    __slots__ = (
        "_nodes",
        "_edges",
        "_out",
        "_index",
        "_node_flag_registry",
        "_edge_flag_registry",
        "_default_seed_mask",
    )

    def __init__(
        self,
        nodes: Sequence[SymbolNode],
        edges: Sequence[tuple[int, int, int]],
        node_flag_registry: Sequence[FlagEntry] = (),
        edge_flag_registry: Sequence[FlagEntry] = (),
    ) -> None:
        self._nodes: list[SymbolNode] = list(nodes)
        self._edges: list[tuple[int, int, int]] = list(edges)
        self._index: dict[SymbolNode, int] = {n: i for i, n in enumerate(self._nodes)}
        out: list[list[tuple[int, int]]] = [[] for _ in self._nodes]
        for src, dst, flags in self._edges:
            out[src].append((dst, flags))
        self._out: list[list[tuple[int, int]]] = out
        self._node_flag_registry: list[FlagEntry] = list(node_flag_registry)
        self._edge_flag_registry: list[FlagEntry] = list(edge_flag_registry)
        self._default_seed_mask: int = _seed_mask_from_registry(self._node_flag_registry)

    def nodes(self) -> list[SymbolNode]:
        return list(self._nodes)

    def edges(self) -> list[tuple[int, int, int]]:
        return list(self._edges)

    def node_flag_registry(self) -> list[FlagEntry]:
        """The node-flag registry loaded from the graph file, so a
        loaded-then-rewritten graph preserves it. Empty for a graph
        written before format version 2."""
        return list(self._node_flag_registry)

    def edge_flag_registry(self) -> list[FlagEntry]:
        """The edge-flag registry loaded from the graph file."""
        return list(self._edge_flag_registry)

    def default_seed_mask(self) -> int:
        """Registry-derived default keepalive mask (see
        :func:`_seed_mask_from_registry`)."""
        return self._default_seed_mask

    def node_flag(self, name: str) -> int | None:
        """Resolve a node-flag bit by its registered ``owner/name`` from
        the loaded registry, mirroring
        :meth:`native.ProjectContext.node_flag`. ``None`` if no flag with
        that name is recorded in the file."""
        for entry_name, bit, *_rest in self._node_flag_registry:
            if entry_name == name:
                return bit
        return None

    def reachable(
        self,
        *,
        seed_flags: int | None = None,
        skip_flags: int = 0,
    ) -> list[SymbolNode]:
        """Forward closure from every node whose ``flags & seed_flags``.

        Mirrors :meth:`native.ProjectContext.reachable` — same seed
        semantics, same ``skip_flags`` masking on edges — implemented
        as a Python BFS over the loaded adjacency list. ``seed_flags``
        defaults to the registry-derived :meth:`default_seed_mask`.
        """
        if seed_flags is None:
            seed_flags = self._default_seed_mask
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


def _flag_registry(obj: object, attr: str) -> list[FlagEntry]:
    """Read a ``node_flag_registry`` / ``edge_flag_registry`` table off a
    duck-typed graph. Both :class:`native.ProjectContext` and
    :class:`LoadedGraph` expose them as methods; a graph without them
    (e.g. a third-party duck type) serializes an empty table."""
    fn = getattr(obj, attr, None)
    if fn is None:
        return []
    return [tuple(entry) for entry in fn()]


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
    node_flag_registry = _flag_registry(graph, "node_flag_registry")
    edge_flag_registry = _flag_registry(graph, "edge_flag_registry")
    _native_mod.write_graph(
        str(path), nodes, edges, list(meta), node_flag_registry, edge_flag_registry
    )


def read_graph(path: Path | str) -> tuple[LoadedGraph, GraphMetadata]:
    """Load a graph previously written by :func:`write_graph`.

    Raises ``ValueError`` when the file's magic bytes or format
    version don't match this build — rebuilding the graph is cheap,
    so the loader is intentionally strict instead of attempting
    in-place migrations.
    """
    native_graph, metadata = _native_mod.read_graph(str(path))
    return (
        LoadedGraph(
            list(native_graph.nodes),
            list(native_graph.edges),
            list(metadata.node_flag_registry),
            list(metadata.edge_flag_registry),
        ),
        metadata,
    )


__all__ = [
    "EdgeFlags",
    "GraphMetadata",
    "Import",
    "LoadedGraph",
    "NodeFlags",
    "SymbolNode",
    "read_graph",
    "write_graph",
]
