"""Public data types of the symbol graph.

:class:`SymbolNode` is what every node in the project graph is.
:class:`Import` captures a cross-file reference (the dotted module
name plus optional decl). :class:`NodeFlags` and :class:`EdgeFlags`
mark structural attributes (``SHADOWED`` decls, ``DEAD_BRANCH`` edges,
explicit ``ENTRYPOINT``\\s).

All four are re-exported straight from the rust extension
(:mod:`dead_cst._native`) — there is no parallel Python copy anymore.

:func:`write_graph` and :func:`read_graph` persist a built graph to
disk. The on-disk format is one header + a bincode body holding the
node / edge lists and a small :class:`GraphMetadata` block; plugins
deliberately do *not* round-trip — load a graph and you get the
materialized adjacency, no ``ProjectContext``.
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
    from ._graphstore import SymbolGraph

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


def write_graph(
    path: Path | str,
    graph: "SymbolGraph",
    meta: Sequence[tuple[str, str]] = (),
) -> None:
    """Persist ``graph`` to ``path`` as a bincode-encoded file.

    ``meta`` is a sequence of ``(key, value)`` pairs stored verbatim
    in the file's metadata block — the CLI threads ``--meta key=value``
    flags here. Plugins are not serialized; loading a graph back gives
    you the adjacency only, so a project that needs plugin-emitted
    edges must rebuild rather than restore.
    """
    nodes = list(graph.nodes)
    edges = graph.weighted_edge_list()
    _native_mod.write_graph(str(path), nodes, edges, list(meta))


def read_graph(path: Path | str) -> "tuple[SymbolGraph, GraphMetadata]":
    """Load a graph previously written by :func:`write_graph`.

    Raises ``ValueError`` when the file's magic bytes or format
    version don't match this build — rebuilding the graph is cheap, so
    the loader is intentionally strict instead of attempting in-place
    migrations.
    """
    from ._graphstore import SymbolGraph

    native_graph, metadata = _native_mod.read_graph(str(path))
    sg = SymbolGraph()
    sg._populate_from_native(list(native_graph.nodes), native_graph.edges)
    return sg, metadata


__all__ = [
    "KEEPALIVE_DEFAULT",
    "EdgeFlags",
    "GraphMetadata",
    "Import",
    "NodeFlags",
    "SymbolNode",
    "read_graph",
    "write_graph",
]
