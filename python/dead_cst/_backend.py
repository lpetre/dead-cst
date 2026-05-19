"""Native (rust) backend bridge.

The rust crate (``dead_cst._native``) builds the project graph
end-to-end using ty's ``SemanticIndex``. This module is a thin
adapter: it instantiates a :class:`native.ProjectContext`, wires
plugins, calls :meth:`materialize`, and folds the rust-shaped
:class:`NativeGraph` envelope into a :class:`SymbolGraph` (a plain
dict-of-lists adjacency keyed on :class:`SymbolNode`).

Nodes / imports / flags are no longer translated — :class:`SymbolNode`
*is* :class:`native.SymbolNode`, :class:`Import` *is*
:class:`native.Import`, etc. The "bridge" today is one pass that
copies the rust node list and edge triples into the adjacency map.

:func:`materialize_project` returns a ``(ctx, graph)`` pair: the
``ctx`` is held by :class:`Analysis` so bulk reachability queries
(:meth:`Analysis.reachable`, :meth:`Analysis.dead`, etc.) can delegate
to the rust BFS one FFI hop at a time instead of walking the Python
adjacency list per node.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ._graphstore import SymbolGraph

if TYPE_CHECKING:
    from dead_cst import _native as native


def materialize_project(
    project_root: Path,
    plugins: Sequence[object] = (),
    src_roots: Sequence[Path] = (),
    *,
    show_progress: bool = False,
) -> tuple["native.ProjectContext", SymbolGraph]:
    """Materialize ``project_root`` end-to-end via the rust backend.

    Builds a :class:`native.ProjectContext` rooted at ``project_root``,
    registers each plugin's ``run(ctx)`` callback, calls
    :meth:`materialize`, and bridges the resulting :class:`NativeGraph`
    into a :class:`SymbolGraph`. Every plugin must be an instance of
    :class:`dead_cst.plugins.Plugin`; anything else raises
    :class:`TypeError` so typos (``Pluign()``) surface immediately
    instead of being silently dropped.

    Returns the ``(ctx, graph)`` pair so the caller (typically
    :class:`Analysis`) can route bulk reachability queries through the
    rust BFS via :meth:`native.ProjectContext.reachable` /
    :meth:`descendants` / :meth:`ancestors`.

    ``show_progress=True`` makes the rust backend draw indicatif progress
    bars to stderr for each of the three per-file phases plus the
    plugin pass. The CLI sets this; the library API leaves it off.
    indicatif auto-hides on non-TTY stderr.
    """
    from dead_cst import _native as native
    from .plugins import Plugin

    ctx = native.ProjectContext(
        str(project_root),
        src_roots=[str(p) for p in src_roots] if src_roots else None,
        show_progress=show_progress,
    )
    for plugin in plugins:
        if not isinstance(plugin, Plugin):
            raise TypeError(
                f"Expected a dead_cst.plugins.Plugin instance, got "
                f"{type(plugin).__name__!r}: {plugin!r}"
            )
        ctx.add_plugin(plugin)
    graph = _bridge(ctx.materialize())
    return ctx, graph


def _bridge(graph: "native.NativeGraph") -> SymbolGraph:
    """Convert a project-wide :class:`NativeGraph` into a fresh :class:`SymbolGraph`.

    Uses :meth:`SymbolGraph._populate_from_native` so the edge fan-out
    works in integer-index space and never re-hashes endpoint nodes --
    on a 10^6-node / 10^7-edge graph that's the difference between
    ~4 s and ~150 ms.
    """
    out = SymbolGraph()
    out._populate_from_native(list(graph.nodes), graph.edges)
    return out
