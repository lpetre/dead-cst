"""Native (rust) backend bridge.

The rust crate (``dead_cst._native``) builds the project graph
end-to-end using ty's ``SemanticIndex``. This module is a thin
adapter: it instantiates a :class:`native.ProjectContext`, wires
plugins, calls :meth:`materialize`, and folds the rust-shaped
:class:`NativeGraph` envelope into a :class:`SymbolGraph` (a plain
dict-of-lists adjacency keyed on :class:`NativeNode`).

Nodes / imports / flags are no longer translated — :class:`SymbolNode`
*is* :class:`native.NativeNode`, :class:`Import` *is*
:class:`native.Import`, etc. The "bridge" today is one pass that
copies the rust node list and edge triples into the adjacency map.
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
) -> SymbolGraph:
    """Materialize ``project_root`` end-to-end via the rust backend.

    Builds a :class:`native.ProjectContext` rooted at ``project_root``,
    registers each plugin's ``run(ctx)`` callback, calls
    :meth:`materialize`, and bridges the resulting :class:`NativeGraph`
    into a :class:`SymbolGraph`. Plugins that don't implement the
    rust ``run(ctx)`` protocol are silently skipped.

    ``show_progress=True`` makes the rust backend draw indicatif progress
    bars to stderr for each of the three per-file phases plus the
    plugin pass. The CLI sets this; the library API leaves it off.
    indicatif auto-hides on non-TTY stderr.
    """
    from dead_cst import _native as native

    ctx = native.ProjectContext(
        str(project_root),
        src_roots=[str(p) for p in src_roots] if src_roots else None,
        show_progress=show_progress,
    )
    for plugin in plugins:
        if hasattr(plugin, "run"):
            ctx.add_plugin(plugin)
    return _bridge(ctx.materialize())


def _bridge(graph: "native.NativeGraph") -> SymbolGraph:
    """Convert a project-wide :class:`NativeGraph` into a fresh :class:`SymbolGraph`."""
    out = SymbolGraph()
    nodes = list(graph.nodes)
    for n in nodes:
        out.add(n)
    for src, dst, flags in graph.edges:
        out.add_edge(nodes[src], nodes[dst], flags)
    return out
