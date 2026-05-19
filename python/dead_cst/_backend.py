"""Native (rust) backend bridge.

The rust crate (``dead_cst._native``) builds the project graph
end-to-end using ty's ``SemanticIndex``. This module instantiates a
:class:`native.ProjectContext`, wires plugins, calls :meth:`materialize`,
and returns the live context.

Bulk reachability queries (:meth:`Analysis.reachable`,
:meth:`Analysis.dead`, :meth:`Analysis.descendants`,
:meth:`Analysis.ancestors`) and node/edge enumeration
(:meth:`ProjectContext.nodes`, :meth:`ProjectContext.edges`) are served
directly from the context — there is no Python-side adjacency copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from dead_cst import _native as native


def materialize_project(
    project_root: Path,
    plugins: Sequence[object] = (),
    src_roots: Sequence[Path] = (),
    *,
    show_progress: bool = False,
) -> native.ProjectContext:
    """Materialize ``project_root`` end-to-end via the rust backend.

    Builds a :class:`native.ProjectContext` rooted at ``project_root``,
    registers each plugin's ``run(ctx)`` callback, and calls
    :meth:`materialize`. Every plugin must be an instance of
    :class:`dead_cst.plugins.Plugin`; anything else raises
    :class:`TypeError` so typos (``Pluign()``) surface immediately
    instead of being silently dropped.

    Returns the live :class:`native.ProjectContext` so the caller
    (typically :class:`Analysis`) can route bulk reachability queries
    through the rust BFS via :meth:`reachable` / :meth:`descendants` /
    :meth:`ancestors`, and enumerate nodes/edges via :meth:`nodes` /
    :meth:`edges`.

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
    ctx.materialize()
    return ctx
