"""Pluggable edge contributors.

An :class:`EdgePlugin` adds (and, optionally, removes) nodes and edges in
the symbol graph after regular analysis is complete. Plugins exist to
encode knowledge the static analyzer cannot infer from CST alone --
dynamic dispatch, framework conventions, entry-point metadata, etc.

Two flavors exist:

* :class:`EdgePlugin` -- receives a :class:`PluginContext` only. Cheap.
* :class:`CSTAwareEdgePlugin` -- additionally receives the per-base
  :class:`FullRepoManager` map, letting the plugin re-walk source with
  libcst metadata providers.

``build_symbol_graph`` iterates the plugin list and dispatches to whichever
protocol each plugin satisfies (``isinstance``).

Plugins emit :class:`GraphOp` values rather than mutating the graph
directly; this keeps them easy to test and lets the analyzer report back
on what was added.

Each builtin plugin lives in its own submodule. Third-party plugins can
register under the ``dead_cst.plugins`` entry-point group; :func:`load_plugin`
checks builtins first, then falls back to entry points.
"""

from __future__ import annotations

from ._core import (
    SYNTHETIC_POSITION,
    AddEdge,
    AddNode,
    CSTAwareEdgePlugin,
    EdgePlugin,
    GraphOp,
    PluginContext,
    RemoveEdge,
    apply_ops,
    synthetic_node,
)
from .explicit import ExplicitEntrypointPlugin
from .fastapi import FastAPIPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin
from .project_scripts import ProjectScriptsPlugin
from .pytest import PytestPlugin

BUILTIN_PLUGINS: dict[str, type] = {
    MainBlockPlugin.name: MainBlockPlugin,
    ProjectScriptsPlugin.name: ProjectScriptsPlugin,
    ExplicitEntrypointPlugin.name: ExplicitEntrypointPlugin,
    ModuleDundersPlugin.name: ModuleDundersPlugin,
    PytestPlugin.name: PytestPlugin,
    FastAPIPlugin.name: FastAPIPlugin,
}


def load_plugin(name: str):
    """Load a plugin by name. Checks builtins first, then entry points."""
    if name in BUILTIN_PLUGINS:
        return BUILTIN_PLUGINS[name]()

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.plugins"):
        if ep.name == name:
            cls = ep.load()
            return cls()
    raise KeyError(f"Unknown edge plugin: {name!r}")


__all__ = [
    "AddEdge",
    "AddNode",
    "BUILTIN_PLUGINS",
    "CSTAwareEdgePlugin",
    "EdgePlugin",
    "ExplicitEntrypointPlugin",
    "FastAPIPlugin",
    "GraphOp",
    "MainBlockPlugin",
    "ModuleDundersPlugin",
    "PluginContext",
    "ProjectScriptsPlugin",
    "PytestPlugin",
    "RemoveEdge",
    "SYNTHETIC_POSITION",
    "apply_ops",
    "load_plugin",
    "synthetic_node",
]
