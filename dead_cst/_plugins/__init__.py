"""Pluggable edge contributors.

An :class:`EdgePlugin` adds (and, optionally, removes) nodes and edges in
the symbol graph after regular analysis is complete. Plugins exist to
encode knowledge the static analyzer cannot infer from CST alone --
dynamic dispatch, framework conventions, entry-point metadata, etc.

The analyzer runs every plugin once per base in topological order, after
that base's import edges have been resolved. Plugins receive a per-base
:class:`PluginContext`; see its docstring for the available helpers
(:meth:`~PluginContext.parse`, :meth:`~PluginContext.importers`,
:meth:`~PluginContext.base_modules`, ...).

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
    EdgePlugin,
    GraphOp,
    PluginContext,
    RemoveEdge,
    UnresolvedDependencyError,
    apply_ops,
    require_resolved_dep,
    synthetic_node,
    walk_to_instance_kind,
)
from .click import ClickPlugin
from .explicit import ExplicitEntrypointPlugin
from .fastapi import FastAPIPlugin
from .flask import FlaskPlugin
from .init_subclass import InitSubclassPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin
from .project_scripts import ProjectScriptsPlugin
from .pytest import PytestPlugin
from .typer import TyperPlugin
from .unittest import UnittestPlugin

BUILTIN_PLUGINS: dict[str, type] = {
    MainBlockPlugin.name: MainBlockPlugin,
    ProjectScriptsPlugin.name: ProjectScriptsPlugin,
    ExplicitEntrypointPlugin.name: ExplicitEntrypointPlugin,
    ModuleDundersPlugin.name: ModuleDundersPlugin,
    PytestPlugin.name: PytestPlugin,
    UnittestPlugin.name: UnittestPlugin,
    FastAPIPlugin.name: FastAPIPlugin,
    FlaskPlugin.name: FlaskPlugin,
    TyperPlugin.name: TyperPlugin,
    ClickPlugin.name: ClickPlugin,
    InitSubclassPlugin.name: InitSubclassPlugin,
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
    "ClickPlugin",
    "EdgePlugin",
    "ExplicitEntrypointPlugin",
    "FastAPIPlugin",
    "FlaskPlugin",
    "GraphOp",
    "InitSubclassPlugin",
    "MainBlockPlugin",
    "ModuleDundersPlugin",
    "PluginContext",
    "ProjectScriptsPlugin",
    "PytestPlugin",
    "RemoveEdge",
    "SYNTHETIC_POSITION",
    "TyperPlugin",
    "UnittestPlugin",
    "UnresolvedDependencyError",
    "apply_ops",
    "load_plugin",
    "require_resolved_dep",
    "synthetic_node",
    "walk_to_instance_kind",
]
