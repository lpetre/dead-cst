"""Pluggable edge contributors.

An :class:`EdgePlugin` adds (and, optionally, removes) nodes and edges in
the symbol graph after regular analysis is complete. Plugins exist to
encode knowledge the static analyzer cannot infer from CST alone --
dynamic dispatch, framework conventions, entry-point metadata, etc.

Two phases:

* :meth:`EdgePlugin.observe` runs once per file inside the visitor
  loop, with the parsed :class:`libcst.Module` and the visitor's
  just-built :class:`~dead_cst.graph.VisitorPayload`. Returns a new
  payload (or ``None``) whose ``nodes`` / ``edges`` extend the file's
  contribution.
* :meth:`EdgePlugin.finalize` runs once per base after the analyzer's
  edge stitching, with the assembled :class:`PluginContext`. Operates
  purely on the graph -- no CST access -- and emits :class:`GraphOp`
  values (:class:`AddNode` / :class:`AddEdge` / :class:`RemoveEdge`).

Builtin plugins covering core Python conventions (``__main__`` blocks,
``[project.scripts]``, explicit entrypoints, module dunders, init-subclass
discovery) live as siblings of this ``__init__``. Plugins that target
specific third-party frameworks (FastAPI, Flask, Click, Typer, pytest,
unittest) live under :mod:`dead_cst.contrib` next to the
``UvWorkspaceResolver``; they are re-exported here for ergonomics so
``from dead_cst.plugins import FastAPIPlugin`` keeps working.

Third-party plugins can register under the ``dead_cst.plugins``
entry-point group; :func:`load_plugin` checks builtins first, then
falls back to entry points.

The :data:`STDLIB_PREFIX` / :data:`EXTERNAL_DIST_PREFIX` /
:data:`EXTERNAL_FILE_PREFIX` / :data:`UNRESOLVED_PREFIX` string
constants name the synthetic-node namespaces the analyzer creates for
non-first-party imports. :data:`EXTERNAL_PREFIXES` and
:data:`SYNTHETIC_PATH_PREFIXES` are convenience tuples;
:data:`SYNTHETIC_POSITION` is the file-wide :class:`CodeRange` sentinel
plugins stamp on synthetic nodes that don't correspond to a specific
source location.
"""

from __future__ import annotations

from ._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    EXTERNAL_PREFIXES,
    STDLIB_PREFIX,
    SYNTHETIC_PATH_PREFIXES,
    SYNTHETIC_POSITION,
    UNRESOLVED_PREFIX,
    AddEdge,
    AddNode,
    EdgePlugin,
    GraphOp,
    ObserveContext,
    PluginContext,
    RemoveEdge,
    UnresolvedDependencyError,
    apply_ops,
    collect_module_imports,
    decls_by_simple_name,
    decorator_owner,
    entrypoint_payload,
    find_call_assignments,
    find_handlers,
    is_from_module,
    is_name,
    make_payload,
    mark_entrypoints,
    matched_attr_call,
    require_resolved_dep,
    simple_name,
    single_target_assignment,
    synthetic_node,
    walk_to_instance_kind,
)
from ..contrib.click import ClickPlugin
from ..contrib.cyclopts import CycloptsPlugin
from ..contrib.fastapi import FastAPIPlugin
from ..contrib.flask import FlaskPlugin
from ..contrib.mock_patch import MockPatchPlugin
from ..contrib.pytest import PytestPlugin
from ..contrib.typer import TyperPlugin
from ..contrib.unittest import UnittestPlugin
from .decl_shapes import DecoratedDeclPlugin, LiteralListPlugin
from .explicit_entrypoint import ExplicitEntrypointPlugin
from .init_subclass import InitSubclassPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin
from .project_scripts import ProjectScriptsPlugin
from .pyi_stub import PyiStubPlugin

BUILTIN_PLUGINS: dict[str, type] = {
    MainBlockPlugin.name: MainBlockPlugin,
    ProjectScriptsPlugin.name: ProjectScriptsPlugin,
    ExplicitEntrypointPlugin.name: ExplicitEntrypointPlugin,
    ModuleDundersPlugin.name: ModuleDundersPlugin,
    PytestPlugin.name: PytestPlugin,
    UnittestPlugin.name: UnittestPlugin,
    MockPatchPlugin.name: MockPatchPlugin,
    FastAPIPlugin.name: FastAPIPlugin,
    FlaskPlugin.name: FlaskPlugin,
    TyperPlugin.name: TyperPlugin,
    ClickPlugin.name: ClickPlugin,
    CycloptsPlugin.name: CycloptsPlugin,
    InitSubclassPlugin.name: InitSubclassPlugin,
    PyiStubPlugin.name: PyiStubPlugin,
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
    "CycloptsPlugin",
    "DecoratedDeclPlugin",
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "EXTERNAL_PREFIXES",
    "EdgePlugin",
    "ExplicitEntrypointPlugin",
    "FastAPIPlugin",
    "FlaskPlugin",
    "GraphOp",
    "InitSubclassPlugin",
    "LiteralListPlugin",
    "MainBlockPlugin",
    "MockPatchPlugin",
    "ModuleDundersPlugin",
    "ObserveContext",
    "PluginContext",
    "ProjectScriptsPlugin",
    "PyiStubPlugin",
    "PytestPlugin",
    "RemoveEdge",
    "STDLIB_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "SYNTHETIC_POSITION",
    "TyperPlugin",
    "UNRESOLVED_PREFIX",
    "UnittestPlugin",
    "UnresolvedDependencyError",
    "apply_ops",
    "collect_module_imports",
    "decls_by_simple_name",
    "decorator_owner",
    "entrypoint_payload",
    "find_call_assignments",
    "find_handlers",
    "is_from_module",
    "is_name",
    "load_plugin",
    "make_payload",
    "mark_entrypoints",
    "matched_attr_call",
    "require_resolved_dep",
    "simple_name",
    "single_target_assignment",
    "synthetic_node",
    "walk_to_instance_kind",
]
