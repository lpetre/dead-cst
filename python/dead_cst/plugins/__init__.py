"""Pluggable edge contributors.

A plugin implements ``run(ctx)`` against a
:class:`dead_cst._native.ProjectContext`, yielding
:class:`AddNode` / :class:`AddEdge` / :class:`AddEntrypoint` ops the
rust backend applies to the in-progress project graph.

Built-in plugins covering core Python conventions (``__main__`` blocks,
``[project.scripts]``, explicit entrypoints, module dunders, init-subclass
discovery) live as siblings of this ``__init__``. Plugins targeting
specific third-party frameworks (FastAPI, Flask, Click, Typer, pytest,
unittest, ...) live under :mod:`dead_cst.contrib`; they are re-exported
here for ergonomics.

Third-party plugins can register under the ``dead_cst.plugins``
entry-point group; :func:`load_plugin` checks builtins first.
"""

from __future__ import annotations

from ._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    EXTERNAL_PREFIXES,
    STDLIB_PREFIX,
    SYNTHETIC_PATH_PREFIXES,
    UNPARSEABLE_PREFIX,
    UNRESOLVED_PREFIX,
    simple_name,
)
from ..contrib.celery import CeleryPlugin
from ..contrib.click import ClickPlugin
from ..contrib.cyclopts import CycloptsPlugin
from ..contrib.discordpy import DiscordPyPlugin
from ..contrib.fastapi import FastAPIPlugin
from ..contrib.fastmcp import FastMCPPlugin
from ..contrib.flask import FlaskPlugin
from ..contrib.mock_patch import MockPatchPlugin
from ..contrib.pytest import PytestPlugin
from ..contrib.server_config import ServerConfigPlugin
from ..contrib.typer import TyperPlugin
from ..contrib.unittest import UnittestPlugin
from .decl_shapes import DecoratedDeclPlugin, DispatchAppPlugin, LiteralListPlugin
from .dynamic_import import DynamicImportFallbackPlugin
from .explicit_entrypoint import ExplicitEntrypointPlugin
from .init_subclass import InitSubclassPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin
from .project_scripts import ProjectScriptsPlugin

BUILTIN_PLUGINS: dict[str, type] = {
    MainBlockPlugin.name: MainBlockPlugin,
    ProjectScriptsPlugin.name: ProjectScriptsPlugin,
    ExplicitEntrypointPlugin.name: ExplicitEntrypointPlugin,
    ModuleDundersPlugin.name: ModuleDundersPlugin,
    PytestPlugin.name: PytestPlugin,
    UnittestPlugin.name: UnittestPlugin,
    MockPatchPlugin.name: MockPatchPlugin,
    ServerConfigPlugin.name: ServerConfigPlugin,
    FastAPIPlugin.name: FastAPIPlugin,
    FastMCPPlugin.name: FastMCPPlugin,
    FlaskPlugin.name: FlaskPlugin,
    TyperPlugin.name: TyperPlugin,
    ClickPlugin.name: ClickPlugin,
    CycloptsPlugin.name: CycloptsPlugin,
    CeleryPlugin.name: CeleryPlugin,
    DiscordPyPlugin.name: DiscordPyPlugin,
    InitSubclassPlugin.name: InitSubclassPlugin,
    DynamicImportFallbackPlugin.name: DynamicImportFallbackPlugin,
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
    "BUILTIN_PLUGINS",
    "CeleryPlugin",
    "ClickPlugin",
    "CycloptsPlugin",
    "DecoratedDeclPlugin",
    "DiscordPyPlugin",
    "DispatchAppPlugin",
    "DynamicImportFallbackPlugin",
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "EXTERNAL_PREFIXES",
    "ExplicitEntrypointPlugin",
    "FastAPIPlugin",
    "FastMCPPlugin",
    "FlaskPlugin",
    "InitSubclassPlugin",
    "LiteralListPlugin",
    "MainBlockPlugin",
    "MockPatchPlugin",
    "ModuleDundersPlugin",
    "ProjectScriptsPlugin",
    "PytestPlugin",
    "STDLIB_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "ServerConfigPlugin",
    "TyperPlugin",
    "UNPARSEABLE_PREFIX",
    "UNRESOLVED_PREFIX",
    "UnittestPlugin",
    "load_plugin",
    "simple_name",
]
