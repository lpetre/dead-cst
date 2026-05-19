"""Pluggable edge contributors.

A plugin subclasses :class:`Plugin` and implements ``run(ctx)``
against a :class:`dead_cst.native.ProjectContext`, yielding
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

from ..contrib.celery import CeleryPlugin
from ..contrib.click import ClickPlugin
from ..contrib.cyclopts import cyclopts_plugin
from ..contrib.discordpy import DiscordPyPlugin
from ..contrib.fastapi import fastapi_plugin
from ..contrib.fastmcp import fastmcp_plugin
from ..contrib.flask import flask_plugin
from ..contrib.mock_patch import MockPatchPlugin
from ..contrib.pytest import PytestPlugin
from ..contrib.server_config import ServerConfigPlugin
from ..contrib.typer import typer_plugin
from ..contrib.unittest import UnittestPlugin
from ._base import Plugin
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
from .decl_shapes import DecoratedDeclPlugin, DispatchAppPlugin, LiteralListPlugin
from .dynamic_import import DynamicImportFallbackPlugin
from .explicit_entrypoint import ExplicitEntrypointPlugin
from .init_subclass import InitSubclassPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin
from .project_scripts import ProjectScriptsPlugin

# ---------------------------------------------------------------------------
# Built-in plugin registry.
#
# Each entry is a fully-configured plugin instance, ready to drop into
# an ``Analysis(..., plugins=...)`` call. The ``_BUILTIN_BY_CLI_KEY``
# dict pairs each entry with the string the CLI's ``--plugin`` flag
# accepts (and that :func:`load_plugin` looks up). CLI keys are
# stable: they mirror the user-facing plugin names that have been
# documented in the README / changelogs.
# ---------------------------------------------------------------------------

_BUILTIN_BY_CLI_KEY: dict[str, Plugin] = {
    "main_block": MainBlockPlugin(),
    "project_scripts": ProjectScriptsPlugin(),
    "explicit": ExplicitEntrypointPlugin(),
    "module_dunders": ModuleDundersPlugin(),
    "pytest": PytestPlugin(),
    "unittest": UnittestPlugin(),
    "mock_patch": MockPatchPlugin(),
    "server_config": ServerConfigPlugin(),
    "fastapi": fastapi_plugin(),
    "fastmcp": fastmcp_plugin(),
    "flask": flask_plugin(),
    "typer": typer_plugin(),
    "click": ClickPlugin(),
    "cyclopts": cyclopts_plugin(),
    "celery": CeleryPlugin(),
    "discordpy": DiscordPyPlugin(),
    "init_subclass": InitSubclassPlugin(),
    "dynamic_import_fallback": DynamicImportFallbackPlugin(),
}

BUILTIN_PLUGINS: list[Plugin] = list(_BUILTIN_BY_CLI_KEY.values())


def load_plugin(name: str) -> Plugin:
    """Load a plugin by name. Checks builtins first, then entry points.

    Builtin CLI keys are the stable strings the ``--plugin`` flag has
    always accepted (``main_block``, ``project_scripts``, ``fastapi``,
    ``flask``, ...) and live in :data:`_BUILTIN_BY_CLI_KEY`.
    """
    builtin = _BUILTIN_BY_CLI_KEY.get(name)
    if builtin is not None:
        return builtin

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.plugins"):
        if ep.name == name:
            loaded = ep.load()
            # Entry points may register either an instance or a factory
            # callable / class. Tolerate both shapes.
            if isinstance(loaded, Plugin):
                return loaded
            if callable(loaded):
                instance = loaded()
                if isinstance(instance, Plugin):
                    return instance
            raise TypeError(
                f"Plugin entry point {name!r} did not resolve to a Plugin instance "
                f"(got {type(loaded).__name__})"
            )
    raise KeyError(f"Unknown edge plugin: {name!r}")


__all__ = [
    "BUILTIN_PLUGINS",
    "CeleryPlugin",
    "ClickPlugin",
    "DecoratedDeclPlugin",
    "DiscordPyPlugin",
    "DispatchAppPlugin",
    "DynamicImportFallbackPlugin",
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "EXTERNAL_PREFIXES",
    "ExplicitEntrypointPlugin",
    "InitSubclassPlugin",
    "LiteralListPlugin",
    "MainBlockPlugin",
    "MockPatchPlugin",
    "ModuleDundersPlugin",
    "Plugin",
    "ProjectScriptsPlugin",
    "PytestPlugin",
    "STDLIB_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "ServerConfigPlugin",
    "UNPARSEABLE_PREFIX",
    "UNRESOLVED_PREFIX",
    "UnittestPlugin",
    "cyclopts_plugin",
    "fastapi_plugin",
    "fastmcp_plugin",
    "flask_plugin",
    "load_plugin",
    "simple_name",
    "typer_plugin",
]
