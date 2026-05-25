"""Pluggable edge contributors.

A plugin subclasses :class:`Plugin` and implements ``run(ctx)``
against a :class:`dead_cst._native.ProjectContext`, yielding
:class:`AddNode` / :class:`AddEdge` / :class:`AddEntrypoint` ops the
rust backend applies to the in-progress project graph.

This package contains plugins covering core Python conventions
(``__main__`` blocks, ``[project.scripts]``, explicit entrypoints,
module dunders, init-subclass discovery, the ``DYNAMIC_IMPORT``
fan-out) plus the reusable :class:`DecoratedDeclPlugin` /
:class:`DispatchAppPlugin` / :class:`LiteralListPlugin` shapes that
framework-specific plugins build on.

Plugins targeting specific third-party frameworks (FastAPI, Flask,
Click, Typer, pytest, unittest, ...) live under
:mod:`dead_cst.contrib`. The CLI's ``--plugin`` flag and its
builtin name → instance map live in :mod:`dead_cst.cli`.
"""

from __future__ import annotations

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
from .decl_shapes import (
    BatchDispatchAppPlugin,
    DecoratedDeclPlugin,
    DispatchAppGather,
    DispatchAppPlugin,
    DispatchAppSpec,
    LiteralListPlugin,
)
from .dynamic_import import DynamicImportFallbackPlugin
from .explicit_entrypoint import ExplicitEntrypointPlugin
from .init_subclass import InitSubclassPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin
from .project_scripts import ProjectScriptsPlugin

__all__ = [
    "BatchDispatchAppPlugin",
    "DecoratedDeclPlugin",
    "DispatchAppGather",
    "DispatchAppPlugin",
    "DispatchAppSpec",
    "DynamicImportFallbackPlugin",
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "EXTERNAL_PREFIXES",
    "ExplicitEntrypointPlugin",
    "InitSubclassPlugin",
    "LiteralListPlugin",
    "MainBlockPlugin",
    "ModuleDundersPlugin",
    "Plugin",
    "ProjectScriptsPlugin",
    "STDLIB_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "UNPARSEABLE_PREFIX",
    "UNRESOLVED_PREFIX",
    "simple_name",
]
