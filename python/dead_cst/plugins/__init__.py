"""Pluggable edge contributors.

A plugin subclasses :class:`Plugin` and implements ``run(ctx)``
against a :class:`dead_cst._native.ProjectContext`, yielding
:class:`AddNode` / :class:`AddEdge` / :class:`AddEntrypoint` ops the
rust backend applies to the in-progress project graph.

This package contains the reusable :class:`DecoratedDeclPlugin` /
:class:`LiteralListPlugin` shapes that framework-specific plugins
build on, plus the parity-tested Python ``main_block`` /
``module_dunders`` / ``init_subclass`` plugins kept alongside their
native ports. Core conventions (``[project.scripts]``, explicit
entrypoints, the ``DYNAMIC_IMPORT`` fan-out, ...) are now native
plugins on :class:`dead_cst._native.NativePlugin`.

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
    DecoratedDeclPlugin,
    LiteralListPlugin,
)
from .init_subclass import InitSubclassPlugin
from .main_block import MainBlockPlugin
from .module_dunders import ModuleDundersPlugin

__all__ = [
    "DecoratedDeclPlugin",
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "EXTERNAL_PREFIXES",
    "InitSubclassPlugin",
    "LiteralListPlugin",
    "MainBlockPlugin",
    "ModuleDundersPlugin",
    "Plugin",
    "STDLIB_PREFIX",
    "SYNTHETIC_PATH_PREFIXES",
    "UNPARSEABLE_PREFIX",
    "UNRESOLVED_PREFIX",
    "simple_name",
]
