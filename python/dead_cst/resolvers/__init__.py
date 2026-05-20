"""Pluggable resolvers that discover the first-party packages of a project.

A :class:`PathResolver` returns a tuple of :class:`Package` objects
(one per first-party workspace member). An
:class:`~dead_cst.analyze.Analysis` takes exactly one resolver.

The :class:`ManualResolver` builtin ships here; tool-specific
resolvers (like :class:`~dead_cst.contrib.uv.UvResolver` for
``uv.lock``) live under :mod:`dead_cst.contrib`. The CLI's
``--resolver`` flag and its builtin name → class map live in
:mod:`dead_cst.cli`.
"""

from __future__ import annotations

from ._core import Package, PathResolver, load_toml
from .manual import ManualResolver

__all__ = [
    "ManualResolver",
    "Package",
    "PathResolver",
    "load_toml",
]
