"""Pluggable resolvers that discover sys.path-like search paths for a project.

A :class:`PathResolver` takes a project root and returns a ``dict[base, [dep_paths]]``
in the same shape :func:`dead_cst.analyze.build_symbol_graph` already
consumes. Multiple resolvers compose by merging dicts -- see
:func:`merge_paths`.

Each builtin resolver lives in its own submodule. Third-party resolvers
can register under the ``dead_cst.resolvers`` entry-point group;
:func:`load_resolver` checks builtins first, then falls back to entry points.

In addition, :func:`exported_roots` -- not a resolver itself -- inspects a
single base's ``pyproject.toml`` to determine which subdirs the build
backend would actually ship. The analyzer calls this per-base to scope
each dep's contribution to consumers' import lookups, so internal dirs
like ``tests/`` stay scoped to their owning member.

Custom resolvers re-implementing :meth:`PathResolver.resolve_import`
can call :func:`default_resolve_import` (the shipped sys.path /
importlib implementation) directly, or compose with the lower-level
:func:`safe_resolve_module`, :func:`distribution_lookup`, and
:func:`editable_distribution_roots` helpers plus the :data:`STDLIB` /
:data:`SITE_PACKAGES_MARKERS` classification constants.
:func:`load_toml` is provided for resolvers that read
``pyproject.toml``-style config.
"""

from __future__ import annotations

from ._core import ImportResolver, PathMap, PathResolver, load_toml, merge_paths
from ._exports import exported_roots
from ._imports import (
    SITE_PACKAGES_MARKERS,
    STDLIB,
    default_resolve_import,
    distribution_lookup,
    editable_distribution_roots,
    safe_resolve_module,
)
from ..contrib.uv_workspace import UvWorkspaceResolver
from .manual import ManualResolver
from .pyproject import PyprojectResolver
from .venv import MissingVenvError, VenvResolver

BUILTIN_RESOLVERS: dict[str, type[PathResolver]] = {
    VenvResolver.name: VenvResolver,
    PyprojectResolver.name: PyprojectResolver,
    UvWorkspaceResolver.name: UvWorkspaceResolver,
}


def load_resolver(name: str) -> PathResolver:
    """Load a resolver by name. Checks builtins first, then entry points."""
    if name in BUILTIN_RESOLVERS:
        return BUILTIN_RESOLVERS[name]()

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.resolvers"):
        if ep.name == name:
            cls = ep.load()
            return cls()
    raise KeyError(f"Unknown path resolver: {name!r}")


__all__ = [
    "BUILTIN_RESOLVERS",
    "ImportResolver",
    "ManualResolver",
    "MissingVenvError",
    "PathMap",
    "PathResolver",
    "PyprojectResolver",
    "SITE_PACKAGES_MARKERS",
    "STDLIB",
    "UvWorkspaceResolver",
    "VenvResolver",
    "default_resolve_import",
    "distribution_lookup",
    "editable_distribution_roots",
    "exported_roots",
    "load_resolver",
    "load_toml",
    "merge_paths",
    "safe_resolve_module",
]
