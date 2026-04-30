"""Pluggable resolvers that discover sys.path-like search paths for a project.

A :class:`PathResolver` takes a project root and returns a ``dict[base, [dep_paths]]``
in the same shape :func:`build_symbol_graph` already consumes. Multiple resolvers
compose by merging dicts -- see :func:`merge_paths`.

Each builtin resolver lives in its own submodule. Third-party resolvers
can register under the ``dead_cst.resolvers`` entry-point group;
:func:`load_resolver` checks builtins first, then falls back to entry points.

In addition, :func:`exported_roots` -- not a resolver itself -- inspects a
single base's ``pyproject.toml`` to determine which subdirs the build
backend would actually ship. The analyzer calls this per-base to scope
each dep's contribution to consumers' import lookups, so internal dirs
like ``tests/`` stay scoped to their owning member.
"""

from __future__ import annotations

from ._core import PathMap, PathResolver, merge_paths
from ._exports import exported_roots
from ._imports import (
    default_resolve_import,
    distribution_lookup,
    safe_resolve_module,
    temp_sys_path,
)
from .manual import ManualResolver
from .pyproject import PyprojectResolver
from .uv_workspace import UvWorkspaceResolver
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
    "ManualResolver",
    "MissingVenvError",
    "PathMap",
    "PathResolver",
    "PyprojectResolver",
    "UvWorkspaceResolver",
    "VenvResolver",
    "default_resolve_import",
    "distribution_lookup",
    "exported_roots",
    "load_resolver",
    "merge_paths",
    "safe_resolve_module",
    "temp_sys_path",
]
