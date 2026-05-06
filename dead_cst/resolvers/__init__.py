"""Pluggable resolvers that discover the :class:`SourceTree` layout of a project.

A :class:`PathResolver` takes a project root and returns a flat
``list[SourceTree]`` describing every directory of first-party source
the analyzer should walk. Each tree carries its own ``package`` name,
:class:`SourceTreeFlags` (notably :data:`SourceTreeFlags.EXPORTED`,
the one-per-package marker that gates cross-package import
visibility), and ``search_trees`` -- the paths of other trees this
one's files can import from. The analyzer routes each ``.py`` file to
its longest-prefix-matching tree.

Each builtin resolver lives in its own submodule. Third-party
resolvers register under the ``dead_cst.resolvers`` entry-point group;
:func:`load_resolver` checks builtins first, then falls back to
entry points.

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

from ..contrib.uv_workspace import UvWorkspaceResolver
from ._core import (
    ImportResolver,
    PathResolver,
    SourceTree,
    SourceTreeFlags,
    assign_file_to_tree,
    load_toml,
    validate_source_trees,
)
from ._imports import (
    SITE_PACKAGES_MARKERS,
    STDLIB,
    default_resolve_import,
    distribution_lookup,
    editable_distribution_roots,
    safe_resolve_module,
)
from .manual import ManualResolver
from .pyproject import PyprojectResolver

BUILTIN_RESOLVERS: dict[str, type[PathResolver]] = {
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
    "PathResolver",
    "PyprojectResolver",
    "SITE_PACKAGES_MARKERS",
    "STDLIB",
    "SourceTree",
    "SourceTreeFlags",
    "UvWorkspaceResolver",
    "assign_file_to_tree",
    "default_resolve_import",
    "distribution_lookup",
    "editable_distribution_roots",
    "load_resolver",
    "load_toml",
    "safe_resolve_module",
    "validate_source_trees",
]
