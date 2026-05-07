"""Pluggable resolvers that discover sys.path-like search paths for a project.

A :class:`PathResolver` returns a tuple of :class:`Package` objects (one
per first-party workspace member) plus an import resolver. Multiple
resolvers compose by merging their package lists -- see
:func:`merge_packages`.

Each builtin resolver lives in its own submodule. Third-party resolvers
can register under the ``dead_cst.resolvers`` entry-point group;
:func:`load_resolver` checks builtins first, then falls back to entry points.

In addition, :func:`exported_roots` -- not a resolver itself -- inspects a
single base's ``pyproject.toml`` to determine which subdirs the build
backend would actually ship. The shipped resolvers call it at
:meth:`PathResolver.resolve` time to populate
:attr:`Package.exported`, so internal dirs like ``tests/`` stay scoped
to their owning member when other packages import from this one.

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

from ..contrib.uv import UvResolver
from ._core import ImportResolver, Package, PathResolver, load_toml, merge_packages
from ._exports import exported_roots
from ._imports import (
    SITE_PACKAGES_MARKERS,
    STDLIB,
    clear_path_caches,
    default_resolve_import,
    distribution_lookup,
    editable_distribution_roots,
    safe_resolve_module,
)
from .manual import ManualResolver

BUILTIN_RESOLVERS: dict[str, type[PathResolver]] = {
    UvResolver.name: UvResolver,
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
    "Package",
    "PathResolver",
    "SITE_PACKAGES_MARKERS",
    "STDLIB",
    "UvResolver",
    "clear_path_caches",
    "default_resolve_import",
    "distribution_lookup",
    "editable_distribution_roots",
    "exported_roots",
    "load_resolver",
    "load_toml",
    "merge_packages",
    "safe_resolve_module",
]
