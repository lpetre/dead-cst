"""Pluggable resolvers that discover the :class:`Package` layout of a project.

A :class:`PathResolver` takes a project root and returns a flat
``list[Package]`` describing every first-party package the analyzer
should walk. Each :class:`Package` owns its directory ``path``,
carries a unique ``name``, lists ``exported`` subdirs whose ``.py``
files ship in the wheel, and lists ``deps`` (other package names) --
the production-only DAG that drives topological parse order.

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
``pyproject.toml``-style config; :func:`exported_roots` and
:func:`exported_tree_root` answer "what does this project's build
backend ship?" so resolvers can pick exported subdir paths
consistently.
"""

from __future__ import annotations

from ..contrib.uv_resolver import UvResolver
from ._core import (
    ImportResolver,
    Package,
    PathResolver,
    assign_file_to_package,
    export_search_root,
    is_exported_file,
    load_toml,
    validate_packages,
)
from ._exports import exported_roots, exported_tree_root
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
    "PyprojectResolver",
    "SITE_PACKAGES_MARKERS",
    "STDLIB",
    "UvResolver",
    "assign_file_to_package",
    "default_resolve_import",
    "distribution_lookup",
    "editable_distribution_roots",
    "export_search_root",
    "exported_roots",
    "exported_tree_root",
    "is_exported_file",
    "load_resolver",
    "load_toml",
    "safe_resolve_module",
    "validate_packages",
]
