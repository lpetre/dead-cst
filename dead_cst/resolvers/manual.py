"""Resolver built from explicit ``-p`` path specs.

Mirrors the CLI's ``-p`` flag as a first-class :class:`PathResolver`.
Specs are intentionally minimal -- they cover the simple
"analyze this directory, with these other directories on the import
path" case. For multi-tree-per-package layouts (``tests/``, ``scripts/``,
etc.), use ``[tool.dead-cst].trees`` in ``pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._core import SourceTree, SourceTreeFlags
from ._imports import default_resolve_import


@dataclass
class ManualResolver:
    """Resolver built from explicit ``path[:search1,search2]`` specs.

    Each spec produces one ``EXPORTED`` :class:`SourceTree` whose
    ``package`` name is the path's final component. Search refs (the
    optional comma-separated list after ``:``) name other trees by
    path; those trees must also be specified as their own ``-p`` spec
    (so the resolver knows about them and they validate as
    ``EXPORTED``).

    Not registered in :data:`BUILTIN_RESOLVERS` -- :func:`load_resolver`
    can't construct it without ``specs``. Instantiate directly when
    composing a resolver chain programmatically.
    """

    specs: list[str] = field(default_factory=list)
    name: str = "manual"
    version: int = 1777985838

    def resolve(self, project_root: Path) -> list[SourceTree]:
        out: list[SourceTree] = []
        for spec in self.specs:
            if ":" in spec:
                path_str, deps_str = spec.split(":", 1)
                deps = tuple(
                    (project_root / d.strip()).resolve() for d in deps_str.split(",") if d.strip()
                )
            else:
                path_str, deps = spec, ()
            path = (project_root / path_str).resolve()
            out.append(
                SourceTree(
                    path=path,
                    package=path.name or "root",
                    flags=SourceTreeFlags.EXPORTED,
                    search_trees=deps,
                )
            )
        return out

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
