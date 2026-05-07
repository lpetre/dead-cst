"""Resolver built from explicit ``-p`` path specs.

Mirrors the CLI's ``-p`` flag as a first-class :class:`PathResolver`.
Specs are intentionally minimal -- they cover the simple
"analyze this directory, with these other packages on the import
path" case. For src/flat-layout splits and per-file exported vs
internal classification, use ``[tool.dead-cst].packages`` in
``pyproject.toml`` (or write a custom :class:`PathResolver`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._core import Package
from ._imports import default_resolve_import


@dataclass
class ManualResolver:
    """Resolver built from explicit ``path[:dep1,dep2]`` specs.

    Each spec produces one :class:`Package` whose ``name`` is the
    path's final component, with the entire directory marked exported
    (``exported = (path,)``) so files under it participate in
    cross-package import resolution. Deps after ``:`` are
    comma-separated *package names* -- the path's final component of
    another spec in the same resolver call.
    """

    specs: list[str] = field(default_factory=list)
    name: str = "manual"
    version: int = 1778025600

    def resolve(self, project_root: Path) -> list[Package]:
        out: list[Package] = []
        for spec in self.specs:
            if ":" in spec:
                path_str, deps_str = spec.split(":", 1)
                deps = tuple(d.strip() for d in deps_str.split(",") if d.strip())
            else:
                path_str, deps = spec, ()
            path = (project_root / path_str).resolve()
            out.append(
                Package(
                    path=path,
                    name=path.name or "root",
                    exported=(path,),
                    deps=deps,
                )
            )
        return out

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
