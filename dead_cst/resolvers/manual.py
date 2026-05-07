"""Resolver built from explicit ``base:dep1,dep2`` path specs.

Mirrors the CLI's ``-p`` flag as a first-class :class:`PathResolver`,
so explicit specs flow through the same pipeline as auto-discovered
layouts and participate in :meth:`~PathResolver.resolve_import` lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._core import Package
from ._exports import exported_roots
from ._imports import default_resolve_import


@dataclass
class ManualResolver:
    """Resolver built from explicit ``base:dep1,dep2`` path specs.

    Each entry in ``specs`` is either ``"base"`` (no deps) or
    ``"base:dep1,dep2,..."``. Bases and deps are joined to
    ``project_root`` at :meth:`resolve` time; whitespace around dep
    names is stripped and trailing empty entries are dropped, matching
    the CLI's ``-p`` parsing.

    Each spec produces one :class:`Package`. Dep paths mentioned only
    inline (not as their own top-level spec) are auto-promoted to a
    :class:`Package` with empty ``deps`` -- so ``"src:dep1"`` yields
    both a ``src`` package (``deps=("dep1",)``) and a ``dep1``
    package, even when ``"dep1"`` isn't explicitly listed.

    Each package's ``name`` is the rightmost path component of its
    ``path`` (e.g. ``/tmp/x/pkg_a -> "pkg_a"``); :func:`merge_packages`
    rejects duplicates. Workspaces with colliding directory names need
    to disambiguate by writing a custom resolver. ``exported`` is
    populated from :func:`exported_roots` (``pyproject.toml`` /
    src-layout discovery) when available, else left empty (no
    restriction).

    Not registered in :data:`BUILTIN_RESOLVERS` -- :func:`load_resolver`
    can't construct it without ``specs``. Instantiate directly when
    composing a resolver chain programmatically.
    """

    specs: list[str] = field(default_factory=list)
    name: str = "manual"
    version: int = 1777985838

    def resolve(self, project_root: Path) -> tuple[Package, ...]:
        # First pass: collect the explicit base path + dep paths from
        # each spec. We delay Package construction until every dep is
        # known so deps can resolve to a Package.name even when that
        # dep was not itself listed as its own spec (auto-promotion).
        explicit: list[tuple[Path, list[Path]]] = []
        for spec in self.specs:
            if ":" in spec:
                base_str, deps_str = spec.split(":", 1)
                base = (project_root / base_str).resolve()
                deps = [
                    (project_root / d.strip()).resolve() for d in deps_str.split(",") if d.strip()
                ]
            else:
                base = (project_root / spec).resolve()
                deps = []
            explicit.append((base, deps))

        # Auto-promote any dep path that isn't itself an explicit base
        # to its own Package (deps=(), exported from pyproject.toml).
        explicit_paths = {b for b, _ in explicit}
        all_paths: list[Path] = []
        seen: set[Path] = set()
        for base, _ in explicit:
            if base not in seen:
                seen.add(base)
                all_paths.append(base)
        for _, deps in explicit:
            for d in deps:
                if d not in seen:
                    seen.add(d)
                    all_paths.append(d)

        names_by_path = {p: p.name for p in all_paths}
        explicit_deps: dict[Path, list[Path]] = {b: deps for b, deps in explicit}

        out: list[Package] = []
        for path in all_paths:
            deps_paths = explicit_deps.get(path, []) if path in explicit_paths else []
            exported = tuple(exported_roots(path) or ())
            out.append(
                Package(
                    path=path,
                    name=names_by_path[path],
                    exported=exported,
                    deps=tuple(names_by_path[d] for d in deps_paths),
                )
            )
        return tuple(out)

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
