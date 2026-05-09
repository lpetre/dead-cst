"""Resolver built from explicit ``package:dep1,dep2`` path specs.

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
    """Resolver built from explicit ``package:dep1,dep2`` path specs.

    Each entry in ``specs`` is either ``"package"`` (no deps) or
    ``"package:dep1,dep2,..."``. Package paths and dep paths are joined
    to ``project_root`` at :meth:`resolve` time; whitespace around dep
    names is stripped and trailing empty entries are dropped, matching
    the CLI's ``-p`` parsing.

    Each spec produces one :class:`Package`. Dep paths mentioned only
    inline (not as their own top-level spec) are auto-promoted to a
    :class:`Package` with empty ``deps`` -- so ``"src:dep1"`` yields
    both a ``src`` package (``deps=("dep1",)``) and a ``dep1``
    package, even when ``"dep1"`` isn't explicitly listed.

    Each package's ``name`` is the rightmost path component of its
    ``path`` (e.g. ``/tmp/x/pkg_a -> "pkg_a"``);
    :class:`~dead_cst.analyze.Analysis` rejects duplicates at
    construction time. Workspaces with colliding directory names need
    to disambiguate by writing a custom resolver. ``exported`` is
    populated from :func:`exported_roots` (``pyproject.toml`` /
    src-layout discovery) when available, else left empty (no
    restriction).

    Not registered in :data:`BUILTIN_RESOLVERS` -- :func:`load_resolver`
    can't construct it without ``specs``. Instantiate directly when
    constructing :class:`~dead_cst.analyze.Analysis` programmatically.
    """

    specs: list[str] = field(default_factory=list)

    def resolve(self, project_root: Path) -> tuple[Package, ...]:
        # Collect (package_path, dep_paths) per spec; Package
        # construction is deferred until every dep is known so
        # auto-promoted deps also get a Package and resolve to a name.
        explicit_deps: dict[Path, list[Path]] = {}
        for spec in self.specs:
            if ":" in spec:
                package_str, deps_str = spec.split(":", 1)
                package_path = (project_root / package_str).resolve()
                deps = [
                    (project_root / d.strip()).resolve() for d in deps_str.split(",") if d.strip()
                ]
            else:
                package_path = (project_root / spec).resolve()
                deps = []
            explicit_deps[package_path] = deps

        all_paths = list(
            dict.fromkeys([*explicit_deps, *(d for deps in explicit_deps.values() for d in deps)])
        )
        names_by_path = {p: p.name for p in all_paths}

        return tuple(
            Package(
                path=path,
                name=names_by_path[path],
                exported=tuple(exported_roots(path) or ()),
                deps=tuple(names_by_path[d] for d in explicit_deps.get(path, [])),
            )
            for path in all_paths
        )

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
