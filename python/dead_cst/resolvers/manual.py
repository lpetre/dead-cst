"""Resolver built from explicit ``package:dep1,dep2`` path specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._core import Package


@dataclass
class ManualResolver:
    """Resolver built from explicit ``package:dep1,dep2`` path specs.

    Each entry in ``specs`` is either ``"package"`` (no deps) or
    ``"package:dep1,dep2,..."``. Package paths and dep paths are joined
    to ``project_root`` at :meth:`resolve` time. Dep paths mentioned only
    inline (not as their own top-level spec) are auto-promoted to a
    :class:`Package` with empty ``deps``.

    Each package's ``name`` is the rightmost path component of its
    ``path``.
    """

    specs: list[str] = field(default_factory=list)

    def resolve(self, project_root: Path) -> tuple[Package, ...]:
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
                deps=tuple(names_by_path[d] for d in explicit_deps.get(path, [])),
            )
            for path in all_paths
        )
