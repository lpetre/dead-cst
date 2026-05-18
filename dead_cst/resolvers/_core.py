"""Shared types for path resolvers.

Defines the :class:`PathResolver` protocol every resolver satisfies and
the :class:`Package` value object they emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Package:
    """One first-party package the analyzer should walk.

    A resolver returns a list of these to describe a project layout.

    * ``path`` -- the package directory. Resolved (absolute) by
      :class:`~dead_cst.analyze.Analysis` at construction time.
    * ``name`` -- a stable identifier for this package, unique within
      one :class:`~dead_cst.analyze.Analysis`. Other packages refer to
      it by name in :attr:`deps`.
    * ``deps`` -- names of other packages this one imports from.
      Cycles are tolerated.
    """

    path: Path
    name: str
    deps: tuple[str, ...] = ()


@runtime_checkable
class PathResolver(Protocol):
    """A resolver returns the first-party packages in a project layout."""

    def resolve(self, project_root: Path) -> tuple[Package, ...]: ...


def load_toml(path: Path) -> dict[str, Any] | None:
    """Read ``path`` as TOML; ``None`` if the file is missing."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return None
    try:
        f = path.open("rb")
    except OSError:
        return None
    with f:
        return tomllib.load(f)


def _validate_packages(packages: Iterable[Package]) -> tuple[Package, ...]:
    """Resolve paths and validate one resolver's :class:`Package` output."""
    by_path: dict[Path, Package] = {}
    name_to_path: dict[str, Path] = {}
    for pkg in packages:
        path = _absolute(pkg.path)
        existing = by_path.get(path)
        if existing is None:
            claimed = name_to_path.get(pkg.name)
            if claimed is not None and claimed != path:
                raise ValueError(
                    f"Duplicate package name {pkg.name!r}: both {claimed} and {path} claim it"
                )
            name_to_path[pkg.name] = path
            by_path[path] = Package(path=path, name=pkg.name, deps=pkg.deps)
        else:
            by_path[path] = Package(
                path=path,
                name=existing.name,
                deps=tuple(dict.fromkeys((*existing.deps, *pkg.deps))),
            )

    for pkg in by_path.values():
        for d in pkg.deps:
            if d not in name_to_path:
                raise ValueError(f"Package {pkg.name!r} references unknown dep {d!r}")
    return tuple(by_path.values())


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else path.resolve()
