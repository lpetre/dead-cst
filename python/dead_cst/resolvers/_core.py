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

    * ``path`` -- the package's *owned* directory. Every source file
      under this prefix is the package's responsibility (so it gets
      processed during the package's edge pass and bucketed under it
      for per-package dead-code reporting). Includes things the wheel
      doesn't ship -- e.g. for an src-layout uv member, ``path`` is
      the member dir and covers both ``src/`` and ``tests/``.
    * ``name`` -- a stable identifier for this package, unique within
      one :class:`~dead_cst.analyze.Analysis`. Other packages refer to
      it by name in :attr:`deps`.
    * ``deps`` -- names of other packages this one imports from.
      Cycles are tolerated.
    * ``exported_paths`` -- directories *consumers* put on their
      search path when they depend on this package. Defaults to
      ``(path,)`` if the resolver doesn't specify them. For an
      src-layout uv member this is ``(member_dir / "src",)`` -- the
      wheel's published contents, NOT the whole member dir. This is
      what keeps a member's ``tests/`` package from bleeding into a
      consumer's lookup namespace.

    All paths are resolved (absolute) by
    :class:`~dead_cst.analyze.Analysis` at construction time.
    """

    path: Path
    name: str
    deps: tuple[str, ...] = ()
    exported_paths: tuple[Path, ...] = ()


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
        # ``exported_paths`` defaults to ``(path,)`` so resolvers that
        # don't care about the owned-vs-exported split (manual specs,
        # single-package layouts) get today's behavior for free. We
        # absolutize each entry the same way ``path`` is normalized.
        exported = (
            tuple(_absolute(p) for p in pkg.exported_paths) if pkg.exported_paths else (path,)
        )
        existing = by_path.get(path)
        if existing is None:
            claimed = name_to_path.get(pkg.name)
            if claimed is not None and claimed != path:
                raise ValueError(
                    f"Duplicate package name {pkg.name!r}: both {claimed} and {path} claim it"
                )
            name_to_path[pkg.name] = path
            by_path[path] = Package(
                path=path, name=pkg.name, deps=pkg.deps, exported_paths=exported
            )
        else:
            # Merge deps + union exported_paths on duplicates -- a
            # resolver that emits the same package twice with extra
            # info gets both sets honored.
            merged_exports = tuple(dict.fromkeys((*existing.exported_paths, *exported)))
            by_path[path] = Package(
                path=path,
                name=existing.name,
                deps=tuple(dict.fromkeys((*existing.deps, *pkg.deps))),
                exported_paths=merged_exports,
            )

    for pkg in by_path.values():
        for d in pkg.deps:
            if d not in name_to_path:
                raise ValueError(f"Package {pkg.name!r} references unknown dep {d!r}")
    return tuple(by_path.values())


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else path.resolve()
