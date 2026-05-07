"""Shared types and helpers for path resolvers.

Defines the :class:`PathResolver` protocol every resolver satisfies, the
:class:`Package` value object they emit, and :func:`merge_packages` for
combining multiple resolvers' outputs into one validated list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from .._cacheable import Cacheable


@dataclass(frozen=True, slots=True)
class Package:
    """One first-party package the analyzer should walk.

    A resolver returns a list of these to describe a project layout.
    Each package contributes one root to the visitor pass and one node
    in the cross-package dep DAG used to scope reachability queries.

    * ``path`` -- the package directory; the visitor walks every
      ``.py`` file under here. Resolved (absolute) by
      :func:`merge_packages`.
    * ``name`` -- a stable identifier for this package, unique within
      one :class:`~dead_cst.analyze.Analysis`. Other packages refer to
      it by name in :attr:`deps`.
    * ``exported`` -- subdirs of ``path`` that ship to consumers; empty
      means "no restriction" (the whole package is exported). The
      analyzer uses this to scope cross-package import lookups, so
      internal dirs like ``tests/`` stay invisible to dependents.
    * ``deps`` -- names of other packages this one imports from.
      External (non-first-party) search paths are *not* expressed
      here; resolvers handle those internally via
      :meth:`PathResolver.resolve_import`.

    Frozen + slotted so packages are hashable / cheap to pass around;
    every ``Path`` field is treated as immutable post-construction.
    """

    path: Path
    name: str
    exported: tuple[Path, ...] = ()
    deps: tuple[str, ...] = ()


# ``name -> path`` lookup callable. ``None`` means "not resolvable here".
ImportResolver = Callable[[str, list[Path]], "str | Path | None"]


@runtime_checkable
class PathResolver(Cacheable, Protocol):
    """A resolver describes a project layout in two complementary ways.

    :meth:`resolve` reports the first-party packages the analyzer
    should walk (one :class:`Package` per workspace member);
    :meth:`resolve_import` answers ``name -> path`` lookups inside that
    layout. Splitting them lets a resolver own both halves -- e.g. a
    vendored-deps resolver can advertise a checked-in ``third_party/``
    package and also redirect imports to it without monkey-patching the
    analyzer.

    Non-first-party search paths (workspace ``.venv/site-packages``,
    vendored bundles, ...) do not appear in the returned
    :class:`Package` list. A resolver that needs such a path during
    classification handles it internally inside
    :meth:`resolve_import` -- e.g. :class:`~dead_cst.contrib.uv.UvResolver`
    splices its workspace ``.venv`` onto ``sys.path`` lazily so
    third-party imports still land in
    :func:`default_resolve_import`'s ``[external dist]`` arm.

    The shipped resolvers all delegate :meth:`resolve_import` to
    :func:`~dead_cst.resolvers._imports.default_resolve_import`, the
    ``sys.path`` + ``importlib`` implementation. Custom resolvers
    typically call it as a fallback after their own layout-specific
    lookups.

    Inherits the ``(name, version)`` contract from :class:`Cacheable`
    so the per-file cache invalidates when a resolver's layout-discovery
    or import-resolution logic changes (bump the epoch ``version``).
    """

    def resolve(self, project_root: Path) -> tuple[Package, ...]: ...

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None: ...


def load_toml(path: Path) -> dict[str, Any] | None:
    """Read ``path`` as TOML; ``None`` if the file is missing or tomllib is unavailable.

    Bad TOML is a programmer/config error and propagates as
    :class:`tomllib.TOMLDecodeError`.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11 not supported
        return None
    try:
        f = path.open("rb")
    except OSError:
        return None
    with f:
        return tomllib.load(f)


def merge_packages(*lists: Sequence[Package]) -> tuple[Package, ...]:
    """Merge multiple :class:`Package` lists into one validated tuple.

    Packages are matched by resolved ``path``; collisions union the
    ``deps`` and ``exported`` tuples (preserving first-seen order) and
    keep the first resolver's ``name``. ``path`` and every entry of
    ``exported`` are resolved before merging so equivalent paths
    written differently still collapse to one entry.

    Raises :class:`ValueError` when:

    * Two distinct paths share a ``name`` (names must be unique within
      one analysis -- :attr:`Package.deps` references go by name).
    * A ``deps`` entry doesn't match any package's ``name`` in the
      merged list.
    * An ``exported`` path isn't equal to or nested under its
      package's ``path``.
    """
    by_path: dict[Path, Package] = {}
    name_to_path: dict[str, Path] = {}
    for lst in lists:
        for pkg in lst:
            path = _absolute(pkg.path)
            exported = tuple(_absolute(e) for e in pkg.exported)
            existing = by_path.get(path)
            if existing is None:
                claimed = name_to_path.get(pkg.name)
                if claimed is not None and claimed != path:
                    raise ValueError(
                        f"Duplicate package name {pkg.name!r}: both {claimed} and {path} claim it"
                    )
                name_to_path[pkg.name] = path
                by_path[path] = Package(path=path, name=pkg.name, exported=exported, deps=pkg.deps)
            else:
                by_path[path] = Package(
                    path=path,
                    name=existing.name,
                    exported=tuple(dict.fromkeys((*existing.exported, *exported))),
                    deps=tuple(dict.fromkeys((*existing.deps, *pkg.deps))),
                )

    for pkg in by_path.values():
        for d in pkg.deps:
            if d not in name_to_path:
                raise ValueError(f"Package {pkg.name!r} references unknown dep {d!r}")
        for e in pkg.exported:
            if e != pkg.path and not e.is_relative_to(pkg.path):
                raise ValueError(f"Package {pkg.name!r}: exported path {e} is not under {pkg.path}")
    return tuple(by_path.values())


def _absolute(path: Path) -> Path:
    """Resolve ``path`` only when not already absolute.

    Resolvers shipped with the analyzer call ``Path.resolve()`` before
    constructing :class:`Package`, so this short-circuits the duplicate
    ``lstat`` for the common case while keeping the safety net for
    third-party resolvers that hand in relative paths.
    """
    return path if path.is_absolute() else path.resolve()
