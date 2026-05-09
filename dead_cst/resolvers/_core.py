"""Shared types and helpers for path resolvers.

Defines the :class:`PathResolver` protocol every resolver satisfies and
the :class:`Package` value object they emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Package:
    """One first-party package the analyzer should walk.

    A resolver returns a list of these to describe a project layout.
    Each package contributes one root to the visitor pass and one node
    in the cross-package dep graph used to scope reachability queries.

    * ``path`` -- the package directory; the visitor walks every
      ``.py`` file under here. Resolved (absolute) by
      :class:`~dead_cst.analyze.Analysis` at construction time.
    * ``name`` -- a stable identifier for this package, unique within
      one :class:`~dead_cst.analyze.Analysis`. Other packages refer to
      it by name in :attr:`deps`.
    * ``exported`` -- subdirs of ``path`` that ship to consumers; empty
      means "no restriction" (the whole package is exported). The
      analyzer uses this to scope cross-package import lookups, so
      internal dirs like ``tests/`` stay invisible to dependents.
    * ``deps`` -- names of other packages this one imports from.
      Cycles are tolerated (e.g. when test or script code in two
      sibling packages cross-import). External (non-first-party)
      search paths are *not* expressed here; resolvers handle those
      internally via :meth:`PathResolver.resolve_import`.

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
class PathResolver(Protocol):
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

    Resolvers do *not* satisfy :class:`~dead_cst._cacheable.Cacheable`:
    their output flows through the (uncached) edge-stitching pass in
    :func:`~dead_cst._edges.resolve_edges`, so swapping a resolver
    re-stitches edges without invalidating any per-file
    :class:`~dead_cst.graph.VisitorPayload` blob. There is no
    ``(name, version)`` knob to bump.
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


def _validate_packages(packages: Iterable[Package]) -> tuple[Package, ...]:
    """Resolve paths and validate one resolver's :class:`Package` output.

    Returns a tuple of packages with ``path`` and every ``exported``
    entry made absolute. Equivalent paths written differently collapse
    to one entry; collisions union the ``deps`` and ``exported``
    tuples (preserving first-seen order) and keep the first occurrence's
    ``name``.

    Raises :class:`ValueError` when:

    * Two distinct paths share a ``name`` (names must be unique within
      one analysis -- :attr:`Package.deps` references go by name).
    * A ``deps`` entry doesn't match any package's ``name``.
    * An ``exported`` path isn't equal to or nested under its
      package's ``path``.
    """
    by_path: dict[Path, Package] = {}
    name_to_path: dict[str, Path] = {}
    for pkg in packages:
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
