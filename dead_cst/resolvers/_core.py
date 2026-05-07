"""Shared types and helpers for path resolvers.

A resolver describes the project layout as a flat list of
:class:`Package` entries. Each package owns one directory tree
(``path``) and optionally lists subdirs whose ``.py`` files ship in
the wheel (``exported``). Files outside the exported subdirs --
tests, scripts, app entrypoints -- are *internal*: they participate
in the analysis but stay invisible to other packages' production
code.

The dependency relation between packages is split in two:

* ``deps`` is a production-only DAG between packages (by name). It
  drives the topological order in which the analyzer parses
  exported files and stitches their cross-package imports.
* Internal files are parsed in a second phase that reads against the
  union of every package's already-built export trie, so non-exported
  code can have apparent cycles between packages (e.g. ``A.tests``
  importing ``B.lib`` while ``B.tests`` imports ``A.lib``) without
  violating the production DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from .._cacheable import Cacheable

# ``name -> path`` lookup callable. ``None`` means "not resolvable here".
ImportResolver = Callable[[str, list[Path]], "str | Path | None"]


@dataclass(frozen=True, slots=True)
class Package:
    """One first-party package.

    ``path`` is the directory that owns every ``.py`` file under it
    (longest-prefix match). ``name`` is the package's unique name in
    this :class:`Analysis`. ``exported`` lists subdirs of ``path``
    whose files ship in the wheel; files under any of those are
    *exported*, everything else under ``path`` is *internal*. ``deps``
    names other packages whose exported code this one's production
    code depends on -- the production DAG.

    Two-phase parsing:

    * Exported files are parsed in topological order over ``deps``,
      with cross-package imports resolved against each dep's
      already-built export trie. The DAG requirement is enforced on
      ``deps``.
    * Internal files are parsed in a second pass against the union of
      every package's export trie plus this package's own files
      (exported + internal). Internal-file imports across packages
      can therefore form apparent cycles -- the trie reads are pure
      lookups, no ordering required.

    For src layout, ``exported = (path / "src",)`` and ``path`` itself
    is the package root that also walks ``tests/``, ``scripts/`` etc.
    For flat layout, ``exported = (path / "<modname>",)`` (or several
    for namespace packages).
    """

    path: Path
    name: str
    exported: tuple[Path, ...] = ()
    deps: tuple[str, ...] = ()


@runtime_checkable
class PathResolver(Cacheable, Protocol):
    """Discover :class:`Package` layout for a project root.

    :meth:`resolve` returns the list of packages the analyzer should
    walk; :meth:`resolve_import` answers ``name -> path`` lookups
    inside that layout. Splitting them lets a resolver own both
    halves -- e.g. a vendored-deps resolver can point at a checked-in
    ``third_party/`` and also redirect imports to it without
    monkey-patching the analyzer.

    The shipped resolvers all delegate :meth:`resolve_import` to
    :func:`~dead_cst.resolvers._imports.default_resolve_import`, the
    ``sys.path`` + ``importlib`` implementation. Custom resolvers
    typically call it as a fallback after their own layout-specific
    lookups.

    Inherits the ``(name, version)`` contract from :class:`Cacheable`
    so the per-file cache invalidates when a resolver's
    layout-discovery or import-resolution logic changes (bump the
    epoch ``version``).
    """

    def resolve(self, project_root: Path) -> list[Package]: ...

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


@dataclass(frozen=True, slots=True)
class _ValidatedPackages:
    """Internal: the validated package list plus precomputed indices.

    :func:`validate_packages` builds this once so the analyzer can
    query ``by_name`` / ``by_path`` / ``topo_order`` without rescanning
    the list.
    """

    packages: tuple[Package, ...]
    by_name: dict[str, Package]
    by_path: dict[Path, Package]
    # Topological order over ``deps``: deps before consumers.
    topo_order: tuple[Package, ...]


def validate_packages(packages: Iterable[Package]) -> _ValidatedPackages:
    """Check the invariants on a resolver's package list and index it.

    Enforces: non-empty unique names; unique paths; every ``exported``
    entry is under (or equal to) ``path``; every ``deps`` name refers
    to another package in the list; no self-dep; ``deps`` is acyclic.

    Raises :class:`ValueError` on the first violation. The analyzer
    calls this once per :class:`Analysis` construction.
    """
    pkg_list = tuple(packages)
    by_name: dict[str, Package] = {}
    by_path: dict[Path, Package] = {}
    for p in pkg_list:
        if not p.name:
            raise ValueError(f"Package.name is empty for {p.path}")
        if p.name in by_name:
            raise ValueError(f"duplicate Package.name: {p.name!r}")
        by_name[p.name] = p
        if p.path in by_path:
            raise ValueError(f"duplicate Package.path: {p.path}")
        by_path[p.path] = p
        for sub in p.exported:
            if sub != p.path and not sub.is_relative_to(p.path):
                raise ValueError(f"Package {p.name!r}: exported {sub} is not under path {p.path}")

    for p in pkg_list:
        for dep in p.deps:
            if dep == p.name:
                raise ValueError(f"Package {p.name!r} lists itself in deps")
            if dep not in by_name:
                raise ValueError(f"Package {p.name!r}: unknown dep {dep!r}")

    color: dict[str, int] = {}  # 0=white, 1=gray, 2=black
    order: list[Package] = []
    for start in by_name:
        if color.get(start, 0) != 0:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = 1
        while stack:
            cur, idx = stack[-1]
            deps = by_name[cur].deps
            if idx >= len(deps):
                color[cur] = 2
                order.append(by_name[cur])
                stack.pop()
                continue
            stack[-1] = (cur, idx + 1)
            nxt = deps[idx]
            c = color.get(nxt, 0)
            if c == 1:
                raise ValueError(f"Package deps has a cycle involving {nxt!r}")
            if c == 0:
                color[nxt] = 1
                stack.append((nxt, 0))

    return _ValidatedPackages(
        packages=pkg_list,
        by_name=by_name,
        by_path=by_path,
        topo_order=tuple(order),
    )


def assign_file_to_package(file: Path, packages: Sequence[Package]) -> Package | None:
    """Longest-prefix match of ``file`` against ``packages``.

    Returns the package whose ``path`` is the longest ancestor of
    ``file`` (including ``file == path``); ``None`` when no package
    contains the file. ``file`` should already be absolute and
    symlink-resolved; the analyzer always passes paths joined under
    a validated (already-resolved) package path, so we skip a per-call
    ``Path.resolve()`` stat in the hot per-file routing loop. Ties on
    prefix length tiebreak by path string to stay deterministic.
    """
    best: Package | None = None
    best_len = -1
    for p in packages:
        try:
            if file == p.path or file.is_relative_to(p.path):
                n = len(p.path.parts)
                if n > best_len or (
                    n == best_len and best is not None and str(p.path) < str(best.path)
                ):
                    best = p
                    best_len = n
        except ValueError:
            continue
    return best


def is_exported_file(file: Path, package: Package) -> bool:
    """``True`` if ``file`` lives under any of ``package.exported``.

    Files inside an exported subdir are part of the wheel-shipped
    surface; everything else under ``package.path`` is internal
    (tests, scripts, app entrypoints).
    """
    for sub in package.exported:
        try:
            if file == sub or file.is_relative_to(sub):
                return True
        except ValueError:
            continue
    return False


def export_search_root(package: Package) -> Path | None:
    """Sys.path entry that resolves cross-package imports of ``package``'s exports.

    For src layout (every ``exported`` entry under ``path / "src"``)
    the entry is ``path / "src"``. Otherwise it's ``path`` itself.
    Returns ``None`` when ``package.exported`` is empty (a virtual
    app/service that ships no wheel).
    """
    if not package.exported:
        return None
    src = package.path / "src"
    if all(sub == src or sub.is_relative_to(src) for sub in package.exported):
        return src
    return package.path
