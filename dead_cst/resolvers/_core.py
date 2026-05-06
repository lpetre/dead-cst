"""Shared types and helpers for path resolvers.

A resolver describes the project layout as a flat list of
:class:`SourceTree` entries. Each tree is a directory + flags +
``search_trees`` pointing at other trees this one's files can import
from. Files are assigned to the longest-prefix-matching tree; files
under no tree are dropped.

A *package* is a logical grouping: every tree in a package shares a
:attr:`SourceTree.package` name, and at most one of those trees may
carry :data:`SourceTreeFlags.EXPORTED`. The exported tree is what
other packages see when they reference this package via
``search_trees``; non-exported trees (tests, scripts) participate in
the analysis but are invisible to consumers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from .._cacheable import Cacheable

# ``name -> path`` lookup callable. ``None`` means "not resolvable here".
ImportResolver = Callable[[str, list[Path]], "str | Path | None"]


class SourceTreeFlags(enum.IntFlag):
    """Per-:class:`SourceTree` flags.

    ``EXPORTED`` marks the one tree per package that other packages may
    import from; only that tree's decls populate the package's export
    trie. Non-exported trees (tests, scripts, internal helpers) are
    analyzed but excluded from cross-package import lookups.
    """

    NONE = 0
    EXPORTED = enum.auto()


@dataclass(frozen=True, slots=True)
class SourceTree:
    """A directory of first-party source files inside a package.

    ``path`` is the directory the tree owns; files under ``path`` are
    routed to this tree via longest-prefix match (a more specific tree
    nested inside ``path`` wins). ``package`` groups trees that share
    invariants (one ``EXPORTED`` per package). ``search_trees`` lists
    the paths of other trees whose decls files in this tree can
    import from -- every referenced path must resolve to an
    :data:`SourceTreeFlags.EXPORTED` tree elsewhere in the resolver's
    output (validated by :func:`validate_source_trees`).

    Construction is two-phase by convention: build every tree with
    its ``path`` / ``package`` / ``flags``, then refer to other trees
    by their ``path`` in ``search_trees``. The analyzer resolves
    those path references against the full list at validation time.
    """

    path: Path
    package: str
    flags: SourceTreeFlags = SourceTreeFlags.NONE
    search_trees: tuple[Path, ...] = ()


@runtime_checkable
class PathResolver(Cacheable, Protocol):
    """Discover :class:`SourceTree` layout for a project root.

    :meth:`resolve` returns the list of trees the analyzer should
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
    so the per-file cache invalidates when a resolver's layout-discovery
    or import-resolution logic changes (bump the epoch ``version``).
    """

    def resolve(self, project_root: Path) -> list[SourceTree]: ...

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
class _ValidatedTrees:
    """Internal: the validated tree list plus precomputed indices.

    :func:`validate_source_trees` builds this once so the analyzer can
    query ``by_path`` / ``by_package`` / ``exported_for`` without
    rescanning the list.
    """

    trees: tuple[SourceTree, ...]
    by_path: dict[Path, SourceTree] = field(default_factory=dict)
    by_package: dict[str, list[SourceTree]] = field(default_factory=dict)
    exported_for: dict[str, SourceTree] = field(default_factory=dict)


def validate_source_trees(trees: Iterable[SourceTree]) -> _ValidatedTrees:
    """Check the invariants on a resolver's tree list and index it.

    Enforces: unique tree paths; non-empty package names; at most one
    ``EXPORTED`` tree per package; every ``search_trees`` entry refers
    to an existing tree; the referenced tree is ``EXPORTED``; no
    self-reference; the ``search_trees`` relation is acyclic.

    Raises :class:`ValueError` on the first violation. The analyzer
    calls this once per :class:`Analysis` construction.
    """
    tree_list = tuple(trees)
    by_path: dict[Path, SourceTree] = {}
    for t in tree_list:
        if t.path in by_path:
            raise ValueError(f"duplicate SourceTree.path: {t.path}")
        by_path[t.path] = t

    by_package: dict[str, list[SourceTree]] = {}
    exported_for: dict[str, SourceTree] = {}
    for t in tree_list:
        if not t.package:
            raise ValueError(f"SourceTree.package is empty for {t.path}")
        by_package.setdefault(t.package, []).append(t)
        if t.flags & SourceTreeFlags.EXPORTED:
            if t.package in exported_for:
                raise ValueError(
                    f"package {t.package!r} has multiple EXPORTED trees: "
                    f"{exported_for[t.package].path} and {t.path}"
                )
            exported_for[t.package] = t

    for t in tree_list:
        for ref in t.search_trees:
            if ref == t.path:
                raise ValueError(f"SourceTree {t.path} references itself in search_trees")
            target = by_path.get(ref)
            if target is None:
                raise ValueError(f"SourceTree {t.path} search_trees references unknown path {ref}")
            if not (target.flags & SourceTreeFlags.EXPORTED):
                raise ValueError(
                    f"SourceTree {t.path} search_trees references {ref} which is not EXPORTED"
                )

    # Cycle detection over the search_trees relation. Iterative DFS
    # keyed on path; detects back-edges into the active stack.
    color: dict[Path, int] = {}  # 0=white, 1=gray, 2=black
    for start in by_path:
        if color.get(start, 0) != 0:
            continue
        stack: list[tuple[Path, int]] = [(start, 0)]
        color[start] = 1
        while stack:
            cur, idx = stack[-1]
            refs = by_path[cur].search_trees
            if idx >= len(refs):
                color[cur] = 2
                stack.pop()
                continue
            stack[-1] = (cur, idx + 1)
            nxt = refs[idx]
            c = color.get(nxt, 0)
            if c == 1:
                raise ValueError(f"SourceTree search_trees has a cycle involving {nxt}")
            if c == 0:
                color[nxt] = 1
                stack.append((nxt, 0))

    return _ValidatedTrees(
        trees=tree_list,
        by_path=by_path,
        by_package=by_package,
        exported_for=exported_for,
    )


def assign_file_to_tree(file: Path, trees: Sequence[SourceTree]) -> SourceTree | None:
    """Longest-prefix match of ``file`` against ``trees``.

    Returns the tree whose ``path`` is the longest ancestor of
    ``file`` (including ``file == path``); ``None`` when no tree
    contains the file. Ties (same prefix length) shouldn't happen for
    a validated tree list -- paths are unique, so two trees can't
    share a path -- but we tiebreak deterministically by path string
    just in case.
    """
    file = file.resolve()
    best: SourceTree | None = None
    best_len = -1
    for t in trees:
        try:
            if file == t.path or file.is_relative_to(t.path):
                n = len(t.path.parts)
                if n > best_len or (
                    n == best_len and best is not None and str(t.path) < str(best.path)
                ):
                    best = t
                    best_len = n
        except ValueError:
            continue
    return best
