"""Per-package contribution build: apply file payloads into raw sets/edges.

Sits between :mod:`dead_cst._refresh` (per-file work) and
:func:`dead_cst.analyze._compose_contribution` (cross-package
composition). Takes a :class:`PackageFiles` plus the visitor payloads
for its miss files and produces one :class:`PackageContribution` --
the per-package trie plus raw nodes / edges / dead-suite map and the
unresolved cross-file import set. The graph itself is built once at
compose time; this stage only accumulates data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from libcst.metadata import CodeRange

from ._notebooks import is_notebook
from ._refresh import PackageFiles
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode, SymbolTrie, VisitorPayload
from .resolvers import Package

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PackageContribution:
    """One package's pre-stitched contribution to the symbol graph.

    Built by :func:`build_contribution` and folded into the target
    graph by :func:`dead_cst.analyze._compose_contribution`. ``trie``
    holds every visible decl; consumer-side merges filter via
    :meth:`SymbolTrie.merge_exported`. ``nodes`` and ``edges`` are raw
    sets; ``dead_suites`` maps each file to its dead-suite positions.
    """

    package: Package
    trie: SymbolTrie
    nodes: frozenset[SymbolNode]
    edges: frozenset[tuple[SymbolNode, SymbolNode, EdgeFlags]]
    dead_suites: Mapping[Path, tuple[CodeRange, ...]]
    import_edges: frozenset[tuple[SymbolNode, Import, EdgeFlags]]


def build_contribution(
    package: Package,
    package_files: PackageFiles,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> PackageContribution:
    """Apply ``package_files``' per-file payloads into raw sets / dicts.

    Hits come straight from :class:`PackageFiles`; the rest are looked
    up in the global ``miss_payloads`` map produced by
    :func:`dead_cst._refresh.process_stale_files`. Empty
    :attr:`Package.exported` means "no restriction" (every file in the
    package is exported to consumers).

    A pre-pass identifies module-FQN collisions (``foo.py`` alongside
    ``foo/__init__.py``) via :func:`eclipsed_paths` so the loser is
    skipped at the trie -- its nodes still land in the contribution so
    observe-time entrypoints (``__main__``, plugin synthetics) keep
    working, but cross-module imports route to the package winner.
    """
    trie = SymbolTrie()
    nodes: set[SymbolNode] = set()
    edges: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = set()
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    dead_suites: dict[Path, tuple[CodeRange, ...]] = {}
    eclipsed = eclipsed_paths(package_files.files)
    for file in package_files.files:
        payload = package_files.hits.get(file)
        if payload is None:
            # ``miss_payloads`` only carries files the visitor managed
            # to ingest -- parse / IO failures in ``_process_one_file``
            # are dropped before reaching here, so a missing key means
            # "skip this file" rather than "lookup error".
            payload = miss_payloads.get(file)
        if payload is None:
            continue
        _apply_payload(
            payload,
            trie=trie,
            eclipsed=file in eclipsed,
            nodes=nodes,
            edges=edges,
            import_edges=import_edges,
            dead_suites=dead_suites,
        )
    for child, parent in trie.module_hierarchy_edges():
        edges.add((child, parent, EdgeFlags.NONE))
    return PackageContribution(
        package=package,
        trie=trie,
        nodes=frozenset(nodes),
        edges=frozenset(edges),
        dead_suites=dead_suites,
        import_edges=frozenset(import_edges),
    )


def eclipsed_paths(files: Sequence[Path]) -> frozenset[Path]:
    """Return every ``.py`` / ``.pyi`` in ``files`` eclipsed by a sibling package.

    Mirrors CPython's :class:`importlib.machinery.FileFinder` precedence:
    a regular package (``__init__.py``) eclipses a sibling module file
    that would claim the same dotted name. Given a directory containing
    both ``foo.py`` and ``foo/__init__.py``, ``foo.py`` is returned --
    Python's import machinery loads the package, so resolving
    ``pkg.foo`` to the ``.py`` would mis-route every consumer import.

    "Eclipsed" disambiguates this file-vs-package precedence from
    :data:`NodeFlags.SHADOWED` (intra-file decl rebinding) and the
    ``.pyi``-vs-``.py`` peer-stub filter inside :func:`enumerate_files`.

    Logs a warning per eclipsed file: this layout is almost always a
    bug, and surfacing it during analysis helps users notice before
    the dead-code report blames the wrong half.
    """
    init_dirs = {f.parent for f in files if f.name == "__init__.py"}
    eclipsed: set[Path] = set()
    for f in files:
        if f.name == "__init__.py":
            continue
        # Notebooks aren't importable modules, so they can't be eclipsed
        # by a sibling ``__init__.py`` -- treat them as orthogonal.
        if is_notebook(f):
            continue
        candidate = f.with_suffix("")
        if candidate in init_dirs:
            init_path = candidate / "__init__.py"
            logger.warning(
                "Module %s eclipsed by sibling package %s; "
                "Python loads the package, so %s will not be "
                "reachable as an importable module",
                f,
                init_path,
                f.name,
            )
            eclipsed.add(f)
    return frozenset(eclipsed)


def _apply_payload(
    payload: VisitorPayload,
    *,
    trie: SymbolTrie,
    eclipsed: bool,
    nodes: set[SymbolNode],
    edges: set[tuple[SymbolNode, SymbolNode, EdgeFlags]],
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
    dead_suites: dict[Path, tuple[CodeRange, ...]],
) -> None:
    """Emit ``payload`` into the in-progress per-package accumulators.

    Routing by ``SymbolNode.type``: ``module`` and other decls go into
    ``nodes`` and (subject to ``eclipsed`` + the per-decl ``SHADOWED`` /
    ``OVERLOAD`` / ``NOTEBOOK`` flags) the trie; non-module decls also
    get a parent-module edge. ``synthetic`` nodes land in ``nodes`` only
    -- no parent edge, no trie entry.

    Each edge gets :data:`EdgeFlags.DEAD_BRANCH` when its access
    position falls inside one of ``payload.dead_suites``. Unresolved
    cross-file imports accumulate into ``import_edges`` for
    :func:`resolve_edges` to stitch later.
    """
    module = next(n for n in payload.nodes if n.type == "module")

    payload_dead_suites = payload.dead_suites
    if payload_dead_suites:

        def flag_for(pos: CodeRange) -> EdgeFlags:
            return (
                EdgeFlags.DEAD_BRANCH
                if any(_contains(s, pos) for s in payload_dead_suites)
                else EdgeFlags.NONE
            )
    else:

        def flag_for(pos: CodeRange) -> EdgeFlags:
            return EdgeFlags.NONE

    for n in payload.nodes:
        nodes.add(n)
        if n.type == "synthetic":
            continue
        if n.type != "module":
            edges.add((n, module, EdgeFlags.NONE))
        if not eclipsed and not (
            n.flags & (NodeFlags.SHADOWED | NodeFlags.OVERLOAD | NodeFlags.NOTEBOOK)
        ):
            trie.add_declaration(n)

    for src, dst, pos in payload.edges:
        edges.add((src, dst, flag_for(pos)))

    for src, imp, pos in payload.imports:
        import_edges.add((src, imp, flag_for(pos)))

    if payload_dead_suites:
        dead_suites[module.path] = payload_dead_suites


def _contains(suite: CodeRange, access: CodeRange) -> bool:
    """``True`` iff ``access`` is fully nested inside ``suite``.

    Compares ``(line, column)`` lexicographically at both ends. Suites
    are line-aligned in practice (libcst positions an ``IndentedBlock``
    at its first statement) so the line check usually decides; the
    column tiebreak handles one-line ``if False: x = 1`` suites.
    """
    s_start = (suite.start.line, suite.start.column)
    s_end = (suite.end.line, suite.end.column)
    a_start = (access.start.line, access.start.column)
    a_end = (access.end.line, access.end.column)
    return s_start <= a_start and a_end <= s_end
