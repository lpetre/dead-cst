"""Per-package contribution build: apply file payloads into a graph slice.

Sits between :mod:`dead_cst._refresh` (per-file work) and
:func:`dead_cst.analyze._compose_contribution` (cross-package
composition). Takes a :class:`PackageFiles` plus the visitor payloads
for its miss files and produces one :class:`PackageContribution` --
the per-package trie + a package-local graph slice + the unresolved
cross-file import set used downstream by
:func:`dead_cst._edges.resolve_edges` and the plugin
:meth:`EdgePlugin.finalize` pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
from libcst.metadata import CodeRange

from ._notebooks import is_notebook
from ._refresh import PackageFiles
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode, SymbolTrie, VisitorPayload
from .resolvers import Package

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PackageContribution:
    """One package's pre-stitched contribution to the symbol graph.

    Built once per package by :func:`build_contribution` and composed
    into a target graph by :func:`dead_cst.analyze._compose_contribution`,
    which adds cross-package edges via :func:`resolve_edges` and runs
    plugin :meth:`EdgePlugin.finalize` against the composed graph.

    ``package_graph.graph["dead_suites"]`` carries this package's
    per-file dead-suite positions; the compose step folds them into the
    target graph's matching key.
    """

    package: Package
    current_trie: SymbolTrie
    export_trie: SymbolTrie
    package_graph: nx.MultiDiGraph
    import_edges: frozenset[tuple[SymbolNode, Import, EdgeFlags]]
    module_nodes: tuple[SymbolNode, ...]


def build_contribution(
    package: Package,
    package_files: PackageFiles,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> PackageContribution:
    """Apply ``package_files``' per-file payloads into a package-local graph slice.

    Hits come straight from :class:`PackageFiles`; the rest are looked
    up in the global ``miss_payloads`` map produced by
    :func:`dead_cst._refresh.process_stale_files`. The package-local
    :class:`nx.MultiDiGraph` is what makes scope-bounded materialization
    cheap: composing it into the full graph or a closure graph doesn't
    redo per-file apply work. Empty :attr:`Package.exported` means
    "no restriction" (every file in the package is exported to consumers).

    A pre-pass identifies module-FQN collisions (``foo.py`` alongside
    ``foo/__init__.py``) via :func:`eclipsed_paths` so the loser is
    skipped at the trie -- the visitor still graphs its nodes so any
    observe-time entrypoints (``__main__``, plugin synthetics) keep
    working, but cross-module imports route to the package winner.
    """
    current_trie = SymbolTrie()
    export_trie = SymbolTrie()
    exported = package.exported
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    package_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    package_graph.graph["dead_suites"] = {}
    module_nodes: list[SymbolNode] = []
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
        file_exported = not exported or _under_any(file, list(exported))
        module_nodes.append(
            _apply_payload(
                payload,
                current_trie=current_trie,
                export_trie=export_trie,
                file_exported=file_exported,
                eclipsed=file in eclipsed,
                symbol_graph=package_graph,
                import_edges=import_edges,
            )
        )
    current_trie.add_module_hierarchy_edges(package_graph)
    return PackageContribution(
        package=package,
        current_trie=current_trie,
        export_trie=export_trie,
        package_graph=package_graph,
        import_edges=frozenset(import_edges),
        module_nodes=tuple(module_nodes),
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
    current_trie: SymbolTrie,
    export_trie: SymbolTrie,
    file_exported: bool,
    eclipsed: bool,
    symbol_graph: nx.MultiDiGraph,
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
) -> SymbolNode:
    """Emit ``payload`` into the in-progress per-package structures.

    Drives all node routing off ``SymbolNode.flags`` and ``type``:

    * ``type == "module"`` goes into the graph and the trie; no parent
      edge (modules are themselves the parent target).
    * ``type == "synthetic"`` (plugin-emitted markers) goes into the
      graph only -- no parent edge, no trie entry. Synthetic fqnames
      don't fit the dotted module hierarchy and aren't lookup targets
      for cross-module imports.
    * Other decls go into the graph with a parent-module edge.
      ``NodeFlags.SHADOWED`` excludes them from the trie -- the graph
      keeps the parent edge so the decl stays well-formed, but
      cross-module imports never resolve to it.
    * ``NodeFlags.ENTRYPOINT`` (typically on plugin synthetics)
      seeds reachability: ``graph.nodes[node]["entrypoint"] = True``
      so :func:`find_reachable` starts its BFS from this node.
    * ``NodeFlags.TESTCASE`` mirrors into a
      ``graph.nodes[node]["testcase"] = True`` attribute so
      :func:`find_reachable_excluding_tests` can drop test seeds when
      computing the "blast radius" of removing the test suite.
    * ``NodeFlags.NOQA`` is read straight off the ``SymbolNode`` (no
      attr-dict mirror) by :func:`find_reachable_excluding`, which
      filters seeds via ``n.flags & exclude_flags`` for the "blast
      radius" of removing every noqa-pinned import.

    Edge flag derivation: each ``(src, dst, access_pos)`` entry has
    its access position tested against ``payload.dead_suites`` for
    containment. If matched, the resulting graph edge gets
    :data:`EdgeFlags.DEAD_BRANCH`. Plugin-emitted edges use
    ``SYNTHETIC_POSITION`` (line 0), which never falls inside a real
    dead suite, so they always land with ``EdgeFlags.NONE``.
    Unresolved cross-file imports accumulate into ``import_edges``
    along with the derived flag and are fed to :func:`resolve_edges`
    once the per-package trie is fully built; resolution preserves
    the flag through every emission.

    Per-file dead-suite positions are stashed on the graph as
    ``graph.graph["dead_suites"][module.path]`` for downstream
    reporting (e.g. "this file has unreachable code at line X").
    """
    module = next(n for n in payload.nodes if n.type == "module")

    dead_suites = payload.dead_suites
    if dead_suites:

        def flag_for(pos: CodeRange) -> EdgeFlags:
            return (
                EdgeFlags.DEAD_BRANCH
                if any(_contains(s, pos) for s in dead_suites)
                else EdgeFlags.NONE
            )
    else:

        def flag_for(pos: CodeRange) -> EdgeFlags:
            return EdgeFlags.NONE

    for n in payload.nodes:
        symbol_graph.add_node(n)
        if n.flags & NodeFlags.ENTRYPOINT:
            symbol_graph.nodes[n]["entrypoint"] = True
        if n.flags & NodeFlags.TESTCASE:
            symbol_graph.nodes[n]["testcase"] = True
        if n.type == "synthetic":
            continue
        if n.type != "module":
            symbol_graph.add_edge(n, module, flags=EdgeFlags.NONE)
        # ``OVERLOAD`` and ``SHADOWED`` exclude a single decl from the
        # cross-module lookup trie; ``eclipsed=True`` is the
        # file-level equivalent for a ``.py`` whose sibling package
        # eclipses it. Either way the graph keeps the parent edge so
        # the decl is well-formed -- only consumer FQN lookups change.
        if not eclipsed and not (
            n.flags & (NodeFlags.SHADOWED | NodeFlags.OVERLOAD | NodeFlags.NOTEBOOK)
        ):
            current_trie.add_declaration(n)
            if file_exported:
                export_trie.add_declaration(n)

    for src, dst, pos in payload.edges:
        symbol_graph.add_edge(src, dst, flags=flag_for(pos))

    for src, imp, pos in payload.imports:
        import_edges.add((src, imp, flag_for(pos)))

    if payload.dead_suites:
        symbol_graph.graph["dead_suites"][module.path] = payload.dead_suites

    return module


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


def _under_any(file: Path, roots: list[Path]) -> bool:
    """True iff ``file`` is equal to or nested under any of ``roots``."""
    f = file.resolve()
    for r in roots:
        if f == r or f.is_relative_to(r):
            return True
    return False
