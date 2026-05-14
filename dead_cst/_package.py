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
from .plugins._core import module_node
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

    Surfaced to plugin authors as :attr:`dead_cst.plugins.PluginContext.contribution`
    so finalize passes can read package-local data (the per-package
    trie, the contributed node set, the raw edge / import-edge triples,
    the dead-suite map) without reaching back into the cross-package
    graph.
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
    *,
    dep_contributions: Sequence[PackageContribution] = (),
) -> PackageContribution:
    """Apply ``package_files``' per-file payloads into raw sets / dicts.

    Two-pass over the files:

    1. **Per-file apply.** Hits come straight from :class:`PackageFiles`;
       the rest are looked up in the global ``miss_payloads`` map produced
       by :func:`dead_cst._refresh.process_stale_files`. A pre-pass
       identifies module-FQN collisions (``foo.py`` alongside
       ``foo/__init__.py``) via :func:`eclipsed_paths` so the loser is
       skipped at the trie -- its nodes still land in the contribution
       so observe-time entrypoints (``__main__``, plugin synthetics)
       keep working, but cross-module imports route to the package
       winner. Empty :attr:`Package.exported` means "no restriction"
       (every file in the package is exported to consumers).
    2. **Star re-export materialization** -- :func:`_materialize_star_reexports`.
       For each module-level ``from X import *`` collected in pass 1,
       look up ``X`` in the package's own trie or in
       ``dep_contributions`` and synthesize a ``"import"``-typed
       :class:`SymbolNode` in the importing module for every name the
       target exposes. ``dep_contributions`` is read directly: callers
       must pass each dep that's already had its own contribution built
       (the refresh loop iterates packages in dep order, so this holds).

    The materialization is what makes cross-module
    ``from <pkg> import <name>`` resolve through to ``<name>``'s real
    source when ``<pkg>/__init__.py`` only re-exports via star, and
    makes downstream ``from <pkg> import *`` fan-outs transitive through
    every re-export chain without any special-casing in the edge
    stitcher.
    """
    trie = SymbolTrie()
    nodes: set[SymbolNode] = set()
    edges: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = set()
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    dead_suites: dict[Path, tuple[CodeRange, ...]] = {}
    star_records: list[tuple[SymbolNode, Import, EdgeFlags]] = []
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
            star_records=star_records,
        )
    _materialize_star_reexports(
        star_records=star_records,
        trie=trie,
        dep_contributions=dep_contributions,
        nodes=nodes,
        edges=edges,
        import_edges=import_edges,
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
    star_records: list[tuple[SymbolNode, Import, EdgeFlags]],
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
    module = module_node(payload)
    assert module is not None, "payload must include a module node"

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
        flags = flag_for(pos)
        import_edges.add((src, imp, flags))
        # Track module-level star imports for pass 2; sub-module-level
        # stars (inside class bodies, conditional blocks) are rare and
        # don't participate in cross-module re-export resolution, so
        # the ``src is module`` guard filters them out.
        if imp.star and src is module and not eclipsed:
            star_records.append((src, imp, flags))

    if payload_dead_suites:
        dead_suites[module.path] = payload_dead_suites


def _materialize_star_reexports(
    *,
    star_records: list[tuple[SymbolNode, Import, EdgeFlags]],
    trie: SymbolTrie,
    dep_contributions: Sequence[PackageContribution],
    nodes: set[SymbolNode],
    edges: set[tuple[SymbolNode, SymbolNode, EdgeFlags]],
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
) -> None:
    """Synthesize ``"import"`` nodes for each name a ``from X import *`` re-exports.

    For each module-level star recorded in pass 1, look ``X`` up in a
    transient lookup trie (this package's own trie plus each dep's
    exported view), then for every name the target's trie node holds,
    create one synthetic re-export node in the importing module:

    * ``type="import"``, ``imports=Import(module=<target>, decl=<name>)``
      -- :func:`dead_cst._edges.resolve_edges` treats it like any other
      re-export decl, chaining the consumer's ``from <importer> import
      <name>`` through to ``<name>``'s real source.
    * ``flags`` carries :data:`NodeFlags.STAR_REEXPORT` so the codemod
      skips it (the file has no literal ``from <target> import <name>``
      line to remove). Inherits :data:`NodeFlags.EXPORTED` from the
      importing module so the re-exports flow through
      :meth:`SymbolTrie.merge_exported` to downstream packages.
    * Two edges: ``synthetic -> module`` (the standard decl parent edge)
      and ``module -> synthetic`` (so the re-export is alive whenever
      the importing module is alive, mirroring today's pessimistic
      "star keeps target decls alive" behavior). The synthetic's
      ``Import`` is added to ``import_edges`` so the edge stitcher
      emits the ``synthetic -> target.<name>`` chain.

    Shadowing rule: if a real decl, an explicit import, or a prior
    star re-export already claims the name in the importing module's
    trie, we skip -- "first writer wins" within the module. Stars
    are processed in source order (by line, column, target FQN) so
    the result is deterministic across runs.

    Star chains within the package converge via fixed-point iteration:
    each round materializes synthetics that the previous round made
    visible, and the loop exits when a pass adds nothing new. Cycles
    (``A: from B import *`` / ``B: from A import *``) terminate after
    one trip around because the ``seen`` set tracks
    ``(importer_fqname, target_fqname, name)`` triples.
    """
    if not star_records:
        return

    lookup = SymbolTrie()
    lookup.merge(trie)
    for dep_contrib in dep_contributions:
        lookup.merge_exported(dep_contrib.trie)

    sorted_records = sorted(
        star_records,
        key=lambda r: (
            r[0].fqname,
            r[1].module,
            r[0].position.start.line,
            r[0].position.start.column,
        ),
    )

    seen: set[tuple[str, str, str]] = set()
    while True:
        progress = False
        for src_module, imp, flags in sorted_records:
            target = lookup._get(imp.module.split("."))
            if target is None or target.module is None:
                continue
            target_fqname = target.module.fqname
            src_trie_node = trie._get(src_module.fqname.split("."))
            if src_trie_node is None:
                continue
            inherited_export = src_module.flags & NodeFlags.EXPORTED
            for name in list(target.declarations.keys()):
                key = (src_module.fqname, target_fqname, name)
                if key in seen:
                    continue
                seen.add(key)
                if name in src_trie_node.declarations:
                    continue
                reexport_import = Import(module=target_fqname, decl=name)
                reexport = SymbolNode(
                    fqname=f"{src_module.fqname}.{name}",
                    type="import",
                    path=src_module.path,
                    position=src_module.position,
                    imports=reexport_import,
                    flags=NodeFlags.STAR_REEXPORT | inherited_export,
                )
                nodes.add(reexport)
                edges.add((reexport, src_module, EdgeFlags.NONE))
                edges.add((src_module, reexport, flags))
                import_edges.add((reexport, reexport_import, flags))
                trie.add_declaration(reexport)
                lookup.add_declaration(reexport)
                progress = True
        if not progress:
            break


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
