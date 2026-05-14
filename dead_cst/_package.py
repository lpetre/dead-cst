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


def compose_lookup(
    own_trie: SymbolTrie, dep_contributions: Sequence[PackageContribution]
) -> SymbolTrie:
    """Build a per-package lookup trie: own trie + each dep's exports.

    ``merge`` pulls in every entry from the package's own trie (the
    package sees itself fully). ``merge_exported`` filters each dep's
    trie to entries flagged :data:`NodeFlags.EXPORTED`, so dep-internal
    decls stay invisible to the consumer. Shared by
    :func:`_materialize_star_reexports` (called during
    :func:`build_contribution`) and
    :meth:`dead_cst.analyze.Analysis._build_symbol_lookup` (called at
    compose time) so the "what's visible to this package" rule lives
    in one place.
    """
    lookup = SymbolTrie()
    lookup.merge(own_trie)
    for dep in dep_contributions:
        lookup.merge_exported(dep.trie)
    return lookup


def build_contribution(
    package: Package,
    package_files: PackageFiles,
    miss_payloads: Mapping[Path, VisitorPayload],
    *,
    dep_contributions: Sequence[PackageContribution] = (),
) -> PackageContribution:
    """Apply ``package_files``' per-file payloads into raw sets / dicts.

    Two passes: per-file ``_apply_payload`` (hits from
    :class:`PackageFiles`, misses from ``miss_payloads``), then
    :func:`_materialize_star_reexports` against the assembled trie plus
    ``dep_contributions``. Callers must pass deps whose contributions
    are already built; the refresh loop walks packages in dep order to
    uphold this. Empty :attr:`Package.exported` means "no restriction"
    -- every file in the package is exported to consumers. Eclipsed
    files (``foo.py`` next to ``foo/__init__.py``, per
    :func:`eclipsed_paths`) still land in the contribution but skip
    the trie so consumer imports route to the package winner.
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
    _materialize_star_reexports(
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
        import_edges.add((src, imp, flag_for(pos)))

    if payload_dead_suites:
        dead_suites[module.path] = payload_dead_suites


def _materialize_star_reexports(
    *,
    trie: SymbolTrie,
    dep_contributions: Sequence[PackageContribution],
    nodes: set[SymbolNode],
    edges: set[tuple[SymbolNode, SymbolNode, EdgeFlags]],
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
) -> None:
    """Synthesize one ``"import"`` decl per name ``from X import *`` re-exports.

    Shadowing: if any decl already claims the name in the importing
    module's trie (a real def, an explicit import, or a prior star
    re-export) the synthetic is skipped -- "first writer wins" within
    the module. Stars sort by source position so the order is
    deterministic; see ``last-star-wins-not-implemented`` in
    ``tests/test_limitations.py`` for the Python-runtime gap this
    leaves.

    Star chains converge via fixed-point iteration -- a star whose
    target itself contains a star needs the inner star to materialize
    before its names become visible. The ``seen`` set keyed on
    ``(importer_fqname, target_fqname, name)`` makes cycles like
    ``A: from B import *`` / ``B: from A import *`` terminate after
    one trip.
    """
    star_records = [
        (src, imp, flags, src.fqname.split("."), imp.module.split("."))
        for src, imp, flags in import_edges
        if imp.star and src.type == "module"
    ]
    if not star_records:
        return

    lookup = compose_lookup(trie, dep_contributions)
    star_records.sort(
        key=lambda r: (
            r[0].fqname,
            r[0].position.start.line,
            r[0].position.start.column,
            r[1].module,
        )
    )

    seen: set[tuple[str, str, str]] = set()
    while True:
        progress = False
        for src_module, imp, flags, src_parts, target_parts in star_records:
            target = lookup._get(target_parts)
            if target is None or target.module is None:
                continue
            src_trie_node = trie._get(src_parts)
            if src_trie_node is None:
                continue
            target_fqname = target.module.fqname
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
