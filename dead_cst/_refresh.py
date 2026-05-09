"""Per-file refresh pipeline: enumerate, parse, cache, apply.

Everything that runs once per file lives here:

* :func:`enumerate_files` walks a package and partitions its ``.py``
  tree into cache hits and misses against the analysis fingerprint.
* :func:`build_stale_tasks` flattens every package's misses into one
  global, deterministic task list -- one ``gen_cache`` call per package
  populates each task's FQN entry.
* :func:`process_stale_files` runs the visitor + observe pass over
  every task (parallel when ``workers`` permits), writing each result
  back into the cache as it lands.
* :func:`build_contribution` applies the per-file payloads (hits +
  fresh misses) into the package-local trie / graph slice / unresolved
  import set used downstream by :func:`dead_cst._edges.resolve_edges`
  and the plugin :meth:`EdgePlugin.finalize` pass.

:class:`Analysis` orchestrates the steps and owns the memo dicts; this
module only knows how to do one package's worth of work given that
package's :class:`Package` and the shared cache / detector / plugins.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import libcst as cst
import networkx as nx
from libcst.helpers.module import ModuleNameAndPackage
from libcst.metadata import CodeRange, MetadataWrapper

from ._fqn import FixedFullyQualifiedNameProvider
from ._progress import progress
from ._visitor import SymbolVisitor
from .branches import UnreachableRegionDetector
from .cache import GraphCache
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode, SymbolTrie, VisitorPayload
from .plugins import EdgePlugin, ObserveContext
from .plugins._core import (
    SYNTHETIC_POSITION,
    UNPARSEABLE_PREFIX,
    make_payload,
    synthetic_node,
)
from .resolvers import Package

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PackageFiles:
    """One package's ``.py`` enumeration partitioned into cache hits and misses.

    Built once per package by :func:`enumerate_files` and parked on
    :class:`Analysis` so :meth:`Analysis.refresh` can rebuild
    contributions later without re-walking the tree.
    """

    package: Package
    files: tuple[Path, ...]
    hits: dict[Path, VisitorPayload]
    miss_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class StaleFile:
    """One stale file ready for the visitor + observe pass.

    ``fqn_entry`` is this file's slice of the per-package FQN cache
    (FQN resolution is package-keyed, hence one ``gen_cache`` call per
    package in :func:`build_stale_tasks`); the runner injects it into
    a :class:`MetadataWrapper` directly. ``package`` rides through to
    :class:`ObserveContext` so plugins see the full :class:`Package`
    (path + exported + deps), not just the directory.
    """

    file: Path
    package: Package
    fqn_entry: ModuleNameAndPackage
    project_root: Path


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


def enumerate_files(
    package: Package,
    cache: GraphCache | None,
    fingerprint: str,
) -> PackageFiles:
    """Walk ``package.path``'s ``.py`` / ``.pyi`` tree, classify each file as cache hit or miss.

    A ``.pyi`` whose ``.py`` twin also exists is skipped at this layer:
    ingesting both would assert in the symbol trie when they claim the
    same FQN, and dead-cst has no peer-stub linker. Orphan ``.pyi``
    (compiled-extension shape) flows through under its natural FQN.

    ``rglob`` matches by name, so a *directory* literally named
    ``something.py`` would otherwise sneak in and crash the visitor on
    ``read_text``. Filter to real files defensively.
    """
    py_files = sorted(p for p in package.path.rglob("*.py") if p.is_file())
    py_stems = {p.with_suffix("") for p in py_files}
    pyi_files = (
        p for p in package.path.rglob("*.pyi") if p.is_file() and p.with_suffix("") not in py_stems
    )
    files = tuple(sorted([*py_files, *pyi_files]))
    hits: dict[Path, VisitorPayload] = {}
    miss_files: list[Path] = []
    for file in files:
        payload = cache.get(file, fingerprint) if cache is not None else None
        if payload is None:
            miss_files.append(file)
        else:
            hits[file] = payload
    return PackageFiles(
        package=package,
        files=files,
        hits=hits,
        miss_files=tuple(miss_files),
    )


def build_stale_tasks(
    package_files: Mapping[Path, PackageFiles],
    project_root: Path,
) -> list[StaleFile]:
    """Flatten every package's miss files into one global, deterministic task list.

    One ``gen_cache`` call per package (FQN resolution is package-keyed)
    populates each task's ``fqn_entry``. Sorting on ``(package_path, file)``
    keeps related tasks together for log readability and makes
    parallel-pool output ordering reproducible.
    """
    tasks: list[StaleFile] = []
    for package_path in sorted(package_files):
        pf = package_files[package_path]
        if not pf.miss_files:
            continue
        fqn_cache = FixedFullyQualifiedNameProvider.gen_cache(
            package_path, [str(f) for f in pf.miss_files], timeout=5
        )
        tasks.extend(
            StaleFile(
                file=file,
                package=pf.package,
                fqn_entry=fqn_cache[str(file)],
                project_root=project_root,
            )
            for file in pf.miss_files
        )
    return tasks


def process_stale_files(
    *,
    tasks: Sequence[StaleFile],
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    cache: GraphCache | None,
    fingerprint: str,
    workers: int | None,
) -> dict[Path, VisitorPayload]:
    """Run visitor + observe across every task; return ``file -> payload``.

    Both branches use :func:`_process_task` for the per-task work;
    they differ only in whether the runner state lives on the main
    process or in :class:`ProcessPoolExecutor` workers. The pool is
    opt-in (``workers >= 2`` and at least two tasks); below that, the
    in-process path avoids pool startup cost.

    Cache writes happen on the main process as each payload arrives,
    so a partial run still warms the cache for files that completed.
    Files ``_process_one_file`` could not even read off disk return
    ``None`` and are dropped here -- not cached, not surfaced to the
    caller -- so the warning re-fires next run. Files that *parse*
    fail come back as an ``[unparseable]`` synthetic payload and ride
    the cache like any other miss.
    """
    if not tasks:
        return {}

    out: dict[Path, VisitorPayload] = {}
    use_pool = workers is not None and workers >= 2 and len(tasks) >= 2

    def _record(file: Path, payload: VisitorPayload | None) -> None:
        if payload is None:
            return
        out[file] = payload
        if cache is not None:
            cache.put(file, payload, fingerprint)

    if use_pool:
        assert workers is not None
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_init_worker,
            initargs=(detector, tuple(plugins)),
        ) as pool:
            for file, payload in progress(
                pool.map(_worker_process_task, tasks),
                total=len(tasks),
                desc="Parsing files",
                unit="file",
            ):
                _record(file, payload)
        return out

    plugins_t = tuple(plugins)
    for task in progress(tasks, total=len(tasks), desc="Parsing files", unit="file"):
        file, payload = _process_task(detector, plugins_t, task)
        _record(file, payload)
    return out


def build_contribution(
    package: Package,
    package_files: PackageFiles,
    miss_payloads: Mapping[Path, VisitorPayload],
) -> PackageContribution:
    """Apply ``package_files``' per-file payloads into a package-local graph slice.

    Hits come straight from :class:`PackageFiles`; the rest are looked
    up in the global ``miss_payloads`` map produced by
    :func:`process_stale_files`. The package-local
    :class:`nx.MultiDiGraph` is what makes scope-bounded materialization
    cheap: composing it into the full graph or a closure graph doesn't
    redo per-file apply work. Empty :attr:`Package.exported` means
    "no restriction" (every file in the package is exported to consumers).
    """
    current_trie = SymbolTrie()
    export_trie = SymbolTrie()
    exported = package.exported
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]] = set()
    package_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    package_graph.graph["dead_suites"] = {}
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
        _apply_payload(
            payload,
            current_trie=current_trie,
            export_trie=export_trie,
            file_exported=file_exported,
            symbol_graph=package_graph,
            import_edges=import_edges,
        )
    current_trie.add_module_hierarchy_edges(package_graph)
    return PackageContribution(
        package=package,
        current_trie=current_trie,
        export_trie=export_trie,
        package_graph=package_graph,
        import_edges=frozenset(import_edges),
    )


# ---------------------------------------------------------------------------
# Per-file work (visitor + observe)
# ---------------------------------------------------------------------------


def _process_one_file(
    file: Path,
    *,
    fqn_entry: ModuleNameAndPackage,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    package: Package,
    project_root: Path,
) -> VisitorPayload | None:
    """Run the visitor + observe pass for a single file and return its payload.

    The caller owns the precomputed FQN entry (built once per package
    by :func:`build_stale_tasks`) so we can construct
    :class:`MetadataWrapper` directly with ``cache=`` injected,
    skipping :class:`FullRepoManager`'s per-instance ``gen_cache``
    rebuild. Same shape ``FullRepoManager.get_metadata_wrapper_for_path``
    builds, just without re-walking the file list every time.

    Cross-file import resolution moved to
    :func:`dead_cst._edges.resolve_edges`, so this pass is purely a
    function of the file's source -- no ``search_paths`` or resolver
    plumbing is needed here.

    Returns ``None`` when the file cannot even be read off disk
    (rare; the directory filter in :func:`enumerate_files` already
    catches the common ``IsADirectoryError`` case). The caller drops
    such entries without caching them, so a later fix is picked up
    on the next run.

    On :class:`libcst.ParserSyntaxError` (e.g. PEP 750 t-strings the
    pinned libcst can't parse) the file is *not* dropped: the analyser
    emits a minimal payload pairing the real module node with a
    synthetic ``[unparseable] <module>`` entrypoint so the file stays
    alive in reachability and importers can still target the module.
    The payload is cached like any other miss -- a fresh source SHA
    re-runs the parse, so fixing the syntax invalidates the entry.
    """
    try:
        source = file.read_text()
    except OSError as exc:
        logger.warning("Skipping %s: could not read file: %s", file, exc)
        return None
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        logger.warning("Could not parse %s: %s; emitting [unparseable] marker", file, exc)
        return _unparseable_payload(file, fqn_entry)
    wrapper = MetadataWrapper(
        module,
        unsafe_skip_copy=True,
        cache={FixedFullyQualifiedNameProvider: fqn_entry},
    )
    visitor = SymbolVisitor(file, unreachable_detector=detector, wrapper=wrapper)
    wrapper.visit(visitor)
    base_payload = visitor.to_payload()
    plugin_payload = _run_observe(
        plugins, file, wrapper.module, base_payload, package, project_root
    )
    return _merge_payloads(base_payload, plugin_payload)


def _unparseable_payload(
    file: Path,
    fqn_entry: ModuleNameAndPackage,
) -> VisitorPayload:
    """Build the placeholder payload for a file libcst could not parse.

    Pairs the real module node (so importers and the symbol trie still
    see the module under its natural FQN) with a synthetic
    ``[unparseable] <module>`` node flagged ``ENTRYPOINT``. The synthetic
    keeps the file alive during reachability -- we cannot prove its
    contents are dead -- and gives downstream queries (``why-alive``,
    reports) a stable handle for "this file did not parse".
    """
    module_node_ = SymbolNode(
        fqname=fqn_entry.name,
        type="module",
        path=file,
        position=SYNTHETIC_POSITION,
    )
    marker = synthetic_node(
        f"{UNPARSEABLE_PREFIX}{fqn_entry.name}",
        file,
        flags=NodeFlags.ENTRYPOINT,
    )
    return VisitorPayload(
        nodes=(module_node_, marker),
        edges=((marker, module_node_, SYNTHETIC_POSITION),),
        imports=(),
        dead_suites=(),
    )


def _run_observe(
    plugins: Sequence[EdgePlugin],
    path: Path,
    module: cst.Module,
    base_payload: VisitorPayload,
    package: Package,
    project_root: Path,
) -> VisitorPayload:
    """Invoke each plugin's :meth:`EdgePlugin.observe` and collect contributions.

    Returns a single :class:`VisitorPayload` that merges every plugin's
    additions for this file. Plugins that return ``None`` contribute
    nothing. The result is concatenated with the visitor's payload by
    :func:`_merge_payloads` and cached together so that warm runs skip
    both the visitor and the observe pass.
    """
    if not plugins:
        return make_payload()
    ctx = ObserveContext(
        path=path,
        module=module,
        payload=base_payload,
        package=package,
        project_root=project_root,
    )
    payloads: list[VisitorPayload] = []
    for plugin in plugins:
        if not isinstance(plugin, EdgePlugin):
            raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
        contribution = plugin.observe(ctx)
        if contribution is not None:
            payloads.append(contribution)
    return _merge_payloads(*payloads) if payloads else make_payload()


def _process_task(
    detector: UnreachableRegionDetector,
    plugins: tuple[EdgePlugin, ...],
    task: StaleFile,
) -> tuple[Path, VisitorPayload | None]:
    """Run one task; pure (no ``sys.path`` mutation, no resolver call)."""
    payload = _process_one_file(
        task.file,
        fqn_entry=task.fqn_entry,
        detector=detector,
        plugins=plugins,
        package=task.package,
        project_root=task.project_root,
    )
    return task.file, payload


_worker_state: tuple[UnreachableRegionDetector, tuple[EdgePlugin, ...]] | None = None


def _init_worker(
    detector: UnreachableRegionDetector,
    plugins: tuple[EdgePlugin, ...],
) -> None:
    """Pool initializer: stash the worker's detector + plugins."""
    global _worker_state
    _worker_state = (detector, plugins)


def _worker_process_task(task: StaleFile) -> tuple[Path, VisitorPayload | None]:
    """Pool task: delegate to :func:`_process_task` against the worker's state."""
    assert _worker_state is not None, "_init_worker must run before _worker_process_task"
    return _process_task(*_worker_state, task)


# ---------------------------------------------------------------------------
# Payload application
# ---------------------------------------------------------------------------


def _apply_payload(
    payload: VisitorPayload,
    *,
    current_trie: SymbolTrie,
    export_trie: SymbolTrie,
    file_exported: bool,
    symbol_graph: nx.MultiDiGraph,
    import_edges: set[tuple[SymbolNode, Import, EdgeFlags]],
) -> None:
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
        # ``OVERLOAD`` and ``SHADOWED`` both keep the decl out of the
        # cross-module lookup trie; the graph keeps the parent edge so
        # the decl is well-formed but consumer imports route to the
        # impl, never the stub.
        if not (n.flags & (NodeFlags.SHADOWED | NodeFlags.OVERLOAD)):
            current_trie.add_declaration(n)
            if file_exported:
                export_trie.add_declaration(n)

    for src, dst, pos in payload.edges:
        symbol_graph.add_edge(src, dst, flags=flag_for(pos))

    for src, imp, pos in payload.imports:
        import_edges.add((src, imp, flag_for(pos)))

    if payload.dead_suites:
        symbol_graph.graph["dead_suites"][module.path] = payload.dead_suites


def _merge_payloads(*payloads: VisitorPayload) -> VisitorPayload:
    """Concatenate the ``nodes``/``edges``/``imports``/``dead_suites`` of every payload."""
    nodes: list[SymbolNode] = []
    edges: list[tuple[SymbolNode, SymbolNode, CodeRange]] = []
    imports: list[tuple[SymbolNode, Import, CodeRange]] = []
    dead_suites: list[CodeRange] = []
    for p in payloads:
        nodes.extend(p.nodes)
        edges.extend(p.edges)
        imports.extend(p.imports)
        dead_suites.extend(p.dead_suites)
    return VisitorPayload(
        nodes=tuple(nodes),
        edges=tuple(edges),
        imports=tuple(imports),
        dead_suites=tuple(dead_suites),
    )


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
