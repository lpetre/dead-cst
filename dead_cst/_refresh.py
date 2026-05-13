"""Per-file refresh pipeline: enumerate, parse, cache.

Everything that runs once per file lives here:

* :func:`enumerate_files` walks a package and partitions its ``.py``
  tree into cache hits and misses against the analysis fingerprint.
* :func:`build_stale_tasks` flattens every package's misses into one
  global, deterministic task list -- one ``gen_cache`` call per package
  populates each task's FQN entry.
* :func:`process_stale_files` runs the visitor + observe pass over
  every task (parallel when ``workers`` permits), writing each result
  back into the cache as it lands.

The per-package apply step (:func:`dead_cst._package.build_contribution`)
consumes these payloads to produce the per-package trie / graph slice /
unresolved import set used downstream by
:func:`dead_cst._edges.resolve_edges` and the plugin
:meth:`EdgePlugin.finalize` pass.

:class:`Analysis` orchestrates the steps and owns the memo dicts; this
module only knows how to do one file's worth of work given the shared
cache / detector / plugins.
"""

from __future__ import annotations

import logging
import signal
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import libcst as cst
from libcst.helpers.module import ModuleNameAndPackage
from libcst.metadata import CodeRange, MetadataWrapper

from ._fqn import FixedFullyQualifiedNameProvider
from ._notebooks import is_notebook, notebook_fqn_entry, notebook_to_module
from ._progress import progress
from ._visitor import SymbolVisitor
from .branches import UnreachableRegionDetector
from .cache import GraphCache
from .graph import Import, NodeFlags, SymbolNode, VisitorPayload
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
    (path + exported + deps), not just the directory. ``fingerprint``
    is the per-package cache key the worker writes alongside the
    payload -- per-package because :attr:`Package.exported` enters the
    fingerprint, so siblings with different export configurations get
    independent invalidation.
    """

    file: Path
    package: Package
    fqn_entry: ModuleNameAndPackage
    project_root: Path
    fingerprint: str


def enumerate_files(
    package: Package,
    cache: GraphCache | None,
    fingerprint: str,
) -> PackageFiles:
    """Walk ``package.path`` once, classifying ``.py`` / ``.pyi`` / ``.ipynb``.

    A ``.pyi`` whose ``.py`` twin also exists is skipped at this layer:
    ingesting both would assert in the symbol trie when they claim the
    same FQN, and dead-cst has no peer-stub linker. Orphan ``.pyi``
    (compiled-extension shape) flows through under its natural FQN.
    Notebooks aren't importable; ``_process_one_file`` stamps them with
    ``NodeFlags.NOTEBOOK`` via the visitor's ``default_flags``.

    ``rglob`` matches by name, so a *directory* literally named
    ``something.py`` would otherwise sneak in and crash the visitor on
    ``read_text``. Filter to real files defensively. One walk over the
    tree (vs three suffix-specific globs) is the cheap path on large
    repos where directory I/O dominates the per-name fnmatch cost.
    """
    py_files: list[Path] = []
    pyi_candidates: list[Path] = []
    ipynb_files: list[Path] = []
    for p in package.path.rglob("*"):
        if not p.is_file():
            continue
        match p.suffix:
            case ".py":
                py_files.append(p)
            case ".pyi":
                pyi_candidates.append(p)
            case ".ipynb":
                ipynb_files.append(p)
    py_stems = {p.with_suffix("") for p in py_files}
    pyi_files = [p for p in pyi_candidates if p.with_suffix("") not in py_stems]
    files = tuple(sorted([*py_files, *pyi_files, *ipynb_files]))
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
    fingerprints: Mapping[Path, str],
) -> list[StaleFile]:
    """Flatten every package's miss files into one global, deterministic task list.

    One ``gen_cache`` call per package (FQN resolution is package-keyed)
    populates each task's ``fqn_entry``. Sorting on ``(package_path, file)``
    keeps related tasks together for log readability and makes
    parallel-pool output ordering reproducible. Each task carries its
    package's fingerprint so the worker writes the correct cache key
    even when packages are batched into one parallel pool.
    """
    tasks: list[StaleFile] = []
    for package_path in sorted(package_files):
        pf = package_files[package_path]
        if not pf.miss_files:
            continue
        # ``gen_cache`` has no notion of notebooks; synthesize FQNs for them.
        gen_cache_files = [str(f) for f in pf.miss_files if not is_notebook(f)]
        fqn_cache: dict[str, ModuleNameAndPackage] = (
            dict(
                FixedFullyQualifiedNameProvider.gen_cache(package_path, gen_cache_files, timeout=5)
            )
            if gen_cache_files
            else {}
        )
        fingerprint = fingerprints[package_path]
        for file in pf.miss_files:
            fqn_entry = notebook_fqn_entry(file) if is_notebook(file) else fqn_cache[str(file)]
            tasks.append(
                StaleFile(
                    file=file,
                    package=pf.package,
                    fqn_entry=fqn_entry,
                    project_root=project_root,
                    fingerprint=fingerprint,
                )
            )
    return tasks


def process_stale_files(
    *,
    tasks: Sequence[StaleFile],
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    cache: GraphCache | None,
    workers: int | None,
) -> dict[Path, VisitorPayload]:
    """Run visitor + observe across every task; return ``file -> payload``.

    Per-task failures other than the ``OSError`` / parse cases that
    ``_process_one_file`` absorbs in-band are collected and re-raised
    as a single :class:`ExceptionGroup` after the run drains, so one
    bad file does not waste the rest of the work. Successful payloads
    are cache-warmed before the group is raised; each task carries its
    package's fingerprint so cache writes use the correct key even
    when tasks from multiple packages share one pool.
    """
    if not tasks:
        return {}

    out: dict[Path, VisitorPayload] = {}
    failures: list[tuple[Path, Exception]] = []
    total = len(tasks)

    def _record(task: StaleFile, payload: VisitorPayload | None) -> None:
        if payload is None:
            return
        out[task.file] = payload
        if cache is not None:
            cache.put(task.file, payload, task.fingerprint)

    use_pool = workers is not None and workers >= 2 and total >= 2

    if use_pool:
        assert workers is not None
        _run_pool(
            tasks=tasks,
            workers=workers,
            detector=detector,
            plugins=plugins,
            record=_record,
            failures=failures,
        )
    else:
        plugins_t = tuple(plugins)
        for idx, task in enumerate(
            progress(tasks, total=total, desc="Parsing files", unit="file"), 1
        ):
            try:
                _, payload = _process_task(detector, plugins_t, task)
            except Exception as exc:
                failures.append((task.file, exc))
                logger.debug("[%d/%d] FAILED %s", idx, total, task.file)
                continue
            _record(task, payload)
            logger.debug("[%d/%d] ok %s", idx, total, task.file)

    if failures:
        raise ExceptionGroup(
            f"dead-cst refresh: {len(failures)}/{total} file(s) failed",
            [exc for _, exc in failures],
        )
    return out


def _run_pool(
    *,
    tasks: Sequence[StaleFile],
    workers: int,
    detector: UnreachableRegionDetector,
    plugins: Sequence[EdgePlugin],
    record: Callable[[StaleFile, VisitorPayload | None], None],
    failures: list[tuple[Path, Exception]],
) -> None:
    """Run ``tasks`` through a :class:`ProcessPoolExecutor`.

    Installs SIGTERM/SIGINT handlers for the pool's lifetime so a
    signal cancels pending futures and re-raises
    :class:`KeyboardInterrupt`; files that completed before the
    signal stay cache-warmed via ``record``. Signal install is
    skipped off the main thread, where :func:`signal.signal` raises.
    """
    cancelled = threading.Event()
    total = len(tasks)

    def _on_signal(signum: int, _frame: object) -> None:
        cancelled.set()
        logger.warning(
            "Received signal %s; cancelling pending dead-cst tasks...",
            signal.Signals(signum).name,
        )

    prev_handlers: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev_handlers[sig] = signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass

    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, total),
            initializer=_init_worker,
            initargs=(detector, tuple(plugins)),
        ) as pool:
            futures = {pool.submit(_worker_process_task, task): task for task in tasks}
            for idx, future in enumerate(
                progress(as_completed(futures), total=total, desc="Parsing files", unit="file"),
                1,
            ):
                if cancelled.is_set():
                    break
                task = futures[future]
                try:
                    _, payload = future.result()
                except Exception as exc:
                    failures.append((task.file, exc))
                    logger.debug("[%d/%d] FAILED %s", idx, total, task.file)
                    continue
                record(task, payload)
                logger.debug("[%d/%d] ok %s", idx, total, task.file)
            if cancelled.is_set():
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise KeyboardInterrupt("dead-cst refresh cancelled")
    finally:
        for sig, prev in prev_handlers.items():
            signal.signal(sig, prev)  # type: ignore[arg-type]


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
    notebook = is_notebook(file)
    default_flags = NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT if notebook else NodeFlags.NONE
    if not package.exported or _under_any(file, package.exported):
        default_flags |= NodeFlags.EXPORTED
    if notebook:
        nb_source = notebook_to_module(file)
        if nb_source is None:
            return _unparseable_payload(file, fqn_entry, default_flags=default_flags)
        source = nb_source
    else:
        try:
            source = file.read_text()
        except OSError as exc:
            logger.warning("Skipping %s: could not read file: %s", file, exc)
            return None
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        logger.warning("Could not parse %s: %s; emitting [unparseable] marker", file, exc)
        return _unparseable_payload(file, fqn_entry, default_flags=default_flags)
    wrapper = MetadataWrapper(
        module,
        unsafe_skip_copy=True,
        cache={FixedFullyQualifiedNameProvider: fqn_entry},
    )
    visitor = SymbolVisitor(
        file, unreachable_detector=detector, wrapper=wrapper, default_flags=default_flags
    )
    wrapper.visit(visitor)
    base_payload = visitor.to_payload()
    plugin_payload = _run_observe(
        plugins, file, wrapper.module, base_payload, package, project_root
    )
    return _merge_payloads(base_payload, plugin_payload)


def _unparseable_payload(
    file: Path,
    fqn_entry: ModuleNameAndPackage,
    *,
    default_flags: NodeFlags = NodeFlags.NONE,
) -> VisitorPayload:
    """Build the placeholder payload for a file libcst could not parse.

    Pairs the real module node (so importers and the symbol trie still
    see the module under its natural FQN) with a synthetic
    ``[unparseable] <module>`` node flagged ``ENTRYPOINT``. The synthetic
    keeps the file alive during reachability -- we cannot prove its
    contents are dead -- and gives downstream queries (``why-alive``,
    reports) a stable handle for "this file did not parse".

    ``default_flags`` mirrors ``SymbolVisitor``'s knob: an unparseable
    notebook still lands flagged ``NOTEBOOK`` so the codemod gate and
    trie exclusion stay consistent with the parseable path.
    """
    module_node_ = SymbolNode(
        fqname=fqn_entry.name,
        type="module",
        path=file,
        position=SYNTHETIC_POSITION,
        flags=default_flags,
    )
    marker = synthetic_node(
        f"{UNPARSEABLE_PREFIX}{fqn_entry.name}",
        file,
        flags=NodeFlags.ENTRYPOINT | default_flags,
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


def _under_any(file: Path, roots: tuple[Path, ...]) -> bool:
    """True iff ``file`` is nested under any of ``roots``.

    Both inputs are absolute by construction (``enumerate_files`` walks
    ``package.path.rglob`` and ``_validate_packages`` absolutizes every
    ``Package.exported`` entry), so no ``resolve()`` is needed.
    """
    return any(file.is_relative_to(r) for r in roots)
