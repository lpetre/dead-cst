"""Standalone benchmark + cProfile for ``_refresh.build_contribution``
and ``PluginContext.package_nodes``.

Times the per-package apply pass -- the loop that stitches every cached
:class:`VisitorPayload` into a per-package graph + trie -- over the
``dead_cst/`` source tree. The visitor pass is run once up front to
produce the payloads, then :func:`build_contribution` is called in a
hot loop with everything already in ``PackageFiles.hits`` so the
measurement isolates apply-pass cost from libcst parse work.

A second section measures :meth:`PluginContext.package_nodes`, the
helper plugins use during :meth:`EdgePlugin.finalize` to iterate
"every node in this package". It scans the composed cross-package
graph and filters by ``node.path.is_relative_to(package_path)``;
plugins like ``ExplicitEntrypointPlugin``, ``MockPatchPlugin``, and
the framework plugins (FastAPI / Flask / discord.py) call it once
per finalize pass.

Run from the repo root:

    uv run python scripts/profile_build_contribution.py
"""

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

from dead_cst._fqn import FixedFullyQualifiedNameProvider
from dead_cst._refresh import (
    PackageContribution,
    PackageFiles,
    StaleFile,
    build_contribution,
    process_stale_files,
)
from dead_cst.branches import DefaultUnreachableRegionDetector
from dead_cst.cache import GraphCache, compute_fingerprint, default_cache_path
from dead_cst.plugins._core import PluginContext
from dead_cst.resolvers._core import Package
from dead_cst.resolvers._exports import exported_roots

REPEATS = 50


def main() -> None:
    project_root = Path(".").resolve()
    target_dir = project_root / "dead_cst"
    assert target_dir.is_dir(), f"expected {target_dir} to exist"

    detector = DefaultUnreachableRegionDetector()
    fingerprint = compute_fingerprint(plugins=(), unreachable_detector=detector)

    package = Package(
        path=project_root,
        name=project_root.name,
        exported=tuple(exported_roots(project_root) or ()),
        deps=(),
    )

    # Walk only ``dead_cst/`` to avoid the venv / examples / tests trees
    # that an ``rglob`` from ``project_root`` would otherwise pull in.
    files = tuple(sorted(p for p in target_dir.rglob("*.py") if p.is_file()))
    print(f"build_contribution over {len(files)} file(s) under {target_dir.name}/")

    with GraphCache(default_cache_path(project_root)) as cache:
        # Visitor pass (one-time): produce a payload for every file.
        hits: dict[Path, object] = {}
        miss_files: list[Path] = []
        for f in files:
            payload = cache.get(f, fingerprint)
            if payload is None:
                miss_files.append(f)
            else:
                hits[f] = payload

        if miss_files:
            fqn_cache = FixedFullyQualifiedNameProvider.gen_cache(
                project_root, [str(f) for f in miss_files], timeout=30
            )
            tasks = [
                StaleFile(
                    file=f,
                    package=package,
                    fqn_entry=fqn_cache[str(f)],
                    project_root=project_root,
                )
                for f in miss_files
            ]
            t0 = time.perf_counter()
            miss_payloads = process_stale_files(
                tasks=tasks,
                detector=detector,
                plugins=(),
                cache=cache,
                fingerprint=fingerprint,
                workers=None,
            )
            print(
                f"  visitor warmup: {(time.perf_counter() - t0) * 1000:.0f} ms over {len(tasks)} miss(es)"
            )
            hits.update(miss_payloads)

        pf = PackageFiles(
            package=package,
            files=files,
            hits=hits,  # type: ignore[arg-type]
            miss_files=(),
        )

    # Warmup call before timing.
    contrib = build_contribution(package, pf, {})

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        build_contribution(package, pf, {})
    wall = time.perf_counter() - t0
    per_call = wall / REPEATS * 1000
    per_file = per_call / len(files)
    print(f"  wall:     {wall * 1000:7.1f} ms over {REPEATS} calls")
    print(f"  per-call: {per_call:7.2f} ms")
    print(f"  per-file: {per_file:7.3f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(REPEATS):
        build_contribution(package, pf, {})
    profiler.disable()

    print("\nTop 25 functions by cumulative time:")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)

    print("\nTop 25 functions by internal (tottime):")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(25)

    _profile_package_nodes(contrib, project_root)


def _profile_package_nodes(contrib: PackageContribution, project_root: Path) -> None:
    """Benchmark :meth:`PluginContext.package_nodes` on both paths.

    Single-package workload: ``contrib.package_graph`` *is* the composed
    graph, so the legacy ``is_relative_to`` filter is a worst-case "every
    node matches" scan. The seeded path (``package_graph=...`` passed to
    :class:`PluginContext`) sidesteps the filter entirely.
    """
    graph = contrib.package_graph
    n_total = graph.number_of_nodes()
    print(f"\npackage_nodes: composed graph has {n_total} node(s)")

    def _make_filter_ctx() -> PluginContext:
        return PluginContext(
            graph=graph,
            symbol_lookup=contrib.current_trie,
            package=contrib.package,
            project_root=project_root,
        )

    def _make_seeded_ctx() -> PluginContext:
        return PluginContext(
            graph=graph,
            symbol_lookup=contrib.current_trie,
            package=contrib.package,
            project_root=project_root,
            package_graph=contrib.package_graph,
        )

    for label, make_ctx in (("filter", _make_filter_ctx), ("seeded", _make_seeded_ctx)):
        make_ctx()  # warm import paths
        t0 = time.perf_counter()
        matched = 0
        for _ in range(REPEATS):
            matched = sum(1 for _ in make_ctx().package_nodes())
        cold_wall = time.perf_counter() - t0
        cold_per_call_us = cold_wall / REPEATS * 1e6
        print(
            f"  {label:6s} cold:  {cold_per_call_us:8.1f} us per call "
            f"({matched}/{n_total} nodes matched)"
        )

        warm_ctx = make_ctx()
        list(warm_ctx.package_nodes())
        iters = 10000
        t0 = time.perf_counter()
        for _ in range(iters):
            for _ in warm_ctx.package_nodes():
                pass
        warm_per_call_us = (time.perf_counter() - t0) / iters * 1e6
        print(f"  {label:6s} warm:  {warm_per_call_us:8.2f} us per call (cached)")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(REPEATS):
        sum(1 for _ in _make_seeded_ctx().package_nodes())
    profiler.disable()

    print("\npackage_nodes seeded cold-call top 15 by tottime:")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
