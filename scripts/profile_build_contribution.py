"""Benchmark + cProfile for ``build_contribution`` and ``package_nodes``.

Run from the repo root:

    uv run python scripts/profile_build_contribution.py
"""

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

from dead_cst._graphstore import SymbolGraph
from dead_cst._package import PackageContribution, build_contribution
from dead_cst._refresh import (
    PackageFiles,
    build_stale_tasks,
    process_stale_files,
)
from dead_cst.branches import DefaultUnreachableRegionDetector
from dead_cst.cache import GraphCache, compute_fingerprint, default_cache_path
from dead_cst.graph import VisitorPayload
from dead_cst.plugins._core import PluginContext
from dead_cst.resolvers._core import Package
from dead_cst.resolvers._exports import exported_roots

REPEATS = 50


def main() -> None:
    project_root = Path(".").resolve()
    # Walk ``dead_cst/`` only; an ``Analysis`` rooted at ``project_root``
    # would rglob ``.venv`` / tests / examples too, but libcst's FQN
    # provider only handles relative imports correctly when the package
    # root is the project root (``from ..resolvers`` in ``dead_cst/contrib/``
    # would escape a ``dead_cst``-rooted package).
    target_dir = project_root / "dead_cst"
    assert target_dir.is_dir(), f"expected {target_dir} to exist"

    detector = DefaultUnreachableRegionDetector()

    package = Package(
        path=project_root,
        name=project_root.name,
        exported=tuple(exported_roots(project_root) or ()),
        deps=(),
    )
    fingerprint = compute_fingerprint(plugins=(), unreachable_detector=detector, package=package)

    files = tuple(sorted(p for p in target_dir.rglob("*.py") if p.is_file()))
    print(f"build_contribution over {len(files)} file(s) under {target_dir.name}/")

    with GraphCache(default_cache_path(project_root)) as cache:
        hits: dict[Path, VisitorPayload] = {}
        miss_files: list[Path] = []
        for f in files:
            payload = cache.get(f, fingerprint)
            if payload is None:
                miss_files.append(f)
            else:
                hits[f] = payload

        pf = PackageFiles(package=package, files=files, hits=hits, miss_files=tuple(miss_files))
        if miss_files:
            tasks = build_stale_tasks({package.path: pf}, project_root, {package.path: fingerprint})
            t0 = time.perf_counter()
            miss_payloads = process_stale_files(
                tasks=tasks,
                detector=detector,
                plugins=(),
                cache=cache,
                workers=None,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"  visitor warmup: {elapsed_ms:.0f} ms over {len(tasks)} miss(es)")
            hits.update(miss_payloads)

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
    """Benchmark :attr:`PluginContext.contribution.nodes` iteration."""
    n_total = len(contrib.nodes)
    print(f"\ncontribution.nodes: contribution has {n_total} node(s)")

    def _make_ctx() -> PluginContext:
        return PluginContext(
            graph=SymbolGraph(),
            symbol_lookup=contrib.trie,
            contribution=contrib,
            project_root=project_root,
        )

    t0 = time.perf_counter()
    matched = 0
    for _ in range(REPEATS):
        matched = sum(1 for _ in _make_ctx().contribution.nodes)
    cold_per_call_us = (time.perf_counter() - t0) / REPEATS * 1e6
    print(f"  cold:  {cold_per_call_us:8.1f} us per call ({matched} nodes)")


if __name__ == "__main__":
    main()
