"""Standalone benchmark for ``DefaultUnreachableRegionDetector.find_regions``.

Times one cold pass of ``find_regions`` over every ``.py`` file under
``dead_cst/``. Run from the repo root:

    uv run python scripts/bench_find_regions.py

Reports total wall time, per-file mean, and a small libcst-internals
breakdown (``ScopeProvider`` / ``ParentNodeProvider`` resolve cost) so
follow-up changes can be compared on equal footing.
"""

from __future__ import annotations

import time
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider, ScopeProvider

from dead_cst.branches import DefaultUnreachableRegionDetector


def main() -> None:
    files = sorted(Path("dead_cst").rglob("*.py"))
    print(f"benchmarking find_regions over {len(files)} files")

    detector = DefaultUnreachableRegionDetector()

    parse_t = scope_t = parent_t = find_t = 0.0
    total_regions = 0
    for f in files:
        src = f.read_text()
        t0 = time.perf_counter()
        module = cst.parse_module(src)
        parse_t += time.perf_counter() - t0

        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)

        t0 = time.perf_counter()
        wrapper.resolve(ScopeProvider)
        scope_t += time.perf_counter() - t0

        t0 = time.perf_counter()
        wrapper.resolve(ParentNodeProvider)
        parent_t += time.perf_counter() - t0

        t0 = time.perf_counter()
        regions = detector.find_regions(wrapper)
        find_t += time.perf_counter() - t0
        total_regions += len(regions)

    n = len(files)
    print(f"  parse:               {parse_t * 1000:7.0f} ms ({parse_t * 1000 / n:5.1f} ms/file)")
    print(f"  ScopeProvider:       {scope_t * 1000:7.0f} ms ({scope_t * 1000 / n:5.1f} ms/file)")
    print(f"  ParentNodeProvider:  {parent_t * 1000:7.0f} ms ({parent_t * 1000 / n:5.1f} ms/file)")
    print(f"  find_regions:        {find_t * 1000:7.0f} ms ({find_t * 1000 / n:5.1f} ms/file)")
    print(f"  total regions found: {total_regions}")


if __name__ == "__main__":
    main()
