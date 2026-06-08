#!/usr/bin/env python
"""Time ``Analysis(corpus).materialize_all()`` on a generated corpus.

Pair with ``scripts/gen_bench_corpus.py``. Reports total wall time, the
per-phase breakdown the rust build already tracks (enumerate / populate /
assemble / fqname index / plugins), node + edge counts, and peak RSS, so
the build is hill-climbable phase by phase.

The per-phase numbers come from ``ctx.read_progress_snapshot()`` (atomic
counters the build stamps as it runs) -- free to read, no allocation.
Exact node/edge counts come from the same snapshot when present and fall
back to ``len(ctx.nodes())`` / ``len(ctx.edges())`` only when ``--count``
is passed (those allocate the full Python lists -- many GB at full scale,
so they default off).

``--incremental`` additionally times a zero-change rescan and a
one-file-change re_materialize on the same ctx. NOTE: the corpus must
live OUTSIDE this repo (see gen_bench_corpus.py) or the rescan re-roots
at the repo and rebuilds the wrong file set; this script resolves the
path to absolute, but it cannot move it out of a nesting project.

Usage::

    uv run python scripts/gen_bench_corpus.py --out /tmp/c --packages 4 \\
        --modules-per-package 8
    uv run python scripts/bench_materialize.py --corpus /tmp/c --count

    # full-scale cold + incremental, with the rust per-phase line
    DEAD_CST_TIMING=1 uv run python scripts/bench_materialize.py \\
        --corpus ~/.cache/dead-cst-bench/corpus --incremental
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import file_count, require_native  # noqa: E402

from dead_cst import Analysis  # noqa: E402
from dead_cst import _native as native  # noqa: E402

# read_progress_snapshot phases, in build order, with the snapshot key
# prefix each one stamps its elapsed-microseconds / item-total under.
_PHASES = [
    ("enumerate", "enum"),
    ("populate", "populate"),
    ("assemble", "assemble"),
    ("fqname idx", "fqname"),
    ("plugins", "plugins"),
]


def _maxrss_bytes() -> int:
    """Peak resident set size of this process. ``ru_maxrss`` is bytes on
    macOS and kilobytes on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _all_builtin_plugins() -> list:
    # Mirror the set bench_parallel_plugins.py uses; resolved through the
    # native registry the CLI drives.
    names = [
        "main_block",
        "init_subclass",
        "server_config",
        "unittest",
        "flask",
        "fastapi",
        "typer",
        "cyclopts",
        "slack_bolt",
        "fastmcp",
        "celery",
        "click",
        "mock_patch",
        "discordpy",
        "pytest",
        "project_scripts",
        "dynamic_import_fallback",
    ]
    return [native._builtin_native_plugin(n) for n in names]


def _cold_build(corpus: Path, plugins: list) -> tuple[float, Analysis, native.ProjectContext]:
    analysis = Analysis(corpus, plugins=plugins)
    start = time.perf_counter()
    ctx = analysis.materialize_all()
    wall = time.perf_counter() - start
    return wall, analysis, ctx


def _incremental(
    analysis: Analysis, ctx: native.ProjectContext, corpus: Path
) -> tuple[float, float, int, int]:
    """Time two re_materialize passes on the already-built ctx and prove
    the edit registered.

    Zero-change: an explicit rescan with nothing touched on disk -- a
    pure salsa cache hit for per-file work (the assemble + fqname passes
    still rebuild the whole graph from cached per-file outputs).

    One-file change: append a top-level decl to one module, then
    ``detect_changes()`` + re_materialize. NOTE: a targeted
    ``ChangeEvent.changed(path)`` does *not* invalidate the file in this
    ty integration (verified -- the node count doesn't move); only the
    rescan path (which ``detect_changes`` returns) picks edits up, so
    salsa re-parses just the changed file under a full-project re-stat.

    Returns ``(zero_s, one_s, nodes_before, nodes_after)``; the node
    delta proves the edit took. Restores the file afterwards."""

    def _nodes() -> int:
        return ctx.read_progress_snapshot()["fqname_total"]

    # Zero-change: force a rebuild with nothing actually modified.
    t = time.perf_counter()
    analysis.re_materialize([native.ChangeEvent.rescan()])
    zero = time.perf_counter() - t
    nodes_before = _nodes()

    # One-file change: append a fresh top-level decl, then autodetect.
    target = (corpus / "pkg_0000" / "mod_0000.py").resolve()
    original = target.read_text()
    try:
        target.write_text(original + "\n\ndef _bench_touch():\n    pass\n")
        t = time.perf_counter()
        analysis.re_materialize(ctx.detect_changes())
        one = time.perf_counter() - t
        nodes_after = _nodes()
    finally:
        target.write_text(original)
    return zero, one, nodes_before, nodes_after


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Generated corpus dir.")
    parser.add_argument("--repeats", type=int, default=1, help="Timed runs; reports best.")
    parser.add_argument(
        "--plugins",
        choices=("none", "all"),
        default="none",
        help="Run with no plugins (pure build) or the full built-in set.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Also materialize ctx.nodes()/edges() for an exact count "
        "(allocates the full lists -- many GB at full scale).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="After the cold build, time a zero-change rescan and a "
        "one-file-change re_materialize on the same ctx.",
    )
    args = parser.parse_args()

    require_native()
    if not args.corpus.is_dir():
        sys.exit(f"ERROR: corpus dir not found: {args.corpus}")
    # Resolve to absolute: a relative root makes ty's rescan re-walk
    # rediscover zero files, so an incremental re_materialize silently
    # rebuilds to an empty graph. Absolute roots are unaffected.
    args.corpus = args.corpus.resolve()

    files = file_count(args.corpus)
    plugins = _all_builtin_plugins() if args.plugins == "all" else []

    walls: list[float] = []
    analysis: Analysis | None = None
    ctx: native.ProjectContext | None = None
    for i in range(args.repeats):
        wall, analysis, ctx = _cold_build(args.corpus, plugins)
        walls.append(wall)
        print(f"  cold run {i + 1}/{args.repeats}: {wall:.2f}s", flush=True)

    assert ctx is not None
    # Capture the cold-build counters before any re_materialize resets them.
    snap = ctx.read_progress_snapshot()
    best = min(walls)

    zero_wall = one_wall = None
    nodes_before = nodes_after = 0
    if args.incremental:
        assert analysis is not None
        zero_wall, one_wall, nodes_before, nodes_after = _incremental(analysis, ctx, args.corpus)
        print(f"  zero-change re_materialize:     {zero_wall:.2f}s", flush=True)
        print(
            f"  one-file-change re_materialize: {one_wall:.2f}s "
            f"(nodes {nodes_before:,} -> {nodes_after:,})",
            flush=True,
        )

    peak = _maxrss_bytes()

    # Counts: the fqname phase walks every node, so fqname_total is the
    # node count (free). No phase counts edges, so an exact edge total
    # needs --count (which allocates the full edge list).
    nodes = snap.get("fqname_total") or 0
    edges = 0
    if args.count:
        nodes = len(ctx.nodes())
        edges = len(ctx.edges())

    print(f"\n  corpus      {args.corpus}")
    print(f"  files       {files:,}")
    print(f"  nodes       {nodes:,}{'' if args.count else '  (fqname_total; --count for exact)'}")
    if args.count:
        print(f"  edges       {edges:,}")
        print(f"  nodes/file  {nodes / files:.1f}")
        print(f"  edges/file  {edges / files:.1f}")
    print(f"  plugins     {args.plugins}")
    print(f"  peak RSS    {peak / 1e9:.2f} GB")
    print(f"  best wall   {best:.2f}s  (median {statistics.median(walls):.2f}s of {args.repeats})")
    if files and best:
        print(f"  throughput  {files / best:,.0f} files/s")
    if zero_wall is not None:
        print(f"  re_mat 0chg {zero_wall:.2f}s  ({best / zero_wall:.1f}x faster than cold)")
        delta = nodes_after - nodes_before
        print(
            f"  re_mat 1chg {one_wall:.2f}s  ({best / one_wall:.1f}x faster than cold; "
            f"+{delta} node{'s' if delta != 1 else ''} from the edit)"
        )

    print("\n  per-phase (best-effort, from last run's snapshot):")
    print(f"    {'phase':<12}{'elapsed':>10}{'items':>14}")
    for label, key in _PHASES:
        us = snap.get(f"{key}_elapsed_us", 0)
        total = snap.get(f"{key}_total", 0)
        print(f"    {label:<12}{us / 1e6:>9.2f}s{total:>14,}")


if __name__ == "__main__":
    main()
