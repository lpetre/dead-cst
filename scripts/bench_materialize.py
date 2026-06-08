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

``--incremental`` additionally runs an LSP-style edit loop on the same
ctx: pay the cold build once, then a series of small one-file edits,
each applied via an explicit ``ChangeEvent.changed(path)`` and timed
individually. Explicit content events are the incremental fast path —
the resolve cache reuses every cross-file resolution whose read set
avoids the edit, and only the touched files re-mint/re-translate.
(``rescan`` / ``detect_changes`` deliberately take the full-resolve
path: a rescan can't bound the blast radius.) The loop edits scattered
files first, then hammers one file repeatedly (the hot-loop shape), and
finishes with a no-op event (the early-return path). Per-round walls,
``(resolved, reused)`` counters, and the tombstone count are reported.
NOTE: the corpus must live OUTSIDE this repo (see gen_bench_corpus.py)
or rescan-style rebuilds re-root at the repo; this script resolves the
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
    analysis: Analysis, ctx: native.ProjectContext, corpus: Path, edits: int
) -> list[tuple[str, float, tuple[int, int], int]]:
    """LSP-style edit loop on the already-built ctx.

    Each round appends a fresh top-level decl to one module and
    re-materializes via an explicit ``ChangeEvent.changed(path)`` — the
    incremental fast path (the resolve cache reuses every entry whose
    read set avoids the edit; only the touched files re-mint). The first
    ``edits`` rounds scatter across packages (cold per-file state each
    time), the next ``edits`` rounds hammer one file (the hot loop), and
    a final round applies a no-op event (early return). Every round
    asserts the node count moved (or, for the no-op, didn't).

    Returns ``[(label, wall_s, (resolved, reused), tombstones), ...]``.
    Edited files are restored afterwards."""

    def _nodes() -> int:
        # Live node count: the raw total includes tombstoned slots (a
        # re-minted file's old block stays in place so live dense ids
        # never remap), so subtract them — an edit that adds one decl
        # grows the *live* count by exactly one.
        return ctx.read_progress_snapshot()["fqname_total"] - len(ctx.tombstoned_indices())

    manifest = sorted((corpus).glob("pkg_*"))
    rows: list[tuple[str, float, tuple[int, int], int]] = []
    originals: dict[Path, str] = {}
    try:
        # Scattered edits: one file per round, spread across packages.
        for i in range(edits):
            pkg = manifest[(i * 97) % len(manifest)]
            target = sorted(pkg.glob("mod_*.py"))[i % 3]
            originals.setdefault(target, target.read_text())
            target.write_text(target.read_text() + f"\n\ndef _bench_s{i}():\n    pass\n")
            before = _nodes()
            t = time.perf_counter()
            analysis.re_materialize([native.ChangeEvent.changed(str(target))])
            wall = time.perf_counter() - t
            assert _nodes() == before + 1, "edit did not register"
            rows.append(
                (
                    f"scatter {target.relative_to(corpus)}",
                    wall,
                    ctx._last_resolve_counts(),
                    len(ctx.tombstoned_indices()),
                )
            )

        # Hot loop: repeated edits to one file.
        target = (corpus / "pkg_0000" / "mod_0000.py").resolve()
        originals.setdefault(target, target.read_text())
        for i in range(edits):
            target.write_text(target.read_text() + f"\n\ndef _bench_h{i}():\n    pass\n")
            before = _nodes()
            t = time.perf_counter()
            analysis.re_materialize([native.ChangeEvent.changed(str(target))])
            wall = time.perf_counter() - t
            assert _nodes() == before + 1, "edit did not register"
            rows.append(
                (
                    f"hot     {target.relative_to(corpus)}",
                    wall,
                    ctx._last_resolve_counts(),
                    len(ctx.tombstoned_indices()),
                )
            )

        # No-op: event on an untouched file -> early return.
        before = _nodes()
        t = time.perf_counter()
        analysis.re_materialize([native.ChangeEvent.changed(str(target))])
        wall = time.perf_counter() - t
        assert _nodes() == before, "no-op event rebuilt something"
        rows.append(("no-op  (early return)", wall, (0, 0), len(ctx.tombstoned_indices())))
    finally:
        for path, text in originals.items():
            path.write_text(text)
    return rows


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
        help="After the cold build, run an LSP-style edit loop: a series "
        "of small one-file edits applied via explicit Changed events, "
        "each timed individually.",
    )
    parser.add_argument(
        "--edits",
        type=int,
        default=5,
        help="Edit rounds per --incremental shape (scattered + hot loop).",
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

    edit_rows: list[tuple[str, float, tuple[int, int], int]] = []
    if args.incremental:
        assert analysis is not None
        edit_rows = _incremental(analysis, ctx, args.corpus, args.edits)
        for label, wall, (resolved, reused), tombs in edit_rows:
            print(
                f"  edit {label:<38} {wall:7.3f}s  "
                f"resolved={resolved:,} reused={reused:,} tombstones={tombs:,}",
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
    if edit_rows:
        # The no-op row is the early-return path; summarize the edits.
        edit_walls = [w for label, w, _, _ in edit_rows if not label.startswith("no-op")]
        med = statistics.median(edit_walls)
        print(
            f"  edit loop   median {med:.3f}s  best {min(edit_walls):.3f}s  "
            f"worst {max(edit_walls):.3f}s  over {len(edit_walls)} edits  "
            f"({best / med:.0f}x faster than cold)"
        )

    print("\n  per-phase (best-effort, from last run's snapshot):")
    print(f"    {'phase':<12}{'elapsed':>10}{'items':>14}")
    for label, key in _PHASES:
        us = snap.get(f"{key}_elapsed_us", 0)
        total = snap.get(f"{key}_total", 0)
        print(f"    {label:<12}{us / 1e6:>9.2f}s{total:>14,}")


if __name__ == "__main__":
    main()
