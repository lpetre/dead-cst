#!/usr/bin/env python
"""Stress-test the parallel plugin executor with N synthetic
"heavy" plugins that simulate the kind of workload an out-of-tree
plugin might have (many find_subclasses / find_decorated_decls
queries). Each plugin's ``run`` body issues a fixed number of
queries; in parallel mode they should overlap on the GIL-releasing
rust paths.

Usage:
    uv run python scripts/bench_heavy_plugins.py --plugins 8 --queries 50
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    require_native,
    stage_dead_cst,
)

from dead_cst import Analysis  # noqa: E402
from dead_cst import _native as native  # noqa: E402
from dead_cst.plugins import Plugin  # noqa: E402


class _HeavyPlugin(Plugin):
    """Synthetic plugin that issues a configurable number of subclass
    queries — each one drops the GIL inside the rust query
    (``find_subclasses_of_class``), so concurrent plugins overlap on
    that work."""

    def __init__(self, queries: int, seeds: list[str]) -> None:
        self.queries = queries
        self.seeds = seeds

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        for _ in range(self.queries):
            for seed in self.seeds:
                ctx.find_subclasses(seed, transitive=True)
        return ()


def _one_run(target_root: Path, plugins: list) -> float:
    start = time.perf_counter()
    Analysis(target_root, plugins=plugins).materialize_all()
    return time.perf_counter() - start


def _measure(target_root: Path, plugins: list, repeats: int, mode: str) -> list[float]:
    if mode == "serial":
        os.environ["DEAD_CST_PLUGINS_SERIAL"] = "1"
    else:
        os.environ.pop("DEAD_CST_PLUGINS_SERIAL", None)
    return [_one_run(target_root, plugins) for _ in range(repeats)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--plugins",
        type=int,
        default=8,
        help="How many synthetic heavy plugins to run.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="How many find_subclasses_of_class calls each plugin makes.",
    )
    args = parser.parse_args()

    require_native()

    target = stage_dead_cst()
    # Pick seeds whose subclasses queries actually do work — these
    # are imported by dead_cst plugins.
    seeds = [
        "dead_cst.plugins._base.Plugin",
        "dead_cst.plugins.decl_shapes.DecoratedDeclPlugin",
        "typer.Typer",
        "click.Command",
        "flask.Flask",
    ]
    plugins = [_HeavyPlugin(args.queries, seeds) for _ in range(args.plugins)]
    print(
        f"\n{args.plugins} plugins x {args.queries} queries x "
        f"{len(seeds)} seeds = {args.plugins * args.queries * len(seeds)} "
        "calls / run"
    )

    print(f"\n{'mode':<10} {'best (ms)':>11} {'median':>10} {'min/max ratio':>14}")
    print("-" * 50)
    for mode in ("serial", "parallel"):
        timings = _measure(target, plugins, args.repeats, mode)
        best = min(timings) * 1000.0
        med = statistics.median(timings) * 1000.0
        ratio = min(timings) / max(timings)
        print(f"{mode:<10} {best:>11.1f} {med:>10.1f} {ratio:>14.2%}")


if __name__ == "__main__":
    main()
