#!/usr/bin/env python
"""Serial vs parallel plugin pass — wall-clock comparison.

Runs the full set of built-in plugins against each target
(``dead_cst`` self, flux0_server, flux0_workspace) in both modes and
prints best-of-N timings. Uses ``DEAD_CST_PLUGINS_SERIAL`` to flip
between the rust-side serial loop and the Python
:class:`concurrent.futures.ThreadPoolExecutor` driven path.

Usage:
    uv run python scripts/bench_parallel_plugins.py --repeats 6
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# Sibling import; make the scripts dir importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    add_common_target_args,
    gather_flux0_targets,
    require_native,
    stage_dead_cst,
    file_count,
    TargetConfig,
)

from dead_cst import Analysis  # noqa: E402
from dead_cst import _native as native  # noqa: E402

# Every built-in plugin name the CLI accepts via ``--plugin``, resolved
# through the same native registry (``_builtin_native_plugin``). Listed
# explicitly so the bench covers the full shipped set.
_BUILTIN_PLUGIN_NAMES = [
    "main_block",
    "module_dunders",
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


def _all_builtin_plugins() -> list:
    """Every built-in plugin, resolved through the native registry
    (``_builtin_native_plugin``) the CLI uses for ``--plugin``. Native
    plugins are stateless after construction, so reusing one instance
    per materialize call is safe."""
    return [native._builtin_native_plugin(name) for name in _BUILTIN_PLUGIN_NAMES]


def _one_run(target: TargetConfig, plugins: list) -> float:
    start = time.perf_counter()
    Analysis(target.root, plugins=plugins).materialize_all()
    return time.perf_counter() - start


def _measure(target: TargetConfig, repeats: int, mode: str) -> list[float]:
    """``repeats`` measurements of ``Analysis(...).materialize_all()``
    with the all-plugin set, in the requested mode."""
    env_before = os.environ.get("DEAD_CST_PLUGINS_SERIAL")
    if mode == "serial":
        os.environ["DEAD_CST_PLUGINS_SERIAL"] = "1"
    else:
        os.environ.pop("DEAD_CST_PLUGINS_SERIAL", None)
    try:
        return [_one_run(target, _all_builtin_plugins()) for _ in range(repeats)]
    finally:
        if env_before is None:
            os.environ.pop("DEAD_CST_PLUGINS_SERIAL", None)
        else:
            os.environ["DEAD_CST_PLUGINS_SERIAL"] = env_before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="How many timed runs per (target, mode) pair (best-of-N).",
    )
    add_common_target_args(parser, default_targets=["dead_cst", "flux0_server", "flux0_workspace"])
    args = parser.parse_args()

    require_native()

    targets: list[TargetConfig] = []
    if "dead_cst" in args.targets:
        targets.append(TargetConfig(name="dead_cst self", root=stage_dead_cst()))
    if not args.no_flux0:
        targets.extend(gather_flux0_targets(args.targets, args.cache_root))

    print(f"\n{'target':<28} {'mode':<8} {'files':>7} {'best (ms)':>11} {'median':>10}")
    print("-" * 70)
    for target in targets:
        files = file_count(target.root)
        for mode in ("serial", "parallel"):
            timings = _measure(target, args.repeats, mode)
            best_ms = min(timings) * 1000.0
            med_ms = statistics.median(timings) * 1000.0
            print(f"{target.name:<28} {mode:<8} {files:>7} {best_ms:>11.1f} {med_ms:>10.1f}")
        print()


if __name__ == "__main__":
    main()
