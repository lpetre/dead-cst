"""Cold-path perf breakdown by plugin on the rust backend.

For every built-in native plugin, measure
``ProjectContext.materialize()`` wall-clock with that plugin enabled
versus the no-plugins baseline. Reports the delta so the per-plugin
cost is comparable across targets.

Cold = fresh ``ProjectContext`` per iteration so ty's Salsa db starts
empty. The first iter prints separately as a "warm-up" line because
the process-global typeshed / module-resolver caches drop run 1 from
~3-4x slower to steady state. Steady-state numbers are reported as
best-of-(repeats - 1).

Usage:
    uv run python scripts/profile_plugins_cold.py
    uv run python scripts/profile_plugins_cold.py --repeats 5
    uv run python scripts/profile_plugins_cold.py --targets flux0_workspace
    uv run python scripts/profile_plugins_cold.py --plugins click pytest fastapi

Every built-in plugin is now native (Rust), resolved by its CLI name
through ``dead_cst._native._builtin_native_plugin``. The ``explicit``
plugin is intentionally absent — it's driven by ``-e`` /
``--entrypoint-regex``, not ``--plugin``. The rust extension must be
importable; if it's not the script exits with a clear error since the
libcst path is irrelevant to a cold-rust profile.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time

from _bench_common import (
    TargetConfig,
    add_common_target_args,
    file_count,
    gather_flux0_targets,
    require_native,
    stage_dead_cst,
)

# Silence the visitor's WARNING-level breadcrumbs (e.g.
# ``importlib.import_module(<not-a-literal>)``) — useful in normal
# runs, pure noise when we want a clean timing table.
logging.getLogger("dead_cst").setLevel(logging.ERROR)

require_native()
from dead_cst import _native as native  # noqa: E402  (require_native gates this)


# ---------------------------------------------------------------------------
# Plugin set
# ---------------------------------------------------------------------------

# Every built-in plugin is now native. These are the CLI ``--plugin``
# names resolved through ``native._builtin_native_plugin``. ``explicit``
# is intentionally absent (driven by ``-e`` / ``--entrypoint-regex``).
_BUILTIN_NATIVE_PLUGIN_NAMES = (
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
)


def _native_builtin_plugins(filter_names: set[str] | None) -> list[tuple[str, object]]:
    """Return ``[(name, native_plugin)]`` for every built-in native
    plugin, optionally filtered by CLI registry name."""
    out: list[tuple[str, object]] = []
    for name in _BUILTIN_NATIVE_PLUGIN_NAMES:
        if filter_names is not None and name not in filter_names:
            continue
        plugin = native._builtin_native_plugin(name)
        if plugin is None:
            continue
        out.append((name, plugin))
    return out


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def _materialize_once(target: TargetConfig, plugin: object | None) -> float:
    """One cold iteration: fresh ProjectContext, optionally one plugin,
    materialize(), return wall-clock seconds.

    Reuses the default-configured native plugin instance resolved from
    ``native._builtin_native_plugin`` — each is ready to add to a
    context and pays its real per-plugin dispatch cost, which is what we
    want to measure."""
    ctx = native.ProjectContext(str(target.root), **target.project_kwargs)
    if plugin is not None:
        ctx.add_plugin(plugin)
    t0 = time.perf_counter()
    ctx.materialize()
    return time.perf_counter() - t0


def _bench_configuration(
    target: TargetConfig, plugin: object | None, repeats: int
) -> tuple[float, list[float]]:
    """Run ``repeats + 1`` cold iterations. Returns ``(warmup_iter1,
    steady_iters)``. The first iter is segregated because it includes
    process-global typeshed loading."""
    iter1 = _materialize_once(target, plugin)
    steady = [_materialize_once(target, plugin) for _ in range(repeats)]
    return iter1, steady


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:7.1f}"


def _fmt_delta(seconds: float) -> str:
    ms = seconds * 1000
    sign = "+" if ms >= 0 else ""
    return f"{sign}{ms:5.1f}"


def _run_target(target: TargetConfig, plugins: list[tuple[str, object]], repeats: int) -> None:
    files = file_count(target.root)
    print(f"\n=== {target.name} === ({files} files)")
    print(f"  path: {target.root}")

    # Baseline first.
    print("  measuring baseline (no plugins)…", flush=True)
    base_iter1, base_steady = _bench_configuration(target, None, repeats)
    base_best = min(base_steady)
    base_mean = statistics.fmean(base_steady)

    rows: list[tuple[str, float, float, float]] = []
    for name, plugin in plugins:
        print(f"  measuring {name}…", flush=True)
        _, steady = _bench_configuration(target, plugin, repeats)
        best = min(steady)
        mean = statistics.fmean(steady)
        rows.append((name, best, mean, best - base_best))

    rows.sort(key=lambda r: r[3], reverse=True)  # most expensive first

    print()
    print(f"  {'plugin':<22} {'best (ms)':>10}  {'mean (ms)':>10}  {'Δ vs base (ms)':>16}")
    print(f"  {'-' * 22} {'-' * 10}  {'-' * 10}  {'-' * 16}")
    print(
        f"  {'(no plugins) ⇣':<22} {_fmt_ms(base_best):>10}  {_fmt_ms(base_mean):>10}  {'  —':>16}"
    )
    for name, best, mean, delta in rows:
        print(f"  {name:<22} {_fmt_ms(best):>10}  {_fmt_ms(mean):>10}  {_fmt_delta(delta):>16}")
    print()
    print(
        f"  warm-up iter 1 (no plugins): {_fmt_ms(base_iter1)} ms — typeshed / module-resolver init"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Steady-state iterations per (target, plugin) combo (default 3). "
        "One extra warm-up iteration is always run and reported separately.",
    )
    add_common_target_args(parser, default_targets=["dead_cst", "flux0_server"])
    parser.add_argument(
        "--plugins",
        nargs="+",
        default=None,
        help="Restrict to a subset of plugins by name (default: all rust-capable builtins)",
    )
    args = parser.parse_args()

    filter_names = set(args.plugins) if args.plugins else None
    plugins = _native_builtin_plugins(filter_names)

    print(f"python: {sys.version.split()[0]}")
    print(f"repeats: {args.repeats}")
    print(f"plugins measured: {len(plugins)}")

    targets: list[TargetConfig] = []
    if "dead_cst" in args.targets:
        targets.append(TargetConfig(name="dead_cst (self)", root=stage_dead_cst()))
    if not args.no_flux0:
        targets.extend(gather_flux0_targets(args.targets, args.cache_root))

    if not targets:
        sys.exit("no targets to profile")

    for target in targets:
        _run_target(target, plugins, args.repeats)


if __name__ == "__main__":
    main()
