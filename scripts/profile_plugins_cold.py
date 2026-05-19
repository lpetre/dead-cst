"""Cold-path perf breakdown by plugin on the rust backend.

For every builtin plugin that supports the rust ``ProjectPlugin``
protocol (i.e. exposes ``run(ctx)``), measure
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

Plugins without a ``run(ctx)`` method (``UnittestPlugin``,
``MockPatchPlugin``, ``DiscordPyPlugin`` today) are listed as
``SKIP``. The rust extension must be importable; if it's not the
script exits with a clear error since the libcst path is irrelevant
to a cold-rust profile.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from dead_cst.plugins import BUILTIN_PLUGINS

# Silence the visitor's WARNING-level breadcrumbs (e.g.
# ``importlib.import_module(<not-a-literal>)``) — useful in normal
# runs, pure noise when we want a clean timing table.
logging.getLogger("dead_cst").setLevel(logging.ERROR)

try:
    from dead_cst import _native as native
except ImportError as exc:
    sys.exit(
        f"ERROR: dead_cst._native not importable ({exc}). "
        "Build with: uv run maturin develop --release "
        "--manifest-path Cargo.toml"
    )

FLUX0_URL = "https://github.com/flux0-ai/flux0.git"
FLUX0_SHA = "8d04176642b091ddb5c5020486f353d4e824460b"


# ---------------------------------------------------------------------------
# Target staging
# ---------------------------------------------------------------------------


def _stage_dead_cst() -> Path:
    """Copy ``dead_cst/`` into a fresh tempdir as a clean project root.

    Mirrors ``profile_backends.py``: a flat copy of the package gives
    the analyzer a project root that contains exactly one package.
    """
    stage = Path(tempfile.mkdtemp(prefix="dead-cst-plugin-bench-"))
    src = Path(__file__).resolve().parent.parent / "dead_cst"
    shutil.copytree(src, stage / "dead_cst", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return stage


def _clone_flux0(cache_root: Path) -> Path:
    """Shallow-clone flux0 at the pinned SHA. Same protocol as
    ``profile_backends.py``."""
    if shutil.which("git") is None:
        raise RuntimeError("git not on PATH")
    dest = cache_root / "flux0"
    marker = dest / ".sha"
    if marker.is_file() and marker.read_text().strip() == FLUX0_SHA:
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def _git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True)

    _git("init", "-q")
    _git("remote", "add", "origin", FLUX0_URL)
    _git("fetch", "--depth=1", "-q", "--filter=blob:none", "origin", FLUX0_SHA)
    _git("checkout", "-q", FLUX0_SHA)
    marker.write_text(FLUX0_SHA + "\n")
    return dest


def _ensure_flux0_venv(flux0_root: Path) -> Path:
    """Run ``uv sync --all-packages`` so :class:`UvResolver` and ty's
    workspace module resolver find a ``.venv``. Idempotent."""
    venv = flux0_root / ".venv"
    if venv.is_dir():
        return venv
    if shutil.which("uv") is None:
        raise RuntimeError("uv not on PATH; cannot create flux0 workspace venv")
    print(f"  (one-time setup: uv sync --all-packages in {flux0_root})")
    subprocess.run(["uv", "sync", "--all-packages"], cwd=flux0_root, check=True)
    return venv


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass
class TargetConfig:
    """Cold-rust target. ``project_kwargs`` are forwarded to
    ``native.ProjectContext`` — typically ``python_env`` for workspace
    targets so ty's module resolver picks up third-party packages from
    the workspace's own ``.venv``."""

    name: str
    root: Path
    project_kwargs: dict[str, object] = field(default_factory=dict)


def _file_count(target_root: Path) -> int:
    import os

    count = 0
    for dirpath, dirnames, filenames in os.walk(target_root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".venv")]
        count += sum(1 for f in filenames if f.endswith(".py"))
    return count


# ---------------------------------------------------------------------------
# Plugin set
# ---------------------------------------------------------------------------


def _plugin_label(plugin: object) -> str:
    """Human-readable label for one builtin plugin instance.

    Class name alone is ambiguous for the shared
    :class:`DispatchAppPlugin` / :class:`DecoratedDeclPlugin` shapes —
    multiple frameworks reuse the same class with different field
    values. Suffix with the most identifying field so the report
    distinguishes ``DispatchAppPlugin(fastapi.FastAPI)`` from
    ``DispatchAppPlugin(flask.Flask)`` etc.
    """
    base = type(plugin).__qualname__
    app_classes = getattr(plugin, "app_classes", None)
    if app_classes:
        return f"{base}({app_classes[0]})"
    decorator_module = getattr(plugin, "decorator_module", None)
    if decorator_module:
        return f"{base}({decorator_module})"
    return base


def _rust_capable_plugins(filter_names: set[str] | None) -> list[tuple[str, object]]:
    """Return ``[(label, plugin_instance)]`` for every builtin plugin
    with a ``run(ctx)`` method. Optionally filtered by class qualname."""
    out: list[tuple[str, object]] = []
    for plugin in BUILTIN_PLUGINS:
        label = _plugin_label(plugin)
        if filter_names is not None and label not in filter_names:
            continue
        if not hasattr(plugin, "run"):
            continue
        out.append((label, plugin))
    return out


def _skipped_plugins(filter_names: set[str] | None) -> list[str]:
    """Builtin plugins explicitly missing rust support."""
    out: list[str] = []
    for plugin in BUILTIN_PLUGINS:
        label = _plugin_label(plugin)
        if filter_names is not None and label not in filter_names:
            continue
        if not hasattr(plugin, "run"):
            out.append(label)
    return out


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def _materialize_once(target: TargetConfig, plugin: object | None) -> float:
    """One cold iteration: fresh ProjectContext, optionally one plugin,
    materialize(), return wall-clock seconds.

    Reuses the configured ``BUILTIN_PLUGINS`` instance — each rust-capable
    builtin is already a ready-to-use, default-configured plugin
    (``ExplicitEntrypointPlugin`` has ``specs=[]``, which makes it a
    no-op but still pays the ``run()`` dispatch cost — that's what we
    want to measure)."""
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
    files = _file_count(target.root)
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


TARGET_NAMES = ("dead_cst", "flux0_server", "flux0_cli", "flux0_workspace")


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
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["dead_cst", "flux0_server"],
        choices=TARGET_NAMES,
        help="Which targets to profile (default: dead_cst flux0_server)",
    )
    parser.add_argument(
        "--plugins",
        nargs="+",
        default=None,
        help="Restrict to a subset of plugins by name (default: all rust-capable builtins)",
    )
    parser.add_argument(
        "--no-flux0",
        action="store_true",
        help="Skip flux0 targets even if requested (offline / no git)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "dead-cst-bench",
        help="Where to cache the flux0 clone (default ~/.cache/dead-cst-bench)",
    )
    args = parser.parse_args()

    filter_names = set(args.plugins) if args.plugins else None
    plugins = _rust_capable_plugins(filter_names)
    skipped = _skipped_plugins(filter_names)

    print(f"python: {sys.version.split()[0]}")
    print(f"repeats: {args.repeats}")
    print(f"plugins measured: {len(plugins)}")
    if skipped:
        print(f"plugins SKIP (no rust run()): {', '.join(skipped)}")

    targets: list[TargetConfig] = []
    if "dead_cst" in args.targets:
        targets.append(TargetConfig(name="dead_cst (self)", root=_stage_dead_cst()))

    flux0_requested = [t for t in args.targets if t.startswith("flux0_")]
    if flux0_requested and not args.no_flux0:
        args.cache_root.mkdir(parents=True, exist_ok=True)
        try:
            flux0_root = _clone_flux0(args.cache_root)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"\nflux0 SKIP: {exc}")
            flux0_root = None
        if flux0_root is not None:
            if "flux0_server" in flux0_requested:
                targets.append(
                    TargetConfig(
                        name="flux0 server",
                        root=flux0_root / "packages" / "server" / "src",
                    )
                )
            if "flux0_cli" in flux0_requested:
                targets.append(
                    TargetConfig(name="flux0 cli", root=flux0_root / "packages" / "cli" / "src")
                )
            if "flux0_workspace" in flux0_requested:
                try:
                    venv = _ensure_flux0_venv(flux0_root)
                except (RuntimeError, subprocess.CalledProcessError) as exc:
                    print(f"\nflux0_workspace SKIP: {exc}")
                else:
                    targets.append(
                        TargetConfig(
                            name="flux0 workspace (uv)",
                            root=flux0_root,
                            project_kwargs={"python_env": str(venv)},
                        )
                    )

    if not targets:
        sys.exit("no targets to profile")

    for target in targets:
        _run_target(target, plugins, args.repeats)


if __name__ == "__main__":
    main()
