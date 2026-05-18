"""Profile the full analysis pass across backends and cache states.

Two targets:

* ``dead_cst`` self — the package itself, staged via a symlink so both
  backends see a project root that contains exactly one package.
* ``flux0_server`` — cloned from ``flux0-ai/flux0`` at the same SHA the
  e2e test pins, ``packages/server/src`` as project root.

Four configurations per target:

* **libcst cold** — ``Analysis(..., cache=None).materialize_all()``;
  visitor + edge stitching from scratch.
* **libcst warm** — same call against a pre-populated ``GraphCache``;
  warm cache hits skip the visitor (the libcst pipeline's per-file
  payload cache).
* **rust cold** — fresh ``native.Project`` per iteration so ty's
  Salsa db starts empty; ``Project.build()`` + the bridge that
  materializes the envelope into a ``SymbolGraph``.
* **rust warm** — same ``Project`` instance reused across iterations;
  ty's Salsa cache returns memoized parses.

Each configuration is timed ``--repeats`` times (default 3) and the
minimum is reported (best-effort suppresses noise from background
load).

Usage:
    uv run python scripts/profile_backends.py
    uv run python scripts/profile_backends.py --repeats 5 --targets dead_cst
    uv run python scripts/profile_backends.py --no-flux0   # skip the clone

Rust path requires ``dead_cst._native``; if not importable the rust
columns print ``SKIP`` and the script continues with libcst only.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dead_cst import Analysis
from dead_cst.cache import GraphCache
from dead_cst.resolvers import ManualResolver, UvResolver
from dead_cst.resolvers._core import PathResolver

# The visitor logs WARNING-level breadcrumbs for things like
# ``importlib.import_module(<not-a-literal>)``. Useful in normal runs,
# pure noise when we want a clean timing table.
logging.getLogger("dead_cst").setLevel(logging.ERROR)

# Rust path is optional. The native extension may not be installed in
# the active venv (CI lint job doesn't build it); fall back to libcst-only.
try:
    from dead_cst import _native as native
    from libcst.metadata import CodePosition, CodeRange

    from dead_cst._graphstore import SymbolGraph
    from dead_cst.graph import EdgeFlags, Import as DCImport, NodeFlags, SymbolNode

    HAS_RUST = True
except ImportError as e:
    HAS_RUST = False
    _RUST_IMPORT_ERROR = str(e)


FLUX0_URL = "https://github.com/flux0-ai/flux0.git"
FLUX0_SHA = "8d04176642b091ddb5c5020486f353d4e824460b"


# ---------------------------------------------------------------------------
# Target staging
# ---------------------------------------------------------------------------


def _stage_dead_cst() -> Path:
    """Copy ``dead_cst/`` into a fresh tempdir as a clean project root.

    Pointing the analyzer at the package dir directly doesn't work for
    ``ManualResolver(specs=["."])`` (it expects subdirs to be packages,
    not the project root itself), and pointing it at the repo root would
    sweep vendor/ruff + tests/ for the rust path. A flat copy gives both
    backends a project root that contains exactly one package — same
    shape flux0's ``packages/server/src/`` has. Symlinks don't work:
    neither libcst's file walker nor ty's project enumeration follows
    them, so the staged tree has to be a real directory.
    """
    stage = Path(tempfile.mkdtemp(prefix="dead-cst-bench-"))
    src = Path(__file__).resolve().parent.parent / "dead_cst"
    shutil.copytree(src, stage / "dead_cst", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return stage


def _clone_flux0(cache_root: Path) -> Path:
    """Shallow-clone flux0 at the pinned SHA. Mirror ``tests/e2e/conftest.py``."""
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
    """Run ``uv sync --all-packages`` in the clone so :class:`UvResolver`
    finds a ``.venv``. Skipped if one already exists at the workspace root.
    Returns the path to the venv."""
    venv = flux0_root / ".venv"
    if venv.is_dir():
        return venv
    if shutil.which("uv") is None:
        raise RuntimeError("uv not on PATH; cannot create flux0 workspace venv")
    print(f"  (one-time setup: uv sync --all-packages in {flux0_root})")
    subprocess.run(
        ["uv", "sync", "--all-packages"],
        cwd=flux0_root,
        check=True,
    )
    return venv


# ---------------------------------------------------------------------------
# Bridge (inlined so the script doesn't depend on the test package layout)
# ---------------------------------------------------------------------------


if HAS_RUST:
    _NO_NODE_FLAGS = NodeFlags(0)
    _NO_EDGE_FLAGS = EdgeFlags(0)

    def _bridge_materialize(graph: "native.NativeGraph") -> "SymbolGraph":
        """Same as ``tests/prototype/_bridge.materialize``, inlined here."""
        out = SymbolGraph()
        path_cache: dict[str, Path] = {}
        symbol_nodes: list[SymbolNode] = []
        for n in graph.nodes:
            path = path_cache.get(n.path)
            if path is None:
                path = Path(n.path)
                path_cache[n.path] = path
            flags = _NO_NODE_FLAGS if n.flags == 0 else NodeFlags(n.flags)
            symbol_nodes.append(
                SymbolNode(
                    fqname=n.fqname,
                    type=n.kind,  # type: ignore[arg-type]
                    path=path,
                    position=CodeRange(
                        CodePosition(n.start_line, n.start_column),
                        CodePosition(n.end_line, n.end_column),
                    ),
                    imports=(
                        DCImport(module=n.imports.module, decl=n.imports.decl, star=n.imports.star)
                        if n.imports is not None
                        else None
                    ),
                    flags=flags,
                )
            )
        for sn in symbol_nodes:
            out.add(sn)
        for src, dst, flags in graph.edges:
            edge_flags = _NO_EDGE_FLAGS if flags == 0 else EdgeFlags(flags)
            out.add_edge(symbol_nodes[src], symbol_nodes[dst], edge_flags)
        return out


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------


def _time(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


@dataclass
class TargetConfig:
    """Per-target settings for the bench drivers.

    ``make_resolver`` is a factory so every libcst run gets a fresh
    resolver instance (some resolvers carry per-analysis state like
    ``UvResolver._site_packages`` that gets primed during ``resolve``).

    ``rust_kwargs`` are forwarded to ``native.Project`` — typically
    ``python_env`` for workspace targets so ty resolves third-party
    imports through the workspace's own ``.venv`` rather than the
    active interpreter's.
    """

    name: str
    root: Path
    make_resolver: Callable[[], PathResolver]
    rust_kwargs: dict[str, object] = field(default_factory=dict)


def _bench_libcst_cold(cfg: TargetConfig, repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        analysis = Analysis(cfg.root, resolver=cfg.make_resolver(), cache=None)
        timings.append(_time(analysis.materialize_all))
    return timings


def _bench_libcst_warm(cfg: TargetConfig, repeats: int) -> list[float]:
    with tempfile.TemporaryDirectory(prefix="dead-cst-bench-cache-") as cd:
        cache_path = Path(cd) / "cache.sqlite"
        with GraphCache(cache_path) as cache:
            # Warm-up: populate the cache with one full run.
            Analysis(cfg.root, resolver=cfg.make_resolver(), cache=cache).materialize_all()
            timings: list[float] = []
            for _ in range(repeats):
                analysis = Analysis(cfg.root, resolver=cfg.make_resolver(), cache=cache)
                timings.append(_time(analysis.materialize_all))
            return timings


def _new_native_project(cfg: TargetConfig) -> "native.Project":
    return native.Project(str(cfg.root), **cfg.rust_kwargs)


def _bench_rust_cold(cfg: TargetConfig, repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        # Fresh Project per iter → fresh ty Salsa db → cold.
        proj = _new_native_project(cfg)

        def _run() -> None:
            g = proj.build()
            _bridge_materialize(g)

        timings.append(_time(_run))
    return timings


def _bench_rust_warm(cfg: TargetConfig, repeats: int) -> list[float]:
    proj = _new_native_project(cfg)
    # Warm-up: prime Salsa.
    g = proj.build()
    _bridge_materialize(g)
    timings: list[float] = []
    for _ in range(repeats):

        def _run() -> None:
            g = proj.build()
            _bridge_materialize(g)

        timings.append(_time(_run))
    return timings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _file_count(target_root: Path) -> int:
    """Count .py files under target_root, following symlinks so the
    staged dead_cst layout reports the real count."""
    import os

    count = 0
    for dirpath, dirnames, filenames in os.walk(target_root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".venv")]
        count += sum(1 for f in filenames if f.endswith(".py"))
    return count


def _fmt_ms(timings: list[float]) -> str:
    best = min(timings) * 1000
    worst = max(timings) * 1000
    return f"{best:7.0f} ms   (best of {len(timings)}, worst {worst:.0f} ms)"


def _run_target(cfg: TargetConfig, repeats: int) -> None:
    files = _file_count(cfg.root)
    print(f"\n=== {cfg.name} ===")
    print(f"  path:  {cfg.root}")
    print(f"  files: {files}")

    print(f"  libcst cold : {_fmt_ms(_bench_libcst_cold(cfg, repeats))}")
    print(f"  libcst warm : {_fmt_ms(_bench_libcst_warm(cfg, repeats))}")
    if HAS_RUST:
        print(f"  rust   cold : {_fmt_ms(_bench_rust_cold(cfg, repeats))}")
        print(f"  rust   warm : {_fmt_ms(_bench_rust_warm(cfg, repeats))}")
    else:
        print(f"  rust   cold : SKIP (dead_cst._native: {_RUST_IMPORT_ERROR})")
        print("  rust   warm : SKIP")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["dead_cst", "flux0_server", "flux0_workspace"],
        choices=["dead_cst", "flux0_server", "flux0_cli", "flux0_workspace"],
        help="Which targets to profile (default: dead_cst flux0_server flux0_workspace)",
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

    print(f"python: {sys.version.split()[0]}")
    print(f"rust extension: {'OK' if HAS_RUST else 'NOT INSTALLED'}")
    print(f"repeats: {args.repeats}")

    if "dead_cst" in args.targets:
        _run_target(
            TargetConfig(
                name="dead_cst (self)",
                root=_stage_dead_cst(),
                make_resolver=lambda: ManualResolver(specs=["."]),
            ),
            args.repeats,
        )

    flux0_targets = [t for t in args.targets if t.startswith("flux0_")]
    if flux0_targets and not args.no_flux0:
        args.cache_root.mkdir(parents=True, exist_ok=True)
        try:
            flux0_root = _clone_flux0(args.cache_root)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            print(f"\nflux0 SKIP: {e}")
            return
        if "flux0_server" in flux0_targets:
            _run_target(
                TargetConfig(
                    name="flux0 server",
                    root=flux0_root / "packages" / "server" / "src",
                    make_resolver=lambda: ManualResolver(specs=["."]),
                ),
                args.repeats,
            )
        if "flux0_cli" in flux0_targets:
            _run_target(
                TargetConfig(
                    name="flux0 cli",
                    root=flux0_root / "packages" / "cli" / "src",
                    make_resolver=lambda: ManualResolver(specs=["."]),
                ),
                args.repeats,
            )
        if "flux0_workspace" in flux0_targets:
            try:
                venv = _ensure_flux0_venv(flux0_root)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                print(f"\nflux0_workspace SKIP: {e}")
                return
            _run_target(
                TargetConfig(
                    name="flux0 workspace (uv)",
                    root=flux0_root,
                    make_resolver=UvResolver,
                    rust_kwargs={"python_env": str(venv)},
                ),
                args.repeats,
            )


if __name__ == "__main__":
    main()
