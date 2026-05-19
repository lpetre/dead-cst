"""Shared infrastructure for the scripts/profile_*.py benchmark scripts.

Both ``profile_plugins_cold.py`` (cold rust path with one plugin at a
time) and ``profile_graph_build.py`` (cold/warm rust-build profile with
per-phase breakdown) need the same harness:

* a tempdir-staged copy of ``python/dead_cst/`` as the "self" target
* a pinned-SHA shallow clone of flux0 (for ``flux0_server`` /
  ``flux0_cli`` / ``flux0_workspace``)
* a ``uv sync --all-packages`` for the workspace target
* a ``.py`` / ``.pyi`` file counter
* a ``TargetConfig`` for the rust constructor kwargs

Kept in one place so a layout change (`#191 moved the package from
``dead_cst/`` to ``python/dead_cst/`` and one of the scripts forgot to
follow) only has to land once.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

FLUX0_URL = "https://github.com/flux0-ai/flux0.git"
FLUX0_SHA = "8d04176642b091ddb5c5020486f353d4e824460b"


@dataclass
class TargetConfig:
    """One project to profile. ``project_kwargs`` is forwarded to
    ``native.ProjectContext`` — typically ``python_env`` for workspace
    targets so ty's module resolver picks up third-party packages from
    the workspace's own ``.venv``."""

    name: str
    root: Path
    project_kwargs: dict[str, object] = field(default_factory=dict)


def require_native() -> None:
    """Exit early when the rust extension isn't importable — measuring
    a libcst fallback isn't useful for a rust-focused profile."""
    try:
        from dead_cst import native  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"ERROR: dead_cst.native not importable ({exc}). "
            "Build with: uv run maturin develop --release"
        )


def stage_dead_cst() -> Path:
    """Copy ``python/dead_cst/`` into a fresh tempdir so the analyzer
    sees one clean package."""
    stage = Path(tempfile.mkdtemp(prefix="dead-cst-bench-"))
    src = Path(__file__).resolve().parent.parent / "python" / "dead_cst"
    if not src.exists():
        raise RuntimeError(f"dead_cst package source not found at {src}")
    shutil.copytree(src, stage / "dead_cst", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return stage


def clone_flux0(cache_root: Path) -> Path:
    """Shallow-clone flux0 at the pinned SHA into ``cache_root/flux0``.
    Skipped when an existing clone already matches the SHA."""
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


def ensure_flux0_venv(flux0_root: Path) -> Path:
    """Run ``uv sync --all-packages`` so ty's workspace module resolver
    finds a populated ``.venv``. Idempotent."""
    venv = flux0_root / ".venv"
    if venv.is_dir():
        return venv
    if shutil.which("uv") is None:
        raise RuntimeError("uv not on PATH; cannot create flux0 workspace venv")
    print(f"  (one-time setup: uv sync --all-packages in {flux0_root})", flush=True)
    subprocess.run(["uv", "sync", "--all-packages"], cwd=flux0_root, check=True)
    return venv


def file_count(target_root: Path) -> int:
    """Number of ``.py``/``.pyi`` files under ``target_root``, ignoring
    ``__pycache__`` and ``.venv``."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(target_root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".venv")]
        count += sum(1 for f in filenames if f.endswith((".py", ".pyi")))
    return count


FLUX0_TARGETS = ("flux0_server", "flux0_cli", "flux0_workspace")


def gather_flux0_targets(requested: Iterable[str], cache_root: Path) -> Iterable[TargetConfig]:
    """Yield ``TargetConfig``s for the requested flux0 targets, cloning
    flux0 and seeding the venv if needed. Swallows network / git /
    uv failures with a ``SKIP`` message so the rest of the run can
    proceed."""
    requested_set = {name for name in requested if name.startswith("flux0_")}
    if not requested_set:
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        flux0_root = clone_flux0(cache_root)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nflux0 SKIP: {exc}", flush=True)
        return

    if "flux0_server" in requested_set:
        yield TargetConfig(name="flux0 server", root=flux0_root / "packages" / "server" / "src")
    if "flux0_cli" in requested_set:
        yield TargetConfig(name="flux0 cli", root=flux0_root / "packages" / "cli" / "src")
    if "flux0_workspace" in requested_set:
        try:
            venv = ensure_flux0_venv(flux0_root)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"\nflux0_workspace SKIP: {exc}", flush=True)
        else:
            yield TargetConfig(
                name="flux0 workspace (uv)",
                root=flux0_root,
                project_kwargs={"python_env": str(venv)},
            )


def add_common_target_args(parser: argparse.ArgumentParser, default_targets: list[str]) -> None:
    """Add the ``--targets`` / ``--no-flux0`` / ``--cache-root`` args
    every bench script accepts. Caller provides the default target list
    so each script picks its own appropriate default."""
    parser.add_argument(
        "--targets",
        nargs="+",
        default=default_targets,
        choices=("dead_cst", *FLUX0_TARGETS),
        help="Which real targets to profile.",
    )
    parser.add_argument(
        "--no-flux0",
        action="store_true",
        help="Skip flux0 targets even if listed (offline / no git).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "dead-cst-bench",
        help="Where to cache the flux0 clone.",
    )
