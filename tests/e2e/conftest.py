"""Shared scaffolding for end-to-end tests against real GitHub repositories.

These tests shallow-clone a target repo at a pinned SHA into a persistent
cache directory, then run dead-cst against it. They're deselected by
default (see the ``e2e`` marker in ``pyproject.toml``); run with::

    uv run pytest -m e2e

Each repo is cloned at most once per cache directory, so the second
invocation of the suite is fast. Override the cache location with
``DEAD_CST_E2E_CACHE``; otherwise it lives under pytest's own cache dir
so ``pytest --cache-clear`` wipes it.

Tests skip (rather than fail) when ``git`` isn't on ``PATH`` or the
clone hits a network / fetch error -- e2e tests are about catching
real-world breakage in dead-cst, not about gating CI on flaky network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pytest


def _cache_root(request: pytest.FixtureRequest) -> Path:
    override = os.environ.get("DEAD_CST_E2E_CACHE")
    if override:
        root = Path(override).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    return Path(request.config.cache.mkdir("e2e-clones"))


def _ensure_clone(name: str, url: str, sha: str, cache_root: Path) -> Path:
    """Return a worktree for ``url`` checked out at ``sha``.

    Idempotent: a second call with the same ``name`` + ``sha`` reuses
    the existing clone. The on-disk marker is the SHA itself -- if a
    previous run left a partial clone behind we wipe and retry rather
    than guess at its state.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available on PATH")

    dest = cache_root / name
    marker = dest / ".dead-cst-e2e-sha"
    if marker.is_file() and marker.read_text().strip() == sha:
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def _git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True)

    try:
        _git("init", "-q")
        _git("remote", "add", "origin", url)
        _git("fetch", "--depth=1", "-q", "--filter=blob:none", "origin", sha)
        _git("checkout", "-q", sha)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        pytest.skip(f"git clone of {url}@{sha} failed: {stderr.strip()}")

    marker.write_text(sha + "\n")
    return dest


@pytest.fixture(scope="session")
def clone_repo(request: pytest.FixtureRequest) -> Callable[[str, str, str], Path]:
    """Return ``clone(name, url, sha) -> Path`` for materialising a pinned repo."""
    root = _cache_root(request)

    def _clone(name: str, url: str, sha: str) -> Path:
        return _ensure_clone(name, url, sha, root)

    return _clone
