"""Resolver that wires a project's active or sibling ``.venv`` into the path map."""

from __future__ import annotations

import sys
from pathlib import Path

from ._core import PathMap


class VenvResolver:
    """Discover a sibling ``.venv`` (or the active venv) and add its
    ``site-packages`` as a dep path of the project root.

    The dep path lets import resolution see third-party distributions, which
    lets the graph correctly classify external imports instead of warning.
    """

    name = "venv"

    def __init__(self, venv_dir: str | None = None) -> None:
        self.venv_dir = venv_dir

    def resolve(self, project_root: Path) -> PathMap:
        project_root = project_root.resolve()
        candidates: list[Path] = []
        if self.venv_dir:
            candidates.append(project_root / self.venv_dir)
        else:
            candidates.extend([project_root / ".venv", project_root / "venv"])
            active = _active_venv()
            if active:
                candidates.append(active)

        for candidate in candidates:
            if not candidate.is_dir():
                continue
            for sp in _site_packages_for(candidate):
                return {project_root: [sp]}
        return {}


def _active_venv() -> Path | None:
    prefix = getattr(sys, "prefix", None)
    base_prefix = getattr(sys, "base_prefix", prefix)
    if prefix and prefix != base_prefix:
        return Path(prefix)
    return None


def _site_packages_for(venv: Path) -> list[Path]:
    """Return the ``site-packages`` dirs inside a venv, if any."""
    paths: list[Path] = []
    lib = venv / "lib"
    if lib.is_dir():
        for py in sorted(lib.glob("python*")):
            sp = py / "site-packages"
            if sp.is_dir():
                paths.append(sp)
    # Windows layout
    win_sp = venv / "Lib" / "site-packages"
    if win_sp.is_dir():
        paths.append(win_sp)
    return paths
