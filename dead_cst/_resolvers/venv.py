"""Resolver that wires a project's active or sibling ``.venv`` into the path map."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ._core import PathMap
from ._imports import default_resolve_import


class MissingVenvError(RuntimeError):
    """Raised by :class:`VenvResolver` (and resolvers that compose it,
    like :class:`~dead_cst._resolvers.uv_workspace.UvWorkspaceResolver`)
    when the user asked for venv-based resolution but no virtual
    environment could be located.

    The user's framework plugins won't function without a populated
    site-packages on the search path, so the resolver errors out with
    an actionable message instead of silently returning ``{}`` and
    letting downstream plugins fail with cryptic
    ``UnresolvedDependencyError``\\s.
    """


@dataclass
class VenvResolver:
    """Discover a sibling ``.venv`` (or the active venv) and add its
    ``site-packages`` as a dep path of the project root.

    The dep path lets import resolution see third-party distributions, which
    lets the graph correctly classify external imports instead of warning.

    When invoked, the resolver raises :class:`MissingVenvError` if no
    venv is found at any of the candidate locations. The user explicitly
    selected ``--resolver venv``, so silently producing an empty path
    map (the previous behavior) just defers the failure into the
    plugin pass with a much less actionable error.
    """

    venv_dir: str | None = None
    name: str = "venv"

    def resolve(self, project_root: Path) -> PathMap:
        project_root = project_root.resolve()
        sp = find_venv_site_packages(project_root, self.venv_dir)
        if sp is None:
            raise MissingVenvError(_missing_venv_message(project_root, self.venv_dir))
        return {project_root: [sp]}

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)


def find_venv_site_packages(project_root: Path, venv_dir: str | None = None) -> Path | None:
    """Locate a ``site-packages`` dir for ``project_root``.

    Returns the first match from (in order): an explicit ``venv_dir``
    under the project, the conventional ``.venv`` / ``venv`` siblings,
    or the currently-active venv. Returns ``None`` if none of those
    point at a real ``site-packages``. Shared with
    :class:`~dead_cst._resolvers.uv_workspace.UvWorkspaceResolver` so
    workspace setups inherit the same venv discovery.
    """
    project_root = project_root.resolve()
    candidates: list[Path] = []
    if venv_dir:
        candidates.append(project_root / venv_dir)
    else:
        candidates.extend([project_root / ".venv", project_root / "venv"])
        active = _active_venv()
        if active:
            candidates.append(active)

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for sp in _site_packages_for(candidate):
            return sp
    return None


def _missing_venv_message(project_root: Path, venv_dir: str | None) -> str:
    if venv_dir:
        return (
            f"venv resolver: '{project_root / venv_dir}' is not a valid virtual "
            f"environment. Create or sync it (e.g. `uv venv {venv_dir}` "
            f"followed by `uv sync`)."
        )
    return (
        f"venv resolver: no virtual environment found for {project_root}. "
        f"Tried '{project_root}/.venv', '{project_root}/venv', and the "
        f"currently-active interpreter, but none had a usable "
        f"'site-packages'. Run `uv sync` (or `python -m venv .venv && "
        f"pip install -e .`) and try again."
    )


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
