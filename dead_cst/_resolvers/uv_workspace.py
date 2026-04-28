"""Resolver that discovers workspace members from ``uv.lock``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._core import PathMap, load_toml
from .venv import MissingVenvError, find_venv_site_packages


@dataclass
class UvWorkspaceResolver:
    """Discover workspace members from ``uv.lock`` and wire their source roots
    together using uv's resolved dependency graph.

    For each ``[[package]]`` entry whose ``source`` is ``{ editable = "..." }``
    or ``{ virtual = "..." }`` -- uv's two markers for a workspace member --
    emit one :class:`PathMap` entry::

        {member_src_root: [direct_workspace_dep_src_roots, workspace_site_packages]}

    ``editable`` members are installable distributions; ``virtual`` members
    are runnable apps/services that aren't shipped as wheels. Both are
    first-party code that needs to be analyzed. The workspace root itself
    (``virtual = "."``) is skipped -- it's a container that holds
    ``[tool.uv.workspace]``, not a member.

    The src root for a member is ``<member_dir>/src`` if that directory
    exists, else ``<member_dir>`` itself (matching the convention
    :class:`PyprojectResolver` uses for single-package projects).

    Direct workspace dependencies come from the lockfile's per-package
    ``dependencies`` array; transitive deps are reachable through the chain
    of returned bases and don't need to be re-listed per member.

    The resolver also requires the workspace's shared venv to be present
    (uv puts a single ``.venv`` at the workspace root). Each member's
    dep list ends with that ``site-packages`` so third-party imports
    resolve to ``[external dist] <pkg>`` synthetics rather than the
    ``[unresolved]`` fallback. If no venv is found,
    :class:`MissingVenvError` is raised -- the user almost certainly
    forgot ``uv sync --all-packages``, and silently returning an empty
    path map just defers a much less actionable failure into the
    plugin pass.
    """

    lock_path: Path | None = None
    name: str = "uv_workspace"

    def resolve(self, project_root: Path) -> PathMap:
        project_root = project_root.resolve()
        data = load_toml(self.lock_path or project_root / "uv.lock")
        if data is None:
            return {}

        site_packages = find_venv_site_packages(project_root)
        if site_packages is None:
            raise MissingVenvError(
                f"uv_workspace resolver: no virtual environment found for "
                f"workspace at {project_root}. Run `uv sync --all-packages` "
                f"to populate the shared `.venv`."
            )

        member_dirs: dict[str, Path] = {}
        member_deps: dict[str, list[str]] = {}
        for pkg in data.get("package", []):
            source = pkg.get("source") or {}
            location = source.get("editable") or source.get("virtual")
            if location is None:
                continue
            member_dir = (project_root / location).resolve()
            # The workspace root itself appears as ``virtual = "."``; it's a
            # container for ``[tool.uv.workspace]``, not a member to analyze.
            if member_dir == project_root:
                continue
            name = pkg["name"]
            member_dirs[name] = member_dir
            member_deps[name] = [d["name"] for d in pkg.get("dependencies", [])]

        out: PathMap = {}
        for name, member_dir in member_dirs.items():
            src_root = _src_root_for(member_dir)
            if src_root is None:
                continue
            deps: list[Path] = []
            for dep_name in member_deps[name]:
                dep_dir = member_dirs.get(dep_name)
                if dep_dir is None:
                    continue
                dep_src = _src_root_for(dep_dir)
                if dep_src is not None:
                    deps.append(dep_src)
            deps.append(site_packages)
            out[src_root] = deps
        return out


def _src_root_for(member_dir: Path) -> Path | None:
    if (member_dir / "src").is_dir():
        return (member_dir / "src").resolve()
    if member_dir.is_dir():
        return member_dir.resolve()
    return None
