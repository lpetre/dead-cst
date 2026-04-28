"""Resolver that discovers workspace members from ``uv.lock``."""

from __future__ import annotations

from pathlib import Path

from ._core import PathMap, load_toml


class UvWorkspaceResolver:
    """Discover workspace members from ``uv.lock`` and wire their source roots
    together using uv's resolved dependency graph.

    For each ``[[package]]`` entry whose ``source`` is ``{ editable = "..." }``
    or ``{ virtual = "..." }`` -- uv's two markers for a workspace member --
    emit one :class:`PathMap` entry::

        {member_src_root: [direct_workspace_dep_src_roots]}

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
    """

    name = "uv_workspace"

    def __init__(self, lock_path: Path | None = None) -> None:
        self.lock_path = lock_path

    def resolve(self, project_root: Path) -> PathMap:
        project_root = project_root.resolve()
        data = load_toml(self.lock_path or project_root / "uv.lock")
        if data is None:
            return {}

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
            out[src_root] = deps
        return out


def _src_root_for(member_dir: Path) -> Path | None:
    if (member_dir / "src").is_dir():
        return (member_dir / "src").resolve()
    if member_dir.is_dir():
        return member_dir.resolve()
    return None
