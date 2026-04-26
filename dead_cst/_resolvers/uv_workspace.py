"""Resolver that discovers workspace members from ``uv.lock``."""

from __future__ import annotations

from pathlib import Path

from ._core import PathMap


class UvWorkspaceResolver:
    """Discover workspace members from ``uv.lock`` and wire their source roots
    together using uv's resolved dependency graph.

    For each ``[[package]]`` entry whose ``source`` is ``{ editable = "..." }``
    -- uv's marker for a workspace member -- emit one :class:`PathMap` entry::

        {member_src_root: [direct_workspace_dep_src_roots]}

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
        lock = self.lock_path or project_root / "uv.lock"
        if not lock.is_file():
            return {}

        try:
            import tomllib
        except ImportError:  # pragma: no cover - py<3.11 not supported
            return {}

        with lock.open("rb") as f:
            data = tomllib.load(f)

        member_dirs: dict[str, Path] = {}
        member_deps: dict[str, list[str]] = {}
        for pkg in data.get("package", []):
            source = pkg.get("source") or {}
            editable = source.get("editable")
            if editable is None:
                continue
            name = pkg["name"]
            member_dirs[name] = (project_root / editable).resolve()
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
