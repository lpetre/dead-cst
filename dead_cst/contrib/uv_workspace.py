"""Resolver that discovers workspace members from ``uv.lock``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..resolvers._core import SourceTree, SourceTreeFlags, load_toml
from ..resolvers._imports import default_resolve_import


@dataclass
class UvWorkspaceResolver:
    """Discover workspace members from ``uv.lock`` and emit one
    :class:`SourceTree` per member, wired together via the lockfile's
    resolved dependency graph.

    For each ``[[package]]`` whose ``source`` is ``{ editable = "..." }``
    or ``{ virtual = "..." }`` -- uv's two markers for a workspace
    member -- emit one ``EXPORTED`` :class:`SourceTree` whose ``path``
    is the member's importable directory (``<member>/src`` if that
    dir exists, else ``<member>`` itself), whose ``package`` is the
    lockfile's package name, and whose ``search_trees`` lists the
    importable directories of the member's first-party workspace
    dependencies. The workspace root itself (``virtual = "."``) is
    skipped -- it's a container that holds ``[tool.uv.workspace]``,
    not a member.

    Run ``dead-cst`` with the workspace's venv active (``uv run
    dead-cst ...``) so third-party imports resolve against the
    workspace's installed distributions via the running Python's
    ``sys.path``.
    """

    lock_path: Path | None = None
    name: str = "uv_workspace"
    version: int = 1777985838

    def resolve(self, project_root: Path) -> list[SourceTree]:
        project_root = project_root.resolve()
        data = load_toml(self.lock_path or project_root / "uv.lock")
        if data is None:
            return []

        member_dirs: dict[str, Path] = {}
        member_deps: dict[str, list[str]] = {}
        for pkg in data.get("package", []):
            if not isinstance(pkg, dict):
                continue
            source = pkg.get("source") or {}
            location = source.get("editable") or source.get("virtual")
            if location is None:
                continue
            member_dir = (project_root / location).resolve()
            if member_dir == project_root:
                continue
            name = pkg["name"]
            member_dirs[name] = member_dir
            member_deps[name] = [
                d["name"] for d in pkg.get("dependencies", []) if isinstance(d, dict)
            ]

        out: list[SourceTree] = []
        for name, member_dir in member_dirs.items():
            src_root = _src_root_for(member_dir)
            if src_root is None:
                continue
            search: list[Path] = []
            for dep_name in member_deps[name]:
                dep_dir = member_dirs.get(dep_name)
                if dep_dir is None:
                    continue
                dep_src = _src_root_for(dep_dir)
                if dep_src is not None:
                    search.append(dep_src)
            out.append(
                SourceTree(
                    path=src_root,
                    package=name,
                    flags=SourceTreeFlags.EXPORTED,
                    search_trees=tuple(search),
                )
            )
        return out

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)


def _src_root_for(member_dir: Path) -> Path | None:
    if (member_dir / "src").is_dir():
        return (member_dir / "src").resolve()
    if member_dir.is_dir():
        return member_dir.resolve()
    return None
