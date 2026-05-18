"""Resolver that discovers workspace members from ``uv.lock``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..resolvers._core import Package, load_toml


class MissingVenvError(RuntimeError):
    """Raised by :class:`UvResolver` when the workspace's shared ``.venv`` is missing."""


@dataclass
class UvResolver:
    """Discover workspace members from ``uv.lock``.

    For each ``[[package]]`` entry whose ``source`` is
    ``{ editable = "..." }`` or ``{ virtual = "..." }`` -- uv's two
    markers for a workspace member -- emit one
    :class:`~dead_cst.resolvers.Package`. The workspace root itself
    (``virtual = "."``) is skipped.

    The src root for a member is ``<member_dir>/src`` if that directory
    exists, else ``<member_dir>`` itself.

    Requires the workspace's shared venv to be present (uv puts a
    single ``.venv`` at the workspace root). If no venv is found,
    :class:`MissingVenvError` is raised.
    """

    lock_path: Path | None = None

    def resolve(self, project_root: Path) -> tuple[Package, ...]:
        project_root = project_root.resolve()
        data = load_toml(self.lock_path or project_root / "uv.lock")
        if data is None:
            return ()

        if _find_venv_site_packages(project_root) is None:
            raise MissingVenvError(
                f"uv resolver: no virtual environment found for "
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
            if member_dir == project_root:
                continue
            name = pkg["name"]
            member_dirs[name] = member_dir
            member_deps[name] = [d["name"] for d in pkg.get("dependencies", [])]

        src_roots: dict[str, Path] = {}
        for name, member_dir in member_dirs.items():
            src_root = _src_root_for(member_dir)
            if src_root is not None:
                src_roots[name] = src_root

        out: list[Package] = []
        for name, src_root in src_roots.items():
            dep_names = [d for d in member_deps[name] if d in src_roots]
            out.append(Package(path=src_root, name=name, deps=tuple(dep_names)))
        return tuple(out)


def _src_root_for(member_dir: Path) -> Path | None:
    if (member_dir / "src").is_dir():
        return member_dir / "src"
    if member_dir.is_dir():
        return member_dir
    return None


def _find_venv_site_packages(project_root: Path) -> Path | None:
    candidates: list[Path] = [project_root / ".venv", project_root / "venv"]
    active = _active_venv()
    if active:
        candidates.append(active)

    for candidate in candidates:
        for sp in _site_packages_for(candidate):
            return sp
    return None


def _active_venv() -> Path | None:
    prefix = getattr(sys, "prefix", None)
    base_prefix = getattr(sys, "base_prefix", prefix)
    if prefix and prefix != base_prefix:
        return Path(prefix)
    return None


def _site_packages_for(venv: Path) -> list[Path]:
    paths: list[Path] = []
    lib = venv / "lib"
    if lib.is_dir():
        for py in sorted(lib.glob("python*")):
            sp = py / "site-packages"
            if sp.is_dir():
                paths.append(sp)
    win_sp = venv / "Lib" / "site-packages"
    if win_sp.is_dir():
        paths.append(win_sp)
    return paths
