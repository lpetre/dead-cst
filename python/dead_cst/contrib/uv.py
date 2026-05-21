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

    * ``Package.path`` is the *member directory* (everything the
      member owns -- ``src/``, ``tests/``, ``scripts/``, ...).
    * ``Package.exported_paths`` is the *published* dirs that
      consumers see when they depend on the member:
      ``(<member_dir>/src,)`` for src-layout (the wheel's contents) or
      ``(<member_dir>,)`` for flat-layout. This keeps a member's
      ``tests/`` package out of consumers' module-resolution namespace.

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

        # Each member contributes (owned_dir, exported_dir) -- they
        # differ for src-layout and coincide for flat. Members whose
        # dir doesn't exist are dropped (matches today's behavior: an
        # editable entry pointing at a deleted path can't be analyzed).
        layouts: dict[str, tuple[Path, Path]] = {}
        for name, member_dir in member_dirs.items():
            exported = _exported_for(member_dir)
            if exported is None:
                continue
            layouts[name] = (member_dir, exported)

        out: list[Package] = []
        for name, (owned, exported) in layouts.items():
            dep_names = [d for d in member_deps[name] if d in layouts]
            out.append(
                Package(
                    path=owned,
                    name=name,
                    deps=tuple(dep_names),
                    exported_paths=(exported,),
                )
            )
        return tuple(out)


def _exported_for(member_dir: Path) -> Path | None:
    """Return the directory consumers should put on their search path.

    Mirrors uv's wheel-build conventions: a ``<member>/src/`` if it
    exists (src-layout), otherwise the member dir itself (flat layout).
    ``None`` if the member directory is missing entirely.
    """
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
