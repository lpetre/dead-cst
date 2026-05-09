"""Resolver that discovers workspace members from ``uv.lock``."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..resolvers._core import Package, load_toml
from ..resolvers._exports import exported_roots
from ..resolvers._imports import clear_path_caches, default_resolve_import


class MissingVenvError(RuntimeError):
    """Raised by :class:`UvResolver` when the workspace's shared ``.venv`` is missing."""


@dataclass
class UvResolver:
    """Discover workspace members from ``uv.lock`` and wire their source roots
    together using uv's resolved dependency graph.

    For each ``[[package]]`` entry whose ``source`` is ``{ editable = "..." }``
    or ``{ virtual = "..." }`` -- uv's two markers for a workspace member --
    emit one :class:`~dead_cst.resolvers.Package`. ``editable`` members are
    installable distributions; ``virtual`` members are runnable apps/services
    that aren't shipped as wheels. Both are first-party code that needs to
    be analyzed. The workspace root itself (``virtual = "."``) is skipped --
    it's a container that holds ``[tool.uv.workspace]``, not a member.

    The src root for a member is ``<member_dir>/src`` if that directory
    exists, else ``<member_dir>`` itself.

    Direct workspace dependencies come from the lockfile's per-package
    ``dependencies`` array (matched against fellow workspace members);
    transitive deps are reachable through the chain of returned packages
    and don't need to be re-listed per member. The package's
    :attr:`~dead_cst.resolvers.Package.deps` carries the dep package
    names (uv's ``[[package]].name`` after PEP 503 canonicalization, but
    here just the lockfile's literal name).

    The resolver also requires the workspace's shared venv to be present
    (uv puts a single ``.venv`` at the workspace root). The venv's
    ``site-packages`` dir is *not* added to any package's ``deps`` -- it
    is non-first-party, so the dep model doesn't represent it. Instead,
    :meth:`resolve_import` lazily splices the venv onto ``sys.path`` on
    first use within an analysis materialization, so third-party
    imports still classify as ``[external dist] <pkg>`` rather than
    ``[unresolved]``. If no venv is found, :class:`MissingVenvError`
    is raised.
    """

    lock_path: Path | None = None
    _site_packages: Path | None = field(default=None, init=False, repr=False)

    def resolve(self, project_root: Path) -> tuple[Package, ...]:
        project_root = project_root.resolve()
        data = load_toml(self.lock_path or project_root / "uv.lock")
        if data is None:
            return ()

        site_packages = _find_venv_site_packages(project_root)
        if site_packages is None:
            raise MissingVenvError(
                f"uv resolver: no virtual environment found for "
                f"workspace at {project_root}. Run `uv sync --all-packages` "
                f"to populate the shared `.venv`."
            )
        self._site_packages = site_packages

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

        src_roots: dict[str, Path] = {}
        for name, member_dir in member_dirs.items():
            src_root = _src_root_for(member_dir)
            if src_root is not None:
                src_roots[name] = src_root

        out: list[Package] = []
        for name, src_root in src_roots.items():
            # Workspace deps only -- regular PyPI deps don't have a
            # member src root and are reached through the venv at
            # resolve_import time.
            dep_names = [d for d in member_deps[name] if d in src_roots]
            exported = tuple(exported_roots(member_dirs[name]) or ())
            out.append(
                Package(
                    path=src_root,
                    name=name,
                    exported=exported,
                    deps=tuple(dep_names),
                )
            )
        return tuple(out)

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        # The workspace ``.venv`` is no longer represented in
        # ``Package.deps``, so :func:`Analysis._rebind_sys_path` does not
        # put it on ``sys.path``. Splice it on lazily here so
        # :func:`default_resolve_import`'s ``importlib`` lookup can still
        # find venv-installed third-party packages.
        if self._site_packages is not None:
            sp_str = str(self._site_packages)
            if sp_str not in sys.path:
                sys.path.append(sp_str)
                # ``sys.path`` mutation invalidates the dist / module
                # caches; clear so the next lookup observes the venv.
                # Subsequent calls within the same package see the venv
                # already on sys.path and skip the clear.
                clear_path_caches()
        return default_resolve_import(name, search_paths)


def _src_root_for(member_dir: Path) -> Path | None:
    """Pick ``<member_dir>/src`` if it exists, else ``member_dir`` itself.

    ``member_dir`` is already absolute (the caller resolves it before lookup),
    so no further normalization is needed.
    """
    if (member_dir / "src").is_dir():
        return member_dir / "src"
    if member_dir.is_dir():
        return member_dir
    return None


def _find_venv_site_packages(project_root: Path) -> Path | None:
    """Locate a ``site-packages`` dir for the workspace at ``project_root``.

    Tries the conventional ``.venv`` / ``venv`` siblings, then falls
    back to the currently-active interpreter. Returns ``None`` if none
    of those point at a real ``site-packages``.
    """
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
