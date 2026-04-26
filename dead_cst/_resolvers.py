"""Pluggable resolvers that discover sys.path-like search paths for a project.

A ``PathResolver`` takes a project root and returns a ``dict[base, [dep_paths]]``
in the same shape ``build_symbol_graph`` already consumes. Multiple resolvers
compose by merging dicts -- see :func:`merge_paths`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

PathMap = dict[Path, list[Path]]


@runtime_checkable
class PathResolver(Protocol):
    name: str

    def resolve(self, project_root: Path) -> PathMap: ...


def merge_paths(*maps: PathMap) -> PathMap:
    """Merge multiple ``PathMap``s, unioning dep-path lists per base."""
    out: PathMap = {}
    for m in maps:
        for base, deps in m.items():
            base = base.resolve()
            existing = out.setdefault(base, [])
            for dep in deps:
                dep = dep.resolve()
                if dep not in existing and dep != base:
                    existing.append(dep)
    return out


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


class PyprojectResolver:
    """Read ``[tool.dead-cst]`` from a project's ``pyproject.toml`` and turn
    its configured paths into a ``PathMap``.

    Example ``pyproject.toml``::

        [tool.dead-cst]
        paths = [
            { base = "src", deps = ["tests"] },
            { base = "scripts" },
        ]

    If no section is configured, fall back to the conventional ``src/``
    layout when present.
    """

    name = "pyproject"

    def resolve(self, project_root: Path) -> PathMap:
        project_root = project_root.resolve()
        pyproject = project_root / "pyproject.toml"
        if not pyproject.is_file():
            return {}

        try:
            import tomllib
        except ImportError:  # pragma: no cover - py<3.11 not supported
            return {}

        with pyproject.open("rb") as f:
            data = tomllib.load(f)

        tool = data.get("tool", {}).get("dead-cst", {})
        entries = tool.get("paths")
        if entries:
            out: PathMap = {}
            for entry in entries:
                base = (project_root / entry["base"]).resolve()
                deps = [(project_root / d).resolve() for d in entry.get("deps", [])]
                out[base] = deps
            return out

        src = project_root / "src"
        if src.is_dir():
            return {src: []}
        return {}


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


BUILTIN_RESOLVERS: dict[str, type[PathResolver]] = {
    VenvResolver.name: VenvResolver,
    PyprojectResolver.name: PyprojectResolver,
    UvWorkspaceResolver.name: UvWorkspaceResolver,
}


def load_resolver(name: str) -> PathResolver:
    """Load a resolver by name. Checks builtins first, then entry points."""
    if name in BUILTIN_RESOLVERS:
        return BUILTIN_RESOLVERS[name]()

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.resolvers"):
        if ep.name == name:
            cls = ep.load()
            return cls()
    raise KeyError(f"Unknown path resolver: {name!r}")
