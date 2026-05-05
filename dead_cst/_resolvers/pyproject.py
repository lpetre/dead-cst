"""Resolver that reads ``[tool.dead-cst]`` from a project's ``pyproject.toml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._core import PathMap, load_toml
from ._imports import default_resolve_import


@dataclass
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

    name: str = "pyproject"
    version: int = 1777973896

    def resolve(self, project_root: Path) -> PathMap:
        project_root = project_root.resolve()
        data = load_toml(project_root / "pyproject.toml")
        if data is None:
            return {}

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

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
