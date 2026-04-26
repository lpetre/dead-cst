"""Resolver that reads ``[tool.dead-cst]`` from a project's ``pyproject.toml``."""

from __future__ import annotations

from pathlib import Path

from ._core import PathMap


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
