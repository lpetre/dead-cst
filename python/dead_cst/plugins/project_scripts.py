"""Plugin: mark every ``[project.scripts]`` entry in ``pyproject.toml`` as an entrypoint."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..graph import NodeFlags
from ._base import Plugin, native


def load_toml(path: Path) -> dict[str, Any] | None:
    """Read ``path`` as TOML; ``None`` if the file is missing."""
    try:
        f = path.open("rb")
    except OSError:
        return None
    with f:
        return tomllib.load(f)


PROJECT_SCRIPTS_PREFIX = "<project.scripts>:"

logger = logging.getLogger(__name__)


@dataclass
class ProjectScriptsPlugin(Plugin):
    """Mark every ``[project.scripts]`` entry in ``pyproject.toml`` as an entrypoint.

    For each ``name = "pkg.mod:func"`` mapping, look up ``pkg.mod.func`` in
    the project graph and wire a synthetic entrypoint node to it.
    """

    pyproject_path: Path | None = None

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        pyproject = self.pyproject_path or Path(ctx.project_root) / "pyproject.toml"
        data = load_toml(pyproject)
        if data is None:
            return

        scripts = data.get("project", {}).get("scripts", {})
        for script_name, target in scripts.items():
            module_part, _, decl_part = target.partition(":")
            fqname = f"{module_part}.{decl_part}" if decl_part else module_part
            targets = ctx.find_declarations(fqname)
            if not targets:
                module_node = ctx.find_module(module_part)
                if module_node is not None:
                    targets = [module_node]
            if not targets:
                logger.warning(
                    "ProjectScriptsPlugin: %s -> %r not found in symbol graph",
                    script_name,
                    target,
                )
                continue
            yield native.AddNode(
                fqname=f"{PROJECT_SCRIPTS_PREFIX}{script_name}",
                path=str(pyproject),
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to=targets,
            )
