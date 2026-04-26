"""Plugin: treat every ``[project.scripts]`` entry in ``pyproject.toml`` as
an entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ._core import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node

logger = logging.getLogger(__name__)


@dataclass
class ProjectScriptsPlugin:
    """Treat every ``[project.scripts]`` entry in ``pyproject.toml`` as an
    entrypoint.

    For each ``name = "pkg.mod:func"`` mapping, look up ``pkg.mod.func`` in the
    symbol trie and wire a synthetic entrypoint node to it.
    """

    name: str = "project_scripts"
    pyproject_path: Path | None = None

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        pyproject = self.pyproject_path or ctx.project_root / "pyproject.toml"
        if not pyproject.is_file():
            return

        try:
            import tomllib
        except ImportError:  # pragma: no cover
            return

        with pyproject.open("rb") as f:
            data = tomllib.load(f)

        scripts = data.get("project", {}).get("scripts", {})
        for script_name, target in scripts.items():
            module_part, _, decl_part = target.partition(":")
            fqname = f"{module_part}.{decl_part}" if decl_part else module_part
            target_nodes = ctx.find_declarations(fqname)
            if not target_nodes:
                module_node = ctx.find_module(module_part)
                target_nodes = [module_node] if module_node else []
            if not target_nodes:
                logger.warning(
                    "ProjectScriptsPlugin: %s -> %r not found in symbol graph",
                    script_name,
                    target,
                )
                continue
            synth = synthetic_node(
                fqname=f"<project.scripts>:{script_name}",
                path=pyproject,
            )
            yield AddNode(synth, entrypoint=True)
            for target_node in target_nodes:
                yield AddEdge(synth, target_node)
