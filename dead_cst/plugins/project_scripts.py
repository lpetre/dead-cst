"""Plugin: treat every ``[project.scripts]`` entry in ``pyproject.toml`` as
an entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..resolvers import load_toml
from ._core import GraphOp, ObserveContext, PluginContext, mark_entrypoints

if TYPE_CHECKING:
    from ..graph import VisitorPayload

PROJECT_SCRIPTS_PREFIX = "<project.scripts>:"

logger = logging.getLogger(__name__)


@dataclass
class ProjectScriptsPlugin:
    """Treat every ``[project.scripts]`` entry in ``pyproject.toml`` as an
    entrypoint.

    For each ``name = "pkg.mod:func"`` mapping, look up ``pkg.mod.func`` in the
    symbol trie and wire a synthetic entrypoint node to it.

    Reads the pyproject in the *current base*. uv workspaces and similar
    layouts are 1:1 with bases, so each member's scripts resolve in its
    own base's symbol lookup -- which is exactly the lookup that contains
    that member plus the deps it can import from. ``pyproject_path`` can
    be set to override the location for non-standard layouts.

    Finalize-only: the pyproject scan runs once per base, after the
    per-file payloads have been applied and import edges resolved, so
    the symbol trie is fully populated.
    """

    name: str = "project_scripts"
    version: int = 1777760307
    pyproject_path: Path | None = None

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        pyproject = self.pyproject_path or ctx.base / "pyproject.toml"
        data = load_toml(pyproject)
        if data is None:
            return

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
            yield from mark_entrypoints(
                f"{PROJECT_SCRIPTS_PREFIX}{script_name}", pyproject, target_nodes
            )
