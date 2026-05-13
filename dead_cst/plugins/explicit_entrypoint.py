"""Plugin: mark user-specified symbols (file paths, FQNs, regexes) as
entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..graph import SymbolNode
from ._core import GraphOp, ObserveContext, PluginContext, mark_entrypoints

if TYPE_CHECKING:
    from ..graph import VisitorPayload

EXPLICIT_PREFIX = "<entrypoint>:"


@dataclass
class ExplicitEntrypointPlugin:
    """Mark user-specified symbols as entrypoints.

    ``specs`` accepts the same three forms the CLI's ``-e`` flag used to:

    * ``str`` -- matches either an exact fully-qualified name
      (``pkg.mod.func``) or a file path relative to ``project_root``.
    * :class:`pathlib.Path` -- matches an exact absolute path.
    * :class:`re.Pattern` -- matched against the file path relative to
      ``project_root``.

    For every matching :class:`SymbolNode`, a synthetic entrypoint node is
    added with an edge pointing at the match. Finalize-only: matches are
    computed against the assembled graph, so user specs that target
    plugin-emitted synthetics are recognized too.
    """

    specs: list[str | Path | re.Pattern[str]] = field(default_factory=list)
    name: str = "explicit"
    version: int = 1777760307

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        root = ctx.project_root
        for node in ctx.package_nodes:
            if not self._matches(node, root):
                continue
            yield from mark_entrypoints(f"{EXPLICIT_PREFIX}{node.fqname}", node.path, [node])

    def _matches(self, sym: SymbolNode, root: Path) -> bool:
        try:
            rel = str(sym.path.relative_to(root))
        except ValueError:
            rel = str(sym.path)
        for spec in self.specs:
            if isinstance(spec, re.Pattern):
                if spec.match(rel):
                    return True
            elif isinstance(spec, Path):
                if spec == sym.path:
                    return True
            elif isinstance(spec, str):
                if spec == rel or spec == sym.fqname:
                    return True
        return False
