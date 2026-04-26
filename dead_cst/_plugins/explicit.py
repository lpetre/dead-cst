"""Plugin: mark user-specified symbols (file paths, FQNs, regexes) as
entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .._symbols import SymbolNode
from ._core import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node


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
    added with an edge pointing at the match.
    """

    specs: list[str | Path | re.Pattern[str]] = field(default_factory=list)
    name: str = "explicit"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        root = ctx.project_root
        for node in ctx.graph.nodes:
            if not self._matches(node, root):
                continue
            synth = synthetic_node(
                fqname=f"<entrypoint>:{node.fqname}",
                path=node.path,
            )
            yield AddNode(synth, entrypoint=True)
            yield AddEdge(synth, node)

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
