"""Plugin: keep module-level ``__all__`` variables alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._core import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node


@dataclass
class DunderAllPlugin:
    """Keep module-level ``__all__`` variables alive.

    ``__all__`` declares a module's public export list; removing it would be
    observable even when no source references it. For each top-level
    ``__all__`` variable, emit a synthetic entrypoint node with an edge to
    the variable so :func:`find_reachable` preserves it.
    """

    name: str = "dunder_all"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for node in ctx.graph.nodes:
            if node.type != "variable":
                continue
            if not node.fqname.endswith(".__all__") and node.fqname != "__all__":
                continue
            synth = synthetic_node(
                fqname=f"<__all__>:{node.fqname}",
                path=node.path,
            )
            yield AddNode(synth, entrypoint=True)
            yield AddEdge(synth, node)
