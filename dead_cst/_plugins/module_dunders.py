"""Plugin: keep module-level dunder variables alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .._symbols import SymbolNode
from ._core import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node


@dataclass
class ModuleDundersPlugin:
    """Keep module-level dunder variables alive.

    Variables named ``__xxx__`` at module scope (``__all__``, ``__version__``,
    ``__author__``, ``__license__``, ...) are read by external tooling
    (packagers, ``importlib.metadata`` fallbacks, doc generators) so removing
    them as dead would be observable even when no source references them. For
    each such top-level variable, emit a synthetic entrypoint node with an
    edge to it so :func:`find_reachable` preserves it.
    """

    name: str = "module_dunders"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for node in ctx.graph.nodes:
            if not node.path.is_relative_to(ctx.base):
                continue
            if not _is_module_dunder(node):
                continue
            synth = synthetic_node(
                fqname=f"<dunder>:{node.fqname}",
                path=node.path,
            )
            yield AddNode(synth, entrypoint=True)
            yield AddEdge(synth, node)


def _is_module_dunder(node: SymbolNode) -> bool:
    """True for module-level variables named like ``__xxx__``."""
    if node.type != "variable":
        return False
    name = node.fqname.rpartition(".")[2]
    return len(name) > 4 and name.startswith("__") and name.endswith("__")
