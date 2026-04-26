"""Plugin: treat ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
from libcst.metadata import FullRepoManager

from .._symbols import SymbolNode
from ._core import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node


@dataclass
class MainBlockPlugin:
    """Treat ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module that contains a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint node with an edge to the containing
    module. The module's existing internal edges (collected by the regular
    visitor) then keep every symbol referenced in the block reachable.

    This is a :class:`CSTAwareEdgePlugin` so it can reuse the
    :class:`FullRepoManager` instances the analyzer already built rather than
    re-parsing files.
    """

    name: str = "main_block"
    cst_aware: bool = True

    def contribute(
        self, ctx: PluginContext, managers: dict[Path, FullRepoManager]
    ) -> Iterable[GraphOp]:
        modules_by_path: dict[Path, SymbolNode] = {}
        for node in ctx.graph.nodes:
            if node.type == "module":
                modules_by_path[node.path] = node

        for base, mgr in managers.items():
            for path, module_node in modules_by_path.items():
                if not path.is_relative_to(base):
                    continue
                try:
                    wrapper = mgr.get_metadata_wrapper_for_path(path)
                except Exception:
                    continue
                if not _has_main_block(wrapper.module):
                    continue
                synth = synthetic_node(
                    fqname=f"<__main__>:{module_node.fqname}",
                    path=path,
                )
                yield AddNode(synth, entrypoint=True)
                yield AddEdge(synth, module_node)


def _has_main_block(module: cst.Module) -> bool:
    for stmt in module.body:
        if not isinstance(stmt, cst.If):
            continue
        if _is_name_eq_main(stmt.test):
            return True
    return False


def _is_name_eq_main(expr: cst.BaseExpression) -> bool:
    if not isinstance(expr, cst.Comparison):
        return False
    if len(expr.comparisons) != 1:
        return False
    op = expr.comparisons[0]
    if not isinstance(op.operator, cst.Equal):
        return False
    left, right = expr.left, op.comparator
    return (_is_dunder_name(left, "__name__") and _is_string(right, "__main__")) or (
        _is_string(left, "__main__") and _is_dunder_name(right, "__name__")
    )


def _is_dunder_name(expr: cst.BaseExpression, name: str) -> bool:
    return isinstance(expr, cst.Name) and expr.value == name


def _is_string(expr: cst.BaseExpression, value: str) -> bool:
    if isinstance(expr, cst.SimpleString):
        return expr.evaluated_value == value
    if isinstance(expr, cst.ConcatenatedString):
        try:
            return expr.evaluated_value == value
        except Exception:
            return False
    return False
