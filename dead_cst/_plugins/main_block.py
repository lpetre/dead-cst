"""Plugin: treat ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst

from ._core import GraphOp, PluginContext, is_name, mark_entrypoints

MAIN_BLOCK_PREFIX = "<__main__>:"


@dataclass
class MainBlockPlugin:
    """Treat ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module that contains a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint node with an edge to the containing
    module. The module's existing internal edges (collected by the regular
    visitor) then keep every symbol referenced in the block reachable.

    There's no useful import-based prefilter here, but the parsed modules
    are already in the per-base cache, so iterating every module is
    effectively free.
    """

    name: str = "main_block"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for path, module_node in ctx.base_modules():
            module = ctx.parse(path)
            if module is None or not _has_main_block(module):
                continue
            yield from mark_entrypoints(
                f"{MAIN_BLOCK_PREFIX}{module_node.fqname}", path, [module_node]
            )


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
    return (is_name(left, "__name__") and _is_string(right, "__main__")) or (
        _is_string(left, "__main__") and is_name(right, "__name__")
    )


def _is_string(expr: cst.BaseExpression, value: str) -> bool:
    if isinstance(expr, cst.SimpleString):
        return expr.evaluated_value == value
    if isinstance(expr, cst.ConcatenatedString):
        try:
            return expr.evaluated_value == value
        except Exception:
            return False
    return False
