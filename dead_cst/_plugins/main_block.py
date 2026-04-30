"""Plugin: treat ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import libcst as cst
from libcst.metadata import CodePosition, CodeRange, PositionProvider

from ._core import GraphOp, PluginContext, is_name, mark_entrypoints

MAIN_BLOCK_PREFIX = "<__main__>:"


@dataclass
class MainBlockPlugin:
    """Treat ``if __name__ == "__main__":`` blocks as entrypoints.

    For each module containing a top-level ``if __name__ == "__main__":``
    block, emit a synthetic entrypoint with edges to (a) the containing
    module and (b) every top-level decl whose binding site falls inside
    the block's source range.

    Why both: the visitor records ``module -> referent`` edges for bare
    references inside the block (so ``main()`` keeps ``main`` alive via
    the module edge alone). But assignments like
    ``app = Foo(fn=main).cli()`` register ``app`` as a top-level decl
    whose value frame produces ``app -> Foo`` / ``app -> main`` -- and
    ``app`` itself has no incoming reference. Without a direct
    ``synth -> app`` edge that chain stays unreachable and ``Foo`` /
    ``main`` look dead.

    There's no useful import-based prefilter here, but the parsed
    modules are already in the per-base cache, so iterating every
    module is effectively free.
    """

    name: str = "main_block"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for path, module_node in ctx.base_modules():
            module = ctx.parse(path)
            if module is None:
                continue
            main_block = _find_main_block(module)
            if main_block is None:
                continue
            block_range = cst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(
                PositionProvider
            )[main_block]
            block_decls = [
                node
                for node in ctx.base_nodes()
                if node.path == path
                and node.type in ("class", "function", "variable", "import")
                and _range_contains(block_range, node.position)
            ]
            yield from mark_entrypoints(
                f"{MAIN_BLOCK_PREFIX}{module_node.fqname}",
                path,
                [module_node, *block_decls],
            )


def _find_main_block(module: cst.Module) -> cst.If | None:
    for stmt in module.body:
        if isinstance(stmt, cst.If) and _is_name_eq_main(stmt.test):
            return stmt
    return None


def _range_contains(outer: CodeRange, inner: CodeRange) -> bool:
    return _le(outer.start, inner.start) and _le(inner.end, outer.end)


def _le(a: CodePosition, b: CodePosition) -> bool:
    return (a.line, a.column) <= (b.line, b.column)


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
