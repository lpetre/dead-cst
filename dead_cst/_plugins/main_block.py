"""Plugin: treat ``if __name__ == "__main__":`` blocks as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import libcst as cst
from libcst.metadata import CodePosition, CodeRange, PositionProvider

from .._symbols import NodeFlags
from ._core import (
    SYNTHETIC_POSITION,
    GraphOp,
    ObserveContext,
    PluginContext,
    make_payload,
    is_name,
    synthetic_node,
)

if TYPE_CHECKING:
    from .._visitor import VisitorPayload

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

    Pure per-file work: the file's CST locates the ``__main__`` block,
    the file's :class:`VisitorPayload` provides the top-level decls,
    and the contribution is added directly to the cached payload.
    """

    name: str = "main_block"
    version: int = 1777760307

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        main_block = _find_main_block(ctx.module)
        if main_block is None:
            return None
        block_range = cst.MetadataWrapper(ctx.module, unsafe_skip_copy=True).resolve(
            PositionProvider
        )[main_block]

        module_node = next((n for n in ctx.payload.nodes if n.type == "module"), None)
        if module_node is None:
            return None

        block_decls = [
            n
            for n in ctx.payload.nodes
            if n.type in ("class", "function", "variable", "import")
            and _range_contains(block_range, n.position)
        ]

        synth = synthetic_node(
            f"{MAIN_BLOCK_PREFIX}{module_node.fqname}",
            ctx.path,
            flags=NodeFlags.ENTRYPOINT,
        )
        edges = [(synth, module_node, SYNTHETIC_POSITION)]
        edges.extend((synth, d, SYNTHETIC_POSITION) for d in block_decls)
        return make_payload(nodes=[synth], edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()


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
