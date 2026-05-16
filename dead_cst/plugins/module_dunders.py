"""Plugin: keep module-level dunder variables and ``__future__`` imports alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..graph import NodeFlags, SymbolNode
from ._core import (
    SYNTHETIC_POSITION,
    GraphOp,
    ObserveContext,
    PluginContext,
    make_payload,
    simple_name,
    synthetic_node,
)

if TYPE_CHECKING:
    import dead_cst_ty_native as native

    from ..graph import VisitorPayload

DUNDER_PREFIX = "<dunder>:"


@dataclass
class ModuleDundersPlugin:
    """Keep module-level dunder variables and ``__future__`` imports alive.

    Variables named ``__xxx__`` at module scope (``__all__``, ``__version__``,
    ``__author__``, ``__license__``, ...) are read by external tooling
    (packagers, ``importlib.metadata`` fallbacks, doc generators) so removing
    them as dead would be observable even when no source references them.

    ``from __future__ import ...`` statements are compile-time directives
    (``annotations``, ``division``, ...): the local binding is never read,
    but the import itself changes how the surrounding module parses, so
    rewriting it away is observable. Both kinds get a synthetic entrypoint
    node with an edge so :func:`find_reachable` preserves them.

    The plugin runs entirely off the per-file :class:`VisitorPayload` --
    no CST inspection is needed -- so the cache hit path skips it for
    free.
    """

    name: str = "module_dunders"
    version: int = 1777760307

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        targets = [n for n in ctx.payload.nodes if _is_kept_alive(n)]
        if not targets:
            return None
        nodes: list[SymbolNode] = []
        edges = []
        for target in targets:
            synth = synthetic_node(
                f"{DUNDER_PREFIX}{target.fqname}", target.path, flags=NodeFlags.ENTRYPOINT
            )
            nodes.append(synth)
            edges.append((synth, target, SYNTHETIC_POSITION))
        return make_payload(nodes=nodes, edges=edges)

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        import dead_cst_ty_native as native

        for target in [*ctx.find_module_dunders(), *ctx.find_imports_of("__future__")]:
            yield native.AddEntrypoint(target, marker="<dunder>")


def _is_kept_alive(node: SymbolNode) -> bool:
    return _is_module_dunder(node) or _is_future_import(node)


def _is_module_dunder(node: SymbolNode) -> bool:
    """True for module-level variables named like ``__xxx__``."""
    if node.type != "variable":
        return False
    name = simple_name(node.fqname)
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _is_future_import(node: SymbolNode) -> bool:
    """True for module-level ``from __future__ import ...`` bindings."""
    return (
        node.type == "import" and node.imports is not None and node.imports.module == "__future__"
    )
