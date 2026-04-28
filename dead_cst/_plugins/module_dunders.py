"""Plugin: keep module-level dunder variables and ``__future__`` imports alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .._symbols import SymbolNode
from ._core import GraphOp, PluginContext, mark_entrypoints, simple_name

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
    """

    name: str = "module_dunders"

    def contribute(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for node in ctx.base_nodes():
            if not _is_kept_alive(node):
                continue
            yield from mark_entrypoints(f"{DUNDER_PREFIX}{node.fqname}", node.path, [node])


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
