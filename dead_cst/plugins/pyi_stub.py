"""Plugin: anchor ``.pyi`` stub declarations to their ``.py`` twins.

Stub files are inert at runtime, so on their own every declaration in
a ``.pyi`` looks unreferenced and the codemod would delete it. This
plugin emits ``runtime_decl -> stub_decl`` edges by simple-name match
within each ``.pyi`` module (FQN ``<runtime>.__pyi__`` -- see
:class:`~dead_cst._fqn.FixedFullyQualifiedNameProvider`) so a stub's
lifetime tracks its ``.py`` twin. Orphan stubs (no matching ``.py``)
are left to whatever reachability the rest of the graph supplies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .._fqn import PYI_FQN_SEGMENT
from ._core import (
    AddEdge,
    GraphOp,
    ObserveContext,
    PluginContext,
    simple_name,
)

if TYPE_CHECKING:
    from ..graph import VisitorPayload


_PYI_SUFFIX = f".{PYI_FQN_SEGMENT}"


@dataclass
class PyiStubPlugin:
    """Link every ``.pyi`` declaration to its same-named ``.py`` twin."""

    name: str = "pyi_stub_link"
    version: int = 1778000000

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        for node in ctx.package_nodes():
            if node.type in ("module", "synthetic", "import"):
                continue
            module_fqname = node.fqname.rpartition(".")[0]
            if not module_fqname.endswith(_PYI_SUFFIX):
                continue
            runtime_fqname = module_fqname.rpartition(".")[0]
            target_fqname = f"{runtime_fqname}.{simple_name(node.fqname)}"
            for runtime in ctx.find_declarations(target_fqname):
                yield AddEdge(runtime, node)
