"""Plugin: anchor ``.pyi`` stub declarations to their ``.py`` twins.

Stub files are inert at runtime -- they exist for type checkers, not for
the interpreter -- so on their own, every declaration in a ``.pyi`` looks
unreferenced and the analyzer would mark them dead. That's the wrong
answer for a project that ships hand-written stubs alongside its
runtime modules: deleting the ``.pyi`` definitions during a codemod
silently drops type information the user wants to keep.

This plugin runs once per base in :meth:`finalize` and wires every
``.pyi`` decl to its same-named ``.py`` sibling. The ``.pyi`` module's
FQN is ``<runtime_module>.__pyi__`` (stamped by
:class:`~dead_cst._fqn.FixedFullyQualifiedNameProvider` so the runtime
and stub modules don't collide in the symbol trie); we strip the
``__pyi__`` segment to find the runtime module and emit
``runtime_decl -> stub_decl`` edges by simple name. After the link, a
stub's lifetime tracks the runtime decl's: keep the runtime decl alive
and the stub stays alive, mark the runtime decl dead and the stub
follows it into the codemod's deletion set.

Orphan stubs -- ``.pyi`` decls with no matching ``.py`` twin -- are
left alone. Their reachability is whatever the user's entrypoints +
edges produce on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .._fqn import PYI_FQN_SEGMENT
from ..graph import SymbolNode
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
    """Link every ``.pyi`` declaration to its same-named ``.py`` twin.

    Pure :meth:`finalize` work -- there is nothing the per-file observe
    pass can do because cross-module name resolution depends on the
    assembled per-base graph + trie. Bumps :attr:`version` whenever the
    edge-emission rule changes (folding additional decl kinds in,
    relaxing the simple-name match, ...) so cached ``observe`` outputs
    that predate the rule change are not served stale.
    """

    name: str = "pyi_stub_link"
    version: int = 1778000000

    def observe(self, ctx: ObserveContext) -> "VisitorPayload | None":
        return None

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        # Index every node under this base by ``(path, simple_name)`` so
        # we can match a stub's ``f`` against the runtime module's ``f``
        # without re-scanning the graph for each link. ``base_nodes``
        # already filters to this base.
        runtime_decls: dict[tuple[str, str], list[SymbolNode]] = {}
        stub_modules: dict[str, SymbolNode] = {}
        stub_decls: dict[str, list[SymbolNode]] = {}

        for node in ctx.base_nodes():
            if node.type == "module":
                if node.fqname.endswith(_PYI_SUFFIX):
                    stub_modules[node.fqname] = node
                continue
            if node.type in ("synthetic", "import"):
                continue

            module_fqname = node.fqname.rpartition(".")[0]
            if module_fqname.endswith(_PYI_SUFFIX):
                stub_decls.setdefault(module_fqname, []).append(node)
            else:
                key = (module_fqname, simple_name(node.fqname))
                runtime_decls.setdefault(key, []).append(node)

        for stub_module_fqname, stubs in stub_decls.items():
            if stub_module_fqname not in stub_modules:
                # Orphan stub module -- the discovery pass found ``.pyi``
                # files but no corresponding ``.py`` module exposes the
                # FQN we'd link against. Without the runtime sibling
                # there's nothing to anchor to; leave the stubs to
                # whatever reachability the rest of the graph supplies.
                continue
            runtime_fqname = stub_module_fqname[: -len(_PYI_SUFFIX)]
            for stub in stubs:
                key = (runtime_fqname, simple_name(stub.fqname))
                for runtime in runtime_decls.get(key, ()):
                    yield AddEdge(runtime, stub)
