"""Bridge from `dead_cst_ty_native.NativeGraph` to `SymbolGraph`.

The native crate now produces one project-wide envelope per
`Project.build()` call. `materialize` is the thin layer that converts
that envelope into a `dead_cst._graphstore.SymbolGraph` of equivalent
shape, so test fixtures can assert on edges in the same vocabulary as
the libcst pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import dead_cst_ty_native as native
from libcst.metadata import CodePosition, CodeRange

from dead_cst._graphstore import SymbolGraph
from dead_cst.graph import EdgeFlags, Import, NodeFlags, SymbolNode


def _to_symbol_node(n: native.NativeNode) -> SymbolNode:
    return SymbolNode(
        fqname=n.fqname,
        type=cast("SymbolNode.type", n.kind),
        path=Path(n.path),
        position=CodeRange(
            CodePosition(n.start_line, n.start_column),
            CodePosition(n.end_line, n.end_column),
        ),
        imports=_to_import(n.imports),
        flags=NodeFlags(n.flags),
    )


def _to_import(native_import: native.Import | None) -> Import | None:
    if native_import is None:
        return None
    return Import(
        module=native_import.module,
        decl=native_import.decl,
        star=native_import.star,
    )


def materialize(graph: native.NativeGraph) -> SymbolGraph:
    """Convert a project-wide `NativeGraph` into a fresh `SymbolGraph`."""
    out = SymbolGraph()
    symbol_nodes = [_to_symbol_node(n) for n in graph.nodes]
    for sn in symbol_nodes:
        out.add(sn)
    for src, dst, flags in graph.edges:
        out.add_edge(symbol_nodes[src], symbol_nodes[dst], EdgeFlags(flags))
    return out
