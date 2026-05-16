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

# Most nodes carry no flags and most edges have flags=0; reusing the
# zero singleton avoids ~5k `EdgeFlags(0)` constructor calls per warm
# build on `dead_cst` itself.
_NO_NODE_FLAGS = NodeFlags(0)
_NO_EDGE_FLAGS = EdgeFlags(0)


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
    # Per-build cache: ~1k nodes typically span ~40 unique paths, so
    # interning here saves ~10ms of pathlib.Path construction +
    # hashing per warm build on `dead_cst` itself.
    path_cache: dict[str, Path] = {}
    symbol_nodes: list[SymbolNode] = []
    for n in graph.nodes:
        path = path_cache.get(n.path)
        if path is None:
            path = Path(n.path)
            path_cache[n.path] = path
        flags = _NO_NODE_FLAGS if n.flags == 0 else NodeFlags(n.flags)
        symbol_nodes.append(
            SymbolNode(
                fqname=n.fqname,
                type=cast("SymbolNode.type", n.kind),
                path=path,
                position=CodeRange(
                    CodePosition(n.start_line, n.start_column),
                    CodePosition(n.end_line, n.end_column),
                ),
                imports=_to_import(n.imports),
                flags=flags,
            )
        )
    for sn in symbol_nodes:
        out.add(sn)
    for src, dst, flags in graph.edges:
        edge_flags = _NO_EDGE_FLAGS if flags == 0 else EdgeFlags(flags)
        out.add_edge(symbol_nodes[src], symbol_nodes[dst], edge_flags)
    return out
