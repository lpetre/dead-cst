"""Bridge from ``dead_cst_ty_native.NativeGraph`` to ``SymbolGraph``.

The Rust side builds a `NativeGraph` envelope (primitive fields,
indices into a `nodes` list, no Python types). Python materializes
that envelope into a real `dead_cst._graphstore.SymbolGraph`
(rustworkx `PyDiGraph` + `SymbolNode -> int` map) here.

Lives under ``tests/prototype/`` rather than inside ``dead_cst/``
because ``dead_cst`` itself does not depend on the experimental
native module today.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from libcst.metadata import CodePosition, CodeRange

from dead_cst._graphstore import SymbolGraph
from dead_cst.graph import EdgeFlags, NodeFlags, SymbolNode

import dead_cst_ty_native as native


def materialize(graph: native.NativeGraph) -> SymbolGraph:
    """Turn a ``NativeGraph`` into a real ``SymbolGraph``."""
    file_path = Path(graph.file_path)
    module_fqname = graph.module_fqname

    out = SymbolGraph()
    symbol_nodes: list[SymbolNode] = []
    for n in graph.nodes:
        if n.kind == "module":
            fqname = module_fqname
        else:
            fqname = f"{module_fqname}.{n.local_name}"
        sn = SymbolNode(
            fqname=fqname,
            type=cast("SymbolNode.type", n.kind),
            path=file_path,
            position=CodeRange(
                CodePosition(n.start_line, n.start_column),
                CodePosition(n.end_line, n.end_column),
            ),
            flags=NodeFlags(n.flags),
        )
        out.add(sn)
        symbol_nodes.append(sn)

    for src, dst, flags in graph.edges:
        out.add_edge(symbol_nodes[src], symbol_nodes[dst], EdgeFlags(flags))

    return out
