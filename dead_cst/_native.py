"""Native (rust) backend bridge.

The rust crate (``dead_cst_ty_native``) builds the project graph
end-to-end using ty's ``SemanticIndex`` instead of libcst's per-file
visitor + cross-file edge stitcher. This module bridges the
rust-shaped ``NativeGraph`` envelope back into the
:class:`SymbolGraph` shape every downstream consumer (codemod, CLI,
plugin queries) speaks.

The rust path replaces every libcst stage that used to build the
graph — visitor, flow-sensitive shadowing, edge stitcher, per-file
SQLite cache, per-package contribution merge. Everything downstream
of the materialized :class:`SymbolGraph` (the codemod's source
rewriter, ``why-alive`` traversals, plugin reachability queries) is
backend-agnostic and keeps working unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from libcst.metadata import CodePosition, CodeRange

from ._graphstore import SymbolGraph
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode

if TYPE_CHECKING:
    import dead_cst_ty_native as native

# Most nodes carry no flags and most edges have flags=0; reusing the
# zero singletons avoids ~5k IntFlag constructor calls per warm build
# on ``dead_cst`` itself.
_NO_NODE_FLAGS = NodeFlags(0)
_NO_EDGE_FLAGS = EdgeFlags(0)


def materialize_project(
    project_root: Path,
    plugins: Sequence[object] = (),
    src_roots: Sequence[Path] = (),
) -> SymbolGraph:
    """Materialize ``project_root`` end-to-end via the rust backend.

    Builds a :class:`native.ProjectContext` rooted at ``project_root``,
    registers each plugin's ``run(ctx)`` callback, calls
    :meth:`materialize`, and bridges the resulting :class:`NativeGraph`
    into a :class:`SymbolGraph`. Plugins that don't implement the
    rust ``run(ctx)`` protocol are silently skipped.
    """
    import dead_cst_ty_native as native

    kwargs = {}
    if src_roots:
        kwargs["src_roots"] = [str(p) for p in src_roots]
    ctx = native.ProjectContext(str(project_root), **kwargs)
    for plugin in plugins:
        if hasattr(plugin, "run"):
            ctx.add_plugin(plugin)
    return _bridge(ctx.materialize())


def _to_import(native_import: "native.Import | None") -> Import | None:
    if native_import is None:
        return None
    return Import(
        module=native_import.module,
        decl=native_import.decl,
        star=native_import.star,
    )


def _bridge(graph: "native.NativeGraph") -> SymbolGraph:
    """Convert a project-wide :class:`NativeGraph` into a fresh :class:`SymbolGraph`."""
    out = SymbolGraph()
    # Per-build cache: ~1k nodes typically span ~40 unique paths, so
    # interning here saves ~10ms of pathlib.Path construction +
    # hashing per warm build on ``dead_cst`` itself.
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
                type=n.kind,
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
