"""Bridge from ``dead_cst_ty_native.NativeGraph`` to ``SymbolGraph``.

Splits cleanly into three layers:

* :func:`materialize` -- one-shot conversion of a single envelope into
  a fresh ``SymbolGraph``. Useful for single-file probing.
* :func:`accumulate` -- merge an envelope into an existing
  ``SymbolGraph``. Returns the local-index -> SymbolNode map so
  callers (and plugins) can look up the materialized SymbolNode that
  the native side produced at index ``i``. Identity-based dedup in
  ``SymbolGraph.add`` means a node already present (because an
  earlier package contributed it) collapses naturally.
* :func:`build_project_graph` -- the per-package orchestrator. Walks
  packages in toposort order; for each one, builds the native graph,
  accumulates it, then calls every registered plugin with a
  :class:`PackageContext` so it can add more nodes / edges.

Plugins implement :class:`Plugin`. They receive a context that exposes
the ty-backed project, the just-accumulated native envelope, the
local-index -> SymbolNode map, and helpers (`add_node`, `add_edge`)
that go straight to the shared ``SymbolGraph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, cast

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
        speculative=native_import.speculative,
    )


def materialize(graph: native.NativeGraph) -> SymbolGraph:
    """Convert a single ``NativeGraph`` into a fresh ``SymbolGraph``."""
    out = SymbolGraph()
    accumulate(out, graph)
    return out


def accumulate(graph: SymbolGraph, native_graph: native.NativeGraph) -> list[SymbolNode]:
    """Merge ``native_graph`` into ``graph``.

    Returns a list aligned with ``native_graph.nodes`` -- entry ``i``
    is the ``SymbolNode`` that index ``i`` in the native envelope
    maps to. Nodes whose identity is already present in ``graph``
    (because an earlier package contributed them) are reused, not
    re-inserted.
    """
    symbol_nodes: list[SymbolNode] = []
    for n in native_graph.nodes:
        sn = _to_symbol_node(n)
        graph.add(sn)
        symbol_nodes.append(sn)
    for src, dst, flags in native_graph.edges:
        graph.add_edge(symbol_nodes[src], symbol_nodes[dst], EdgeFlags(flags))
    return symbol_nodes


@dataclass(slots=True)
class PackageContext:
    """One package's slice of state, passed to plugins.

    ``package_graph`` is the native envelope the visitor produced for
    this package. ``symbol_nodes[i]`` is the SymbolNode that the
    accumulator mapped ``package_graph.nodes[i]`` to. ``graph`` is
    the shared SymbolGraph -- plugins mutate it directly through the
    helpers on this context.
    """

    project: native.Project
    package_name: str
    package_graph: native.NativeGraph
    graph: SymbolGraph
    symbol_nodes: list[SymbolNode]

    def add_node(self, node: SymbolNode) -> None:
        self.graph.add(node)

    def add_edge(
        self,
        src: SymbolNode,
        dst: SymbolNode,
        flags: EdgeFlags = EdgeFlags.NONE,
    ) -> None:
        self.graph.add_edge(src, dst, flags)

    def symbol_node_for(self, native_node: native.NativeNode) -> SymbolNode:
        """Look up the SymbolNode that the accumulator made for ``native_node``.

        Linear scan; only convenient for tests / sparse use. Real
        plugins should index ``self.symbol_nodes`` directly with the
        local index they already have.
        """
        for i, n in enumerate(self.package_graph.nodes):
            if n is native_node:
                return self.symbol_nodes[i]
        raise ValueError(f"node not in package_graph: {native_node!r}")


class Plugin(Protocol):
    """Per-package plugin hook.

    Plugins run once per package, *after* the visitor's native build
    has been accumulated into the shared ``SymbolGraph``. Use
    ``ctx.add_node`` / ``ctx.add_edge`` to mutate the graph;
    inspect ``ctx.package_graph`` / ``ctx.symbol_nodes`` to read
    what the visitor produced for this package.
    """

    name: str
    version: int

    def contribute(self, ctx: PackageContext) -> None: ...


@dataclass(slots=True)
class BuildReport:
    """Result of a project-wide build.

    ``graph`` is the assembled symbol graph. ``order`` is the
    toposort the project used (mostly useful for tests / debugging).
    """

    graph: SymbolGraph
    order: list[str] = field(default_factory=list)


def build_project_graph(
    project: native.Project,
    plugins: Iterable[Plugin] = (),
) -> BuildReport:
    """Walk packages in toposort order, accumulating into one graph."""
    plugins = list(plugins)
    graph = SymbolGraph()
    order = project.package_order()
    for pkg_name in order:
        pkg_graph = project.build_package_graph(pkg_name)
        symbol_nodes = accumulate(graph, pkg_graph)
        ctx = PackageContext(
            project=project,
            package_name=pkg_name,
            package_graph=pkg_graph,
            graph=graph,
            symbol_nodes=symbol_nodes,
        )
        for plugin in plugins:
            plugin.contribute(ctx)
    return BuildReport(graph=graph, order=order)
