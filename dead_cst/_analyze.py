import contextlib
import re
import sys
from pathlib import Path
from typing import Set

import networkx as nx
from libcst.metadata import FullRepoManager, FullyQualifiedNameProvider

from ._visitor import SymbolNode, SymbolVisitor


@contextlib.contextmanager
def temp_sys_path(paths: list[Path]):
    old = list(sys.path)
    seen = set(old)
    sys.path = [str(p) for p in paths if str(p) not in seen] + sys.path
    try:
        yield
    finally:
        sys.path = old


def build_symbol_graph(paths: dict[Path, list[Path]]) -> nx.DiGraph:
    symbol_graph = nx.DiGraph()
    import_edges = set()
    for base, search_paths in paths.items():
        with temp_sys_path(search_paths):
            paths = list(base.rglob("*.py"))
            mgr = FullRepoManager(base, paths, {FullyQualifiedNameProvider})
            for file in paths:
                wrapper = mgr.get_metadata_wrapper_for_path(file)
                visitor = SymbolVisitor(file, search_paths)
                wrapper.visit(visitor)
                for decl_node in visitor.decls:
                    symbol_graph.add_node(decl_node)
                for src, dst in visitor.internal_edges:
                    symbol_graph.add_edge(src, dst)

                # collect all the intra module edges
                import_edges = import_edges | visitor.import_edges

    # now resolve all the import edges
    edge_lookup = {(n.fqname, n.path): n for n in symbol_graph.nodes}
    for edge in import_edges:
        if not isinstance(edge.dst_path, Path):
            continue

        if dst := edge_lookup.get((edge.dst_fqname, edge.dst_path)):
            symbol_graph.add_edge(edge.src, dst)
        else:
            print(f"Failed to resolve import edge: {edge}")

    return symbol_graph


def find_reachable(
    graph: nx.DiGraph, root: Path, entrypoints: list[Path | re.Pattern]
) -> Set[SymbolNode]:
    visited = set()

    def _is_entrypoint(sym: SymbolNode) -> bool:
        rel = str(sym.path.relative_to(root))
        for e in entrypoints:
            if isinstance(e, str):
                if e == rel:
                    return True
            elif isinstance(e, re.Pattern):
                if e.match(rel):
                    return True
        return False

    stack = [e for e in graph.nodes if _is_entrypoint(e)]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph.successors(node))

    return visited


def count_nodes(graph: nx.DiGraph, prefix: Path | None) -> dict[str, int]:
    counts = {}
    for node in graph.nodes:
        if prefix and not node.path.is_relative_to(prefix):
            continue
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts
