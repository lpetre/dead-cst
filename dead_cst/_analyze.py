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
    import_edges = []
    for base, search_paths in paths.items():
        with temp_sys_path(search_paths):
            paths = list(base.rglob("*.py"))
            mgr = FullRepoManager(base, paths, {FullyQualifiedNameProvider})
            for file in paths:
                print(file)
                wrapper = mgr.get_metadata_wrapper_for_path(file)
                decl_visitor = SymbolVisitor(file, search_paths)
                wrapper.visit(decl_visitor)
                for decl_node in decl_visitor.decls:
                    symbol_graph.add_node(decl_node)
                for src, dst in decl_visitor.internal_edges:
                    symbol_graph.add_edge(src, dst)

                # collect all the intra module edges
                import_edges.extend(decl_visitor.import_edges)

    # now resolve all the import edges
    fqname_to_node = {n.fqname: n for n in symbol_graph.nodes}
    for src, fqname in import_edges:
        dst = fqname_to_node.get(fqname)
        if not dst:
            print(f"Failed to resolve import edge: {src.fqname} -> {fqname}")
            continue
        symbol_graph.add_edge(src, dst)

    return symbol_graph


def find_reachable(graph: nx.DiGraph, root: Path, entrypoints: list[Path | re.Pattern]) -> Set[str]:
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
