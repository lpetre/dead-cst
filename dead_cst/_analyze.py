import re
from pathlib import Path

import networkx as nx
from libcst.metadata import FullRepoManager, FullyQualifiedNameProvider

from ._resolve import resolve_edges, safe_resolve_module, temp_sys_path
from ._symbols import SymbolNode, SymbolTrie
from ._visitor import SymbolVisitor


def order_paths(paths: dict[Path, list[Path]]) -> list[Path]:
    path_order = nx.DiGraph()
    for base, search_paths in paths.items():
        path_order.add_node(base)
        for sp in search_paths:
            path_order.add_edge(sp, base)
    return list(nx.topological_sort(path_order))


def build_symbol_graph(paths: dict[Path, list[Path]]) -> nx.DiGraph:
    symbol_graph = nx.DiGraph()
    base_tries = dict()
    for base in order_paths(paths):
        search_paths = [base] + paths.get(base, [])
        safe_resolve_module.cache_clear()
        with temp_sys_path(search_paths):
            base_tries[base] = current_trie = SymbolTrie()
            import_edges = set()
            files = list(sorted(base.rglob("*.py")))
            mgr = FullRepoManager(base, files, {FullyQualifiedNameProvider})
            for file in files:
                # if str(file) != "/home/lpetre_midjourney_com/dev/src/github.com/midjourney/image-generation/ml/src/kdj_v7/omini/model.py":
                #     continue
                # if str(file) != "/home/lpetre_midjourney_com/dev/src/github.com/midjourney/image-generation/ml/src/kdpt_model.py":
                #     continue
                # if "libs/kdj-minimal/v6/nntree/" not in str(file):
                #     continue
                print(file)
                wrapper = mgr.get_metadata_wrapper_for_path(file)
                visitor = SymbolVisitor(file, search_paths)
                wrapper.visit(visitor)
                curr = [visitor.trie]
                while curr:
                    node = curr.pop()
                    if node.module is not None:
                        symbol_graph.add_node(node.module)
                        current_trie.add_declaration(node.module)
                    for decl in node.declarations.values():
                        symbol_graph.add_node(decl)
                        symbol_graph.add_edge(decl, node.module)
                        current_trie.add_declaration(decl)
                    curr.extend(node.children.values())

                for src, dst in visitor.internal_edges:
                    symbol_graph.add_edge(src, dst)

                # collect all the intra module edges
                import_edges = import_edges | visitor.import_edges

            # add edges to keep __init__.py files alive
            current_trie.add_module_hierarchy_edges(symbol_graph)

            # now merge all the lookup tries
            symbol_lookup = SymbolTrie()
            for sp in search_paths:
                symbol_lookup.merge(base_tries[sp])

            # resolve all the import edges
            for src, dst in resolve_edges(import_edges, symbol_lookup):
                symbol_graph.add_edge(src, dst)

    return symbol_graph


def find_reachable(
    graph: nx.DiGraph, root: Path, entrypoints: list[Path | re.Pattern]
) -> set[SymbolNode]:
    visited = set()

    def _is_entrypoint(sym: SymbolNode) -> bool:
        rel = str(sym.path.relative_to(root))
        for e in entrypoints:
            if isinstance(e, str):
                if e == rel or e == sym.fqname:
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
