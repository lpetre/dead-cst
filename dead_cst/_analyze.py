import logging
from pathlib import Path
from typing import Sequence

import networkx as nx
from libcst.metadata import FullRepoManager

from ._fqn import FixedFullyQualifiedNameProvider
from ._plugins import (
    CSTAwareEdgePlugin,
    EdgePlugin,
    PluginContext,
    apply_ops,
)
from ._resolve import resolve_edges, safe_resolve_module, temp_sys_path
from ._symbols import SymbolNode, SymbolTrie
from ._visitor import SymbolVisitor

logger = logging.getLogger(__name__)


def order_paths(paths: dict[Path, list[Path]]) -> list[Path]:
    path_order = nx.DiGraph()
    for base, search_paths in paths.items():
        path_order.add_node(base)
        for sp in search_paths:
            path_order.add_edge(sp, base)
    return list(nx.topological_sort(path_order))


def build_symbol_graph(
    paths: dict[Path, list[Path]],
    *,
    plugins: Sequence[EdgePlugin | CSTAwareEdgePlugin] = (),
    project_root: Path | None = None,
) -> nx.DiGraph:
    symbol_graph = nx.DiGraph()
    base_tries: dict[Path, SymbolTrie] = {}
    base_managers: dict[Path, FullRepoManager] = {}
    symbol_lookup: SymbolTrie = SymbolTrie()
    for base in order_paths(paths):
        logger.debug("Processing base path: %s", base)
        search_paths = [base] + paths.get(base, [])
        safe_resolve_module.cache_clear()
        with temp_sys_path(search_paths):
            base_tries[base] = current_trie = SymbolTrie()
            import_edges = set()
            files = list(sorted(base.rglob("*.py")))
            mgr = FullRepoManager(base, files, {FixedFullyQualifiedNameProvider})
            base_managers[base] = mgr
            for file in files:
                wrapper = mgr.get_metadata_wrapper_for_path(file)
                visitor = SymbolVisitor(file, search_paths)
                wrapper.visit(visitor)
                curr = [visitor.trie]
                while curr:
                    node = curr.pop()
                    if node.module is not None:
                        symbol_graph.add_node(node.module)
                        current_trie.add_declaration(node.module)
                    for decls in node.declarations.values():
                        for decl in decls:
                            symbol_graph.add_node(decl)
                            symbol_graph.add_edge(decl, node.module)
                            current_trie.add_declaration(decl)
                    # Shadowed decls still belong to this module: emit them
                    # so the graph stays well-formed (every decl has a
                    # parent-module edge). They aren't re-added to
                    # ``current_trie`` -- only live-at-exit decls should
                    # win project-wide lookups.
                    for decl in node.shadowed:
                        symbol_graph.add_node(decl)
                        symbol_graph.add_edge(decl, node.module)
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

    if plugins:
        root = project_root or _infer_project_root(paths)
        ctx = PluginContext(
            graph=symbol_graph,
            symbol_lookup=symbol_lookup,
            paths=paths,
            project_root=root,
        )
        for plugin in plugins:
            if isinstance(plugin, CSTAwareEdgePlugin):
                ops = list(plugin.contribute(ctx, base_managers))
            elif isinstance(plugin, EdgePlugin):
                ops = list(plugin.contribute(ctx))
            else:
                raise TypeError(f"Plugin {plugin!r} does not satisfy EdgePlugin protocol")
            # Materialize before applying so plugins can iterate ctx.graph.nodes
            # without tripping "dictionary changed size during iteration".
            apply_ops(symbol_graph, ops)

    return symbol_graph


def _infer_project_root(paths: dict[Path, list[Path]]) -> Path:
    bases = list(paths)
    if not bases:
        return Path.cwd()
    return min(bases, key=lambda p: len(p.parts))


def find_reachable(graph: nx.DiGraph) -> set[SymbolNode]:
    """BFS forward from every node tagged as an entrypoint by a plugin.

    Plugins mark seeds by setting ``graph.nodes[node]["entrypoint"] = True``
    (see :func:`dead_cst._plugins.apply_ops`). There is no longer any
    built-in matching against file paths or FQNs -- that lives in
    :class:`ExplicitEntrypointPlugin`.
    """
    visited: set[SymbolNode] = set()
    stack = [n for n, attrs in graph.nodes(data=True) if attrs.get("entrypoint")]
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
