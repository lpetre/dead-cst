import logging
from pathlib import Path
from typing import Sequence

import networkx as nx
from libcst.metadata import FullRepoManager

from ._fqn import FixedFullyQualifiedNameProvider
from ._plugins import (
    CSTAwareEdgePlugin,
    EdgePlugin,
    FileTextCache,
    PluginContext,
    apply_ops,
)
from ._resolve import resolve_edges, safe_resolve_module, temp_sys_path
from ._resolvers import exported_roots
from ._symbols import SymbolNode, SymbolTrie
from ._visitor import SymbolVisitor

logger = logging.getLogger(__name__)


def order_paths(paths: dict[Path, list[Path]]) -> list[Path]:
    """Topologically sort base paths so dependencies are processed first.

    ``paths`` maps each base directory to the list of other base directories
    it imports from (added to ``sys.path`` while it is processed). The
    returned order ensures that when a base is processed every base it
    depends on has already contributed its symbols to the lookup tables, so
    cross-package import resolution sees them.
    """
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
    """Build a directed reachability graph of every top-level symbol under ``paths``.

    Each ``.py`` file under each base in ``paths`` is parsed with LibCST;
    modules, classes, functions, top-level variables, and module-level imports
    become :class:`SymbolNode` graph nodes. Edges encode "keeps alive"
    relationships:

    * a reference points at its referent,
    * a declaration points at its containing module,
    * a submodule points at its parent package, and
    * synthetic ``unreachable`` nodes own edges into symbols referenced from
      statically-dead suites (``if False:``, ``raise``-only branches, ...) so
      those references don't keep the targets alive.

    Third-party imports are surfaced as synthetic ``[external dist] <name>``
    / ``[external file] <name>`` nodes so callers can audit the
    project's dependency surface (see the ``dependencies`` CLI command).

    Parameters
    ----------
    paths:
        Mapping from base directory to its first-party search-path
        dependencies. For a single-package project, pass ``{root: []}``. For
        a monorepo, list the dependencies so they're added to ``sys.path``
        and resolved as first-party. ``order_paths`` orders the bases.
    plugins:
        Sequence of :class:`EdgePlugin` / :class:`CSTAwareEdgePlugin`
        instances run after analysis. Plugins emit :class:`AddNode`,
        :class:`AddEdge`, and :class:`RemoveEdge` ops; ``AddNode(...,
        entrypoint=True)`` seeds :func:`find_reachable`.
    project_root:
        Project root used by plugins for path-relative matching and for
        locating ``pyproject.toml``. If omitted, inferred as the shortest
        path in ``paths``.

    Returns
    -------
    networkx.DiGraph
        Nodes are :class:`SymbolNode` instances; entrypoint seeds carry
        ``graph.nodes[node]["entrypoint"] = True``.
    """
    symbol_graph = nx.DiGraph()
    base_tries: dict[Path, SymbolTrie] = {}
    export_tries: dict[Path, SymbolTrie] = {}
    base_managers: dict[Path, FullRepoManager] = {}
    all_files: list[Path] = []
    symbol_lookup: SymbolTrie = SymbolTrie()
    for base in order_paths(paths):
        logger.debug("Processing base path: %s", base)
        search_paths = [base] + paths.get(base, [])
        safe_resolve_module.cache_clear()
        with temp_sys_path(search_paths):
            base_tries[base] = current_trie = SymbolTrie()
            export_trie = SymbolTrie()
            export_roots = exported_roots(base)
            import_edges = set()
            files = list(sorted(base.rglob("*.py")))
            mgr = FullRepoManager(base, files, {FixedFullyQualifiedNameProvider})
            base_managers[base] = mgr
            all_files.extend(files)
            for file in files:
                wrapper = mgr.get_metadata_wrapper_for_path(file)
                visitor = SymbolVisitor(file, search_paths)
                wrapper.visit(visitor)
                # A file's decls go into ``export_trie`` only when the file
                # lives under one of ``base``'s exported dirs (or when the
                # base has no export restriction). This is what hides
                # ``tests/`` from dependents in the typical workspace
                # layout: the member analyzes its own ``tests/`` (decls
                # land in ``current_trie`` and the graph), but consumers'
                # lookup tries never see them.
                file_exported = export_roots is None or _under_any(file, export_roots)
                curr = [visitor.trie]
                while curr:
                    node = curr.pop()
                    if node.module is not None:
                        symbol_graph.add_node(node.module)
                        current_trie.add_declaration(node.module)
                        if file_exported:
                            export_trie.add_declaration(node.module)
                    for decls in node.declarations.values():
                        for decl in decls:
                            symbol_graph.add_node(decl)
                            symbol_graph.add_edge(decl, node.module)
                            current_trie.add_declaration(decl)
                            if file_exported:
                                export_trie.add_declaration(decl)
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

                # Synthetic ``unreachable`` nodes for statically-dead suites.
                # They live in the graph as orphan sources -- nothing points
                # at them, so reachability never visits them, but the edges
                # they own surface which symbols are referenced from inside
                # dead branches. Existing top-level decl edges are unchanged.
                for unreachable in visitor.unreachable_nodes:
                    symbol_graph.add_node(unreachable)
                for src, dst in visitor.unreachable_internal_edges:
                    symbol_graph.add_edge(src, dst)

                # collect all the intra module edges
                import_edges = import_edges | visitor.import_edges
                import_edges = import_edges | visitor.unreachable_import_edges

            # add edges to keep __init__.py files alive
            current_trie.add_module_hierarchy_edges(symbol_graph)
            export_tries[base] = export_trie

            # Per-consumer lookup trie: this base's full trie (everything
            # in scope when resolving its own imports) plus each dep's
            # *exported* trie (what the dep ships to consumers). Deps are
            # processed earlier by topological order, so their export
            # tries already exist.
            symbol_lookup = SymbolTrie()
            symbol_lookup.merge(current_trie)
            for dep in paths.get(base, []):
                symbol_lookup.merge(export_tries.get(dep, base_tries[dep]))

            # resolve all the import edges
            for src, dst in resolve_edges(import_edges, symbol_lookup, base):
                symbol_graph.add_edge(src, dst)

    if plugins:
        root = project_root or _infer_project_root(paths)
        ctx = PluginContext(
            graph=symbol_graph,
            symbol_lookup=symbol_lookup,
            paths=paths,
            project_root=root,
            file_cache=FileTextCache(all_files),
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


def _under_any(file: Path, roots: list[Path]) -> bool:
    """True iff ``file`` is equal to or nested under any of ``roots``."""
    f = file.resolve()
    for r in roots:
        if f == r or f.is_relative_to(r):
            return True
    return False


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
    """Count nodes in ``graph`` by ``SymbolNode.type``, optionally restricted by path.

    If ``prefix`` is given, only nodes whose ``path`` is under ``prefix`` are
    counted -- useful for per-base summaries when several packages are
    analysed together. Includes the synthetic ``"synthetic"`` type contributed
    by plugins and third-party-dep markers; the CLI suppresses that key when
    rendering summaries.
    """
    counts = {}
    for node in graph.nodes:
        if prefix and not node.path.is_relative_to(prefix):
            continue
        counts[node.type] = counts.get(node.type, 0) + 1
    return counts
