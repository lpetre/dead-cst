"""Dead code analysis for Python using LibCST.

`dead-cst` builds a symbol-level reachability graph of a Python codebase and
reports (or removes) anything not reachable from a configurable set of
entrypoints.

The public API has three layers:

* :func:`build_symbol_graph` parses every ``.py`` file under each base in a
  ``{base: [dep_paths]}`` map and returns a :class:`networkx.DiGraph` of
  :class:`SymbolNode` instances. Edges encode "keeps alive" relationships.
* Edge plugins (:class:`MainBlockPlugin`, :class:`ProjectScriptsPlugin`,
  :class:`ExplicitEntrypointPlugin`, :class:`ModuleDundersPlugin`,
  :class:`PytestPlugin`, :class:`FastAPIPlugin`, :class:`TyperPlugin`)
  extend the graph with edges that pure CST analysis can't infer --
  entry points, framework conventions, dynamic dispatch. Custom plugins
  implement the :class:`EdgePlugin` or :class:`CSTAwareEdgePlugin`
  protocol.
* Path resolvers (:class:`VenvResolver`, :class:`PyprojectResolver`,
  :class:`UvWorkspaceResolver`) discover the ``{base: [dep_paths]}`` map
  itself from a project root, so callers don't have to hand-build it.

After plugins run, :func:`find_reachable` walks successors from every node
tagged with ``entrypoint=True`` and returns the live set;
:func:`remove_code` rewrites source files in place to delete the rest via
a LibCST codemod that preserves surrounding formatting.

See the README for the CLI, the entrypoint and search-path specs, and the
limitations of static analysis.
"""

from ._analyze import build_symbol_graph, count_nodes, find_reachable, order_paths
from ._codemod import remove_code
from ._plugins import (
    AddEdge,
    AddNode,
    BUILTIN_PLUGINS,
    CSTAwareEdgePlugin,
    EdgePlugin,
    ExplicitEntrypointPlugin,
    FastAPIPlugin,
    FileTextCache,
    GraphOp,
    MainBlockPlugin,
    ModuleDundersPlugin,
    PluginContext,
    ProjectScriptsPlugin,
    PytestPlugin,
    RemoveEdge,
    TyperPlugin,
    load_plugin,
)
from ._resolvers import (
    BUILTIN_RESOLVERS,
    PathResolver,
    PyprojectResolver,
    UvWorkspaceResolver,
    VenvResolver,
    load_resolver,
    merge_paths,
)
from ._version import __version__

__all__ = [
    "__version__",
    "AddEdge",
    "AddNode",
    "BUILTIN_PLUGINS",
    "BUILTIN_RESOLVERS",
    "CSTAwareEdgePlugin",
    "EdgePlugin",
    "ExplicitEntrypointPlugin",
    "FastAPIPlugin",
    "FileTextCache",
    "GraphOp",
    "MainBlockPlugin",
    "ModuleDundersPlugin",
    "PathResolver",
    "PluginContext",
    "ProjectScriptsPlugin",
    "PyprojectResolver",
    "PytestPlugin",
    "RemoveEdge",
    "TyperPlugin",
    "UvWorkspaceResolver",
    "VenvResolver",
    "build_symbol_graph",
    "count_nodes",
    "find_reachable",
    "load_plugin",
    "load_resolver",
    "merge_paths",
    "order_paths",
    "remove_code",
]
