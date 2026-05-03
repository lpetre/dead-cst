"""Dead code analysis for Python using LibCST.

`dead-cst` builds a symbol-level reachability graph of a Python codebase and
reports (or removes) anything not reachable from a configurable set of
entrypoints.

The public API has three layers:

* :func:`build_symbol_graph` parses every ``.py`` file under each base in a
  ``{base: [dep_paths]}`` map and returns a :class:`networkx.MultiDiGraph` of
  :class:`SymbolNode` instances. Edges encode "keeps alive" relationships
  and carry a :class:`EdgeFlags` ``flags`` attribute. References from
  inside statically-dead suites are flagged ``DEAD_BRANCH``.
* Edge plugins (:class:`MainBlockPlugin`, :class:`ProjectScriptsPlugin`,
  :class:`ExplicitEntrypointPlugin`, :class:`ModuleDundersPlugin`,
  :class:`PytestPlugin`, :class:`UnittestPlugin`, :class:`FastAPIPlugin`,
  :class:`FlaskPlugin`, :class:`TyperPlugin`, :class:`ClickPlugin`)
  extend the graph with edges that pure CST analysis can't infer --
  entry points, framework conventions, dynamic dispatch. Custom
  plugins implement the :class:`EdgePlugin` protocol.
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

from ._analyze import (
    build_symbol_graph,
    count_nodes,
    find_kept_alive_by_dead_branches,
    find_reachable,
    order_paths,
)
from ._branches import (
    DefaultUnreachableRegionDetector,
    UnreachableRegionDetector,
)
from ._cacheable import Cacheable
from ._codemod import remove_code
from ._symbols import EdgeFlags, NodeFlags
from ._plugins import (
    AddEdge,
    AddNode,
    BUILTIN_PLUGINS,
    ClickPlugin,
    DecoratedDeclPlugin,
    EdgePlugin,
    ExplicitEntrypointPlugin,
    FastAPIPlugin,
    FlaskPlugin,
    GraphOp,
    InitSubclassPlugin,
    LiteralListPlugin,
    MainBlockPlugin,
    ModuleDundersPlugin,
    ObserveContext,
    PluginContext,
    ProjectScriptsPlugin,
    PytestPlugin,
    RemoveEdge,
    TyperPlugin,
    UnittestPlugin,
    entrypoint_payload,
    load_plugin,
    mark_entrypoints,
    synthetic_node,
)
from ._resolvers import (
    BUILTIN_RESOLVERS,
    ManualResolver,
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
    "Cacheable",
    "ClickPlugin",
    "DecoratedDeclPlugin",
    "DefaultUnreachableRegionDetector",
    "EdgeFlags",
    "EdgePlugin",
    "ExplicitEntrypointPlugin",
    "FastAPIPlugin",
    "FlaskPlugin",
    "GraphOp",
    "InitSubclassPlugin",
    "LiteralListPlugin",
    "MainBlockPlugin",
    "ManualResolver",
    "ModuleDundersPlugin",
    "NodeFlags",
    "ObserveContext",
    "PathResolver",
    "PluginContext",
    "ProjectScriptsPlugin",
    "PyprojectResolver",
    "PytestPlugin",
    "RemoveEdge",
    "TyperPlugin",
    "UnittestPlugin",
    "UnreachableRegionDetector",
    "UvWorkspaceResolver",
    "VenvResolver",
    "build_symbol_graph",
    "count_nodes",
    "entrypoint_payload",
    "find_kept_alive_by_dead_branches",
    "find_reachable",
    "load_plugin",
    "load_resolver",
    "mark_entrypoints",
    "merge_paths",
    "order_paths",
    "remove_code",
    "synthetic_node",
]
