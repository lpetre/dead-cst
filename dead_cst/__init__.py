"""Dead code analysis for Python using LibCST.

``dead-cst`` builds a symbol-level reachability graph of a Python codebase
and reports (or removes) anything not reachable from a configurable set
of entrypoints.

The top-level package re-exports the names most callers need:
:func:`build_symbol_graph` to run the analysis,
:func:`find_reachable` / :func:`find_kept_alive_by_dead_branches` /
:func:`count_nodes` / :func:`order_paths` to inspect the result,
:func:`remove_code` to apply the codemod, and the graph data types
(:class:`SymbolNode`, :class:`Import`, :class:`NodeFlags`,
:class:`EdgeFlags`).

The deeper public surface lives in focused sub-packages; pull from
those when writing extensions:

* :mod:`dead_cst.graph` -- node and payload data types.
* :mod:`dead_cst.analyze` -- the analysis driver and reachability helpers.
* :mod:`dead_cst.codemod` -- the LibCST-based source rewriter.
* :mod:`dead_cst.cache` -- :class:`~dead_cst.cache.GraphCache` and the
  :func:`~dead_cst.cache.compute_fingerprint` helper for callers that
  share or inspect the per-file cache.
* :mod:`dead_cst.branches` -- the :class:`UnreachableRegionDetector`
  protocol, :class:`DefaultUnreachableRegionDetector`, and the
  truthiness / fold helpers a from-scratch detector needs.
* :mod:`dead_cst.plugins` -- the :class:`EdgePlugin` protocol, the
  :class:`PluginContext` / :class:`ObserveContext` types,
  :class:`GraphOp` value objects, and the synthetic-node prefix
  constants the analyzer uses for non-first-party imports.
* :mod:`dead_cst.resolvers` -- the :class:`PathResolver` protocol,
  builtin resolvers (:class:`VenvResolver`, :class:`PyprojectResolver`,
  :class:`ManualResolver`), and the ``sys.path`` / ``importlib``
  helpers a custom resolver may want to reuse.
* :mod:`dead_cst.contrib` -- extensions targeting specific third-party
  tools: framework plugins (:class:`FastAPIPlugin`, :class:`FlaskPlugin`,
  :class:`ClickPlugin`, :class:`TyperPlugin`, :class:`PytestPlugin`,
  :class:`UnittestPlugin`) and tool-specific resolvers
  (:class:`UvWorkspaceResolver`). All contrib classes are also
  re-exported from ``dead_cst.plugins`` / ``dead_cst.resolvers`` for
  ergonomics.

The :class:`Cacheable` protocol (``name: str`` + ``version: int``) is
the common contract every extension point inherits from -- the
``(name, version)`` pair feeds the per-file cache fingerprint, so
swapping or reconfiguring any of them invalidates stale entries.

See the README for the CLI, the entrypoint and search-path specs, and
the limitations of static analysis. ``dead-cst`` is alpha; APIs, CLI
flags, and output formats may change without notice.
"""

from ._cacheable import Cacheable
from ._version import __version__
from .analyze import (
    build_symbol_graph,
    count_nodes,
    find_kept_alive_by_dead_branches,
    find_reachable,
    order_paths,
)
from .codemod import remove_code
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode

__all__ = [
    "Cacheable",
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolNode",
    "__version__",
    "build_symbol_graph",
    "count_nodes",
    "find_kept_alive_by_dead_branches",
    "find_reachable",
    "order_paths",
    "remove_code",
]
