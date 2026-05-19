"""Dead code analysis for Python.

``dead-cst`` builds a symbol-level reachability graph of a Python
codebase via a rust-backed extension (:mod:`dead_cst._native`, which
uses ty's ``SemanticIndex``) and reports (or removes) anything not
reachable from a configurable set of entrypoints.

The top-level package re-exports the names most callers need:
:class:`Analysis` is the lazy entry point and the graph data types
(:class:`SymbolNode`, :class:`Import`, :class:`NodeFlags`,
:class:`EdgeFlags`) describe the materialized graph.

The deeper public surface lives in focused sub-packages:

* :mod:`dead_cst.graph` -- node and edge data types.
* :mod:`dead_cst.analyze` -- :class:`Analysis`.
* :mod:`dead_cst.codemod` -- the LibCST-based source rewriter.
* :mod:`dead_cst.plugins` -- the synthetic-node prefix constants and
  every built-in plugin.
* :mod:`dead_cst.resolvers` -- the :class:`PathResolver` protocol,
  :class:`ManualResolver`, :class:`UvResolver`.
* :mod:`dead_cst.contrib` -- third-party-aware extensions.

``dead-cst`` is alpha; APIs, CLI flags, and output formats may change.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .analyze import Analysis
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode

try:
    __version__: str = _pkg_version("dead-cst")
except PackageNotFoundError:  # editable / source-tree fallback
    __version__ = "0.0.0+unknown"

__all__ = [
    "Analysis",
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "SymbolNode",
    "__version__",
]
