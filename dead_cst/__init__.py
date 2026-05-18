"""Dead code analysis for Python.

``dead-cst`` builds a symbol-level reachability graph of a Python
codebase via the rust-backed :mod:`dead_cst_ty_native` crate (which
uses ty's ``SemanticIndex``) and reports (or removes) anything not
reachable from a configurable set of entrypoints.

The top-level package re-exports the names most callers need:
:class:`Analysis` is the lazy entry point, :class:`PackageView` is the
per-package query view, and the graph data types
(:class:`SymbolNode`, :class:`Import`, :class:`NodeFlags`,
:class:`EdgeFlags`) describe the materialized graph.

The deeper public surface lives in focused sub-packages:

* :mod:`dead_cst.graph` -- node and edge data types.
* :mod:`dead_cst.analyze` -- :class:`Analysis` and :class:`PackageView`.
* :mod:`dead_cst.codemod` -- the LibCST-based source rewriter.
* :mod:`dead_cst.plugins` -- the synthetic-node prefix constants and
  every built-in plugin.
* :mod:`dead_cst.resolvers` -- the :class:`PathResolver` protocol,
  :class:`ManualResolver`, :class:`UvResolver`.
* :mod:`dead_cst.contrib` -- third-party-aware extensions.

``dead-cst`` is alpha; APIs, CLI flags, and output formats may change.
"""

from ._version import __version__
from .analyze import Analysis, PackageView
from .graph import EdgeFlags, Import, NodeFlags, SymbolNode

__all__ = [
    "Analysis",
    "EdgeFlags",
    "Import",
    "NodeFlags",
    "PackageView",
    "SymbolNode",
    "__version__",
]
