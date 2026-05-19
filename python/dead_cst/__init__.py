"""Dead code analysis for Python.

``dead-cst`` builds a symbol-level reachability graph of a Python
codebase via a rust-backed extension (:mod:`dead_cst.native`, which
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
* :mod:`dead_cst.native` -- the rust-backed extension. Plugin authors
  doing a deep integration import this directly
  (``from dead_cst import native``) to construct ``GraphOp``\\s and
  query the in-progress :class:`native.ProjectContext`.

``dead-cst`` is alpha; APIs, CLI flags, and output formats may change.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import native
from .analyze import Analysis, PackageView
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
    "PackageView",
    "SymbolNode",
    "__version__",
    "native",
]
