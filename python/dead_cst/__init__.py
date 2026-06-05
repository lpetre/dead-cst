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
* :mod:`dead_cst.plugins` -- the external-node prefix constants and
  every built-in plugin.

For multi-package monorepos, the caller is responsible for setting
up a venv with editable ``.pth`` entries pointing at each member's
published source dir (``uv sync --all-packages`` produces this
layout). Pass the venv path to :class:`Analysis` and ty's resolver
discovers every member via the ``.pth`` files.

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
