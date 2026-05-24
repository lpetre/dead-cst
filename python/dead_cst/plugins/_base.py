"""Plugin base class.

Every concrete plugin subclasses :class:`Plugin`. The base exists so
the rust backend can do an :func:`isinstance` check (typos like
``Pluign()`` raise :class:`TypeError` instead of being silently
skipped) and so plugin authors don't have to repeat the
``from dead_cst import _native as native`` lazy import at the top of
every ``run()`` body.

The base is intentionally tiny — it does not enforce the ``run``
signature today. Concrete plugins are expected to expose
``run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]``.
Optionally, ``prepare(self, repo_root)`` may be overridden to do
pre-graph work (config scans, etc.); the default is a no-op.
"""

from __future__ import annotations

from pathlib import Path

from dead_cst import _native as native

__all__ = ["Plugin", "native"]


class Plugin:
    """Base class every concrete plugin subclasses.

    Subclasses implement
    ``run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]``.
    The base class itself does nothing — it just provides a runtime
    type the backend can check.

    Access the native extension as ``self.native`` (or via the
    re-export from :mod:`dead_cst.plugins._base`) instead of repeating
    ``from dead_cst import _native as native`` inside every ``run()``.
    """

    native = native

    def prepare(self, repo_root: Path) -> None:
        """Pre-graph hook. Called once per plugin per
        :meth:`Analysis.materialize_all` invocation, before the graph
        is built and before ``run`` is dispatched.

        Receives the repository root the :class:`Analysis` was
        constructed with (i.e. :attr:`Analysis.project_root`).
        Subclasses that need to scan the repo for config files
        (``pyproject.toml``, framework-specific manifests, etc.) or
        compute setup state independent of the graph should override.

        The default implementation is a no-op. Plugins that don't
        need pre-graph work can ignore it.

        Exceptions raised here propagate out of
        :meth:`Analysis.materialize_all` before any graph construction
        happens, so they're a clean failure mode for "config
        invalid; can't proceed" cases.
        """
