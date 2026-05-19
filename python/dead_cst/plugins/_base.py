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
"""

from __future__ import annotations

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
