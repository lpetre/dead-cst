"""Re-export the pyo3 extension's public surface.

Maturin generates this file automatically when ``python-source`` isn't
set; we ship our own so a sibling ``__init__.pyi`` (the type stub) can
provide static information ty needs when checking
``dead_cst/plugins/*.py``.
"""

from .dead_cst_ty_native import *  # noqa: F401, F403
from . import dead_cst_ty_native as _native

__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
