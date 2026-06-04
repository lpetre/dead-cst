"""External-node prefix constants shared with the native backend.

Every built-in plugin is now a native (Rust) plugin on
:class:`dead_cst._native.NativePlugin` (``NativePlugin.main_block()`` …
``NativePlugin.celery()``); there is no Python ``Plugin`` protocol any
more. This package keeps only the external-node fqname prefixes the
rust backend mints for resolved site-packages imports
(``[external dist] ``, ``[external file] ``) and the :func:`simple_name`
helper, which callers use to filter the graph for external dependencies.

Out-of-tree plugins are external native plugins compiled against the
shipped runtime dylib and loaded via
:func:`dead_cst._native.load_native_plugins` — see ``NATIVE_PLUGINS.md``.
"""

from __future__ import annotations

from ._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    EXTERNAL_PREFIXES,
    simple_name,
)

__all__ = [
    "EXTERNAL_DIST_PREFIX",
    "EXTERNAL_FILE_PREFIX",
    "EXTERNAL_PREFIXES",
    "simple_name",
]
