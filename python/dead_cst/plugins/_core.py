"""Synthetic-node prefixes and the :func:`simple_name` helper.

The rust backend emits these fqname prefixes for non-first-party
imports; callers consume them to filter the graph for external
dependencies (``[external dist] requests``, ``[stdlib] os``, …).
"""

from __future__ import annotations

# Synthetic-node fqname prefixes used by the rust backend for non-first-
# party imports. Plugins consume these to filter the graph for external
# dependencies.
STDLIB_PREFIX = "[stdlib] "
EXTERNAL_DIST_PREFIX = "[external dist] "
EXTERNAL_FILE_PREFIX = "[external file] "
EXTERNAL_PREFIXES = (EXTERNAL_DIST_PREFIX, EXTERNAL_FILE_PREFIX)
UNRESOLVED_PREFIX = "[unresolved] "
UNPARSEABLE_PREFIX = "[unparseable] "
SYNTHETIC_PATH_PREFIXES = (*EXTERNAL_PREFIXES, UNRESOLVED_PREFIX)


def simple_name(fqname: str) -> str:
    """Return the rightmost dotted segment of ``fqname`` (``pkg.mod.f`` -> ``f``)."""
    return fqname.rpartition(".")[2]
