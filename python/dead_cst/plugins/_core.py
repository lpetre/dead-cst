"""External-node fqname prefixes and the :func:`simple_name` helper.

The rust backend mints a ``kind="external"`` node for every resolved
site-packages import, named with one of these prefixes; callers consume
them to filter the graph for external dependencies
(``[external dist] requests``, ``[external file] vendored_mod``, …).
"""

from __future__ import annotations

# External-node fqname prefixes used by the rust backend for resolved
# site-packages imports. Each such node carries ``kind="external"`` and a
# real site-packages path; plugins consume these to filter the graph for
# external dependencies. (Stdlib imports mint no node, and imports that
# don't resolve carry no external node — their alias is flagged
# ``NodeFlags.UNRESOLVED`` instead.)
EXTERNAL_DIST_PREFIX = "[external dist] "
EXTERNAL_FILE_PREFIX = "[external file] "
EXTERNAL_PREFIXES = (EXTERNAL_DIST_PREFIX, EXTERNAL_FILE_PREFIX)


def simple_name(fqname: str) -> str:
    """Return the rightmost dotted segment of ``fqname`` (``pkg.mod.f`` -> ``f``)."""
    return fqname.rpartition(".")[2]
