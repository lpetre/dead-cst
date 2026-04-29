"""Default ``name -> path`` import resolution.

Provides :func:`default_resolve_import`, the ``sys.path`` + ``importlib``
implementation that every shipped :class:`PathResolver` delegates to.
Custom resolvers can call it directly, replace it with their own logic
(for vendored deps, ``.pyi`` siblings, ...), or compose both.

The supporting helpers (:func:`safe_resolve_module`,
:func:`distribution_lookup`, :func:`temp_sys_path`) are kept here so the
analyzer and the resolvers share a single cache and one canonical
treatment of distribution-name normalization.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import sysconfig
from functools import cache
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Generator

from .._plugins._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    STDLIB_PREFIX,
)

STDLIB = Path(sysconfig.get_path("stdlib")).resolve()
SITE_PACKAGES_MARKERS = ("site-packages", "dist-packages")


def _canonical_dist_name(name: str) -> str:
    """Normalize a PyPI distribution name per PEP 503.

    PyPI treats ``Flask`` / ``flask`` / ``FLASK`` as the same project;
    plugins query the analyzer by the lowercase import name (``flask``)
    so the synthetic ``[external dist] <name>`` node has to match. PEP
    503's canonical form is ``re.sub(r"[-_.]+", "-", name).lower()`` --
    we apply that to the raw ``Name`` from each dist's metadata so
    plugin lookups don't depend on whatever casing the package author
    chose.

    Note: this still uses the *distribution* name, not the import
    (top-level) name. For most third-party packages they match after
    canonicalization (``fastapi``, ``flask``, ``click``, ``typer``,
    ``networkx``, ...). They differ for a handful of historic names
    (``Pillow`` → import ``PIL``, ``PyYAML`` → import ``yaml``); plugins
    targeting those would need an explicit dist-name override, which
    we don't yet support.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


@contextlib.contextmanager
def temp_sys_path(paths: list[Path]) -> Generator[None, None, None]:
    """Prepend ``paths`` to ``sys.path`` for the duration of the ``with`` block.

    Used by the analyzer to scope :func:`safe_resolve_module`'s lookups
    to the current base's search paths without leaking entries between
    bases.
    """
    old = list(sys.path)
    seen = set(old)
    sys.path = [str(p) for p in paths if str(p) not in seen] + sys.path
    try:
        yield
    finally:
        sys.path = old


@cache
def safe_resolve_module(fullname: str) -> ModuleSpec | None:
    """Locate a :class:`ModuleSpec` for ``fullname`` against the current ``sys.path``.

    Cached, so a base's full sweep over import edges only pays the
    finder cost once per unique name. The analyzer clears the cache
    between bases (search paths change) via
    :meth:`safe_resolve_module.cache_clear`.
    """
    parts = fullname.split(".")
    search_paths = list(sys.path)

    # emulate namespace __path__ resolution
    for i, part in enumerate(parts[:-1]):
        candidate_paths = []
        for base in search_paths:
            subdir = os.path.join(base, parts[i])
            if os.path.isdir(subdir):
                candidate_paths.append(subdir)
        search_paths = candidate_paths

    # Final part resolution
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if not find_spec:
            continue
        try:
            spec = find_spec(fullname, search_paths)
            if spec:
                return spec
        except Exception:
            continue

    return None


@cache
def distribution_lookup() -> dict[Path, str]:
    """Map every installed distribution file to its canonical project name.

    Used by :func:`default_resolve_import` to classify a resolved path
    as ``[external dist] <name>`` when the file came from an installed
    third-party distribution.
    """
    from importlib import metadata

    lookup = {}
    for dist in metadata.distributions():
        canonical = _canonical_dist_name(dist.metadata["Name"])
        for file in dist.files or ():
            abs_path = Path(str(dist.locate_file(file))).resolve()
            lookup[abs_path] = canonical
    return lookup


def default_resolve_import(name: str, search_paths: list[Path]) -> str | Path | None:
    """Resolve a dotted module name against ``sys.path`` + the importlib finders.

    Returns one of:

    * a :class:`Path` when ``name`` is a first-party module under one of
      ``search_paths`` (the analyzer ingests it as a regular graph node);
    * a synthetic ``[stdlib] <name>`` / ``[external dist] <pkg>`` /
      ``[external file] <name>`` string when ``name`` resolves outside
      the project (collapsed into one synthetic node per group);
    * ``None`` when the importlib finders can't locate ``name`` at all.

    This is the implementation every shipped :class:`PathResolver`
    delegates to. Third-party resolvers can call it as a fallback after
    their own layout-specific lookups, or replace it entirely.
    """
    spec = safe_resolve_module(name)
    if spec is None:
        return None
    if spec.origin is None:
        return None
    if spec.origin in {"built-in", "frozen"}:
        return f"{STDLIB_PREFIX}{name}"
    path = Path(spec.origin).resolve()
    if path.is_relative_to(STDLIB):
        return f"{STDLIB_PREFIX}{name}"

    lookup = distribution_lookup()
    if dist := lookup.get(path):
        return f"{EXTERNAL_DIST_PREFIX}{dist}"

    path_str = str(path)
    if any(m in path_str for m in SITE_PACKAGES_MARKERS):
        return f"{EXTERNAL_FILE_PREFIX}{name}"

    for search in search_paths:
        if path.is_relative_to(search):
            return path
    raise Exception(f"Module {name} resolved to an unexpected path: {path}")
