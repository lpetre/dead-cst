"""Default ``name -> path`` import resolution.

Provides :func:`default_resolve_import`, the ``sys.path`` + ``importlib``
implementation that every shipped :class:`PathResolver` delegates to.
Custom resolvers can call it directly, replace it with their own logic
(for vendored deps, ``.pyi`` siblings, ...), or compose both.

The supporting helpers (:func:`safe_resolve_module` and
:func:`distribution_lookup`) live here so the analyzer and the
resolvers share a single cache and one canonical treatment of
distribution-name normalization.
"""

from __future__ import annotations

import json
import os
import re
import sys
import sysconfig
from functools import cache
from importlib.machinery import ModuleSpec
from pathlib import Path

from ..plugins._core import (
    EXTERNAL_DIST_PREFIX,
    EXTERNAL_FILE_PREFIX,
    STDLIB_PREFIX,
)


def _resolved_sysconfig_path(name: str) -> Path | None:
    raw = sysconfig.get_path(name)
    if not raw:
        return None
    try:
        return Path(raw).resolve()
    except OSError:
        return None


# A system Python install (no venv) typically nests purelib/platlib *inside*
# the stdlib root (e.g. ``/usr/local/lib/python3.13/site-packages`` under
# ``/usr/local/lib/python3.13``). Capturing both lets us classify a resolved
# path as stdlib only when it isn't actually a third-party package living
# under that same root.
STDLIB = Path(sysconfig.get_path("stdlib")).resolve()
PURELIB = _resolved_sysconfig_path("purelib")
PLATLIB = _resolved_sysconfig_path("platlib")
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

    Cached process-wide. The analyzer clears it in worker transitions
    (see :func:`dead_cst.analyze._on_search_paths_change`) so a worker
    that crosses a venv boundary -- e.g. between two uv-workspace
    members each with their own ``.venv`` -- doesn't keep the prior
    venv's dist map. Serial single-process runs already have a stable
    ``sys.path`` at the venv level (only the first-party prefix moves
    between bases), so the cache survives across bases there.
    """
    from importlib import metadata

    lookup = {}
    for dist in metadata.distributions():
        canonical = _canonical_dist_name(dist.metadata["Name"])
        for file in dist.files or ():
            abs_path = Path(str(dist.locate_file(file))).resolve()
            lookup[abs_path] = canonical
    return lookup


@cache
def editable_distribution_roots() -> tuple[tuple[Path, str], ...]:
    """Source-directory roots for editably-installed distributions.

    A modern ``pip install -e`` / ``uv pip install -e`` only records
    metadata files plus a ``.pth`` (or PEP 660 finder proxy) in the
    dist's ``RECORD``. The actual ``.py`` source lives in the user's
    project directory and never appears in :func:`distribution_lookup`.
    We surface it here by reading each dist's ``direct_url.json``
    (PEP 610) and any ``.pth`` shims it ships, so an importer of an
    editable third-party package resolves to ``[external dist] <name>``
    instead of raising on an "unexpected path" or being silently
    misclassified.

    :func:`default_resolve_import` consults this *after* the
    ``search_paths`` check, so a first-party file that happens to nest
    under an editable root (the project being analyzed living inside
    another editable install's checkout) still classifies as
    first-party.

    Returns ``(root, canonical_name)`` tuples sorted longest-path first
    so prefix-matching against nested editable layouts picks the most
    specific dist. Cached alongside :func:`distribution_lookup` and
    cleared together when the analyzer transitions across venvs.
    """
    from importlib import metadata

    roots: list[tuple[Path, str]] = []
    for dist in metadata.distributions():
        canonical = _canonical_dist_name(dist.metadata["Name"])
        for root in _editable_source_roots(dist):
            roots.append((root, canonical))
    roots.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    return tuple(roots)


def _editable_source_roots(dist) -> list[Path]:
    """Discover editable source directories advertised by ``dist``.

    Honors PEP 610 (``direct_url.json`` with ``dir_info.editable=true``)
    and the ``.pth`` shim style used by uv / setuptools-legacy
    editables. Both are best-effort: malformed metadata or missing
    files just yield no roots for this dist.
    """
    roots: list[Path] = []

    direct_url = _safe_read_dist_text(dist, "direct_url.json")
    if direct_url:
        try:
            data = json.loads(direct_url)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict) and data.get("dir_info", {}).get("editable"):
            url = data.get("url")
            if isinstance(url, str) and url.startswith("file://"):
                # ``file:///abs/path`` -> ``/abs/path``; ``file://host/path`` is
                # not portable on POSIX but we let Path normalize whatever's
                # left of the scheme prefix.
                candidate = Path(url[len("file://") :])
                if candidate.is_absolute() and candidate.is_dir():
                    roots.append(candidate.resolve())

    for file in dist.files or ():
        if not str(file).endswith(".pth"):
            continue
        try:
            text = Path(str(dist.locate_file(file))).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(("import ", "import\t")):
                continue
            candidate = Path(line)
            if candidate.is_absolute() and candidate.is_dir():
                roots.append(candidate.resolve())

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _safe_read_dist_text(dist, name: str) -> str | None:
    try:
        return dist.read_text(name)
    except (OSError, UnicodeDecodeError):
        return None


def _is_site_packages_path(path: Path) -> bool:
    """True when ``path`` lives inside an interpreter's third-party install dir.

    We check ``purelib`` / ``platlib`` first since they're the
    interpreter-blessed locations, then fall back to the conventional
    directory names for layouts where sysconfig disagrees with reality
    (Debian ``dist-packages`` siblings, vendored bundles, ...).
    """
    if PURELIB is not None and path.is_relative_to(PURELIB):
        return True
    if PLATLIB is not None and path.is_relative_to(PLATLIB):
        return True
    return any(part in SITE_PACKAGES_MARKERS for part in path.parts)


def _is_stdlib_path(path: Path) -> bool:
    """True when ``path`` is part of the standard library.

    Excludes nested site-packages because a system Python install lays
    them out *inside* ``STDLIB`` (e.g. ``.../python3.13/site-packages``
    under ``.../python3.13``); a naive ``is_relative_to(STDLIB)`` would
    otherwise misclassify every third-party package as stdlib.
    """
    return path.is_relative_to(STDLIB) and not _is_site_packages_path(path)


def clear_path_caches() -> None:
    """Drop the ``sys.path``-derived resolver caches.

    :func:`safe_resolve_module` keys on fullname and
    :func:`distribution_lookup` / :func:`editable_distribution_roots`
    key on ``()`` -- all three read live ``sys.path`` (or
    :mod:`importlib.metadata` against it). Anything mutating
    ``sys.path`` (the analyzer's per-base rebind, a resolver splicing
    in its own venv) must call this to keep the next lookup honest.
    """
    safe_resolve_module.cache_clear()
    distribution_lookup.cache_clear()
    editable_distribution_roots.cache_clear()


def default_resolve_import(name: str, search_paths: list[Path]) -> str | Path | None:
    """Resolve a dotted module name against ``sys.path`` + the importlib finders.

    Returns one of:

    * a :class:`Path` when ``name`` is a first-party module under one of
      ``search_paths`` (the analyzer ingests it as a regular graph node);
    * a synthetic ``[stdlib] <name>`` / ``[external dist] <pkg>`` /
      ``[external file] <name>`` string when ``name`` resolves outside
      the project (collapsed into one synthetic node per group);
    * ``None`` when the importlib finders can't locate ``name`` at all.

    Classification precedence (first match wins):

    1. ``distribution_lookup`` -- the resolved path appears in an
       installed dist's ``RECORD``.
    2. stdlib (``_is_stdlib_path``).
    3. site-packages (``_is_site_packages_path``).
    4. ``search_paths`` -- first-party project file.
    5. ``editable_distribution_roots`` -- editable third-party whose
       source dir lives outside ``search_paths``.

    Stdlib / site-packages precede ``search_paths`` because they're
    interpreter-blessed locations a user-configured ``search_paths``
    shouldn't overlap with. ``search_paths`` precedes editable roots so
    that a project whose source happens to nest under another editable
    install's root (e.g. an e2e fixture cloned into ``.pytest_cache/``
    of an editable ``dead-cst`` checkout) is still treated as
    first-party.

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

    # Distribution lookup runs before the stdlib check so that, on a
    # system Python where site-packages is nested inside the stdlib
    # root, third-party packages are still correctly attributed to
    # their distribution rather than swallowed into ``[stdlib]``.
    if dist := distribution_lookup().get(path):
        return f"{EXTERNAL_DIST_PREFIX}{dist}"

    if _is_stdlib_path(path):
        return f"{STDLIB_PREFIX}{name}"

    if _is_site_packages_path(path):
        return f"{EXTERNAL_FILE_PREFIX}{name}"

    # First-party wins over editable-dist matching: the project being
    # analyzed may itself live under another editable install's root
    # (e.g. an e2e fixture clones a third-party repo into
    # ``.pytest_cache/`` inside an editable ``dead-cst`` checkout). Those
    # files are first-party for the run, not third-party. ``stdlib`` /
    # ``site-packages`` still win above because those are
    # interpreter-blessed locations and shouldn't appear in a
    # user-configured ``search_paths``.
    for search in search_paths:
        if path.is_relative_to(search):
            return path

    for root, dist in editable_distribution_roots():
        if path.is_relative_to(root):
            return f"{EXTERNAL_DIST_PREFIX}{dist}"

    raise Exception(f"Module {name} resolved to an unexpected path: {path}")
