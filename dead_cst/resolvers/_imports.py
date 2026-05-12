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

    Cached, so a package's full sweep over import edges only pays the
    finder cost once per unique name. The analyzer clears the cache
    between packages (search paths change) via
    :meth:`safe_resolve_module.cache_clear`.
    """
    parts = fullname.split(".")
    search_paths = list(sys.path)

    # emulate namespace __path__ resolution
    for i, part in enumerate(parts[:-1]):
        candidate_paths = []
        for entry in search_paths:
            subdir = os.path.join(entry, parts[i])
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


def _dist_relevant_sys_path() -> tuple[str, ...]:
    """Snapshot of ``sys.path`` entries that can host installed distributions.

    :func:`distribution_lookup` and :func:`editable_distribution_roots`
    depend on the *dist-bearing* portion of ``sys.path`` --
    ``importlib.metadata`` discovers distributions by walking
    site-packages-style directories. The analyzer's per-package
    ``sys.path`` rebind only moves the first-party project prefix,
    which never matches this filter, so the dist caches keyed on this
    snapshot survive package transitions for free. A real venv change
    (uv resolver splicing in a workspace ``.venv``, see
    :func:`dead_cst.contrib.uv.UvResolver.resolve_import`) adds a new
    site-packages entry, which flips the key and triggers a single
    rebuild.
    """
    purelib_str = str(PURELIB) if PURELIB is not None else None
    platlib_str = str(PLATLIB) if PLATLIB is not None else None
    out: list[str] = []
    for entry in sys.path:
        if entry == purelib_str or entry == platlib_str:
            out.append(entry)
            continue
        if any(marker in entry for marker in SITE_PACKAGES_MARKERS):
            out.append(entry)
    return tuple(out)


@cache
def _distribution_lookup_for(_key: tuple[str, ...]) -> dict[Path, str]:
    """Cached worker for :func:`distribution_lookup`, keyed on the venv slice.

    ``_key`` is the :func:`_dist_relevant_sys_path` snapshot at call
    time. The body still reads :mod:`importlib.metadata` against the
    live ``sys.path`` -- the key is purely a cache-invalidation
    fingerprint so the same dist-bearing layout maps to the same
    entry.
    """
    from importlib import metadata

    lookup: dict[Path, str] = {}
    for dist in metadata.distributions():
        canonical = _canonical_dist_name(dist.metadata["Name"])
        for file in dist.files or ():
            abs_path = Path(str(dist.locate_file(file))).resolve()
            lookup[abs_path] = canonical
    return lookup


def distribution_lookup() -> dict[Path, str]:
    """Map every installed distribution file to its canonical project name.

    Used by :func:`default_resolve_import` to classify a resolved path
    as ``[external dist] <name>`` when the file came from an installed
    third-party distribution.

    Cached process-wide and keyed on :func:`_dist_relevant_sys_path` --
    the dist-bearing slice of ``sys.path``. A worker crossing a venv
    boundary (two uv-workspace members each with their own ``.venv``)
    sees a different key and rebuilds. Single-venv runs hit the cache
    on every package transition: the analyzer's per-package rebind
    only moves the first-party prefix, which doesn't enter the key.
    """
    return _distribution_lookup_for(_dist_relevant_sys_path())


@cache
def _editable_distribution_roots_for(
    _key: tuple[str, ...],
) -> tuple[tuple[Path, str], ...]:
    """Cached worker for :func:`editable_distribution_roots`. See
    :func:`_distribution_lookup_for` for the keying contract."""
    from importlib import metadata

    roots: list[tuple[Path, str]] = []
    for dist in metadata.distributions():
        canonical = _canonical_dist_name(dist.metadata["Name"])
        for root in _editable_source_roots(dist):
            roots.append((root, canonical))
    roots.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    return tuple(roots)


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
    specific dist. Cached alongside :func:`distribution_lookup` on the
    same dist-bearing ``sys.path`` slice.
    """
    return _editable_distribution_roots_for(_dist_relevant_sys_path())


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


def clear_module_specs_cache() -> None:
    """Clear the fullname-keyed module-spec cache only.

    :func:`safe_resolve_module` is keyed on the import fullname; it
    reads ``sys.path`` live, so anything mutating ``sys.path`` needs to
    invalidate it. The analyzer calls this on every package transition
    where only the first-party prefix moves -- the dist caches are
    keyed on :func:`_dist_relevant_sys_path` and survive that
    transition automatically, so a narrower clear is enough to keep
    every name resolution honest without re-paying the
    ``importlib.metadata`` walk.
    """
    safe_resolve_module.cache_clear()


def clear_path_caches() -> None:
    """Drop every ``sys.path``-derived resolver cache.

    The thorough variant of :func:`clear_module_specs_cache`. Useful
    when ``sys.path`` mutation actually changes the visible
    distributions (e.g. a uv resolver splicing in a workspace
    ``.venv``) or when a caller wants a full reset (tests). Note that
    :func:`distribution_lookup` / :func:`editable_distribution_roots`
    *also* self-invalidate via their dist-bearing ``sys.path`` slice,
    so callers that just want correctness across a real venv change
    don't strictly need to call this -- the next lookup will rebuild
    on its own. It is kept for explicit-reset semantics.
    """
    safe_resolve_module.cache_clear()
    _distribution_lookup_for.cache_clear()
    _editable_distribution_roots_for.cache_clear()


def default_resolve_import(name: str, search_paths: list[Path]) -> str | Path | None:
    """Resolve a dotted module name against ``sys.path`` + the importlib finders.

    Returns one of:

    * a :class:`Path` when ``name`` is a first-party module under one of
      ``search_paths`` (the analyzer ingests it as a regular graph node);
    * a synthetic ``[stdlib] <name>`` / ``[external dist] <pkg>`` /
      ``[external file] <name>`` string when ``name`` resolves outside
      the project (collapsed into one synthetic node per group);
    * ``None`` when the importlib finders can't locate ``name`` at all.

    A dotted name whose own ``find_spec`` returns nothing (``collections.abc``
    and friends, synthesized in their parent's ``__init__.py``) inherits
    its parent's classification — the child collapses onto the parent's
    synthetic node for dist / file, or gets its own ``[stdlib] <name>``
    entry for stdlib.

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
    if spec is None or spec.origin is None:
        # ``collections.abc``, ``importlib.resources.abc`` and friends are
        # synthesized in their parent's ``__init__.py`` and can't be
        # located by a static ``find_spec``. Fall back to the parent: if
        # it classifies, the child inherits.
        if "." in name:
            parent = default_resolve_import(name.rsplit(".", 1)[0], search_paths)
            if isinstance(parent, str):
                if parent.startswith(STDLIB_PREFIX):
                    return f"{STDLIB_PREFIX}{name}"
                # Dist / file synthetic-node keys are the dist / top-level
                # name -- children collapse onto the parent's node.
                if parent.startswith((EXTERNAL_DIST_PREFIX, EXTERNAL_FILE_PREFIX)):
                    return parent
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
