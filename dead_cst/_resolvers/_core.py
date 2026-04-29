"""Shared types and helpers for path resolvers.

Defines the :class:`PathResolver` protocol every resolver satisfies, the
:data:`PathMap` shape they return, and :func:`merge_paths` for combining
multiple resolvers' outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

PathMap = dict[Path, list[Path]]


@runtime_checkable
class PathResolver(Protocol):
    """A resolver describes a project layout in two complementary ways.

    :meth:`resolve` reports the search paths the analyzer should walk
    (``{base: [dep_paths]}``); :meth:`resolve_import` answers
    ``name -> path`` lookups inside that layout. Splitting them lets a
    resolver own both halves -- e.g. a vendored-deps resolver can point
    at a checked-in ``third_party/`` and also redirect imports to it
    without monkey-patching the analyzer.

    The shipped resolvers all delegate :meth:`resolve_import` to
    :func:`~dead_cst._resolvers._imports.default_resolve_import`, the
    ``sys.path`` + ``importlib`` implementation. Custom resolvers
    typically call it as a fallback after their own layout-specific
    lookups.
    """

    name: str

    def resolve(self, project_root: Path) -> PathMap: ...

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None: ...


def load_toml(path: Path) -> dict[str, Any] | None:
    """Read ``path`` as TOML; ``None`` if the file is missing or tomllib is unavailable.

    Bad TOML is a programmer/config error and propagates as
    :class:`tomllib.TOMLDecodeError`.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11 not supported
        return None
    try:
        f = path.open("rb")
    except OSError:
        return None
    with f:
        return tomllib.load(f)


def merge_paths(*maps: PathMap) -> PathMap:
    """Merge multiple ``PathMap``s, unioning dep-path lists per base."""
    out: PathMap = {}
    for m in maps:
        for base, deps in m.items():
            base = base.resolve()
            existing = out.setdefault(base, [])
            for dep in deps:
                dep = dep.resolve()
                if dep not in existing and dep != base:
                    existing.append(dep)
    return out
