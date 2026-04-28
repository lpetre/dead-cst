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
    name: str

    def resolve(self, project_root: Path) -> PathMap: ...


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
