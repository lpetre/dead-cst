"""Resolver built from explicit ``base:dep1,dep2`` path specs.

Mirrors the CLI's ``-p`` flag as a first-class :class:`PathResolver`,
so explicit specs flow through the same pipeline as auto-discovered
layouts and participate in :meth:`~PathResolver.resolve_import` lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._core import PathMap
from ._imports import default_resolve_import


@dataclass
class ManualResolver:
    """Resolver built from explicit ``base:dep1,dep2`` path specs.

    Each entry in ``specs`` is either ``"base"`` (no deps) or
    ``"base:dep1,dep2,..."``. Bases and deps are joined to
    ``project_root`` at :meth:`resolve` time; whitespace around dep
    names is stripped and trailing empty entries are dropped, matching
    the CLI's ``-p`` parsing.

    Not registered in :data:`BUILTIN_RESOLVERS` -- :func:`load_resolver`
    can't construct it without ``specs``. Instantiate directly when
    composing a resolver chain programmatically.
    """

    specs: list[str] = field(default_factory=list)
    name: str = "manual"
    version: int = 1777760307

    def resolve(self, project_root: Path) -> PathMap:
        out: PathMap = {}
        for spec in self.specs:
            if ":" in spec:
                base_str, deps_str = spec.split(":", 1)
                base = project_root / base_str
                deps = [project_root / d.strip() for d in deps_str.split(",") if d.strip()]
            else:
                base = project_root / spec
                deps = []
            out[base] = deps
        return out

    def resolve_import(self, name: str, search_paths: list[Path]) -> str | Path | None:
        return default_resolve_import(name, search_paths)
