"""Pluggable resolvers that discover the first-party packages of a project.

A :class:`PathResolver` returns a tuple of :class:`Package` objects
(one per first-party workspace member). An
:class:`~dead_cst.analyze.Analysis` takes exactly one resolver.

Builtins: :class:`ManualResolver` (explicit ``-p`` specs) and
:class:`UvResolver` (workspace members discovered from ``uv.lock``).
Third-party resolvers register under the ``dead_cst.resolvers``
entry-point group; :func:`load_resolver` checks builtins first.
"""

from __future__ import annotations

from ..contrib.uv import UvResolver
from ._core import Package, PathResolver, load_toml
from .manual import ManualResolver

BUILTIN_RESOLVERS: dict[str, type[PathResolver]] = {
    "uv": UvResolver,
}


def load_resolver(name: str) -> PathResolver:
    """Load a resolver by name. Checks builtins first, then entry points."""
    if name in BUILTIN_RESOLVERS:
        return BUILTIN_RESOLVERS[name]()

    from importlib.metadata import entry_points

    for ep in entry_points(group="dead_cst.resolvers"):
        if ep.name == name:
            cls = ep.load()
            return cls()
    raise KeyError(f"Unknown path resolver: {name!r}")


__all__ = [
    "BUILTIN_RESOLVERS",
    "ManualResolver",
    "Package",
    "PathResolver",
    "UvResolver",
    "load_resolver",
    "load_toml",
]
