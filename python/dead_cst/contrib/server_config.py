"""Plugin: mark WSGI/ASGI server config modules as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..graph import NodeFlags
from ..plugins._base import Plugin, native

SERVER_CONFIG_PREFIX = "<server-config>:"

# Conventional filenames Gunicorn and Hypercorn load at startup.
_DEFAULT_FILENAMES: tuple[str, ...] = (
    "gunicorn.conf.py",
    "gunicorn_conf.py",
    "hypercorn.conf.py",
    "hypercorn_conf.py",
)


@dataclass
class ServerConfigPlugin(Plugin):
    """Mark WSGI/ASGI server config modules as entrypoints.

    Files like ``gunicorn.conf.py`` and ``hypercorn.conf.py`` are loaded
    by the server process at startup -- nothing in the project imports
    them, so the static analyzer sees the whole file as unreachable.
    The plugin matches a configurable set of filenames and marks the
    matched module's top-level surface as alive.
    """

    filenames: tuple[str, ...] = _DEFAULT_FILENAMES

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        targets_by_path: dict[str, list[native.SymbolNode]] = {}
        for n in ctx.nodes():
            if Path(n.path).name not in self.filenames:
                continue
            if n.kind in ("module", "function", "class", "variable", "import"):
                targets_by_path.setdefault(n.path, []).append(n)

        for path, targets in targets_by_path.items():
            module = ctx.module_for(path)
            if module is None:
                continue
            yield native.AddNode(
                fqname=f"{SERVER_CONFIG_PREFIX}{module.fqname}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to=targets,
            )
