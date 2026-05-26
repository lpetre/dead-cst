"""Plugin: mark WSGI/ASGI server config modules as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
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

_TARGET_KINDS: tuple[str, ...] = ("module", "function", "class", "variable", "import")


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
        targets_by_path: dict[str, list[int]] = {}
        idxs = (
            native.query(ctx)
            .decls()
            .with_filenames(list(self.filenames))
            .with_kinds(list(_TARGET_KINDS))
            .indices()
        )
        if not idxs:
            return
        for idx, path in zip(idxs, ctx.node_paths(idxs), strict=True):
            targets_by_path.setdefault(path, []).append(idx)

        paths = list(targets_by_path.keys())
        module_idxs = ctx.modules_for_paths(paths)
        present = [(p, idx) for p, idx in zip(paths, module_idxs) if idx is not None]
        if not present:
            return
        module_attrs = ctx.node_attrs([idx for _p, idx in present])
        for (path, _idx), (_k, _p, module_fqname, _f) in zip(present, module_attrs, strict=True):
            yield native.AddNodeByIdx(
                fqname=f"{SERVER_CONFIG_PREFIX}{module_fqname}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
                edges_to_idx=targets_by_path[path],
            )
