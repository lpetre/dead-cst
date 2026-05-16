"""Plugin: treat WSGI/ASGI server config modules as entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..graph import NodeFlags
from ..plugins._core import (
    GraphOp,
    ObserveContext,
    PluginContext,
    entrypoint_payload,
    module_node,
)

if TYPE_CHECKING:
    import dead_cst_ty_native as native

    from ..graph import VisitorPayload

SERVER_CONFIG_PREFIX = "<server-config>:"

# Conventional filenames Gunicorn and Hypercorn load at startup. Both
# servers also accept arbitrary paths via ``--config``; users can extend
# this set on the plugin instance for non-standard layouts.
_DEFAULT_FILENAMES: tuple[str, ...] = (
    "gunicorn.conf.py",
    "gunicorn_conf.py",
    "hypercorn.conf.py",
    "hypercorn_conf.py",
)

_TOPLEVEL_DECL_TYPES: frozenset[str] = frozenset({"function", "class", "variable", "import"})


@dataclass
class ServerConfigPlugin:
    """Treat WSGI/ASGI server config modules as entrypoints.

    Files like ``gunicorn.conf.py`` and ``hypercorn.conf.py`` are loaded
    by the server process at startup (Docker entrypoint, Cloud Run,
    systemd unit, ...) -- nothing in the project imports them, so the
    static analyzer sees the whole file as unreachable. This plugin
    matches a configurable set of filenames and marks the matched
    module's top-level surface as alive:

    * the module node itself (so ``dead-cst remove`` never deletes the
      file);
    * every top-level function (hook callbacks like ``on_starting``,
      ``post_fork``, ``when_ready``, ``worker_exit``, ``pre_request``,
      ``post_request``, ...);
    * every top-level class (rare, but custom logging classes do appear
      in production configs);
    * every top-level variable (config settings like ``bind``,
      ``workers``, ``logconfig_dict``, ...);
    * every top-level import (so helpers used only to build config
      values, e.g. ``os.environ.get``, stay alive transitively).

    The default ``filenames`` set covers the conventions Gunicorn and
    Hypercorn document. Override or extend it for non-standard layouts:

    .. code-block:: python

        ServerConfigPlugin(filenames=("gunicorn.conf.py", "deploy/prod_gunicorn.py"))

    Pure per-file work: filename + payload top-level decls. Cross-file
    behavior (an importer of the config module getting an extra edge,
    etc.) is not modeled -- gunicorn/hypercorn read the file by path,
    not by import.
    """

    filenames: tuple[str, ...] = _DEFAULT_FILENAMES
    name: str = "server_config"
    version: int = 1778573528

    def observe(self, ctx: ObserveContext) -> VisitorPayload | None:
        if ctx.path.name not in self.filenames:
            return None
        module = module_node(ctx.payload)
        if module is None:
            return None
        decls = [n for n in ctx.payload.nodes if n.type in _TOPLEVEL_DECL_TYPES]
        return entrypoint_payload(
            f"{SERVER_CONFIG_PREFIX}{module.fqname}",
            ctx.path,
            [module, *decls],
        )

    def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
        return ()

    def run(self, ctx: native.ProjectContext) -> None:
        nodes = list(ctx.nodes())
        # Group decls by their module so we can mint one synthetic per
        # config file rather than per-decl.
        modules_by_path: dict[str, native.NativeNode] = {
            n.path: n for n in nodes if n.kind == "module"
        }
        targets_by_path: dict[str, list[native.NativeNode]] = {}
        for n in nodes:
            if Path(n.path).name not in self.filenames:
                continue
            if n.kind == "module":
                targets_by_path.setdefault(n.path, []).append(n)
            elif n.kind in ("function", "class", "variable", "import"):
                targets_by_path.setdefault(n.path, []).append(n)

        for path, targets in targets_by_path.items():
            module = modules_by_path.get(path)
            if module is None:
                continue
            marker = ctx.add_node(
                fqname=f"{SERVER_CONFIG_PREFIX}{module.fqname}",
                path=path,
                flags=int(NodeFlags.ENTRYPOINT),
            )
            for target in targets:
                ctx.add_edge(marker, target)
