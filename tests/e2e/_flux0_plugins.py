"""Project-specific plugin subclasses for the flux0 e2e tests.

The two abstract bases (:class:`DecoratedDeclPlugin`,
:class:`LiteralListPlugin`) live in :mod:`dead_cst.plugins.decl_shapes`
and were promoted out of this file once they earned a second user. The
flux0 subclasses below are now pure configuration -- four to five lines
each.
"""

from __future__ import annotations

from dataclasses import dataclass

from dead_cst.plugins import DecoratedDeclPlugin, LiteralListPlugin


@dataclass
class Flux0CliCommandsPlugin(DecoratedDeclPlugin):
    """Mirror ``flux0_cli/main.py:register_commands``.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/cli/src/flux0_cli/main.py#L61
    """

    marker_prefix: str = "flux0_cli_cmds"
    package_prefix: str = "flux0_cli.cmds"
    decorator_module: str = "click"
    decorator_names: frozenset[str] = frozenset({"group", "Group"})
    constructor_names: frozenset[str] = frozenset({"Group"})


@dataclass
class Flux0InternalModulesPlugin(LiteralListPlugin):
    """Mirror ``flux0_server.main.INTERNAL_MODULES``.

    https://github.com/flux0-ai/flux0/blob/8d04176642b091ddb5c5020486f353d4e824460b/packages/server/src/flux0_server/main.py#L51
    """

    marker_prefix: str = "flux0_internal_modules"
    owner_fqname: str = "flux0_server.main"
    variable_name: str = "INTERNAL_MODULES"
