"""Plugin: keep discord.py bot/client handlers, Cogs, and extension hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..graph import NodeFlags

if TYPE_CHECKING:
    from dead_cst import _native as native


_COMMANDS_BOT_KINDS: frozenset[str] = frozenset({"Bot", "AutoShardedBot"})
_DISCORD_CLIENT_KINDS: frozenset[str] = frozenset({"Client", "AutoShardedClient"})
_COG_BASES: frozenset[str] = frozenset({"Cog", "GroupCog"})

_BOT_DECORATORS: frozenset[str] = frozenset(
    {
        "command",
        "event",
        "listen",
        "group",
        "hybrid_command",
        "hybrid_group",
        "check",
        "check_once",
        "before_invoke",
        "after_invoke",
    }
)

_TREE_DECORATORS: frozenset[str] = frozenset({"command", "context_menu"})

DISCORDPY_APP_PREFIX = "<discordpy-app>:"
DISCORDPY_COG_PREFIX = "<discordpy-cog>:"
DISCORDPY_EXTENSION_PREFIX = "<discordpy-extension>:"


@dataclass
class DiscordPyPlugin:
    """Wire discord.py bots, Cogs, and extension hooks into reachability."""

    name: str = "discordpy"
    version: int = 1778566342

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        from dead_cst import _native as native

        # Per-file gate: only fire on files that import discord.
        discord_paths: set[str] = set()
        for module in ("discord", "discord.ext", "discord.ext.commands"):
            for imp in native.query(ctx).imports().of(module).collect():
                discord_paths.add(imp.path)
        if not discord_paths:
            return

        # 1. Bot / Client constructions.
        bot_constructions: list[native.ConstructionRef] = []
        bot_constructions.extend(
            native.query(ctx)
            .constructions()
            .where_module("discord.ext.commands")
            .where_name(sorted(_COMMANDS_BOT_KINDS))
        )
        bot_constructions.extend(
            native.query(ctx)
            .constructions()
            .where_module("discord")
            .where_name(sorted(_DISCORD_CLIENT_KINDS))
        )

        bot_vars_by_file: dict[str, dict[str, native.SymbolNode]] = {}
        for cref in bot_constructions:
            if cref.path not in discord_paths:
                continue
            simple = cref.var.fqname.rsplit(".", 1)[-1]
            bot_vars_by_file.setdefault(cref.path, {})[simple] = cref.var
            yield native.AddEntrypoint(cref.var, marker="<discordpy-app>")

        # 2. Single-attr decorators @<bot>.<verb>(...).
        for h in native.query(ctx).decorators().where_owner_attr(sorted(_BOT_DECORATORS)):
            owner_node = bot_vars_by_file.get(h.decorated.path, {}).get(h.decorator_owner or "")
            if owner_node is not None:
                yield native.AddEdge(owner_node, h.decorated)

        # 3. Two-level slash-command decorators @<bot>.tree.<verb>(...).
        for h in (
            native.query(ctx).decorators().where_owner_attr_via("tree", sorted(_TREE_DECORATORS))
        ):
            owner_node = bot_vars_by_file.get(h.decorated.path, {}).get(h.decorator_owner or "")
            if owner_node is not None:
                yield native.AddEdge(owner_node, h.decorated)

        # 4. Cog subclasses + module-level setup / teardown hooks.
        cogs_by_path: dict[str, list[native.SymbolNode]] = {}
        for base in ("discord.ext.commands.Cog", "discord.ext.commands.GroupCog"):
            for cog in native.query(ctx).subclasses().of_fqn(base).transitive(True).collect():
                cogs_by_path.setdefault(cog.path, []).append(cog)

        if cogs_by_path:
            hook_funcs_by_path: dict[str, list[native.SymbolNode]] = {}
            for n in ctx.nodes():
                if n.kind != "function" or n.path not in cogs_by_path:
                    continue
                if n.fqname.rsplit(".", 1)[-1] in ("setup", "teardown"):
                    hook_funcs_by_path.setdefault(n.path, []).append(n)

            for path, cogs in cogs_by_path.items():
                targets = list(cogs) + hook_funcs_by_path.get(path, [])
                filename = path.rsplit("/", 1)[-1]
                yield native.AddNode(
                    fqname=f"{DISCORDPY_COG_PREFIX}{filename}",
                    path=path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to=targets,
                )

        # 5. load_extension / load_extensions string-literal targets.
        seen_extensions: set[str] = set()
        for attr in ("load_extension", "load_extensions"):
            for cref in native.query(ctx).calls().where_attr(attr).string_arg_at(0):
                if cref.owner.path not in discord_paths:
                    continue
                if cref.string_arg in seen_extensions:
                    continue
                targets = list(ctx.module_surface(cref.string_arg))
                if not targets:
                    continue
                seen_extensions.add(cref.string_arg)
                yield native.AddNode(
                    fqname=f"{DISCORDPY_EXTENSION_PREFIX}{cref.string_arg}",
                    path=cref.owner.path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to=targets,
                )
