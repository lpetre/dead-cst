"""Plugin: keep discord.py bot/client handlers, Cogs, and extension hooks alive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..graph import NodeFlags
from ..plugins._base import Plugin, native

_COMMANDS_BOT_KINDS: frozenset[str] = frozenset({"Bot", "AutoShardedBot"})
_DISCORD_CLIENT_KINDS: frozenset[str] = frozenset({"Client", "AutoShardedClient"})

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
DISCORDPY_COG_PREFIX = "<discordpy-cog>:"
DISCORDPY_EXTENSION_PREFIX = "<discordpy-extension>:"


@dataclass
class DiscordPyPlugin(Plugin):
    """Wire discord.py bots, Cogs, and extension hooks into reachability."""

    def run(self, ctx: native.ProjectContext) -> Iterable[native.GraphOp]:
        # Cheap presence probe — O(1) hashmap lookup per module name
        # against the pre-built ``imports_by_module`` index. Short-
        # circuits before paying for the per-file path scan below.
        if not any(
            native.query(ctx).imports().of(m).exists()
            for m in ("discord", "discord.ext", "discord.ext.commands")
        ):
            return

        # Per-file gate: only fire on files that import discord.
        # ``node_paths`` rather than ``node_attrs`` — we only need
        # path here; the other fields would be allocated then dropped.
        discord_paths: set[str] = set()
        for module in ("discord", "discord.ext", "discord.ext.commands"):
            idxs = native.query(ctx).imports().of(module).indices()
            discord_paths.update(ctx.node_paths(idxs))

        # 1. Bot / Client constructions.
        bot_rows: list[native.ConstructionIdxRef] = []
        bot_rows.extend(
            native.query(ctx)
            .constructions()
            .where_module("discord.ext.commands")
            .where_name(sorted(_COMMANDS_BOT_KINDS))
            .row_indices()
        )
        bot_rows.extend(
            native.query(ctx)
            .constructions()
            .where_module("discord")
            .where_name(sorted(_DISCORD_CLIENT_KINDS))
            .row_indices()
        )

        # bot_vars_by_file: path -> {simple_name -> var_idx}.
        bot_vars_by_file: dict[str, dict[str, int]] = {}
        if bot_rows:
            bot_attrs = ctx.node_attrs([r.var_idx for r in bot_rows])
            for row, (_k, _p, fqname, _f) in zip(bot_rows, bot_attrs, strict=True):
                if row.path not in discord_paths:
                    continue
                simple = fqname.rsplit(".", 1)[-1]
                bot_vars_by_file.setdefault(row.path, {})[simple] = row.var_idx
                yield native.AddEntrypointByIdx(row.var_idx, marker="<discordpy-app>")

        # 2. Single-attr decorators @<bot>.<verb>(...).
        for h in (
            native.query(ctx).decorators().where_owner_attr(sorted(_BOT_DECORATORS)).row_indices()
        ):
            owner_idx = bot_vars_by_file.get(h.path, {}).get(h.decorator_owner or "")
            if owner_idx is not None:
                yield native.AddEdgeByIdx(owner_idx, h.decorated_idx)

        # 3. Two-level slash-command decorators @<bot>.tree.<verb>(...).
        for h in (
            native.query(ctx)
            .decorators()
            .where_owner_attr_via("tree", sorted(_TREE_DECORATORS))
            .row_indices()
        ):
            owner_idx = bot_vars_by_file.get(h.path, {}).get(h.decorator_owner or "")
            if owner_idx is not None:
                yield native.AddEdgeByIdx(owner_idx, h.decorated_idx)

        # 4. Cog subclasses + module-level setup / teardown hooks.
        cogs_by_path: dict[str, list[int]] = {}
        for base in ("discord.ext.commands.Cog", "discord.ext.commands.GroupCog"):
            cog_idxs = native.query(ctx).subclasses().of_fqn(base).transitive(True).indices()
            if not cog_idxs:
                continue
            for cog_idx, cog_path in zip(cog_idxs, ctx.node_paths(cog_idxs), strict=True):
                cogs_by_path.setdefault(cog_path, []).append(cog_idx)

        if cogs_by_path:
            hook_funcs_by_path: dict[str, list[int]] = {}
            hook_idxs = (
                native.query(ctx)
                .decls()
                .with_kind("function")
                .with_paths(list(cogs_by_path.keys()))
                .with_simple_names(["setup", "teardown"])
                .indices()
            )
            if hook_idxs:
                for hook_idx, hook_path in zip(hook_idxs, ctx.node_paths(hook_idxs), strict=True):
                    hook_funcs_by_path.setdefault(hook_path, []).append(hook_idx)

            for path, cog_idxs in cogs_by_path.items():
                targets = list(cog_idxs) + hook_funcs_by_path.get(path, [])
                filename = Path(path).name
                yield native.AddNodeByIdx(
                    fqname=f"{DISCORDPY_COG_PREFIX}{filename}",
                    path=path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to_idx=targets,
                )

        # 5. load_extension / load_extensions string-literal targets.
        # Collect the (owner_path, extension_fqname) pairs in order
        # first so the module-surface lookup runs in a single batched
        # scan instead of one scan per call site.
        seen_extensions: set[str] = set()
        pending: list[tuple[str, str]] = []
        for attr in ("load_extension", "load_extensions"):
            for cref in native.query(ctx).calls().where_attr(attr).string_arg_at(0).row_indices():
                if cref.path not in discord_paths:
                    continue
                if cref.string_arg in seen_extensions:
                    continue
                seen_extensions.add(cref.string_arg)
                pending.append((cref.path, cref.string_arg))
        if pending:
            surfaces = ctx.module_surfaces_indices([ext for _, ext in pending])
            for owner_path, ext_fqname in pending:
                target_idxs = surfaces.get(ext_fqname, [])
                if not target_idxs:
                    continue
                yield native.AddNodeByIdx(
                    fqname=f"{DISCORDPY_EXTENSION_PREFIX}{ext_fqname}",
                    path=owner_path,
                    flags=int(NodeFlags.ENTRYPOINT),
                    edges_to_idx=list(target_idxs),
                )
