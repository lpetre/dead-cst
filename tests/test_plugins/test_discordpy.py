"""Tests for the native ``discordpy`` plugin."""

from __future__ import annotations

from dead_cst import _native as native


def test_discordpy_plugin_marks_bot_command_handlers(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext import commands

            bot = commands.Bot(command_prefix="!")

            @bot.command()
            async def ping(ctx):
                await ctx.send("pong")

            @bot.event
            async def on_ready():
                print("ready")

            @bot.listen("on_message")
            async def echo(message):
                pass

            def helper(): pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.bot" in reached
    assert "app.main.ping" in reached
    assert "app.main.on_ready" in reached
    assert "app.main.echo" in reached
    # Undecorated helper stays dead.
    assert "app.main.helper" not in reached


def test_discordpy_plugin_marks_slash_command_tree_handlers(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext import commands

            bot = commands.Bot(command_prefix="!")

            @bot.tree.command()
            async def slash_ping(interaction):
                pass

            @bot.tree.context_menu(name="Greet")
            async def greet(interaction, member):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.slash_ping" in reached
    assert "app.main.greet" in reached


def test_discordpy_plugin_handles_direct_client(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            import discord

            client = discord.Client(intents=discord.Intents.default())

            @client.event
            async def on_ready():
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.client" in reached
    assert "app.main.on_ready" in reached


def test_discordpy_plugin_handles_aliased_class_import(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext.commands import Bot as B

            bot = B(command_prefix="!")

            @bot.command()
            async def ping(ctx):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    assert "app.main.ping" in reachable_fqnames(graph)


def test_discordpy_plugin_handles_autosharded_variants(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            import discord
            from discord.ext import commands

            bot = commands.AutoShardedBot(command_prefix="!")
            client = discord.AutoShardedClient()

            @bot.command()
            async def b_cmd(ctx): pass

            @client.event
            async def on_ready(): pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.b_cmd" in reached
    assert "app.main.on_ready" in reached


def test_discordpy_plugin_keeps_cog_class_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "cogs/__init__.py": "",
            "cogs/greetings.py": """
            from discord.ext import commands

            class Greetings(commands.Cog):
                def __init__(self, bot):
                    self.bot = bot

                @commands.command()
                async def hello(self, ctx):
                    await ctx.send("hi")

            async def setup(bot):
                await bot.add_cog(Greetings(bot))
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    # Cog class + module-level setup function are both stamped ENTRYPOINT
    # directly (keep_alive) for any file holding a Cog subclass.
    assert "cogs.greetings.Greetings" in reached
    assert "cogs.greetings.setup" in reached


def test_discordpy_plugin_marks_teardown_alongside_setup(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "cogs/__init__.py": "",
            "cogs/admin.py": """
            from discord.ext.commands import Cog

            class Admin(Cog):
                pass

            async def setup(bot):
                await bot.add_cog(Admin())

            async def teardown(bot):
                await bot.remove_cog("Admin")
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "cogs.admin.Admin" in reached
    assert "cogs.admin.setup" in reached
    assert "cogs.admin.teardown" in reached


def test_discordpy_plugin_setup_without_cog_stays_dead(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": """
            from discord.ext import commands

            # No Cog subclass -- a function happening to be named ``setup``
            # is not a discord.py extension entrypoint.
            def setup(bot):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    assert "pkg.helpers.setup" not in reachable_fqnames(graph)


def test_discordpy_plugin_load_extension_pulls_target_module(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext import commands

            bot = commands.Bot(command_prefix="!")

            async def main():
                await bot.load_extension("app.cogs.greet")
                await bot.load_extensions(["app.cogs.admin", "app.cogs.misc"])
            """,
            "app/cogs/__init__.py": "",
            "app/cogs/greet.py": """
            def helper_for_greet():
                pass

            def setup(bot):
                pass
            """,
            "app/cogs/admin.py": """
            def admin_helper():
                pass

            def setup(bot):
                pass
            """,
            "app/cogs/misc.py": """
            def misc_helper():
                pass

            def setup(bot):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "app.cogs.greet" in reached
    assert "app.cogs.greet.setup" in reached
    assert "app.cogs.greet.helper_for_greet" in reached
    assert "app.cogs.admin.setup" in reached
    assert "app.cogs.admin.admin_helper" in reached
    assert "app.cogs.misc.setup" in reached
    assert "app.cogs.misc.misc_helper" in reached


def test_discordpy_plugin_load_extension_module_constant_resolves(
    build_plugin_graph, reachable_fqnames
):
    """A module-level string constant folds (see ``string_fold``), so
    ``load_extension(EXT_NAME)`` keeps the cog alive like a literal would."""
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext import commands

            bot = commands.Bot(command_prefix="!")

            EXT_NAME = "app.cogs.dynamic"

            async def main():
                await bot.load_extension(EXT_NAME)
            """,
            "app/cogs/__init__.py": "",
            "app/cogs/dynamic.py": """
            def setup(bot):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    assert "app.cogs.dynamic.setup" in reachable_fqnames(graph)


def test_discordpy_plugin_load_extension_non_literal_dropped(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext import commands

            bot = commands.Bot(command_prefix="!")

            async def main(ext_name: str):
                await bot.load_extension(ext_name)
            """,
            "app/cogs/__init__.py": "",
            "app/cogs/dynamic.py": """
            def setup(bot):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    # Non-literal argument is dropped silently; nothing keeps the cog alive.
    assert "app.cogs.dynamic.setup" not in reachable_fqnames(graph)


def test_discordpy_plugin_ignores_unrelated_bare_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from discord.ext import commands

            bot = commands.Bot(command_prefix="!")

            class Thing:
                def register(self, fn):
                    return fn

            t = Thing()

            @t.register
            def not_a_handler(): pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    assert "pkg.mod.not_a_handler" not in reachable_fqnames(graph)


def test_discordpy_plugin_ignores_files_without_discord_imports(
    build_plugin_graph, reachable_fqnames
):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            # No discord import in scope -- the plugin must not fire on
            # arbitrary load_extension lookalikes.

            class Loader:
                def load_extension(self, name):
                    pass

            loader = Loader()
            loader.load_extension("pkg.target")
            """,
            "pkg/target.py": """
            def setup(bot):
                pass
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    assert "pkg.target.setup" not in reachable_fqnames(graph)


def test_discordpy_plugin_handler_dependencies_stay_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/util.py": """
            def render_greeting() -> str:
                return "hello"

            def unused_util() -> str:
                return "bye"
            """,
            "app/main.py": """
            from discord.ext import commands
            from app.util import render_greeting

            bot = commands.Bot(command_prefix="!")

            @bot.command()
            async def hello(ctx):
                await ctx.send(render_greeting())
            """,
        },
        [native.NativePlugin.discordpy()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.hello" in reached
    # Handler transitively pulls render_greeting alive.
    assert "app.util.render_greeting" in reached
    # Unrelated util stays dead.
    assert "app.util.unused_util" not in reached


def test_discordpy_plugin_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("discordpy")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "DiscordPyPlugin"
