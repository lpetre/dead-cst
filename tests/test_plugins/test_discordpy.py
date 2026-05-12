"""Tests for :class:`DiscordPyPlugin`."""

from __future__ import annotations

from dead_cst.plugins import DiscordPyPlugin


def test_discordpy_plugin_marks_bot_command_handlers(make_analysis, write_files, reachable_fqnames):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.bot" in reached
    assert "app.main.ping" in reached
    assert "app.main.on_ready" in reached
    assert "app.main.echo" in reached
    # Undecorated helper stays dead.
    assert "app.main.helper" not in reached


def test_discordpy_plugin_marks_slash_command_tree_handlers(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.slash_ping" in reached
    assert "app.main.greet" in reached


def test_discordpy_plugin_handles_direct_client(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            import discord

            client = discord.Client(intents=discord.Intents.default())

            @client.event
            async def on_ready():
                pass
            """,
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.client" in reached
    assert "app.main.on_ready" in reached


def test_discordpy_plugin_handles_aliased_class_import(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from discord.ext.commands import Bot as B

            bot = B(command_prefix="!")

            @bot.command()
            async def ping(ctx):
                pass
            """,
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    assert "app.main.ping" in reachable_fqnames(graph)


def test_discordpy_plugin_handles_autosharded_variants(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.b_cmd" in reached
    assert "app.main.on_ready" in reached


def test_discordpy_plugin_keeps_cog_class_alive(make_analysis, write_files, reachable_fqnames):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    # Cog class + module-level setup function both kept alive by the
    # cog-module synthetic entrypoint.
    assert "cogs.greetings.Greetings" in reached
    assert "cogs.greetings.setup" in reached


def test_discordpy_plugin_marks_teardown_alongside_setup(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cogs.admin.Admin" in reached
    assert "cogs.admin.setup" in reached
    assert "cogs.admin.teardown" in reached


def test_discordpy_plugin_setup_without_cog_stays_dead(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": """
            from discord.ext import commands

            # No Cog subclass -- a function happening to be named ``setup``
            # is not a discord.py extension entrypoint.
            def setup(bot):
                pass
            """,
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    assert "pkg.helpers.setup" not in reachable_fqnames(graph)


def test_discordpy_plugin_load_extension_pulls_target_module(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.cogs.greet" in reached
    assert "app.cogs.greet.setup" in reached
    assert "app.cogs.greet.helper_for_greet" in reached
    assert "app.cogs.admin.setup" in reached
    assert "app.cogs.admin.admin_helper" in reached
    assert "app.cogs.misc.setup" in reached
    assert "app.cogs.misc.misc_helper" in reached


def test_discordpy_plugin_load_extension_non_literal_dropped(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    # Non-literal argument is dropped silently; nothing keeps the cog alive.
    assert "app.cogs.dynamic.setup" not in reachable_fqnames(graph)


def test_discordpy_plugin_ignores_unrelated_bare_decorators(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    assert "pkg.mod.not_a_handler" not in reachable_fqnames(graph)


def test_discordpy_plugin_ignores_files_without_discord_imports(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    assert "pkg.target.setup" not in reachable_fqnames(graph)


def test_discordpy_plugin_handler_dependencies_stay_alive(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
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
        }
    )
    graph = make_analysis(plugins=[DiscordPyPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.hello" in reached
    # Handler transitively pulls render_greeting alive.
    assert "app.util.render_greeting" in reached
    # Unrelated util stays dead.
    assert "app.util.unused_util" not in reached


def test_discordpy_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("discordpy")
    assert isinstance(plugin, DiscordPyPlugin)
