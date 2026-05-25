"""Tests for :func:`slack_bolt_plugin`."""

from __future__ import annotations

import pytest

from dead_cst.contrib import slack_bolt_plugin
from dead_cst.plugins import DispatchAppPlugin


def test_slack_bolt_plugin_marks_handlers(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "bot/__init__.py": "",
            "bot/main.py": """
            import slack_bolt

            app = slack_bolt.App(token="x", signing_secret="y")

            @app.event("app_mention")
            def handle_mention(event, say):
                pass

            @app.message("hello")
            def message_hello(message, say):
                pass

            @app.command("/echo")
            def echo_command(ack, command):
                ack()

            @app.action("button_click")
            def handle_button(ack, body, client):
                ack()

            @app.shortcut("open_modal")
            def open_modal(ack, shortcut, client):
                pass

            @app.view("view_submission")
            def handle_submission(ack, body):
                ack()

            @app.options("external_select")
            def options_handler(ack, body):
                ack({})

            @app.error
            def custom_error_handler(error, body, logger):
                pass

            @app.step(callback_id="cb", edit=None)
            def step_handler(step, edit):
                pass

            @app.function("my_function")
            def function_handler(args):
                pass

            def helper():
                pass
            """,
        },
        [slack_bolt_plugin()],
    )
    reached = reachable_fqnames(graph)
    assert "bot.main.app" in reached
    assert "bot.main.handle_mention" in reached
    assert "bot.main.message_hello" in reached
    assert "bot.main.echo_command" in reached
    assert "bot.main.handle_button" in reached
    assert "bot.main.open_modal" in reached
    assert "bot.main.handle_submission" in reached
    assert "bot.main.options_handler" in reached
    assert "bot.main.custom_error_handler" in reached
    assert "bot.main.step_handler" in reached
    assert "bot.main.function_handler" in reached
    # Undecorated helper not referenced by any handler stays dead.
    assert "bot.main.helper" not in reached


def test_slack_bolt_plugin_handles_async_app(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "bot/__init__.py": "",
            "bot/main.py": """
            from slack_bolt.async_app import AsyncApp

            app = AsyncApp(token="x", signing_secret="y")

            @app.event("app_mention")
            async def handle_mention(event, say):
                pass

            @app.message("hello")
            async def message_hello(message, say):
                pass

            @app.command("/echo")
            async def echo_command(ack, command):
                await ack()

            @app.action("button_click")
            async def handle_button(ack, body, client):
                await ack()
            """,
        },
        [slack_bolt_plugin()],
    )
    reached = reachable_fqnames(graph)
    assert "bot.main.app" in reached
    assert "bot.main.handle_mention" in reached
    assert "bot.main.message_hello" in reached
    assert "bot.main.echo_command" in reached
    assert "bot.main.handle_button" in reached


def test_slack_bolt_plugin_keeps_handler_dependencies_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "bot/__init__.py": "",
            "bot/models.py": """
            class Greeting:
                pass

            class Unused:
                pass
            """,
            "bot/main.py": """
            from slack_bolt import App
            from bot.models import Greeting

            app = App(token="x", signing_secret="y")

            def build_greeting() -> Greeting:
                return Greeting()

            @app.event("app_mention")
            def run(event, say):
                return build_greeting()
            """,
        },
        [slack_bolt_plugin()],
    )
    reached = reachable_fqnames(graph)
    assert "bot.main.run" in reached
    # Symbols transitively referenced from the handler stay alive.
    assert "bot.main.build_greeting" in reached
    assert "bot.models.Greeting" in reached
    # Unrelated module symbol is still dead.
    assert "bot.models.Unused" not in reached


def test_slack_bolt_plugin_auto_seeds_app_as_entrypoint(build_plugin_graph, reachable_fqnames):
    """Direct ``X = App(...)`` is an entrypoint even without a
    ``__main__`` block or ``[project.scripts]`` entry. Slack Bolt apps
    are typically launched via ``app.start(...)`` from a module-level
    invocation, mirroring how Flask / FastAPI / FastMCP apps are
    framework-visible the moment they're constructed."""
    graph = build_plugin_graph(
        {
            "bot/__init__.py": "",
            "bot/main.py": """
            from slack_bolt import App

            app = App(token="x", signing_secret="y")

            @app.event("app_mention")
            def hello(event, say):
                pass
            """,
        },
        [slack_bolt_plugin()],
    )
    reached = reachable_fqnames(graph)
    assert "bot.main.app" in reached
    assert "bot.main.hello" in reached


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            """
            from slack_bolt import App as Bolt

            app = Bolt(token="x", signing_secret="y")

            @app.event("app_mention")
            def hello(event, say): pass
            """,
            id="aliased-class-import",
        ),
        pytest.param(
            """
            import slack_bolt

            app = slack_bolt.App(token="x", signing_secret="y")

            @app.event("app_mention")
            def hello(event, say): pass
            """,
            id="module-import",
        ),
        pytest.param(
            """
            from slack_bolt import App

            app: App = App(token="x", signing_secret="y")

            @app.event("app_mention")
            def hello(event, say): pass
            """,
            id="annotated-assignment",
        ),
        pytest.param(
            """
            from slack_bolt.async_app import AsyncApp as AsyncBolt

            app = AsyncBolt(token="x", signing_secret="y")

            @app.event("app_mention")
            async def hello(event, say): pass
            """,
            id="async-aliased-class-import",
        ),
    ],
)
def test_slack_bolt_plugin_handles_import_variants(build_plugin_graph, reachable_fqnames, src):
    graph = build_plugin_graph(
        {"bot/__init__.py": "", "bot/main.py": src},
        [slack_bolt_plugin()],
    )
    assert "bot.main.hello" in reachable_fqnames(graph)


def test_slack_bolt_plugin_ignores_bare_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from slack_bolt import App

            app = App(token="x", signing_secret="y")

            def event(fn):
                return fn

            @event
            def looks_like_event(): pass
            """,
        },
        [slack_bolt_plugin()],
    )
    # Bare ``@event`` (no attribute access) is not a Bolt registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_event" not in reachable_fqnames(graph)


def test_slack_bolt_plugin_ignores_unrelated_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Thing:
                def event(self, fn):
                    return fn

            t = Thing()

            @t.event
            def not_a_handler(): pass
            """,
        },
        [slack_bolt_plugin()],
    )
    # ``t`` isn't a ``slack_bolt.App`` instance, so its ``.event`` decorator
    # is ignored.
    assert "pkg.mod.not_a_handler" not in reachable_fqnames(graph)


def test_slack_bolt_plugin_does_nothing_without_slack_bolt_imports(
    build_plugin_graph, reachable_fqnames
):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def event(self, name):
                    def wrap(fn): return fn
                    return wrap

            app = App()

            @app.event("ping")
            def looks_like_handler(): pass
            """,
        },
        [slack_bolt_plugin()],
    )
    # ``app`` here is not a slack_bolt App -- no ``slack_bolt`` import in scope.
    assert "pkg.mod.looks_like_handler" not in reachable_fqnames(graph)


def test_slack_bolt_plugin_handles_factory_function(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "bot/__init__.py": "",
            "bot/factory.py": """
            from slack_bolt import App

            def make_app() -> App:
                return App(token="x", signing_secret="y")
            """,
            "bot/main.py": """
            from bot.factory import make_app

            app = make_app()

            @app.event("app_mention")
            def handle_mention(event, say):
                pass

            @app.command("/echo")
            def echo_command(ack, command):
                ack()
            """,
        },
        [slack_bolt_plugin()],
    )
    reached = reachable_fqnames(graph)
    assert "bot.main.app" in reached
    assert "bot.main.handle_mention" in reached
    assert "bot.main.echo_command" in reached


def test_slack_bolt_plugin_factory_returns_configured_dispatch_app():
    plugin = slack_bolt_plugin()
    assert isinstance(plugin, DispatchAppPlugin)
    assert plugin.marker_prefix == "slack-bolt"
    assert plugin.app_classes == ("slack_bolt.App", "slack_bolt.async_app.AsyncApp")
    assert plugin.seed_as_entrypoint is True
