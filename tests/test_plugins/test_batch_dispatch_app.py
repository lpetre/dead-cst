"""Tests for :class:`BatchDispatchAppPlugin`.

The plugin's contract is parity with running each wrapped
``DispatchAppPlugin`` individually — same reachable set, same dead
set — but with fused queries under the hood. The tests below pin
that observable equivalence across the framework patterns that
exercise the trickier branches: ``seed_as_entrypoint=True`` factories,
``seed_as_entrypoint=False`` dispatch, multiple plugins targeting the
same module, and plugins targeting disjoint modules.
"""

from __future__ import annotations

from dead_cst.contrib import (
    CeleryPlugin,
    cyclopts_plugin,
    fastapi_plugin,
    flask_plugin,
    typer_plugin,
)
from dead_cst.plugins import BatchDispatchAppPlugin, DispatchAppPlugin


def _plugins_flask_fastapi() -> list[DispatchAppPlugin]:
    return [flask_plugin(), fastapi_plugin()]


def _plugins_seed_and_dispatch() -> list[DispatchAppPlugin]:
    # Mixes seed_as_entrypoint=True (Flask, Celery) with
    # seed_as_entrypoint=False (Cyclopts, Typer) — exercises both
    # branches of the per-plugin emission.
    return [flask_plugin(), CeleryPlugin(), cyclopts_plugin(), typer_plugin()]


def test_batch_matches_individual_for_flask_fastapi(build_plugin_graph, reachable_fqnames):
    files = {
        "app/__init__.py": "",
        "app/flask_app.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/hello")
        def hello(): pass

        @app.get("/items/<id>")
        def get_item(id): pass

        def helper(): pass
        """,
        "app/fastapi_app.py": """
        from fastapi import FastAPI

        api = FastAPI()

        @api.get("/health")
        def health(): pass

        @api.post("/items")
        def create_item(): pass

        def helper(): pass
        """,
    }

    individual = reachable_fqnames(build_plugin_graph(files, _plugins_flask_fastapi()))
    batched = reachable_fqnames(
        build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=_plugins_flask_fastapi())])
    )
    assert individual == batched


def test_batch_matches_individual_for_seed_and_dispatch_mix(
    build_plugin_graph, reachable_fqnames
):
    files = {
        "app/__init__.py": "",
        "app/flask_app.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/hello")
        def hello(): pass
        """,
        "app/celery_app.py": """
        from celery import Celery

        celery_app = Celery("tasks")

        @celery_app.task
        def add(x, y): return x + y
        """,
        "app/cyclopts_app.py": """
        from cyclopts import App

        cli = App()

        @cli.command
        def hello(): pass

        def helper_used_by_hello():
            hello()
        """,
        "app/typer_app.py": """
        import typer

        app = typer.Typer()

        @app.command()
        def serve(): pass
        """,
    }

    plugins = _plugins_seed_and_dispatch()
    individual = reachable_fqnames(build_plugin_graph(files, plugins))
    batched = reachable_fqnames(
        build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=_plugins_seed_and_dispatch())])
    )
    assert individual == batched


def test_batch_with_factory_chains(build_plugin_graph, reachable_fqnames):
    """Factory-app promotion (``app = create_app()``) must survive the
    batched path — exercises the step-6 ``direct_predecessors_idx`` walk
    via per-plugin emission."""
    files = {
        "app/__init__.py": "",
        "app/factory.py": """
        from flask import Flask

        def create_app() -> Flask:
            return Flask(__name__)

        app = create_app()

        @app.route("/")
        def index(): pass
        """,
    }
    individual = reachable_fqnames(build_plugin_graph(files, [flask_plugin()]))
    batched = reachable_fqnames(
        build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=[flask_plugin()])])
    )
    assert individual == batched
    assert "app.factory.index" in batched
    assert "app.factory.create_app" in batched
    assert "app.factory.app" in batched


def test_batch_skips_when_no_framework_imported(build_plugin_graph, reachable_fqnames):
    """The import-presence guard short-circuits every wrapped plugin
    cleanly. Parity check against the unbatched run is the contract:
    same dead set with or without the framework imports."""
    files = {
        "app/__init__.py": "",
        "app/plain.py": """
        def unused(): pass

        def used():
            unused()

        used()
        """,
    }
    individual = reachable_fqnames(build_plugin_graph(files, _plugins_flask_fastapi()))
    batched = reachable_fqnames(
        build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=_plugins_flask_fastapi())])
    )
    assert individual == batched


def test_batch_with_one_plugin_matches_individual(build_plugin_graph, reachable_fqnames):
    """Degenerate batch (single wrapped plugin) must match the
    individual run exactly — the simplest equivalence check."""
    files = {
        "app/__init__.py": "",
        "app/main.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/x")
        def x(): pass
        """,
    }
    individual = reachable_fqnames(build_plugin_graph(files, [flask_plugin()]))
    batched = reachable_fqnames(
        build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=[flask_plugin()])])
    )
    assert individual == batched
