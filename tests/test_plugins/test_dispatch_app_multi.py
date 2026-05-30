"""Multiple native dispatch-app plugins registered together.

Each framework (``NativePlugin.flask()``, ``.fastapi()``, …) is an
independent project-wide native plugin. ``Analysis.materialize_all``
runs them all against the same in-progress graph, so the tests below
pin the observable contract: routes wired correctly across frameworks,
seed vs. non-seed apps behaving per their config, factory-app
promotion, and inactive plugins (framework not imported) no-op cleanly.
"""

from __future__ import annotations

from dead_cst import _native as native


def _plugins_flask_fastapi() -> list[native.NativePlugin]:
    return [native.NativePlugin.flask(), native.NativePlugin.fastapi()]


def _plugins_seed_and_dispatch() -> list[native.NativePlugin]:
    # Mixes seed_as_entrypoint=True (Flask, Celery) with
    # seed_as_entrypoint=False (Cyclopts, Typer) -- exercises both
    # branches of the per-plugin emission.
    return [
        native.NativePlugin.flask(),
        native.NativePlugin.celery(),
        native.NativePlugin.cyclopts(),
        native.NativePlugin.typer(),
    ]


def test_multiple_dispatch_plugins_wire_routes(build_plugin_graph, reachable_fqnames):
    """Two dispatch-app plugins registered together produce a combined
    reachable set covering both frameworks' routes."""
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

    alive = reachable_fqnames(build_plugin_graph(files, _plugins_flask_fastapi()))
    # Every Flask + FastAPI handler should be kept alive.
    for fq in (
        "app.flask_app.hello",
        "app.flask_app.get_item",
        "app.fastapi_app.health",
        "app.fastapi_app.create_item",
    ):
        assert fq in alive, fq
    # Unused helpers stay dead.
    assert "app.flask_app.helper" not in alive
    assert "app.fastapi_app.helper" not in alive


def test_mixed_seed_and_dispatch_modes(build_plugin_graph, reachable_fqnames):
    """``seed_as_entrypoint=True`` (Flask, Celery) and
    ``seed_as_entrypoint=False`` (Cyclopts, Typer) plugins coexisting
    in one registration list."""
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

    alive = reachable_fqnames(build_plugin_graph(files, _plugins_seed_and_dispatch()))
    # seed_as_entrypoint=True frameworks promote their app instances.
    assert "app.flask_app.app" in alive
    assert "app.celery_app.celery_app" in alive
    assert "app.flask_app.hello" in alive
    assert "app.celery_app.add" in alive
    # seed_as_entrypoint=False frameworks don't promote the app -- the
    # Cyclopts / Typer apps stay dead without ``[project.scripts]`` /
    # ``__main__:`` wiring, and that's the documented contract.


def test_factory_chains(build_plugin_graph, reachable_fqnames):
    """Factory-app promotion (``app = create_app()``) survives -- exercises
    the step-6 ``direct_predecessors_idx`` walk."""
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
    alive = reachable_fqnames(build_plugin_graph(files, [native.NativePlugin.flask()]))
    assert "app.factory.index" in alive
    assert "app.factory.create_app" in alive
    assert "app.factory.app" in alive


def test_inactive_dispatch_plugins_skip_cleanly(build_plugin_graph, reachable_fqnames):
    """When no project file imports the framework, every dispatch plugin's
    import guard returns ``False`` and the run is a no-op -- same reachable
    set as registering no plugins at all."""
    files = {
        "app/__init__.py": "",
        "app/plain.py": """
        def unused(): pass

        def used():
            unused()

        used()
        """,
    }
    with_dispatch = reachable_fqnames(build_plugin_graph(files, _plugins_flask_fastapi()))
    without_dispatch = reachable_fqnames(build_plugin_graph(files, []))
    assert with_dispatch == without_dispatch


def test_single_dispatch_plugin_path(build_plugin_graph, reachable_fqnames):
    """A single dispatch plugin wires the same routes a standalone run would."""
    files = {
        "app/__init__.py": "",
        "app/main.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/x")
        def x(): pass
        """,
    }
    alive = reachable_fqnames(build_plugin_graph(files, [native.NativePlugin.flask()]))
    assert "app.main.app" in alive
    assert "app.main.x" in alive


def test_celery_shared_task_fires_alongside_another_plugin(build_plugin_graph, reachable_fqnames):
    """Celery's ``@shared_task`` fan-out fires whether it runs alone or
    alongside another dispatch plugin."""
    files = {
        "app/__init__.py": "",
        "app/tasks.py": """
        from celery import shared_task

        @shared_task
        def send_email(addr): pass
        """,
    }
    alone = reachable_fqnames(build_plugin_graph(files, [native.NativePlugin.celery()]))
    paired = reachable_fqnames(
        build_plugin_graph(files, [native.NativePlugin.celery(), native.NativePlugin.flask()])
    )
    assert "app.tasks.send_email" in alone
    assert "app.tasks.send_email" in paired
