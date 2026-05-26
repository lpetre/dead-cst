"""Tests for the harness's automatic batching of
:class:`DispatchAppPlugin` instances.

Registering multiple ``DispatchAppPlugin``\\ s with :class:`Analysis`
no longer requires any wrapper — ``Analysis.materialize_all`` detects
every dispatch plugin, runs one fused ``_gather_batched`` on the main
thread, and fans the per-plugin :meth:`policy` calls out through the
same ``ThreadPoolExecutor`` it uses for non-dispatch plugins.

The tests below pin the observable contract: routes wired correctly
across frameworks, subclass ``policy()`` overrides honored, and inactive
plugins (framework not imported) no-op cleanly.
"""

from __future__ import annotations

from dead_cst.contrib import (
    CeleryPlugin,
    cyclopts_plugin,
    fastapi_plugin,
    flask_plugin,
    typer_plugin,
)
from dead_cst.plugins import DispatchAppPlugin


def _plugins_flask_fastapi() -> list[DispatchAppPlugin]:
    return [flask_plugin(), fastapi_plugin()]


def _plugins_seed_and_dispatch() -> list[DispatchAppPlugin]:
    # Mixes seed_as_entrypoint=True (Flask, Celery) with
    # seed_as_entrypoint=False (Cyclopts, Typer) — exercises both
    # branches of the per-plugin emission.
    return [flask_plugin(), CeleryPlugin(), cyclopts_plugin(), typer_plugin()]


def test_multiple_dispatch_plugins_wire_routes(build_plugin_graph, reachable_fqnames):
    """Two ``DispatchAppPlugin`` instances registered together produce
    a combined reachable set covering both frameworks' routes."""
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
    # seed_as_entrypoint=False frameworks don't promote the app — the
    # handler is only alive because no entrypoint reaches it here.
    # (Cyclopts / Typer apps stay dead without ``[project.scripts]`` /
    # ``__main__:`` wiring, and that's the documented contract.)


def test_factory_chains(build_plugin_graph, reachable_fqnames):
    """Factory-app promotion (``app = create_app()``) survives the
    harness's auto-batched gather — exercises the step-6
    ``direct_predecessors_idx`` walk inside :meth:`policy`."""
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
    alive = reachable_fqnames(build_plugin_graph(files, [flask_plugin()]))
    assert "app.factory.index" in alive
    assert "app.factory.create_app" in alive
    assert "app.factory.app" in alive


def test_inactive_dispatch_plugins_skip_cleanly(build_plugin_graph, reachable_fqnames):
    """When no project file imports the framework, every dispatch
    plugin's ``_is_active(ctx)`` returns ``False`` and the harness's
    shim becomes a no-op — same reachable set as registering no
    plugins at all."""
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
    """Degenerate batch (one ``DispatchAppPlugin``) — the auto-batching
    path with a single plugin must wire the same routes a standalone
    run would."""
    files = {
        "app/__init__.py": "",
        "app/main.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/x")
        def x(): pass
        """,
    }
    alive = reachable_fqnames(build_plugin_graph(files, [flask_plugin()]))
    assert "app.main.app" in alive
    assert "app.main.x" in alive


# ---------------------------------------------------------------------------
# spec / policy split — subclass overrides of policy() must be honored
# uniformly by the harness's auto-batched gather. The previous design
# (composing plugins inside an explicit BatchDispatchAppPlugin wrapper)
# got this wrong; the spec/policy split fixes it and the harness now
# inherits the same guarantee for free.
# ---------------------------------------------------------------------------


def test_subclass_policy_override_fires_under_auto_batch(build_plugin_graph, reachable_fqnames):
    """A subclass that extends ``policy`` to emit extra ops must have
    those ops fire whether it runs alongside other dispatch plugins
    (auto-batched) or as the only plugin."""
    from collections.abc import Iterable
    from dataclasses import dataclass

    from dead_cst import _native as native
    from dead_cst.plugins import DispatchAppGather, DispatchAppPlugin

    @dataclass
    class FlaskPlusMarker(DispatchAppPlugin):
        """Flask spec + an extra synthetic node per discovered app."""

        marker_prefix: str = "flask_plus"
        app_classes: tuple[str, ...] = ("flask.Flask",)
        registration_decorators: frozenset[str] = frozenset({"route"})

        def policy(
            self, ctx: native.ProjectContext, gathered: DispatchAppGather
        ) -> Iterable[native.GraphOp]:
            yield from super().policy(ctx, gathered)
            # Subclass extension: mint a uniquely-named marker per
            # direct construction so we can prove the override fired.
            for ref in gathered.direct:
                yield native.AddNodeByIdx(
                    fqname=f"<flask-plus-extra>:{ref.var_idx}",
                    path=ref.path,
                )

    files = {
        "app/__init__.py": "",
        "app/main.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/x")
        def x(): pass
        """,
    }

    def _extra_fqnames(ctx) -> set[str]:
        return {n.fqname for n in ctx.nodes() if n.fqname.startswith("<flask-plus-extra>:")}

    # Run the subclass on its own AND alongside another DispatchAppPlugin —
    # both paths hit the harness's auto-batching code, but the second
    # path exercises the spec/policy fan-out across N > 1 dispatch plugins.
    solo_ctx = build_plugin_graph(files, [FlaskPlusMarker()])
    batched_ctx = build_plugin_graph(files, [FlaskPlusMarker(), fastapi_plugin()])

    solo_extras = _extra_fqnames(solo_ctx)
    batched_extras = _extra_fqnames(batched_ctx)

    # Both paths emit the same set of extension markers — the
    # additional fastapi_plugin slot adds no Flask markers and doesn't
    # suppress FlaskPlusMarker's extras.
    assert solo_extras
    assert solo_extras == batched_extras
    # And the underlying dispatch reachability matches too.
    assert reachable_fqnames(solo_ctx) == reachable_fqnames(batched_ctx)


def test_celery_shared_task_fires_in_batch(build_plugin_graph, reachable_fqnames):
    """CeleryPlugin's ``@shared_task`` fan-out lives in its policy()
    override. The harness's auto-batching must still trigger that
    fan-out when CeleryPlugin sits in a list of dispatch plugins."""
    files = {
        "app/__init__.py": "",
        "app/tasks.py": """
        from celery import shared_task

        @shared_task
        def send_email(addr): pass
        """,
    }
    alone = reachable_fqnames(build_plugin_graph(files, [CeleryPlugin()]))
    # Pair Celery with another dispatch plugin to hit the multi-plugin
    # auto-batch path explicitly.
    batched = reachable_fqnames(build_plugin_graph(files, [CeleryPlugin(), flask_plugin()]))
    assert "app.tasks.send_email" in alone
    assert "app.tasks.send_email" in batched


def test_spec_property_packages_class_attrs():
    """The ``spec`` property is a derived view of the plugin's class
    attributes — same values, packaged in a frozen DispatchAppSpec.
    Subclasses that override the attributes get a spec that reflects
    those overrides; no extra wiring needed.
    """
    from dataclasses import dataclass

    from dead_cst.plugins import DispatchAppPlugin, DispatchAppSpec

    @dataclass
    class _Custom(DispatchAppPlugin):
        marker_prefix: str = "my"
        app_classes: tuple[str, ...] = ("pkg.App",)
        registration_decorators: frozenset[str] = frozenset({"handler"})
        seed_as_entrypoint: bool = False

    plugin = _Custom()
    spec = plugin.spec
    assert isinstance(spec, DispatchAppSpec)
    assert spec.marker_prefix == "my"
    assert spec.app_classes == ("pkg.App",)
    assert spec.registration_decorators == frozenset({"handler"})
    assert spec.seed_as_entrypoint is False
    # Spec is frozen → hashable; equal-by-value.
    assert spec == DispatchAppSpec(
        marker_prefix="my",
        app_classes=("pkg.App",),
        registration_decorators=frozenset({"handler"}),
        seed_as_entrypoint=False,
    )
