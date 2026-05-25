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


def test_batch_matches_individual_for_seed_and_dispatch_mix(build_plugin_graph, reachable_fqnames):
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


# ---------------------------------------------------------------------------
# spec / policy split — subclass overrides of policy() must be honored
# uniformly by both the standalone DispatchAppPlugin.run path and the
# batched BatchDispatchAppPlugin.run path. This is the bit the previous
# "compose existing plugins" design got wrong: it called
# ``plugin._emit_ops`` rather than letting subclasses override behavior.
# ---------------------------------------------------------------------------


def test_subclass_policy_override_honored_by_batch(build_plugin_graph, reachable_fqnames):
    """A subclass that extends ``policy`` to emit additional ops should
    have those extra ops fire whether it's invoked standalone OR
    wrapped in ``BatchDispatchAppPlugin``. This is the regression the
    spec/policy split fixes — the previous batch design only invoked
    the standard policy and skipped subclass extensions.
    """
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

    standalone_ctx = build_plugin_graph(files, [FlaskPlusMarker()])
    batched_ctx = build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=[FlaskPlusMarker()])])

    standalone_extras = _extra_fqnames(standalone_ctx)
    batched_extras = _extra_fqnames(batched_ctx)

    # Both paths must emit the same set of extension markers.
    assert standalone_extras
    assert standalone_extras == batched_extras
    # And the underlying dispatch reachability must still match.
    assert reachable_fqnames(standalone_ctx) == reachable_fqnames(batched_ctx)


def test_celery_shared_task_fires_in_batch(build_plugin_graph, reachable_fqnames):
    """CeleryPlugin's ``@shared_task`` fan-out lives in its policy()
    override. Wrapping it in BatchDispatchAppPlugin must still trigger
    that fan-out — the original 'compose existing plugins' design
    skipped subclass run() overrides; the spec/policy split fixes it.
    """
    files = {
        "app/__init__.py": "",
        "app/tasks.py": """
        from celery import shared_task

        @shared_task
        def send_email(addr): pass
        """,
    }
    individual = reachable_fqnames(build_plugin_graph(files, [CeleryPlugin()]))
    batched = reachable_fqnames(
        build_plugin_graph(files, [BatchDispatchAppPlugin(plugins=[CeleryPlugin()])])
    )
    assert "app.tasks.send_email" in individual
    assert "app.tasks.send_email" in batched
    assert individual == batched


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
