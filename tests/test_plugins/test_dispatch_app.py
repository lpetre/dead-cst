"""Tests for the generic dispatch-app factory (``NativePlugin.dispatch_app()``).

The dedicated ``NativePlugin.flask()`` … ``celery()`` factories are thin wrappers
over the same engine; these tests drive the engine directly through the public
``dispatch_app`` factory to prove the Python-supplied config (``app_classes``,
``registration_decorators``, ``seed_as_entrypoint``) is what actually steers it.
"""

from __future__ import annotations

from dead_cst import _native as native

# A custom framework reusing Flask's class (known to resolve) but a *restricted*
# decorator set: only ``route``/``get`` register handlers, so a ``@app.post``
# handler must stay dead — that gap is the proof the config is honoured.
_SOURCE = {
    "svc/__init__.py": "",
    "svc/main.py": """
    from flask import Flask

    app = Flask(__name__)

    @app.route("/a")
    def handler_route(): pass

    @app.get("/b")
    def handler_get(): pass

    @app.post("/c")
    def handler_post(): pass

    def helper(): pass
    """,
}


def test_dispatch_app_threads_app_classes_and_decorators(build_plugin_graph, reachable_fqnames):
    plugin = native.NativePlugin.dispatch_app(
        name="myframework",
        app_classes=["flask.Flask"],
        registration_decorators=["route", "get"],
        seed_as_entrypoint=True,
    )
    reached = reachable_fqnames(build_plugin_graph(_SOURCE, [plugin]))

    # seed_as_entrypoint=True promotes the constructed app instance.
    assert "svc.main.app" in reached
    # Decorators in the config wire their handlers alive.
    assert "svc.main.handler_route" in reached
    assert "svc.main.handler_get" in reached
    # `post` is absent from registration_decorators → its handler stays dead.
    assert "svc.main.handler_post" not in reached
    # Undecorated, unreferenced helper stays dead.
    assert "svc.main.helper" not in reached


def test_dispatch_app_seed_as_entrypoint_false_does_not_seed(build_plugin_graph, reachable_fqnames):
    # Same source, same decorators, but pure-dispatch mode: the app instance is
    # not auto-seeded, so with no other entrypoint nothing comes alive.
    plugin = native.NativePlugin.dispatch_app(
        name="myframework",
        app_classes=["flask.Flask"],
        registration_decorators=["route", "get"],
        seed_as_entrypoint=False,
    )
    reached = reachable_fqnames(build_plugin_graph(_SOURCE, [plugin]))

    assert "svc.main.app" not in reached
    assert "svc.main.handler_route" not in reached
    assert "svc.main.handler_get" not in reached
