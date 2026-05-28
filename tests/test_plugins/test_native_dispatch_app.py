"""Tests for the per-file native ``NativePlugin.dispatch_app(...)``.

This is the native counterpart of
:class:`dead_cst.plugins.DispatchAppPlugin` for the portion of the
work that is genuinely per-file: direct construction promotion
(``app = App(...)`` -> entrypoint) and same-file handler wiring
(``@app.deco(...)`` -> the ``app`` binding). The cross-file pieces —
subclass-closure expansion of ``app_classes`` and the
``app = create_app()`` factory walk — stay on the Python plugin, so
the native plugin takes an *already-resolved* ``module_to_names`` map.

The matchers are purely syntactic (they resolve the ``from flask
import Flask`` statement textually), so none of these tests need a
framework actually installed.
"""

from __future__ import annotations

import textwrap

from dead_cst import Analysis, _native as native
from dead_cst.contrib import flask_plugin


def _flask_native() -> native.NativePlugin:
    return native.NativePlugin.dispatch_app(
        marker_prefix="flask",
        module_to_names={"flask": ["Flask"]},
        registration_decorators=["route", "get", "post"],
        seed_as_entrypoint=True,
    )


def test_native_dispatch_promotes_app_and_wires_handlers(build_plugin_graph, reachable_fqnames):
    """A direct ``app = Flask(...)`` becomes an entrypoint and its
    ``@app.route`` / ``@app.get`` handlers are kept alive; an unrelated
    helper stays dead."""
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
    }
    alive = reachable_fqnames(build_plugin_graph(files, [_flask_native()]))
    assert "app.flask_app.app" in alive
    assert "app.flask_app.hello" in alive
    assert "app.flask_app.get_item" in alive
    assert "app.flask_app.helper" not in alive


def test_native_dispatch_matches_python_plugin_on_direct_case(
    build_plugin_graph, reachable_fqnames
):
    """On a direct-construction Flask app (no subclasses, no factory),
    the per-file native plugin produces the same reachable set as the
    Python ``flask_plugin()`` — the cross-file machinery the native
    plugin omits doesn't fire on this shape."""
    files = {
        "app/__init__.py": "",
        "app/flask_app.py": """
        from flask import Flask

        app = Flask(__name__)

        @app.route("/hello")
        def hello(): pass

        @app.post("/items")
        def create_item(): pass

        def helper(): pass
        """,
    }
    py_alive = reachable_fqnames(build_plugin_graph(files, [flask_plugin()]))
    rs_alive = reachable_fqnames(build_plugin_graph(files, [_flask_native()]))
    assert py_alive == rs_alive


def test_native_dispatch_emits_entrypoint_marker(build_plugin_graph):
    """The synthetic ``<flask-app>:<var-fqname>`` marker lands in the
    graph and carries ``ENTRYPOINT`` — same marker shape the Python
    plugin emits."""
    ctx = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/flask_app.py": """
            from flask import Flask
            app = Flask(__name__)
            @app.route("/x")
            def x(): pass
            """,
        },
        [_flask_native()],
    )
    markers = [n for n in ctx.nodes() if n.fqname == "<flask-app>:app.flask_app.app"]
    assert len(markers) == 1
    assert markers[0].flags & native.NodeFlags.ENTRYPOINT


def test_native_dispatch_seed_false_does_not_promote_app(build_plugin_graph, reachable_fqnames):
    """With ``seed_as_entrypoint=False`` (pure-dispatch frameworks), the
    app is not entrypoint-promoted, so nothing keeps the app or its
    handlers alive on their own — they surface as dead unless reached
    by other means."""
    plugin = native.NativePlugin.dispatch_app(
        marker_prefix="cli",
        module_to_names={"cyclopts": ["App"]},
        registration_decorators=["command"],
        seed_as_entrypoint=False,
    )
    files = {
        "app/__init__.py": "",
        "app/cli.py": """
        from cyclopts import App

        app = App()

        @app.command
        def run(): pass
        """,
    }
    alive = reachable_fqnames(build_plugin_graph(files, [plugin]))
    # No entrypoint marker => app + handler are not kept alive.
    assert "app.cli.app" not in alive
    assert "app.cli.run" not in alive


def test_native_dispatch_name_attribute():
    assert _flask_native().name == "DispatchAppPlugin"


# ---------------------------------------------------------------------------
# Per-file salsa caching: the dispatch impl is invoked through a salsa-tracked
# query keyed on (file, kind=DispatchApp(config_id)). Unchanged files reuse
# cached ops across re_materialize without re-running the impl.
# ---------------------------------------------------------------------------


def _write(path, src: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src).strip() + "\n")


def test_per_file_dispatch_caches_unchanged_files(tmp_path):
    """Editing one file re-runs the per-file dispatch impl for *only*
    that file on ``re_materialize``; every other file's wiring is served
    from the salsa cache."""
    _write(tmp_path / "app/__init__.py", "")
    _write(
        tmp_path / "app/a.py",
        """
        from flask import Flask
        app = Flask(__name__)
        @app.route("/a")
        def a(): pass
        """,
    )
    _write(
        tmp_path / "app/b.py",
        """
        from flask import Flask
        app = Flask(__name__)
        @app.route("/b")
        def b(): pass
        """,
    )
    _write(tmp_path / "app/c.py", "def helper(): pass\n")

    analysis = Analysis(tmp_path, plugins=[_flask_native()])
    native._reset_dispatch_app_run_count()
    analysis.materialize_all()
    # Cold build: the impl runs once per project file (a, b, c, __init__).
    assert native._dispatch_app_run_count() >= 3

    native._reset_dispatch_app_run_count()
    _write(tmp_path / "app/c.py", "def helper(): pass\ndef extra(): pass\n")
    analysis.re_materialize(analysis.materialize_all().detect_changes())
    second_pass = native._dispatch_app_run_count()
    assert second_pass == 1, (
        f"expected exactly 1 per-file re-run (the edited c.py), got {second_pass} "
        "— unchanged a.py/b.py/__init__.py should have hit the salsa cache"
    )


def test_per_file_dispatch_cache_invalidates_on_edit(tmp_path, reachable_fqnames):
    """Editing the dispatch file re-runs its per-file impl and reflects
    the new wiring (a removed handler stops being kept alive)."""
    _write(tmp_path / "app/__init__.py", "")
    _write(
        tmp_path / "app/a.py",
        """
        from flask import Flask
        app = Flask(__name__)
        @app.route("/a")
        def a(): pass
        @app.route("/b")
        def b(): pass
        """,
    )

    analysis = Analysis(tmp_path, plugins=[_flask_native()])
    ctx = analysis.materialize_all()
    assert {"app.a.a", "app.a.b"} <= reachable_fqnames(ctx)

    native._reset_dispatch_app_run_count()
    _write(
        tmp_path / "app/a.py",
        """
        from flask import Flask
        app = Flask(__name__)
        @app.route("/a")
        def a(): pass
        def b(): pass
        """,
    )
    ctx2 = analysis.re_materialize(analysis.materialize_all().detect_changes())
    assert native._dispatch_app_run_count() >= 1  # a.py re-ran
    alive = reachable_fqnames(ctx2)
    assert "app.a.a" in alive
    assert "app.a.b" not in alive  # decorator removed -> no longer wired
