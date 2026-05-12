"""Tests for :class:`FlaskPlugin`."""

from __future__ import annotations

from dead_cst.plugins import FlaskPlugin


def test_flask_plugin_marks_route_handlers(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from flask import Flask

            app = Flask(__name__)

            @app.route("/items")
            def list_items(): pass

            @app.get("/items/<id>")
            def get_item(id): pass

            @app.post("/items")
            def create_item(): pass

            @app.put("/items/<id>")
            def update_item(id): pass

            @app.delete("/items/<id>")
            def delete_item(id): pass

            @app.patch("/items/<id>")
            def patch_item(id): pass

            def helper(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.main.list_items" in reached
    assert "app.main.get_item" in reached
    assert "app.main.create_item" in reached
    assert "app.main.update_item" in reached
    assert "app.main.delete_item" in reached
    assert "app.main.patch_item" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "app.main.helper" not in reached


def test_flask_plugin_marks_lifecycle_and_template_helpers(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from flask import Flask

            app = Flask(__name__)

            @app.before_request
            def before(): pass

            @app.after_request
            def after(response): return response

            @app.teardown_request
            def teardown(exc): pass

            @app.teardown_appcontext
            def teardown_ctx(exc): pass

            @app.errorhandler(404)
            def not_found(error): pass

            @app.context_processor
            def inject(): return {}

            @app.template_filter("reverse")
            def reverse_filter(s): return s[::-1]

            @app.template_test("even")
            def even_test(n): return n % 2 == 0

            @app.template_global()
            def now(): pass

            @app.url_value_preprocessor
            def pull_lang(endpoint, values): pass

            @app.url_defaults
            def add_lang(endpoint, values): pass

            @app.shell_context_processor
            def make_shell_context(): return {}
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.before" in reached
    assert "app.main.after" in reached
    assert "app.main.teardown" in reached
    assert "app.main.teardown_ctx" in reached
    assert "app.main.not_found" in reached
    assert "app.main.inject" in reached
    assert "app.main.reverse_filter" in reached
    assert "app.main.even_test" in reached
    assert "app.main.now" in reached
    assert "app.main.pull_lang" in reached
    assert "app.main.add_lang" in reached
    assert "app.main.make_shell_context" in reached


def test_flask_plugin_keeps_handler_dependencies_alive(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/models.py": """
            class Item:
                pass

            class Unused:
                pass
            """,
            "app/main.py": """
            from flask import Flask
            from app.models import Item

            app = Flask(__name__)

            def build_item() -> Item:
                return Item()

            @app.route("/item")
            def get_item():
                return build_item()
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.get_item" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "app.main.build_item" in reached
    assert "app.models.Item" in reached
    # Unrelated module symbol is still dead
    assert "app.models.Unused" not in reached


def test_flask_plugin_ignores_bare_decorators(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from flask import Flask

            app = Flask(__name__)

            def route(fn):
                return fn

            @route
            def looks_like_route(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    # Bare ``@route`` (no attribute access) is not a Flask registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_flask_plugin_ignores_unrelated_decorators(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Thing:
                def route(self, path):
                    def wrap(fn): return fn
                    return wrap

            t = Thing()

            @t.route("/")
            def not_a_route(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "pkg.mod.not_a_route" not in reachable_fqnames(graph)


def test_flask_plugin_unused_blueprint_stays_dead(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from flask import Blueprint

            bp = Blueprint("bp", __name__)

            @bp.route("/")
            def orphan(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    # No Flask app reaches this blueprint, so it (and its handler) are dead.
    assert "app.routes.bp" not in reached
    assert "app.routes.orphan" not in reached


def test_flask_plugin_blueprint_reachable_via_register_blueprint(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from flask import Blueprint

            bp = Blueprint("bp", __name__)

            @bp.route("/")
            def index(): pass

            @bp.route("/things", methods=["GET", "POST"])
            def things(): pass
            """,
            "app/main.py": """
            from flask import Flask
            from app.routes import bp

            app = Flask(__name__)
            app.register_blueprint(bp)
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.routes.bp" in reached
    assert "app.routes.index" in reached
    assert "app.routes.things" in reached


def test_flask_plugin_handles_aliased_class_import(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from flask import Flask as F

            app = F(__name__)

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "app.main.index" in reachable_fqnames(graph)


def test_flask_plugin_handles_module_import(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            import flask

            app = flask.Flask(__name__)

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "app.main.index" in reachable_fqnames(graph)


def test_flask_plugin_handles_annotated_assignment(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from flask import Flask

            app: Flask = Flask(__name__)

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "app.main.index" in reachable_fqnames(graph)


def test_flask_plugin_does_nothing_without_flask_imports(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def route(self, path):
                    def wrap(fn): return fn
                    return wrap

            app = App()

            @app.route("/")
            def looks_like_route(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    # ``app`` here is not a Flask instance -- no ``flask`` import in scope.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_flask_plugin_ignores_import_star(make_analysis, write_files, reachable_fqnames):
    """``from flask import *`` doesn't bind ``Flask`` for the plugin's
    purposes. The ``import *`` analyzer logic is pessimistic enough on
    its own; the plugin shouldn't infer Flask wiring from a star import."""
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from flask import *

            app = Flask(__name__)

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "app.main.index" not in reachable_fqnames(graph)


def test_flask_plugin_does_nothing_when_flask_not_installed(
    make_analysis, write_files, reachable_fqnames
):
    """If no file imports ``flask``, the plugin's import-graph prefilter
    short-circuits before touching any module."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "pkg.mod.index" not in reachable_fqnames(graph)


def test_flask_plugin_ignores_relative_imports_and_unrelated_names(
    make_analysis, write_files, reachable_fqnames
):
    """Imports that look superficially like ``flask`` -- relative imports,
    sibling packages, non-``Flask`` names from ``flask`` -- shouldn't make
    the plugin synthesize wiring."""
    write_files(
        {
            "app/__init__.py": "from flask import url_for",
            "app/sibling.py": """
            import os as flask  # rebinds the name; not actually flask
            """,
            "app/main.py": """
            from . import flask as _local  # relative; not flask
            from flask import url_for  # not Flask/Blueprint

            def helper(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    # No Flask app or Blueprint anywhere -> helper stays dead.
    assert "app.main.helper" not in reachable_fqnames(graph)


def test_flask_plugin_ignores_non_decorator_assignment_shapes(
    make_analysis, write_files, reachable_fqnames
):
    """Multi-target assignments, tuple unpacks, bare annotations, and
    nested attribute decorators all bypass the plugin's wiring rules."""
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from flask import Flask, Blueprint

            # Multi-target -- ignored by _single_target_assignment
            x = y = Flask(__name__)

            # Tuple unpacking -- not a Name target
            (a, b) = (Flask(__name__), Blueprint("b", __name__))

            # Bare annotation, no value -- not an assignment
            late: Flask

            # Attribute call that isn't Flask/Blueprint
            from flask import url_for
            link = url_for("x")

            # Real app, used as a sanity check that the rest of the module still parses
            app = Flask(__name__)

            @app.route("/")
            def index(): pass

            # Nested-attribute decorator: @app.a.b.route(...) is not matched
            class Holder:
                cli = app
            holder = Holder()

            @holder.cli.route("/nope")
            def nested(): pass

            # Decorator on a known instance but with an unknown attr
            @app.unknown_thing("/nope")
            def unknown(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    # The real app + its ``index`` route are kept alive.
    assert "app.main.app" in reached
    assert "app.main.index" in reached
    # The nested-attribute and unknown-attr decorators are ignored.
    assert "app.main.nested" not in reached
    assert "app.main.unknown" not in reached
    # Multi-target / tuple-target / bare-annotation forms produce no instance
    # nodes, so they all stay dead.
    assert "app.main.x" not in reached
    assert "app.main.a" not in reached
    assert "app.main.late" not in reached


def test_flask_plugin_module_prefixed_unknown_attr(make_analysis, write_files, reachable_fqnames):
    """``flask.SomethingElse(...)`` should not be classified as Flask/Blueprint."""
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            import flask

            cfg = flask.Config({})  # not Flask/Blueprint
            app = flask.Flask(__name__)

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.index" in reached
    # ``cfg`` is not classified as a Flask/Blueprint instance and stays dead.
    assert "app.main.cfg" not in reached


def test_flask_plugin_handles_factory_function(make_analysis, write_files, reachable_fqnames):
    """The canonical Flask factory pattern: ``def create_app(): return Flask(__name__)``."""
    write_files(
        {
            "app/__init__.py": "",
            "app/factory.py": """
            from flask import Flask

            def create_app() -> Flask:
                return Flask(__name__)
            """,
            "app/main.py": """
            from app.factory import create_app

            app = create_app()

            @app.route("/items")
            def list_items(): pass

            @app.post("/items")
            def create_item(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    from dead_cst.analyze import _find_reachable as find_reachable

    reached = {n.fqname for n in find_reachable(graph) if n.type != "synthetic"}
    assert "app.main.app" in reached
    assert "app.main.list_items" in reached
    assert "app.main.create_item" in reached


def test_flask_plugin_factory_returning_blueprint_stays_dead(
    make_analysis, write_files, reachable_fqnames
):
    """Factory-produced Blueprint is treated like a literal Blueprint --
    never auto-seeded as an entrypoint, so an unregistered one stays dead."""
    write_files(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from flask import Blueprint

            def make_bp() -> Blueprint:
                return Blueprint("orphan", __name__)

            bp = make_bp()

            @bp.route("/orphan")
            def orphan(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.routes.bp" not in reached
    assert "app.routes.orphan" not in reached


def test_flask_plugin_ignores_non_app_flask_users(make_analysis, write_files, reachable_fqnames):
    """Variables that touch ``flask`` for unrelated reasons stay dead.

    Walking only to the ``flask`` synthetic isn't enough -- the plugin
    must require a discriminating ``Flask`` / ``Blueprint`` import on
    the path before treating ``X`` as an instance. Otherwise any value
    derived from e.g. ``request`` would get marked as an app.
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from flask import request

            current_path = request

            class Decorated:
                def route(self, path):
                    def wrap(fn): return fn
                    return wrap

            thing = Decorated()

            @thing.route("/")
            def handler(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[FlaskPlugin()]).materialize_all()
    assert "pkg.mod.handler" not in reachable_fqnames(graph)


def test_flask_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("flask")
    assert isinstance(plugin, FlaskPlugin)
    assert plugin.name == "flask"


def test_flask_plugin_factory_in_different_package(make_analysis, write_files, reachable_fqnames):
    """Factory in dep package, consumer in dependent package."""
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            from flask import Flask

            def create_app() -> Flask:
                return Flask(__name__)
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.factory import create_app

            app = create_app()

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.app" in reached
    assert "pkg_b.main.index" in reached


def test_flask_plugin_factory_module_form_in_different_package(
    make_analysis, write_files, reachable_fqnames
):
    """Factory in dep package uses ``import flask; flask.Flask()``.

    Without the factory-marker synthetic the cross-package walk has no
    discriminator -- the external-edge classifier drops the
    ``decl='Flask'`` half of the access.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            import flask

            def create_app():
                return flask.Flask(__name__)
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.factory import create_app

            app = create_app()

            @app.route("/")
            def index(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.app" in reached
    assert "pkg_b.main.index" in reached


def test_flask_plugin_blueprint_factory_in_different_package(
    make_analysis, write_files, reachable_fqnames
):
    """Blueprint factory in dep package, consumer wires it via ``register_blueprint``."""
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/routes.py": """
            from flask import Blueprint

            def make_blueprint() -> Blueprint:
                bp = Blueprint("bp", __name__)

                @bp.route("/")
                def index(): pass

                return bp
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from flask import Flask
            from pkg_a.routes import make_blueprint

            app = Flask(__name__)
            bp = make_blueprint()
            app.register_blueprint(bp)
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.app" in reached
    assert "pkg_b.main.bp" in reached


def test_flask_plugin_orphan_blueprint_factory_stays_dead_cross_package(
    make_analysis, write_files, reachable_fqnames
):
    """Blueprint factory in dep package that nobody ``register_blueprint``s.

    The factory marker carries the ``Blueprint`` kind, which finalize
    explicitly leaves un-entrypointed -- so a downstream consumer that
    only constructs the blueprint (without registering it on a Flask
    app) still gets flagged dead.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/routes.py": """
            from flask import Blueprint

            def make_blueprint() -> Blueprint:
                return Blueprint("bp", __name__)
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.routes import make_blueprint

            bp = make_blueprint()

            @bp.route("/orphan")
            def orphan(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[FlaskPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.bp" not in reached
    assert "pkg_b.main.orphan" not in reached
