"""Tests for :class:`CeleryPlugin`."""

from __future__ import annotations

from dead_cst.plugins import CeleryPlugin


def test_celery_plugin_marks_task_handlers(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            from celery import Celery

            app = Celery("worker")

            @app.task
            def send_email(to): pass

            @app.task(bind=True)
            def crunch(self, payload): pass

            @app.task(name="explicit.name")
            def named_task(): pass

            def helper(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.celery.app" in reached
    assert "app.celery.send_email" in reached
    assert "app.celery.crunch" in reached
    assert "app.celery.named_task" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "app.celery.helper" not in reached


def test_celery_plugin_keeps_handler_dependencies_alive(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/models.py": """
            class Job:
                pass

            class Unused:
                pass
            """,
            "app/celery.py": """
            from celery import Celery
            from app.models import Job

            app = Celery("worker")

            def build_job() -> Job:
                return Job()

            @app.task
            def run_job():
                return build_job()
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.celery.run_job" in reached
    assert "app.celery.build_job" in reached
    assert "app.models.Job" in reached
    assert "app.models.Unused" not in reached


def test_celery_plugin_handles_aliased_class_import(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            from celery import Celery as C

            app = C("worker")

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "app.celery.run" in reachable_fqnames(graph)


def test_celery_plugin_handles_module_import(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            import celery

            app = celery.Celery("worker")

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "app.celery.run" in reachable_fqnames(graph)


def test_celery_plugin_handles_annotated_assignment(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            from celery import Celery

            app: Celery = Celery("worker")

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "app.celery.run" in reachable_fqnames(graph)


def test_celery_plugin_marks_shared_tasks(make_analysis, write_files, reachable_fqnames):
    """``@shared_task`` registers into Celery's global registry and is
    invoked by name by the worker -- no owning app variable needed."""
    write_files(
        {
            "app/__init__.py": "",
            "app/tasks.py": """
            from celery import shared_task

            @shared_task
            def bare_decorator(): pass

            @shared_task(name="explicit")
            def called_decorator(): pass

            @shared_task(bind=True, ignore_result=True)
            def with_options(self): pass

            def helper(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.tasks.bare_decorator" in reached
    assert "app.tasks.called_decorator" in reached
    assert "app.tasks.with_options" in reached
    assert "app.tasks.helper" not in reached


def test_celery_plugin_shared_task_aliased(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/tasks.py": """
            from celery import shared_task as task

            @task
            def alias_bare(): pass

            @task(bind=True)
            def alias_called(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.tasks.alias_bare" in reached
    assert "app.tasks.alias_called" in reached


def test_celery_plugin_shared_task_module_form(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/tasks.py": """
            import celery

            @celery.shared_task
            def via_module(): pass

            @celery.shared_task(bind=True)
            def via_module_called(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.tasks.via_module" in reached
    assert "app.tasks.via_module_called" in reached


def test_celery_plugin_ignores_unrelated_task_decorators(
    make_analysis, write_files, reachable_fqnames
):
    """A ``.task`` attribute on a non-Celery instance shouldn't be wired."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Queue:
                def task(self, fn):
                    return fn

            q = Queue()

            @q.task
            def not_a_celery_task(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "pkg.mod.not_a_celery_task" not in reachable_fqnames(graph)


def test_celery_plugin_ignores_bare_shared_task_name(make_analysis, write_files, reachable_fqnames):
    """A local ``shared_task`` name not imported from ``celery`` shouldn't trigger the plugin."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            def shared_task(fn):
                return fn

            @shared_task
            def looks_like_shared(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "pkg.mod.looks_like_shared" not in reachable_fqnames(graph)


def test_celery_plugin_does_nothing_without_celery_imports(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def task(self, fn):
                    return fn

            app = App()

            @app.task
            def fake(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "pkg.mod.fake" not in reachable_fqnames(graph)


def test_celery_plugin_ignores_import_star(make_analysis, write_files, reachable_fqnames):
    """``from celery import *`` shouldn't synthesize Celery wiring."""
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            from celery import *

            app = Celery("worker")

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "app.celery.run" not in reachable_fqnames(graph)


def test_celery_plugin_does_nothing_when_celery_not_imported_anywhere(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            def index(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "pkg.mod.index" not in reachable_fqnames(graph)


def test_celery_plugin_module_prefixed_unknown_attr(make_analysis, write_files, reachable_fqnames):
    """``celery.SomethingElse(...)`` should not be classified as a Celery app."""
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            import celery

            cfg = celery.Task()  # not Celery
            app = celery.Celery("worker")

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.celery.run" in reached
    assert "app.celery.cfg" not in reached


def test_celery_plugin_handles_factory_function(make_analysis, write_files, reachable_fqnames):
    """``def make_celery(): return Celery(...)``."""
    write_files(
        {
            "app/__init__.py": "",
            "app/factory.py": """
            from celery import Celery

            def make_celery() -> Celery:
                return Celery("worker")
            """,
            "app/celery.py": """
            from app.factory import make_celery

            app = make_celery()

            @app.task
            def run(): pass

            @app.task(bind=True)
            def bound(self): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    from dead_cst.analyze import _entrypoint_seeds, _find_reachable as find_reachable

    reached = {
        n.fqname for n in find_reachable(graph, _entrypoint_seeds(graph)) if n.type != "synthetic"
    }
    assert "app.celery.app" in reached
    assert "app.celery.run" in reached
    assert "app.celery.bound" in reached


def test_celery_plugin_factory_in_different_package(make_analysis, write_files, reachable_fqnames):
    """Factory in dep package, consumer in dependent package."""
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            from celery import Celery

            def make_celery() -> Celery:
                return Celery("worker")
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/celery.py": """
            from pkg_a.factory import make_celery

            app = make_celery()

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.celery.app" in reached
    assert "pkg_b.celery.run" in reached


def test_celery_plugin_factory_module_form_in_different_package(
    make_analysis, write_files, reachable_fqnames
):
    """Factory uses ``import celery; celery.Celery()`` -- factory marker required.

    See ``test_flask_plugin_factory_module_form_in_different_package``
    for the rationale; the external-edge classifier drops the
    ``decl='Celery'`` half of the access.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            import celery

            def make_celery():
                return celery.Celery("worker")
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/celery.py": """
            from pkg_a.factory import make_celery

            app = make_celery()

            @app.task
            def run(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.celery.app" in reached
    assert "pkg_b.celery.run" in reached


def test_celery_plugin_ignores_non_decorator_assignment_shapes(
    make_analysis, write_files, reachable_fqnames
):
    """Multi-target / tuple / bare-annotation forms shouldn't produce instance nodes."""
    write_files(
        {
            "app/__init__.py": "",
            "app/celery.py": """
            from celery import Celery

            # Multi-target -- ignored by single_target_assignment
            x = y = Celery("worker")

            # Tuple unpacking -- not a Name target
            (a, b) = (Celery("a"), Celery("b"))

            # Bare annotation, no value -- not an assignment
            late: Celery

            # Real app, used as a sanity check
            app = Celery("worker")

            @app.task
            def run(): pass

            # Nested-attribute decorator: not matched
            class Holder:
                cli = app
            holder = Holder()

            @holder.cli.task
            def nested(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.celery.app" in reached
    assert "app.celery.run" in reached
    assert "app.celery.nested" not in reached
    assert "app.celery.x" not in reached
    assert "app.celery.a" not in reached
    assert "app.celery.late" not in reached


def test_celery_plugin_ignores_non_app_celery_users(make_analysis, write_files, reachable_fqnames):
    """Variables that touch ``celery`` for unrelated reasons stay dead.

    The pending-walk must require a discriminating ``Celery`` import on
    the path before classifying ``X`` as an instance -- otherwise any
    value derived from e.g. ``celery.signals`` would get marked as an
    app.
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from celery import signals

            current = signals

            class Decorated:
                def task(self, fn):
                    return fn

            thing = Decorated()

            @thing.task
            def handler(): pass
            """,
        }
    )
    graph = make_analysis(plugins=[CeleryPlugin()]).materialize_all()
    assert "pkg.mod.handler" not in reachable_fqnames(graph)


def test_celery_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("celery")
    assert isinstance(plugin, CeleryPlugin)
    assert plugin.name == "celery"
