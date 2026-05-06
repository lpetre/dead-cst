"""Tests for :class:`CycloptsPlugin`."""

from __future__ import annotations

from dead_cst import Analysis
from dead_cst.resolvers import ManualResolver
from dead_cst.plugins import (
    CycloptsPlugin,
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
)


def test_cyclopts_plugin_marks_command_handlers(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import cyclopts

            app = cyclopts.App()

            @app.command
            def hello(): pass

            @app.command(name="bye")
            def goodbye(): pass

            @app.default
            def root(): pass

            def helper(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.app" in reached
    assert "cli.main.hello" in reached
    assert "cli.main.goodbye" in reached
    assert "cli.main.root" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "cli.main.helper" not in reached


def test_cyclopts_plugin_keeps_handler_dependencies_alive(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/models.py": """
            class Item:
                pass

            class Unused:
                pass
            """,
            "cli/main.py": """
            from cyclopts import App
            from cli.models import Item

            app = App()

            def build_item() -> Item:
                return Item()

            @app.command
            def show():
                return build_item()

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.show" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "cli.main.build_item" in reached
    assert "cli.models.Item" in reached
    # Unrelated module symbol is still dead
    assert "cli.models.Unused" not in reached


def test_cyclopts_plugin_reachable_via_explicit_entrypoint(
    tmp_path, write_files, reachable_fqnames
):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from cyclopts import App

            app = App()

            @app.command
            def hello(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[ExplicitEntrypointPlugin(specs=["cli.main.app"]), CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.app" in reached
    assert "cli.main.hello" in reached


def test_cyclopts_plugin_does_not_seed_entrypoint(tmp_path, write_files, reachable_fqnames):
    """Without an external reach (no main block, no project.scripts, no -e),
    the cyclopts instance itself stays dead -- and so do its commands."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from cyclopts import App

            app = App()

            @app.command
            def orphan(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.app" not in reached
    assert "cli.main.orphan" not in reached


def test_cyclopts_plugin_unused_subapp_stays_dead(tmp_path, write_files, reachable_fqnames):
    """A sub-App that's never registered has no path from the root app and
    stays dead, along with its commands."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/sub.py": """
            from cyclopts import App

            sub = App()

            @sub.command
            def orphan(): pass
            """,
            "cli/main.py": """
            from cyclopts import App

            app = App()

            @app.command
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.hello" in reached
    assert "cli.sub.sub" not in reached
    assert "cli.sub.orphan" not in reached


def test_cyclopts_plugin_subapp_reachable_via_command_attach(
    tmp_path, write_files, reachable_fqnames
):
    """``app.command(sub)`` attaches a sub-App by reference -- the analyzer
    tracks that ordinary call argument, so the sub-App becomes reachable
    through the root app and its own ``@sub.command`` handlers go live."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/sub.py": """
            from cyclopts import App

            sub = App()

            @sub.command
            def index(): pass

            @sub.command
            def things(): pass
            """,
            "cli/main.py": """
            from cyclopts import App
            from cli.sub import sub

            app = App()
            app.command(sub)

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.app" in reached
    assert "cli.sub.sub" in reached
    assert "cli.sub.index" in reached
    assert "cli.sub.things" in reached


def test_cyclopts_plugin_handles_aliased_class_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from cyclopts import App as A

            app = A()

            @app.command
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_cyclopts_plugin_handles_aliased_module_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import cyclopts as cy

            app = cy.App()

            @app.command
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_cyclopts_plugin_handles_annotated_assignment(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import cyclopts

            app: cyclopts.App = cyclopts.App()

            @app.command
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_cyclopts_plugin_ignores_bare_decorators(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from cyclopts import App

            app = App()

            def command(fn):
                return fn

            @command
            def looks_like_command(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    # Bare ``@command`` (no attribute access) is not a cyclopts registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_command" not in reachable_fqnames(graph)


def test_cyclopts_plugin_ignores_unrelated_decorators(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Thing:
                def command(self, fn):
                    return fn

            t = Thing()

            @t.command
            def not_a_command(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[CycloptsPlugin()],
    ).materialize_all()
    # ``t`` isn't a cyclopts ``App`` instance, so its ``.command`` decorator is ignored.
    assert "pkg.mod.not_a_command" not in reachable_fqnames(graph)


def test_cyclopts_plugin_does_nothing_without_cyclopts_imports(
    tmp_path, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def command(self, fn):
                    return fn

            app = App()

            @app.command
            def looks_like_command(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[CycloptsPlugin()],
    ).materialize_all()
    # ``app`` here is not a cyclopts instance -- no ``cyclopts`` import in scope.
    assert "pkg.mod.looks_like_command" not in reachable_fqnames(graph)


def test_cyclopts_plugin_multiple_instances_in_one_module(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from cyclopts import App

            app = App()
            other = App()

            @app.command
            def from_app(): pass

            @other.command
            def from_other(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    # ``app`` is reached via the main block; its command is alive.
    assert "cli.main.from_app" in reached
    # ``other`` is never reached, so its command stays dead -- handler edges
    # are scoped to their own instance.
    assert "cli.main.other" not in reached
    assert "cli.main.from_other" not in reached


def test_cyclopts_plugin_ignores_import_star(tmp_path, write_files, reachable_fqnames):
    """``from cyclopts import *`` doesn't bind ``App`` for the plugin's
    purposes. The ``import *`` analyzer logic is pessimistic enough on its
    own; the plugin shouldn't infer cyclopts wiring from the star import."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from cyclopts import *

            app = App()

            @app.command
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[MainBlockPlugin(), CycloptsPlugin()],
    ).materialize_all()
    # No instance edge from ``app`` to ``hello`` because the plugin ignores
    # star imports. ``hello`` is not referenced by anything reachable.
    assert "cli.main.hello" not in reachable_fqnames(graph)


def test_cyclopts_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("cyclopts")
    assert isinstance(plugin, CycloptsPlugin)
    assert plugin.name == "cyclopts"
