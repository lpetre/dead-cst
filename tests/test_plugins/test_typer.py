"""Tests for :class:`TyperPlugin`."""

from __future__ import annotations

from dead_cst import build_symbol_graph
from dead_cst.plugins import (
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
    TyperPlugin,
)


def test_typer_plugin_marks_command_handlers(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import typer

            app = typer.Typer()

            @app.command()
            def hello(): pass

            @app.command("bye")
            def goodbye(): pass

            @app.callback()
            def root(): pass

            def helper(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.app" in reached
    assert "cli.main.hello" in reached
    assert "cli.main.goodbye" in reached
    assert "cli.main.root" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "cli.main.helper" not in reached


def test_typer_plugin_keeps_handler_dependencies_alive(tmp_path, write_files, reachable_fqnames):
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
            from typer import Typer
            from cli.models import Item

            app = Typer()

            def build_item() -> Item:
                return Item()

            @app.command()
            def show():
                return build_item()

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.show" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "cli.main.build_item" in reached
    assert "cli.models.Item" in reached
    # Unrelated module symbol is still dead
    assert "cli.models.Unused" not in reached


def test_typer_plugin_reachable_via_explicit_entrypoint(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from typer import Typer

            app = Typer()

            @app.command()
            def hello(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[ExplicitEntrypointPlugin(specs=["cli.main.app"]), TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.app" in reached
    assert "cli.main.hello" in reached


def test_typer_plugin_does_not_seed_entrypoint(tmp_path, write_files, reachable_fqnames):
    """Without an external reach (no main block, no project.scripts, no -e),
    the Typer instance itself stays dead -- and so do its commands. Mirrors
    the ``APIRouter`` behavior in :class:`FastAPIPlugin`."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from typer import Typer

            app = Typer()

            @app.command()
            def orphan(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.app" not in reached
    assert "cli.main.orphan" not in reached


def test_typer_plugin_unused_subapp_stays_dead(tmp_path, write_files, reachable_fqnames):
    """A sub-Typer that's never ``add_typer``'d has no path from the root
    app and stays dead, along with its commands."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/sub.py": """
            from typer import Typer

            sub = Typer()

            @sub.command()
            def orphan(): pass
            """,
            "cli/main.py": """
            from typer import Typer

            app = Typer()

            @app.command()
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.hello" in reached
    assert "cli.sub.sub" not in reached
    assert "cli.sub.orphan" not in reached


def test_typer_plugin_subapp_reachable_via_add_typer(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/sub.py": """
            from typer import Typer

            sub = Typer()

            @sub.command()
            def index(): pass

            @sub.command()
            def things(): pass
            """,
            "cli/main.py": """
            from typer import Typer
            from cli.sub import sub

            app = Typer()
            app.add_typer(sub, name="sub")

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.app" in reached
    assert "cli.sub.sub" in reached
    assert "cli.sub.index" in reached
    assert "cli.sub.things" in reached


def test_typer_plugin_handles_aliased_class_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from typer import Typer as T

            app = T()

            @app.command()
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_typer_plugin_handles_aliased_module_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import typer as ty

            app = ty.Typer()

            @app.command()
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_typer_plugin_handles_annotated_assignment(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import typer

            app: typer.Typer = typer.Typer()

            @app.command()
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_typer_plugin_ignores_bare_decorators(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from typer import Typer

            app = Typer()

            def command(fn):
                return fn

            @command
            def looks_like_command(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    # Bare ``@command`` (no attribute access) is not a Typer registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_command" not in reachable_fqnames(graph)


def test_typer_plugin_ignores_unrelated_decorators(tmp_path, write_files, reachable_fqnames):
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
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[TyperPlugin()],
        project_root=tmp_path,
    )
    # ``t`` isn't a ``Typer`` instance, so its ``.command`` decorator is ignored.
    assert "pkg.mod.not_a_command" not in reachable_fqnames(graph)


def test_typer_plugin_does_nothing_without_typer_imports(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def command(self):
                    def wrap(fn): return fn
                    return wrap

            app = App()

            @app.command()
            def looks_like_command(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[TyperPlugin()],
        project_root=tmp_path,
    )
    # ``app`` here is not a Typer instance -- no ``typer`` import in scope.
    assert "pkg.mod.looks_like_command" not in reachable_fqnames(graph)


def test_typer_plugin_multiple_instances_in_one_module(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from typer import Typer

            app = Typer()
            other = Typer()

            @app.command()
            def from_app(): pass

            @other.command()
            def from_other(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    # ``app`` is reached via the main block; its command is alive.
    assert "cli.main.from_app" in reached
    # ``other`` is never reached, so its command stays dead -- handler edges
    # are scoped to their own instance.
    assert "cli.main.other" not in reached
    assert "cli.main.from_other" not in reached


def test_typer_plugin_ignores_import_star(tmp_path, write_files, reachable_fqnames):
    """``from typer import *`` doesn't bind ``Typer`` for the plugin's
    purposes. The ``import *`` analyzer logic is pessimistic enough on its
    own; the plugin shouldn't infer Typer wiring from the star import."""
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from typer import *

            app = Typer()

            @app.command()
            def hello(): pass

            if __name__ == "__main__":
                app()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), TyperPlugin()],
        project_root=tmp_path,
    )
    # No instance edge from ``app`` to ``hello`` because the plugin ignores
    # star imports. ``hello`` is not referenced by anything reachable.
    assert "cli.main.hello" not in reachable_fqnames(graph)


def test_typer_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("typer")
    assert isinstance(plugin, TyperPlugin)
    assert plugin.name == "typer"
