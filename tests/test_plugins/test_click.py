"""Tests for the native Click plugin (``NativePlugin.click()``)."""

from __future__ import annotations

import pytest

from dead_cst import _native as native
from dead_cst.plugins import (
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
)


def test_click_plugin_marks_command_handlers(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click

            @click.group()
            def cli(): pass

            @cli.command()
            def hello(): pass

            @cli.command("bye")
            def goodbye(): pass

            @cli.result_callback()
            def on_result(result, **kwargs): pass

            def helper(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.cli" in reached
    assert "cli.main.hello" in reached
    assert "cli.main.goodbye" in reached
    assert "cli.main.on_result" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "cli.main.helper" not in reached


def test_click_plugin_keeps_handler_dependencies_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/models.py": """
            class Item:
                pass

            class Unused:
                pass
            """,
            "cli/main.py": """
            import click
            from cli.models import Item

            @click.group()
            def cli(): pass

            def build_item() -> Item:
                return Item()

            @cli.command()
            def show():
                return build_item()

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.show" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "cli.main.build_item" in reached
    assert "cli.models.Item" in reached
    # Unrelated module symbol is still dead
    assert "cli.models.Unused" not in reached


def test_click_plugin_reachable_via_explicit_entrypoint(
    make_analysis, write_files, reachable_fqnames
):
    write_files(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click

            @click.group()
            def cli(): pass

            @cli.command()
            def hello(): pass
            """,
        }
    )
    graph = make_analysis(
        plugins=[ExplicitEntrypointPlugin(specs=["cli.main.cli"]), native.NativePlugin.click()]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "cli.main.cli" in reached
    assert "cli.main.hello" in reached


def test_click_plugin_does_not_seed_entrypoint(build_plugin_graph, reachable_fqnames):
    """Without an external reach (no main block, no project.scripts, no -e),
    the Click group itself stays dead -- and so do its commands. Mirrors
    the ``APIRouter`` behavior in ``NativePlugin.fastapi()``."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click

            @click.group()
            def cli(): pass

            @cli.command()
            def orphan(): pass
            """,
        },
        [native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.cli" not in reached
    assert "cli.main.orphan" not in reached


def test_click_plugin_unused_subgroup_stays_dead(build_plugin_graph, reachable_fqnames):
    """A sub-group that's never ``add_command``'d has no path from the root
    group and stays dead, along with its commands."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/sub.py": """
            import click

            @click.group()
            def sub(): pass

            @sub.command()
            def orphan(): pass
            """,
            "cli/main.py": """
            import click

            @click.group()
            def cli(): pass

            @cli.command()
            def hello(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.hello" in reached
    assert "cli.sub.sub" not in reached
    assert "cli.sub.orphan" not in reached


def test_click_plugin_subgroup_reachable_via_add_command(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/sub.py": """
            import click

            @click.group()
            def sub(): pass

            @sub.command()
            def index(): pass

            @sub.command()
            def things(): pass
            """,
            "cli/main.py": """
            import click
            from cli.sub import sub

            @click.group()
            def cli(): pass

            cli.add_command(sub)

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.cli" in reached
    assert "cli.sub.sub" in reached
    assert "cli.sub.index" in reached
    assert "cli.sub.things" in reached


def test_click_plugin_subgroup_via_decorator(build_plugin_graph, reachable_fqnames):
    """``@<group>.group(...)`` registers a nested group inline."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click

            @click.group()
            def cli(): pass

            @cli.group()
            def admin(): pass

            @admin.command()
            def reset(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.cli" in reached
    assert "cli.main.admin" in reached
    assert "cli.main.reset" in reached


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            """
            from click import group

            @group()
            def cli(): pass

            @cli.command()
            def hello(): pass

            if __name__ == "__main__":
                cli()
            """,
            id="from-click-import-group",
        ),
        pytest.param(
            """
            import click as c

            @c.group()
            def cli(): pass

            @cli.command()
            def hello(): pass

            if __name__ == "__main__":
                cli()
            """,
            id="aliased-module-import",
        ),
        pytest.param(
            """
            from click import group as g

            @g()
            def cli(): pass

            @cli.command()
            def hello(): pass

            if __name__ == "__main__":
                cli()
            """,
            id="aliased-decorator-import",
        ),
        pytest.param(
            """
            from click import Group

            cli = Group("cli")

            @cli.command()
            def hello(): pass

            if __name__ == "__main__":
                cli()
            """,
            id="explicit-constructor",
        ),
    ],
)
def test_click_plugin_handles_import_variants(build_plugin_graph, reachable_fqnames, src):
    graph = build_plugin_graph(
        {"cli/__init__.py": "", "cli/main.py": src},
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    assert "cli.main.hello" in reachable_fqnames(graph)


def test_click_plugin_ignores_bare_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            import click

            @click.group()
            def cli(): pass

            def command(fn):
                return fn

            @command
            def looks_like_command(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    # Bare ``@command`` (no attribute access) is not a Click registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_command" not in reachable_fqnames(graph)


def test_click_plugin_ignores_unrelated_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.click()],
    )
    # ``t`` isn't a Click group, so its ``.command`` decorator is ignored.
    assert "pkg.mod.not_a_command" not in reachable_fqnames(graph)


def test_click_plugin_does_nothing_without_click_imports(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Group:
                def command(self):
                    def wrap(fn): return fn
                    return wrap

            cli = Group()

            @cli.command()
            def looks_like_command(): pass
            """,
        },
        [native.NativePlugin.click()],
    )
    # ``cli`` here is not a Click group -- no ``click`` import in scope.
    assert "pkg.mod.looks_like_command" not in reachable_fqnames(graph)


def test_click_plugin_multiple_groups_in_one_module(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click

            @click.group()
            def cli(): pass

            @click.group()
            def other(): pass

            @cli.command()
            def from_cli(): pass

            @other.command()
            def from_other(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    # ``cli`` is reached via the main block; its command is alive.
    assert "cli.main.from_cli" in reached
    # ``other`` is never reached, so its command stays dead -- handler edges
    # are scoped to their own group.
    assert "cli.main.other" not in reached
    assert "cli.main.from_other" not in reached


def test_click_plugin_ignores_import_star(build_plugin_graph, reachable_fqnames):
    """``from click import *`` doesn't bind ``group`` for the plugin's
    purposes. The ``import *`` analyzer logic is pessimistic enough on
    its own; the plugin shouldn't infer Click wiring from the star import."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            from click import *

            @group()
            def cli(): pass

            @cli.command()
            def hello(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    # No instance edge from ``cli`` to ``hello`` because the plugin ignores
    # star imports.
    assert "cli.main.hello" not in reachable_fqnames(graph)


def test_click_plugin_does_nothing_when_click_not_installed(build_plugin_graph, reachable_fqnames):
    """If no file imports ``click``, the plugin's import-graph prefilter
    short-circuits before touching any module."""
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            def helper(): pass
            """,
        },
        [native.NativePlugin.click()],
    )
    assert "pkg.mod.helper" not in reachable_fqnames(graph)


def test_click_plugin_ignores_relative_imports_and_unrelated_names(
    build_plugin_graph, reachable_fqnames
):
    """Imports that look superficially like ``click`` -- relative imports,
    non-group names from ``click`` -- shouldn't make the plugin synthesize
    wiring."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "from click import echo",
            "cli/main.py": """
            from . import click as _local  # relative; not click
            from click import echo  # not group/Group

            def helper(): pass
            """,
        },
        [native.NativePlugin.click()],
    )
    assert "cli.main.helper" not in reachable_fqnames(graph)


def test_click_plugin_no_groups_when_click_imported_but_unused(
    build_plugin_graph, reachable_fqnames
):
    """A file that imports ``click`` but never declares a group is a
    no-op for the plugin (exercising the early return when no group
    declarations or constructions are found)."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click

            echo = click.echo

            def helper(): pass
            """,
        },
        [native.NativePlugin.click()],
    )
    assert "cli.main.helper" not in reachable_fqnames(graph)


def test_click_plugin_ignores_non_group_assignment_shapes(build_plugin_graph, reachable_fqnames):
    """Multi-target assignments, tuple unpacks, bare annotations, and
    nested-attribute decorators all bypass the plugin's wiring rules."""
    graph = build_plugin_graph(
        {
            "cli/__init__.py": "",
            "cli/main.py": """
            import click
            from click import Group

            # Multi-target -- ignored by single-target assignment matching
            x = y = Group("x")

            # Tuple unpacking -- not a Name target
            (a, b) = (Group("a"), Group("b"))

            # Bare annotation, no value -- not an assignment
            late: Group

            # Attribute call that isn't Group: click.echo(...)
            msg = click.echo("hi")

            # Module-prefixed attr that isn't Group/group: ignored
            cfg = click.Context(None)

            # Real group, sanity check
            @click.group()
            def cli(): pass

            @cli.command()
            def hello(): pass

            # Nested-attribute decorator: @cli.a.command(...) is not matched
            class Holder:
                inner = cli
            holder = Holder()

            @holder.inner.command()
            def nested(): pass

            # Known instance but unknown attr
            @cli.unknown_thing()
            def unknown(): pass

            if __name__ == "__main__":
                cli()
            """,
        },
        [MainBlockPlugin(), native.NativePlugin.click()],
    )
    reached = reachable_fqnames(graph)
    assert "cli.main.cli" in reached
    assert "cli.main.hello" in reached
    # Nested-attribute and unknown-attr decorators are ignored.
    assert "cli.main.nested" not in reached
    assert "cli.main.unknown" not in reached
    # Multi-target / tuple / bare-annotation forms don't create groups.
    assert "cli.main.x" not in reached
    assert "cli.main.a" not in reached
    assert "cli.main.late" not in reached
    # ``click.Context(...)`` is not a Group constructor.
    assert "cli.main.cfg" not in reached


def test_click_plugin_loads_via_cli_loader():
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("click")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "click"


def test_click_plugin_factory_returns_native_plugin():
    plugin = native.NativePlugin.click()
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "click"
