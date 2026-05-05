"""Tests for :class:`InitSubclassPlugin`."""

from __future__ import annotations

from dead_cst import build_symbol_graph
from dead_cst.plugins import (
    ExplicitEntrypointPlugin,
    InitSubclassPlugin,
    MainBlockPlugin,
)
from dead_cst.plugins.init_subclass import INIT_SUBCLASS_PREFIX


def test_init_subclass_keeps_subclass_alive_via_parent(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plugin:
                registry: list[type] = []

                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
                    Plugin.registry.append(cls)
            """,
            "pkg/impls.py": """
            from pkg.base import Plugin

            class Foo(Plugin):
                pass

            class Bar(Plugin):
                pass

            class Unrelated:
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plugin"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    # Parent is alive via the explicit entrypoint, so its subclasses come along.
    assert "pkg.base.Plugin" in reached
    assert "pkg.impls.Foo" in reached
    assert "pkg.impls.Bar" in reached
    # A class without any registry-class base stays dead.
    assert "pkg.impls.Unrelated" not in reached


def test_init_subclass_transitive_subclasses(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Root:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)

            class Mid(Root):
                pass

            class Leaf(Mid):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.mod.Root"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.mod.Root" in reached
    assert "pkg.mod.Mid" in reached
    assert "pkg.mod.Leaf" in reached


def test_init_subclass_does_not_seed_parent_entrypoint(tmp_path, write_files, reachable_fqnames):
    """The plugin only emits inverse edges; if nothing else keeps the parent
    alive, neither parent nor subclasses become reachable."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plugin:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
            """,
            "pkg/impls.py": """
            from pkg.base import Plugin

            class Foo(Plugin):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[InitSubclassPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.base.Plugin" not in reached
    assert "pkg.impls.Foo" not in reached


def test_init_subclass_via_main_block(tmp_path, write_files, reachable_fqnames):
    """End-to-end: parent reached via a __main__ block, subclasses come along."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Handler:
                registry: list[type] = []

                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
                    Handler.registry.append(cls)
            """,
            "pkg/impls.py": """
            from pkg.base import Handler

            class JSONHandler(Handler):
                pass

            class XMLHandler(Handler):
                pass
            """,
            "pkg/main.py": """
            from pkg.base import Handler
            from pkg import impls  # noqa: F401  -- side-effect import

            def run():
                for cls in Handler.registry:
                    cls()

            if __name__ == "__main__":
                run()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[MainBlockPlugin(), InitSubclassPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.base.Handler" in reached
    assert "pkg.impls.JSONHandler" in reached
    assert "pkg.impls.XMLHandler" in reached


def test_init_subclass_aliased_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plugin:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
            """,
            "pkg/impls.py": """
            from pkg.base import Plugin as P

            class Foo(P):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plugin"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    assert "pkg.impls.Foo" in reachable_fqnames(graph)


def test_init_subclass_dotted_attribute_base(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plugin:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
            """,
            "pkg/impls.py": """
            from pkg import base

            class Foo(base.Plugin):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plugin"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    assert "pkg.impls.Foo" in reachable_fqnames(graph)


def test_init_subclass_class_without_init_subclass_no_edges(
    tmp_path, write_files, reachable_fqnames
):
    """A regular base class with no ``__init_subclass__`` produces no edges,
    so a subclass that nothing else references stays dead even when the
    parent is alive."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plain:
                pass
            """,
            "pkg/impls.py": """
            from pkg.base import Plain

            class Sub(Plain):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plain"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.base.Plain" in reached
    assert "pkg.impls.Sub" not in reached


def test_init_subclass_keeps_subclass_method_references_alive(
    tmp_path, write_files, reachable_fqnames
):
    """Methods of a subclass kept alive by ``__init_subclass__`` reach
    through to whatever they reference -- exactly the registry-dispatch
    pattern this plugin is meant to support."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plugin:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
            """,
            "pkg/helpers.py": """
            def helper():
                return 42

            def unused_helper():
                return 0
            """,
            "pkg/impls.py": """
            from pkg.base import Plugin
            from pkg.helpers import helper

            class Foo(Plugin):
                def run(self):
                    return helper()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plugin"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "pkg.impls.Foo" in reached
    assert "pkg.helpers.helper" in reached
    assert "pkg.helpers.unused_helper" not in reached


def test_init_subclass_subscripted_base(tmp_path, write_files, reachable_fqnames):
    """``Subscript`` bases like ``Generic[T]`` are unwrapped to their value
    so a class that inherits from a generic registry base is still wired."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            from typing import Generic, TypeVar

            T = TypeVar("T")

            class Plugin(Generic[T]):
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
            """,
            "pkg/impls.py": """
            from pkg.base import Plugin

            class Foo(Plugin[int]):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plugin"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    assert "pkg.impls.Foo" in reachable_fqnames(graph)


def test_init_subclass_marker_in_predecessor_chain(tmp_path, write_files):
    """Reachability of a subclass routes through a labeled marker node so
    ``why-alive`` chains read ``Foo <- <__init_subclass__>:Plugin <- Plugin``."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/base.py": """
            class Plugin:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__(**kwargs)
            """,
            "pkg/impls.py": """
            from pkg.base import Plugin

            class Foo(Plugin):
                pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[
            ExplicitEntrypointPlugin(specs=["pkg.base.Plugin"]),
            InitSubclassPlugin(),
        ],
        project_root=tmp_path,
    )
    foo = next(n for n in graph.nodes if n.fqname == "pkg.impls.Foo")
    preds = list(graph.predecessors(foo))
    marker = next(
        (p for p in preds if p.type == "synthetic" and p.fqname.startswith(INIT_SUBCLASS_PREFIX)),
        None,
    )
    assert marker is not None, f"expected a marker predecessor, got {preds!r}"
    assert marker.fqname == f"{INIT_SUBCLASS_PREFIX}pkg.base.Plugin"

    marker_preds = list(graph.predecessors(marker))
    parent = next(p for p in marker_preds if p.fqname == "pkg.base.Plugin")
    assert parent.type == "class"


def test_init_subclass_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("init_subclass")
    assert isinstance(plugin, InitSubclassPlugin)
    assert plugin.name == "init_subclass"
