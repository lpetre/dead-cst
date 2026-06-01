"""Tests for :class:`InitSubclassPlugin`."""

from __future__ import annotations

from dead_cst import _native as native
from dead_cst.plugins import (
    InitSubclassPlugin,
    MainBlockPlugin,
)
from dead_cst.plugins.init_subclass import INIT_SUBCLASS_PREFIX


def test_init_subclass_keeps_subclass_alive_via_parent(
    make_analysis, write_files, reachable_fqnames
):
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plugin"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    # Parent is alive via the explicit entrypoint, so its subclasses come along.
    assert "pkg.base.Plugin" in reached
    assert "pkg.impls.Foo" in reached
    assert "pkg.impls.Bar" in reached
    # A class without any registry-class base stays dead.
    assert "pkg.impls.Unrelated" not in reached


def test_init_subclass_transitive_subclasses(make_analysis, write_files, reachable_fqnames):
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.mod.Root"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.mod.Root" in reached
    assert "pkg.mod.Mid" in reached
    assert "pkg.mod.Leaf" in reached


def test_init_subclass_does_not_seed_parent_entrypoint(build_plugin_graph, reachable_fqnames):
    """The plugin only emits inverse edges; if nothing else keeps the parent
    alive, neither parent nor subclasses become reachable."""
    graph = build_plugin_graph(
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
        },
        [InitSubclassPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.base.Plugin" not in reached
    assert "pkg.impls.Foo" not in reached


def test_init_subclass_via_main_block(build_plugin_graph, reachable_fqnames):
    """End-to-end: parent reached via a __main__ block, subclasses come along."""
    graph = build_plugin_graph(
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
        },
        [MainBlockPlugin(), InitSubclassPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.base.Handler" in reached
    assert "pkg.impls.JSONHandler" in reached
    assert "pkg.impls.XMLHandler" in reached


def test_init_subclass_aliased_import(make_analysis, write_files, reachable_fqnames):
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plugin"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    assert "pkg.impls.Foo" in reachable_fqnames(graph)


def test_init_subclass_dotted_attribute_base(make_analysis, write_files, reachable_fqnames):
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plugin"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    assert "pkg.impls.Foo" in reachable_fqnames(graph)


def test_init_subclass_class_without_init_subclass_no_edges(
    make_analysis, write_files, reachable_fqnames
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plain"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.base.Plain" in reached
    assert "pkg.impls.Sub" not in reached


def test_init_subclass_keeps_subclass_method_references_alive(
    make_analysis, write_files, reachable_fqnames
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plugin"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg.impls.Foo" in reached
    assert "pkg.helpers.helper" in reached
    assert "pkg.helpers.unused_helper" not in reached


def test_init_subclass_subscripted_base(make_analysis, write_files, reachable_fqnames):
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plugin"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    assert "pkg.impls.Foo" in reachable_fqnames(graph)


def test_init_subclass_marker_in_predecessor_chain(make_analysis, write_files, predecessors_of):
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
    graph = make_analysis(
        plugins=[
            native.NativePlugin.explicit([], ["pkg.base.Plugin"], []),
            InitSubclassPlugin(),
        ]
    ).materialize_all()
    foo = next(n for n in graph.nodes() if n.fqname == "pkg.impls.Foo")
    preds = predecessors_of(graph, foo)
    marker = next(
        (p for p in preds if p.kind == "synthetic" and p.fqname.startswith(INIT_SUBCLASS_PREFIX)),
        None,
    )
    assert marker is not None, f"expected a marker predecessor, got {preds!r}"
    assert marker.fqname == f"{INIT_SUBCLASS_PREFIX}pkg.base.Plugin"

    marker_preds = predecessors_of(graph, marker)
    parent = next(p for p in marker_preds if p.fqname == "pkg.base.Plugin")
    assert parent.kind == "class"


def test_init_subclass_loads_via_cli_loader():
    from dead_cst import _native as native
    from dead_cst.cli import _load_plugin

    plugin = _load_plugin("init_subclass")
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "InitSubclassPlugin"
