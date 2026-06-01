"""Tests for the native ``main_block`` plugin (``NativePlugin.main_block``)."""

from __future__ import annotations

from dead_cst import _native as native


def test_main_block_plugin_marks_module_entrypoint(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            def unused(): pass
            if __name__ == "__main__":
                main()
            """,
            "pkg/other.py": "def g(): pass",
        },
        [native.NativePlugin.main_block()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.script" in reached
    assert "pkg.script.main" in reached
    # `unused` has no reference from inside __main__, so stays dead
    assert "pkg.script.unused" not in reached
    # modules without a main block are not entrypoints
    assert "pkg.other" not in reached


def test_main_block_keeps_block_decls_alive(build_plugin_graph, reachable_fqnames):
    # ``app`` is bound inside the block but never read elsewhere in the
    # module, so the only edges pointing in are the visitor's
    # ``app -> Foo`` / ``app -> main`` from the assignment frame. Without
    # a direct ``synth -> app`` edge from the plugin, the chain is
    # unreachable and ``Foo`` / ``main`` look dead.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): ...

            class Foo:
                def __init__(self, fn): self.fn = fn
                def cli(self): return self

            class Unused: ...

            if __name__ == "__main__":
                app = Foo(fn=main).cli()
            """,
        },
        [native.NativePlugin.main_block()],
    )
    reached = reachable_fqnames(graph)
    assert {"pkg.script", "pkg.script.app", "pkg.script.Foo", "pkg.script.main"} <= reached
    assert "pkg.script.Unused" not in reached


def test_main_block_keeps_nested_block_decls_alive(build_plugin_graph, reachable_fqnames):
    # Nested compound statements inside the main block still hold
    # top-level decls (the visitor doesn't push a frame for if/for/with),
    # so position-based filtering catches them too.
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            class Foo: ...
            class Unused: ...

            if __name__ == "__main__":
                if True:
                    app = Foo()
            """,
        },
        [native.NativePlugin.main_block()],
    )
    reached = reachable_fqnames(graph)
    assert {"pkg.script.app", "pkg.script.Foo"} <= reached
    assert "pkg.script.Unused" not in reached


def test_main_block_reversed_comparison(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def main(): pass
            if "__main__" == __name__:
                main()
            """,
        },
        [native.NativePlugin.main_block()],
    )
    assert "pkg.script" in reachable_fqnames(graph)
