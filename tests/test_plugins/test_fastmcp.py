"""Tests for :class:`FastMCPPlugin`."""

from __future__ import annotations

from dead_cst.plugins import FastMCPPlugin


def test_fastmcp_plugin_marks_tool_handlers(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/main.py": """
            import fastmcp

            mcp = fastmcp.FastMCP("demo")

            @mcp.tool()
            def add(a: int, b: int) -> int:
                return a + b

            @mcp.tool
            def bare_tool(): pass

            @mcp.resource("res://config")
            def config() -> dict:
                return {}

            @mcp.prompt()
            def greet(name: str) -> str:
                return f"hi {name}"

            @mcp.completion()
            def complete(): pass

            def helper(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "server.main.mcp" in reached
    assert "server.main.add" in reached
    assert "server.main.bare_tool" in reached
    assert "server.main.config" in reached
    assert "server.main.greet" in reached
    assert "server.main.complete" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "server.main.helper" not in reached


def test_fastmcp_plugin_keeps_handler_dependencies_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/models.py": """
            class Result:
                pass

            class Unused:
                pass
            """,
            "server/main.py": """
            from fastmcp import FastMCP
            from server.models import Result

            mcp = FastMCP("demo")

            def build_result() -> Result:
                return Result()

            @mcp.tool()
            def run():
                return build_result()
            """,
        },
        [FastMCPPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "server.main.run" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "server.main.build_result" in reached
    assert "server.models.Result" in reached
    # Unrelated module symbol is still dead
    assert "server.models.Unused" not in reached


def test_fastmcp_plugin_auto_seeds_server_as_entrypoint(build_plugin_graph, reachable_fqnames):
    """Direct ``X = FastMCP(...)`` is an entrypoint even without a
    ``__main__`` block or ``[project.scripts]`` entry. The ``fastmcp``
    CLI loads ``module:mcp`` by import path, mirroring how ``uvicorn``
    loads a FastAPI ``module:app``, so every FastMCP instance is
    framework-visible the moment it's constructed."""
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/main.py": """
            from fastmcp import FastMCP

            mcp = FastMCP("demo")

            @mcp.tool()
            def hello(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "server.main.mcp" in reached
    assert "server.main.hello" in reached


def test_fastmcp_plugin_handles_aliased_class_import(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/main.py": """
            from fastmcp import FastMCP as Server

            mcp = Server("demo")

            @mcp.tool()
            def hello(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    assert "server.main.hello" in reachable_fqnames(graph)


def test_fastmcp_plugin_handles_module_import(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/main.py": """
            import fastmcp

            mcp = fastmcp.FastMCP("demo")

            @mcp.tool()
            def hello(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    assert "server.main.hello" in reachable_fqnames(graph)


def test_fastmcp_plugin_handles_annotated_assignment(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/main.py": """
            from fastmcp import FastMCP

            mcp: FastMCP = FastMCP("demo")

            @mcp.tool()
            def hello(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    assert "server.main.hello" in reachable_fqnames(graph)


def test_fastmcp_plugin_ignores_bare_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from fastmcp import FastMCP

            mcp = FastMCP("demo")

            def tool(fn):
                return fn

            @tool
            def looks_like_tool(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    # Bare ``@tool`` (no attribute access) is not a FastMCP registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_tool" not in reachable_fqnames(graph)


def test_fastmcp_plugin_ignores_unrelated_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Thing:
                def tool(self, fn):
                    return fn

            t = Thing()

            @t.tool
            def not_a_tool(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    # ``t`` isn't a ``FastMCP`` instance, so its ``.tool`` decorator is ignored.
    assert "pkg.mod.not_a_tool" not in reachable_fqnames(graph)


def test_fastmcp_plugin_does_nothing_without_fastmcp_imports(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def tool(self):
                    def wrap(fn): return fn
                    return wrap

            mcp = App()

            @mcp.tool()
            def looks_like_tool(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    # ``mcp`` here is not a FastMCP instance -- no ``fastmcp`` import in scope.
    assert "pkg.mod.looks_like_tool" not in reachable_fqnames(graph)


def test_fastmcp_plugin_ignores_import_star(build_plugin_graph, reachable_fqnames):
    """``from fastmcp import *`` doesn't bind ``FastMCP`` for the plugin's
    purposes. The ``import *`` analyzer logic is pessimistic enough on its
    own; the plugin shouldn't infer FastMCP wiring from the star import."""
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/main.py": """
            from fastmcp import *

            mcp = FastMCP("demo")

            @mcp.tool()
            def hello(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    # No instance edge from ``mcp`` to ``hello`` because the plugin ignores
    # star imports. ``hello`` is not referenced by anything reachable.
    assert "server.main.hello" not in reachable_fqnames(graph)


def test_fastmcp_plugin_handles_factory_function(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "server/__init__.py": "",
            "server/factory.py": """
            from fastmcp import FastMCP

            def create_server() -> FastMCP:
                return FastMCP("demo")
            """,
            "server/main.py": """
            from server.factory import create_server

            mcp = create_server()

            @mcp.tool()
            def add(a: int, b: int) -> int:
                return a + b

            @mcp.resource("res://config")
            def config() -> dict:
                return {}
            """,
        },
        [FastMCPPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "server.main.mcp" in reached
    assert "server.main.add" in reached
    assert "server.main.config" in reached


def test_fastmcp_plugin_ignores_non_server_fastmcp_users(build_plugin_graph, reachable_fqnames):
    """Variables that touch ``fastmcp`` for unrelated reasons stay dead.

    Walking only to the ``fastmcp`` synthetic isn't enough -- the plugin
    must require a discriminating ``FastMCP`` import on the path before
    treating ``X`` as an instance. Otherwise any value derived from some
    other ``fastmcp`` export would get marked as a server.
    """
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            import fastmcp

            settings = fastmcp.__version__

            class Decorated:
                def tool(self):
                    def wrap(fn): return fn
                    return wrap

            thing = Decorated()

            @thing.tool()
            def handler(): pass
            """,
        },
        [FastMCPPlugin()],
    )
    assert "pkg.mod.handler" not in reachable_fqnames(graph)


def test_fastmcp_plugin_factory_in_different_package(
    tmp_path, make_analysis, write_files, reachable_fqnames
):
    """Factory in dep package, consumer in dependent package.

    The classic uv-workspace layout: ``pkg_a`` owns the FastMCP factory
    and ``pkg_b`` imports + calls it. Walking from the pending marker
    in ``pkg_b`` must reach the ``FastMCP`` import inside ``pkg_a``'s
    factory body.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            from fastmcp import FastMCP

            def create_server() -> FastMCP:
                return FastMCP("demo")
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.factory import create_server

            mcp = create_server()

            @mcp.tool()
            def hello(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[FastMCPPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.mcp" in reached
    assert "pkg_b.main.hello" in reached


def test_fastmcp_plugin_factory_module_form_in_different_package(
    tmp_path, make_analysis, write_files, reachable_fqnames
):
    """Factory in dep package uses ``import fastmcp; fastmcp.FastMCP()``.

    Without the factory-marker synthetic the cross-package walk has no
    discriminator: the external-edge classifier drops the
    ``decl='FastMCP'`` half of the access, so every reference to
    ``fastmcp`` would collapse to the same ``[external dist] fastmcp``
    node.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            import fastmcp

            def create_server():
                return fastmcp.FastMCP("demo")
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.factory import create_server

            mcp = create_server()

            @mcp.tool()
            def hello(): pass
            """,
        }
    )
    graph = make_analysis(["pkg_a", "pkg_b:pkg_a"], plugins=[FastMCPPlugin()]).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.mcp" in reached
    assert "pkg_b.main.hello" in reached


def test_fastmcp_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("fastmcp")
    assert isinstance(plugin, FastMCPPlugin)
    assert plugin.name == "fastmcp"
