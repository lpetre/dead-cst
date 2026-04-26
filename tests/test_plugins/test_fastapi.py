"""Tests for :class:`FastAPIPlugin`."""

from __future__ import annotations

from dead_cst import FastAPIPlugin, build_symbol_graph


def test_fastapi_plugin_marks_route_handlers(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from fastapi import FastAPI

            app = FastAPI()

            @app.get("/items")
            def list_items(): pass

            @app.post("/items")
            def create_item(): pass

            @app.put("/items/{id}")
            def update_item(): pass

            @app.delete("/items/{id}")
            def delete_item(): pass

            @app.patch("/items/{id}")
            def patch_item(): pass

            def helper(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.main.list_items" in reached
    assert "app.main.create_item" in reached
    assert "app.main.update_item" in reached
    assert "app.main.delete_item" in reached
    assert "app.main.patch_item" in reached
    # Undecorated helper not referenced by any handler stays dead
    assert "app.main.helper" not in reached


def test_fastapi_plugin_marks_websocket_and_lifecycle(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from fastapi import FastAPI, WebSocket

            app = FastAPI()

            @app.websocket("/ws")
            async def ws_endpoint(websocket: WebSocket): pass

            @app.middleware("http")
            async def add_header(request, call_next): pass

            @app.exception_handler(ValueError)
            async def value_error_handler(request, exc): pass

            @app.on_event("startup")
            async def startup(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "app.main.ws_endpoint" in reached
    assert "app.main.add_header" in reached
    assert "app.main.value_error_handler" in reached
    assert "app.main.startup" in reached


def test_fastapi_plugin_keeps_handler_dependencies_alive(tmp_path, write_files, reachable_fqnames):
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
            from fastapi import FastAPI
            from app.models import Item

            app = FastAPI()

            def build_item() -> Item:
                return Item()

            @app.get("/item")
            def get_item():
                return build_item()
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "app.main.get_item" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "app.main.build_item" in reached
    assert "app.models.Item" in reached
    # Unrelated module symbol is still dead
    assert "app.models.Unused" not in reached


def test_fastapi_plugin_ignores_bare_decorators(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from fastapi import FastAPI

            app = FastAPI()

            def get(fn):
                return fn

            @get
            def looks_like_route(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    # Bare ``@get`` (no attribute access) is not a FastAPI registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_ignores_unrelated_decorators(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Thing:
                def register(self, fn):
                    return fn

            t = Thing()

            @t.register
            def not_a_route(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    assert "pkg.mod.not_a_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_unused_router_stays_dead(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from fastapi import APIRouter

            router = APIRouter()

            @router.get("/")
            def orphan(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    # No FastAPI app reaches this router, so it (and its handler) are dead.
    assert "app.routes.router" not in reached
    assert "app.routes.orphan" not in reached


def test_fastapi_plugin_router_reachable_via_include_router(
    tmp_path, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from fastapi import APIRouter

            router = APIRouter()

            @router.get("/")
            def index(): pass

            @router.api_route("/things", methods=["GET", "POST"])
            def things(): pass
            """,
            "app/main.py": """
            from fastapi import FastAPI
            from app.routes import router

            app = FastAPI()
            app.include_router(router)
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.routes.router" in reached
    assert "app.routes.index" in reached
    assert "app.routes.things" in reached


def test_fastapi_plugin_handles_aliased_class_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from fastapi import FastAPI as F

            app = F()

            @app.get("/")
            def index(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    assert "app.main.index" in reachable_fqnames(graph)


def test_fastapi_plugin_handles_module_import(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            import fastapi

            app = fastapi.FastAPI()

            @app.get("/")
            def index(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    assert "app.main.index" in reachable_fqnames(graph)


def test_fastapi_plugin_handles_annotated_assignment(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/main.py": """
            from fastapi import FastAPI

            app: FastAPI = FastAPI()

            @app.get("/")
            def index(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    assert "app.main.index" in reachable_fqnames(graph)


def test_fastapi_plugin_does_nothing_without_fastapi_imports(
    tmp_path, write_files, reachable_fqnames
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class App:
                def get(self, path):
                    def wrap(fn): return fn
                    return wrap

            app = App()

            @app.get("/")
            def looks_like_route(): pass
            """,
        }
    )
    graph = build_symbol_graph(
        {tmp_path: []},
        plugins=[FastAPIPlugin()],
        project_root=tmp_path,
    )
    # ``app`` here is not a FastAPI instance -- no ``fastapi`` import in scope.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_loads_via_load_plugin():
    from dead_cst import load_plugin

    plugin = load_plugin("fastapi")
    assert isinstance(plugin, FastAPIPlugin)
