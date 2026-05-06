"""Tests for :class:`FastAPIPlugin`."""

from __future__ import annotations

from dead_cst import Analysis
from dead_cst.plugins import FastAPIPlugin
from conftest import manual


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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
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
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
    # ``app`` here is not a FastAPI instance -- no ``fastapi`` import in scope.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_handles_factory_function(tmp_path, write_files, reachable_fqnames):
    write_files(
        {
            "app/__init__.py": "",
            "app/factory.py": """
            from fastapi import FastAPI

            def create_app() -> FastAPI:
                return FastAPI()
            """,
            "app/main.py": """
            from app.factory import create_app

            app = create_app()

            @app.get("/items")
            def list_items(): pass

            @app.post("/items")
            def create_item(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.main.list_items" in reached
    assert "app.main.create_item" in reached


def test_fastapi_plugin_factory_returning_router_stays_dead(
    tmp_path, write_files, reachable_fqnames
):
    write_files(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from fastapi import APIRouter

            def make_router() -> APIRouter:
                return APIRouter()

            router = make_router()

            @router.get("/orphan")
            def orphan(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
    # Factory-produced router is treated like a literal APIRouter --
    # never auto-seeded as an entrypoint, so an unincluded one stays dead.
    reached = reachable_fqnames(graph)
    assert "app.routes.router" not in reached
    assert "app.routes.orphan" not in reached


def test_fastapi_plugin_ignores_non_app_fastapi_users(tmp_path, write_files, reachable_fqnames):
    """Variables that touch ``fastapi`` for unrelated reasons stay dead.

    Walking only to the ``fastapi`` synthetic isn't enough -- the plugin
    must require a discriminating ``FastAPI``/``APIRouter`` import on
    the path before treating ``X`` as an instance. Otherwise any value
    derived from e.g. ``HTTPException`` would get marked as an app.
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from fastapi import HTTPException

            err = HTTPException(404)

            class Decorated:
                def get(self, path):
                    def wrap(fn): return fn
                    return wrap

            thing = Decorated()

            @thing.get("/")
            def handler(): pass
            """,
        }
    )
    graph = Analysis(
        tmp_path,
        resolvers=manual(),
        plugins=[FastAPIPlugin()],
    ).materialize_all()
    assert "pkg.mod.handler" not in reachable_fqnames(graph)


def test_fastapi_plugin_loads_via_load_plugin():
    from dead_cst.plugins import load_plugin

    plugin = load_plugin("fastapi")
    assert isinstance(plugin, FastAPIPlugin)
