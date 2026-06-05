"""Tests for the native FastAPI dispatch-app plugin (``NativePlugin.fastapi()``)."""

from __future__ import annotations

import pytest

from dead_cst import _native as native


def test_fastapi_plugin_marks_route_handlers(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
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


def test_fastapi_plugin_marks_websocket_and_lifecycle(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.ws_endpoint" in reached
    assert "app.main.add_header" in reached
    assert "app.main.value_error_handler" in reached
    assert "app.main.startup" in reached


def test_fastapi_plugin_keeps_handler_dependencies_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.get_item" in reached
    # Symbols transitively referenced from the handler stay alive
    assert "app.main.build_item" in reached
    assert "app.models.Item" in reached
    # Unrelated module symbol is still dead
    assert "app.models.Unused" not in reached


def test_fastapi_plugin_ignores_bare_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    # Bare ``@get`` (no attribute access) is not a FastAPI registration --
    # matching it would clobber unrelated decorators with the same name.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_ignores_unrelated_decorators(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    assert "pkg.mod.not_a_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_unused_router_stays_dead(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "app/__init__.py": "",
            "app/routes.py": """
            from fastapi import APIRouter

            router = APIRouter()

            @router.get("/")
            def orphan(): pass
            """,
        },
        [native.NativePlugin.fastapi()],
    )
    reached = reachable_fqnames(graph)
    # No FastAPI app reaches this router, so it (and its handler) are dead.
    assert "app.routes.router" not in reached
    assert "app.routes.orphan" not in reached


def test_fastapi_plugin_router_reachable_via_include_router(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.routes.router" in reached
    assert "app.routes.index" in reached
    assert "app.routes.things" in reached


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            """
            from fastapi import FastAPI as F

            app = F()

            @app.get("/")
            def index(): pass
            """,
            id="aliased-class-import",
        ),
        pytest.param(
            """
            import fastapi

            app = fastapi.FastAPI()

            @app.get("/")
            def index(): pass
            """,
            id="module-import",
        ),
        pytest.param(
            """
            from fastapi import FastAPI

            app: FastAPI = FastAPI()

            @app.get("/")
            def index(): pass
            """,
            id="annotated-assignment",
        ),
    ],
)
def test_fastapi_plugin_handles_import_variants(build_plugin_graph, reachable_fqnames, src):
    graph = build_plugin_graph(
        {"app/__init__.py": "", "app/main.py": src},
        [native.NativePlugin.fastapi()],
    )
    assert "app.main.index" in reachable_fqnames(graph)


def test_fastapi_plugin_does_nothing_without_fastapi_imports(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    # ``app`` here is not a FastAPI instance -- no ``fastapi`` import in scope.
    assert "pkg.mod.looks_like_route" not in reachable_fqnames(graph)


def test_fastapi_plugin_handles_factory_function(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    reached = reachable_fqnames(graph)
    assert "app.main.app" in reached
    assert "app.main.list_items" in reached
    assert "app.main.create_item" in reached


def test_fastapi_plugin_factory_returning_router_stays_dead(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    # Factory-produced router is treated like a literal APIRouter --
    # never auto-seeded as an entrypoint, so an unincluded one stays dead.
    reached = reachable_fqnames(graph)
    assert "app.routes.router" not in reached
    assert "app.routes.orphan" not in reached


def test_fastapi_plugin_ignores_non_app_fastapi_users(build_plugin_graph, reachable_fqnames):
    """Variables that touch ``fastapi`` for unrelated reasons stay dead.

    Walking only to the ``fastapi`` external node isn't enough -- the plugin
    must require a discriminating ``FastAPI``/``APIRouter`` import on
    the path before treating ``X`` as an instance. Otherwise any value
    derived from e.g. ``HTTPException`` would get marked as an app.
    """
    graph = build_plugin_graph(
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
        },
        [native.NativePlugin.fastapi()],
    )
    assert "pkg.mod.handler" not in reachable_fqnames(graph)


def test_fastapi_plugin_factory_returns_native_plugin():
    plugin = native.NativePlugin.fastapi()
    assert isinstance(plugin, native.NativePlugin)
    assert plugin.name == "fastapi"


def test_fastapi_plugin_factory_in_different_package(
    tmp_path, make_analysis, write_files, reachable_fqnames
):
    """Factory in dep package, consumer in dependent package.

    The classic uv-workspace layout: ``pkg_a`` owns the FastAPI factory
    and ``pkg_b`` imports + calls it. Walking from the pending marker
    in ``pkg_b`` must reach the ``FastAPI`` import inside ``pkg_a``'s
    factory body.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            from fastapi import FastAPI

            def create_app() -> FastAPI:
                return FastAPI()
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.factory import create_app

            app = create_app()

            @app.get("/items")
            def list_items(): pass
            """,
        }
    )
    graph = make_analysis(
        ["pkg_a", "pkg_b:pkg_a"], plugins=[native.NativePlugin.fastapi()]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.app" in reached
    assert "pkg_b.main.list_items" in reached


def test_fastapi_plugin_factory_module_form_in_different_package(
    tmp_path, make_analysis, write_files, reachable_fqnames
):
    """Factory in dep package uses ``import fastapi; fastapi.FastAPI()``.

    In module-attribute form the external-edge classifier drops the
    ``decl='FastAPI'`` half of the access, so every reference to
    ``fastapi`` collapses to the same ``[external dist] fastapi``
    node regardless of which class is being constructed.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/factory.py": """
            import fastapi

            def create_app():
                return fastapi.FastAPI()
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.factory import create_app

            app = create_app()

            @app.get("/items")
            def list_items(): pass
            """,
        }
    )
    graph = make_analysis(
        ["pkg_a", "pkg_b:pkg_a"], plugins=[native.NativePlugin.fastapi()]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.app" in reached
    assert "pkg_b.main.list_items" in reached


def test_fastapi_plugin_router_factory_in_different_package(
    tmp_path, make_analysis, write_files, reachable_fqnames
):
    """APIRouter factory in dep package, consumer wires it via include_router.

    A router factory shouldn't promote routers to entrypoints on its
    own -- only a FastAPI app that ``include_router``s it keeps the
    router alive.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/routes.py": """
            from fastapi import APIRouter

            def make_router() -> APIRouter:
                r = APIRouter()

                @r.get("/")
                def index(): pass

                return r
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from fastapi import FastAPI
            from pkg_a.routes import make_router

            app = FastAPI()
            router = make_router()
            app.include_router(router)
            """,
        }
    )
    graph = make_analysis(
        ["pkg_a", "pkg_b:pkg_a"], plugins=[native.NativePlugin.fastapi()]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.app" in reached
    assert "pkg_b.main.router" in reached


def test_fastapi_plugin_orphan_router_factory_stays_dead_cross_package(
    tmp_path, make_analysis, write_files, reachable_fqnames
):
    """APIRouter factory in dep package that nobody ``include_router``s.

    The factory walk only seeds factories returning the app class
    (``FastAPI``); an ``APIRouter`` factory is never entrypointed on
    its own -- so a downstream consumer that only constructs the router
    (without registering it on a FastAPI app) still gets flagged dead.
    """
    write_files(
        {
            "pkg_a/pkg_a/__init__.py": "",
            "pkg_a/pkg_a/routes.py": """
            from fastapi import APIRouter

            def make_router() -> APIRouter:
                return APIRouter()
            """,
            "pkg_b/pkg_b/__init__.py": "",
            "pkg_b/pkg_b/main.py": """
            from pkg_a.routes import make_router

            router = make_router()

            @router.get("/orphan")
            def orphan(): pass
            """,
        }
    )
    graph = make_analysis(
        ["pkg_a", "pkg_b:pkg_a"], plugins=[native.NativePlugin.fastapi()]
    ).materialize_all()
    reached = reachable_fqnames(graph)
    assert "pkg_b.main.router" not in reached
    assert "pkg_b.main.orphan" not in reached
