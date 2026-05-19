"""Tests for the ``.where_kwarg(...)`` chainable filter on
:class:`DecoratorQuery` and :class:`CallQuery`, plus the ``args`` /
``kwargs`` payload on :class:`DecoratorRef` and :class:`CallRef`.

Each test materializes a small project through ``ProjectContext``, runs
one plugin that captures refs in a module-level holder, and asserts on
the captured payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

native = pytest.importorskip("dead_cst.native")


@pytest.fixture
def make_ctx(tmp_path: Path):
    """Write ``{relpath: source}`` files and return a fresh ProjectContext."""

    def make(files: dict[str, str], **kwargs) -> native.ProjectContext:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.ProjectContext(str(tmp_path), **kwargs)

    return make


class _CapturePlugin:
    """Run a caller-supplied callable on ``ctx`` and stash its result."""

    name = "capture"

    def __init__(self, fn) -> None:
        self.fn = fn
        self.result: Any = None

    def run(self, ctx: "native.ProjectContext"):
        self.result = self.fn(ctx)
        return ()


# ---------------------------------------------------------------------------
# DecoratorQuery.where_kwarg
# ---------------------------------------------------------------------------


def test_where_kwarg_matches_list_literal(make_ctx):
    """A function decorated ``@app.route("/x", methods=["GET"])`` is
    matched by ``.where_kwarg("methods", ["GET"])`` and is NOT matched
    by ``.where_kwarg("methods", ["POST"])``."""

    def capture(ctx):
        get_refs = (
            native.query(ctx)
            .decorators()
            .where_owner_attr(["route"])
            .where_kwarg("methods", ["GET"])
            .collect()
        )
        post_refs = (
            native.query(ctx)
            .decorators()
            .where_owner_attr(["route"])
            .where_kwarg("methods", ["POST"])
            .collect()
        )
        any_route = native.query(ctx).decorators().where_owner_attr(["route"]).collect()
        return {
            "get": [r.decorated.fqname for r in get_refs],
            "post": [r.decorated.fqname for r in post_refs],
            "any": [r.decorated.fqname for r in any_route],
        }

    ctx = make_ctx(
        {
            "app.py": (
                "class App:\n"
                "    def route(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "app = App()\n"
                "\n"
                "@app.route('/x', methods=['GET'])\n"
                "def get_handler(): pass\n"
                "\n"
                "@app.route('/x', methods=['POST'])\n"
                "def post_handler(): pass\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result["get"] == ["app.get_handler"]
    assert plugin.result["post"] == ["app.post_handler"]
    # Sanity: without the filter both decorated funcs land.
    assert sorted(plugin.result["any"]) == [
        "app.get_handler",
        "app.post_handler",
    ]


def test_where_kwarg_missing_kwarg_never_matches(make_ctx):
    """``@app.route("/x")`` (no ``methods=`` kwarg) is NOT matched by
    ``.where_kwarg("methods", ["GET"])``."""

    def capture(ctx):
        refs = (
            native.query(ctx)
            .decorators()
            .where_owner_attr(["route"])
            .where_kwarg("methods", ["GET"])
            .collect()
        )
        return [r.decorated.fqname for r in refs]

    ctx = make_ctx(
        {
            "app.py": (
                "class App:\n"
                "    def route(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "app = App()\n"
                "\n"
                "@app.route('/x')\n"
                "def handler(): pass\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result == []


def test_decorator_ref_args_kwargs_populated(make_ctx):
    """``@app.route("/x", methods=["GET"], strict_slashes=False)``
    surfaces both ``args`` and ``kwargs`` on the matched ref."""

    def capture(ctx):
        refs = native.query(ctx).decorators().where_owner_attr(["route"]).collect()
        assert len(refs) == 1
        ref = refs[0]
        return {
            "args": list(ref.args),
            "kwargs": dict(ref.kwargs),
        }

    ctx = make_ctx(
        {
            "app.py": (
                "class App:\n"
                "    def route(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "app = App()\n"
                "\n"
                "@app.route('/x', methods=['GET'], strict_slashes=False)\n"
                "def handler(): pass\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result["args"] == ["/x"]
    assert plugin.result["kwargs"] == {
        "methods": ["GET"],
        "strict_slashes": False,
    }


def test_decorator_ref_args_kwargs_empty_for_bare_decorator(make_ctx):
    """A bare ``@app.route`` (no ``()``) surfaces empty ``args`` /
    ``kwargs`` on the matched ref."""

    def capture(ctx):
        refs = native.query(ctx).decorators().where_owner_attr(["route"]).collect()
        assert len(refs) == 1
        ref = refs[0]
        return {"args": list(ref.args), "kwargs": dict(ref.kwargs)}

    ctx = make_ctx(
        {
            "app.py": (
                "class App:\n"
                "    def route(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "app = App()\n"
                "\n"
                "@app.route\n"
                "def handler(): pass\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result == {"args": [], "kwargs": {}}


def test_kwarg_payload_surfaces_nativenode_for_imported_symbol(make_ctx):
    """``@register(handler=ImportedClass)`` exposes ImportedClass as a
    SymbolNode in ``ref.kwargs["handler"]`` so plugins can anchor inverted
    edges off the resolved decl."""

    def capture(ctx):
        refs = native.query(ctx).decorators().where_owner_attr(["register"]).collect()
        out = []
        for r in refs:
            handler = r.kwargs.get("handler")
            out.append(
                {
                    "decorated": r.decorated.fqname,
                    "handler_fqname": getattr(handler, "fqname", None),
                }
            )
        return out

    ctx = make_ctx(
        {
            "events.py": "class UserCreated: pass\n",
            "registry.py": (
                "class Registry:\n"
                "    def register(self, *_a, **_kw):\n"
                "        def deco(f): return f\n"
                "        return deco\n"
                "registry = Registry()\n"
            ),
            "handlers.py": (
                "from events import UserCreated\n"
                "from registry import registry\n"
                "\n"
                "@registry.register(handler=UserCreated)\n"
                "def on_user_created(evt): pass\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    # SymbolNode resolution finds the local import alias (handlers.UserCreated),
    # not the upstream class (events.UserCreated) — the alias is the codemod
    # invariant target. Either is acceptable; the test asserts the fqname is one
    # of those two so the resolution succeeded.
    assert plugin.result[0]["decorated"] == "handlers.on_user_created"
    assert plugin.result[0]["handler_fqname"] in {
        "handlers.UserCreated",
        "events.UserCreated",
    }


# ---------------------------------------------------------------------------
# CallQuery.where_kwarg with literal values
# ---------------------------------------------------------------------------


def test_call_query_where_kwarg_bool(make_ctx):
    """``mocker.patch("X", autospec=True)`` is matched by
    ``.where_kwarg("autospec", True)``; the ``autospec=False`` form is
    not."""

    def capture(ctx):
        true_refs = (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("autospec", True)
            .collect()
        )
        false_refs = (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("autospec", False)
            .collect()
        )
        return {
            "true": [r.string_arg for r in true_refs],
            "false": [r.string_arg for r in false_refs],
        }

    ctx = make_ctx(
        {
            "tests.py": (
                "def test_a(mocker):\n"
                "    mocker.patch('pkg.a', autospec=True)\n"
                "def test_b(mocker):\n"
                "    mocker.patch('pkg.b', autospec=False)\n"
                "def test_c(mocker):\n"
                "    mocker.patch('pkg.c')\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result["true"] == ["pkg.a"]
    assert plugin.result["false"] == ["pkg.b"]


def test_call_query_where_kwarg_multiple_and_together(make_ctx):
    """Two ``.where_kwarg`` calls AND together — both kwargs must match."""

    def capture(ctx):
        refs = (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("autospec", True)
            .where_kwarg("create", True)
            .collect()
        )
        return [r.string_arg for r in refs]

    ctx = make_ctx(
        {
            "tests.py": (
                "def test_a(mocker):\n"
                "    mocker.patch('pkg.a', autospec=True, create=True)\n"
                "def test_b(mocker):\n"
                "    mocker.patch('pkg.b', autospec=True, create=False)\n"
                "def test_c(mocker):\n"
                "    mocker.patch('pkg.c', autospec=True)\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result == ["pkg.a"]


def test_call_ref_args_kwargs_populated(make_ctx):
    """``mocker.patch("X", autospec=True, foo=1)`` surfaces all args
    and kwargs on the matched ref."""

    def capture(ctx):
        refs = (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .collect()
        )
        assert len(refs) == 1
        ref = refs[0]
        return {
            "string_arg": ref.string_arg,
            "args": list(ref.args),
            "kwargs": dict(ref.kwargs),
        }

    ctx = make_ctx(
        {
            "tests.py": (
                "def test_a(mocker):\n    mocker.patch('pkg.a', autospec=True, count=3)\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result["string_arg"] == "pkg.a"
    assert plugin.result["args"] == ["pkg.a"]
    assert plugin.result["kwargs"] == {"autospec": True, "count": 3}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_where_kwarg_with_nativenode_raises(make_ctx):
    """``where_kwarg`` is literal-only; passing a ``SymbolNode`` errors."""

    captured: list[Exception] = []

    def capture(ctx):
        mod = ctx.find_module("tests")
        assert mod is not None
        try:
            (
                native.query(ctx)
                .calls()
                .where_owner("mocker")
                .where_attr("patch")
                .string_arg_at(0)
                .where_kwarg("new_callable", mod)
            )
        except Exception as exc:
            captured.append(exc)
        return None

    ctx = make_ctx({"tests.py": "x = 1\n"})
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert captured, "expected an error from where_kwarg(SymbolNode)"
    assert "where_kwarg value must be" in str(captured[0])


def test_where_kwarg_rejects_unknown_value_type(make_ctx):
    """``where_kwarg`` errors on a Python value that's not a literal."""

    captured: list[Exception] = []

    def capture(ctx):
        try:
            (
                native.query(ctx)
                .calls()
                .where_owner("mocker")
                .where_attr("patch")
                .string_arg_at(0)
                .where_kwarg("foo", object())
            )
        except Exception as exc:
            captured.append(exc)
        return None

    ctx = make_ctx({"tests.py": "x = 1\n"})
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert captured, "expected an error from where_kwarg(object())"
    assert "where_kwarg value must be" in str(captured[0])
