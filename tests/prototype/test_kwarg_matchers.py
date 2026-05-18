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

native = pytest.importorskip("dead_cst._native")


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
# Where_kwarg with NativeNode value (DeclRef matching)
# ---------------------------------------------------------------------------


def test_call_query_where_kwarg_native_node(make_ctx):
    """A ``mocker.patch("X", new_callable=Replacement)`` call where
    ``Replacement`` is a project-local imported decl is matched by
    ``.where_kwarg("new_callable", replacement_node)``."""

    def capture(ctx):
        # Resolve a project-local decl. The decl lives in ``mocks.py``,
        # so its fqname is ``mocks.Replacement``.
        repls = native.query(ctx).declarations("mocks.Replacement")
        assert repls, "expected Replacement to be resolvable via declarations()"
        repl_node = repls[0]
        refs = (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("new_callable", repl_node)
            .collect()
        )
        # The matched kwarg should also surface as the same NativeNode
        # in ``ref.kwargs["new_callable"]``.
        kwarg_fqns = [
            r.kwargs["new_callable"].fqname
            for r in refs
            if r.kwargs.get("new_callable") is not None
            and hasattr(r.kwargs["new_callable"], "fqname")
        ]
        return {
            "matched": [r.string_arg for r in refs],
            "kwarg_fqns": kwarg_fqns,
        }

    ctx = make_ctx(
        {
            "mocks.py": "class Replacement: pass\n",
            "tests.py": (
                "from mocks import Replacement\n"
                "\n"
                "def test_a(mocker):\n"
                "    mocker.patch('pkg.a', new_callable=Replacement)\n"
                "def test_b(mocker):\n"
                "    mocker.patch('pkg.b')\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result["matched"] == ["pkg.a"]
    assert plugin.result["kwarg_fqns"] == ["mocks.Replacement"]


def test_call_query_where_kwarg_native_node_does_not_match_literal(make_ctx):
    """A literal kwarg value never matches a ``NativeNode`` filter."""

    def capture(ctx):
        repls = native.query(ctx).declarations("mocks.Replacement")
        assert repls
        repl_node = repls[0]
        refs = (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("new_callable", repl_node)
            .collect()
        )
        return [r.string_arg for r in refs]

    ctx = make_ctx(
        {
            "mocks.py": "class Replacement: pass\n",
            "tests.py": (
                "from mocks import Replacement\n"
                "\n"
                "def test_a(mocker):\n"
                # ``new_callable=None`` — literal, not the imported decl.
                "    mocker.patch('pkg.a', new_callable=None)\n"
            ),
        }
    )
    plugin = _CapturePlugin(capture)
    ctx.add_plugin(plugin)
    ctx.materialize()
    assert plugin.result == []


def test_where_kwarg_rejects_unknown_value_type(make_ctx):
    """``where_kwarg`` errors on a Python value that's neither a
    literal nor a ``NativeNode``."""

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
