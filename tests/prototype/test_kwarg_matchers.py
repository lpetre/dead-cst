"""Tests for the ``.where_kwarg(...)`` chainable filter on
:class:`DecoratorQuery` and :class:`CallQuery`, plus the ``args`` /
``kwargs`` payload on :class:`DecoratorRef` and :class:`CallRef`.

Each test materializes a small project, then drives the chainable
``native.query(ctx)`` DSL directly against the built graph and asserts
on the matched refs / their arg payloads.
"""

from __future__ import annotations

import pytest

native = pytest.importorskip("dead_cst._native")


# ---------------------------------------------------------------------------
# DecoratorQuery.where_kwarg
# ---------------------------------------------------------------------------


def test_where_kwarg_matches_list_literal(build_decl_graph):
    """A function decorated ``@app.route("/x", methods=["GET"])`` is
    matched by ``.where_kwarg("methods", ["GET"])`` and is NOT matched
    by ``.where_kwarg("methods", ["POST"])``."""
    ctx = build_decl_graph(
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
    nodes = ctx.nodes()
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
    assert [nodes[r.decorated_idx].fqname for r in get_refs] == ["app.get_handler"]
    assert [nodes[r.decorated_idx].fqname for r in post_refs] == ["app.post_handler"]
    # Sanity: without the filter both decorated funcs land.
    assert sorted(nodes[r.decorated_idx].fqname for r in any_route) == [
        "app.get_handler",
        "app.post_handler",
    ]


def test_where_kwarg_missing_kwarg_never_matches(build_decl_graph):
    """``@app.route("/x")`` (no ``methods=`` kwarg) is NOT matched by
    ``.where_kwarg("methods", ["GET"])``."""
    ctx = build_decl_graph(
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
    refs = (
        native.query(ctx)
        .decorators()
        .where_owner_attr(["route"])
        .where_kwarg("methods", ["GET"])
        .collect()
    )
    assert [r.decorated.fqname for r in refs] == []


def _unwrap_arg(arg):
    """Helper: peel off the ``ArgLiteral`` / ``ArgNodeRef`` /
    ``ArgOpaque`` wrapper to compare against raw Python values. Used by
    the assertion-level tests below.
    """
    if isinstance(arg, native.ArgLiteral):
        v = arg.value
        if isinstance(v, list):
            return [_unwrap_arg(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_unwrap_arg(x) for x in v)
        return v
    if isinstance(arg, native.ArgNodeRef):
        return ("node", arg.idx)
    if isinstance(arg, native.ArgOpaque):
        return ("opaque",)
    raise TypeError(f"unexpected arg shape: {type(arg)!r}")


def test_decorator_ref_args_kwargs_populated(build_decl_graph):
    """``@app.route("/x", methods=["GET"], strict_slashes=False)``
    surfaces both ``args`` and ``kwargs`` as the discriminated-union
    shape on the matched ref."""
    ctx = build_decl_graph(
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
    refs = native.query(ctx).decorators().where_owner_attr(["route"]).with_args(True).collect()
    assert len(refs) == 1
    ref = refs[0]
    assert [_unwrap_arg(a) for a in ref.args] == ["/x"]
    assert {k: _unwrap_arg(v) for k, v in ref.kwargs.items()} == {
        "methods": ["GET"],
        "strict_slashes": False,
    }


def test_decorator_ref_args_kwargs_empty_for_bare_decorator(build_decl_graph):
    """A bare ``@app.route`` (no ``()``) surfaces empty ``args`` /
    ``kwargs`` on the matched ref."""
    ctx = build_decl_graph(
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
    refs = native.query(ctx).decorators().where_owner_attr(["route"]).collect()
    assert len(refs) == 1
    ref = refs[0]
    assert list(ref.args) == []
    assert dict(ref.kwargs) == {}


def test_kwarg_payload_surfaces_nativenode_for_imported_symbol(build_decl_graph):
    """``@register(handler=ImportedClass)`` exposes ImportedClass as an
    :class:`ArgNodeRef` in ``ref.kwargs["handler"]`` so plugins can
    anchor inverted edges off the resolved decl."""
    ctx = build_decl_graph(
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
    refs = native.query(ctx).decorators().where_owner_attr(["register"]).with_args(True).collect()
    nodes = ctx.nodes()
    assert len(refs) == 1
    ref = refs[0]
    handler = ref.kwargs.get("handler")
    assert isinstance(handler, native.ArgNodeRef)
    assert nodes[ref.decorated_idx].fqname == "handlers.on_user_created"
    # SymbolNode resolution finds the local import alias (handlers.UserCreated),
    # not the upstream class (events.UserCreated) — the alias is the codemod
    # invariant target. Either is acceptable; the test asserts the fqname is one
    # of those two so the resolution succeeded.
    assert nodes[handler.idx].fqname in {
        "handlers.UserCreated",
        "events.UserCreated",
    }


# ---------------------------------------------------------------------------
# CallQuery.where_kwarg with literal values
# ---------------------------------------------------------------------------


def test_call_query_where_kwarg_bool(build_decl_graph):
    """``mocker.patch("X", autospec=True)`` is matched by
    ``.where_kwarg("autospec", True)``; the ``autospec=False`` form is
    not."""
    ctx = build_decl_graph(
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
    assert [r.string_arg for r in true_refs] == ["pkg.a"]
    assert [r.string_arg for r in false_refs] == ["pkg.b"]


def test_call_query_where_kwarg_multiple_and_together(build_decl_graph):
    """Two ``.where_kwarg`` calls AND together — both kwargs must match."""
    ctx = build_decl_graph(
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
    assert [r.string_arg for r in refs] == ["pkg.a"]


def test_call_ref_args_kwargs_populated(build_decl_graph):
    """``mocker.patch("X", autospec=True, foo=1)`` surfaces all args
    and kwargs on the matched ref (as the discriminated-union shape)."""
    ctx = build_decl_graph(
        {
            "tests.py": (
                "def test_a(mocker):\n    mocker.patch('pkg.a', autospec=True, count=3)\n"
            ),
        }
    )
    refs = (
        native.query(ctx)
        .calls()
        .where_owner("mocker")
        .where_attr("patch")
        .string_arg_at(0)
        .with_args(True)
        .collect()
    )
    assert len(refs) == 1
    ref = refs[0]
    assert ref.string_arg == "pkg.a"
    assert [_unwrap_arg(a) for a in ref.args] == ["pkg.a"]
    assert {k: _unwrap_arg(v) for k, v in ref.kwargs.items()} == {"autospec": True, "count": 3}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_where_kwarg_with_nativenode_raises(build_decl_graph):
    """``where_kwarg`` is literal-only; passing a ``SymbolNode`` errors."""
    ctx = build_decl_graph({"tests.py": "x = 1\n"})
    mod_idx = native.query(ctx).modules().with_fqn("tests").first_idx()
    assert mod_idx is not None
    mod = ctx.nodes_at([mod_idx])[0]
    with pytest.raises(Exception, match="where_kwarg value must be"):
        (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("new_callable", mod)
        )


def test_where_kwarg_rejects_unknown_value_type(build_decl_graph):
    """``where_kwarg`` errors on a Python value that's not a literal."""
    ctx = build_decl_graph({"tests.py": "x = 1\n"})
    with pytest.raises(Exception, match="where_kwarg value must be"):
        (
            native.query(ctx)
            .calls()
            .where_owner("mocker")
            .where_attr("patch")
            .string_arg_at(0)
            .where_kwarg("foo", object())
        )
