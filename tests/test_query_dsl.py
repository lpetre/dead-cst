"""Tests for the rust-backed chainable query DSL on ``ProjectContext``.

Covers the three correctness fixes documented in
``QUERY_FIXES_REPORT.md``:

* ``where_module`` / ``of_module`` accept either a single string or a
  list of modules (OR semantics).
* ``DecoratorQuery.where_module(M).where_name(N)`` matches relative
  imports (``from .foo import N``).
* ``ConstructionQuery.where_module(M).where_name(N)`` matches
  subscripted generic constructors (``Generic[T]()``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dead_cst import _native as native

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Bug 1: where_module / of_module accept list[str]
# ---------------------------------------------------------------------------


def test_decorator_where_module_accepts_single_string(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from flask import route
            @route
            def handler():
                pass
            """,
        }
    )
    refs = list(native.query(ctx).decorators().where_module("flask").where_name("route"))
    fqnames = {r.decorated.fqname for r in refs}
    assert "pkg.mod.handler" in fqnames


def test_decorator_where_module_accepts_single_element_list(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from flask import route
            @route
            def handler():
                pass
            """,
        }
    )
    refs = list(native.query(ctx).decorators().where_module(["flask"]).where_name("route"))
    fqnames = {r.decorated.fqname for r in refs}
    assert "pkg.mod.handler" in fqnames


def test_decorator_where_module_accepts_multi_element_list(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/flask_app.py": """
            from flask import route
            @route
            def f_handler(): pass
            """,
            "pkg/quart_app.py": """
            from quart import route
            @route
            def q_handler(): pass
            """,
            "pkg/other.py": """
            from other import route
            @route
            def o_handler(): pass
            """,
        }
    )
    refs = list(native.query(ctx).decorators().where_module(["flask", "quart"]).where_name("route"))
    fqnames = {r.decorated.fqname for r in refs}
    assert "pkg.flask_app.f_handler" in fqnames
    assert "pkg.quart_app.q_handler" in fqnames
    assert "pkg.other.o_handler" not in fqnames


def test_decorator_where_module_empty_list_matches_nothing(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            from flask import route
            @route
            def handler(): pass
            """,
        }
    )
    refs = list(native.query(ctx).decorators().where_module([]).where_name("route"))
    assert refs == []


def test_construction_where_module_accepts_list(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/flask_app.py": """
            from flask import Flask
            app = Flask(__name__)
            """,
            "pkg/quart_app.py": """
            from quart import Quart
            app = Quart(__name__)
            """,
        }
    )
    refs = list(
        native.query(ctx)
        .constructions()
        .where_module(["flask", "quart"])
        .where_name(["Flask", "Quart"])
    )
    fqnames = {r.var.fqname for r in refs}
    assert "pkg.flask_app.app" in fqnames
    assert "pkg.quart_app.app" in fqnames


def test_call_where_module_accepts_list(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
            from mod_a import use
            use("token_a")
            """,
            "pkg/b.py": """
            from mod_b import use
            use("token_b")
            """,
        }
    )
    refs = list(
        native.query(ctx)
        .calls()
        .where_module(["mod_a", "mod_b"])
        .where_name("use")
        .string_arg_at(0)
    )
    captured = {r.string_arg for r in refs}
    assert "token_a" in captured
    assert "token_b" in captured


def test_factory_of_module_accepts_list(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/factories.py": """
            from flask import Flask
            from quart import Quart

            def make_flask():
                app = Flask(__name__)
                return app

            def make_quart():
                app = Quart(__name__)
                return app
            """,
        }
    )
    refs = list(
        native.query(ctx).factories().of_module(["flask", "quart"]).where_name(["Flask", "Quart"])
    )
    by_fq = {r.decl.fqname: r.kinds for r in refs}
    assert "pkg.factories.make_flask" in by_fq
    assert "pkg.factories.make_quart" in by_fq
    assert "Flask" in by_fq["pkg.factories.make_flask"]
    assert "Quart" in by_fq["pkg.factories.make_quart"]


# ---------------------------------------------------------------------------
# Bug 2: relative-import resolution in decorator matchers
# ---------------------------------------------------------------------------


def test_decorator_matches_relative_import(build_decl_graph):
    """``from .foo import route`` is the relative-import form for
    ``from pkg.foo import route``. ``where_module("pkg.foo")`` should
    catch a decorator that resolves through it."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo.py": "def route(fn): return fn\n",
            "pkg/handlers.py": """
            from .foo import route

            @route
            def handler():
                pass
            """,
        }
    )
    refs = list(native.query(ctx).decorators().where_module("pkg.foo").where_name("route"))
    fqnames = {r.decorated.fqname for r in refs}
    assert "pkg.handlers.handler" in fqnames


def test_construction_matches_relative_import(build_decl_graph):
    """Same relative-import path, for the construction matcher."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo.py": "class Worker: ...\n",
            "pkg/handlers.py": """
            from .foo import Worker

            w = Worker()
            """,
        }
    )
    refs = list(native.query(ctx).constructions().where_module("pkg.foo").where_name("Worker"))
    fqnames = {r.var.fqname for r in refs}
    assert "pkg.handlers.w" in fqnames


# ---------------------------------------------------------------------------
# Bug 3: subscripted generic constructors in where_name
# ---------------------------------------------------------------------------


def test_construction_matches_subscripted_generic(build_decl_graph):
    """``Worker[T](...)`` should match ``where_name("Worker")`` just
    like ``Worker(...)`` does."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "class Worker: ...\n",
            "pkg/uses.py": """
            from pkg.lib import Worker

            w = Worker[int]()
            """,
        }
    )
    refs = list(native.query(ctx).constructions().where_module("pkg.lib").where_name("Worker"))
    fqnames = {r.var.fqname for r in refs}
    assert "pkg.uses.w" in fqnames


def test_construction_matches_subscripted_generic_via_module_attr(build_decl_graph):
    """``mod.Worker[T](...)`` should also match."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "class Worker: ...\n",
            "pkg/uses.py": """
            from pkg import lib

            w = lib.Worker[int]()
            """,
        }
    )
    refs = list(native.query(ctx).constructions().where_module("pkg.lib").where_name("Worker"))
    fqnames = {r.var.fqname for r in refs}
    assert "pkg.uses.w" in fqnames


# ---------------------------------------------------------------------------
# DeclQuery.where_fqname — str | list[str] | re.Pattern | list[re.Pattern]
# ---------------------------------------------------------------------------


def test_where_fqname_str_literal(build_decl_graph):
    """Single str matches by exact fqname equality."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def foo(): ...\ndef bar(): ...\n",
            "pkg/b.py": "def foo(): ...\n",
        }
    )
    refs = list(native.query(ctx).decls().where_fqname("pkg.a.foo"))
    assert {r.fqname for r in refs} == {"pkg.a.foo"}


def test_where_fqname_list_of_str(build_decl_graph):
    """List of strs matches if fqname equals any element."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def foo(): ...\ndef bar(): ...\n",
            "pkg/b.py": "def baz(): ...\n",
        }
    )
    refs = list(native.query(ctx).decls().where_fqname(["pkg.a.foo", "pkg.b.baz"]))
    assert {r.fqname for r in refs} == {"pkg.a.foo", "pkg.b.baz"}


def test_where_fqname_regex(build_decl_graph):
    """A single ``re.Pattern`` matches by regex search."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def foo(): ...\ndef foobar(): ...\ndef bar(): ...\n",
        }
    )
    refs = list(native.query(ctx).decls().where_fqname(re.compile(r"^pkg\.a\.foo")))
    assert {r.fqname for r in refs} == {"pkg.a.foo", "pkg.a.foobar"}


def test_where_fqname_list_of_regex(build_decl_graph):
    """List of ``re.Pattern`` matches if fqname matches any pattern."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def foo(): ...\ndef bar(): ...\ndef baz(): ...\n",
        }
    )
    refs = list(
        native.query(ctx).decls().where_fqname([re.compile(r"\.foo$"), re.compile(r"\.bar$")])
    )
    assert {r.fqname for r in refs} == {"pkg.a.foo", "pkg.a.bar"}


def test_where_fqname_mixed_str_and_regex(build_decl_graph):
    """A mixed sequence applies literal-equality OR regex-search."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def foo(): ...\ndef bar(): ...\ndef baz(): ...\n",
        }
    )
    refs = list(native.query(ctx).decls().where_fqname(["pkg.a.foo", re.compile(r"\.baz$")]))
    assert {r.fqname for r in refs} == {"pkg.a.foo", "pkg.a.baz"}


def test_where_fqname_empty_list_matches_nothing(build_decl_graph):
    """``where_fqname([])`` is the matches-nothing sentinel."""
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def foo(): ...\n"})
    refs = list(native.query(ctx).decls().where_fqname([]))
    assert refs == []


def test_where_fqname_invalid_type_raises(build_decl_graph):
    """Non-str / non-Pattern raises ``TypeError`` at the call site."""
    import pytest

    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def foo(): ...\n"})
    with pytest.raises(TypeError):
        native.query(ctx).decls().where_fqname(123)
    with pytest.raises(TypeError):
        native.query(ctx).decls().where_fqname(["ok", 42])
