"""Tests for the fqname-tree-backed ``ProjectContext.module_surface``
and ``find_module_top_level_decls`` queries, and the matching
``DeclQuery.with_fqname_under`` DSL predicate.

All three are backed by ``BuildOutputs.children_by_parent``, a parent
fqname -> [child node idx] index built in ``build_fqname_indices``.
The behavior contracts these tests pin:

* ``module_surface(M)``: module ``M`` plus every transitive descendant
  in the fqname tree. Decls under ``M`` are included but their
  sub-fqnames (synthetic method-of-class names, etc.) are NOT recursed
  into — the BFS only steps through module children.
* ``find_module_top_level_decls(M)``: just the immediate top-level
  decls (non-module children of ``M``). Submodules are excluded.
* ``DeclQuery.with_fqname_under(P)``: ``P`` itself plus every
  segment-bounded descendant. Distinct from
  ``with_fqname_prefix(P)`` which is raw ``starts_with``.
"""

from __future__ import annotations

from dead_cst import _native as native


# ---------------------------------------------------------------------------
# module_surface
# ---------------------------------------------------------------------------


def test_module_surface_returns_module_and_all_decls(build_decl_graph):
    """The module node itself + every top-level decl is included."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            def foo(): ...
            def bar(): ...
            X = 1
            """,
        }
    )
    surface = ctx.module_surface("pkg.mod")
    fqnames = {n.fqname for n in surface}
    assert fqnames == {"pkg.mod", "pkg.mod.foo", "pkg.mod.bar", "pkg.mod.X"}


def test_module_surface_includes_transitive_submodules(build_decl_graph):
    """``importlib.import_module(pkg)`` would pull every submodule and
    submodule decl — same here."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): ...\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/b.py": "def g(): ...\n",
        }
    )
    surface = ctx.module_surface("pkg")
    fqnames = {n.fqname for n in surface}
    assert {"pkg", "pkg.a", "pkg.a.f", "pkg.sub", "pkg.sub.b", "pkg.sub.b.g"} <= fqnames


def test_module_surface_segment_bounded(build_decl_graph):
    """``module_surface(pkg.foo)`` must NOT pick up ``pkg.foobar`` —
    the old linear-prefix-scan would have. The fqname-tree BFS is
    segment-bounded by construction."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo/__init__.py": "X = 1\n",
            "pkg/foobar.py": "Y = 2\n",
        }
    )
    surface = ctx.module_surface("pkg.foo")
    fqnames = {n.fqname for n in surface}
    assert "pkg.foo" in fqnames
    assert "pkg.foo.X" in fqnames
    assert "pkg.foobar" not in fqnames
    assert "pkg.foobar.Y" not in fqnames


def test_module_surface_unknown_module_returns_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): ...\n"})
    assert ctx.module_surface("does.not.exist") == []


def test_module_surface_does_not_recurse_into_decls(build_decl_graph):
    """A class decl ``pkg.mod.Klass`` has fqname-tree children only if
    it shadows a submodule path — which it can't here. The BFS shouldn't
    chase decl children expecting deeper names: nested methods aren't
    graph nodes."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
            class Klass:
                def method(self): ...
                def other(self): ...
            """,
        }
    )
    surface = ctx.module_surface("pkg.mod")
    fqnames = {n.fqname for n in surface}
    # The Klass node is in; its method nodes don't exist (nested defs
    # are deliberately not graph nodes per the architecture invariant).
    assert fqnames == {"pkg.mod", "pkg.mod.Klass"}


# ---------------------------------------------------------------------------
# find_module_top_level_decls
# ---------------------------------------------------------------------------


def test_find_top_level_decls_excludes_submodules(build_decl_graph):
    """``from pkg import *`` doesn't pull in submodules — top-level
    decls only."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "X = 1\n",
            "pkg/sub.py": "def f(): ...\n",
        }
    )
    decls = ctx.find_module_top_level_decls("pkg")
    fqnames = {n.fqname for n in decls}
    # X is top-level; sub (a submodule) is NOT included.
    assert "pkg.X" in fqnames
    assert "pkg.sub" not in fqnames


def test_find_top_level_decls_does_not_descend(build_decl_graph):
    """Transitive decls (``pkg.sub.x`` when querying ``pkg``) are
    excluded — only immediate children."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "X = 1\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): ...\n",
        }
    )
    decls = ctx.find_module_top_level_decls("pkg")
    fqnames = {n.fqname for n in decls}
    assert "pkg.sub.inner" not in fqnames
    assert "pkg.sub.inner.deep" not in fqnames


def test_find_top_level_decls_segment_bounded(build_decl_graph):
    """Same segment-bounded guarantee as ``module_surface``: the
    sibling ``pkg.foobar`` is not pulled in by querying ``pkg.foo``."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo/__init__.py": "A = 1\n",
            "pkg/foobar.py": "B = 2\n",
        }
    )
    decls = ctx.find_module_top_level_decls("pkg.foo")
    fqnames = {n.fqname for n in decls}
    assert "pkg.foo.A" in fqnames
    assert "pkg.foobar" not in fqnames
    assert "pkg.foobar.B" not in fqnames


def test_find_top_level_decls_unknown_module_returns_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): ...\n"})
    assert ctx.find_module_top_level_decls("does.not.exist") == []


# ---------------------------------------------------------------------------
# DeclQuery.with_fqname_under (new) vs with_fqname_prefix (raw)
# ---------------------------------------------------------------------------


def test_with_fqname_under_segment_bounded(build_decl_graph):
    """``with_fqname_under("pkg.foo")`` matches ``pkg.foo`` and its
    descendants — but NOT the sibling ``pkg.foobar``."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo/__init__.py": "A = 1\n",
            "pkg/foo/inner.py": "B = 2\n",
            "pkg/foobar.py": "C = 3\n",
        }
    )
    fqnames = {r.fqname for r in native.query(ctx).decls().with_fqname_under("pkg.foo")}
    assert "pkg.foo.A" in fqnames
    assert "pkg.foo.inner.B" in fqnames
    assert "pkg.foobar" not in fqnames
    assert "pkg.foobar.C" not in fqnames


def test_with_fqname_under_includes_root_module(build_decl_graph):
    """The module ``pkg.foo`` itself is included (when its kind isn't
    filtered out), not just its descendants."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo/__init__.py": "A = 1\n",
        }
    )
    nodes = list(
        native.query(ctx).decls().with_fqname_under("pkg.foo").with_kinds(["module", "variable"])
    )
    fqnames = {n.fqname for n in nodes}
    assert "pkg.foo" in fqnames
    assert "pkg.foo.A" in fqnames


def test_with_fqname_under_unknown_parent_yields_empty(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): ...\n",
        }
    )
    nodes = list(native.query(ctx).decls().with_fqname_under("does.not.exist"))
    assert nodes == []


def test_with_fqname_prefix_remains_raw_starts_with(build_decl_graph):
    """``with_fqname_prefix`` is documented as raw ``starts_with``;
    this test pins that behavior so the segment-bounded
    ``with_fqname_under`` doesn't accidentally take its place."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/foo/__init__.py": "A = 1\n",
            "pkg/foobar.py": "C = 3\n",
        }
    )
    fqnames = {
        r.fqname
        for r in native.query(ctx).decls().with_kind("module").with_fqname_prefix("pkg.foo")
    }
    # Raw starts_with picks up both ``pkg.foo`` AND ``pkg.foobar``.
    assert "pkg.foo" in fqnames
    assert "pkg.foobar" in fqnames


def test_with_fqname_under_composes_with_other_predicates(build_decl_graph):
    """Predicates AND together. ``with_fqname_under("pkg") &
    with_kind("function")`` yields only function decls under pkg."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
            def foo(): ...
            X = 1
            class Klass: ...
            """,
        }
    )
    fqnames = {
        r.fqname for r in native.query(ctx).decls().with_fqname_under("pkg").with_kind("function")
    }
    assert fqnames == {"pkg.a.foo"}
