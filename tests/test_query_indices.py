"""Tests for the index-returning terminals on the chainable query DSL,
the ``ctx.reachable_indices`` / ``ctx.indices_where`` / ``ctx.nodes_at``
sibling helpers on :class:`ProjectContext`, and the
:class:`AddEdgeByIdx` graph op.

These cover the additive surface added to let plugins do index-level set
operations without paying the per-row ``Py<SymbolNode>`` allocation that
``.collect()`` does.
"""

from __future__ import annotations

import pytest

from dead_cst import _native as native


# ---------------------------------------------------------------------------
# DeclQuery.indices — parity with collect() + lookup
# ---------------------------------------------------------------------------


def test_decl_query_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def alpha(): pass\n",
            "pkg/b.py": "def beta(): pass\n",
        }
    )
    q = native.query(ctx).decls().with_kind("function")
    nodes = q.collect()
    indices = q.indices()
    assert len(nodes) == len(indices)
    all_nodes = ctx.nodes()
    for node, idx in zip(nodes, indices, strict=True):
        assert all_nodes[idx].fqname == node.fqname
        assert all_nodes[idx].path == node.path
        assert all_nodes[idx].start_line == node.start_line


def test_decl_query_indices_predicate_combos(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
            "pkg/util.py": "def helper(): pass\n",
        }
    )
    # kind + path filename narrows to just pkg.svc.handler.
    indices = native.query(ctx).decls().with_kind("function").with_filename("svc.py").indices()
    nodes = ctx.nodes_at(indices)
    fqnames = {n.fqname for n in nodes}
    assert fqnames == {"pkg.svc.handler"}


# ---------------------------------------------------------------------------
# SubclassQuery.indices
# ---------------------------------------------------------------------------


def test_subclass_query_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/bases.py": "class Base: pass\n",
            "pkg/sub.py": """
            from pkg.bases import Base
            class Mid(Base): pass
            class Leaf(Mid): pass
            """,
        }
    )
    q = native.query(ctx).subclasses().of_fqn("pkg.bases.Base")
    indices = q.indices()
    nodes = q.collect()
    assert len(indices) == len(nodes)
    # Round-trip back through nodes_at — same fqnames.
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}


# ---------------------------------------------------------------------------
# ImportQuery.indices
# ---------------------------------------------------------------------------


def test_import_query_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from os.path import join\n",
            "pkg/b.py": "from os.path import join as j2\n",
        }
    )
    q = native.query(ctx).imports().of("os.path")
    indices = q.indices()
    nodes = q.collect()
    assert len(indices) == len(nodes)
    assert len(indices) >= 2  # at least the two import nodes
    revived = ctx.nodes_at(indices)
    assert {n.kind for n in revived} == {"import"}


# ---------------------------------------------------------------------------
# ClassQuery.indices
# ---------------------------------------------------------------------------


def test_class_query_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
            class Greeter:
                def greet(self): pass
            class Counter:
                def increment(self): pass
            """,
        }
    )
    q = native.query(ctx).classes().defining_method("greet")
    indices = q.indices()
    nodes = q.collect()
    assert len(indices) == len(nodes) == 1
    revived = ctx.nodes_at(indices)
    assert revived[0].fqname == "pkg.a.Greeter"


# ---------------------------------------------------------------------------
# EdgeQuery.index_triples
# ---------------------------------------------------------------------------


def test_edge_query_index_triples_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\ndef g(): f()\n",
        }
    )
    q = native.query(ctx).edges().with_src_kind("function")
    triples = q.index_triples()
    refs = q.collect()
    assert len(triples) == len(refs)
    all_nodes = ctx.nodes()
    for (src_idx, dst_idx, flags), ref in zip(triples, refs, strict=True):
        assert all_nodes[src_idx].fqname == ref.src.fqname
        assert all_nodes[dst_idx].fqname == ref.dst.fqname
        assert flags == ref.flags


# ---------------------------------------------------------------------------
# ctx.reachable_indices parity with ctx.reachable
# ---------------------------------------------------------------------------


def test_reachable_indices_matches_reachable(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": """
            def used(): pass
            def unused(): pass
            used()
            """,
        }
    )
    # Same seed_flags / skip_flags: idx and node forms must agree.
    nodes = ctx.reachable()
    indices = ctx.reachable_indices()
    assert {ctx.nodes()[i].fqname for i in indices} == {n.fqname for n in nodes}


# ---------------------------------------------------------------------------
# ctx.indices_where predicate combos
# ---------------------------------------------------------------------------


def test_indices_where_kind_and_filename(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/util.py": "def helper(): pass\n",
        }
    )
    indices = ctx.indices_where(kind="function", filename="svc.py")
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {"pkg.svc.handler"}


def test_indices_where_fqname_prefix(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): pass\n",
            "pkg/top.py": "def shallow(): pass\n",
        }
    )
    indices = ctx.indices_where(kind="function", fqname_prefix="pkg.sub")
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {"pkg.sub.inner.deep"}


def test_indices_where_unconstrained_yields_all(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    indices = ctx.indices_where()
    assert len(indices) == len(ctx.nodes())


# ---------------------------------------------------------------------------
# ctx.nodes_at — bounds + happy path
# ---------------------------------------------------------------------------


def test_nodes_at_happy_path(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    all_nodes = ctx.nodes()
    revived = ctx.nodes_at([0, len(all_nodes) - 1])
    assert revived[0].fqname == all_nodes[0].fqname
    assert revived[-1].fqname == all_nodes[-1].fqname


def test_nodes_at_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.nodes_at([n])
    with pytest.raises(IndexError, match="out of range"):
        ctx.nodes_at([0, n + 100])


# ---------------------------------------------------------------------------
# AddEdgeByIdx — graph op
# ---------------------------------------------------------------------------


def test_add_edge_by_idx_round_trip(tmp_path):
    """``AddEdgeByIdx`` accepts positional indices into ``ctx.nodes()``
    and materializes an edge in the snapshot just like ``AddEdge``."""
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text("def f(): pass\ndef g(): pass\n")

    class WireByIdx(Plugin):
        name = "wire_by_idx"
        version = 1

        def run(self, ctx: native.ProjectContext):
            # Find f and g by fqname using indices_where + nodes_at.
            f_idx = ctx.indices_where(fqname_prefix="mod.f")
            g_idx = ctx.indices_where(fqname_prefix="mod.g")
            assert len(f_idx) == 1
            assert len(g_idx) == 1
            yield native.AddEdgeByIdx(f_idx[0], g_idx[0])

    analysis = Analysis(tmp_path, plugins=[WireByIdx()])
    ctx = analysis.materialize_all()
    fqnames = [n.fqname for n in ctx.nodes()]
    f_idx = fqnames.index("mod.f")
    g_idx = fqnames.index("mod.g")
    # Edge present in the snapshot.
    assert any(s == f_idx and d == g_idx for (s, d, _f) in ctx.edges())


def test_add_edge_by_idx_carries_flags(tmp_path):
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text("def f(): pass\ndef g(): pass\n")

    class WireWithFlags(Plugin):
        name = "wire_with_flags"
        version = 1

        def run(self, ctx: native.ProjectContext):
            f_idx = ctx.indices_where(fqname_prefix="mod.f")[0]
            g_idx = ctx.indices_where(fqname_prefix="mod.g")[0]
            yield native.AddEdgeByIdx(f_idx, g_idx, flags=native.EdgeFlags.DEAD_BRANCH)

    analysis = Analysis(tmp_path, plugins=[WireWithFlags()])
    ctx = analysis.materialize_all()
    fqnames = [n.fqname for n in ctx.nodes()]
    f_idx = fqnames.index("mod.f")
    g_idx = fqnames.index("mod.g")
    matching = [
        (s, d, f)
        for (s, d, f) in ctx.edges()
        if s == f_idx and d == g_idx and f == native.EdgeFlags.DEAD_BRANCH
    ]
    assert matching
