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


# ---------------------------------------------------------------------------
# AddNodeByIdx — graph op
# ---------------------------------------------------------------------------


def test_add_node_by_idx_round_trip(tmp_path):
    """``AddNodeByIdx`` mints a synthetic node and wires its edges from
    positional indices, the same way ``AddNode`` does with
    ``SymbolNode`` endpoints."""
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text("def f(): pass\ndef g(): pass\n")

    class MintByIdx(Plugin):
        name = "mint_by_idx"
        version = 1

        def run(self, ctx: native.ProjectContext):
            f_idx = ctx.indices_where(fqname_prefix="mod.f")[0]
            g_idx = ctx.indices_where(fqname_prefix="mod.g")[0]
            yield native.AddNodeByIdx(
                "<marker>:mint",
                path="mod.py",
                edges_from_idx=[f_idx],
                edges_to_idx=[g_idx],
            )

    analysis = Analysis(tmp_path, plugins=[MintByIdx()])
    ctx = analysis.materialize_all()
    fqnames = [n.fqname for n in ctx.nodes()]
    assert "<marker>:mint" in fqnames

    marker_idx = fqnames.index("<marker>:mint")
    f_idx = fqnames.index("mod.f")
    g_idx = fqnames.index("mod.g")
    edges = [(s, d) for (s, d, _f) in ctx.edges()]
    assert (f_idx, marker_idx) in edges
    assert (marker_idx, g_idx) in edges


def test_add_node_by_idx_entrypoint_flag(tmp_path):
    """``AddNodeByIdx`` honors ``NodeFlags.ENTRYPOINT`` so a multi-target
    marker (the idiomatic non-sugar reason to prefer it over
    ``AddEntrypoint``) makes its targets reachable."""
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text(
        "def alive(): pass\ndef also_alive(): pass\ndef dead(): pass\n"
    )

    class SeedBoth(Plugin):
        name = "seed_both"
        version = 1

        def run(self, ctx: native.ProjectContext):
            a = ctx.indices_where(fqname_prefix="mod.alive")[0]
            b = ctx.indices_where(fqname_prefix="mod.also_alive")[0]
            yield native.AddNodeByIdx(
                "<seed>:multi",
                path="mod.py",
                flags=native.NodeFlags.ENTRYPOINT,
                edges_to_idx=[a, b],
            )

    analysis = Analysis(tmp_path, plugins=[SeedBoth()])
    analysis.materialize_all()
    dead_fqnames = {n.fqname for n in analysis.dead()}
    assert "mod.alive" not in dead_fqnames
    assert "mod.also_alive" not in dead_fqnames
    assert "mod.dead" in dead_fqnames


def test_add_node_by_idx_out_of_range_raises(tmp_path):
    """An out-of-range ``edges_*_idx`` raises ``IndexError`` at apply
    time and does *not* leave an orphan synthetic node behind."""
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text("def f(): pass\n")

    class BadIdx(Plugin):
        name = "bad_idx"
        version = 1

        def run(self, ctx: native.ProjectContext):
            yield native.AddNodeByIdx(
                "<should-not-land>",
                path="mod.py",
                edges_to_idx=[10**9],
            )

    analysis = Analysis(tmp_path, plugins=[BadIdx()])
    with pytest.raises(IndexError, match="edges_to_idx"):
        analysis.materialize_all()


def test_add_node_by_idx_default_no_edges(tmp_path):
    """A bare ``AddNodeByIdx(fqname, path=...)`` with no edges mints a
    standalone synthetic node — same defaults as ``AddNode``."""
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text("x = 1\n")

    class JustMint(Plugin):
        name = "just_mint"
        version = 1

        def run(self, ctx: native.ProjectContext):
            yield native.AddNodeByIdx("<bare>:marker", path="mod.py")

    analysis = Analysis(tmp_path, plugins=[JustMint()])
    ctx = analysis.materialize_all()
    fqnames = [n.fqname for n in ctx.nodes()]
    assert "<bare>:marker" in fqnames


# ---------------------------------------------------------------------------
# ctx.find_declarations_indices
# ---------------------------------------------------------------------------


def test_find_declarations_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
        }
    )
    nodes = ctx.find_declarations("pkg.svc.handler")
    indices = ctx.find_declarations_indices("pkg.svc.handler")
    assert len(nodes) == len(indices) == 1
    all_nodes = ctx.nodes()
    assert all_nodes[indices[0]].fqname == nodes[0].fqname


def test_find_declarations_indices_walks_back(build_decl_graph):
    """``pkg.svc.Cls.method`` resolves to ``pkg.svc.Cls`` — methods don't
    get their own graph node."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "class Cls:\n    def method(self): pass\n",
        }
    )
    indices = ctx.find_declarations_indices("pkg.svc.Cls.method")
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {"pkg.svc.Cls"}


def test_find_declarations_indices_unknown_returns_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.find_declarations_indices("nothing.here") == []


# ---------------------------------------------------------------------------
# ctx.module_for_indices + modules_for_paths
# ---------------------------------------------------------------------------


def test_module_for_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def f(): pass\n",
        }
    )
    # ``module_for`` keys on the absolute path stored on the SymbolNode.
    svc_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.svc")
    node = ctx.module_for(svc_path)
    assert node is not None
    idx = ctx.module_for_indices(svc_path)
    assert idx is not None
    assert ctx.nodes()[idx].fqname == node.fqname


def test_module_for_indices_missing_path(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.module_for_indices("/not/a/real/path.py") is None


def test_modules_for_paths_bulk(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def f(): pass\n",
            "pkg/util.py": "def g(): pass\n",
        }
    )
    svc_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.svc")
    util_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.util")
    results = ctx.modules_for_paths([svc_path, util_path, "/missing.py"])
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    all_nodes = ctx.nodes()
    assert all_nodes[results[0]].fqname == "pkg.svc"
    assert all_nodes[results[1]].fqname == "pkg.util"


# ---------------------------------------------------------------------------
# ctx.module_surface_indices + module_surfaces_indices
# ---------------------------------------------------------------------------


def test_module_surface_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): pass\n",
        }
    )
    nodes = ctx.module_surface("pkg")
    indices = ctx.module_surface_indices("pkg")
    assert len(nodes) == len(indices)
    assert {ctx.nodes()[i].fqname for i in indices} == {n.fqname for n in nodes}


def test_module_surface_indices_unknown_returns_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.module_surface_indices("nothing.here") == []


def test_module_surfaces_indices_bulk(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/util.py": "def helper(): pass\n",
            "other/__init__.py": "",
            "other/m.py": "def m(): pass\n",
        }
    )
    node_buckets = ctx.module_surfaces(["pkg", "other", "missing"])
    idx_buckets = ctx.module_surfaces_indices(["pkg", "other", "missing"])
    assert set(node_buckets.keys()) == set(idx_buckets.keys())
    for key in node_buckets:
        node_fqs = {n.fqname for n in node_buckets[key]}
        idx_fqs = {ctx.nodes()[i].fqname for i in idx_buckets[key]}
        assert node_fqs == idx_fqs


# ---------------------------------------------------------------------------
# ctx.find_main_blocks_indices
# ---------------------------------------------------------------------------


def test_find_main_blocks_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/m.py": (
                'def helper(): pass\nif __name__ == "__main__":\n    helper()\n    x = 1\n'
            ),
        }
    )
    node_pairs = ctx.find_main_blocks()
    idx_pairs = ctx.find_main_blocks_indices()
    assert len(node_pairs) == len(idx_pairs) == 1
    (mod_node, decls) = node_pairs[0]
    (mod_idx, decl_idxs) = idx_pairs[0]
    assert ctx.nodes()[mod_idx].fqname == mod_node.fqname
    assert {ctx.nodes()[i].fqname for i in decl_idxs} == {d.fqname for d in decls}


def test_find_main_blocks_indices_no_main_block(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.find_main_blocks_indices() == []


# ---------------------------------------------------------------------------
# ctx.node_attrs — batched node-field snapshot
# ---------------------------------------------------------------------------


def test_node_attrs_round_trip(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def alpha(): pass\nclass Beta: pass\n",
        }
    )
    all_nodes = ctx.nodes()
    indices = list(range(len(all_nodes)))
    rows = ctx.node_attrs(indices)
    assert len(rows) == len(indices)
    for idx, (kind, path, fqname, flags) in zip(indices, rows, strict=True):
        n = all_nodes[idx]
        assert kind == n.kind
        assert path == n.path
        assert fqname == n.fqname
        assert flags == n.flags


def test_node_attrs_subset(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def alpha(): pass\n"})
    indices = ctx.indices_where(kind="function", fqname_prefix="pkg.a")
    assert len(indices) == 1
    [(kind, path, fqname, _flags)] = ctx.node_attrs(indices)
    assert kind == "function"
    assert fqname == "pkg.a.alpha"
    assert path.endswith("a.py")


def test_node_attrs_bounds_check(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.node_attrs([n])


# ---------------------------------------------------------------------------
# *Query.row_indices — index-form row terminals
# ---------------------------------------------------------------------------


def test_decorator_query_row_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": ("import functools\n@functools.lru_cache\ndef cached(): pass\n"),
        }
    )
    q = native.query(ctx).decorators().where_module("functools").where_name("lru_cache")
    refs = q.collect()
    rows = q.row_indices()
    assert len(refs) == len(rows) == 1
    all_nodes = ctx.nodes()
    for ref, row in zip(refs, rows, strict=True):
        assert all_nodes[row.decorated_idx].fqname == ref.decorated.fqname
        assert row.decorator_name == ref.decorator_name
        assert row.decorator_owner == ref.decorator_owner
        assert row.decorator_via == ref.decorator_via


def test_construction_query_row_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/app.py": "import flask\napp = flask.Flask(__name__)\n",
        }
    )
    q = native.query(ctx).constructions().where_module("flask").where_name("Flask")
    refs = q.collect()
    rows = q.row_indices()
    assert len(refs) == len(rows) == 1
    all_nodes = ctx.nodes()
    for ref, row in zip(refs, rows, strict=True):
        assert all_nodes[row.var_idx].fqname == ref.var.fqname
        assert row.class_name == ref.class_name


def test_call_query_row_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/app.py": (
                'import flask\napp = flask.Flask(__name__)\napp.config.from_object("settings")\n'
            ),
        }
    )
    q = (
        native.query(ctx)
        .calls()
        .where_owner("app.config")
        .where_attr("from_object")
        .string_arg_at(0)
    )
    refs = q.collect()
    rows = q.row_indices()
    assert len(refs) == len(rows)
    if refs:
        all_nodes = ctx.nodes()
        for ref, row in zip(refs, rows, strict=True):
            assert all_nodes[row.owner_idx].fqname == ref.owner.fqname
            assert row.string_arg == ref.string_arg


def test_factory_query_row_indices_matches_collect(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/factory.py": ("import flask\ndef make_app():\n    return flask.Flask(__name__)\n"),
        }
    )
    q = native.query(ctx).factories().of_module("flask").where_name("Flask")
    refs = q.collect()
    rows = q.row_indices()
    assert len(refs) == len(rows)
    all_nodes = ctx.nodes()
    for ref, row in zip(refs, rows, strict=True):
        assert all_nodes[row.decl_idx].fqname == ref.decl.fqname
        assert row.kinds == ref.kinds


# ---------------------------------------------------------------------------
# AddEntrypointByIdx — graph op
# ---------------------------------------------------------------------------


def test_add_entrypoint_by_idx_promotes_seed(tmp_path):
    """``AddEntrypointByIdx(idx, marker=...)`` mints the same
    ``<marker>:<decl.fqname>`` synthetic the node-form ``AddEntrypoint``
    does, and the decl + everything it reaches stay alive."""
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text(
        "def used_by_seed(): pass\ndef seed(): used_by_seed()\ndef dead(): pass\n"
    )

    class SeedByIdx(Plugin):
        name = "seed_by_idx"
        version = 1

        def run(self, ctx: native.ProjectContext):
            (seed_idx,) = ctx.indices_where(fqname_prefix="mod.seed")
            yield native.AddEntrypointByIdx(seed_idx, marker="<seed>")

    analysis = Analysis(tmp_path, plugins=[SeedByIdx()])
    ctx = analysis.materialize_all()
    fqnames = [n.fqname for n in ctx.nodes()]
    # Marker is composed as ``<marker>:<decl.fqname>`` — same shape
    # ``AddEntrypoint`` uses.
    assert "<seed>:mod.seed" in fqnames
    dead_fqnames = {n.fqname for n in analysis.dead()}
    assert "mod.seed" not in dead_fqnames
    assert "mod.used_by_seed" not in dead_fqnames
    assert "mod.dead" in dead_fqnames


def test_add_entrypoint_by_idx_out_of_range_raises(tmp_path):
    from dead_cst.analyze import Analysis
    from dead_cst.plugins._base import Plugin

    (tmp_path / "mod.py").write_text("def f(): pass\n")

    class BadIdx(Plugin):
        name = "bad_idx"
        version = 1

        def run(self, ctx: native.ProjectContext):
            yield native.AddEntrypointByIdx(10**9, marker="<seed>")

    analysis = Analysis(tmp_path, plugins=[BadIdx()])
    with pytest.raises(IndexError, match="decl_idx"):
        analysis.materialize_all()


# ---------------------------------------------------------------------------
# ctx.find_module_idx
# ---------------------------------------------------------------------------


def test_find_module_idx_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {"pkg/__init__.py": "", "pkg/sub/__init__.py": "", "pkg/sub/inner.py": "x = 1\n"}
    )
    for fqn in ("pkg", "pkg.sub", "pkg.sub.inner"):
        node = ctx.find_module(fqn)
        idx = ctx.find_module_idx(fqn)
        assert node is not None
        assert idx is not None
        assert ctx.nodes()[idx].fqname == node.fqname


def test_find_module_idx_unknown_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.find_module_idx("nothing.here") is None


# ---------------------------------------------------------------------------
# ctx.find_module_dunders_indices
# ---------------------------------------------------------------------------


def test_find_module_dunders_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "__all__ = []\n__version__ = '1.0'\n",
            "pkg/a.py": "def __getattr__(name): pass\nx = 1\n",
        }
    )
    nodes = ctx.find_module_dunders()
    indices = ctx.find_module_dunders_indices()
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    # ``x = 1`` is a plain top-level variable — must not be surfaced.
    assert all("x" not in n.fqname.rsplit(".", 1)[-1] for n in revived)


# ---------------------------------------------------------------------------
# ctx.find_nodes_matching_specs_indices
# ---------------------------------------------------------------------------


def test_find_nodes_matching_specs_indices_matches_node_form(build_decl_graph, tmp_path):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/util.py": "def helper(): pass\n",
        }
    )
    # Use the same three spec buckets the ExplicitEntrypointPlugin
    # exercises: regex, str (relative path or fqname), abs path.
    nodes = ctx.find_nodes_matching_specs(
        str(tmp_path),
        regexes=[r"pkg/svc\.py"],
        str_specs=["pkg.util.helper"],
        abs_paths=[],
    )
    indices = ctx.find_nodes_matching_specs_indices(
        str(tmp_path),
        regexes=[r"pkg/svc\.py"],
        str_specs=["pkg.util.helper"],
        abs_paths=[],
    )
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    assert {"pkg.svc.handler", "pkg.util.helper"} <= {n.fqname for n in revived}


# ---------------------------------------------------------------------------
# ctx.subclasses_of_fqn_indices + SubclassQuery.of_idx +
# ctx.find_subclasses_of_idx
# ---------------------------------------------------------------------------


def test_subclasses_of_fqn_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/bases.py": "class Base: pass\n",
            "pkg/sub.py": (
                "from pkg.bases import Base\nclass Mid(Base): pass\nclass Leaf(Mid): pass\n"
            ),
        }
    )
    nodes = ctx.subclasses_of_fqn("pkg.bases.Base", transitive=True)
    indices = ctx.subclasses_of_fqn_indices("pkg.bases.Base", transitive=True)
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}


def test_subclasses_of_fqn_indices_transitive_flag(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/bases.py": "class Base: pass\n",
            "pkg/sub.py": (
                "from pkg.bases import Base\nclass Mid(Base): pass\nclass Leaf(Mid): pass\n"
            ),
        }
    )
    direct = ctx.subclasses_of_fqn_indices("pkg.bases.Base", transitive=False)
    direct_revived = {n.fqname for n in ctx.nodes_at(direct)}
    assert direct_revived == {"pkg.sub.Mid"}  # Leaf is not a *direct* subclass of Base


def test_find_subclasses_of_idx_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "class Base: pass\nclass Sub(Base): pass\n",
        }
    )
    (base_idx,) = ctx.indices_where(fqname_prefix="pkg.a.Base", kind="class")
    base_node = ctx.nodes_at([base_idx])[0]
    nodes = ctx.find_subclasses_of(base_node)
    idx_results = ctx.find_subclasses_of_idx(base_idx)
    assert len(nodes) == len(idx_results)
    revived = ctx.nodes_at(idx_results)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}


def test_find_subclasses_of_idx_non_class_returns_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    (f_idx,) = ctx.indices_where(fqname_prefix="pkg.a.f", kind="function")
    # Same contract as the node-form: non-class seed returns empty.
    assert ctx.find_subclasses_of_idx(f_idx) == []


def test_find_subclasses_of_idx_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "class X: pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.find_subclasses_of_idx(n)


def test_subclass_query_of_idx_matches_of_node(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "class Base: pass\nclass Sub(Base): pass\n",
        }
    )
    (base_idx,) = ctx.indices_where(fqname_prefix="pkg.a.Base", kind="class")
    base_node = ctx.nodes_at([base_idx])[0]
    by_node = native.query(ctx).subclasses().of_node(base_node).indices()
    by_idx = native.query(ctx).subclasses().of_idx(base_idx).indices()
    assert sorted(by_node) == sorted(by_idx)


# ---------------------------------------------------------------------------
# ctx.direct_predecessors_idx
# ---------------------------------------------------------------------------


def test_direct_predecessors_idx_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def callee(): pass\ndef caller(): callee()\ncaller()\n",
        }
    )
    (callee_idx,) = ctx.indices_where(fqname_prefix="pkg.a.callee", kind="function")
    callee_node = ctx.nodes_at([callee_idx])[0]
    nodes = ctx.direct_predecessors(callee_node)
    indices = ctx.direct_predecessors_idx(callee_idx)
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    # ``caller`` definitely calls ``callee`` — sanity check.
    assert "pkg.a.caller" in {n.fqname for n in revived}


def test_direct_predecessors_idx_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.direct_predecessors_idx(n)


# ---------------------------------------------------------------------------
# DecoratorQuery.in_decl_idx
# ---------------------------------------------------------------------------


def test_decorator_query_in_decl_idx_matches_in_decl(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/app.py": (
                "from cyclopts import App\n"
                "cli = App()\n"
                "@cli.command\n"
                "def hello(): pass\n"
                "@cli.command\n"
                "def bye(): pass\n"
            ),
        }
    )
    (cli_idx,) = ctx.indices_where(fqname_prefix="pkg.app.cli", kind="variable")
    cli_node = ctx.nodes_at([cli_idx])[0]
    by_node = native.query(ctx).decorators().in_decl(cli_node).where_name("command").row_indices()
    by_idx = native.query(ctx).decorators().in_decl_idx(cli_idx).where_name("command").row_indices()
    assert len(by_node) == len(by_idx)
    assert sorted(r.decorated_idx for r in by_node) == sorted(r.decorated_idx for r in by_idx)


def test_decorator_query_in_decl_idx_requires_where_name(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    (x_idx,) = ctx.indices_where(fqname_prefix="pkg.a.x", kind="variable")
    with pytest.raises(ValueError, match="where_name"):
        native.query(ctx).decorators().in_decl_idx(x_idx).row_indices()


def test_decorator_query_in_decl_idx_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        native.query(ctx).decorators().in_decl_idx(n).where_name("command").row_indices()


# ---------------------------------------------------------------------------
# ctx.resolve_idx
# ---------------------------------------------------------------------------


def test_resolve_idx_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
        }
    )
    # Exact decl, exact module, walk-back from method-level fqname.
    for fqn in ("pkg.svc.handler", "pkg.svc", "pkg.svc.Service.method"):
        node = ctx.resolve(fqn)
        idx = ctx.resolve_idx(fqn)
        assert node is not None
        assert idx is not None
        assert ctx.nodes()[idx].fqname == node.fqname


def test_resolve_idx_unknown_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.resolve_idx("nothing.here") is None


# ---------------------------------------------------------------------------
# ctx.decls_under_indices / decls_matching_indices / decls_matching_name_indices
# ---------------------------------------------------------------------------


def test_decls_under_indices_matches_node_form(build_decl_graph, tmp_path):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/util.py": "def helper(): pass\n",
            "other/__init__.py": "",
            "other/m.py": "def m(): pass\n",
        }
    )
    prefix = str(tmp_path / "pkg")
    nodes = ctx.decls_under(prefix)
    indices = ctx.decls_under_indices(prefix)
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    # Sanity: nothing under "other" leaked into the bucket.
    assert all(not n.fqname.startswith("other.") for n in revived)


def test_decls_matching_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/test_a.py": "def test_one(): pass\n",
            "pkg/main.py": "def main(): pass\n",
        }
    )
    nodes = ctx.decls_matching("test_a")
    indices = ctx.decls_matching_indices("test_a")
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}


def test_decls_matching_name_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def test_one(): pass\ndef helper(): pass\nclass TestX: pass\n",
        }
    )
    nodes = ctx.decls_matching_name(r"^(test_|Test)")
    indices = ctx.decls_matching_name_indices(r"^(test_|Test)")
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    assert {"pkg.a.test_one", "pkg.a.TestX"} <= {n.fqname for n in revived}
    # The kind filter excludes module nodes — verify a module's name
    # isn't accidentally surfaced even though "a" matches no pattern here.
    assert all(n.kind != "module" for n in revived)


# ---------------------------------------------------------------------------
# ctx.descendants_indices / ancestors_indices
# ---------------------------------------------------------------------------


def test_descendants_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": ("def leaf(): pass\ndef middle(): leaf()\ndef root(): middle()\nroot()\n"),
        }
    )
    (root_idx,) = ctx.indices_where(fqname_prefix="pkg.a.root", kind="function")
    root_node = ctx.nodes_at([root_idx])[0]
    nodes = ctx.descendants(root_node)
    indices = ctx.descendants_indices(root_idx)
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    # Sanity: the forward closure includes the transitive targets.
    assert {"pkg.a.middle", "pkg.a.leaf"} <= {n.fqname for n in revived}


def test_descendants_indices_skip_flags(build_decl_graph):
    """``skip_flags`` parity: DEAD_BRANCH edges are filtered the same
    way by both forms."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": (
                "def alive_only_in_dead_branch(): pass\n"
                "def caller():\n"
                "    if False:\n"
                "        alive_only_in_dead_branch()\n"
                "caller()\n"
            ),
        }
    )
    (caller_idx,) = ctx.indices_where(fqname_prefix="pkg.a.caller", kind="function")
    caller_node = ctx.nodes_at([caller_idx])[0]
    skip = int(native.EdgeFlags.DEAD_BRANCH)
    nodes = ctx.descendants(caller_node, skip_flags=skip)
    indices = ctx.descendants_indices(caller_idx, skip_flags=skip)
    assert {n.fqname for n in nodes} == {ctx.nodes()[i].fqname for i in indices}


def test_descendants_indices_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.descendants_indices(n)


def test_ancestors_indices_matches_node_form(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": ("def leaf(): pass\ndef middle(): leaf()\ndef root(): middle()\nroot()\n"),
        }
    )
    (leaf_idx,) = ctx.indices_where(fqname_prefix="pkg.a.leaf", kind="function")
    leaf_node = ctx.nodes_at([leaf_idx])[0]
    nodes = ctx.ancestors(leaf_node)
    indices = ctx.ancestors_indices(leaf_idx)
    assert len(nodes) == len(indices)
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {n.fqname for n in nodes}
    assert {"pkg.a.middle", "pkg.a.root"} <= {n.fqname for n in revived}


def test_ancestors_indices_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.ancestors_indices(n)
