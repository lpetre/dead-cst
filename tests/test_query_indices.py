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
# ctx.modules_for_paths — bulk path → module idx lookup
# ---------------------------------------------------------------------------


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
# ctx.module_surfaces_indices — bulk module-surface lookup
# ---------------------------------------------------------------------------


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
    idx_buckets = ctx.module_surfaces_indices(["pkg", "other", "missing"])
    assert "missing" in idx_buckets and idx_buckets["missing"] == []
    fqs_pkg = {ctx.nodes()[i].fqname for i in idx_buckets["pkg"]}
    fqs_other = {ctx.nodes()[i].fqname for i in idx_buckets["other"]}
    # ``pkg`` surface is the module + every transitive decl under it.
    assert "pkg.svc.handler" in fqs_pkg
    assert "pkg.util.helper" in fqs_pkg
    assert "other.m.m" in fqs_other
    # Buckets must not bleed across keys.
    assert not (fqs_pkg & fqs_other)


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
# ctx.node_paths — slim variant of node_attrs
# ---------------------------------------------------------------------------


def test_node_paths_matches_node_attrs_path_field(build_decl_graph):
    """``node_paths(idxs)`` must return the same ``path`` value
    ``node_attrs(idxs)`` returns at row position 1 — parity is the
    whole contract; the only difference is allocation count."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def f(): pass\nclass C: pass\n",
            "pkg/b.py": "x = 1\n",
        }
    )
    all_idxs = list(range(len(ctx.nodes())))
    paths = ctx.node_paths(all_idxs)
    attrs = ctx.node_attrs(all_idxs)
    assert len(paths) == len(attrs) == len(all_idxs)
    for path, (_kind, attr_path, _fq, _flags) in zip(paths, attrs, strict=True):
        assert path == attr_path


def test_node_paths_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.node_paths([]) == []


def test_node_paths_bounds_check(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.node_paths([n])


# ---------------------------------------------------------------------------
# Ref-query .collect() terminals — idx-shape contract
# ---------------------------------------------------------------------------


def test_decorator_query_collect_returns_idx_rows(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": ("import functools\n@functools.lru_cache\ndef cached(): pass\n"),
        }
    )
    rows = (
        native.query(ctx).decorators().where_module("functools").where_name("lru_cache").collect()
    )
    assert len(rows) == 1
    [row] = rows
    assert isinstance(row, native.DecoratorIdxRef)
    assert ctx.nodes()[row.decorated_idx].fqname == "pkg.svc.cached"
    assert row.path.endswith("svc.py")


def test_construction_query_collect_returns_idx_rows(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/app.py": "import flask\napp = flask.Flask(__name__)\n",
        }
    )
    rows = native.query(ctx).constructions().where_module("flask").where_name("Flask").collect()
    assert len(rows) == 1
    [row] = rows
    assert isinstance(row, native.ConstructionIdxRef)
    assert ctx.nodes()[row.var_idx].fqname == "pkg.app.app"
    assert row.class_name == "Flask"


def test_call_query_collect_returns_idx_rows(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/app.py": (
                'import flask\napp = flask.Flask(__name__)\napp.config.from_object("settings")\n'
            ),
        }
    )
    rows = (
        native.query(ctx)
        .calls()
        .where_owner("app.config")
        .where_attr("from_object")
        .string_arg_at(0)
        .collect()
    )
    if rows:
        [row] = rows
        assert isinstance(row, native.CallIdxRef)
        assert ctx.nodes()[row.owner_idx].fqname == "pkg.app"
        assert row.string_arg == "settings"


def test_with_args_opt_in_populates_args_kwargs(build_decl_graph):
    """``with_args(True)`` opts into the rust-side
    ``extract_call_args_kwargs`` walk. Default (``with_args(False)``)
    skips it; row ``args`` / ``kwargs`` getters surface empty
    containers, but node-identity + metadata strings populate
    normally."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": (
                "import functools\n@functools.lru_cache(maxsize=128)\ndef cached(): pass\n"
            ),
        }
    )
    default_rows = (
        native.query(ctx).decorators().where_module("functools").where_name("lru_cache").collect()
    )
    assert len(default_rows) == 1
    # Default skips extraction.
    assert list(default_rows[0].args) == []
    assert dict(default_rows[0].kwargs) == {}

    with_args_rows = (
        native.query(ctx)
        .decorators()
        .where_module("functools")
        .where_name("lru_cache")
        .with_args(True)
        .collect()
    )
    assert len(with_args_rows) == 1
    # ``with_args(True)`` populates args/kwargs.
    assert dict(with_args_rows[0].kwargs)
    # Identity + metadata fields stable across both calls.
    assert with_args_rows[0].decorated_idx == default_rows[0].decorated_idx
    assert with_args_rows[0].decorator_owner == default_rows[0].decorator_owner


def test_where_kwarg_forces_extraction(build_decl_graph):
    """``.where_kwarg(...)`` filters even at the default
    ``with_args=False`` — the rust side forces extraction back on
    when any kwarg matcher is set."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": (
                "import functools\n"
                "@functools.lru_cache(maxsize=128)\n"
                "def big(): pass\n"
                "@functools.lru_cache(maxsize=1)\n"
                "def small(): pass\n"
            ),
        }
    )
    rows = (
        native.query(ctx)
        .decorators()
        .where_module("functools")
        .where_name("lru_cache")
        .where_kwarg("maxsize", 128)
        .collect()
    )
    assert len(rows) == 1
    fqnames = {ctx.nodes()[r.decorated_idx].fqname for r in rows}
    assert fqnames == {"pkg.svc.big"}


def test_factory_query_collect_returns_idx_rows(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/factory.py": ("import flask\ndef make_app():\n    return flask.Flask(__name__)\n"),
        }
    )
    rows = native.query(ctx).factories().of_module("flask").where_name("Flask").collect()
    assert len(rows) == 1
    [row] = rows
    assert isinstance(row, native.FactoryIdxRef)
    assert ctx.nodes()[row.decl_idx].fqname == "pkg.factory.make_app"
    assert "Flask" in row.kinds


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
# SubclassQuery.of_fqn / of_idx / of_node
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
    indices = native.query(ctx).subclasses().of_fqn("pkg.bases.Base").indices()
    revived = ctx.nodes_at(indices)
    assert {n.fqname for n in revived} == {"pkg.sub.Mid", "pkg.sub.Leaf"}


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
    direct = native.query(ctx).subclasses().of_fqn("pkg.bases.Base").transitive(False).indices()
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
    idx_results = native.query(ctx).subclasses().of_idx(base_idx).indices()
    revived = ctx.nodes_at(idx_results)
    assert {n.fqname for n in revived} == {"pkg.a.Sub"}


def test_find_subclasses_of_idx_non_class_returns_empty(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    (f_idx,) = ctx.indices_where(fqname_prefix="pkg.a.f", kind="function")
    # Non-class seed → empty result.
    assert native.query(ctx).subclasses().of_idx(f_idx).indices() == []


def test_find_subclasses_of_idx_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "class X: pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        native.query(ctx).subclasses().of_idx(n).indices()


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
    by_node = native.query(ctx).decorators().in_decl(cli_node).where_name("command").collect()
    by_idx = native.query(ctx).decorators().in_decl_idx(cli_idx).where_name("command").collect()
    assert len(by_node) == len(by_idx)
    assert sorted(r.decorated_idx for r in by_node) == sorted(r.decorated_idx for r in by_idx)


def test_decorator_query_in_decl_idx_requires_where_name(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    (x_idx,) = ctx.indices_where(fqname_prefix="pkg.a.x", kind="variable")
    with pytest.raises(ValueError, match="where_name"):
        native.query(ctx).decorators().in_decl_idx(x_idx).collect()


def test_decorator_query_in_decl_idx_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        native.query(ctx).decorators().in_decl_idx(n).where_name("command").collect()


# ---------------------------------------------------------------------------
# TraverseQuery bounds — out-of-range seeds raise IndexError
# ---------------------------------------------------------------------------


def test_traverse_descendants_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        native.query(ctx).from_idx(n).descendants()


def test_traverse_ancestors_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        native.query(ctx).from_idx(n).ancestors()


def test_traverse_direct_predecessors_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        native.query(ctx).from_idx(n).direct_predecessors()


# ---------------------------------------------------------------------------
# ModuleQuery — DSL stream for module-shaped lookups
# ---------------------------------------------------------------------------


def test_module_query_with_fqn_returns_module_idx(build_decl_graph):
    ctx = build_decl_graph(
        {"pkg/__init__.py": "", "pkg/sub/__init__.py": "", "pkg/sub/inner.py": "x = 1\n"}
    )
    idx = native.query(ctx).modules().with_fqn("pkg.sub.inner").first_idx()
    assert idx is not None
    assert ctx.nodes()[idx].fqname == "pkg.sub.inner"


def test_module_query_with_fqn_missing_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert native.query(ctx).modules().with_fqn("nothing.here").first_idx() is None
    assert native.query(ctx).modules().with_fqn("nothing.here").indices() == []


def test_module_query_with_path(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/svc.py": "def f(): pass\n"})
    svc_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.svc")
    idx = native.query(ctx).modules().with_path(svc_path).first_idx()
    assert idx is not None
    assert ctx.nodes()[idx].fqname == "pkg.svc"


def test_module_query_surface_includes_submodules(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): pass\n",
        }
    )
    indices = native.query(ctx).modules().with_fqn("pkg").surface().indices()
    fqnames = {ctx.nodes()[i].fqname for i in indices}
    assert "pkg" in fqnames
    assert "pkg.svc" in fqnames
    assert "pkg.svc.handler" in fqnames
    assert "pkg.sub.inner" in fqnames
    assert "pkg.sub.inner.deep" in fqnames


def test_module_query_top_level_excludes_submodules(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): pass\n",
        }
    )
    indices = native.query(ctx).modules().with_fqn("pkg").top_level().indices()
    fqnames = {ctx.nodes()[i].fqname for i in indices}
    # ``pkg/__init__.py`` has no top-level decls of its own; the
    # submodule node itself is excluded by the contract.
    assert "pkg.svc" not in fqnames
    assert "pkg.sub" not in fqnames


def test_module_query_dunder_all_returns_listed_exports(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/m.py": '__all__ = ["f"]\ndef f(): pass\ndef g(): pass\n',
        }
    )
    idxs = native.query(ctx).modules().with_fqn("pkg.m").dunder_all()
    assert idxs is not None
    fqnames = {ctx.nodes()[i].fqname for i in idxs}
    assert fqnames == {"pkg.m.f"}


def test_module_query_dunder_all_none_when_unset(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    assert native.query(ctx).modules().with_fqn("pkg.m").dunder_all() is None


def test_module_query_with_dunders_project_wide(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "__all__ = []\n__version__ = '1.0'\n",
            "pkg/a.py": "def __getattr__(name): pass\nx = 1\n",
        }
    )
    idxs = native.query(ctx).modules().with_dunders().indices()
    fqnames = {ctx.nodes()[i].fqname for i in idxs}
    assert "pkg.__all__" in fqnames
    assert "pkg.__version__" in fqnames
    assert "pkg.a.__getattr__" in fqnames
    assert "pkg.a.x" not in fqnames


def test_module_query_no_filter_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    with pytest.raises(ValueError, match="with_fqn"):
        native.query(ctx).modules().indices()


# ---------------------------------------------------------------------------
# TraverseQuery — parity with ctx.{descendants,ancestors,direct_predecessors}_*
# ---------------------------------------------------------------------------


def test_traverse_descendants_matches_ctx_descendants_indices(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": """
            def used(): pass
            def caller(): used()
            caller()
            """,
        }
    )
    main_idx = native.query(ctx).modules().with_fqn("pkg.main").first_idx()
    assert main_idx is not None
    via_traverse = native.query(ctx).from_idx(main_idx).descendants()
    via_ctx = ctx.descendants_indices(main_idx)
    assert sorted(via_traverse) == sorted(via_ctx)


def test_traverse_ancestors_matches_ctx_ancestors_indices(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": """
            def used(): pass
            def caller(): used()
            caller()
            """,
        }
    )
    used_idx = native.query(ctx).declarations().with_fqname("pkg.main.used").resolve_idx()
    assert used_idx is not None
    via_traverse = native.query(ctx).from_idx(used_idx).ancestors()
    via_ctx = ctx.ancestors_indices(used_idx)
    assert sorted(via_traverse) == sorted(via_ctx)


def test_traverse_direct_predecessors_matches_ctx(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": "def f(): pass\ndef g(): f()\ndef h(): f()\n",
        }
    )
    f_idx = native.query(ctx).declarations().with_fqname("pkg.main.f").resolve_idx()
    assert f_idx is not None
    via_traverse = native.query(ctx).from_idx(f_idx).direct_predecessors()
    via_ctx = ctx.direct_predecessors_idx(f_idx)
    assert sorted(via_traverse) == sorted(via_ctx)


def test_traverse_descendants_skip_flags(build_decl_graph):
    """``skip_flags`` plumbs through to the underlying BFS."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
            def live(): pass
            if False:
                def dead(): pass
                live()
            """,
        }
    )
    mod_idx = native.query(ctx).modules().with_fqn("pkg.m").first_idx()
    assert mod_idx is not None
    full = native.query(ctx).from_idx(mod_idx).descendants()
    strict = (
        native.query(ctx).from_idx(mod_idx).descendants(skip_flags=native.EdgeFlags.DEAD_BRANCH)
    )
    # Strict closure is a subset of the dead-branch-traversing closure.
    assert set(strict).issubset(set(full))


def test_query_reachable_matches_ctx_reachable_indices(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": """
            def alive(): pass
            def dead(): pass
            alive()
            """,
        }
    )
    via_query = native.query(ctx).reachable()
    via_ctx = ctx.reachable_indices()
    assert sorted(via_query) == sorted(via_ctx)


def test_query_matching_specs_or_form(build_decl_graph, tmp_path):
    """``QueryBuilder.matching_specs`` ORs across the three buckets:
    a node matches if it satisfies any of the regex / str / abs_path
    sets. Mirrors ``ExplicitEntrypointPlugin`` shape."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/util.py": "def helper(): pass\n",
        }
    )
    util_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.util")
    indices = native.query(ctx).matching_specs(
        str(tmp_path),
        regexes=[r"pkg/svc\.py"],
        str_specs=["pkg.util.helper"],
        abs_paths=[util_path],
    )
    fqnames = {n.fqname for n in ctx.nodes_at(indices)}
    # Regex bucket pulls pkg.svc + its decl; str-spec bucket pulls
    # pkg.util.helper; abs-path bucket pulls the pkg.util module.
    assert {"pkg.svc.handler", "pkg.util.helper", "pkg.util"} <= fqnames


def test_query_reachable_seed_flags_kwarg(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": "def f(): pass\nf()\n",
        }
    )
    via_query = native.query(ctx).reachable(
        skip_flags=native.EdgeFlags.DEAD_BRANCH,
        seed_flags=native.NodeFlags.ENTRYPOINT,
    )
    via_ctx = ctx.reachable_indices(
        skip_flags=native.EdgeFlags.DEAD_BRANCH,
        seed_flags=native.NodeFlags.ENTRYPOINT,
    )
    assert sorted(via_query) == sorted(via_ctx)


# ---------------------------------------------------------------------------
# DeclarationsQuery — parity with ctx.find_declarations_indices / resolve_idx
# ---------------------------------------------------------------------------


def test_declarations_query_indices_matches_ctx(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "class Cls:\n    def method(self): pass\n",
        }
    )
    # Walk-back: pkg.lib.Cls.method → pkg.lib.Cls.
    via_query = native.query(ctx).declarations().with_fqname("pkg.lib.Cls.method").indices()
    via_ctx = ctx.find_declarations_indices("pkg.lib.Cls.method")
    assert sorted(via_query) == sorted(via_ctx)
    assert len(via_query) == 1
    assert ctx.nodes()[via_query[0]].fqname == "pkg.lib.Cls"


def test_declarations_query_resolve_idx_falls_back_to_module(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/lib.py": "def f(): pass\n"})
    # find_declarations excludes modules — resolve_idx includes them.
    decls = native.query(ctx).declarations().with_fqname("pkg.lib").indices()
    assert decls == []
    resolved = native.query(ctx).declarations().with_fqname("pkg.lib").resolve_idx()
    assert resolved is not None
    assert ctx.nodes()[resolved].kind == "module"


def test_declarations_query_resolve_idx_unknown_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    assert native.query(ctx).declarations().with_fqname("nowhere.such.name").resolve_idx() is None


def test_declarations_query_no_fqname_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    with pytest.raises(ValueError, match="with_fqname"):
        native.query(ctx).declarations().indices()


# ---------------------------------------------------------------------------
# MainBlockQuery — parity with ctx.find_main_blocks_indices
# ---------------------------------------------------------------------------


def test_main_block_query_matches_ctx(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/script.py": """
            def helper(): pass

            if __name__ == "__main__":
                def inner_decl(): pass
                helper()
            """,
            "pkg/lib.py": "def lib_fn(): pass\n",
        }
    )
    via_query = native.query(ctx).main_blocks().index_pairs()
    via_ctx = ctx.find_main_blocks_indices()
    # Same pairs, same shapes.
    assert len(via_query) == len(via_ctx) == 1
    q_mod, q_decls = via_query[0]
    c_mod, c_decls = via_ctx[0]
    assert q_mod == c_mod
    assert sorted(q_decls) == sorted(c_decls)


def test_main_block_query_no_main_block(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/lib.py": "def f(): pass\n"})
    assert native.query(ctx).main_blocks().index_pairs() == []


# ---------------------------------------------------------------------------
# LiteralListQuery — parity with ctx.find_literal_list_entries
# ---------------------------------------------------------------------------


def test_literal_list_query_matches_ctx(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
            __all__ = ["a", "b"]
            def a(): pass
            def b(): pass
            """,
        }
    )
    via_query = native.query(ctx).literal_lists().for_fqn("pkg.m.__all__").entries()
    via_ctx = ctx.find_literal_list_entries("pkg.m.__all__")
    assert via_query == via_ctx == ["a", "b"]


def test_literal_list_query_unknown_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    assert native.query(ctx).literal_lists().for_fqn("pkg.nope").entries() is None


def test_literal_list_query_no_fqn_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    with pytest.raises(ValueError, match="for_fqn"):
        native.query(ctx).literal_lists().entries()


# ---------------------------------------------------------------------------
# DeclQuery.with_path_prefix / with_path_contains / with_simple_name_regex
# ---------------------------------------------------------------------------


def test_decl_query_with_path_prefix_matches_decls_under(build_decl_graph, tmp_path):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): pass\n",
            "pkg/top.py": "def shallow(): pass\n",
        }
    )
    prefix = str(tmp_path / "pkg" / "sub")
    via_query = native.query(ctx).decls().with_path_prefix(prefix).indices()
    via_ctx = ctx.decls_under_indices(prefix)
    assert sorted(via_query) == sorted(via_ctx)


def test_decl_query_with_path_contains_matches_decls_matching(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/test_a.py": "def t_a(): pass\n",
            "pkg/lib.py": "def f(): pass\n",
        }
    )
    via_query = native.query(ctx).decls().with_path_contains("test_a").indices()
    via_ctx = ctx.decls_matching_indices("test_a")
    assert sorted(via_query) == sorted(via_ctx)


def test_decl_query_with_simple_name_regex_matches_decls_matching_name(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def test_one(): pass\nclass TestThing: pass\ndef helper(): pass\n",
        }
    )
    # decls_matching_name implicitly filters to function|class|variable|import|type_alias;
    # DeclQuery composes — apply the same kind filter explicitly.
    via_query = (
        native.query(ctx)
        .decls()
        .with_simple_name_regex(r"^(test_|Test)")
        .with_kinds(["function", "class", "variable", "import", "type_alias"])
        .indices()
    )
    via_ctx = ctx.decls_matching_name_indices(r"^(test_|Test)")
    assert sorted(via_query) == sorted(via_ctx)


def test_decl_query_path_and_simple_name_compose(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/test_a.py": "def test_one(): pass\ndef helper(): pass\n",
            "pkg/lib.py": "def test_one(): pass\n",
        }
    )
    indices = (
        native.query(ctx)
        .decls()
        .with_path_contains("test_a")
        .with_simple_name_regex(r"^test_")
        .with_kind("function")
        .indices()
    )
    fqnames = {n.fqname for n in ctx.nodes_at(indices)}
    assert fqnames == {"pkg.test_a.test_one"}


def test_decl_query_simple_name_regex_invalid_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    with pytest.raises(ValueError, match="invalid simple-name regex"):
        native.query(ctx).decls().with_simple_name_regex(r"(unclosed").indices()


# ---------------------------------------------------------------------------
# NodeAttrs — tuple-like row with named fields
# ---------------------------------------------------------------------------


def test_node_attrs_attribute_access(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    (idx,) = ctx.indices_where(fqname_prefix="pkg.a.f", kind="function")
    (attr,) = ctx.node_attrs([idx])
    assert attr.kind == "function"
    assert attr.fqname == "pkg.a.f"
    assert attr.path.endswith("pkg/a.py")
    assert isinstance(attr.flags, int)


def test_node_attrs_tuple_unpacking(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    (idx,) = ctx.indices_where(fqname_prefix="pkg.a.f", kind="function")
    (attr,) = ctx.node_attrs([idx])
    kind, path, fqname, flags = attr
    assert kind == attr.kind
    assert path == attr.path
    assert fqname == attr.fqname
    assert flags == attr.flags


def test_node_attrs_subscript_and_len(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    (idx,) = ctx.indices_where(fqname_prefix="pkg.a.f", kind="function")
    (attr,) = ctx.node_attrs([idx])
    assert len(attr) == 4
    assert attr[0] == attr.kind
    assert attr[2] == attr.fqname
    assert attr[-1] == attr.flags
    with pytest.raises(IndexError):
        attr[4]


# ---------------------------------------------------------------------------
# .attrs() / .first_idx() / .indices_by_path() — uniform terminals
# ---------------------------------------------------------------------------


def test_decl_query_attrs_matches_node_attrs(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
        }
    )
    q = native.query(ctx).decls().with_kind("function")
    via_terminal = q.attrs()
    via_ctx = ctx.node_attrs(q.indices())
    assert [a.fqname for a in via_terminal] == [a.fqname for a in via_ctx]


def test_decl_query_first_idx(build_decl_graph):
    ctx = build_decl_graph(
        {"pkg/__init__.py": "", "pkg/a.py": "def alpha(): pass\ndef beta(): pass\n"}
    )
    idx = native.query(ctx).decls().with_fqname_prefix("pkg.a.alpha").first_idx()
    assert idx is not None
    assert ctx.nodes()[idx].fqname == "pkg.a.alpha"


def test_decl_query_first_idx_none_when_no_match(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    assert native.query(ctx).decls().with_fqname_prefix("nope").first_idx() is None


def test_decl_query_indices_by_path(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\nclass Service: pass\n",
            "pkg/util.py": "def helper(): pass\n",
        }
    )
    buckets = native.query(ctx).decls().with_kind("function").indices_by_path()
    fqnames_by_path = {
        path: sorted(ctx.nodes()[i].fqname for i in idxs) for path, idxs in buckets.items()
    }
    assert any(fqs == ["pkg.svc.handler"] for fqs in fqnames_by_path.values()), fqnames_by_path
    assert any(fqs == ["pkg.util.helper"] for fqs in fqnames_by_path.values()), fqnames_by_path


def test_import_query_attrs_first_idx_and_indices_by_path(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from os.path import join\n",
            "pkg/b.py": "from os.path import join as j2\n",
        }
    )
    q = native.query(ctx).imports().of("os.path")
    attrs = q.attrs()
    assert all(a.kind == "import" for a in attrs)
    assert q.first_idx() is not None
    buckets = q.indices_by_path()
    # Two distinct files both import os.path.
    assert len(buckets) == 2


def test_decorator_query_indices_by_path(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": (
                "import functools\n@functools.lru_cache(maxsize=128)\ndef cached(): pass\n"
            ),
        }
    )
    buckets = (
        native.query(ctx)
        .decorators()
        .where_module("functools")
        .where_name("lru_cache")
        .indices_by_path()
    )
    assert len(buckets) == 1
    (path, idxs) = next(iter(buckets.items()))
    assert path.endswith("pkg/svc.py")
    assert ctx.nodes()[idxs[0]].fqname == "pkg.svc.cached"
