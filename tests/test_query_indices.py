"""Tests for the idx-returning query pymethods on
:class:`ProjectContext` — the low-level surface plugins and
:class:`Analysis` drive directly.

Every method here returns positional indices into
:meth:`ProjectContext.nodes` (or one of the named-tuple ``NodeAttrs``
rows). These tests pin that idx-form contract and the behavior of each
helper.
"""

from __future__ import annotations

import pytest

from dead_cst import _native as native


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
# ctx.find_module_idx / ctx.module_for_indices / ctx.modules_for_paths
# ---------------------------------------------------------------------------


def test_find_module_idx_returns_module(build_decl_graph):
    ctx = build_decl_graph(
        {"pkg/__init__.py": "", "pkg/sub/__init__.py": "", "pkg/sub/inner.py": "x = 1\n"}
    )
    idx = ctx.find_module_idx("pkg.sub.inner")
    assert idx is not None
    assert ctx.nodes()[idx].fqname == "pkg.sub.inner"


def test_find_module_idx_missing_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    assert ctx.find_module_idx("nothing.here") is None


def test_module_for_indices_by_path(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/svc.py": "def f(): pass\n"})
    svc_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.svc")
    idx = ctx.module_for_indices(svc_path)
    assert idx is not None
    assert ctx.nodes()[idx].fqname == "pkg.svc"


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
# ctx.find_module_dunder_all_exports_indices / find_module_dunders_indices
# ---------------------------------------------------------------------------


def test_dunder_all_exports_returns_listed_exports(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/m.py": '__all__ = ["f"]\ndef f(): pass\ndef g(): pass\n',
        }
    )
    idxs = ctx.find_module_dunder_all_exports_indices("pkg.m")
    assert idxs is not None
    fqnames = {ctx.nodes()[i].fqname for i in idxs}
    assert fqnames == {"pkg.m.f"}


def test_dunder_all_exports_none_when_unset(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/m.py": "def f(): pass\n"})
    assert ctx.find_module_dunder_all_exports_indices("pkg.m") is None


def test_module_dunders_indices_project_wide(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "__all__ = []\n__version__ = '1.0'\n",
            "pkg/a.py": "def __getattr__(name): pass\nx = 1\n",
        }
    )
    idxs = ctx.find_module_dunders_indices()
    fqnames = {ctx.nodes()[i].fqname for i in idxs}
    assert "pkg.__all__" in fqnames
    assert "pkg.__version__" in fqnames
    assert "pkg.a.__getattr__" in fqnames
    assert "pkg.a.x" not in fqnames


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
# ctx.find_declarations_indices / ctx.resolve_idx
# ---------------------------------------------------------------------------


def test_find_declarations_indices_walks_back(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "class Cls:\n    def method(self): pass\n",
        }
    )
    # Walk-back: pkg.lib.Cls.method → pkg.lib.Cls.
    indices = ctx.find_declarations_indices("pkg.lib.Cls.method")
    assert len(indices) == 1
    assert ctx.nodes()[indices[0]].fqname == "pkg.lib.Cls"


def test_resolve_idx_falls_back_to_module(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/lib.py": "def f(): pass\n"})
    # find_declarations excludes modules — resolve_idx includes them.
    assert ctx.find_declarations_indices("pkg.lib") == []
    resolved = ctx.resolve_idx("pkg.lib")
    assert resolved is not None
    assert ctx.nodes()[resolved].kind == "module"


def test_resolve_idx_unknown_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    assert ctx.resolve_idx("nowhere.such.name") is None


# ---------------------------------------------------------------------------
# ctx.{descendants,ancestors,direct_predecessors} idx-form traversal
# ---------------------------------------------------------------------------


def test_descendants_indices_walks_forward_closure(build_decl_graph):
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
    main_idx = ctx.find_module_idx("pkg.main")
    assert main_idx is not None
    fqnames = {n.fqname for n in ctx.nodes_at(ctx.descendants_indices(main_idx))}
    # module → caller (top-level call) → used.
    assert {"pkg.main.caller", "pkg.main.used"} <= fqnames


def test_ancestors_indices_walks_back(build_decl_graph):
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
    (used_idx,) = ctx.find_declarations_indices("pkg.main.used")
    fqnames = {n.fqname for n in ctx.nodes_at(ctx.ancestors_indices(used_idx))}
    # caller reaches used.
    assert "pkg.main.caller" in fqnames


def test_direct_predecessors_idx_immediate_only(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": "def f(): pass\ndef g(): f()\ndef h(): f()\n",
        }
    )
    (f_idx,) = ctx.find_declarations_indices("pkg.main.f")
    fqnames = {n.fqname for n in ctx.nodes_at(ctx.direct_predecessors_idx(f_idx))}
    assert {"pkg.main.g", "pkg.main.h"} <= fqnames


def test_descendants_indices_skip_flags(build_decl_graph):
    """``skip_flags`` plumbs through to the underlying BFS — the
    dead-branch-skipping closure is a subset of the full closure."""
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
    mod_idx = ctx.find_module_idx("pkg.m")
    assert mod_idx is not None
    full = ctx.descendants_indices(mod_idx)
    strict = ctx.descendants_indices(mod_idx, skip_flags=native.EdgeFlags.DEAD_BRANCH)
    assert set(strict).issubset(set(full))


def test_descendants_indices_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.descendants_indices(n)


def test_ancestors_indices_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.ancestors_indices(n)


def test_direct_predecessors_idx_out_of_range_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass\n"})
    n = len(ctx.nodes())
    with pytest.raises(IndexError, match="out of range"):
        ctx.direct_predecessors_idx(n)


# ---------------------------------------------------------------------------
# ctx.reachable_indices
# ---------------------------------------------------------------------------


def test_reachable_indices_returns_valid_subset(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/main.py": "def alive(): pass\ndef dead(): pass\nalive()\n",
        }
    )
    n = len(ctx.nodes())
    reachable = ctx.reachable_indices()
    assert all(0 <= i < n for i in reachable)
    # Indices are revivable and unique.
    assert len(set(reachable)) == len(reachable)
    assert len(ctx.nodes_at(reachable)) == len(reachable)


def test_reachable_indices_seed_flags_kwarg(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/main.py": "def f(): pass\nf()\n"})
    n = len(ctx.nodes())
    reachable = ctx.reachable_indices(
        skip_flags=native.EdgeFlags.DEAD_BRANCH,
        seed_flags=native.NodeFlags.ENTRYPOINT,
    )
    assert all(0 <= i < n for i in reachable)


# ---------------------------------------------------------------------------
# ctx.find_nodes_matching_specs_indices — OR-form entrypoint matcher
# ---------------------------------------------------------------------------


def test_find_nodes_matching_specs_indices_or_form(build_decl_graph, tmp_path):
    """The matcher ORs across the three buckets: a node matches if it
    satisfies any of the regex / str-spec / abs-path sets."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/svc.py": "def handler(): pass\n",
            "pkg/util.py": "def helper(): pass\n",
        }
    )
    util_path = next(n.path for n in ctx.nodes() if n.fqname == "pkg.util")
    indices = ctx.find_nodes_matching_specs_indices(
        str(tmp_path),
        [r"pkg/svc\.py"],
        ["pkg.util.helper"],
        [util_path],
    )
    fqnames = {n.fqname for n in ctx.nodes_at(indices)}
    # Regex bucket pulls pkg.svc + its decl; str-spec bucket pulls
    # pkg.util.helper; abs-path bucket pulls the pkg.util module.
    assert {"pkg.svc.handler", "pkg.util.helper", "pkg.util"} <= fqnames


# ---------------------------------------------------------------------------
# ctx.find_main_blocks_indices
# ---------------------------------------------------------------------------


def test_find_main_blocks_indices(build_decl_graph):
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
    pairs = ctx.find_main_blocks_indices()
    assert len(pairs) == 1
    mod_idx, decl_idxs = pairs[0]
    assert ctx.nodes()[mod_idx].fqname == "pkg.script"
    decl_fqnames = {ctx.nodes()[i].fqname for i in decl_idxs}
    assert "pkg.script.inner_decl" in decl_fqnames


def test_find_main_blocks_indices_no_main_block(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/lib.py": "def f(): pass\n"})
    assert ctx.find_main_blocks_indices() == []


# ---------------------------------------------------------------------------
# ctx.find_literal_list_entries
# ---------------------------------------------------------------------------


def test_find_literal_list_entries(build_decl_graph):
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
    assert ctx.find_literal_list_entries("pkg.m.__all__") == ["a", "b"]


def test_find_literal_list_entries_unknown_returns_none(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": ""})
    assert ctx.find_literal_list_entries("pkg.nope") is None


# ---------------------------------------------------------------------------
# ctx.decls_under_indices / decls_matching_indices / decls_matching_name_indices
# ---------------------------------------------------------------------------


def test_decls_under_indices_path_prefix(build_decl_graph, tmp_path):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/inner.py": "def deep(): pass\n",
            "pkg/top.py": "def shallow(): pass\n",
        }
    )
    prefix = str(tmp_path / "pkg" / "sub")
    fqnames = {n.fqname for n in ctx.nodes_at(ctx.decls_under_indices(prefix))}
    assert "pkg.sub.inner.deep" in fqnames
    assert "pkg.top.shallow" not in fqnames


def test_decls_matching_indices_path_substring(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/test_a.py": "def t_a(): pass\n",
            "pkg/lib.py": "def f(): pass\n",
        }
    )
    fqnames = {n.fqname for n in ctx.nodes_at(ctx.decls_matching_indices("test_a"))}
    assert "pkg.test_a.t_a" in fqnames
    assert "pkg.lib.f" not in fqnames


def test_decls_matching_name_indices_simple_name_regex(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def test_one(): pass\nclass TestThing: pass\ndef helper(): pass\n",
        }
    )
    indices = ctx.decls_matching_name_indices(r"^(test_|Test)")
    fqnames = {n.fqname for n in ctx.nodes_at(indices)}
    assert fqnames == {"pkg.a.test_one", "pkg.a.TestThing"}


def test_decls_matching_name_indices_invalid_regex_raises(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    with pytest.raises(ValueError, match="invalid regex"):
        ctx.decls_matching_name_indices(r"(unclosed")
