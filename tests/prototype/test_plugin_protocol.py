"""Smoke tests for the prototype plugin protocol.

Each test materializes a small project through `ProjectContext`, runs
one or two plugins, and asserts on the post-plugin edge set. The
plugin runs entirely through the rust-side context — these tests are
exercising the four ty-backed queries (`find_module_dunders`,
`find_classes_defining_method`, `find_subclasses_of`,
`find_comment_patterns`) plus the `add_node` / `add_edge` mutation
path and the rust → Python → rust re-entry through `materialize`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("dead_cst_ty_native")

from dead_cst.plugins import InitSubclassPlugin, ModuleDundersPlugin  # noqa: E402

from .plugin_protocol import KeepAliveCommentPlugin, run_plugins  # noqa: E402


@pytest.fixture
def make_ctx(tmp_path: Path):
    """Write `{relpath: source}` files and return a fresh ProjectContext."""

    def make(files: dict[str, str], **kwargs) -> native.ProjectContext:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.ProjectContext(str(tmp_path), **kwargs)

    return make


def _edges(graph: native.NativeGraph) -> set[str]:
    return {f"{graph.nodes[s].fqname} -> {graph.nodes[d].fqname}" for s, d, _ in graph.edges}


def _fqnames(graph: native.NativeGraph) -> set[str]:
    return {n.fqname for n in graph.nodes}


# ---------------------------------------------------------------------------
# Materialize round-trips back to Python and re-enters rust
# ---------------------------------------------------------------------------


def test_materialize_with_no_plugins_matches_project_build(make_ctx, tmp_path):
    """Empty plugin list should produce the same graph as `Project.build`."""
    files = {"mod.py": "def f(): pass\nclass C: pass\n"}
    ctx = make_ctx(files)
    via_ctx = ctx.materialize()

    project = native.Project(str(tmp_path))
    via_project = project.build()

    assert _fqnames(via_ctx) == _fqnames(via_project)
    assert _edges(via_ctx) == _edges(via_project)


def test_plugin_runs_once_per_project(make_ctx):
    """A plugin sees a single ctx; counts add up across files."""
    calls: list[int] = []

    class Counter:
        name = "counter"

        def run(self, ctx: native.ProjectContext) -> None:
            calls.append(len(ctx.find_module_dunders()))

    ctx = make_ctx(
        {
            "a.py": "__version__ = '1'\n__all__ = ['x']\nx = 1\n",
            "b.py": "__author__ = 'me'\ny = 2\n",
        }
    )
    ctx.add_plugin(Counter())
    ctx.materialize()
    assert calls == [3]


# ---------------------------------------------------------------------------
# find_module_dunders + ModuleDundersPlugin
# ---------------------------------------------------------------------------


def test_module_dunders_plugin_marks_each_dunder(make_ctx):
    ctx = make_ctx(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "__version__ = '1'\n__all__ = ['public']\npublic = 1\n_private = 2\n",
        }
    )
    graph = run_plugins(ctx, [ModuleDundersPlugin()])

    edges = _edges(graph)
    assert "<dunder>:pkg.mod.__version__ -> pkg.mod.__version__" in edges
    assert "<dunder>:pkg.mod.__all__ -> pkg.mod.__all__" in edges
    # The non-dunder name must NOT have been marked.
    assert "<dunder>:pkg.mod.public -> pkg.mod.public" not in edges
    assert "<dunder>:pkg.mod._private -> pkg.mod._private" not in edges


def test_module_dunders_skips_function_with_dunder_name(make_ctx):
    """`def __init_subclass__` at module scope is not a variable dunder."""
    ctx = make_ctx({"mod.py": "def __init__(): pass\n__version__ = '1'\n"})
    graph = run_plugins(ctx, [ModuleDundersPlugin()])
    assert "<dunder>:mod.__version__ -> mod.__version__" in _edges(graph)
    assert "<dunder>:mod.__init__ -> mod.__init__" not in _edges(graph)


# ---------------------------------------------------------------------------
# find_classes_defining_method + find_subclasses_of
# ---------------------------------------------------------------------------


def test_init_subclass_plugin_wires_transitive_subclasses(make_ctx):
    ctx = make_ctx(
        {
            "registry.py": (
                "class Base:\n"
                "    def __init_subclass__(cls): pass\n"
                "class Mid(Base): pass\n"
                "class Leaf(Mid): pass\n"
            ),
        }
    )
    graph = run_plugins(ctx, [InitSubclassPlugin()])
    edges = _edges(graph)

    # Marker is wired to the parent, then to each transitive subclass.
    assert "registry.Base -> <__init_subclass__>:registry.Base" in edges
    assert "<__init_subclass__>:registry.Base -> registry.Mid" in edges
    assert "<__init_subclass__>:registry.Base -> registry.Leaf" in edges


def test_init_subclass_plugin_no_marker_when_no_dunder(make_ctx):
    ctx = make_ctx({"mod.py": "class Base: pass\nclass Sub(Base): pass\n"})
    graph = run_plugins(ctx, [InitSubclassPlugin()])
    assert not any(n.fqname.startswith("<__init_subclass__>:") for n in graph.nodes)


def test_find_subclasses_of_returns_empty_for_non_class(make_ctx):
    """Asking for subclasses of a function-kind node yields []."""
    ctx = make_ctx({"mod.py": "def f(): pass\n"})

    captured: list[list] = []

    class Inspect:
        name = "inspect"

        def run(self, ctx: native.ProjectContext) -> None:
            for node in ctx.nodes():
                if node.fqname == "mod.f":
                    captured.append(ctx.find_subclasses_of(node))

    ctx.add_plugin(Inspect())
    ctx.materialize()
    assert captured == [[]]


# ---------------------------------------------------------------------------
# find_comment_patterns + KeepAliveCommentPlugin
# ---------------------------------------------------------------------------


def test_keep_alive_comment_plugin_attaches_to_following_decl(make_ctx):
    ctx = make_ctx(
        {
            "mod.py": (
                "# dead-cst: keep\n"
                "def kept(): pass\n"
                "\n"
                "def normal(): pass\n"
                "\n"
                "# dead-cst: keep\n"
                "class AlsoKept: pass\n"
            ),
        }
    )
    graph = run_plugins(ctx, [KeepAliveCommentPlugin()])
    edges = _edges(graph)
    assert "<keep>:mod.kept -> mod.kept" in edges
    assert "<keep>:mod.AlsoKept -> mod.AlsoKept" in edges
    # `normal` has no preceding directive.
    assert "<keep>:mod.normal -> mod.normal" not in edges


def test_find_comment_patterns_skips_unmatched_comments(make_ctx):
    """A regex miss must yield no edges."""
    ctx = make_ctx(
        {
            "mod.py": "# regular comment\ndef f(): pass\n# dead-cst: keep\ndef g(): pass\n",
        }
    )
    graph = run_plugins(ctx, [KeepAliveCommentPlugin()])
    edges = _edges(graph)
    assert "<keep>:mod.g -> mod.g" in edges
    assert "<keep>:mod.f -> mod.f" not in edges


# ---------------------------------------------------------------------------
# AddNode / AddEdge / re-entry mechanics
# ---------------------------------------------------------------------------


def test_add_node_and_add_edge_round_trip(make_ctx):
    """`AddEdge` resolves node identity by content (no idx field needed)."""

    class MintAndCheck:
        name = "mint_and_check"

        def run(self, ctx: native.ProjectContext):
            decls = [n for n in ctx.nodes() if n.fqname == "mod.f"]
            assert len(decls) == 1
            decl = decls[0]
            yield native.AddNode(fqname="<seed>", path=decl.path, edges_to=[decl])

    ctx = make_ctx({"mod.py": "def f(): pass\n"})
    ctx.add_plugin(MintAndCheck())
    graph = ctx.materialize()
    # The minted edge survived to the snapshot.
    fqnames = [n.fqname for n in graph.nodes]
    synth_idx = fqnames.index("<seed>")
    decl_idx = fqnames.index("mod.f")
    assert (synth_idx, decl_idx, 0) in graph.edges


def test_add_edge_dedups_by_content(make_ctx):
    """Two AddNode ops with the same key alias the same interned node;
    the resulting edges collapse to one."""

    class MintTwice:
        name = "mint_twice"

        def run(self, ctx: native.ProjectContext):
            decl = next(n for n in ctx.nodes() if n.fqname == "mod.f")
            yield native.AddNode(fqname="<seed>", path=decl.path, edges_to=[decl])
            yield native.AddNode(fqname="<seed>", path=decl.path, edges_to=[decl])

    ctx = make_ctx({"mod.py": "def f(): pass\n"})
    ctx.add_plugin(MintTwice())
    graph = ctx.materialize()
    fqnames = [n.fqname for n in graph.nodes]
    # The synthetic was interned exactly once.
    assert fqnames.count("<seed>") == 1
    synth_idx = fqnames.index("<seed>")
    decl_idx = fqnames.index("mod.f")
    seed_to_f = [(s, d, f) for (s, d, f) in graph.edges if s == synth_idx and d == decl_idx]
    assert len(seed_to_f) == 1


def test_query_outside_materialize_raises(make_ctx):
    """Calling a query method on an un-materialized context errors clearly."""
    ctx = make_ctx({"mod.py": "x = 1\n"})
    with pytest.raises(RuntimeError, match="ProjectContext.find_module_dunders"):
        ctx.find_module_dunders()


def test_materialize_is_idempotent(make_ctx):
    """Calling materialize twice rebuilds; the same plugin re-runs once per call."""
    counts: list[int] = []

    class CountAdded:
        name = "count"

        def run(self, ctx: native.ProjectContext):
            yield native.AddNode(fqname="<seed>", path="/")
            counts.append(sum(1 for n in ctx.nodes() if n.fqname == "<seed>"))

    ctx = make_ctx({"mod.py": "x = 1\n"})
    ctx.add_plugin(CountAdded())
    g1 = ctx.materialize()
    g2 = ctx.materialize()
    # The second run gets a fresh outputs (not the accumulated state).
    assert counts == [1, 1]
    assert {n.fqname for n in g1.nodes} == {n.fqname for n in g2.nodes}
