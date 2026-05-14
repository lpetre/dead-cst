"""Tests for the plugin surface itself: empty seeds, registry lookup,
composition across multiple plugins, plus unit tests for the small
AST helpers in :mod:`dead_cst.plugins._core`."""

from __future__ import annotations

from pathlib import Path

import libcst as cst
import networkx as nx
import pytest
from libcst.metadata import CodePosition, CodeRange

from dead_cst.analyze import _find_reachable as find_reachable
from dead_cst.plugins import MainBlockPlugin, ProjectScriptsPlugin
from dead_cst.plugins._core import (
    AddEdge,
    AddNode,
    PluginContext,
    RemoveEdge,
    apply_ops,
    collect_module_imports,
    decorator_owner,
    find_call_assignments,
    find_handlers,
    is_from_module,
    is_name,
    mark_entrypoints,
    matched_attr_call,
    single_target_assignment,
)
from dead_cst.graph import EdgeFlags, NodeFlags, SymbolNode, SymbolTrie
from dead_cst.resolvers import Package


def test_no_plugins_means_nothing_reachable(make_analysis, write_files):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    graph = make_analysis().materialize_all()
    assert find_reachable(graph) == set()


def test_plugins_compose(make_analysis, write_files, reachable_fqnames):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/cli.py": "def main(): pass",
            "pkg/runner.py": """
            import pkg.cli
            if __name__ == "__main__":
                pkg.cli.main()
            """,
            "pyproject.toml": """
            [project]
            name = "x"
            [project.scripts]
            mytool = "pkg.cli:main"
            """,
        }
    )
    graph = make_analysis(plugins=[MainBlockPlugin(), ProjectScriptsPlugin()]).materialize_all()
    assert {"pkg.runner", "pkg.cli", "pkg.cli.main"} <= reachable_fqnames(graph)


def test_unknown_plugin_raises():
    from dead_cst.plugins import load_plugin

    with pytest.raises(KeyError):
        load_plugin("does-not-exist")


# ---------------------------------------------------------------------------
# require_resolved_dep
# ---------------------------------------------------------------------------


def _ctx_with_synthetic(fqname: str, base: Path, make_contribution) -> PluginContext:
    """Build a minimal PluginContext containing a single synthetic node."""
    graph = nx.DiGraph()
    node = SymbolNode(
        fqname=fqname,
        type="synthetic",
        path=base,
        position=CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0)),
    )
    graph.add_node(node)
    return PluginContext(
        graph=graph,
        symbol_lookup=SymbolTrie(),
        contribution=make_contribution(Package(path=base, name="pkg"), frozenset({node})),
        project_root=base,
    )


def test_require_resolved_dep_returns_external_dist(tmp_path, make_contribution):
    from dead_cst.plugins._core import EXTERNAL_DIST_PREFIX, require_resolved_dep

    ctx = _ctx_with_synthetic(f"{EXTERNAL_DIST_PREFIX}fastapi", tmp_path, make_contribution)
    node = require_resolved_dep(ctx, "fastapi")
    assert node is not None
    assert node.fqname == f"{EXTERNAL_DIST_PREFIX}fastapi"


def test_require_resolved_dep_returns_external_file(tmp_path, make_contribution):
    from dead_cst.plugins._core import EXTERNAL_FILE_PREFIX, require_resolved_dep

    ctx = _ctx_with_synthetic(f"{EXTERNAL_FILE_PREFIX}fastapi", tmp_path, make_contribution)
    assert require_resolved_dep(ctx, "fastapi") is not None


def test_require_resolved_dep_returns_none_if_not_imported(tmp_path, make_contribution):
    from dead_cst.plugins._core import EXTERNAL_DIST_PREFIX, require_resolved_dep

    ctx = _ctx_with_synthetic(f"{EXTERNAL_DIST_PREFIX}something_else", tmp_path, make_contribution)
    assert require_resolved_dep(ctx, "fastapi") is None


def test_require_resolved_dep_raises_on_unresolved(tmp_path, make_contribution):
    from dead_cst.plugins._core import (
        UNRESOLVED_PREFIX,
        UnresolvedDependencyError,
        require_resolved_dep,
    )

    ctx = _ctx_with_synthetic(f"{UNRESOLVED_PREFIX}fastapi", tmp_path, make_contribution)
    with pytest.raises(UnresolvedDependencyError, match="uv sync"):
        require_resolved_dep(ctx, "fastapi")


# ---------------------------------------------------------------------------
# AST helpers in _plugins._core
# ---------------------------------------------------------------------------


def _parse(src: str) -> cst.Module:
    import textwrap

    return cst.parse_module(textwrap.dedent(src).strip() + "\n")


def _pos() -> CodeRange:
    return CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0))


@pytest.mark.parametrize(
    "expr, value, expected",
    [
        ("foo", "foo", True),
        ("foo", "bar", False),
        ("foo.bar", "foo", False),  # Attribute, not Name
        ("foo()", "foo", False),  # Call, not Name
    ],
)
def test_is_name(expr, value, expected):
    node = cst.parse_expression(expr)
    assert is_name(node, value) is expected


def test_is_name_handles_none():
    assert is_name(None, "anything") is False


@pytest.mark.parametrize(
    "src, module_name, expected",
    [
        ("from flask import Flask", "flask", True),
        ("from flask import Flask", "fastapi", False),
        ("from .pkg import x", "pkg", False),  # relative
        ("from a.b import c", "a", False),  # dotted module name -- prefix doesn't match
        ("from a.b import c", "a.b", True),  # dotted module name -- exact match
        ("from a.b.c import d", "a.b.c", True),  # deeper dotted module name
    ],
)
def test_is_from_module(src, module_name, expected):
    module = _parse(src)
    stmt = module.body[0].body[0]
    assert isinstance(stmt, cst.ImportFrom)
    assert is_from_module(stmt, module_name) is expected


@pytest.mark.parametrize(
    "src, expected_name, expected_rhs",
    [
        ("x = 1", "x", "1"),
        ("x: int = 1", "x", "1"),
        ("x, y = 1, 2", None, None),  # tuple target
        ("x = y = 1", None, None),  # chained assign -> 2 targets
        ("x: int", None, None),  # AnnAssign without value
        ("foo.bar = 1", None, None),  # not a bare Name target
    ],
)
def test_single_target_assignment(src, expected_name, expected_rhs):
    module = _parse(src)
    stmt = module.body[0].body[0]
    name, rhs = single_target_assignment(stmt)
    assert name == expected_name
    if expected_rhs is None:
        assert rhs is None
    else:
        assert rhs is not None
        assert cst.Module([]).code_for_node(rhs).strip() == expected_rhs


def test_single_target_assignment_rejects_non_assign():
    module = _parse("def f(): pass")
    stmt = module.body[0]
    assert single_target_assignment(stmt) == (None, None)


@pytest.mark.parametrize(
    "expr, valid, expected",
    [
        ("app.route", {"route"}, "app"),
        ("app.route(...)", {"route"}, "app"),  # Call unwrapped
        ("app.route", {"get"}, None),  # attr not in valid
        ("app.route.sub", {"route"}, None),  # owner not bare Name
        ("route", {"route"}, None),  # bare Name, no owner
    ],
)
def test_decorator_owner(expr, valid, expected):
    node = cst.parse_expression(expr)
    assert decorator_owner(node, valid) == expected


def test_find_handlers_collects_per_instance():
    module = _parse(
        """
        @app.route("/a")
        def a(): pass

        @app.get("/b")
        def b(): pass

        @other.route("/c")
        def c(): pass

        @app.unknown("/d")
        def d(): pass
        """
    )
    handlers = find_handlers(module, {"app", "other"}, {"route", "get"})
    assert handlers == {"app": ["a", "b"], "other": ["c"]}


def test_find_handlers_skips_non_top_level_functions():
    module = _parse(
        """
        class C:
            @app.route("/a")
            def a(self): pass
        """
    )
    assert find_handlers(module, {"app"}, {"route"}) == {}


def test_find_handlers_one_decorator_per_function_max():
    """If a function has multiple matching decorators, count it once per first match."""
    module = _parse(
        """
        @app.route("/a")
        @app.get("/a")
        def a(): pass
        """
    )
    handlers = find_handlers(module, {"app"}, {"route", "get"})
    assert handlers == {"app": ["a"]}


def test_collect_module_imports_from_form():
    module = _parse(
        """
        from flask import Flask, Blueprint as BP, Other
        """
    )
    bindings = collect_module_imports(module, "flask", {"Flask", "Blueprint"})
    assert bindings == {"Flask": "Flask", "BP": "Blueprint"}


def test_collect_module_imports_module_form():
    module = _parse("import flask as fl")
    bindings = collect_module_imports(module, "flask", {"Flask"})
    assert bindings == {"fl": "<module>"}


def test_collect_module_imports_skips_other_modules_and_star():
    module = _parse(
        """
        from other import Flask
        from flask import *
        """
    )
    assert collect_module_imports(module, "flask", {"Flask"}) == {}


def test_mark_entrypoints_emits_synthetic_node_and_edges(tmp_path):
    target = SymbolNode("pkg.f", "function", tmp_path / "f.py", _pos())
    ops = list(mark_entrypoints("seed", tmp_path / "p.toml", [target]))
    assert len(ops) == 2
    assert isinstance(ops[0], AddNode)
    assert ops[0].node.flags & NodeFlags.ENTRYPOINT
    assert ops[0].node.fqname == "seed"
    assert isinstance(ops[1], AddEdge)
    assert ops[1].src is ops[0].node
    assert ops[1].dst is target


def test_mark_entrypoints_empty_targets_yields_nothing():
    assert list(mark_entrypoints("seed", Path("/x"), [])) == []


@pytest.mark.parametrize(
    "expr, imports, valid, expected",
    [
        ("Flask()", {"Flask": "Flask"}, {"Flask", "Blueprint"}, "Flask"),
        ("BP()", {"BP": "Blueprint"}, {"Flask", "Blueprint"}, "Blueprint"),
        ("flask.Flask()", {"flask": "<module>"}, {"Flask", "Blueprint"}, "Flask"),
        ("Flask()", {"Flask": "Other"}, {"Flask"}, None),  # binding maps elsewhere
        ("flask.Flask()", {"flask": "<module>"}, {"Blueprint"}, None),  # attr not valid
        ("flask.sub.Flask()", {"flask": "<module>"}, {"Flask"}, None),  # nested attr
        ("Other()", {"Flask": "Flask"}, {"Flask"}, None),  # unknown name
    ],
)
def test_matched_attr_call_unwraps_call(expr, imports, valid, expected):
    node = cst.parse_expression(expr)
    assert matched_attr_call(node, imports, valid) == expected


def test_matched_attr_call_no_unwrap():
    """With ``unwrap_call=False`` a Call expression itself does not match -- callers
    must hand in the func directly."""
    call = cst.parse_expression("Flask()")
    assert matched_attr_call(call, {"Flask": "Flask"}, {"Flask"}, unwrap_call=False) is None
    assert matched_attr_call(call.func, {"Flask": "Flask"}, {"Flask"}, unwrap_call=False) == "Flask"


def test_find_call_assignments():
    module = _parse(
        """
        app = Flask(__name__)
        bp: Blueprint = Blueprint("bp", __name__)
        app2 = flask.Flask()
        not_a_target = SomethingElse()
        x, y = Flask(), Flask()  # multi-target -- ignored
        """
    )
    imports = {"Flask": "Flask", "Blueprint": "Blueprint", "flask": "<module>"}
    assignments = find_call_assignments(module, imports, {"Flask", "Blueprint"})
    assert assignments == {"app": "Flask", "bp": "Blueprint", "app2": "Flask"}


def test_find_call_assignments_ignores_non_call_rhs():
    module = _parse(
        """
        x = Flask  # bare reference, not a call
        y = 1
        """
    )
    assert find_call_assignments(module, {"Flask": "Flask"}, {"Flask"}) == {}


def test_apply_ops_dedupes_add_edge_against_emitted_set(tmp_path):
    """``apply_ops`` keys plugin ``AddEdge`` against
    ``(src, dst, EdgeFlags.NONE)`` -- the second emission for the same
    pair becomes a no-op when the dedup set already has the key, so
    ``MultiDiGraph`` doesn't accumulate parallel edges from cross-
    source duplicates.
    """
    graph = nx.MultiDiGraph()
    src = SymbolNode("pkg.a", "function", tmp_path / "a.py", _pos())
    dst = SymbolNode("pkg.b", "function", tmp_path / "b.py", _pos())
    graph.add_nodes_from([src, dst])
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = set()

    apply_ops(graph, [AddEdge(src, dst), AddEdge(src, dst)], emitted)

    assert graph.number_of_edges(src, dst) == 1
    assert (src, dst, EdgeFlags.NONE) in emitted


def test_apply_ops_dedup_respects_prior_contribution_edge(tmp_path):
    """A visitor / import edge already in the emitted set blocks a
    later plugin ``AddEdge`` for the same pair. The dedup set is the
    single source of truth across all three edge sources composed in
    one pass.
    """
    graph = nx.MultiDiGraph()
    src = SymbolNode("pkg.a", "function", tmp_path / "a.py", _pos())
    dst = SymbolNode("pkg.b", "function", tmp_path / "b.py", _pos())
    graph.add_nodes_from([src, dst])
    graph.add_edge(src, dst, flags=EdgeFlags.NONE)
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = {(src, dst, EdgeFlags.NONE)}

    apply_ops(graph, [AddEdge(src, dst)], emitted)

    assert graph.number_of_edges(src, dst) == 1


def test_apply_ops_remove_edge_drops_dedup_key(tmp_path):
    """``RemoveEdge`` clears the matching ``(src, dst, NONE)`` entry
    so a subsequent ``AddEdge`` for the same pair is re-added rather
    than silently swallowed.
    """
    graph = nx.MultiDiGraph()
    src = SymbolNode("pkg.a", "function", tmp_path / "a.py", _pos())
    dst = SymbolNode("pkg.b", "function", tmp_path / "b.py", _pos())
    graph.add_nodes_from([src, dst])
    graph.add_edge(src, dst)
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = {(src, dst, EdgeFlags.NONE)}

    apply_ops(graph, [RemoveEdge(src, dst), AddEdge(src, dst)], emitted)

    assert graph.number_of_edges(src, dst) == 1
    assert (src, dst, EdgeFlags.NONE) in emitted


def test_materialize_dedupes_edges_across_visitor_and_plugin(make_analysis, write_files, tmp_path):
    """Plugin ``AddEdge`` re-asserting an edge the visitor already
    contributed does not produce a parallel multi-edge -- the compose-
    pass dedup set spans all three edge sources.
    """
    from dead_cst.plugins._core import ObserveContext

    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": """
                from pkg.b import target
                def caller():
                    target()
            """,
            "pkg/b.py": "def target(): pass",
        }
    )

    class _ReAssertImportPlugin:
        name = "re_assert_import"
        version = 1

        def observe(self, ctx: ObserveContext) -> None:
            return None

        def finalize(self, ctx: PluginContext):
            # Re-assert the cross-file edge ``pkg.a.caller -> pkg.b.target``
            # that the visitor + import resolution already contributed.
            # Without compose-time dedup this produces a parallel edge.
            caller = next((n for n in ctx.graph.nodes if n.fqname == "pkg.a.caller"), None)
            target = next((n for n in ctx.graph.nodes if n.fqname == "pkg.b.target"), None)
            if caller is None or target is None:
                return ()
            return [AddEdge(caller, target)]

    graph = make_analysis(plugins=[_ReAssertImportPlugin()]).materialize_all()
    caller = next(n for n in graph.nodes if n.fqname == "pkg.a.caller")
    target = next(n for n in graph.nodes if n.fqname == "pkg.b.target")
    assert graph.number_of_edges(caller, target) == 1


def test_apply_ops_dedup_keeps_parallel_edges_with_distinct_flags(tmp_path):
    """Plugin edges key as ``NONE``; a pre-existing flagged edge for
    the same pair (e.g. ``DEAD_BRANCH`` from the visitor) does not
    block the plugin emission because the keys differ. MultiDiGraph
    keeps both parallel edges -- the flag is what distinguishes them
    downstream.
    """
    graph = nx.MultiDiGraph()
    src = SymbolNode("pkg.a", "function", tmp_path / "a.py", _pos())
    dst = SymbolNode("pkg.b", "function", tmp_path / "b.py", _pos())
    graph.add_nodes_from([src, dst])
    graph.add_edge(src, dst, flags=EdgeFlags.DEAD_BRANCH)
    emitted: set[tuple[SymbolNode, SymbolNode, EdgeFlags]] = {(src, dst, EdgeFlags.DEAD_BRANCH)}

    apply_ops(graph, [AddEdge(src, dst)], emitted)

    assert graph.number_of_edges(src, dst) == 2


def test_plugin_context_contribution_exposes_nodes(tmp_path, make_contribution):
    """``contribution.nodes`` is the immutable frozenset passed in."""
    pkg = Package(path=tmp_path, name="pkg")
    mod = SymbolNode("pkg", "module", tmp_path / "__init__.py", _pos())
    fn = SymbolNode("pkg.f", "function", tmp_path / "a.py", _pos())
    nodes = frozenset({mod, fn})

    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        contribution=make_contribution(pkg, nodes),
        project_root=tmp_path,
    )
    assert ctx.contribution.nodes == nodes
    assert ctx.contribution.package is pkg
