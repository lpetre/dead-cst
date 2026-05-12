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
from dead_cst.graph import SymbolNode, SymbolTrie
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


def _ctx_with_synthetic(fqname: str, base: Path) -> PluginContext:
    """Build a minimal PluginContext containing a single synthetic node."""
    graph = nx.DiGraph()
    node = SymbolNode(
        fqname=fqname,
        type="synthetic",
        path=base,
        position=CodeRange(start=CodePosition(0, 0), end=CodePosition(0, 0)),
    )
    graph.add_node(node)
    package_graph = nx.MultiDiGraph()
    package_graph.add_node(node)
    return PluginContext(
        graph=graph,
        symbol_lookup=SymbolTrie(),
        package=Package(path=base, name="pkg"),
        project_root=base,
        package_graph=package_graph,
    )


def test_require_resolved_dep_returns_external_dist(tmp_path):
    from dead_cst.plugins._core import EXTERNAL_DIST_PREFIX, require_resolved_dep

    ctx = _ctx_with_synthetic(f"{EXTERNAL_DIST_PREFIX}fastapi", tmp_path)
    node = require_resolved_dep(ctx, "fastapi")
    assert node is not None
    assert node.fqname == f"{EXTERNAL_DIST_PREFIX}fastapi"


def test_require_resolved_dep_returns_external_file(tmp_path):
    from dead_cst.plugins._core import EXTERNAL_FILE_PREFIX, require_resolved_dep

    ctx = _ctx_with_synthetic(f"{EXTERNAL_FILE_PREFIX}fastapi", tmp_path)
    assert require_resolved_dep(ctx, "fastapi") is not None


def test_require_resolved_dep_returns_none_if_not_imported(tmp_path):
    from dead_cst.plugins._core import EXTERNAL_DIST_PREFIX, require_resolved_dep

    ctx = _ctx_with_synthetic(f"{EXTERNAL_DIST_PREFIX}something_else", tmp_path)
    assert require_resolved_dep(ctx, "fastapi") is None


def test_require_resolved_dep_raises_on_unresolved(tmp_path):
    from dead_cst.plugins._core import (
        UNRESOLVED_PREFIX,
        UnresolvedDependencyError,
        require_resolved_dep,
    )

    ctx = _ctx_with_synthetic(f"{UNRESOLVED_PREFIX}fastapi", tmp_path)
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
    assert ops[0].entrypoint is True
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


# ---------------------------------------------------------------------------
# PluginContext: package_modules() caches its first scan
# ---------------------------------------------------------------------------


def test_plugin_context_package_modules_yields_from_package_graph(tmp_path):
    """``package_modules`` sources from ``package_graph`` and snapshots once."""
    pkg = Package(path=tmp_path, name="pkg")
    inside = SymbolNode("pkg.a", "module", tmp_path / "a.py", _pos())
    fn = SymbolNode("pkg.a.f", "function", tmp_path / "a.py", _pos())
    package_graph = nx.MultiDiGraph()
    package_graph.add_node(inside)
    package_graph.add_node(fn)

    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        package=pkg,
        project_root=tmp_path,
        package_graph=package_graph,
    )
    first = list(ctx.package_modules())
    assert first == [(inside.path, inside)]

    # Nodes added after the first call must not appear -- the cache is
    # intentional, so plugins see a stable snapshot.
    late = SymbolNode("pkg.late", "module", tmp_path / "late.py", _pos())
    package_graph.add_node(late)
    assert list(ctx.package_modules()) == first


def test_plugin_context_package_nodes_yields_from_package_graph(tmp_path):
    """``package_nodes`` yields the whole ``package_graph`` node set and snapshots once."""
    pkg = Package(path=tmp_path, name="pkg")
    mod = SymbolNode("pkg", "module", tmp_path / "__init__.py", _pos())
    fn = SymbolNode("pkg.f", "function", tmp_path / "a.py", _pos())
    package_graph = nx.MultiDiGraph()
    package_graph.add_node(mod)
    package_graph.add_node(fn)

    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        package=pkg,
        project_root=tmp_path,
        package_graph=package_graph,
    )
    first = sorted(ctx.package_nodes(), key=lambda n: n.fqname)
    assert first == [mod, fn]

    # Cached: nodes added after the first scan don't appear.
    late = SymbolNode("pkg.late", "function", tmp_path / "late.py", _pos())
    package_graph.add_node(late)
    assert sorted(ctx.package_nodes(), key=lambda n: n.fqname) == first
