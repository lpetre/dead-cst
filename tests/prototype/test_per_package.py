"""Per-package orchestration tests.

Exercise:
  * ``Project.package_order()`` toposort + cycle detection
  * ``Project.build_package_graph(name)`` walking every .py file
  * the Python-side accumulator with cross-package node dedup
  * the plugin protocol (one demo plugin)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

native = pytest.importorskip("dead_cst_ty_native")

from dead_cst.graph import EdgeFlags, NodeFlags, SymbolNode  # noqa: E402

from ._bridge import PackageContext, accumulate, build_project_graph  # noqa: E402


@pytest.fixture
def make_project(tmp_path: Path):
    """Materialize a multi-package project under tmp_path."""

    def make(
        layout: dict[str, str],
        packages: list[tuple[str, str, list[str]]],
    ) -> tuple[native.Project, Path]:
        for relpath, source in layout.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        specs = [native.PackageSpec(name, path, deps) for name, path, deps in packages]
        proj = native.Project(str(tmp_path), packages=specs)
        return proj, tmp_path

    return make


# ---------------------------------------------------------------------
# Toposort
# ---------------------------------------------------------------------


def test_package_order_respects_deps(tmp_path):
    """Deps are placed before the package that declares them."""
    proj = native.Project(
        str(tmp_path),
        packages=[
            native.PackageSpec("c", "c", ["b"]),
            native.PackageSpec("a", "a", []),
            native.PackageSpec("b", "b", ["a"]),
        ],
    )
    order = proj.package_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_package_order_handles_diamond(tmp_path):
    """A diamond shape produces some valid linearization."""
    proj = native.Project(
        str(tmp_path),
        packages=[
            native.PackageSpec("base", "base", []),
            native.PackageSpec("left", "left", ["base"]),
            native.PackageSpec("right", "right", ["base"]),
            native.PackageSpec("top", "top", ["left", "right"]),
        ],
    )
    order = proj.package_order()
    assert order[0] == "base"
    assert order[-1] == "top"
    assert set(order[1:3]) == {"left", "right"}


def test_unknown_dep_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="unknown dep"):
        native.Project(
            str(tmp_path),
            packages=[native.PackageSpec("a", "a", ["nope"])],
        )


def test_dep_cycle_detected_in_package_order(tmp_path):
    proj = native.Project(
        str(tmp_path),
        packages=[
            native.PackageSpec("a", "a", ["b"]),
            native.PackageSpec("b", "b", ["a"]),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        proj.package_order()


def test_duplicate_package_names_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate package name"):
        native.Project(
            str(tmp_path),
            packages=[
                native.PackageSpec("a", "a", []),
                native.PackageSpec("a", "other", []),
            ],
        )


# ---------------------------------------------------------------------
# Per-package build
# ---------------------------------------------------------------------


def test_build_package_walks_all_py_files(make_project):
    """build_package_graph picks up every .py file under the package."""
    proj, root = make_project(
        layout={
            "pkg/__init__.py": "def init_fn(): pass\n",
            "pkg/mod.py": "def mod_fn(): pass\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/leaf.py": "class Leaf: pass\n",
        },
        packages=[("pkg", "pkg", [])],
    )
    g = proj.build_package_graph("pkg")
    fqnames = sorted(n.fqname for n in g.nodes)
    assert fqnames == [
        "pkg",
        "pkg.init_fn",
        "pkg.mod",
        "pkg.mod.mod_fn",
        "pkg.sub",
        "pkg.sub.leaf",
        "pkg.sub.leaf.Leaf",
    ]


def test_build_package_skips_pycache(make_project):
    proj, root = make_project(
        layout={
            "pkg/__init__.py": "",
            "pkg/__pycache__/cached.cpython-313.pyc": "junk",
            "pkg/__pycache__/cached.cpython-313.py": "def junk(): pass\n",
        },
        packages=[("pkg", "pkg", [])],
    )
    g = proj.build_package_graph("pkg")
    fqnames = {n.fqname for n in g.nodes}
    assert "pkg" in fqnames
    assert not any("junk" in n for n in fqnames)


def test_build_unknown_package_raises(make_project):
    proj, _ = make_project(layout={}, packages=[])
    with pytest.raises(ValueError, match="unknown package"):
        proj.build_package_graph("nope")


# ---------------------------------------------------------------------
# Accumulation across packages
# ---------------------------------------------------------------------


def test_accumulate_returns_index_aligned_symbol_nodes(make_project):
    proj, root = make_project(
        layout={"pkg/mod.py": "def f(): pass\nclass C: pass\n"},
        packages=[("pkg", "pkg", [])],
    )
    g = proj.build_package_graph("pkg")
    from dead_cst._graphstore import SymbolGraph

    graph = SymbolGraph()
    symbol_nodes = accumulate(graph, g)
    assert len(symbol_nodes) == len(g.nodes)
    for native_node, sn in zip(g.nodes, symbol_nodes):
        assert sn.fqname == native_node.fqname
        assert sn.type == native_node.kind


def test_cross_package_node_dedup(make_project):
    """A node shared by two packages collapses on accumulate."""
    proj, root = make_project(
        layout={
            "a/__init__.py": "def shared(): pass\n",
            "b/__init__.py": "",
        },
        packages=[("a", "a", []), ("b", "b", ["a"])],
    )

    from dead_cst._graphstore import SymbolGraph

    pkg_a = proj.build_package_graph("a")

    graph = SymbolGraph()
    sn_a = accumulate(graph, pkg_a)
    nodes_after_a = len(graph)

    # Simulate B referencing A by manually re-introducing one of A's
    # nodes into B's envelope: build_package_graph does NOT do this
    # today (B only has its own files). We exercise the accumulator's
    # dedup behavior directly by passing pkg_a again -- equivalent to
    # B emitting a duplicate of A's node to hang an edge off.
    sn_b = accumulate(graph, pkg_a)
    assert len(graph) == nodes_after_a, "duplicate accumulate added new nodes"
    # Both materializations point at the SAME SymbolNode instances.
    assert sn_a == sn_b


def test_full_project_build_with_two_packages(make_project):
    proj, root = make_project(
        layout={
            "a/__init__.py": "def af(): pass\n",
            "b/__init__.py": "class BC: pass\n",
        },
        packages=[("b", "b", ["a"]), ("a", "a", [])],
    )
    report = build_project_graph(proj)
    assert report.order == ["a", "b"]
    fqnames = {n.fqname for n in report.graph.nodes}
    assert {"a", "a.af", "b", "b.BC"}.issubset(fqnames)


# ---------------------------------------------------------------------
# Plugin protocol
# ---------------------------------------------------------------------


@dataclass(slots=True)
class _MarkClassesEntrypoint:
    """Demo plugin: every class becomes an entrypoint.

    For each class node in the package, add a synthetic
    ``"<entrypoint>:<fqname>"`` node flagged ENTRYPOINT and an edge
    from it to the class. This mirrors the shape real entrypoint
    plugins (ProjectScripts, ClickCommand, ...) use today.
    """

    name: str = "entrypoint-stub"
    version: int = 1
    package_order_seen: list[str] = field(default_factory=list)

    def contribute(self, ctx: PackageContext) -> None:
        self.package_order_seen.append(ctx.package_name)
        for i, native_node in enumerate(ctx.package_graph.nodes):
            if native_node.kind != "class":
                continue
            class_sn = ctx.symbol_nodes[i]
            entrypoint = SymbolNode(
                fqname=f"<entrypoint>:{class_sn.fqname}",
                type="synthetic",
                path=class_sn.path,
                position=class_sn.position,
                flags=NodeFlags.ENTRYPOINT,
            )
            ctx.add_edge(entrypoint, class_sn, EdgeFlags.NONE)


def test_plugin_runs_once_per_package_in_order(make_project):
    proj, root = make_project(
        layout={
            "a/__init__.py": "class A: pass\n",
            "b/__init__.py": "class B: pass\n",
        },
        packages=[("b", "b", ["a"]), ("a", "a", [])],
    )
    plugin = _MarkClassesEntrypoint()
    build_project_graph(proj, plugins=[plugin])
    assert plugin.package_order_seen == ["a", "b"]


def test_plugin_adds_synthetic_entrypoints(make_project):
    proj, root = make_project(
        layout={
            "pkg/__init__.py": "class Foo: pass\nclass Bar: pass\n",
        },
        packages=[("pkg", "pkg", [])],
    )
    report = build_project_graph(proj, plugins=[_MarkClassesEntrypoint()])
    by_fqname = {n.fqname: n for n in report.graph.nodes}
    assert "<entrypoint>:pkg.Foo" in by_fqname
    assert "<entrypoint>:pkg.Bar" in by_fqname
    assert by_fqname["<entrypoint>:pkg.Foo"].flags == NodeFlags.ENTRYPOINT
    # Each synthetic entrypoint points at its class.
    entry = by_fqname["<entrypoint>:pkg.Foo"]
    cls = by_fqname["pkg.Foo"]
    successors = list(report.graph.raw.successor_indices(report.graph.index(entry)))
    assert report.graph.index(cls) in successors


def test_plugins_compose(make_project):
    """Two plugins both contribute; both end up in the graph."""

    @dataclass(slots=True)
    class TagModuleAsExported:
        name: str = "tag-exported"
        version: int = 1

        def contribute(self, ctx: PackageContext) -> None:
            module_sn = next(
                ctx.symbol_nodes[i]
                for i, n in enumerate(ctx.package_graph.nodes)
                if n.kind == "module"
            )
            marker = SymbolNode(
                fqname=f"<exported>:{module_sn.fqname}",
                type="synthetic",
                path=module_sn.path,
                position=module_sn.position,
                flags=NodeFlags.EXPORTED,
            )
            ctx.add_edge(marker, module_sn, EdgeFlags.NONE)

    proj, root = make_project(
        layout={"pkg/__init__.py": "class C: pass\n"},
        packages=[("pkg", "pkg", [])],
    )
    report = build_project_graph(
        proj,
        plugins=[_MarkClassesEntrypoint(), TagModuleAsExported()],
    )
    fqnames = {n.fqname for n in report.graph.nodes}
    assert "<entrypoint>:pkg.C" in fqnames
    assert "<exported>:pkg" in fqnames
