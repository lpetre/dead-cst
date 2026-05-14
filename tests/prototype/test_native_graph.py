"""Round-trip tests for ``NativeGraph -> SymbolGraph``.

Validates that a single file's worth of graph contribution -- built
entirely on the Rust side and crossing the FFI boundary as one
envelope -- materializes correctly into a real
``dead_cst._graphstore.SymbolGraph``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("dead_cst_ty_native")

from dead_cst.graph import EdgeFlags, NodeFlags, SymbolNode  # noqa: E402

from ._bridge import materialize  # noqa: E402


@pytest.fixture
def project_factory(tmp_path: Path):
    def make(files: dict[str, str], **kwargs) -> tuple[native.Project, Path]:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.Project(str(tmp_path), **kwargs), tmp_path

    return make


def test_envelope_shape(project_factory):
    """Rust returns the right envelope shape; nodes carry pre-composed fqnames."""
    proj, root = project_factory({"mod.py": "def foo(): pass\nclass Bar: pass\n"})
    g = proj.build_file_graph(str(root / "mod.py"), "smoke.mod")

    assert g.file_path.endswith("mod.py")
    assert g.module_fqname == "smoke.mod"
    # module node + 2 decls
    assert len(g.nodes) == 3
    assert g.nodes[0].kind == "module"
    assert g.nodes[0].fqname == "smoke.mod"
    assert g.nodes[1].kind == "function"
    assert g.nodes[1].fqname == "smoke.mod.foo"
    assert g.nodes[2].kind == "class"
    assert g.nodes[2].fqname == "smoke.mod.Bar"
    # Each decl emits one decl -> module edge.
    assert g.edges == [(1, 0, 0), (2, 0, 0)]


def test_materialize_round_trip(project_factory):
    """NativeGraph -> SymbolGraph builds the correct nodes + edges."""
    proj, root = project_factory({"pkg/mod.py": "def alpha(): pass\nclass Beta: pass\nGAMMA = 1\n"})
    native_graph = proj.build_file_graph(str(root / "pkg/mod.py"), "pkg.mod")

    graph = materialize(native_graph)

    # 4 nodes: module + alpha + Beta + GAMMA
    assert len(graph) == 4

    fqnames = {n.fqname for n in graph.nodes}
    assert fqnames == {"pkg.mod", "pkg.mod.alpha", "pkg.mod.Beta", "pkg.mod.GAMMA"}

    by_fqname = {n.fqname: n for n in graph.nodes}
    assert by_fqname["pkg.mod"].type == "module"
    assert by_fqname["pkg.mod.alpha"].type == "function"
    assert by_fqname["pkg.mod.Beta"].type == "class"
    assert by_fqname["pkg.mod.GAMMA"].type == "variable"

    # Every node points at the module.
    module_idx = graph.index(by_fqname["pkg.mod"])
    decl_nodes = [n for n in graph.nodes if n.type != "module"]
    for decl in decl_nodes:
        successors = graph.raw.successor_indices(graph.index(decl))
        assert module_idx in list(successors), f"{decl.fqname} missing module edge"


def test_node_positions_preserved(project_factory):
    """Positions on the materialized SymbolNodes match what Rust emitted."""
    proj, root = project_factory({"mod.py": "def f():\n    pass\n"})
    native_graph = proj.build_file_graph(str(root / "mod.py"), "mod")
    graph = materialize(native_graph)

    by_fqname = {n.fqname: n for n in graph.nodes}
    f_node = by_fqname["mod.f"]
    assert (f_node.position.start.line, f_node.position.start.column) == (1, 0)
    assert (f_node.position.end.line, f_node.position.end.column) == (2, 8)


def test_node_paths_preserved(project_factory):
    """Materialized SymbolNodes carry the file path the Rust side reported."""
    proj, root = project_factory({"a/b/c.py": "def x(): pass\n"})
    file_path = root / "a/b/c.py"
    native_graph = proj.build_file_graph(str(file_path), "a.b.c")
    graph = materialize(native_graph)

    for n in graph.nodes:
        assert n.path == file_path


def test_default_flags_are_none(project_factory):
    """The prototype emits flags=0; bridge maps to NodeFlags.NONE."""
    proj, root = project_factory({"mod.py": "X = 1\n"})
    native_graph = proj.build_file_graph(str(root / "mod.py"), "mod")
    graph = materialize(native_graph)
    for n in graph.nodes:
        assert n.flags == NodeFlags.NONE


def test_edge_flags_are_none(project_factory):
    """Decl -> module edges land as EdgeFlags.NONE."""
    proj, root = project_factory({"mod.py": "def f(): pass\n"})
    native_graph = proj.build_file_graph(str(root / "mod.py"), "mod")
    graph = materialize(native_graph)
    f = next(n for n in graph.nodes if n.fqname == "mod.f")
    m = next(n for n in graph.nodes if n.fqname == "mod")
    edge_flags = graph.raw.get_all_edge_data(graph.index(f), graph.index(m))
    assert edge_flags == [EdgeFlags.NONE]


def test_empty_module_has_only_module_node(project_factory):
    """A file with no top-level decls still gets one module node, no edges."""
    proj, root = project_factory({"empty.py": ""})
    native_graph = proj.build_file_graph(str(root / "empty.py"), "empty")
    graph = materialize(native_graph)
    assert len(graph) == 1
    assert next(iter(graph.nodes)).fqname == "empty"
    assert len(native_graph.edges) == 0


def test_materialize_is_idempotent(project_factory):
    """Calling materialize twice produces equivalent graphs (no Salsa side effects)."""
    proj, root = project_factory({"mod.py": "def f(): pass\nclass C: pass\n"})
    native_graph = proj.build_file_graph(str(root / "mod.py"), "mod")
    g1 = materialize(native_graph)
    g2 = materialize(native_graph)
    assert {n.fqname for n in g1.nodes} == {n.fqname for n in g2.nodes}
    assert g1.raw.num_edges() == g2.raw.num_edges()


def _node_identity(n: native.NativeNode) -> tuple:
    """The tuple the Rust builder dedupes nodes by."""
    return (
        n.fqname,
        n.kind,
        n.path,
        n.start_line,
        n.start_column,
        n.end_line,
        n.end_column,
        n.flags,
    )


def test_nodes_are_unique(project_factory):
    """The Rust builder guarantees unique node identities."""
    proj, root = project_factory(
        {
            "mod.py": (
                "def alpha(): pass\nclass Beta: pass\nGAMMA = 1\nDELTA: int = 2\nE, F = 3, 4\n"
            )
        }
    )
    g = proj.build_file_graph(str(root / "mod.py"), "mod")

    identities = [_node_identity(n) for n in g.nodes]
    assert len(identities) == len(set(identities)), "duplicate node identities"


def test_edges_are_unique(project_factory):
    """The Rust builder guarantees unique edge triples."""
    proj, root = project_factory({"mod.py": "def a(): pass\nclass B: pass\nC = 1\nD, E = 2, 3\n"})
    g = proj.build_file_graph(str(root / "mod.py"), "mod")
    assert len(g.edges) == len(set(g.edges)), "duplicate edge triples"


def test_node_path_field_is_absolute(project_factory):
    """Each NativeNode carries the file's absolute path."""
    proj, root = project_factory({"x/y.py": "def f(): pass\n"})
    file_path = root / "x/y.py"
    g = proj.build_file_graph(str(file_path), "x.y")
    for n in g.nodes:
        assert n.path == str(file_path)
