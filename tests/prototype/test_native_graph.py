"""End-to-end tests for `Project.build()` in the ty-backed native crate.

The crate is governed by the three principles in
`src/CLAUDE.md`. These tests assert that the
project-wide graph respects each one:

* ty drives every piece of semantics (decl enumeration, module
  resolution, star-import expansion).
* Every import — explicit and implicit — binds a `kind="import"`
  node local to the importing file.
* Shadowed decls remain as first-class nodes; use-site edges go to
  the reaching def only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("dead_cst.native")

from dead_cst.graph import EdgeFlags, NodeFlags  # noqa: E402


@pytest.fixture
def project_factory(tmp_path: Path):
    """Materialize `{relpath: source}` files under tmp_path and yield a Project."""

    def make(files: dict[str, str], **kwargs) -> tuple[native.Project, Path]:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.Project(str(tmp_path), **kwargs), tmp_path

    return make


def _edges(graph) -> set[str]:
    return {f"{graph.nodes[s].fqname} -> {graph.nodes[d].fqname}" for s, d, _ in graph.edges}


def _fqnames(graph) -> set[str]:
    return {n.fqname for n in graph.nodes}


# ---------------------------------------------------------------------------
# Decl enumeration (Principle 1 — ty's SemanticIndex is the source of truth)
# ---------------------------------------------------------------------------


def test_function_class_variable_nodes(project_factory):
    proj, _ = project_factory({"mod.py": "def f(): pass\nclass C: pass\nX = 1\n"})
    g = proj.build()
    kinds = {n.fqname: n.kind for n in g.nodes}
    assert kinds == {
        "mod": "module",
        "mod.f": "function",
        "mod.C": "class",
        "mod.X": "variable",
    }


def test_same_file_reference_routes_through_decl(project_factory):
    proj, _ = project_factory({"mod.py": "def a(): pass\ndef b(): a()\n"})
    g = proj.build()
    assert "mod.b -> mod.a" in _edges(g)


def test_decls_inside_if_typechecking_still_register(project_factory):
    """`if TYPE_CHECKING:` doesn't create a new scope — bindings inside it
    still land in the file's global scope per ty's index."""
    proj, _ = project_factory(
        {
            "mod.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            "    def f(): pass\ndef use(): f()\n"
        }
    )
    g = proj.build()
    fqs = _fqnames(g)
    assert "mod.f" in fqs
    assert "mod.use -> mod.f" in _edges(g)


# ---------------------------------------------------------------------------
# Imports (Principle 2 — every import binds a local decl, including stars)
# ---------------------------------------------------------------------------


def test_from_import_binds_local_decl(project_factory):
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "from .other import g\ndef use(): g()\n",
            "pkg/other.py": "def g(): pass\n",
        }
    )
    g = proj.build()
    edges = _edges(g)
    # Codemod invariant: every use of an imported name emits an edge
    # to the local alias.
    assert "pkg.mod.use -> pkg.mod.g" in edges
    # Alias edges to the upstream module / decl.
    assert "pkg.mod.g -> pkg.other.g" in edges
    assert "pkg.mod.g -> pkg.other" in edges
    # Parallel reachability edges: the use *also* links directly to
    # the upstream decl + its enclosing module so reachability can
    # see what each call site actually depends on. The alias edge
    # above is what keeps `pkg.mod.g` from being marked unused.
    assert "pkg.mod.use -> pkg.other.g" in edges
    assert "pkg.mod.use -> pkg.other" in edges


def test_dotted_import_binds_first_segment(project_factory):
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "import pkg.other\ndef use(): pkg.other.f()\n",
            "pkg/other.py": "def f(): pass\n",
        }
    )
    g = proj.build()
    fqs = _fqnames(g)
    assert "pkg.mod.pkg" in fqs  # bound name is "pkg"
    assert "pkg.mod.pkg -> pkg.other" in _edges(g)  # alias → module
    assert "pkg.mod.use -> pkg.mod.pkg" in _edges(g)  # use → local alias


def test_aliased_import_binds_asname(project_factory):
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "from .other import g as gee\ndef use(): gee()\n",
            "pkg/other.py": "def g(): pass\n",
        }
    )
    g = proj.build()
    fqs = _fqnames(g)
    assert "pkg.mod.gee" in fqs
    assert "pkg.mod.g" not in fqs  # original name not bound locally
    assert "pkg.mod.use -> pkg.mod.gee" in _edges(g)
    assert "pkg.mod.gee -> pkg.other.g" in _edges(g)


def test_star_import_mints_one_node_per_statement(project_factory):
    """`from X import *` mints exactly one `kind=import` node per
    statement (named `<importing_module>.*<source>`), not one per
    name `X` exports — ty already resolves uses of star-bound names
    to their specific upstream definitions, so the per-name aliases
    libcst minted as a workaround aren't needed here. The single
    star node carries one outgoing edge to the upstream module; uses
    of star-bound names emit `use → star_node` plus the standard
    parallel `use → upstream_module` / `use → upstream_decl` edges
    via Principle 2.
    """
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/other.py": "def g(): pass\ndef h(): pass\ndef _private(): pass\n",
            "pkg/mod.py": "from .other import *\ndef use(): g(); h()\n",
        }
    )
    g = proj.build()
    fqs = _fqnames(g)
    # One star node per statement — the underscored-private decl
    # didn't matter for node count, and per-name `pkg.mod.g` /
    # `pkg.mod.h` aliases are gone.
    assert "pkg.mod.*pkg.other" in fqs
    assert "pkg.mod.g" not in fqs
    assert "pkg.mod.h" not in fqs
    edges = _edges(g)
    # Star alias's one outgoing edge: to the upstream module.
    assert "pkg.mod.*pkg.other -> pkg.other" in edges
    # Use sites: alias edge through the star + parallel upstream
    # (ty's resolution finds `g` / `h` in `pkg.other`).
    assert "pkg.mod.use -> pkg.mod.*pkg.other" in edges
    assert "pkg.mod.use -> pkg.other.g" in edges
    assert "pkg.mod.use -> pkg.other.h" in edges


def test_star_import_node_carries_star_spec(project_factory):
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/other.py": "def g(): pass\n",
            "pkg/mod.py": "from .other import *\n",
        }
    )
    g = proj.build()
    [node] = [n for n in g.nodes if n.fqname == "pkg.mod.*pkg.other"]
    assert node.kind == "import"
    assert node.imports is not None
    assert node.imports.star is True
    assert node.imports.module == "pkg.other"


def test_relative_import_resolves_to_absolute_module(project_factory):
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/mod.py": "from ..other import g\n",
            "pkg/other.py": "def g(): pass\n",
        }
    )
    g = proj.build()
    [node] = [n for n in g.nodes if n.fqname == "pkg.sub.mod.g"]
    assert node.imports.module == "pkg.other"


# ---------------------------------------------------------------------------
# Module hierarchy
# ---------------------------------------------------------------------------


def test_submodule_edges_to_parent(project_factory):
    proj, _ = project_factory(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/mod.py": "",
        }
    )
    g = proj.build()
    edges = _edges(g)
    assert "pkg.sub -> pkg" in edges
    assert "pkg.sub.mod -> pkg.sub" in edges


# ---------------------------------------------------------------------------
# Shadowed declarations (Principle 3 — first-class graph nodes)
# ---------------------------------------------------------------------------


def test_redefined_function_mints_two_nodes(project_factory):
    """Two `def f` at different lines stay distinct (positional identity)."""
    proj, _ = project_factory({"mod.py": "def f(): pass\ndef f(): pass\n"})
    g = proj.build()
    f_nodes = [n for n in g.nodes if n.fqname == "mod.f"]
    assert len(f_nodes) == 2
    assert sorted(n.start_line for n in f_nodes) == [1, 2]


def test_shadowed_def_has_no_in_edges_from_use(project_factory):
    """A dead-shadowed `def f` stays in the graph but receives no use edges;
    only the live, end-of-scope binding gets them."""
    proj, _ = project_factory({"mod.py": "def f(): pass\ndef f(): pass\ndef use(): f()\n"})
    g = proj.build()

    f_nodes = [(i, n) for i, n in enumerate(g.nodes) if n.fqname == "mod.f"]
    use_node_idx = next(i for i, n in enumerate(g.nodes) if n.fqname == "mod.use")
    edges_from_use = {dst for (src, dst, _) in g.edges if src == use_node_idx}

    live_idx = next(i for i, n in f_nodes if n.start_line == 2)
    shadowed_idx = next(i for i, n in f_nodes if n.start_line == 1)
    assert live_idx in edges_from_use
    assert shadowed_idx not in edges_from_use


# ---------------------------------------------------------------------------
# NativeGraph smoke
# ---------------------------------------------------------------------------


def test_native_graph_exposes_nodes_and_edges(project_factory):
    proj, _ = project_factory({"mod.py": "def f(): pass\nclass C: pass\n"})
    graph = proj.build()
    fqnames = {n.fqname for n in graph.nodes}
    assert fqnames == {"mod", "mod.f", "mod.C"}


def test_default_flags_are_none(project_factory):
    proj, _ = project_factory({"mod.py": "X = 1\n"})
    graph = proj.build()
    for n in graph.nodes:
        assert n.flags == NodeFlags.NONE


def test_edge_flags_are_none(project_factory):
    proj, _ = project_factory({"mod.py": "def f(): pass\n"})
    graph = proj.build()
    f_idx = next(i for i, n in enumerate(graph.nodes) if n.fqname == "mod.f")
    m_idx = next(i for i, n in enumerate(graph.nodes) if n.fqname == "mod")
    edge_flags = [f for u, v, f in graph.edges if u == f_idx and v == m_idx]
    assert edge_flags == [EdgeFlags.NONE]
