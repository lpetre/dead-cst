"""Tests for import resolution in the symbol graph.

Every case adds a ``p/x.py`` file on top of the shared package fixture
below and asserts the complete set of edges the graph contains.
``IMPORT_BASE_EDGES`` captures the edges that are always present from
the fixture so individual cases only list the edges they introduce.
"""

import logging

import pytest

from dead_cst.plugins._core import EXTERNAL_PREFIXES, STDLIB_PREFIX, UNRESOLVED_PREFIX

IMPORT_TEST_FILES = {
    "p/__init__.py": "",
    "p/functions.py": "def f(): pass\ndef g(): pass",
    "p/classes.py": "class C(): pass",
    "p/chain.py": "from . import functions",
}

# Edges always produced by IMPORT_TEST_FILES plus an (empty-or-populated)
# p/x.py file: module-hierarchy edges, the `p.chain` re-import, and the
# parent edges for p.x itself.
IMPORT_BASE_EDGES = frozenset(
    {
        "p.chain -> p",
        "p.chain.functions -> p.chain",
        "p.chain.functions -> p.functions",
        "p.classes -> p",
        "p.classes.C -> p.classes",
        "p.functions -> p",
        "p.functions.f -> p.functions",
        "p.functions.g -> p.functions",
        "p.x -> p",
    }
)

# Edges from materializing ``from p.functions import *`` (or a synonym)
# into ``p/x.py``: one synthetic re-export per name plus the
# ``module -> synthetic -> target`` chain. Used by every star-shaped
# test case in this file.
STAR_REEXPORT_EDGES = frozenset(
    {
        "p.x -> p.x.f",
        "p.x -> p.x.g",
        "p.x.f -> p.functions",
        "p.x.f -> p.functions.f",
        "p.x.f -> p.x",
        "p.x.g -> p.functions",
        "p.x.g -> p.functions.g",
        "p.x.g -> p.x",
    }
)


@pytest.mark.parametrize(
    "src, expected_extra_edges",
    [
        pytest.param(
            "import p.functions\ndef a(): p.functions.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.p",
                "p.x.p -> p.functions",
                "p.x.p -> p.x",
            },
            id="cst.Import-dotted-module",
        ),
        pytest.param(
            "from p.functions import f\ndef a(): f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="cst.ImportFrom-function",
        ),
        pytest.param(
            "from p.classes import C\ndef a(): C.f()",
            {
                "p.x.C -> p.classes",
                "p.x.C -> p.classes.C",
                "p.x.C -> p.x",
                "p.x.a -> p.classes",
                "p.x.a -> p.classes.C",
                "p.x.a -> p.x",
                "p.x.a -> p.x.C",
            },
            id="cst.ImportFrom-class-attribute-access",
        ),
        pytest.param(
            "import p.functions as f\ndef a(): f.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.x",
            },
            id="cst.Import-with-alias",
        ),
        pytest.param(
            "from p import functions as f\ndef a(): f.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.x",
            },
            id="cst.ImportFrom-with-alias",
        ),
        pytest.param(
            "def a(): import p.functions; p.functions.f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.Import",
        ),
        pytest.param(
            "def a(): from p.functions import f; f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.ImportFrom",
        ),
        pytest.param(
            # Nested import with no use of the bound name still
            # creates a dep on the upstream module — the import
            # statement itself is a side effect attributed to the
            # enclosing decl.
            "def a():\n    import p.functions\n",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
            },
            id="nested-cst.Import-no-use",
        ),
        pytest.param(
            # `from X import Y` with no use still emits both the
            # parent module and the upstream decl edges, just like
            # the module-scope ImportFrom alias does.
            "def a():\n    from p.functions import f\n",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.ImportFrom-no-use",
        ),
        pytest.param(
            # Bare-`import p` then dotted access through it inside
            # the function body. The chain `.functions.f` walks
            # submodule then decl from the bound name `p`.
            "def a():\n    import p\n    p.functions.f()\n",
            {
                "p.x.a -> p",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
            },
            id="nested-cst.Import-bare-then-dotted",
        ),
        pytest.param(
            # Nested import shadows a module-scope import of the
            # same root name. Inside `a` the use resolves to the
            # nested binding; outside it the module-scope alias
            # still points at its own upstream.
            "import p.functions\ndef a():\n    import p.classes\n    p.classes.C()\n",
            {
                "p.x.a -> p.classes",
                "p.x.a -> p.classes.C",
                "p.x.a -> p.x",
                "p.x.p -> p.functions",
                "p.x.p -> p.x",
            },
            id="nested-cst.Import-shadows-module-scope",
        ),
        pytest.param(
            # Import in a class body. The class is the owner;
            # the chain walks normally.
            "class A:\n    import p.functions\n    p.functions.f()\n",
            {
                "p.x.A -> p.functions",
                "p.x.A -> p.functions.f",
                "p.x.A -> p.x",
            },
            id="class-body-cst.Import",
        ),
        pytest.param(
            "class A:\n    from p.functions import f\n    f()\n",
            {
                "p.x.A -> p.functions",
                "p.x.A -> p.functions.f",
                "p.x.A -> p.x",
            },
            id="class-body-cst.ImportFrom",
        ),
        pytest.param(
            # Imports inside method bodies attribute to the
            # enclosing class — methods are not separate top-level
            # nodes, per the project's "nested defs are folded
            # into the enclosing top-level decl" convention.
            "class A:\n    def m(self):\n        import p.functions\n        p.functions.f()\n",
            {
                "p.x.A -> p.functions",
                "p.x.A -> p.functions.f",
                "p.x.A -> p.x",
            },
            id="method-body-cst.Import",
        ),
        pytest.param(
            # `try: import X as A; except: import Y as A` — both
            # branches reach end-of-scope with `A` bound, so a use
            # of `A` (or the import statements alone) attributes
            # edges to *both* upstreams (Principle 3 — every
            # reaching def gets edges).
            "def a():\n"
            "    try:\n"
            "        import p.functions as src\n"
            "    except ImportError:\n"
            "        import p.classes as src\n"
            "    src\n",
            {
                "p.x.a -> p.classes",
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
            },
            id="nested-try-except-cst.Import",
        ),
        pytest.param(
            # Nested star import fans out to every name `p.functions`
            # exports, attributed to the enclosing function. No alias
            # node is minted.
            "def a():\n    from p.functions import *\n    f()\n",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.functions.g",
                "p.x.a -> p.x",
            },
            id="nested-star-import",
        ),
        pytest.param(
            "from .functions import f\ndef a(): f()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="relative-import",
        ),
        # libcst follows the reexport chain at the use site, so `g.f()`
        # picks up parallel `p.x.a -> p.chain.functions` /
        # `p.x.a -> p.functions.f` edges (in addition to the alias
        # edges on `p.x.g`). The rust backend stops at the alias edge
        # — `p.x.g` already points at `p.chain.functions`, but no
        # parallel use-site edges fan through that chain.
        pytest.param(
            "from p.chain import functions as g\ndef a(): g.f()",
            {
                "p.x.a -> p.chain",
                "p.x.a -> p.chain.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.g",
                "p.x.g -> p.chain",
                "p.x.g -> p.chain.functions",
                "p.x.g -> p.x",
            },
            id="import-chain-via-reexport-rust",
        ),
        pytest.param(
            "from p.functions import f\nfrom p.classes import C\ndef a(): f(); C()",
            {
                "p.x.C -> p.classes",
                "p.x.C -> p.classes.C",
                "p.x.C -> p.x",
                "p.x.a -> p.classes",
                "p.x.a -> p.classes.C",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.C",
                "p.x.a -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="multiple-from-imports",
        ),
        pytest.param(
            "import p.functions\n",
            {
                "p.x.p -> p.functions",
                "p.x.p -> p.x",
            },
            id="bare-cst.Import-keeps-module-alive",
        ),
        pytest.param(
            "from p.functions import f\n",
            {
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="bare-cst.ImportFrom-keeps-module-alive",
        ),
        pytest.param(
            "from p.functions import f\nf()",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="module-level-call-of-imported-symbol",
        ),
        # Same reexport-chain divergence as `import-chain-via-reexport`
        # above: libcst emits parallel use-site edges through the
        # chain (`p.x.a -> p.chain.functions`, `p.x.a -> p.functions.f`),
        # rust stops at the alias edge on `p.x.functions`.
        pytest.param(
            "from p.chain import functions\ndef a(): functions.f()",
            {
                "p.x.a -> p.chain",
                "p.x.a -> p.chain.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.functions",
                "p.x.functions -> p.chain",
                "p.x.functions -> p.chain.functions",
                "p.x.functions -> p.x",
            },
            id="reexport-through-package-init-rust",
        ),
        pytest.param(
            "import p\ndef a(): p.functions.f()",
            # The access ``p.functions.f`` synthesizes an
            # ``Import(module="p", decl="functions.f")``; the stitcher
            # canonicalizes it to ``module="p.functions"`` because the
            # whole prefix resolves as a submodule, so the edge points
            # at the deepest reached module rather than at ``p``.
            # Reachability of ``p`` itself is still preserved through
            # ``p.x.p -> p`` plus the ``p.functions -> p`` parent-edge.
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.p",
                "p.x.p -> p",
                "p.x.p -> p.x",
            },
            id="import-package-then-dotted-access",
        ),
        # libcst minted per-name synthetic aliases as a workaround for
        # its inability to resolve uses *through* a star import; the
        # rust backend leans on ty and mints one node per
        # `from X import *` statement (named `<mod>.*<src>`) with a
        # single outgoing edge to the upstream module. Use sites
        # route through this node and emit the standard parallel
        # upstream module/decl edges via ty's name resolution
        # (Principle 2).
        pytest.param(
            "from p.functions import *\ndef a(): f()",
            {
                "p.x.*p.functions -> p.x",
                "p.x.*p.functions -> p.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.*p.functions",
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
            },
            id="star-import-fans-out-to-all-decls-rust",
        ),
        # ------------------------------------------------------------------
        # Dynamic imports (`__import__` / `importlib.import_module`).
        #
        # libcst minted per-name synthetic aliases that fanned out to
        # every export of the target module. The rust backend leans on
        # ty: one edge per *explicit symbol* the call mentions, tagged
        # with `EdgeFlags.DYNAMIC_IMPORT` so a contrib plugin can fan
        # out further if a project wants the old semantic.
        # ------------------------------------------------------------------
        pytest.param(
            "__import__('p.functions')",
            {"p.x -> p.functions"},
            id="dunder-import-call-fans-out-like-star-rust",
        ),
        pytest.param(
            "def a(): getattr(__import__('p.functions'), 'f')()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
            },
            id="dunder-import-call-inside-function-attributes-to-enclosing-decl-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('p.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-fans-out-like-star-rust",
        ),
        pytest.param(
            "import importlib\ndef a(): importlib.import_module('p.functions')",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.x",
                "p.x.a -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-inside-function-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions', 'p')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-positional-package-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions', package='p')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-keyword-package-rust",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-uses-enclosing-package-rust",
        ),
        pytest.param(
            "__import__('functions', globals(), locals(), [], 1)",
            {"p.x -> p.functions"},
            id="dunder-import-positional-level-resolves-relative-rust",
        ),
        pytest.param(
            "__import__('functions', level=1)",
            {"p.x -> p.functions"},
            id="dunder-import-keyword-level-resolves-relative-rust",
        ),
    ],
)
def test_imports(build_decl_graph, assert_edges, src, expected_extra_edges):
    graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    assert_edges(graph, IMPORT_BASE_EDGES | expected_extra_edges)


@pytest.mark.parametrize(
    "src, expected_extra_edges",
    [
        pytest.param(
            'from p.functions import f\n__all__ = ["f"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-references-import",
        ),
        pytest.param(
            'from p.functions import f\nfrom p.classes import C\n__all__ = ("f", "C")',
            {
                "p.x.C -> p.classes",
                "p.x.C -> p.classes.C",
                "p.x.C -> p.x",
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.C",
                "p.x.__all__ -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-tuple-of-imports",
        ),
        pytest.param(
            'def g(): pass\nfrom p.functions import f\n__all__ = ["f", "g"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.f",
                "p.x.__all__ -> p.x.g",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
                "p.x.g -> p.x",
            },
            id="dunder-all-mixes-local-and-imported",
        ),
        pytest.param(
            'from p.functions import f\n__all__: list[str] = ["f"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.__all__ -> p.x.f",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-annotated-assignment",
        ),
        pytest.param(
            'from p.functions import f\n__all__ = ["missing"]',
            {
                "p.x.__all__ -> p.x",
                "p.x.f -> p.functions",
                "p.x.f -> p.functions.f",
                "p.x.f -> p.x",
            },
            id="dunder-all-unknown-name-is-ignored",
        ),
    ],
)
def test_dunder_all_edges(build_decl_graph, assert_edges, src, expected_extra_edges):
    graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    assert_edges(graph, IMPORT_BASE_EDGES | expected_extra_edges)


@pytest.mark.parametrize(
    "src",
    [
        pytest.param("__import__('p', None, None, ['functions'])", id="fromlist-positional"),
        pytest.param("__import__('p', fromlist=['functions'])", id="fromlist-keyword"),
    ],
)
def test_dunder_import_fromlist_resolves_submodules_rust(build_decl_graph, assert_edges, src):
    """Rust emits one edge per explicit symbol the call mentions: the
    base module plus each literal fromlist entry that resolves as a
    submodule, all tagged ``EdgeFlags.DYNAMIC_IMPORT``."""
    graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    assert_edges(
        graph,
        IMPORT_BASE_EDGES
        | {
            "p.x -> p",
            "p.x -> p.functions",
        },
    )


def test_dunder_import_fromlist_attribute_entries_silent_rust(build_decl_graph, assert_edges):
    """The rust backend looks each entry up as a global-scope decl in
    the base module and emits an edge if found (no fan-out to g),
    otherwise drops silently — and never warns on attribute-style
    entries."""
    graph = build_decl_graph(
        {**IMPORT_TEST_FILES, "p/x.py": "__import__('p.functions', fromlist=['f', ''])"}
    )
    assert_edges(
        graph,
        IMPORT_BASE_EDGES
        | {
            "p.x -> p.functions",
            "p.x -> p.functions.f",
        },
    )


def test_third_party_import_creates_synthetic_node(build_decl_graph):
    graph = build_decl_graph(
        {
            "p/__init__.py": "",
            "p/uses_rx.py": "import rustworkx as rx\ndef build(): return rx.PyDiGraph()",
        }
    )
    rx_nodes = {
        n
        for n in graph.nodes
        if n.kind == "synthetic"
        and n.fqname.startswith(EXTERNAL_PREFIXES)
        and "rustworkx" in n.fqname
    }
    assert rx_nodes, (
        "expected an external-dep synthetic node for rustworkx, got "
        f"{[n.fqname for n in graph.nodes if n.kind == 'synthetic']}"
    )

    edge_srcs = {
        graph.node(u).fqname for u, v in graph.raw.edge_list() if graph.node(v) in rx_nodes
    }
    assert {"p.uses_rx.rx", "p.uses_rx.build"} <= edge_srcs


def test_third_party_import_uses_canonical_dist_name(build_decl_graph):
    """``import yaml`` lands on ``[external dist] pyyaml`` (PEP 503).

    The distribution's top-level module (``yaml``) and its canonical
    PyPI name (``PyYAML`` → ``pyyaml``) differ. Plugins query by the
    canonical name so the synthetic must match what
    :data:`importlib.metadata`'s ``Name:`` header normalizes to.
    """
    graph = build_decl_graph(
        {
            "p/__init__.py": "",
            "p/uses_yaml.py": "import yaml\nDATA = yaml.safe_load('a: 1')\n",
        }
    )
    fqnames = {n.fqname for n in graph.nodes if n.kind == "synthetic"}
    assert "[external dist] pyyaml" in fqnames, fqnames


def test_stdlib_imports_are_silent(build_decl_graph, caplog):
    """Stdlib imports drop without surfacing a synthetic node or a warning."""
    with caplog.at_level(logging.WARNING, logger="dead_cst._edges"):
        graph = build_decl_graph(
            {
                "p/__init__.py": "",
                "p/uses_stdlib.py": (
                    "import datetime\n"
                    "import os\n"
                    "from pathlib import Path\n"
                    "from collections.abc import Iterable\n"
                ),
            }
        )

    assert [r.getMessage() for r in caplog.records] == []
    synthetics = {n.fqname for n in graph.nodes if n.kind == "synthetic"}
    # No stdlib ever surfaces as a graph node, and ``collections.abc``
    # must not fall through to ``[unresolved] collections`` (regression
    # against the synthesized-submodule parent-fallback).
    assert not [fq for fq in synthetics if fq.startswith(STDLIB_PREFIX)]
    assert f"{UNRESOLVED_PREFIX}collections" not in synthetics


def test_unresolved_import_emits_synthetic_silently(build_decl_graph, caplog):
    """A genuinely-missing top-level import gets a ``[unresolved]`` node, no warning."""
    with caplog.at_level(logging.WARNING, logger="dead_cst._edges"):
        graph = build_decl_graph(
            {
                "p/__init__.py": "",
                "p/uses_missing.py": "from unknown_pkg_xyz import thing\n",
            }
        )

    assert [r.getMessage() for r in caplog.records] == []
    assert any(
        n.kind == "synthetic" and n.fqname == f"{UNRESOLVED_PREFIX}unknown_pkg_xyz"
        for n in graph.nodes
    )


def test_module_runtime_dunder_access_is_module_dep(build_decl_graph, assert_edges, caplog):
    """``pkg.__file__`` collapses to a plain ``Import(module=pkg, decl=None)``.

    The import machinery injects ``__file__`` / ``__name__`` / ``__spec__`` /
    etc. onto every module object at runtime -- attribute access past one of
    these is a path / string op, not a symbol reference. The visitor truncates
    the access chain at the dunder so the edge stitcher sees a clean module-
    level dependency (no speculative ``decl="__file__"`` that it would warn
    about, no synthetic node for the missing attribute).
    """
    with caplog.at_level(logging.WARNING, logger="dead_cst._edges"):
        graph = build_decl_graph(
            {
                "pkg/__init__.py": "",
                "pkg/config.py": (
                    "from pathlib import Path\n"
                    "import pkg as pkg_alias\n"
                    "FILE_PATH = Path(pkg_alias.__file__).parent\n"
                    "NAME = pkg_alias.__name__\n"
                    "SPEC = pkg_alias.__spec__\n"
                ),
            }
        )

    assert [r.getMessage() for r in caplog.records] == []
    # No synthetic was minted for the missing-dunder lookup.
    assert not [n.fqname for n in graph.nodes if n.fqname.endswith(".__file__")]
    # The module-level dependency edges remain intact for each user of a dunder.
    edge_strs = {
        f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
    }
    assert "pkg.config.FILE_PATH -> pkg" in edge_strs
    assert "pkg.config.NAME -> pkg" in edge_strs
    assert "pkg.config.SPEC -> pkg" in edge_strs


def test_dunder_on_imported_symbol_strips_dunder_tail(build_decl_graph, assert_edges):
    """``from pkg import Cls; Cls.__name__`` -> edge to ``Cls``, not ``Cls.__name__``.

    Regression guard for the visitor-level dunder strip: the truncation
    drops the dunder *and everything after it*, so an access through an
    imported symbol still resolves to that symbol (not past it).
    """
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "class Cls:\n    pass\n",
            "pkg/uses.py": ("from pkg.lib import Cls\nWHO = Cls.__name__\nDOCSTR = Cls.__doc__\n"),
        }
    )
    edge_strs = {
        f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
    }
    assert "pkg.uses.WHO -> pkg.lib.Cls" in edge_strs
    assert "pkg.uses.DOCSTR -> pkg.lib.Cls" in edge_strs


def test_cyclic_reexport_terminates_rust(build_decl_graph, assert_edges):
    """Re-export cycle terminates without spinning (rust behavior).

    The rust backend resolves each alias once and stops — no
    reexport-chain self-edges (``A.x -> A.x``), no transitive walk
    through the cycle (no ``main -> B.x``). The cycle itself is still
    represented (``A.x -> B.x``, ``B.x -> A.x``) so reachability from
    `main` can still walk it. The use-site does emit the standard
    Principle 2 parallel-upstream edge to ``A.x`` (the decl in A's
    namespace that the import resolved to) — same shape as
    ``main -> A`` for the upstream module.
    """
    graph = build_decl_graph(
        {
            "A.py": "from B import x",
            "B.py": "from A import x",
            "main.py": "from A import x\nx()",
        }
    )

    assert_edges(
        graph,
        {
            "A.x -> A",
            "A.x -> B",
            "A.x -> B.x",
            "B.x -> A",
            "B.x -> A.x",
            "B.x -> B",
            "main -> A",
            "main -> A.x",
            "main -> main.x",
            "main.x -> A",
            "main.x -> A.x",
            "main.x -> main",
        },
    )


def test_import_resolves_through_star_reexport_rust(build_decl_graph, assert_edges):
    """``from pkg import g`` resolves through the star node to ``pkg._internal.g``.

    The rust backend mints one ``*pkg._internal`` node per
    ``from pkg._internal import *`` statement rather than a per-name
    ``pkg.g`` alias; ty resolves ``from pkg import g`` straight to
    ``pkg._internal.g``, so the consumer alias edges directly to the
    upstream decl (no intermediate ``pkg.g``).
    """
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "from pkg._internal import *\n",
            "pkg/_internal.py": "def g(): pass\n",
            "consumer.py": "from pkg import g\ng()\n",
        }
    )
    edge_strs = {
        f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
    }
    assert "consumer.g -> pkg._internal.g" in edge_strs
    assert "consumer.g -> pkg" in edge_strs
    # No `pkg.g` alias is minted; the star node carries the reexport.
    assert "consumer.g -> pkg.g" not in edge_strs
    assert any(n.fqname == "pkg.*pkg._internal" for n in graph.nodes)


def test_import_resolves_through_chained_star_reexports(build_decl_graph, assert_edges):
    """``from A import g`` follows ``A -> B -> C`` star chain to ``C.g``."""
    graph = build_decl_graph(
        {
            "A.py": "from B import *\n",
            "B.py": "from C import *\n",
            "C.py": "def g(): pass\n",
            "consumer.py": "from A import g\ng()\n",
        }
    )
    edge_strs = {
        f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
    }
    assert "consumer.g -> C.g" in edge_strs


def test_star_reexport_cycle_terminates(build_decl_graph, assert_edges):
    """Mutual ``from B import *`` / ``from A import *`` terminates without spinning."""
    graph = build_decl_graph(
        {
            "A.py": "from B import *\ndef a(): pass\n",
            "B.py": "from A import *\ndef b(): pass\n",
            "consumer.py": "from A import b\n",
        }
    )
    edge_strs = {
        f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
    }
    assert "consumer.b -> B.b" in edge_strs


def test_star_reexport_shadowed_by_real_decl(build_decl_graph, assert_edges):
    """A real decl in the importing module wins over a star re-export of the same name."""
    graph = build_decl_graph(
        {
            "other.py": "def g(): pass\n",
            "mod.py": "from other import *\ndef g(): return 1\n",
            "consumer.py": "from mod import g\ng()\n",
        }
    )
    edge_strs = {
        f"{graph.node(u).fqname} -> {graph.node(v).fqname}" for u, v in graph.raw.edge_list()
    }
    assert "consumer.g -> mod.g" in edge_strs
    assert "consumer.g -> other.g" not in edge_strs


def test_from_import_prefers_namespace_binding_over_submodule(build_decl_graph):
    """``from p import q`` where ``p/__init__.py`` binds ``q`` (e.g. to an
    int) and ``p/q.py`` *also* exists: CPython's ``_handle_fromlist``
    binds ``q`` to the int via the namespace, the submodule is never
    imported, and dead code in ``p/q.py`` stays dead.

    Pins the rust backend's matching behavior (``resolve_from_imported``
    probes ``globals_by_name`` before falling back to the submodule
    lookup). The libcst path canonicalizes the import the other way
    (`_edges.py` pushes ``q`` into the module name as long as it
    resolves as a submodule in the trie) and is wrong for this corner
    case — when libcst is updated this test should drop the
    backend-skip and start enforcing the assertion everywhere.
    """
    graph = build_decl_graph(
        {
            "p/__init__.py": "q = 42\n",
            "p/q.py": "def dead(): pass\n",
            "consumer.py": "from p import q\nprint(q)\n",
        }
    )
    # Same fqname appears twice (the `q` variable in p/__init__.py and
    # the `q` module from p/q.py), so we have to disambiguate on type.
    consumer_q_alias = next(
        n for n in graph.nodes if n.fqname == "consumer.q" and n.kind == "import"
    )
    targets = {
        (graph.node(v).fqname, graph.node(v).kind)
        for u, v in graph.raw.edge_list()
        if graph.index(consumer_q_alias) == u
    }
    assert ("p.q", "variable") in targets, targets
    # Crucial: the submodule must NOT be linked. The old submodule-first
    # order would wrongly add ("p.q", "module"); the namespace-first
    # order matches CPython and skips it.
    assert ("p.q", "module") not in targets, targets


def test_star_reexport_is_skipped_by_codemod_rust(build_decl_graph):
    """The rust backend mints one node per ``from X import *`` named
    ``<mod>.*<src>`` with ``kind="import"`` and an ``imports.star=True``
    payload — the ``*`` prefix is the codemod's marker."""
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "from pkg._internal import *\n",
            "pkg/_internal.py": "def g(): pass\n",
        }
    )
    star_nodes = [n for n in graph.nodes if n.fqname == "pkg.*pkg._internal" and n.kind == "import"]
    assert len(star_nodes) == 1, [n.fqname for n in graph.nodes if "pkg" in n.fqname]
    assert star_nodes[0].imports is not None
    assert star_nodes[0].imports.star is True
    assert star_nodes[0].imports.module == "pkg._internal"


def test_cross_dep_submodule_import(tmp_path, make_analysis, assert_edges):
    """Importing a submodule from a dep base resolves through the dep's exported trie.

    Layout:
        pkg_a/A/__init__.py  -- empty package
        pkg_a/A/sub.py       -- def f(): ...
        pkg_b/B/__init__.py  -- from A import sub; sub.f()

    Packages: pkg_b(deps=("pkg_a",)), pkg_a(deps=())

    ``A.sub`` lives in the dep's exported trie, not the consumer's own
    trie. ``resolve_edges`` must find it via the merged ``symbol_lookup``
    and follow the submodule through ``cur.children`` (since ``A/__init__.py``
    has no explicit ``import sub`` declaration -- the name resolves as a
    real submodule path).
    """
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    (pkg_a / "A").mkdir(parents=True)
    (pkg_b / "B").mkdir(parents=True)
    (pkg_a / "A" / "__init__.py").write_text("")
    (pkg_a / "A" / "sub.py").write_text("def f(): ...\n")
    (pkg_b / "B" / "__init__.py").write_text("from A import sub\nsub.f()\n")

    graph = make_analysis(["pkg_b:pkg_a", "pkg_a"]).materialize_all()
    assert_edges(
        graph,
        {
            "A.sub -> A",
            "A.sub.f -> A.sub",
            "B -> A.sub",
            "B -> A.sub.f",
            "B -> B.sub",
            "B.sub -> A.sub",
            "B.sub -> B",
        },
    )
