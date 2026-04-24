"""Tests for import resolution in the symbol graph.

Every case adds a ``p/x.py`` file on top of the shared package fixture
below and asserts the complete set of edges the graph contains.
``IMPORT_BASE_EDGES`` captures the edges that are always present from
the fixture so individual cases only list the edges they introduce.
"""

import pytest

IMPORT_TEST_FILES = {
    "p/__init__.py": "",
    "p/functions.py": "def f(): pass",
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
        "p.x -> p",
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
        pytest.param(
            "from p.chain import functions as g\ndef a(): g.f()",
            {
                "p.x.a -> p.chain",
                "p.x.a -> p.chain.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.g",
                "p.x.g -> p.chain",
                "p.x.g -> p.chain.functions",
                "p.x.g -> p.x",
            },
            id="import-chain-via-reexport",
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
        pytest.param(
            "from p.chain import functions\ndef a(): functions.f()",
            {
                "p.x.a -> p.chain",
                "p.x.a -> p.chain.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.functions",
                "p.x.functions -> p.chain",
                "p.x.functions -> p.chain.functions",
                "p.x.functions -> p.x",
            },
            id="reexport-through-package-init",
        ),
        pytest.param(
            "import p\ndef a(): p.functions.f()",
            {
                "p.x.a -> p",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.x",
                "p.x.a -> p.x.p",
                "p.x.p -> p",
                "p.x.p -> p.x",
            },
            id="import-package-then-dotted-access",
        ),
    ],
)
def test_imports(build_decl_graph, assert_edges, src, expected_extra_edges):
    graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    assert_edges(graph, IMPORT_BASE_EDGES | expected_extra_edges)
