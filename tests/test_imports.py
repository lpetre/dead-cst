"""Tests for import resolution in the symbol graph.

Every case adds a ``p/x.py`` file on top of the shared package fixture
below and asserts the complete set of edges the graph contains.
``IMPORT_BASE_EDGES`` captures the edges that are always present from
the fixture so individual cases only list the edges they introduce.
"""

import pytest

from dead_cst.plugins._core import EXTERNAL_PREFIXES

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
        pytest.param(
            "from p.functions import *\ndef a(): f()",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
                "p.x.a -> p.x",
            },
            id="star-import-fans-out-to-all-decls",
        ),
        pytest.param(
            "__import__('p.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
            },
            id="dunder-import-call-fans-out-like-star",
        ),
        pytest.param(
            "def a(): getattr(__import__('p.functions'), 'f')()",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.functions.g",
                "p.x.a -> p.x",
            },
            id="dunder-import-call-inside-function-attributes-to-enclosing-decl",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('p.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-fans-out-like-star",
        ),
        pytest.param(
            "import importlib\ndef a(): importlib.import_module('p.functions')",
            {
                "p.x.a -> p.functions",
                "p.x.a -> p.functions.f",
                "p.x.a -> p.functions.g",
                "p.x.a -> p.x",
                "p.x.a -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-inside-function",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions', 'p')",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-positional-package",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions', package='p')",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-keyword-package",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module('.functions')",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
                "p.x -> p.x.importlib",
                "p.x.importlib -> p.x",
            },
            id="importlib-import-module-relative-uses-enclosing-package",
        ),
        pytest.param(
            "__import__('functions', globals(), locals(), [], 1)",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
            },
            id="dunder-import-positional-level-resolves-relative",
        ),
        pytest.param(
            "__import__('functions', level=1)",
            {
                "p.x -> p.functions",
                "p.x -> p.functions.f",
                "p.x -> p.functions.g",
            },
            id="dunder-import-keyword-level-resolves-relative",
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


def test_dynamic_import_non_literal_warns(build_decl_graph, visitor_warnings):
    """Non-literal ``__import__(name)`` / ``importlib.import_module(name)`` skip with a warning."""
    build_decl_graph(
        {
            "p/__init__.py": "",
            "p/x.py": (
                "import importlib\n"
                "name = 'p.functions'\n"
                "def a(): __import__(name)\n"
                "def b(): importlib.import_module(name)\n"
            ),
        }
    )

    messages = visitor_warnings()
    assert any("Skipping dynamic import '__import__(...)'" in m for m in messages), messages
    assert any("Skipping dynamic import 'importlib.import_module(...)'" in m for m in messages), (
        messages
    )


@pytest.mark.parametrize(
    "src",
    [
        pytest.param("__import__('p', None, None, ['functions'])", id="fromlist-positional"),
        pytest.param("__import__('p', fromlist=['functions'])", id="fromlist-keyword"),
    ],
)
def test_dunder_import_fromlist_resolves_submodules(build_decl_graph, assert_edges, src):
    """Literal ``fromlist`` entries that resolve as submodules are fanned out too."""
    graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    # Fan-out from ``p`` (empty ``__init__.py``) plus fan-out from
    # ``p.functions`` (the resolved fromlist submodule).
    assert_edges(
        graph,
        IMPORT_BASE_EDGES
        | {
            "p.x -> p.functions",
            "p.x -> p.functions.f",
            "p.x -> p.functions.g",
        },
    )


def test_dunder_import_fromlist_attribute_entries_silent(
    build_decl_graph, assert_edges, visitor_warnings
):
    """Fromlist entries that are not submodules don't warn (already covered by name fan-out)."""
    graph = build_decl_graph(
        {**IMPORT_TEST_FILES, "p/x.py": "__import__('p.functions', fromlist=['f', ''])"}
    )

    assert visitor_warnings() == []
    assert_edges(
        graph,
        IMPORT_BASE_EDGES
        | {
            "p.x -> p.functions",
            "p.x -> p.functions.f",
            "p.x -> p.functions.g",
        },
    )


@pytest.mark.parametrize(
    "src, fragment",
    [
        pytest.param(
            "__import__('.functions')",
            "leading dots are invalid for __import__",
            id="dunder-import-leading-dot-name",
        ),
        pytest.param(
            "level = 1\n__import__('functions', level=level)",
            "level is not an int literal",
            id="dunder-import-non-literal-level",
        ),
        pytest.param(
            "import importlib\npkg = 'p'\nimportlib.import_module('.functions', package=pkg)",
            "package is not a string literal",
            id="importlib-non-literal-package",
        ),
    ],
)
def test_dynamic_import_relative_warnings(build_decl_graph, visitor_warnings, src, fragment):
    build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    messages = visitor_warnings()
    assert any(fragment in m for m in messages), messages


def test_dunder_import_fromlist_non_literal_warns(build_decl_graph, visitor_warnings):
    """Non-literal fromlists warn (we can't enumerate entries)."""
    build_decl_graph(
        {
            **IMPORT_TEST_FILES,
            "p/x.py": "names = ['functions']\n__import__('p', fromlist=names)",
        }
    )

    messages = visitor_warnings()
    assert any("fromlist is not a literal" in m and "'p'" in m for m in messages), messages


def test_third_party_import_creates_synthetic_node(build_decl_graph):
    graph = build_decl_graph(
        {
            "p/__init__.py": "",
            "p/uses_nx.py": "import networkx as nx\ndef build(): return nx.DiGraph()",
        }
    )
    nx_nodes = {
        n
        for n in graph.nodes
        if n.type == "synthetic"
        and n.fqname.startswith(EXTERNAL_PREFIXES)
        and "networkx" in n.fqname
    }
    assert nx_nodes, (
        "expected an external-dep synthetic node for networkx, got "
        f"{[n.fqname for n in graph.nodes if n.type == 'synthetic']}"
    )

    edge_srcs = {src.fqname for src, dst in graph.edges(keys=False) if dst in nx_nodes}
    assert {"p.uses_nx.nx", "p.uses_nx.build"} <= edge_srcs
