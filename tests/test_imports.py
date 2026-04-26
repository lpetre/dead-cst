import pytest

IMPORT_TEST_FILES = {
    "p/__init__.py": "",
    "p/functions.py": "def f(): pass",
    "p/classes.py": "class C(): pass",
    "p/chain.py": "from . import functions",
}


@pytest.mark.parametrize(
    "src, symbol_edges",
    [
        pytest.param(
            "import p.functions\ndef a(): p.functions.f()",
            {"p.x.a": {"p.x", "p.x.p", "p.functions", "p.functions.f"}},
            id="simple-cst.Import",
        ),
        pytest.param(
            "from p.functions import f\ndef a(): f()",
            {
                "p.x.f": {"p.x", "p.functions", "p.functions.f"},
                "p.x.a": {"p.x", "p.x.f", "p.functions", "p.functions.f"},
            },
            id="simple-cst.ImportFrom",
        ),
        pytest.param(
            "from p.classes import C\ndef a(): C.f()",
            {
                "p.x.C": {"p.x", "p.classes", "p.classes.C"},
                "p.x.a": {"p.x", "p.classes", "p.classes.C", "p.x.C"},
            },
            id="simple cst.ImportFrom for classmethod",
        ),
        # import as
        pytest.param(
            "import p.functions as f\ndef a(): f.f()",
            {"p.x.a": {"p.x", "p.x.f", "p.functions", "p.functions.f"}},
            id="import module with alias",
        ),
        pytest.param(
            "from p import functions as f\ndef a(): f.f()",
            {"p.x.a": {"p.x", "p.x.f", "p.functions", "p.functions.f"}},
            id="import module from module with alias",
        ),
        # nested import
        pytest.param(
            "def a(): import p.functions; p.functions.f()",
            {"p.x.a": {"p.x", "p.functions", "p.functions.f"}},
            id="nested cst.Import",
        ),
        pytest.param(
            "def a(): from p.functions import f; f()",
            {"p.x.a": {"p.x", "p.functions", "p.functions.f"}},
            id="nested cst.ImportFrom",
        ),
        # relative import
        pytest.param(
            "from .functions import f\ndef a(): f()",
            {"p.x.f": {"p.x", "p.functions", "p.functions.f"}},
            id="relative import",
        ),
        # import chain
        pytest.param(
            "from p.chain import functions as g\ndef a(): g.f()",
            {"p.x.a": {"p.x", "p.x.g", "p.chain", "p.chain.functions", "p.functions.f"}},
            id="import chain",
        ),
    ],
)
def test_imports(build_decl_graph, src, symbol_edges):
    import_test_graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    found_symbols = {n.fqname for n in import_test_graph.nodes}
    for expected_symbol, expected_edges in symbol_edges.items():
        assert expected_symbol in found_symbols
        found_edges = {
            dst.fqname for src, dst in import_test_graph.edges if src.fqname == expected_symbol
        }
        assert found_edges == expected_edges


@pytest.mark.parametrize(
    "src, symbol_edges",
    [
        pytest.param(
            'from p.functions import f\n__all__ = ["f"]',
            {"p.x.__all__": {"p.x", "p.x.f"}},
            id="dunder-all references import",
        ),
        pytest.param(
            'from p.functions import f\nfrom p.classes import C\n__all__ = ("f", "C")',
            {"p.x.__all__": {"p.x", "p.x.f", "p.x.C"}},
            id="dunder-all tuple of imports",
        ),
        pytest.param(
            'def g(): pass\nfrom p.functions import f\n__all__ = ["f", "g"]',
            {"p.x.__all__": {"p.x", "p.x.f", "p.x.g"}},
            id="dunder-all mixes local and imported",
        ),
        pytest.param(
            'from p.functions import f\n__all__: list[str] = ["f"]',
            {"p.x.__all__": {"p.x", "p.x.f"}},
            id="dunder-all annotated assignment",
        ),
        pytest.param(
            'from p.functions import f\n__all__ = ["missing"]',
            {"p.x.__all__": {"p.x"}},
            id="dunder-all unknown name is ignored",
        ),
    ],
)
def test_dunder_all_edges(build_decl_graph, src, symbol_edges):
    import_test_graph = build_decl_graph({**IMPORT_TEST_FILES, "p/x.py": src})
    found_symbols = {n.fqname for n in import_test_graph.nodes}
    for expected_symbol, expected_edges in symbol_edges.items():
        assert expected_symbol in found_symbols
        found_edges = {
            dst.fqname for src, dst in import_test_graph.edges if src.fqname == expected_symbol
        }
        assert found_edges == expected_edges


def test_module_keeps_dunder_all_alive(build_decl_graph):
    graph = build_decl_graph(
        {**IMPORT_TEST_FILES, "p/x.py": 'from p.functions import f\n__all__ = ["f"]'}
    )
    edges = {(src.fqname, dst.fqname) for src, dst in graph.edges}
    assert ("p.x", "p.x.__all__") in edges
