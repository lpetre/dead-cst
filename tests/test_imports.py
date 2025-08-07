def test_simple_import(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "x.py": "import y, nested.z\ndef a(): y.b()",
            "y.py": "def b(): pass",
            "nested/z.py": "",
        }
    )
    assert_edges(
        graph,
        {
            "x.y -> x",
            "x.nested.z -> x",
            "x.a -> x",
            "x.a -> x.y",
            "x.a -> y.b",
            "x.y -> y",
            "y.b -> y",
            "x.nested.z -> nested.z",
        },
    )


def test_asname_import(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "x.py": "import nested.y as z\ndef a(): z.b()",
            "nested/__init__.py": "",
            "nested/y.py": "def b(): pass",
        }
    )
    assert_edges(
        graph,
        {
            "x.a -> x",
            "x.z -> x",
            "x.a -> x.z",
            "x.a -> nested.y.b",
            "x.z -> nested",
            "nested.y.b -> nested.y",
        },
    )


def test_import_from(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "x.py": "from y import b as d, c\ndef a(): d()",
            "y.py": "from nested import z\ndef b(): pass\ndef c(): pass",
            "nested/z.py": "",
        }
    )
    assert_edges(
        graph,
        {
            "x.a -> x",
            "x.d -> x",
            "x.c -> x",
            "x.a -> x.d",
            "x.d -> y.b",
            "x.c -> y.c",
            "y.b -> y",
            "y.c -> y",
            "y.z -> y",
            "y.z -> nested.z",
        },
    )


def test_nested_import(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "x.py": "def a(): import y; y.b()",
            "y.py": "def b(): pass",
        }
    )
    assert_edges(
        graph,
        {
            "x.a -> x",
            "x.a -> y",
            "x.a -> y.b",
            "y.b -> y",
        },
    )


def test_relative_import(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "pkg/__init__.py": "",
            "pkg/x.py": "def a(): pass",
            "pkg/y/__init__.py": "",
            "pkg/y/z.py": "from ..x import a\ndef b(): a()",
        }
    )
    assert_edges(
        graph,
        {
            "pkg.x.a -> pkg.x",
            "pkg.y.z.a -> pkg.y.z",
            "pkg.y.z.b -> pkg.y.z",
            "pkg.y.z.b -> pkg.y.z.a",
            "pkg.y.z.a -> pkg.x.a",
        },
    )
