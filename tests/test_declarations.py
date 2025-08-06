def test_func(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
def a(): pass
def b(): a()
b()
"""
        }
    )
    assert_edges(
        graph,
        {
            "mod.a -> mod",
            "mod.b -> mod",
            "mod.b -> mod.a",
            "mod -> mod.b",
        },
    )


def test_class(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
class A: pass
class B(A): pass
"""
        }
    )
    assert_edges(
        graph,
        {
            "mod.A -> mod",
            "mod.B -> mod",
            "mod.B -> mod.A",
        },
    )


def test_simple_variable(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
a = 1
b = a
"""
        }
    )
    assert_edges(
        graph,
        {
            "mod.a -> mod",
            "mod.b -> mod",
            "mod.b -> mod.a",
        },
    )


def test_tuple_variable(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
a, b = 1, 2
c, d = a, b
"""
        }
    )
    assert_edges(
        graph,
        {
            "mod.a -> mod",
            "mod.b -> mod",
            "mod.c -> mod",
            "mod.d -> mod",
            "mod.c -> mod.a",
            "mod.d -> mod.b",
        },
    )


def test_multiple_assign_variable(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
a = 1
b = c = a
"""
        }
    )
    assert_edges(
        graph,
        {
            "mod.a -> mod",
            "mod.b -> mod",
            "mod.c -> mod",
            "mod.b -> mod.a",
            "mod.c -> mod.a",
        },
    )


def test_nested_decl(build_decl_graph, assert_edges):
    graph = build_decl_graph(
        {
            "mod.py": """
def a(): return 1
def b():
    def c():
        return a()
"""
        }
    )
    assert_edges(
        graph,
        {
            "mod.a -> mod",
            "mod.b -> mod",
            "mod.b -> mod.a",
        },
    )
