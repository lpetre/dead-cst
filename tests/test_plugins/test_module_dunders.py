"""Tests for :class:`ModuleDundersPlugin`.

Runs against both backends via :func:`build_plugin_graph` so the rust
``ProjectContext`` plugin protocol is exercised against the same scenarios
as libcst's ``observe`` / ``finalize`` pair.
"""

from __future__ import annotations

from dead_cst.plugins import ModuleDundersPlugin


def test_keeps_all_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {"pkg/__init__.py": '__all__ = ["a"]\na = 1'},
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg.__all__" in reached
    # The visitor wires __all__ -> decl edges for string-literal entries,
    # so preserving __all__ transitively preserves the listed names too.
    assert "pkg.a" in reached


def test_keeps_other_dunders_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": (
                '__version__ = "1.0.0"\n__author__ = "someone"\n__license__ = "MIT"\nunused = 1\n'
            ),
        },
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert {"pkg.__version__", "pkg.__author__", "pkg.__license__"} <= reached
    # plain (non-dunder) variables are still dead absent another entrypoint
    assert "pkg.unused" not in reached


def test_keeps_future_imports_alive(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": (
                "from __future__ import annotations\nfrom __future__ import division\nunused = 1\n"
            ),
        },
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    # The local bindings of ``from __future__ import X`` are kept alive
    # even though ``X`` is not a dunder name -- the import itself is a
    # compile-time directive that can't be rewritten away.
    assert {"pkg.annotations", "pkg.division"} <= reached
    assert "pkg.unused" not in reached


def test_ignores_non_future_imports_with_plain_names(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {"pkg/__init__.py": "from os import path\n"},
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    # Non-``__future__`` imports of plain names stay dead absent another entrypoint.
    assert "pkg.path" not in reached


def test_keeps_module_level_dunder_functions_alive(build_plugin_graph, reachable_fqnames):
    """PEP 562 ``__getattr__`` and ``__dir__`` are module-level *functions*
    called by the import / attribute-access machinery — observable side
    effects with no source reference, exactly like module dunder
    *variables*. The plugin must pin them too.
    """
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": (
                "_EXPORTS = {'Foo': '.foo'}\n"
                "def __getattr__(name):\n"
                "    return _EXPORTS[name]\n"
                "def __dir__():\n"
                "    return list(_EXPORTS)\n"
                "def regular():\n"
                "    return _EXPORTS\n"
            ),
        },
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert {"pkg.__getattr__", "pkg.__dir__"} <= reached
    # The supporting variable they reference rides along.
    assert "pkg._EXPORTS" in reached
    # Plain functions with non-dunder names stay dead absent another
    # entrypoint — the plugin only pins the dunder names.
    assert "pkg.regular" not in reached


def test_does_not_pin_class_dunder_methods(build_plugin_graph, reachable_fqnames):
    """Dunder *methods* inside a class (``__init__``, ``__call__``) are
    not module-level decls — they ride on their enclosing class's
    reachability. The plugin must not pin them as separate entrypoints.
    """
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": (
                "class C:\n    def __init__(self): pass\n    def __call__(self): pass\n"
            ),
        },
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    # The class itself isn't pinned absent another entrypoint.
    assert "pkg.C" not in reached


def test_ignores_non_dunder_underscore_names(build_plugin_graph, reachable_fqnames):
    graph = build_plugin_graph(
        {
            "pkg/__init__.py": (
                "_private = 1\n"
                "__mangled = 2\n"  # leading dunder only
                "trailing__ = 3\n"  # trailing dunder only
            ),
        },
        [ModuleDundersPlugin()],
    )
    reached = reachable_fqnames(graph)
    assert "pkg._private" not in reached
    assert "pkg.__mangled" not in reached
    assert "pkg.trailing__" not in reached
