"""Engine keep-alive for module-level dunders and ``__future__`` imports.

Module-scope dunders (``__all__``, ``__version__``, PEP 562
``__getattr__``/``__dir__``, …) and ``__future__`` imports are kept alive by
an edge *from their module node*, emitted at edge-collection time. They are
**not** reachability seeds of their own: a dunder survives only while its
module is reachable. A module nothing reaches dies, and its dunders die with
it. (This used to be ``NativePlugin.module_dunders``, which flagged them as
standalone entrypoints; it is now part of the engine's edge-emission path.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dead_cst import _native as native

if TYPE_CHECKING:
    from dead_cst import _native as native_t


def _module_successors(ctx: "native_t.ProjectContext", module_fqname: str) -> set[str]:
    """One-hop targets of the ``module_fqname`` module node — i.e. the dst
    fqnames of every ``module -> X`` edge the engine emitted."""
    nodes = ctx.nodes()
    midx = next(i for i, n in enumerate(nodes) if n.kind == "module" and n.fqname == module_fqname)
    return {nodes[v].fqname for u, v, _ in ctx.edges() if u == midx}


# ---------------------------------------------------------------------------
# Structural: the engine emits a `module -> dunder` edge for the right decls
# and withholds it for everything else. These don't need an entrypoint — they
# assert the edge exists, independent of reachability seeding.
# ---------------------------------------------------------------------------


def test_emits_module_edge_to_dunder_all(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": '__all__ = ["a"]\na = 1'})
    assert "pkg.__all__" in _module_successors(ctx, "pkg")


def test_emits_module_edge_to_other_dunders(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": (
                '__version__ = "1.0.0"\n__author__ = "someone"\n__license__ = "MIT"\nunused = 1\n'
            ),
        }
    )
    succ = _module_successors(ctx, "pkg")
    assert {"pkg.__version__", "pkg.__author__", "pkg.__license__"} <= succ
    # Plain (non-dunder) variables get no module keep-alive edge.
    assert "pkg.unused" not in succ


def test_emits_module_edge_to_future_imports(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": (
                "from __future__ import annotations\nfrom __future__ import division\nunused = 1\n"
            ),
        }
    )
    succ = _module_successors(ctx, "pkg")
    # ``from __future__ import X`` local bindings get the edge even though
    # ``X`` is not a dunder name — the import is a compile-time directive
    # that can't be rewritten away.
    assert {"pkg.annotations", "pkg.division"} <= succ
    assert "pkg.unused" not in succ


def test_no_module_edge_to_non_future_imports(build_decl_graph):
    ctx = build_decl_graph({"pkg/__init__.py": "from os import path\n"})
    # Non-``__future__`` imports of plain names get no keep-alive edge.
    assert "pkg.path" not in _module_successors(ctx, "pkg")


def test_emits_module_edge_to_pep562_functions(build_decl_graph):
    """PEP 562 ``__getattr__`` and ``__dir__`` are module-level *functions*
    called by the import / attribute-access machinery — observable side
    effects with no source reference, exactly like module dunder
    *variables*. The engine must edge to them too."""
    ctx = build_decl_graph(
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
        }
    )
    succ = _module_successors(ctx, "pkg")
    assert {"pkg.__getattr__", "pkg.__dir__"} <= succ
    # Plain functions with non-dunder names get no edge.
    assert "pkg.regular" not in succ


def test_no_module_edge_to_class_dunder_methods(build_decl_graph):
    """Dunder *methods* inside a class (``__init__``, ``__call__``) are not
    module-level decls — they ride on their enclosing class's reachability
    and must not get a module keep-alive edge."""
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": (
                "class C:\n    def __init__(self): pass\n    def __call__(self): pass\n"
            ),
        }
    )
    assert _module_successors(ctx, "pkg") == set()


def test_no_module_edge_to_non_dunder_underscore_names(build_decl_graph):
    ctx = build_decl_graph(
        {
            "pkg/__init__.py": (
                "_private = 1\n"
                "__mangled = 2\n"  # leading dunder only
                "trailing__ = 3\n"  # trailing dunder only
            ),
        }
    )
    assert _module_successors(ctx, "pkg") == set()


# ---------------------------------------------------------------------------
# Reachability: the edge keeps a dunder alive *iff its module is reached*.
# A dead module's dunders die with it (the whole point of routing through an
# edge instead of a standalone entrypoint flag).
# ---------------------------------------------------------------------------


def test_dunders_track_module_reachability(write_files, make_analysis):
    """A dunder in a reachable module stays alive; the identical dunder in an
    unreachable sibling module dies."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/live.py": '__all__ = ["foo"]\n__version__ = "1.0"\ndef foo(): pass\ndef bar(): pass\n',
            "pkg/dead.py": '__all__ = ["helper"]\n__version__ = "9.9"\ndef helper(): pass\n',
        }
    )
    # Pin only `pkg.live.foo`; nothing imports `pkg.dead`.
    analysis = make_analysis(plugins=[native.NativePlugin.explicit([], ["pkg.live.foo"], [])])
    graph = analysis.materialize_all()
    reached = {n.fqname for n in graph.reachable(seed_flags=graph.default_seed_mask())}

    # Live module reached (via the foo entrypoint's decl -> module edge), so
    # its dunders ride along.
    assert "pkg.live" in reached
    assert {"pkg.live.__all__", "pkg.live.__version__"} <= reached
    # `__all__ -> foo` keeps the listed name alive transitively.
    assert "pkg.live.foo" in reached
    # A non-dunder, non-listed sibling decl stays dead even in a live module.
    assert "pkg.live.bar" not in reached

    # Dead module never reached -> its dunders die with it. This is the
    # behavior the edge (vs. a standalone entrypoint flag) buys us.
    assert "pkg.dead" not in reached
    assert "pkg.dead.__all__" not in reached
    assert "pkg.dead.__version__" not in reached
    assert "pkg.dead.helper" not in reached


def test_future_import_dies_with_unreachable_module(write_files, make_analysis):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/live.py": "from __future__ import annotations\ndef foo(): pass\n",
            "pkg/dead.py": "from __future__ import annotations\ndef helper(): pass\n",
        }
    )
    analysis = make_analysis(plugins=[native.NativePlugin.explicit([], ["pkg.live.foo"], [])])
    graph = analysis.materialize_all()
    reached = {n.fqname for n in graph.reachable(seed_flags=graph.default_seed_mask())}

    assert "pkg.live.annotations" in reached
    assert "pkg.dead.annotations" not in reached
