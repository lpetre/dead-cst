"""Tests for ``foo.py`` eclipsed by a sibling ``foo/__init__.py`` package.

CPython's ``FileFinder`` resolves ``import pkg.foo`` to the package
directory whenever both shapes coexist, so the analyzer mirrors that
precedence: the trie holds the ``__init__.py`` and any cross-module
import of ``pkg.foo`` (or of a name re-exported from it) routes to the
package alone. The eclipsed file is still parsed -- its nodes appear in
the graph and observe-time entrypoints (``__main__``, plugin
synthetics) keep working -- but consumer imports never see its decls.
"""

from __future__ import annotations

from dead_cst.graph import KEEPALIVE_DEFAULT
from dead_cst.plugins import ExplicitEntrypointPlugin, MainBlockPlugin


def test_package_wins_trie_slot_for_cross_module_imports(
    tmp_path, write_files, make_analysis, successors_of
):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/foo.py": "def x(): pass\n",
            "pkg/foo/__init__.py": "def x(): pass\n",
            "caller.py": "from pkg.foo import x\nx()\n",
        }
    )
    analysis = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["caller"])])
    graph = analysis.materialize_all()

    caller_x = next(n for n in graph.nodes() if n.fqname == "caller.x")
    targets = [s for s in successors_of(graph, caller_x) if s.fqname == "pkg.foo.x"]
    assert len(targets) == 1
    assert targets[0].path.endswith("/__init__.py")

    reachable = set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))
    package_x = next(
        n for n in graph.nodes() if n.fqname == "pkg.foo.x" and n.path.endswith("/__init__.py")
    )
    eclipsed_x = next(
        n for n in graph.nodes() if n.fqname == "pkg.foo.x" and n.path.endswith("/foo.py")
    )
    assert package_x in reachable
    assert eclipsed_x not in reachable


def test_eclipsed_file_keeps_main_block_entrypoint(tmp_path, write_files, make_analysis):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/foo.py": "def helper():\n    pass\nif __name__ == '__main__':\n    helper()\n",
            "pkg/foo/__init__.py": "value = 1\n",
        }
    )
    analysis = make_analysis(plugins=[MainBlockPlugin()])
    graph = analysis.materialize_all()
    reachable = set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))

    helper = next(n for n in graph.nodes() if n.fqname == "pkg.foo.helper")
    eclipsed_module = next(
        n for n in graph.nodes() if n.fqname == "pkg.foo" and n.path.endswith("/foo.py")
    )
    package_module = next(
        n for n in graph.nodes() if n.fqname == "pkg.foo" and n.path.endswith("/__init__.py")
    )
    main_synth = next(
        n for n in graph.nodes() if n.kind == "synthetic" and n.fqname.endswith(":pkg.foo")
    )

    assert main_synth in reachable, "MainBlockPlugin synthetic must seed reachability"
    assert helper in reachable, "decls in the eclipsed file are alive when the synth reaches them"
    assert eclipsed_module in reachable, "eclipsed module node stays alive via the synth"
    # The package itself is unreached: nothing imports pkg.foo from outside.
    assert package_module not in reachable


def test_name_only_in_eclipsed_file_is_unresolvable(
    tmp_path, write_files, make_analysis, successors_of
):
    """A name only the eclipsed ``.py`` defines does not satisfy a consumer
    import -- mirrors what Python would do at runtime (``ImportError``).
    The eclipsed file's decl does not surface in the package's lookup,
    so the caller's import resolves only to the package (no decl edge).
    """
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/foo.py": "ONLY_IN_PY = 1\n",
            "pkg/foo/__init__.py": "ONLY_IN_PKG = 2\n",
            "caller.py": "from pkg.foo import ONLY_IN_PY\n",
        }
    )
    analysis = make_analysis(plugins=[ExplicitEntrypointPlugin(specs=["caller"])])
    graph = analysis.materialize_all()
    # The import doesn't bind a real decl, so the caller's import node
    # does not edge to any ``ONLY_IN_PY`` decl.
    caller_module = next(n for n in graph.nodes() if n.fqname == "caller")
    targets = {s.fqname for s in successors_of(graph, caller_module)}
    assert "pkg.foo.ONLY_IN_PY" not in targets
