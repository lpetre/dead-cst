"""End-to-end tests for :data:`NodeFlags.NOQA` and ``kept_alive_by_flags_only(NodeFlags.NOQA)``.

Imports preserved by a ruff/pyflakes ``# noqa[: ...F401...]`` (per-line)
or a file-level ``# ruff: noqa`` / ``# flake8: noqa`` are stamped with
:data:`NodeFlags.NOQA` -- one of the keepalive bits in
:data:`KEEPALIVE_DEFAULT`, so reachability seeds from them by default.
The flag-taking blast-radius query returns modules and decls currently
kept alive only because of those pinned imports.
"""

from __future__ import annotations

from dead_cst import NodeFlags
from dead_cst.graph import KEEPALIVE_DEFAULT


def find_reachable_excluding_noqa(graph):
    return set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT & ~NodeFlags.NOQA))


def find_kept_alive_by_noqa_only(graph):
    full = set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))
    return full - find_reachable_excluding_noqa(graph)


def test_noqa_pin_keeps_module_alive_only_via_noqa(make_analysis, write_files):
    """A module that's only kept alive by a ``# noqa: F401`` import shows up
    in ``kept_alive_by_noqa_only`` but not in the strict reachable set."""
    write_files(
        {
            "pkg/__init__.py": """
            import pkg.side_effect  # noqa: F401
            """,
            "pkg/side_effect.py": """
            CONST = 1
            """,
        }
    )
    graph = make_analysis().materialize_all()
    side = next(n for n in graph.nodes() if n.fqname == "pkg.side_effect")
    assert side in set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))
    assert side not in find_reachable_excluding_noqa(graph)
    assert side in find_kept_alive_by_noqa_only(graph)


def test_production_only_decl_survives_strict_pass(make_analysis, write_files):
    """A decl reachable from a non-noqa entrypoint is not in
    ``kept_alive_by_noqa_only``."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": """
            def helper(): return 1

            if __name__ == "__main__":
                helper()
            """,
        }
    )
    from dead_cst.plugins import MainBlockPlugin

    graph = make_analysis(plugins=[MainBlockPlugin()]).materialize_all()
    helper = next(n for n in graph.nodes() if n.fqname == "pkg.lib.helper")
    assert helper in find_reachable_excluding_noqa(graph)
    assert helper not in find_kept_alive_by_noqa_only(graph)


def test_pinned_import_carries_noqa_flag(build_decl_graph):
    """Per-line ``# noqa: F401`` stamps the NOQA keepalive bit on the import node."""
    graph = build_decl_graph({"m.py": "import os  # noqa: F401\n"})
    pinned = next(n for n in graph.nodes() if n.fqname == "m.os")
    assert pinned.flags & NodeFlags.NOQA


def test_file_level_directive_stamps_noqa(build_decl_graph):
    """File-level ``# ruff: noqa`` stamps NOQA on every import in the file."""
    src = "# ruff: noqa\nimport os\nimport sys\n"
    graph = build_decl_graph({"m.py": src})
    for fqn in ("m.os", "m.sys"):
        node = next(n for n in graph.nodes() if n.fqname == fqn)
        assert node.flags & NodeFlags.NOQA, fqn


def test_unpinned_import_has_no_noqa_flag(build_decl_graph):
    graph = build_decl_graph({"m.py": "import os\n"})
    node = next(n for n in graph.nodes() if n.fqname == "m.os")
    assert not (node.flags & NodeFlags.NOQA)


def test_excluding_multiple_flags_in_one_pass(make_analysis, write_files):
    """``reachable(seed_flags=...)`` accepts a combined ``IntFlag``
    value so callers can drop several entrypoint classes at once."""
    write_files(
        {
            "pkg/__init__.py": """
            import pkg.side_effect  # noqa: F401
            """,
            "pkg/side_effect.py": "X = 1",
            "tests/__init__.py": "",
            "tests/test_x.py": """
            from pkg.side_effect import X

            def test_x(): assert X == 1
            """,
        }
    )
    from dead_cst.contrib import PytestPlugin

    graph = make_analysis(plugins=[PytestPlugin()]).materialize_all()
    side = next(n for n in graph.nodes() if n.fqname == "pkg.side_effect")
    assert side in set(graph.reachable(seed_flags=KEEPALIVE_DEFAULT))
    excluded = set(
        graph.reachable(seed_flags=KEEPALIVE_DEFAULT & ~(NodeFlags.TESTCASE | NodeFlags.NOQA))
    )
    assert side not in excluded


def test_analysis_kept_alive_by_flags_only_noqa(make_analysis, write_files):
    write_files(
        {
            "pkg/__init__.py": """
            import pkg.side_effect  # noqa: F401
            """,
            "pkg/side_effect.py": "X = 1",
        }
    )
    analysis = make_analysis()
    diff = analysis.kept_alive_by_flags_only(NodeFlags.NOQA)
    ctx = analysis.materialize_all()
    fqnames = {a.fqname for a in ctx.node_attrs(list(diff))}
    assert "pkg.side_effect" in fqnames
