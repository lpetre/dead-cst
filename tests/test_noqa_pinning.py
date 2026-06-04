"""Imports flagged with a ruff/pyflakes ``# noqa`` directive that silences
F401 are pinned alive (flagged with :data:`NodeFlags.NOQA`, a seed flag in
the default seed mask), matching the semantics ruff itself uses for the
unused-import rule.

Both per-line directives (``# noqa``, ``# noqa: F401``, multi-rule
``# noqa: E501, F401``, case-insensitive ``noqa`` keyword) and file-level
directives (``# ruff: noqa``, ``# flake8: noqa``, with or without an
explicit ``F401`` code list) are supported. The file-level prefix is
case-sensitive per ruff's documented behavior.
"""

from __future__ import annotations

import pytest

from dead_cst import NodeFlags


def _import_nodes(graph):
    return [n for n in graph.nodes() if n.kind == "import"]


def _entrypoint_imports(graph) -> set[str]:
    return {n.fqname for n in graph.nodes() if n.kind == "import" and n.flags & NodeFlags.NOQA}


@pytest.mark.parametrize(
    "src, expected_pinned",
    [
        pytest.param(
            "import os  # noqa: F401\nimport sys\n",
            {"m.os"},
            id="explicit-F401",
        ),
        pytest.param(
            "import os  # noqa\nimport sys\n",
            {"m.os"},
            id="bare-noqa",
        ),
        pytest.param(
            "import os  # noqa: E501, F401\nimport sys\n",
            {"m.os"},
            id="multi-rule",
        ),
        pytest.param(
            "import os  # noqa: E501\n",
            set(),
            id="other-rule-only",
        ),
        pytest.param(
            "import os  # NOQA: F401\nimport sys  # NoQA\n",
            {"m.os", "m.sys"},
            id="case-insensitive-noqa",
        ),
        pytest.param(
            "import os  # noqa:F401\n",
            {"m.os"},
            id="no-space-after-colon",
        ),
        pytest.param(
            "import os, sys  # noqa: F401\n",
            {"m.os", "m.sys"},
            id="multi-alias-single-line",
        ),
        pytest.param(
            "import os\n# noqa: F401\n",
            set(),
            id="comment-on-different-line",
        ),
    ],
)
def test_per_line_noqa(build_decl_graph, src, expected_pinned):
    graph = build_decl_graph({"m.py": src})
    assert _entrypoint_imports(graph) == expected_pinned


def test_per_alias_noqa_in_parenthesized_from_import(build_decl_graph):
    src = """
        from os.path import (
            join,  # noqa: F401
            sep,
            dirname,  # noqa
        )
    """
    graph = build_decl_graph({"m.py": src})
    assert _entrypoint_imports(graph) == {"m.join", "m.dirname"}


@pytest.mark.parametrize(
    "header, expect_pinned",
    [
        pytest.param("# ruff: noqa: F401", True, id="ruff-explicit"),
        pytest.param("# ruff: noqa", True, id="ruff-bare"),
        pytest.param("# flake8: noqa: F401", True, id="flake8-explicit"),
        pytest.param("# flake8: noqa", True, id="flake8-bare"),
        pytest.param("# ruff: NoQA: F401", True, id="ruff-mixed-case-noqa"),
        pytest.param("# ruff: noqa: E501", False, id="ruff-other-rule-only"),
        pytest.param("# RUFF: noqa: F401", False, id="ruff-prefix-case-sensitive"),
        pytest.param("# noqa: F401", False, id="not-file-level"),
    ],
)
def test_file_level_directive(build_decl_graph, header, expect_pinned):
    src = f"{header}\nimport os\nimport sys\nfrom json import dumps\n"
    graph = build_decl_graph({"m.py": src})
    pinned = _entrypoint_imports(graph)
    if expect_pinned:
        assert pinned == {"m.os", "m.sys", "m.dumps"}
    else:
        assert pinned == set()


def test_file_level_directive_anywhere_in_file(build_decl_graph):
    """Ruff scans the whole file for the directive; not just the header."""
    src = """
        import os
        import sys

        # ruff: noqa: F401

        def f(): ...
    """
    graph = build_decl_graph({"m.py": src})
    assert _entrypoint_imports(graph) == {"m.os", "m.sys"}


def test_pinned_import_keeps_module_alive(make_analysis, write_files):
    """A pinned import is itself an entrypoint, so the module survives reachability."""
    write_files({"m.py": "import os  # noqa: F401\n"})
    analysis = make_analysis()
    assert list(analysis.dead()) == []


def test_unpinned_import_in_dead_module_stays_dead(make_analysis, write_files):
    """Without the directive, an unused import in an otherwise-dead module is dead."""
    write_files({"m.py": "import os\n"})
    analysis = make_analysis()
    ctx = analysis.materialize_all()
    dead_fqs = {a.fqname for a in ctx.node_attrs(list(analysis.dead()))}
    assert "m.os" in dead_fqs


def test_pinned_node_carries_noqa_flag(build_decl_graph):
    graph = build_decl_graph({"m.py": "import os  # noqa: F401\n"})
    pinned = [n for n in _import_nodes(graph) if n.fqname == "m.os"]
    assert pinned and pinned[0].flags & NodeFlags.NOQA
