"""Tests for Jupyter ``.ipynb`` ingestion.

Notebooks are concatenated into a single Python source per file, parsed
with libcst, and every emitted node is flagged
``NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT``. Notebook decls stay out
of the cross-module lookup trie, but cross-file references *from* a
notebook into a real package still resolve via the normal import path,
which makes notebooks act as reachability seeds for downstream code.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from dead_cst.analyze import _find_reachable as find_reachable
from dead_cst.graph import NodeFlags


def _write_notebook(tmp_path, relpath: str, cells: list[str | dict]) -> None:
    """Write an nbformat-4 notebook with the given code-cell sources.

    A ``str`` entry becomes a code cell with that source; a ``dict`` is
    written through unmodified (use this to test markdown cells, raw
    cells, or malformed shapes).
    """
    nb_cells = []
    for cell in cells:
        if isinstance(cell, str):
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": textwrap.dedent(cell).strip() + "\n",
                }
            )
        else:
            nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb))


@pytest.fixture
def write_notebook(tmp_path):
    def _write(relpath: str, cells: list[str | dict]) -> None:
        _write_notebook(tmp_path, relpath, cells)

    return _write


def test_every_notebook_node_is_notebook_and_entrypoint(write_notebook, make_analysis):
    write_notebook(
        "explore.ipynb",
        [
            "def helper():\n    return 1\n",
            "helper()\n",
        ],
    )
    graph = make_analysis().materialize_all()
    notebook_nodes = [n for n in graph.nodes if n.path.suffix == ".ipynb"]
    assert notebook_nodes, "expected at least one node from the notebook"
    for n in notebook_nodes:
        assert n.flags & NodeFlags.NOTEBOOK, f"{n.fqname} missing NOTEBOOK"
        assert n.flags & NodeFlags.ENTRYPOINT, f"{n.fqname} missing ENTRYPOINT"


def test_notebook_keeps_referenced_py_code_alive(write_notebook, write_files, make_analysis):
    write_files({"lib.py": "def used(): return 1\ndef unused(): return 2\n"})
    write_notebook(
        "use.ipynb",
        ["from lib import used\nused()\n"],
    )
    graph = make_analysis().materialize_all()
    reachable = find_reachable(graph)
    used = next(n for n in graph.nodes if n.fqname == "lib.used")
    unused = next(n for n in graph.nodes if n.fqname == "lib.unused")
    assert used in reachable
    assert unused not in reachable


def test_magics_do_not_break_parse(write_notebook, make_analysis):
    write_notebook(
        "magics.ipynb",
        [
            "%timeit 1 + 1\n!ls -la\n?print\nprint?\n",
            "%%bash\necho hi\necho still in cell magic\n",
            "x = 1 != 2\n",  # ``!=`` is not a shell escape
        ],
    )
    graph = make_analysis().materialize_all()
    # The notebook parsed and produced at least a module node.
    assert any(n.type == "module" and n.path.suffix == ".ipynb" for n in graph.nodes)
    # The ``x`` assignment in cell 3 survived (proving ``!=`` wasn't clobbered).
    assert any(n.fqname.endswith(".x") and n.path.suffix == ".ipynb" for n in graph.nodes)


def test_malformed_notebook_falls_through_to_unparseable(
    write_notebook, tmp_path, make_analysis, caplog
):
    # Write invalid JSON manually -- the fixture would produce valid JSON.
    bad = tmp_path / "broken.ipynb"
    bad.write_text("{this is not json}")
    import logging

    with caplog.at_level(logging.WARNING):
        graph = make_analysis().materialize_all()
    # The unparseable placeholder synthetic stays in the graph so the
    # module is reachable even when ingestion fails.
    assert any(n.fqname.startswith("[unparseable]") and n.path == bad for n in graph.nodes)


def test_notebook_decls_excluded_from_cross_module_imports(
    write_notebook, write_files, make_analysis
):
    """A ``from <notebook_stem> import foo`` from a real .py must NOT resolve.

    Notebooks aren't importable; their decls are deliberately excluded
    from the lookup trie so a sibling .py file that happens to share a
    stem can't accidentally route through the notebook.
    """
    write_notebook("nb.ipynb", ["def secret():\n    return 1\n"])
    write_files({"caller.py": "from nb import secret\nsecret()\n"})
    graph = make_analysis().materialize_all()
    # The caller's import resolves to an [unresolved] synthetic, not to
    # the notebook's ``nb.secret`` decl.
    notebook_secret = [
        n for n in graph.nodes if n.fqname == "nb.secret" and n.path.suffix == ".ipynb"
    ]
    assert notebook_secret, "notebook decl still exists in the graph"
    caller_secret = next(n for n in graph.nodes if n.fqname == "caller.secret")
    targets = list(graph.successors(caller_secret))
    # No outgoing edge from the caller's import lands on the notebook decl.
    assert all(t not in notebook_secret for t in targets)


def test_codemod_skips_notebook_paths(write_notebook, make_analysis):
    """``generate_patch`` must not emit hunks for ``.ipynb`` source.

    Even if a notebook node ended up in the unreachable subgraph (it
    shouldn't, because we flag ENTRYPOINT, but defensively), the codemod
    should pre-filter on the suffix / NOTEBOOK flag.
    """
    import networkx as nx

    from dead_cst.codemod import generate_patch

    write_notebook("nb.ipynb", ["def dead():\n    return 1\n"])
    analysis = make_analysis()
    graph = analysis.materialize_all()
    nb_nodes = [n for n in graph.nodes if n.path.suffix == ".ipynb"]
    # Force-feed every notebook node into a synthetic "unreachable"
    # subgraph; the codemod must still produce an empty patch.
    sub = nx.MultiDiGraph()
    sub.add_nodes_from(nb_nodes)
    patch = generate_patch(sub, analysis.project_root)
    assert patch == ""
