"""End-to-end tests for Jupyter ``.ipynb`` ingestion via ``Analysis``."""

from __future__ import annotations

import logging

from dead_cst._graph_impl import MultiDiGraph
from dead_cst.analyze import _find_reachable as find_reachable
from dead_cst.codemod import generate_patch
from dead_cst.graph import NodeFlags


def test_every_notebook_node_is_notebook_and_entrypoint(write_notebook, make_analysis):
    write_notebook(
        "explore.ipynb",
        [
            "def helper():\n    return 1\n",
            "helper()\n",
        ],
    )
    graph = make_analysis().materialize_all()
    notebook_nodes = [n for n in graph.nodes if n.flags & NodeFlags.NOTEBOOK]
    assert notebook_nodes
    for n in notebook_nodes:
        assert n.flags & NodeFlags.ENTRYPOINT, f"{n.fqname} missing ENTRYPOINT"


def test_notebook_keeps_referenced_py_code_alive(write_notebook, write_files, make_analysis):
    write_files({"lib.py": "def used(): return 1\ndef unused(): return 2\n"})
    write_notebook("use.ipynb", ["from lib import used\nused()\n"])
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
            "x = 1 != 2\n",
        ],
    )
    graph = make_analysis().materialize_all()
    assert any(n.type == "module" and n.flags & NodeFlags.NOTEBOOK for n in graph.nodes)
    # ``x`` survives, proving ``!=`` wasn't mistaken for a shell escape.
    assert any(n.fqname.endswith(".x") and n.flags & NodeFlags.NOTEBOOK for n in graph.nodes)


def test_malformed_notebook_falls_through_to_unparseable(tmp_path, make_analysis, caplog):
    bad = tmp_path / "broken.ipynb"
    bad.write_text("{this is not json}")
    with caplog.at_level(logging.WARNING):
        graph = make_analysis().materialize_all()
    assert any(n.fqname.startswith("[unparseable]") and n.path == bad for n in graph.nodes)


def test_notebook_decls_excluded_from_cross_module_imports(
    write_notebook, write_files, make_analysis
):
    """``from <notebook_stem> import foo`` from a real .py must NOT resolve."""
    write_notebook("nb.ipynb", ["def secret():\n    return 1\n"])
    write_files({"caller.py": "from nb import secret\nsecret()\n"})
    graph = make_analysis().materialize_all()
    notebook_secret = [
        n for n in graph.nodes if n.fqname == "nb.secret" and n.flags & NodeFlags.NOTEBOOK
    ]
    assert notebook_secret
    caller_secret = next(n for n in graph.nodes if n.fqname == "caller.secret")
    targets = list(graph.successors(caller_secret))
    assert all(t not in notebook_secret for t in targets)


def test_codemod_skips_notebook_nodes(write_notebook, make_analysis):
    write_notebook("nb.ipynb", ["def dead():\n    return 1\n"])
    analysis = make_analysis()
    graph = analysis.materialize_all()
    nb_nodes = [n for n in graph.nodes if n.flags & NodeFlags.NOTEBOOK]
    sub = MultiDiGraph()
    sub.add_nodes_from(nb_nodes)
    patch = generate_patch(sub, analysis.project_root)
    assert patch == ""
