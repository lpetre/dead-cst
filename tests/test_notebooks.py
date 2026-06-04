"""End-to-end tests for Jupyter ``.ipynb`` ingestion via ``Analysis``."""

from __future__ import annotations


from dead_cst.codemod import generate_patch
from dead_cst.graph import NodeFlags


def test_every_notebook_node_carries_notebook_flag(write_notebook, make_analysis):
    write_notebook(
        "explore.ipynb",
        [
            "def helper():\n    return 1\n",
            "helper()\n",
        ],
    )
    ctx = make_analysis().materialize_all()
    notebook_nodes = [n for n in ctx.nodes() if n.flags & NodeFlags.NOTEBOOK]
    assert notebook_nodes
    # ``NOTEBOOK`` alone is enough — it's a seed flag, so the BFS seeds
    # from these nodes by default (it's in the registry-derived
    # ``default_seed_mask``) without needing an explicit ``ENTRYPOINT``
    # overlay.
    assert any(n.flags & ctx.default_seed_mask() for n in notebook_nodes), (
        "notebook nodes should be keepalive seeds"
    )


def test_notebook_keeps_referenced_py_code_alive(write_notebook, write_files, make_analysis):
    write_files({"lib.py": "def used(): return 1\ndef unused(): return 2\n"})
    write_notebook("use.ipynb", ["from lib import used\nused()\n"])
    ctx = make_analysis().materialize_all()
    reachable = set(ctx.reachable(seed_flags=ctx.default_seed_mask()))
    used = next(n for n in ctx.nodes() if n.fqname == "lib.used")
    unused = next(n for n in ctx.nodes() if n.fqname == "lib.unused")
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
    ctx = make_analysis().materialize_all()
    assert any(n.kind == "module" and n.flags & NodeFlags.NOTEBOOK for n in ctx.nodes())
    # ``x`` survives, proving ``!=`` wasn't mistaken for a shell escape.
    assert any(n.fqname.endswith(".x") and n.flags & NodeFlags.NOTEBOOK for n in ctx.nodes())


def test_codemod_skips_notebook_nodes(write_notebook, make_analysis):
    write_notebook("nb.ipynb", ["def dead():\n    return 1\n"])
    analysis = make_analysis()
    ctx = analysis.materialize_all()
    nb_nodes = [n for n in ctx.nodes() if n.flags & NodeFlags.NOTEBOOK]
    patch = generate_patch(nb_nodes, analysis.project_root)
    assert patch == ""
