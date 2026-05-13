"""Unit tests for :mod:`dead_cst._notebooks` (the ingestion helper).

Pure-function surface: JSON loading, magic neutralization, FQN
synthesis. End-to-end shape is exercised by ``tests/test_notebooks.py``.
"""

from __future__ import annotations

import json

from dead_cst._notebooks import (
    is_notebook,
    notebook_fqn_entry,
    notebook_to_module,
)


def test_is_notebook(tmp_path):
    assert is_notebook(tmp_path / "x.ipynb")
    assert not is_notebook(tmp_path / "x.py")


def test_concat_joins_cells_in_order(tmp_path, write_notebook):
    write_notebook("a.ipynb", ["def f():\n    return 1", "def g():\n    return 2"])
    src = notebook_to_module(tmp_path / "a.ipynb")
    assert src is not None
    assert "def f" in src
    assert "def g" in src
    assert src.index("def f") < src.index("def g")


def test_markdown_cells_ignored(tmp_path, write_notebook):
    write_notebook("a.ipynb", [{"cell_type": "markdown", "source": "# heading"}, "x = 1"])
    src = notebook_to_module(tmp_path / "a.ipynb")
    assert src is not None
    assert "heading" not in src
    assert "x = 1" in src


def test_line_magic_neutralized(tmp_path, write_notebook):
    write_notebook("a.ipynb", ["%timeit 1 + 1\nx = 1"])
    src = notebook_to_module(tmp_path / "a.ipynb")
    assert src is not None
    assert "pass  # %timeit" in src
    assert "x = 1" in src


def test_shell_escape_neutralized_but_neq_preserved(tmp_path, write_notebook):
    write_notebook("a.ipynb", ["!ls -la\nflag = 1 != 0"])
    src = notebook_to_module(tmp_path / "a.ipynb")
    assert src is not None
    assert "pass  # !ls -la" in src
    assert "flag = 1 != 0" in src


def test_cell_magic_swallows_rest_of_cell(tmp_path, write_notebook):
    write_notebook(
        "a.ipynb",
        [
            "%%bash\necho one\necho two\n",
            "x = 1",
        ],
    )
    src = notebook_to_module(tmp_path / "a.ipynb")
    assert src is not None
    assert "pass  # %%bash" in src
    assert "pass  # echo one" in src
    assert "pass  # echo two" in src
    assert "x = 1" in src


def test_help_suffix_neutralized(tmp_path, write_notebook):
    write_notebook("a.ipynb", ["print?", "obj.attr??", "x = 1"])
    src = notebook_to_module(tmp_path / "a.ipynb")
    assert src is not None
    assert "pass  # print?" in src
    assert "pass  # obj.attr??" in src


def test_invalid_json_returns_none(tmp_path):
    nb = tmp_path / "broken.ipynb"
    nb.write_text("{not json")
    assert notebook_to_module(nb) is None


def test_empty_cells_returns_none(tmp_path):
    nb = tmp_path / "empty.ipynb"
    nb.write_text(json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}))
    assert notebook_to_module(nb) is None


def test_fqn_sanitizes_path(tmp_path):
    entry = notebook_fqn_entry(tmp_path / "explore-data.ipynb")
    assert entry.name == "explore_data"
    assert entry.package == ""

    entry2 = notebook_fqn_entry(tmp_path / "1st_run.ipynb")
    assert entry2.name.startswith("_")
