"""Unit tests for :mod:`dead_cst._notebooks` (the ingestion helper).

These cover the pure-function surface -- JSON loading, magic
neutralization, line-to-cell mapping -- without going through the
full ``Analysis`` pipeline. The end-to-end shape is exercised by
``tests/test_notebooks.py``.
"""

from __future__ import annotations

import json
import textwrap

from dead_cst._notebooks import (
    is_notebook,
    notebook_fqn_entry,
    notebook_to_module,
)


def _write_nb(path, cells):
    nb = {
        "cells": [
            (
                cell
                if isinstance(cell, dict)
                else {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": textwrap.dedent(cell).strip() + "\n",
                }
            )
            for cell in cells
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


def test_is_notebook(tmp_path):
    assert is_notebook(tmp_path / "x.ipynb")
    assert not is_notebook(tmp_path / "x.py")


def test_concat_preserves_cell_boundaries(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(nb, ["def f():\n    return 1", "def g():\n    return 2"])
    src = notebook_to_module(nb)
    assert src is not None
    assert "def f" in src.text
    assert "def g" in src.text
    # Two code cells preserved in order.
    assert len(src.cell_indices) == 2
    assert src.cell_indices == (0, 1)
    # Line-start of cell 2 lands after cell 1's lines.
    assert src.cell_line_starts[0] == 1
    assert src.cell_line_starts[1] > 1


def test_locate_round_trip(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(nb, ["x = 1\ny = 2", "z = 3"])
    src = notebook_to_module(nb)
    assert src is not None
    # First line is in cell 0.
    assert src.locate(1) == (0, 1)
    # The "z = 3" line is in cell 1.
    z_line = src.text.splitlines().index("z = 3") + 1
    assert src.locate(z_line) == (1, 1)


def test_markdown_cells_ignored(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(nb, [{"cell_type": "markdown", "source": "# heading"}, "x = 1"])
    src = notebook_to_module(nb)
    assert src is not None
    assert "heading" not in src.text
    assert src.cell_indices == (1,)


def test_line_magic_neutralized(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(nb, ["%timeit 1 + 1\nx = 1"])
    src = notebook_to_module(nb)
    assert src is not None
    assert "pass  # %timeit" in src.text
    assert "x = 1" in src.text


def test_shell_escape_neutralized_but_neq_preserved(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(nb, ["!ls -la\nflag = 1 != 0"])
    src = notebook_to_module(nb)
    assert src is not None
    assert "pass  # !ls -la" in src.text
    assert "flag = 1 != 0" in src.text


def test_cell_magic_swallows_rest_of_cell(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(
        nb,
        [
            "%%bash\necho one\necho two\n",
            "x = 1",  # next cell stays plain Python
        ],
    )
    src = notebook_to_module(nb)
    assert src is not None
    assert "pass  # %%bash" in src.text
    assert "pass  # echo one" in src.text
    assert "pass  # echo two" in src.text
    assert "x = 1" in src.text


def test_help_suffix_neutralized(tmp_path):
    nb = tmp_path / "a.ipynb"
    _write_nb(nb, ["print?", "obj.attr??", "x = 1"])
    src = notebook_to_module(nb)
    assert src is not None
    assert "pass  # print?" in src.text
    assert "pass  # obj.attr??" in src.text


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
    assert entry2.name.startswith("_") or entry2.name == "_1st_run"
