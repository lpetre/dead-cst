"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

import textwrap

import pytest

from dead_cst import find_reachable


@pytest.fixture
def write_files(tmp_path):
    """Write a ``{relpath: source}`` mapping under ``tmp_path``.

    Each value is dedented and stripped, with a trailing newline appended,
    matching the inline-source convention used across these tests.
    """

    def _write(files: dict[str, str]) -> None:
        for name, src in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src).strip() + "\n")

    return _write


@pytest.fixture
def reachable_fqnames():
    """Return ``{fqname for n in find_reachable(graph) if not synthetic}``."""

    def _reachable(graph) -> set[str]:
        return {n.fqname for n in find_reachable(graph) if n.type != "synthetic"}

    return _reachable
