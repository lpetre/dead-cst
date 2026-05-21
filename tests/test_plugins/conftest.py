"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dead_cst import Analysis
from dead_cst.graph import KEEPALIVE_DEFAULT

if TYPE_CHECKING:
    from dead_cst import _native as native


@pytest.fixture
def reachable_fqnames():
    """``{fqname for n in ctx.reachable() if n.kind != "synthetic"}``."""

    def _reachable(ctx: "native.ProjectContext") -> set[str]:
        reached = ctx.reachable(seed_flags=KEEPALIVE_DEFAULT)
        return {n.fqname for n in reached if n.kind != "synthetic"}

    return _reachable


@pytest.fixture
def build_plugin_graph(tmp_path, write_files):
    """Materialise inline files with the given plugins applied and return
    the live :class:`native.ProjectContext`."""

    def _build(files: dict[str, str], plugins: list) -> "native.ProjectContext":
        write_files(files)
        return Analysis(tmp_path, plugins=plugins).materialize_all()

    return _build
