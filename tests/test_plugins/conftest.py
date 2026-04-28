"""Shared fixtures for the plugin test suite."""

from __future__ import annotations

import pytest

from dead_cst import find_reachable


@pytest.fixture
def reachable_fqnames():
    """Return ``{fqname for n in find_reachable(graph) if not synthetic}``."""

    def _reachable(graph) -> set[str]:
        return {n.fqname for n in find_reachable(graph) if n.type != "synthetic"}

    return _reachable
