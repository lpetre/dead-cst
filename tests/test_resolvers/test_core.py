"""Tests for :mod:`dead_cst._resolvers._core` and :func:`load_resolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst import (
    PyprojectResolver,
    UvWorkspaceResolver,
    VenvResolver,
    load_resolver,
    merge_paths,
)


def test_merge_paths_unions_deps():
    base = Path("/a")
    d1 = Path("/b")
    d2 = Path("/c")
    result = merge_paths({base: [d1]}, {base: [d2]}, {base: [d1]})
    assert result == {base: [d1, d2]}


def test_merge_paths_drops_self():
    base = Path("/a")
    result = merge_paths({base: [base]})
    assert result == {base: []}


def test_load_resolver_known():
    assert isinstance(load_resolver("venv"), VenvResolver)
    assert isinstance(load_resolver("pyproject"), PyprojectResolver)
    assert isinstance(load_resolver("uv_workspace"), UvWorkspaceResolver)


def test_load_resolver_unknown_raises():
    with pytest.raises(KeyError):
        load_resolver("does-not-exist")
