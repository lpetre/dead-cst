"""Tests for :mod:`dead_cst.resolvers._core` and :func:`load_resolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst.resolvers import (
    UvResolver,
    load_resolver,
    load_toml,
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
    assert isinstance(load_resolver("uv"), UvResolver)


def test_load_resolver_unknown_raises():
    with pytest.raises(KeyError):
        load_resolver("does-not-exist")


def test_load_toml_returns_parsed_data(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('[project]\nname = "x"\n')
    assert load_toml(p) == {"project": {"name": "x"}}


def test_load_toml_missing_file_returns_none(tmp_path):
    assert load_toml(tmp_path / "missing.toml") is None


def test_load_toml_directory_returns_none(tmp_path):
    """``is_file()`` is False for a directory, so we get None rather than IsADirectoryError."""
    assert load_toml(tmp_path) is None


def test_load_toml_invalid_syntax_propagates(tmp_path):
    """Bad TOML is a programmer/config error -- callers see the parse exception."""
    import tomllib

    p = tmp_path / "bad.toml"
    p.write_text("this is = not = valid\n")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_toml(p)
