"""Tests for :mod:`dead_cst.resolvers._core` and :func:`load_resolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst.resolvers import (
    Package,
    UvResolver,
    load_resolver,
    load_toml,
    merge_packages,
)


def test_merge_packages_unions_deps():
    a = Package(path=Path("/a"), name="a", deps=("b",))
    b = Package(path=Path("/b"), name="b")
    a_again = Package(path=Path("/a"), name="a", deps=("b",))
    result = merge_packages([a, b], [a_again])
    assert {p.name for p in result} == {"a", "b"}
    a_merged = next(p for p in result if p.name == "a")
    assert a_merged.deps == ("b",)


def test_merge_packages_resolves_paths():
    a = Package(path=Path("/a"), name="a")
    result = merge_packages([a])
    assert result[0].path == Path("/a").resolve()


def test_merge_packages_unknown_dep_raises():
    a = Package(path=Path("/a"), name="a", deps=("missing",))
    with pytest.raises(ValueError, match="unknown dep"):
        merge_packages([a])


def test_merge_packages_duplicate_name_distinct_path_raises():
    a = Package(path=Path("/a"), name="x")
    b = Package(path=Path("/b"), name="x")
    with pytest.raises(ValueError, match="Duplicate package name"):
        merge_packages([a, b])


def test_merge_packages_exported_must_be_under_path(tmp_path: Path):
    pkg = Package(path=tmp_path / "a", name="a", exported=(tmp_path / "b",))
    with pytest.raises(ValueError, match="not under"):
        merge_packages([pkg])


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
