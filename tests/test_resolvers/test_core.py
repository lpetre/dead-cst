"""Tests for :mod:`dead_cst.resolvers._core` and :func:`load_resolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst.resolvers import (
    PyprojectResolver,
    SourceTree,
    SourceTreeFlags,
    UvWorkspaceResolver,
    assign_file_to_tree,
    load_resolver,
    load_toml,
    validate_source_trees,
)


def test_load_resolver_known():
    assert isinstance(load_resolver("pyproject"), PyprojectResolver)
    assert isinstance(load_resolver("uv_workspace"), UvWorkspaceResolver)


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


def _t(path: Path, package: str, **kw) -> SourceTree:
    return SourceTree(
        path=path,
        package=package,
        flags=kw.get("flags", SourceTreeFlags.EXPORTED),
        search_trees=tuple(kw.get("search", ())),
    )


def test_validate_rejects_duplicate_paths(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError, match="duplicate"):
        validate_source_trees([_t(a, "p1"), _t(a, "p2")])


def test_validate_rejects_empty_package(tmp_path):
    with pytest.raises(ValueError, match="package"):
        validate_source_trees([_t(tmp_path / "a", "")])


def test_validate_rejects_two_exported_in_same_package(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(ValueError, match="multiple EXPORTED"):
        validate_source_trees([_t(a, "p"), _t(b, "p")])


def test_validate_rejects_search_ref_to_unknown_path(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError, match="unknown path"):
        validate_source_trees([_t(a, "p", search=[tmp_path / "missing"])])


def test_validate_rejects_search_ref_to_non_exported_tree(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    trees = [
        _t(a, "p1", flags=SourceTreeFlags.NONE),
        _t(b, "p2", search=[a]),
    ]
    with pytest.raises(ValueError, match="not EXPORTED"):
        validate_source_trees(trees)


def test_validate_rejects_self_reference(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError, match="references itself"):
        validate_source_trees([_t(a, "p", search=[a])])


def test_validate_rejects_cycle(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(ValueError, match="cycle"):
        validate_source_trees([_t(a, "p1", search=[b]), _t(b, "p2", search=[a])])


def test_validate_returns_indices(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    trees = [_t(a, "p1"), _t(b, "p1", flags=SourceTreeFlags.NONE)]
    v = validate_source_trees(trees)
    assert v.by_path[a].package == "p1"
    assert len(v.by_package["p1"]) == 2
    assert v.exported_for["p1"].path == a


def test_assign_file_to_tree_picks_longest_prefix(tmp_path):
    pkg, tests = tmp_path / "pkg", tmp_path / "pkg" / "tests"
    pkg.mkdir()
    tests.mkdir()
    trees = [_t(pkg, "p"), _t(tests, "p_tests", flags=SourceTreeFlags.NONE)]
    f1 = pkg / "a.py"
    f2 = tests / "b.py"
    f1.write_text("")
    f2.write_text("")
    assert assign_file_to_tree(f1, trees).path == pkg.resolve()
    assert assign_file_to_tree(f2, trees).path == tests.resolve()


def test_assign_file_to_tree_returns_none_when_outside(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    other = tmp_path / "other.py"
    other.write_text("")
    assert assign_file_to_tree(other, [_t(a, "p")]) is None
