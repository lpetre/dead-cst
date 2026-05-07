"""Tests for :mod:`dead_cst.resolvers._core` and :func:`load_resolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst.resolvers import (
    Package,
    UvResolver,
    assign_file_to_package,
    export_search_root,
    is_exported_file,
    load_resolver,
    load_toml,
    validate_packages,
)


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


def _p(path: Path, name: str, **kw) -> Package:
    return Package(
        path=path,
        name=name,
        exported=tuple(kw.get("exported", (path,))),
        deps=tuple(kw.get("deps", ())),
    )


def test_validate_rejects_duplicate_paths(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError, match="duplicate Package.path"):
        validate_packages([_p(a, "p1"), _p(a, "p2")])


def test_validate_rejects_duplicate_names(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(ValueError, match="duplicate Package.name"):
        validate_packages([_p(a, "same"), _p(b, "same")])


def test_validate_rejects_empty_name(tmp_path):
    with pytest.raises(ValueError, match="name is empty"):
        validate_packages([_p(tmp_path / "a", "")])


def test_validate_rejects_exported_not_under_path(tmp_path):
    a = tmp_path / "a"
    other = tmp_path / "other"
    with pytest.raises(ValueError, match="exported"):
        validate_packages([_p(a, "p", exported=(other,))])


def test_validate_rejects_unknown_dep(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError, match="unknown dep"):
        validate_packages([_p(a, "p", deps=("missing",))])


def test_validate_rejects_self_dep(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError, match="lists itself"):
        validate_packages([_p(a, "p", deps=("p",))])


def test_validate_rejects_cycle(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(ValueError, match="cycle"):
        validate_packages([_p(a, "p1", deps=("p2",)), _p(b, "p2", deps=("p1",))])


def test_validate_returns_indices(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    pkgs = [_p(a, "p1"), _p(b, "p2", deps=("p1",))]
    v = validate_packages(pkgs)
    assert v.by_name["p1"].path == a
    assert v.by_path[b].name == "p2"
    # topo order: deps before consumers.
    assert [p.name for p in v.topo_order] == ["p1", "p2"]


def test_assign_file_to_package_picks_longest_prefix(tmp_path):
    pkg, sub = tmp_path / "pkg", tmp_path / "pkg" / "sub"
    pkg.mkdir()
    sub.mkdir()
    pkgs = [_p(pkg, "p"), _p(sub, "p_sub")]
    f1 = pkg / "a.py"
    f2 = sub / "b.py"
    f1.write_text("")
    f2.write_text("")
    assert assign_file_to_package(f1, pkgs).path == pkg.resolve()
    assert assign_file_to_package(f2, pkgs).path == sub.resolve()


def test_assign_file_to_package_returns_none_when_outside(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    other = tmp_path / "other.py"
    other.write_text("")
    assert assign_file_to_package(other, [_p(a, "p")]) is None


def test_is_exported_file_under_exported_subdir(tmp_path):
    pkg = (tmp_path / "pkg").resolve()
    pkg.mkdir()
    src = pkg / "src"
    src.mkdir()
    p = Package(path=pkg, name="pkg", exported=(src,))
    assert is_exported_file(src / "mod.py", p) is True
    assert is_exported_file(pkg / "tests" / "test.py", p) is False


def test_export_search_root_src_layout(tmp_path):
    pkg = (tmp_path / "pkg").resolve()
    src = pkg / "src"
    p = Package(path=pkg, name="pkg", exported=(src,))
    assert export_search_root(p) == src


def test_export_search_root_flat_layout(tmp_path):
    pkg = (tmp_path / "pkg").resolve()
    p = Package(path=pkg, name="pkg", exported=(pkg / "modname",))
    assert export_search_root(p) == pkg


def test_export_search_root_no_exports(tmp_path):
    pkg = (tmp_path / "pkg").resolve()
    p = Package(path=pkg, name="pkg", exported=())
    assert export_search_root(p) is None
