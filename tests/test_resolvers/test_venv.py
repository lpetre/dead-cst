"""Tests for :class:`dead_cst._resolvers.venv.VenvResolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dead_cst import VenvResolver
from dead_cst._resolvers import MissingVenvError


def test_venv_resolver_finds_site_packages(tmp_path: Path):
    venv = tmp_path / ".venv"
    sp = venv / "lib" / "python3.13" / "site-packages"
    sp.mkdir(parents=True)

    result = VenvResolver().resolve(tmp_path)
    assert tmp_path.resolve() in result
    assert sp.resolve() in result[tmp_path.resolve()]


def test_venv_resolver_missing_raises(tmp_path: Path):
    # Passing an explicit (nonexistent) venv_dir skips the active-venv probe,
    # so there's nowhere for the resolver to find a venv -- it raises rather
    # than silently producing an empty path map (which would just defer the
    # failure into the plugin pass).
    with pytest.raises(MissingVenvError, match="nope"):
        VenvResolver(venv_dir="nope").resolve(tmp_path)


def test_venv_resolver_custom_dir(tmp_path: Path):
    venv = tmp_path / "envs" / "myenv"
    sp = venv / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)

    result = VenvResolver(venv_dir="envs/myenv").resolve(tmp_path)
    assert sp.resolve() in result[tmp_path.resolve()]
