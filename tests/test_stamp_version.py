"""The publish workflow's version stamper.

`dead-cst` and `dead-cst-plugin-host` are built in separate CI jobs but MUST
ship the identical version (the runtime version is baked into the ABI
fingerprint a native plugin is checked against). `scripts/stamp_version.py` is
the single source of truth for that; these tests pin the lockstep + the
Cargo<->PEP 440 spelling so a refactor can't silently desync them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stamp_version.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stamp_version", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stamper(tmp_path, monkeypatch):
    """The stamp module pointed at throwaway Cargo/pyproject copies."""
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        '[workspace]\nmembers = ["."]\n\n[package]\nname = "dead-cst-native"\nversion = "9.9.9"\n'
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project.optional-dependencies]\nbuild-plugin = ["dead-cst-plugin-host"]\n'
    )
    ph = tmp_path / "plugin-host" / "pyproject.toml"
    ph.parent.mkdir()
    ph.write_text('[project]\nname = "dead-cst-plugin-host"\nversion = "0.0.0"\n')

    mod = _load_module()
    monkeypatch.setattr(mod, "CARGO", cargo)
    monkeypatch.setattr(mod, "PYPROJECT", pyproject)
    monkeypatch.setattr(mod, "PLUGIN_HOST_PYPROJECT", ph)
    return mod, cargo, pyproject, ph


def test_release_stamp_locks_all_three_to_cargo(stamper, monkeypatch):
    mod, cargo, pyproject, ph = stamper
    monkeypatch.setattr(sys, "argv", ["stamp_version.py", "--release"])

    mod.main()

    assert 'version = "9.9.9"' in cargo.read_text()  # source of truth, untouched
    assert 'version = "9.9.9"' in ph.read_text()
    assert 'build-plugin = ["dead-cst-plugin-host==9.9.9"]' in pyproject.read_text()


def test_release_normalizes_cargo_dev_spelling_to_pep440(stamper, monkeypatch):
    # A stamped dev Cargo version uses SemVer `-dev`; the Python artifacts must
    # use PEP 440 `.dev` so the pin actually matches the published wheel.
    mod, cargo, pyproject, ph = stamper
    cargo.write_text('[package]\nname = "dead-cst-native"\nversion = "0.13.1-dev5"\n')
    monkeypatch.setattr(sys, "argv", ["stamp_version.py", "--release"])

    mod.main()

    assert 'version = "0.13.1.dev5"' in ph.read_text()
    assert 'build-plugin = ["dead-cst-plugin-host==0.13.1.dev5"]' in pyproject.read_text()


def test_pin_is_idempotent(stamper, monkeypatch):
    mod, _cargo, pyproject, _ph = stamper
    pyproject.write_text('build-plugin = ["dead-cst-plugin-host==1.0.0"]\n')
    monkeypatch.setattr(sys, "argv", ["stamp_version.py", "--release"])

    mod.main()  # cargo is 9.9.9

    assert 'build-plugin = ["dead-cst-plugin-host==9.9.9"]' in pyproject.read_text()
