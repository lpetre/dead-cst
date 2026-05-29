#!/usr/bin/env python3
"""Stamp dead-cst's version into every artifact that must share it.

``dead-cst`` and ``dead-cst-plugin-host`` must carry the *identical* version.
A native plugin bakes an ABI fingerprint (``rustc`` commit + runtime version)
at compile time, and ``native.load_native_plugins`` refuses to load a plugin
whose runtime version differs from the running host's. The two packages are
built in *separate* CI jobs (the host wheel via maturin, the plugin-host wheel
via ``dead-cst bundle-plugin-host``), so the only safe contract is a single
deterministic source of truth for the version — this script, run identically
in every job.

The resolved version depends on the event:

* **push to main** -> a dev version from ``setuptools-scm`` (``<next>.devN``,
  deterministic for a given commit), written into ``Cargo.toml`` (maturin's
  version source) in Cargo's SemVer spelling (``<next>-devN``; maturin
  normalizes it back to PEP 440).
* **release / tag** -> the version already committed in ``Cargo.toml`` is the
  source of truth and is left untouched.

Either way the resolved PEP 440 version is mirrored into
``plugin-host/pyproject.toml`` and pinned into the ``dead-cst[build-plugin]``
extra (``dead-cst-plugin-host == <version>``), and printed on stdout so the
caller can capture it.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARGO = ROOT / "Cargo.toml"
PYPROJECT = ROOT / "pyproject.toml"
PLUGIN_HOST_PYPROJECT = ROOT / "plugin-host" / "pyproject.toml"


def _replace_version_line(path: Path, version: str, *, what: str) -> None:
    """Rewrite the first top-level ``version = "..."`` line in ``path``.

    Top-level (column-0) ``version`` keys only appear under the leading
    ``[package]`` / ``[project]`` table; dependency versions are nested, so a
    first-match replace is safe and preserves comments + formatting.
    """
    text = path.read_text()
    new, n = re.subn(r'(?m)^version = "[^"]*"', f'version = "{version}"', text, count=1)
    if n != 1:
        raise SystemExit(f"failed to stamp {what}: no top-level version line in {path}")
    path.write_text(new)


def _pin_extra(version: str) -> None:
    """Pin ``dead-cst-plugin-host`` to ``== version`` in the build-plugin extra."""
    text = PYPROJECT.read_text()
    new, n = re.subn(
        r'(?m)^(build-plugin = \[")dead-cst-plugin-host[^"]*("\])',
        rf"\g<1>dead-cst-plugin-host=={version}\g<2>",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit("failed to pin dead-cst-plugin-host in the [build-plugin] extra")
    PYPROJECT.write_text(new)


def _cargo_version() -> str:
    # The first top-level `version = "..."` line is `[package].version`
    # ([workspace] carries no version key); avoid a tomllib dep so the script
    # runs on any runner Python.
    m = re.search(r'(?m)^version = "([^"]*)"', CARGO.read_text())
    if m is None:
        raise SystemExit("could not find [package].version in Cargo.toml")
    return m.group(1)


def _dev_version() -> str:
    from setuptools_scm import get_version

    return get_version(root=str(ROOT), local_scheme="no-local-version")


def _is_dev_stamp() -> bool:
    if "--dev" in sys.argv[1:]:
        return True
    if "--release" in sys.argv[1:]:
        return False
    # Auto-detect: a push to main gets a fresh dev version; everything else
    # (release, tag, local run) uses the committed Cargo.toml version.
    return (
        os.environ.get("GITHUB_EVENT_NAME") == "push"
        and os.environ.get("GITHUB_REF") == "refs/heads/main"
    )


def main() -> None:
    if _is_dev_stamp():
        pep440 = _dev_version()  # e.g. 0.13.1.dev5
        cargo = pep440.replace(".dev", "-dev")  # 0.13.1-dev5 (Cargo SemVer)
        _replace_version_line(CARGO, cargo, what="Cargo.toml [package].version")
    else:
        pep440 = _cargo_version().replace("-dev", ".dev")

    _replace_version_line(PLUGIN_HOST_PYPROJECT, pep440, what="plugin-host version")
    _pin_extra(pep440)
    print(pep440)


if __name__ == "__main__":
    main()
