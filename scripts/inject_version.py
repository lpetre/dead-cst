"""Stamp ``native/Cargo.toml`` with a setuptools-scm-style version.

Maturin reads the package version from ``native/Cargo.toml``'s
``[package].version`` field. We don't hand-edit that field; instead
CI runs this script before each build to compute the version from
git history.

Output forms:

* **On a tag** (``v1.2.3``): ``1.2.3``.
* **Between tags**: ``<last-tag>.dev<N>`` where ``N`` is the number
  of commits since the tag. PEP 440 dev versions sort correctly on
  PyPI and TestPyPI, both of which reject local ``+gSHA`` suffixes.
* **No tags yet**: ``0.0.0.dev<commit-count>``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parent.parent / "native" / "Cargo.toml"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def compute_version() -> str:
    try:
        latest_tag = _git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
    except subprocess.CalledProcessError:
        # No matching tag in history.
        total = _git("rev-list", "--count", "HEAD")
        return f"0.0.0.dev{total}"

    base = latest_tag.lstrip("v")
    distance = int(_git("rev-list", "--count", f"{latest_tag}..HEAD"))
    if distance == 0:
        return base
    return f"{base}.dev{distance}"


def stamp(version: str) -> None:
    text = CARGO_TOML.read_text()
    new = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if new == text:
        raise SystemExit(f"failed to find version line in {CARGO_TOML}")
    CARGO_TOML.write_text(new)


def main() -> int:
    version = compute_version()
    stamp(version)
    print(f"injected version={version} into {CARGO_TOML.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
