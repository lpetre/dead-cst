"""Tests for :meth:`native.ProjectContext.set_src_roots`.

The setter is the foundation of the per-package pipeline design. It
mutates ``Program::search_paths`` on the live Salsa db so the parse +
``semantic_index`` queries (keyed on ``File``) survive an env swap,
while module-resolution answers (keyed on the env) re-execute.

Subtlety pinned by these tests: ``environment.root`` controls
*classification* (first-party vs third-party) and ``file_to_module``
fqname derivation -- it does NOT control file enumeration. ty walks
the project root regardless and applies include / exclude filters.
That means per-package scoping doesn't shrink the file walk; it
narrows which targets the resolver finds first-party and how dotted
names get derived.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst import _native as native
from dead_cst.plugins import SYNTHETIC_PATH_PREFIXES, UNRESOLVED_PREFIX


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def test_set_src_roots_smoke(tmp_path):
    """The setter accepts a new root list and doesn't crash. Sanity
    floor for the wiring -- if this fails, the Program / Project
    setters or the make_metadata rebuild are broken."""
    _write(tmp_path, {"a/__init__.py": "", "a/m.py": "x = 1"})
    ctx = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path / "a")])
    ctx.set_src_roots([str(tmp_path / "a")])  # no-op swap
    ctx.set_src_roots([str(tmp_path)])  # widen
    ctx.set_src_roots([str(tmp_path / "a")])  # narrow again


def test_set_src_roots_changes_fqname_derivation(tmp_path):
    """A file's dotted name comes from stripping the matching
    ``environment.root`` prefix. Same file, two roots -> two different
    module fqnames in the resulting graph.

    This is the contract that lets per-package iteration mount each
    package's files at the right top-level name: when the active root
    is ``packages/A/src``, ``packages/A/src/A/mod.py`` is ``A.mod``;
    when the active root is ``packages/``, the same file is
    ``A.src.A.mod``.
    """
    _write(tmp_path, {"pkg/sub/m.py": "x = 1"})

    narrow = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path / "pkg")])
    narrow.materialize()
    assert "sub.m" in {n.fqname for n in narrow.nodes() if n.kind == "module"}

    wide = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path)])
    wide.materialize()
    assert "pkg.sub.m" in {n.fqname for n in wide.nodes() if n.kind == "module"}


def test_set_src_roots_changes_import_classification(tmp_path):
    """A cross-package import classifies as first-party iff the
    target's dir is in ``environment.root``. With both pkgs as roots
    the import resolves first-party; with only one pkg's root, the
    target lands on a synthetic ``[unresolved]`` / ``[external]``
    node.
    """
    _write(
        tmp_path,
        {
            "app/main.py": "from lib_mod import value",
            "lib/lib_mod.py": "value = 1",
        },
    )

    wide = native.ProjectContext(
        str(tmp_path), src_roots=[str(tmp_path / "app"), str(tmp_path / "lib")]
    )
    wide.materialize()
    wide_imports = [n for n in wide.nodes() if n.kind == "import" and "main.py" in n.path]
    assert {n.imports.module for n in wide_imports if n.imports is not None} == {"lib_mod"}

    narrow = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path / "app")])
    narrow.materialize()
    # ``from lib_mod import value`` lands on a synthetic node prefixed
    # with one of the documented ``SYNTHETIC_PATH_PREFIXES`` -- in this
    # venv-less tmp, the ``[unresolved] lib_mod`` synthetic specifically.
    synthetic_fqs = {
        n.fqname
        for n in narrow.nodes()
        if n.kind == "synthetic" and any(n.fqname.startswith(p) for p in SYNTHETIC_PATH_PREFIXES)
    }
    assert f"{UNRESOLVED_PREFIX}lib_mod" in synthetic_fqs
