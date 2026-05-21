"""Tests for the per-package file partition built during materialize.

The partition is computed once after ty enumerates the project's
file set: each ``File`` is assigned to the package whose ``path``
prefix is the longest match. Files outside every owned path land
in ``unowned_files``.

The partition is the data structure the per-package edge pass
iterates over -- each iteration scopes the resolver to the
package's exports + deps and processes only that package's
bucket.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from dead_cst import _native as native


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src).strip() + "\n")


def test_files_partition_assigns_each_file_to_owning_package(tmp_path):
    """Two non-overlapping packages partition the file list cleanly:
    each file belongs to exactly one package, no overlap."""
    _write(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/m1.py": "x = 1",
            "pkg_a/m2.py": "y = 2",
            "pkg_b/__init__.py": "",
            "pkg_b/m1.py": "z = 3",
        },
    )
    a = tmp_path / "pkg_a"
    b = tmp_path / "pkg_b"
    ctx = native.ProjectContext(
        str(tmp_path),
        src_roots=[str(a), str(b)],
        package_owned_paths=[str(a), str(b)],
    )
    ctx.materialize()

    assert set(ctx.files_for_package(str(a))) == {
        str(a / "__init__.py"),
        str(a / "m1.py"),
        str(a / "m2.py"),
    }
    assert set(ctx.files_for_package(str(b))) == {
        str(b / "__init__.py"),
        str(b / "m1.py"),
    }


def test_files_partition_longest_prefix_wins_for_nested_owned_dirs(tmp_path):
    """When one package's owned dir sits inside another's, the
    nested package wins ownership for its own files.

    Realistic shape: monorepo with ``packages/lib`` and a nested
    subpackage ``packages/lib/internal`` that the resolver treats
    as its own ownership unit. A file in ``packages/lib/internal/x.py``
    must be owned by ``lib/internal``, not ``lib``.
    """
    _write(
        tmp_path,
        {
            "lib/__init__.py": "",
            "lib/top.py": "x = 1",
            "lib/internal/__init__.py": "",
            "lib/internal/deep.py": "y = 2",
        },
    )
    outer = tmp_path / "lib"
    inner = tmp_path / "lib" / "internal"
    ctx = native.ProjectContext(
        str(tmp_path),
        src_roots=[str(outer)],
        package_owned_paths=[str(outer), str(inner)],
    )
    ctx.materialize()

    assert set(ctx.files_for_package(str(outer))) == {
        str(outer / "__init__.py"),
        str(outer / "top.py"),
    }
    assert set(ctx.files_for_package(str(inner))) == {
        str(inner / "__init__.py"),
        str(inner / "deep.py"),
    }


def test_files_partition_unowned_bucket_catches_stray_files(tmp_path):
    """Files outside every owned path land in ``unowned_files``.
    Common shape: ``conftest.py`` at the project root, scripts in
    ``./scripts/`` that aren't anyone's owned dir.
    """
    _write(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/mod.py": "x = 1",
            "conftest.py": "",
            "scripts/build.py": "y = 2",
        },
    )
    a = tmp_path / "pkg_a"
    ctx = native.ProjectContext(
        str(tmp_path),
        src_roots=[str(a)],
        package_owned_paths=[str(a)],
    )
    ctx.materialize()

    assert set(ctx.files_for_package(str(a))) == {
        str(a / "__init__.py"),
        str(a / "mod.py"),
    }
    assert set(ctx.unowned_files()) == {
        str(tmp_path / "conftest.py"),
        str(tmp_path / "scripts" / "build.py"),
    }


def test_files_partition_no_owned_paths_puts_everything_in_unowned(tmp_path):
    """When no owned paths are passed (legacy shape), every file
    lands in ``unowned``. Pins that today's callers (which don't
    pass ``package_owned_paths``) keep working."""
    _write(tmp_path, {"m.py": "x = 1", "n.py": "y = 2"})
    ctx = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path)])
    ctx.materialize()

    assert set(ctx.unowned_files()) == {str(tmp_path / "m.py"), str(tmp_path / "n.py")}
    assert ctx.files_for_package(str(tmp_path / "nonexistent")) == []
