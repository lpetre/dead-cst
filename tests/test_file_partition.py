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
    a = str(tmp_path / "pkg_a")
    b = str(tmp_path / "pkg_b")
    ctx = native.ProjectContext(
        str(tmp_path),
        src_roots=[a, b],
        package_owned_paths=[a, b],
    )
    ctx.materialize()

    a_files = set(ctx.files_for_package(a))
    b_files = set(ctx.files_for_package(b))

    # Every pkg_a file is in A's bucket, no pkg_b leak.
    assert any(p.endswith("pkg_a/m1.py") for p in a_files)
    assert any(p.endswith("pkg_a/m2.py") for p in a_files)
    assert any(p.endswith("pkg_a/__init__.py") for p in a_files)
    assert not any("pkg_b" in p for p in a_files)

    # Same for pkg_b.
    assert any(p.endswith("pkg_b/m1.py") for p in b_files)
    assert any(p.endswith("pkg_b/__init__.py") for p in b_files)
    assert not any("pkg_a" in p for p in b_files)

    # Disjoint partitions.
    assert a_files.isdisjoint(b_files)


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
    outer = str(tmp_path / "lib")
    inner = str(tmp_path / "lib" / "internal")
    ctx = native.ProjectContext(
        str(tmp_path),
        src_roots=[outer],
        package_owned_paths=[outer, inner],
    )
    ctx.materialize()

    outer_files = set(ctx.files_for_package(outer))
    inner_files = set(ctx.files_for_package(inner))

    # The deep file belongs to the inner package (longest prefix), not
    # the outer one -- otherwise per-package processing would double-
    # count it or miss it under the wrong scope.
    assert any(p.endswith("lib/internal/deep.py") for p in inner_files)
    assert any(p.endswith("lib/internal/__init__.py") for p in inner_files)
    assert not any("internal" in p for p in outer_files)

    # The outer package keeps its own top-level files.
    assert any(p.endswith("lib/top.py") for p in outer_files)


def test_files_partition_unowned_bucket_catches_stray_files(tmp_path):
    """Files outside every owned path land in ``unowned_files``.
    Common shape: ``conftest.py`` at the project root, scripts in
    ``./scripts/`` that aren't anyone's owned dir, etc. The build
    still has to process them somewhere -- the catch-all pass uses
    the union of all exported paths.
    """
    _write(
        tmp_path,
        {
            "pkg_a/__init__.py": "",
            "pkg_a/mod.py": "x = 1",
            "conftest.py": "",  # root-level, no owner
            "scripts/build.py": "y = 2",  # not under pkg_a
        },
    )
    a = str(tmp_path / "pkg_a")
    ctx = native.ProjectContext(
        str(tmp_path),
        src_roots=[a],
        package_owned_paths=[a],
    )
    ctx.materialize()

    a_files = set(ctx.files_for_package(a))
    unowned = set(ctx.unowned_files())

    assert any(p.endswith("pkg_a/mod.py") for p in a_files)
    # The strays show up in the unowned bucket, NOT in pkg_a's.
    assert any(p.endswith("conftest.py") for p in unowned)
    assert any(p.endswith("scripts/build.py") for p in unowned)
    assert not any("conftest.py" in p for p in a_files)
    assert a_files.isdisjoint(unowned)


def test_files_partition_no_owned_paths_puts_everything_in_unowned(tmp_path):
    """Backward-compat: when no owned paths are passed (legacy or
    single-package shape), the partition is still computed -- every
    file lands in ``unowned``, no per-package buckets exist. Pins
    that today's callers (which don't pass ``package_owned_paths``)
    keep working without surprises."""
    _write(tmp_path, {"m.py": "x = 1", "n.py": "y = 2"})
    ctx = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path)])
    ctx.materialize()

    # No owned paths -> every project file is unowned.
    unowned = set(ctx.unowned_files())
    assert any(p.endswith("m.py") for p in unowned)
    assert any(p.endswith("n.py") for p in unowned)

    # Looking up an unknown owned path returns an empty list (no crash).
    assert ctx.files_for_package(str(tmp_path / "nonexistent")) == []
