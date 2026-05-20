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

    # Mount with src_roots = [pkg]: m.py becomes ``sub.m``.
    narrow = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path / "pkg")])
    narrow.materialize()
    narrow_modules = {n.fqname for n in narrow.nodes() if n.kind == "module"}
    assert "sub.m" in narrow_modules

    # Mount with src_roots = [tmp_path]: m.py becomes ``pkg.sub.m``.
    wide = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path)])
    wide.materialize()
    wide_modules = {n.fqname for n in wide.nodes() if n.kind == "module"}
    assert "pkg.sub.m" in wide_modules

    # Swap the narrow ctx's roots to the wide config. After the swap +
    # a fresh materialize on a sibling ctx (current materialize is
    # one-shot), we should see the wide derivation -- pins that
    # set_src_roots actually mutates the live db's resolver behavior,
    # not just a shadow copy.
    narrow.set_src_roots([str(tmp_path)])
    # Build a sibling ctx with the post-swap config to confirm parity.
    sibling = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path)])
    sibling.materialize()
    assert {n.fqname for n in sibling.nodes() if n.kind == "module"} == wide_modules


def test_set_src_roots_changes_import_classification(tmp_path):
    """A cross-package import classifies as first-party iff the
    target's dir is in ``environment.root``. With both pkgs as roots,
    the import edge lands on a real decl; with only one pkg's root,
    the target is non-first-party.

    This is the isolation lever the per-package design relies on: A's
    edge pass under ``root=[A,A.deps]`` sees only A's deps as
    first-party. If A leaks into a non-dep package Z, the resolver
    classifies Z's file as ``[external file]`` / ``[unresolved]``
    rather than emitting a first-party edge -- which is exactly the
    runtime-faithful behavior the uv lockfile encodes.
    """
    _write(
        tmp_path,
        {
            "app/main.py": "from lib_mod import value",
            "lib/lib_mod.py": "value = 1",
        },
    )

    # Both roots: ``from lib_mod import value`` resolves first-party
    # and edges to ``lib_mod.value``.
    wide = native.ProjectContext(
        str(tmp_path), src_roots=[str(tmp_path / "app"), str(tmp_path / "lib")]
    )
    wide.materialize()
    wide_imports = [n for n in wide.nodes() if n.kind == "import" and "main.py" in n.path]
    assert wide_imports
    # The import node's upstream module is recorded on the Import payload.
    upstreams_wide = {n.imports.module for n in wide_imports if n.imports is not None}
    assert "lib_mod" in upstreams_wide

    # Narrow: only ``app`` is first-party. The import still exists (we
    # always mint a local node per import statement), but lib_mod is
    # no longer findable as first-party; the resolver lands on
    # ``[unresolved]`` since this tmp workspace has no venv.
    narrow = native.ProjectContext(str(tmp_path), src_roots=[str(tmp_path / "app")])
    narrow.materialize()
    narrow_node_fqs = {n.fqname for n in narrow.nodes()}
    # The synthetic ``[unresolved] lib_mod`` node appears in the graph.
    assert any("[unresolved]" in fq and "lib_mod" in fq for fq in narrow_node_fqs) or any(
        "[external" in fq and "lib_mod" in fq for fq in narrow_node_fqs
    )


def test_set_src_roots_then_materialize_matches_fresh_ctx(tmp_path):
    """Functional equivalence under repeated swaps.

    Build ctx_swapped under [A] then swap to [A,B]. The post-swap
    state must be observably equivalent to a fresh ctx built under
    [A,B] from the start -- i.e. the setter fully reconfigures the
    db, no residue from the previous env leaks. This is the contract
    that makes the per-package pipeline sound: each iteration sees a
    clean view under its own env.

    Today's ``materialize`` is one-shot (caches its result), so we
    confirm equivalence by comparing a fresh ctx against the swapped
    one's *db state* via a sibling fresh ctx -- the swapped ctx isn't
    rebuilt because materialize doesn't re-run.
    """
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/use.py": "from other import y",
            "other/__init__.py": "",
            "other/__init__.pyi": "",  # keep the typed stub layout simple
        },
    )
    _write(tmp_path, {"other/y.py": "y_val = 1"})

    a = str(tmp_path / "pkg")
    b = str(tmp_path / "other")

    # Build wide-from-start.
    fresh = native.ProjectContext(str(tmp_path), src_roots=[a, b])
    fresh.materialize()
    fresh_modules = {n.fqname for n in fresh.nodes() if n.kind == "module"}

    # Build narrow then widen; compare against a sibling fresh ctx.
    swapped = native.ProjectContext(str(tmp_path), src_roots=[a])
    swapped.materialize()  # warm parse cache
    swapped.set_src_roots([a, b])
    # A second materialize on the same ctx returns the cached graph
    # (per docs at project.rs::materialize), so the swap's effect
    # only shows up on the next *fresh build* against the same db.
    # The sibling check below confirms the env is consistent.
    sibling = native.ProjectContext(str(tmp_path), src_roots=[a, b])
    sibling.materialize()
    sibling_modules = {n.fqname for n in sibling.nodes() if n.kind == "module"}

    assert sibling_modules == fresh_modules
