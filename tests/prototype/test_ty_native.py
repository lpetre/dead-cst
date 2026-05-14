"""Smoke tests for the experimental ty-backed native parser.

Skipped unless ``dead_cst_ty_native`` has been built and installed
(``maturin develop`` from ``crates/dead-cst-ty-native``). The native
crate is not part of the standard ``uv sync`` flow today -- it's a
prototype probing whether ty can replace stage 2 of the analyzer's
pipeline.

The prototype's ``Project`` takes injected configuration (root +
src_roots + extra_paths + python_env + python_version + typeshed)
instead of reading ``pyproject.toml`` / ``ty.toml`` from disk. This
matches dead-cst's ``Analysis(project_root, resolver=...)`` shape
where a ``PathResolver`` is the single source of truth for what's
first-party.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("dead_cst_ty_native")


@pytest.fixture
def project_factory(tmp_path: Path):
    """Write sources under tmp_path and return a Project with default config.

    Tests that need extra config (src_roots, extra_paths, python_env, ...)
    skip this fixture and build their Project directly.
    """

    def make(files: dict[str, str], **kwargs) -> tuple[native.Project, Path]:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.Project(str(tmp_path), **kwargs), tmp_path

    return make


def _decls(proj: native.Project, path: Path):
    return [
        (d.name, d.kind, (d.start_line, d.start_column), (d.end_line, d.end_column))
        for d in proj.extract_top_level_decls(str(path))
    ]


def test_function_and_class(project_factory):
    proj, root = project_factory({"mod.py": "def foo():\n    pass\n\nclass Bar:\n    pass\n"})
    assert _decls(proj, root / "mod.py") == [
        ("foo", "function", (1, 0), (2, 8)),
        ("Bar", "class", (4, 0), (5, 8)),
    ]


def test_async_function_is_function(project_factory):
    proj, root = project_factory({"mod.py": "async def go(): pass\n"})
    assert _decls(proj, root / "mod.py") == [("go", "function", (1, 0), (1, 20))]


def test_assignments(project_factory):
    proj, root = project_factory({"mod.py": "X = 1\nY: int = 2\nA, B = 3, 4\n"})
    assert _decls(proj, root / "mod.py") == [
        ("X", "variable", (1, 0), (1, 1)),
        ("Y", "variable", (2, 0), (2, 1)),
        ("A", "variable", (3, 0), (3, 1)),
        ("B", "variable", (3, 3), (3, 4)),
    ]


def test_positions_match_libcst(project_factory):
    """Cross-check ty/ruff-derived positions against libcst's PositionProvider."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider

    src = "def foo():\n    pass\n\nclass Bar:\n    def method(self):\n        return 1\n"
    proj, root = project_factory({"mod.py": src})

    wrapper = MetadataWrapper(cst.parse_module(src))
    pos = wrapper.resolve(PositionProvider)
    libcst_positions = {
        stmt.name.value: (
            (pos[stmt].start.line, pos[stmt].start.column),
            (pos[stmt].end.line, pos[stmt].end.column),
        )
        for stmt in wrapper.module.body
        if isinstance(stmt, (cst.FunctionDef, cst.ClassDef))
    }

    native_positions = {
        d.name: ((d.start_line, d.start_column), (d.end_line, d.end_column))
        for d in proj.extract_top_level_decls(str(root / "mod.py"))
        if d.kind in {"function", "class"}
    }

    assert native_positions == libcst_positions


def test_repeated_queries_reuse_db(project_factory):
    """Same project handle answers multiple queries (Salsa stays warm)."""
    proj, root = project_factory(
        {
            "a.py": "def fa(): pass\n",
            "b.py": "class CB: pass\n",
        }
    )
    assert _decls(proj, root / "a.py") == [("fa", "function", (1, 0), (1, 14))]
    assert _decls(proj, root / "b.py") == [("CB", "class", (1, 0), (1, 14))]
    # Second query against same file is served from Salsa's memoized parse.
    assert _decls(proj, root / "a.py") == [("fa", "function", (1, 0), (1, 14))]


def test_missing_file_raises(project_factory):
    proj, root = project_factory({})
    with pytest.raises(OSError):
        proj.extract_top_level_decls(str(root / "nope.py"))


def test_no_pyproject_required(tmp_path):
    """A bare directory (no pyproject.toml, no ty.toml) is a valid project."""
    (tmp_path / "mod.py").write_text("def f(): pass\n", encoding="utf-8")
    proj = native.Project(str(tmp_path))
    assert _decls(proj, tmp_path / "mod.py") == [("f", "function", (1, 0), (1, 13))]


def test_pyproject_in_root_is_ignored(tmp_path):
    """A pyproject.toml in root is not consulted (no discovery happens)."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ty.environment]\npython-version = "3.7"\n',
        encoding="utf-8",
    )
    (tmp_path / "mod.py").write_text("def f(): pass\n", encoding="utf-8")
    # If discovery were happening, the pyproject's `python-version = "3.7"`
    # would take effect. We assert it didn't by passing 3.13 explicitly and
    # confirming the file still parses (a no-op check, but if discovery
    # were silently overriding our injected value the call would fail when
    # we extend the prototype to surface version-conditional behavior).
    proj = native.Project(str(tmp_path), python_version="3.13")
    assert _decls(proj, tmp_path / "mod.py") == [("f", "function", (1, 0), (1, 13))]


def test_src_roots_injected(project_factory):
    """src_roots accepts a list of paths; equivalent to resolver Package.path."""
    proj, root = project_factory(
        {"pkg_a/mod.py": "def a(): pass\n", "pkg_b/mod.py": "def b(): pass\n"},
        src_roots=["pkg_a", "pkg_b"],
    )
    assert _decls(proj, root / "pkg_a/mod.py") == [("a", "function", (1, 0), (1, 13))]
    assert _decls(proj, root / "pkg_b/mod.py") == [("b", "function", (1, 0), (1, 13))]


def test_invalid_python_version_raises(tmp_path):
    with pytest.raises(ValueError, match="invalid python_version"):
        native.Project(str(tmp_path), python_version="not-a-version")
