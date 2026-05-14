"""Smoke tests for the experimental ty-backed native parser.

Skipped unless ``dead_cst_ty_native`` has been built and installed
(``maturin develop`` from ``crates/dead-cst-ty-native``). The native
crate is not part of the standard ``uv sync`` flow today -- it's a
prototype probing whether ty can replace stage 2 of the analyzer's
pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("dead_cst_ty_native")


@pytest.fixture
def project_factory(tmp_path: Path):
    """Write a minimal project rooted at tmp_path and return a Project.

    Each call materializes one file at ``tmp_path/<name>`` and writes a
    bare pyproject.toml so ProjectMetadata::discover finds the root.
    """

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "smoke"\nversion = "0"\n', encoding="utf-8"
    )

    def make(files: dict[str, str]) -> tuple[native.Project, Path]:
        for relpath, source in files.items():
            target = tmp_path / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        return native.Project(str(tmp_path)), tmp_path

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


def test_discovery_failure_raises(tmp_path):
    with pytest.raises(ValueError, match="project discovery failed"):
        native.Project(str(tmp_path / "does-not-exist"))
