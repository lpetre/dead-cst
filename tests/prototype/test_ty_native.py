"""Smoke tests for the experimental ruff-backed native parser.

Skipped unless ``dead_cst_ty_native`` has been built and installed
(``maturin develop`` from ``crates/dead-cst-ty-native``). The native
crate is not part of the standard ``uv sync`` flow today -- it's a
prototype probing whether ruff/ty can replace stage 2 of the
analyzer's pipeline.
"""

from __future__ import annotations

import pytest

native = pytest.importorskip("dead_cst_ty_native")


def _decls(source: str) -> list[tuple[str, str, tuple[int, int], tuple[int, int]]]:
    return [
        (d.name, d.kind, (d.start_line, d.start_column), (d.end_line, d.end_column))
        for d in native.extract_top_level_decls(source)
    ]


def test_function_and_class():
    src = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
    assert _decls(src) == [
        ("foo", "function", (1, 0), (2, 8)),
        ("Bar", "class", (4, 0), (5, 8)),
    ]


def test_async_function_is_function():
    assert _decls("async def go(): pass\n") == [
        ("go", "function", (1, 0), (1, 20)),
    ]


def test_assignments():
    src = "X = 1\nY: int = 2\nA, B = 3, 4\n"
    assert _decls(src) == [
        ("X", "variable", (1, 0), (1, 1)),
        ("Y", "variable", (2, 0), (2, 1)),
        ("A", "variable", (3, 0), (3, 1)),
        ("B", "variable", (3, 3), (3, 4)),
    ]


def test_positions_match_libcst():
    """Cross-check ruff-derived positions against libcst's PositionProvider."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider

    src = "def foo():\n    pass\n\nclass Bar:\n    def method(self):\n        return 1\n"

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
        for d in native.extract_top_level_decls(src)
        if d.kind in {"function", "class"}
    }

    assert native_positions == libcst_positions


def test_parse_error_raises():
    with pytest.raises(ValueError, match="parse error"):
        native.extract_top_level_decls("def foo(:\n")
