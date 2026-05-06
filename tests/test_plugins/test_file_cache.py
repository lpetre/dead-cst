"""Tests for the per-base ``PluginContext`` surface: ``parse``, ``importers``,
``base_modules``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
import networkx as nx

from dead_cst import Analysis
from dead_cst.resolvers import ManualResolver
from dead_cst.plugins import GraphOp, ObserveContext, PluginContext
from dead_cst.graph import SymbolTrie


def _ctx(tmp_path):
    return PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        base=tmp_path,
        project_root=tmp_path,
    )


def test_parse_memoizes_within_a_pass(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def f(): pass\n")
    ctx = _ctx(tmp_path)
    module = ctx.parse(p)
    assert isinstance(module, cst.Module)
    # Mutating the file afterwards doesn't affect the cached parse.
    p.write_text("def g(): pass\n")
    assert ctx.parse(p) is module


def test_parse_handles_syntax_error(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("def : pass\n")
    ctx = _ctx(tmp_path)
    assert ctx.parse(p) is None
    # Failure is also cached.
    assert ctx.parse(p) is None


def test_base_modules_only_yields_under_base(tmp_path, write_files):
    """``ctx.base_modules()`` filters to the current base, not the full graph."""
    write_files(
        {
            "a/pkg/__init__.py": "",
            "a/pkg/m.py": "def f(): pass",
            "b/pkg/__init__.py": "",
            "b/pkg/m.py": "def g(): pass",
        }
    )
    seen_per_base: dict[Path, set[str]] = {}

    @dataclass
    class _Capture:
        name: str = "capture"
        version: str = "1"

        def observe(self, ctx: ObserveContext):
            return None

        def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
            seen_per_base[ctx.base] = {p.name for p, _ in ctx.base_modules()}
            return ()

    Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["a", "b"])],
        plugins=[_Capture()],
    ).materialize_all()
    # Each base only sees its own files, even though the full graph
    # contains both bases' nodes by the time the second base runs.
    assert seen_per_base[tmp_path / "a"] == {"__init__.py", "m.py"}
    assert seen_per_base[tmp_path / "b"] == {"__init__.py", "m.py"}


def test_importers_finds_first_party_imports(tmp_path, write_files):
    """``ctx.importers(fqname)`` returns paths whose imports reach the target module."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def util(): pass",
            "pkg/uses_lib.py": "from pkg.lib import util\nutil()",
            "pkg/no_lib.py": "def f(): pass",
        }
    )
    seen: set[Path] = set()

    @dataclass
    class _Capture:
        name: str = "capture"
        version: str = "1"

        def observe(self, ctx: ObserveContext):
            return None

        def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
            seen.update(ctx.importers("pkg.lib"))
            return ()

    Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[_Capture()],
    ).materialize_all()
    assert {p.name for p in seen} == {"uses_lib.py"}


def test_importers_finds_third_party_dist(tmp_path, write_files):
    """``ctx.importers("typer")`` resolves to the synthetic external dep node."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/uses_typer.py": "import typer\ncli = typer.Typer()",
            "pkg/no_typer.py": "def f(): pass",
        }
    )
    seen: set[Path] = set()

    @dataclass
    class _Capture:
        name: str = "capture"
        version: str = "1"

        def observe(self, ctx: ObserveContext):
            return None

        def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
            seen.update(ctx.importers("typer"))
            return ()

    Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[_Capture()],
    ).materialize_all()
    assert {p.name for p in seen} == {"uses_typer.py"}


def test_importers_unknown_returns_empty(tmp_path, write_files):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    saw_empty = False

    @dataclass
    class _Capture:
        name: str = "capture"
        version: str = "1"

        def observe(self, ctx: ObserveContext):
            return None

        def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
            nonlocal saw_empty
            saw_empty = ctx.importers("definitely-not-a-module") == set()
            return ()

    Analysis(
        tmp_path,
        resolvers=[ManualResolver(specs=["."])],
        plugins=[_Capture()],
    ).materialize_all()
    assert saw_empty
