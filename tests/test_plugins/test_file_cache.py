"""Tests for the per-package ``PluginContext`` surface: ``parse``,
``contribution.nodes``, and the analyzer-side ``PackageView.importers_of``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst

from dead_cst._graphstore import SymbolGraph
from dead_cst.graph import SymbolTrie
from dead_cst.plugins import GraphOp, ObserveContext, PluginContext
from dead_cst.resolvers import Package


def _ctx(tmp_path, make_contribution):
    return PluginContext(
        graph=SymbolGraph(),
        symbol_lookup=SymbolTrie(),
        contribution=make_contribution(Package(path=tmp_path, name="pkg")),
        project_root=tmp_path,
    )


def test_parse_memoizes_within_a_pass(tmp_path, make_contribution):
    p = tmp_path / "a.py"
    p.write_text("def f(): pass\n")
    ctx = _ctx(tmp_path, make_contribution)
    module = ctx.parse(p)
    assert isinstance(module, cst.Module)
    # Mutating the file afterwards doesn't affect the cached parse.
    p.write_text("def g(): pass\n")
    assert ctx.parse(p) is module


def test_parse_handles_syntax_error(tmp_path, make_contribution):
    p = tmp_path / "broken.py"
    p.write_text("def : pass\n")
    ctx = _ctx(tmp_path, make_contribution)
    assert ctx.parse(p) is None
    # Failure is also cached.
    assert ctx.parse(p) is None


def test_package_nodes_only_yields_under_package(tmp_path, make_analysis, write_files):
    """``ctx.contribution.nodes`` filters to the current package, not the full graph."""
    write_files(
        {
            "a/pkg/__init__.py": "",
            "a/pkg/m.py": "def f(): pass",
            "b/pkg/__init__.py": "",
            "b/pkg/m.py": "def g(): pass",
        }
    )
    seen_per_package: dict[Path, set[str]] = {}

    @dataclass
    class _Capture:
        name: str = "capture"
        version: str = "1"

        def observe(self, ctx: ObserveContext):
            return None

        def finalize(self, ctx: PluginContext) -> Iterable[GraphOp]:
            seen_per_package[ctx.contribution.package.path] = {
                n.path.name for n in ctx.contribution.nodes if n.type == "module"
            }
            return ()

    make_analysis(["a", "b"], plugins=[_Capture()]).materialize_all()
    assert seen_per_package[tmp_path / "a"] == {"__init__.py", "m.py"}
    assert seen_per_package[tmp_path / "b"] == {"__init__.py", "m.py"}


def test_importers_of_finds_first_party_imports(make_analysis, write_files, tmp_path):
    """``PackageView.importers_of(fqname)`` returns paths whose imports reach the target."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": "def util(): pass",
            "pkg/uses_lib.py": "from pkg.lib import util\nutil()",
            "pkg/no_lib.py": "def f(): pass",
        }
    )
    seen = make_analysis().package(tmp_path).importers_of("pkg.lib")
    assert {p.name for p in seen} == {"uses_lib.py"}


def test_importers_of_finds_third_party_dist(make_analysis, write_files, tmp_path):
    """``PackageView.importers_of("typer")`` resolves to the synthetic external dep node."""
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/uses_typer.py": "import typer\ncli = typer.Typer()",
            "pkg/no_typer.py": "def f(): pass",
        }
    )
    seen = make_analysis().package(tmp_path).importers_of("typer")
    assert {p.name for p in seen} == {"uses_typer.py"}


def test_importers_of_unknown_returns_empty(make_analysis, write_files, tmp_path):
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    assert make_analysis().package(tmp_path).importers_of("definitely-not-a-module") == set()
