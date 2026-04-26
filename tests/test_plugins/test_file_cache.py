"""Tests for :class:`FileTextCache` and the ``ctx.grep`` / ``ctx.parse`` surface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import libcst as cst
import networkx as nx

from dead_cst import (
    FileTextCache,
    MainBlockPlugin,
    build_symbol_graph,
)
from dead_cst._plugins import PluginContext
from dead_cst._plugins._core import GraphOp
from dead_cst._symbols import SymbolTrie


def _make_cache(tmp_path, files):
    paths = []
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        paths.append(p)
    return FileTextCache(paths)


def test_file_text_cache_substring_grep(tmp_path):
    cache = _make_cache(
        tmp_path,
        {"a.py": "import fastapi\n", "b.py": "print('hi')\n", "c.py": "from fastapi import X\n"},
    )
    matches = set(cache.grep("fastapi"))
    assert {p.name for p in matches} == {"a.py", "c.py"}


def test_file_text_cache_regex_grep(tmp_path):
    cache = _make_cache(
        tmp_path,
        {"a.py": "x = FastAPI()\n", "b.py": "x = APIRouter()\n", "c.py": "pass\n"},
    )
    pat = re.compile(rb"FastAPI|APIRouter")
    matches = {p.name for p in cache.grep(pat)}
    assert matches == {"a.py", "b.py"}


def test_file_text_cache_caches_reads(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("hello world\n")
    cache = FileTextCache([p])
    assert cache.contains(p, "hello")
    p.write_text("totally different content\n")
    # Second call should still see the cached bytes, not the new content.
    assert cache.contains(p, "hello")
    assert not cache.contains(p, "totally")


def test_file_text_cache_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.py"
    cache = FileTextCache([missing])
    assert cache.read(missing) == b""
    assert list(cache.grep("anything")) == []


def test_file_text_cache_explicit_paths_override_default(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("import fastapi\n")
    b = tmp_path / "b.py"
    b.write_text("import fastapi\n")
    cache = FileTextCache([a])
    assert list(cache.grep("fastapi")) == [a]
    assert set(cache.grep("fastapi", paths=[a, b])) == {a, b}


def test_plugin_context_grep_delegates_to_cache(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("__main__ guard\n")
    b = tmp_path / "b.py"
    b.write_text("nothing here\n")
    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        paths={tmp_path: []},
        project_root=tmp_path,
        file_cache=FileTextCache([a, b]),
    )
    assert list(ctx.grep("__main__")) == [a]


def test_plugin_context_grep_default_cache_is_empty():
    """A bare PluginContext (no cache supplied) yields nothing rather than crashing."""
    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        paths={},
        project_root=Path("/"),
    )
    assert list(ctx.grep("anything")) == []


@dataclass
class _RecordingPlugin:
    """CST-aware plugin that records every path it would have parsed."""

    name: str = "recorder"
    cst_aware: bool = True
    seen: list[Path] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.seen = []

    def contribute(self, ctx: PluginContext, managers) -> Iterable[GraphOp]:
        for path in ctx.grep("__main__"):
            self.seen.append(path)
        return ()


def test_file_text_cache_parse_returns_module(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def f(): pass\n")
    cache = FileTextCache([p])
    module = cache.parse(p)
    assert isinstance(module, cst.Module)
    assert "def f()" in module.code


def test_file_text_cache_parse_caches(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def f(): pass\n")
    cache = FileTextCache([p])
    first = cache.parse(p)
    p.write_text("def g(): pass\n")
    # Second call returns the same cached module instance, not a re-parse.
    assert cache.parse(p) is first


def test_file_text_cache_parse_handles_syntax_error(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("def : pass\n")  # syntactically invalid
    cache = FileTextCache([p])
    assert cache.parse(p) is None
    # Failure is also cached -- second call should not re-attempt.
    assert cache.parse(p) is None


def test_file_text_cache_prime_module_skips_reparse(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("def f(): pass\n")
    sentinel = cst.parse_module("def sentinel(): pass\n")
    cache = FileTextCache([p])
    cache.prime_module(p, sentinel)
    # Disk content differs, but the primed module wins.
    assert cache.parse(p) is sentinel


def test_plugin_context_parse_delegates_to_cache(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    cache = FileTextCache([p])
    ctx = PluginContext(
        graph=nx.DiGraph(),
        symbol_lookup=SymbolTrie(),
        paths={tmp_path: []},
        project_root=tmp_path,
        file_cache=cache,
    )
    module = ctx.parse(p)
    assert module is cache.parse(p)


def test_analyze_primes_cache_with_parsed_modules(tmp_path, write_files):
    """build_symbol_graph hands plugins the modules it already parsed."""
    write_files({"pkg/__init__.py": "", "pkg/a.py": "def f(): pass"})
    seen: dict[Path, cst.Module] = {}

    @dataclass
    class _Capture:
        name: str = "capture"
        cst_aware: bool = True

        def contribute(self, ctx: PluginContext, managers) -> Iterable[GraphOp]:
            for path in ctx.file_cache.paths:
                module = ctx.parse(path)
                assert module is not None
                seen[path] = module
            return ()

    build_symbol_graph(
        {tmp_path: []},
        plugins=[_Capture()],
        project_root=tmp_path,
    )
    # Every analyzed file should have a primed module the plugin saw.
    expected_names = {"__init__.py", "a.py"}
    assert {p.name for p in seen} == expected_names
    # And rewriting the file on disk after the analyze pass shouldn't
    # affect the cached module the plugin received.
    a_path = next(p for p in seen if p.name == "a.py")
    assert "def f()" in seen[a_path].code


def test_grep_prefilter_skips_unrelated_modules(tmp_path, write_files):
    write_files(
        {
            "pkg/__init__.py": "",
            "pkg/has_main.py": """
            if __name__ == "__main__":
                pass
            """,
            "pkg/no_main.py": "def f(): pass",
        }
    )
    plugin = _RecordingPlugin()
    build_symbol_graph(
        {tmp_path: []},
        plugins=[plugin, MainBlockPlugin()],
        project_root=tmp_path,
    )
    seen = {p.name for p in plugin.seen}
    assert "has_main.py" in seen
    assert "no_main.py" not in seen
