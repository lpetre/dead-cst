# Contributing to dead-cst

Thanks for your interest. `dead-cst` is small and pre-release — bug
reports with minimal repros, real-world test cases, and PRs are all
welcome. Expect APIs and CLI flags to keep moving until the first
stable release. See [`ROADMAP.md`](ROADMAP.md) for the planned
trajectory.

## Development setup

`dead-cst` uses [uv](https://github.com/astral-sh/uv) and
[maturin](https://github.com/PyO3/maturin) — the rust extension under
`src/` builds into `python/dead_cst/_native.{abi3.so,pyd}` on
`uv sync`. You need a Rust toolchain (`rustup`); everything else (uv,
pytest, ruff, prek, ty, maturin) is installed by `uv sync`.

```bash
git clone https://github.com/lpetre/dead-cst
cd dead-cst
uv sync
```

## Running tests

```bash
uv run pytest                          # full suite (e2e is deselected by default)
uv run pytest tests/test_imports.py    # one file
uv run pytest -k name_substring        # one test by name
uv run pytest -m e2e                   # opt-in e2e suite (clones real GitHub repos)
uv run pytest --cov=dead_cst --cov-branch --cov-report=term-missing
uv run ptw                             # pytest-watcher for a tight inner loop
```

`pyproject.toml` pins `addopts = "-m 'not e2e'"`, so the standard
`uv run pytest` is hermetic. CI runs the matrix `pytest` on Python
3.11–3.14 plus `prek run --all-files` on every push and pull request.

## Linting, formatting, type-checking

Ruff (lint + format), `ty` (type check), and a small set of hooks run
on every commit via [prek](https://github.com/j178/prek), a
Rust-based drop-in replacement for `pre-commit`:

```bash
uv run prek install        # one-time, sets up the git hook
uv run prek run --all-files
```

`ty` type-checks `dead_cst/` only — tests, examples, and workspace
fixtures intentionally exercise untyped third-party internals.

## Project layout

```
src/lib.rs            # thin pyo3 cdylib shim -> dead_cst._native (builds via maturin)
runtime/              # dead-cst-runtime crate: the whole impl, built as rlib + dylib
  src/lib.rs          # register() (the pymodule body) + top-level query() helper
  src/project.rs      # Project / ProjectContext / build() pipeline
  src/builder.rs      # GraphBuilder, AddNode / AddEdge / AddEntrypoint, BFS
  src/graph.rs        # SymbolNode / Import / NativeGraph / NodeFlags / EdgeFlags
  src/ingest.rs       # the three build phases (decls / chain / references)
  src/query.rs        # plugin-facing chainable query builder
  src/native_plugins.rs  # in-tree + external native plugins (plugin_api, ABI airlock)
  src/helpers.rs      # noqa parser, notebook decoder, dist-info lookup, …
  src/io.rs           # write_graph / read_graph (bincode + versioned header)
examples/main_block_plugin/  # worked example external native plugin
plugin-host/          # dead-cst-plugin-host package: the `[build-plugin]` extra payload
python/dead_cst/      # Python source tree (ships alongside _native.so in the wheel)
  __init__.py         # public API: Analysis, SymbolNode, Import, NodeFlags, EdgeFlags
  analyze.py          # Analysis (wraps the rust ProjectContext for BFS queries)
  cli.py              # build / analyze / remove + builtin plugin & resolver maps
  codemod.py          # remove_code, generate_patch (only stage still on libcst)
  graph.py            # graph data types + write_graph / read_graph / LoadedGraph
  _native.pyi         # hand-written type stubs for the rust extension
  plugins/            # generic-Python plugins (main_block, project_scripts, …)
  contrib/            # framework-aware plugins + UvResolver
  resolvers/          # PathResolver, ManualResolver
tests/                # fixture-driven pytest suite (e2e under tests/e2e/)
```

Modules whose name starts with `_` are internal. The supported surface
is the names re-exported from `dead_cst/__init__.py` plus the focused
public submodules listed in `python/dead_cst/__init__.py`'s docstring.
`tests/test_public_api.py` pins each module's `__all__` against a
snapshot.

## Adding a plugin

Define a class that subclasses `dead_cst.plugins.Plugin` and
implements `run(ctx) -> Iterable[GraphOp]`:

```python
from dead_cst.plugins import Plugin
from dead_cst._native import AddEntrypoint, query

class MyPlugin(Plugin):
    def run(self, ctx):
        for ref in query(ctx).decorators().where_module("myframework").collect():
            yield AddEntrypoint(ref.owner)
```

For common shapes, subclass one of the declarative bases in
`dead_cst/plugins/decl_shapes.py`:

* `DecoratedDeclPlugin` — decorated decls in files matching a search
  path.
* `LiteralListPlugin` — `<owner>.<var> = ['fqn', ...]` registries.

The `@<instance>.<reg>(...)` dispatch-app shape (Flask / FastAPI /
Typer / Cyclopts / Slack Bolt / FastMCP / Celery) is a native (Rust)
plugin — `NativePlugin.flask()` … `NativePlugin.celery()` — not a
Python base.

Drop generic-Python plugins in `dead_cst/plugins/<name>.py`
(re-exported from `dead_cst/plugins/__init__.py`); drop
framework-aware plugins in `dead_cst/contrib/<name>.py` (re-exported
from `dead_cst/contrib/__init__.py`). Register the CLI key in
`_BUILTIN_PLUGINS` in `dead_cst/cli.py`, importing the plugin
directly.

Out-of-tree plugins register under the `dead_cst.plugins`
entry-point group:

```toml
[project.entry-points."dead_cst.plugins"]
my_plugin = "myproj.plugins:MyPlugin"
```

The CLI's `_load_plugin` checks `_BUILTIN_PLUGINS` first, then the
entry-point group.

The reusable `DecoratedDeclPlugin` / `LiteralListPlugin` shapes in
`dead_cst/plugins/decl_shapes.py` are the reference scaffolding for a
Python framework plugin (decorated-decl → handler edges, or
`<owner>.<var> = [...]` literal lists). Every in-tree framework plugin
is now native — Click and the dispatch-app frameworks (Flask / FastAPI
/ Typer / Cyclopts / Slack Bolt / FastMCP / Celery), plus pytest,
mock_patch, and discordpy — see `ClickPluginImpl`,
`DispatchAppPluginImpl`, and friends in `runtime/src/native_plugins.rs`.

### Native plugins

For hot logic — or plugins that want their own salsa-cached queries over ty's
database — there's a Rust path: an external native plugin compiled against the
runtime `dylib` and loaded via `native.load_native_plugins(...)`. It's a
preview (macOS only) and a bigger commitment (a pinned Rust toolchain,
recompile per release). See **[`NATIVE_PLUGINS.md`](NATIVE_PLUGINS.md)** and the
worked **`examples/main_block_plugin/`**. Prefer a Python plugin unless you
specifically need native speed or plugin-defined salsa queries.

## Adding a resolver

Implement `PathResolver`: a `resolve(project_root) -> tuple[Package, ...]`
method. Drop generic resolvers in `dead_cst/resolvers/<name>.py`
(re-exported from `dead_cst/resolvers/__init__.py`); resolvers that
target a specific external tool (like `UvResolver` for `uv.lock`)
belong in `dead_cst/contrib/<name>.py` (re-exported from
`dead_cst/contrib/__init__.py`). Register the CLI key in
`_BUILTIN_RESOLVERS` in `dead_cst/cli.py`.

## Adding a test

Tests are fixture-driven from inline source snippets — no checked-in
`.py` fixture files under `tests/`. The `build_decl_graph` fixture
(in `tests/conftest.py`) writes a `{filename: source}` dict to a
tmpdir, runs `Analysis(...).materialize_all()` on it, and returns the
resulting `ProjectContext`. The `assert_edges` family compares edges
as `"src.fqname -> dst.fqname"` strings.

```python
def test_something(build_decl_graph, assert_edges):
    graph = build_decl_graph({"mod.py": "def a(): pass\na()"})
    assert_edges(graph, {"mod.a -> mod", "mod -> mod.a"})
```

Plugin tests live in `tests/test_plugins/` with their own
`conftest.py`; resolver tests in `tests/test_resolvers/`. Codemod
tests (`tests/test_codemod.py`) write a snippet, call `remove_code`,
and assert on the rewritten text. E2E tests (`tests/e2e/`) shallow-
clone real repos at pinned SHAs behind the `e2e` marker (deselected
by default).

## Reporting bugs

A good bug report contains a minimal `.py` file (or pair of files)
and the entrypoint flag you ran, plus the actual vs. expected dead-
symbol output. The smaller the repro, the faster it gets fixed.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Add or update tests for behaviour changes.
- Run `prek run --all-files` and `pytest` before pushing.
- If your change is user-visible, add an entry to `CHANGELOG.md`
  under `[Unreleased]`.

## Releasing

The version is read from `Cargo.toml`'s `[package].version` by maturin
(`pyproject.toml` is `dynamic = ["version"]`). Bump that field by
hand before tagging a release. On every push to `main` the publish
workflow rewrites the version to `<base>-dev${{ github.run_number }}`
in-CI (never committed) so TestPyPI gets a unique version per commit;
the SemVer pre-release suffix is normalized by maturin to PEP 440
`<base>.dev<N>` for the wheel.

Publishing a GitHub Release with a `vX.Y.Z` tag triggers
`.github/workflows/publish.yml`, which builds wheels + sdist with
`uv build` / `maturin`, publishes to PyPI via OIDC, and attaches the
artifacts to the release.
