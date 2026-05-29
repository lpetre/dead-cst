# dead-cst

[![PyPI](https://img.shields.io/pypi/v/dead-cst.svg)](https://pypi.org/project/dead-cst/)
[![Python](https://img.shields.io/pypi/pyversions/dead-cst.svg)](https://pypi.org/project/dead-cst/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/lpetre/dead-cst/actions/workflows/ci.yml/badge.svg)](https://github.com/lpetre/dead-cst/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lpetre/dead-cst/branch/main/graph/badge.svg)](https://codecov.io/gh/lpetre/dead-cst)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Python dead code analysis backed by a Rust extension built on top of [ty](https://github.com/astral-sh/ruff)'s `SemanticIndex` and [libcst](https://github.com/Instagram/LibCST) (codemod only).

`dead-cst` builds a full symbol graph of your Python codebase, walks from your entrypoints, and reports (or removes) anything unreachable.

> **Pre-release software.** `dead-cst` is in early alpha. APIs, CLI flags, and output formats may change without notice, and bugs are expected. `dead-cst remove` itself is non-destructive — it emits a patch to stdout — but apply the patch on a clean working tree so you can inspect (and easily revert) the result.

## Installation

```bash
pip install dead-cst
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add dead-cst
```

## Quick start

```bash
# Find dead code in your project
dead-cst analyze ./src --entrypoint-regex ".*__main__\.py"

# Generate a patch that removes dead code, then apply it
dead-cst remove ./src --entrypoint-regex ".*__main__\.py" | git apply
# Or, outside a git checkout: dead-cst remove ./src ... | patch -p1

# Drop the test suite and every helper it kept alive
dead-cst remove ./src --plugin pytest --query test-only | git apply

# Build the graph once, reuse it for several reachability queries
dead-cst build ./src -o graph.bin --plugin pytest --plugin fastapi
dead-cst analyze ./src --graph graph.bin
dead-cst analyze ./src --graph graph.bin --query test-only
dead-cst remove ./src --graph graph.bin --query test-only -o test-only.patch
```

## CLI reference

The CLI surfaces three commands — `build`, `analyze`, `remove` — plus a `--graph` flag that lets `analyze` / `remove` reuse a graph `build` already wrote to disk.

### `dead-cst build`

Materialize the project graph and persist it to disk. Subsequent `analyze` / `remove` runs can load the saved file with `--graph PATH` to skip the build step. Useful for CI shapes that run several reachability queries against the same code state. The on-disk format captures only the graph (nodes + edges + a small metadata block); plugins do **not** round-trip — load a graph and you get the materialized adjacency only.

```
dead-cst build ROOT -o PATH [OPTIONS]
```

| Option | Description |
|---|---|
| `-o, --output` | **Required.** Write the graph to this file |
| `-e, --entrypoint` | Entrypoint as a file path or FQN (repeatable) |
| `--entrypoint-regex` | Entrypoint as a regex over fqnames / file paths (repeatable) |
| `-p, --path` | Search path spec: `package:dep1,dep2` or `package` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` (mutually exclusive with `-p`) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `pytest`, `fastapi` (repeatable) |
| `--meta` | Stash `key=value` metadata in the graph file (repeatable) |
| `-v, --verbose` | Enable verbose logging |

Graph files start with the magic bytes `DEADCSTG` and a `u32` format version; mismatched versions raise on load with no migration path (rebuilding a graph is cheap).

### `dead-cst analyze`

Report dead code for a Python codebase.

```
dead-cst analyze ROOT [OPTIONS]
```

| Option | Description |
|---|---|
| `--graph` | Load a pre-built graph from disk; mutually exclusive with all build inputs |
| `--query` | Reachability question: `dead` (default) or `test-only` |
| `-e, --entrypoint` | Entrypoint as a file path or FQN (repeatable) |
| `--entrypoint-regex` | Entrypoint as a regex (repeatable) |
| `-p, --path` | Search path spec (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` |
| `--plugin` | Edge plugin to run (repeatable) |
| `--format` | Output format: `text` or `json` |
| `--exit-zero` | Always exit 0, even when dead code is found |
| `-v, --verbose` | Enable verbose logging |

Without `--exit-zero`, the exit code is 1 if dead code is found, 0 otherwise.

`--query test-only` runs `kept_alive_by_flags_only(NodeFlags.TESTCASE)` — the blast-radius diff between "alive with the test suite" and "alive without it" — and reports the set that would go dark if the tests went away. The test functions themselves are included (they are `TESTCASE`-flagged), so the result is the wholesale "the test suite and everything that backs it" set.

### `dead-cst remove`

Emit a unified diff that removes the dead code. The command never touches source files itself — pipe the patch into `git apply` (preferred in a git checkout) or POSIX `patch -p1` (handy when running from a subdirectory or outside git), or write it to a file with `-o` and apply later.

```
dead-cst remove ROOT [OPTIONS]
```

| Option | Description |
|---|---|
| `--graph` | Load a pre-built graph from disk; mutually exclusive with all build inputs |
| `--query` | Reachability question: `dead` (default) or `test-only` |
| `-e, --entrypoint` | Entrypoint as a file path or FQN (repeatable) |
| `--entrypoint-regex` | Entrypoint as a regex (repeatable) |
| `-p, --path` | Search path spec (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` |
| `--plugin` | Edge plugin to run (repeatable) |
| `-o, --output` | Write the patch to this file instead of stdout |
| `-v, --verbose` | Enable verbose logging |

```bash
dead-cst remove ./src -e mypkg.__main__ | git apply
dead-cst remove ./src -e mypkg.__main__ -o dead.patch && git apply dead.patch

# Outside git, or when running from a sub-directory whose CWD is the patch root:
dead-cst remove . -e mypkg.__main__ | patch -p1
```

## Python API

```python
import re
from pathlib import Path
from dead_cst import Analysis
from dead_cst.plugins import ExplicitEntrypointPlugin, MainBlockPlugin
from dead_cst.resolvers import ManualResolver

root = Path("./src")
analysis = Analysis(
    root,
    resolver=ManualResolver(specs=["."]),
    plugins=[
        MainBlockPlugin(),
        ExplicitEntrypointPlugin(specs=[re.compile(r".*__main__\.py")]),
    ],
)

# Whole-project queries: cheap to construct, lazy to materialize.
for node in analysis.dead():
    print(f"dead: {node.fqname} ({node.kind}) at {node.path}")

# "What's only alive because of the tests?" — the same blast-radius
# query the ``dead-cst remove --query test-only`` workflow uses.
from dead_cst.graph import NodeFlags
for node in analysis.kept_alive_by_flags_only(NodeFlags.TESTCASE):
    print(f"test-only: {node.fqname}")

# Rewrite source files to drop every unreachable decl + now-unused import.
from dead_cst.codemod import remove_code
remove_code(list(analysis.dead()), root)
```

For non-destructive review, `dead_cst.codemod.generate_patch(G, root)` returns the same removal as a `git apply`-compatible unified diff. Selection is driven entirely by `G.nodes`, so you can pass any subgraph slice to review a large codebase as a series of focused patches.

### Graph persistence

`dead_cst.graph.write_graph(path, graph, meta)` and
`dead_cst.graph.read_graph(path)` mirror the `dead-cst build` /
`--graph` workflow at the library level. The body is a bincode-encoded
blob prefixed with magic bytes + a `u32` format version; loaders
reject mismatched versions outright (no migration logic — rebuilding a
graph is cheap). The metadata block records `created_at`, node / edge
/ file / line counts, and the user-supplied `(key, value)` pairs the
CLI's `--meta` flag threads through.

```python
from dead_cst import Analysis
from dead_cst.graph import read_graph, write_graph
from dead_cst.resolvers import ManualResolver

graph = Analysis(root, resolver=ManualResolver(specs=["."])).materialize_all()
write_graph("graph.bin", graph, meta=[("branch", "main"), ("sha", "abc123")])

loaded, meta = read_graph("graph.bin")
print(meta.node_count, meta.edge_count, dict(meta.user_meta))
```

`Analysis(...).materialize_all()` returns the live
`dead_cst._native.ProjectContext`; queries on the `Analysis`
(`reachable` / `dead` / `descendants` / `ancestors` /
`kept_alive_by_flags_only`) delegate to the rust BFS without copying
the graph into Python.

### Incremental rebuilds

For long-lived `Analysis` instances (LSP harnesses, watch loops),
`Analysis.re_materialize(events)` rebuilds the graph against the
same `ProjectContext` without tearing down ty's salsa db. The caller
supplies the change events — either an explicit list (from an LSP
consuming `didChangeWatchedFiles`, a file-watcher, or a CI script
that knows which files it touched), or `ctx.detect_changes()` to let
ty figure it out:

```python
from dead_cst import _native as native

ctx = analysis.materialize_all()
# ... source files change on disk ...

analysis.re_materialize(ctx.detect_changes())              # autodetect via ty
analysis.re_materialize([                                  # or explicit
    native.ChangeEvent.changed("/abs/foo.py"),
    native.ChangeEvent.created("/abs/new.py"),
    native.ChangeEvent.deleted("/abs/gone.py"),
])
```

`ctx.detect_changes()` currently returns a single
`ChangeEvent.rescan()`, which ty's apply handler turns into an
mtime-checked `Files::sync_all` + project file re-walk + metadata
rediscovery. Salsa's per-file cache survives across calls, so
unchanged files skip parsing / `file_to_nodes` / `file_to_edges` /
`file_to_ref_edges` recomputation; cross-file importers invalidate
transitively through salsa's read-tracking. The assemble pass and
plugin pass run on every call, but they're cheap O(N) walks over the
warm cache.

Entrypoint detection is fully plugin-driven. Builtins:

| Plugin | Purpose |
|---|---|
| `MainBlockPlugin` | Mark modules containing `if __name__ == "__main__":` as entrypoints |
| `ProjectScriptsPlugin` | Read `pyproject.toml [project.scripts]` and mark each target as an entrypoint |
| `ExplicitEntrypointPlugin` | Match user-supplied file paths / FQNs / regexes (powers `-e` and `--entrypoint-regex`) |
| `ModuleDundersPlugin` | Keep top-level dunder variables (`__all__`, `__version__`, etc.) alive (always on) |
| `PytestPlugin` | Keep pytest-discovered tests, `conftest.py` decls, and `@pytest.fixture` functions alive (`--plugin pytest`) |
| `UnittestPlugin` | Keep stdlib `unittest.TestCase` / `IsolatedAsyncioTestCase` subclasses and `setUpModule` / `tearDownModule` / `load_tests` hooks alive. Transitive: a class extending a project-local `TestCase` mixin or a re-exported `TestCase` is detected (`--plugin unittest`) |
| `MockPatchPlugin` | Resolve string-fqname patch targets so symbols whose only consumers are tests stay alive. Recognizes `unittest.mock.patch` / `mock.patch` (decorator and context-manager forms, plus aliased imports), pytest-mock's `mocker.patch`, and pytest's `monkeypatch.setattr("X.Y", v)` / `monkeypatch.delattr("X.Y")` (`--plugin mock_patch`) |
| `ServerConfigPlugin` | Mark Gunicorn / Hypercorn config modules (`gunicorn.conf.py`, `hypercorn.conf.py`, and the `*_conf.py` variants) as entrypoints. The server loads these files by path at startup — nothing imports them statically, so the whole top-level surface would otherwise look dead. Override the `filenames` tuple for non-standard layouts (`--plugin server_config`) |
| `fastapi_plugin()` | Detect top-level `FastAPI()` / `APIRouter()` instances; mark `FastAPI` apps as entrypoints and add `instance -> handler` edges for every `@app.get(...)`-style decorator (HTTP methods, websockets, middleware, exception handlers, `on_event`). Routers stay pass-through, so an `APIRouter` that's never `include_router`'d remains dead (`--plugin fastapi`) |
| `flask_plugin()` | Detect top-level `Flask()` / `Blueprint()` instances; mark `Flask` apps as entrypoints and add `instance -> handler` edges for every `@app.route(...)` / `@app.get(...)` / lifecycle / errorhandler / template-helper / URL-processor decorator. Blueprints stay pass-through, so a `Blueprint` that's never `register_blueprint`'d remains dead (`--plugin flask`) |
| `typer_plugin()` | Detect top-level `Typer()` instances and add `instance -> handler` edges for every `@app.command(...)` / `@app.callback(...)` decorator. Typer apps are pass-through (reach them via `[project.scripts]` or `if __name__ == "__main__": app()`), so a sub-typer that's never `add_typer`'d stays dead (`--plugin typer`) |
| `ClickPlugin` | Detect top-level Click `Group` instances (functions decorated `@click.group(...)` or `X = click.Group(...)`) and add `instance -> handler` edges for every `@cli.command(...)` / `@cli.group(...)` / `@cli.result_callback(...)` decorator. Groups are pass-through, so a sub-group that's never `add_command`'d stays dead (`--plugin click`) |
| `cyclopts_plugin()` | Detect top-level `cyclopts.App()` instances and add `instance -> handler` edges for every `@app.command(...)` / `@app.default(...)` decorator. Apps are pass-through, mirroring the Typer plugin (`--plugin cyclopts`) |
| `DiscordPyPlugin` | Detect top-level `commands.Bot()` / `discord.Client()` (and `AutoSharded*`) instances, mark them as entrypoints, and add `instance -> handler` edges for `@bot.command()` / `event` / `listen()` / `group()` / `hybrid_command()` / `before_invoke` / `after_invoke` / `check` decorators plus the two-level `@bot.tree.command()` / `@bot.tree.context_menu()` slash-command form (`--plugin discordpy`) |
| `fastmcp_plugin()` | Detect top-level `FastMCP()` server instances, mark them as entrypoints, and add `instance -> handler` edges for every `@mcp.tool` / `@mcp.resource` / `@mcp.prompt` / `@mcp.completion` decorator. Factory-aware: `def create_server() -> FastMCP: ...` chains classify across packages (`--plugin fastmcp`) |
| `InitSubclassPlugin` | Detect classes that define `__init_subclass__` and add `parent -> subclass` edges for every (transitive) first-party subclass. Parents stay pass-through, so a registry base class only keeps subclasses alive once something else keeps the parent alive (`--plugin init_subclass`) |

For project-specific dynamic-import patterns, three abstract bases ship as scaffolding that subclasses configure in 4-5 lines:

| Abstract base | Use it for |
|---|---|
| `DecoratedDeclPlugin` | "Find decorated decls in files matching a search path." Subclass with `package_prefix`, `decorator_module`, `decorator_names`, `marker_prefix`. |
| `LiteralListPlugin` | "Read `<owner>.<var> = ['fqn', ...]` and treat each entry as alive." Subclass with `owner_fqname`, `variable_name`, `marker_prefix`. |
| `DispatchAppPlugin` | "Wire `@<instance>.<reg>(...)` handlers to an app instance." Subclass with `app_module`, `registration_decorators`, plus either `constructor_targets` (pure-dispatch — powers `typer_plugin` / `cyclopts_plugin`, app instances stay pass-through) or `instance_kinds: Mapping[str, bool]` (factory-aware — powers `flask_plugin` / `fastapi_plugin` / `CeleryPlugin`; emits `<{name}-app>` / `<{name}-pending>` / `<{name}-factory>` synthetics and runs a per-package finalize walk that classifies factory chains across files and promotes auto-entrypoint kinds). |

Write your own plugin from scratch by subclassing `dead_cst.plugins.Plugin` and implementing `run(ctx)` to yield `AddNode` / `AddEdge` / `AddEntrypoint` ops. Register under the `dead_cst.plugins` entry-point group for CLI discovery.

Path resolution is similarly pluggable. `PathResolver` implementations return a tuple of `Package` records (`path`, `name`, `deps`) to feed `Analysis`. Builtins: `ManualResolver` (explicit `package:dep` specs from `-p`) and `UvResolver` (parses `uv.lock` to discover workspace members and their inter-member dep edges). Third-party resolvers register under `dead_cst.resolvers`.

Unreachable code regions (`if False:`, the else of `if True:`, statements after an unconditional `return` / `raise` / `break` / `continue`, ...) are detected by the rust backend and stamped onto the contributing edges with `EdgeFlags.DEAD_BRANCH`. Default reachability follows those edges so the metadata stays informational; `Analysis.kept_alive_by_dead_branches()` is the opt-in inverse query that returns the set of nodes only reachable through dead-branch edges.

## Graph model

The graph has one node per top-level declaration plus a synthetic module node per file. Edges run from a declaration to each symbol it references, and from every submodule to its parent package so `__init__.py` stays alive as long as anything in the package does. Entrypoints seed the reachability walk; every node not reached is reported as dead.

A module-level `import` / `from ... import ...` is itself a declaration of type `"import"` in the current module. Uses of the imported name inside the file are wired through that local import node, and the import node in turn points at the upstream module (and, when applicable, at the specific imported symbol). Removing the last local use therefore makes the import itself dead, which is how `dead-cst remove` knows to drop now-unused import lines.

Imports whose source line carries a ruff/pyflakes `# noqa` directive that silences F401 (`# noqa`, `# noqa: F401`, multi-rule `# noqa: E501, F401`, case-variant `# NOQA`) are pinned alive. File-level `# ruff: noqa` and `# flake8: noqa` directives (`ruff:` / `flake8:` is matched case-sensitively per ruff; `noqa` is not) pin every import in the file. This matches ruff's own semantics: an import you have explicitly preserved (re-exports, side-effect imports, `TYPE_CHECKING` shims guarded by F401) is not surfaced as dead and `dead-cst remove` will not drop it.

Pinned imports are tagged with `NodeFlags.NOQA` (in addition to `NodeFlags.ENTRYPOINT`). The opt-in `Analysis.kept_alive_by_flags_only(flags)` query takes any `NodeFlags` combination and returns the "blast radius" of dropping every entrypoint with those flag bits. Pass `NodeFlags.NOQA` to audit stale F401 pins ("if I removed every `# noqa: F401`, what would actually become dead?"), `NodeFlags.TESTCASE` for "what would die if you dropped the test suite", or `NodeFlags.TESTCASE | NodeFlags.NOQA` to combine in one pass.

## Scope

`dead-cst` tracks top-level declarations only -- module-level functions, classes, and variables. Nested definitions (inner functions, methods, nested classes) are deliberately not given their own nodes; references made from inside those nested scopes are attributed to the enclosing top-level declaration. Keeping the containing top-level symbol alive keeps its nested source alive with it.

`.pyi` stub files are ingested for the **compiled-extension** layout: a binary like `mypkg/_native.so` shipping next to `mypkg/_native.pyi`. With no matching `.py`, the stub is the only Python-discoverable description of those names, so the analyzer parses it under its natural module FQN (`mypkg._native`) and `from mypkg._native import compute` resolves to the stub's declaration through the normal import path. A `.pyi` shipped alongside a real `.py` is dropped during file enumeration -- the runtime always wins, and the peer stub is invisible to dead-cst.

Jupyter notebooks (`.ipynb`) are ingested too: code cells are concatenated in document order into a single libcst-parseable module, IPython line / cell magics, shell escapes (`!ls`), and trailing-help forms (`obj?`, `obj.attr??`) are neutralized line-for-line into `pass  # <line>` comments, and every emitted `SymbolNode` is flagged `NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT`. Notebooks aren't importable, so their decls stay out of the cross-module lookup trie — but `.py` code a notebook imports stays alive, which makes notebooks effective reachability seeds. `dead-cst remove` skips `.ipynb` paths (cell-aware writeback into the notebook JSON envelope is out of scope today).

`@typing.overload`-decorated functions in `.py` files are flagged so a dead implementation drags its overloads along during `dead-cst remove` instead of leaving them behind as orphans.

## Limitations

- `from X import *` is treated pessimistically (every top-level declaration in the target stays alive). Cross-module `from <importer> import <name>` still resolves through to `<name>`'s real source — even across chained `__init__.py` star re-exports — via ty's name resolution, and bare-name uses inside function bodies (`from x import *; def a(): g()`) edge to the upstream decl. When two stars in the same module export the same name, "first writer wins"; Python's runtime "last star wins" semantics is not implemented.
- `__import__('pkg.mod')` and `importlib.import_module('pkg.mod')` with string-literal arguments emit `EdgeFlags.DYNAMIC_IMPORT` edges to the resolved module / decls; non-literal arguments fall back to opaque external nodes. `DynamicImportFallbackPlugin` opts into per-name fan-out for the dynamic-import case.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- Only first-party code is analysed; third-party dependencies appear as synthetic `[external dist] <name>` / `[external file] <name>` nodes in the graph (and are the targets of every import that resolves outside the project).
- `__all__` is followed only when assigned a list/tuple of string literals; dynamic mutation (`__all__.append`, comprehensions, etc.) is not tracked.
- PEP 572 walrus targets inside a comprehension (`[last := n for n in nums]`) don't surface as top-level decls and module-level uses of the name go unresolved. ty has a `// TODO walrus in comprehensions is implicitly nonlocal`, so the `last` binding stays scoped to the comprehension instead of leaking to the enclosing module per PEP 572 (see `tests/test_limitations.py::comprehension-walrus-doesnt-leak-to-enclosing-scope`).

## Native plugins (experimental)

Most extensions are [Python plugins](CONTRIBUTING.md#adding-a-plugin) — they
ship in the wheel and need no toolchain. For hot logic, or plugins that want
their own salsa-cached queries over ty's database, there's also a **native
(Rust) plugin** path: a separately-compiled plugin that links the dead-cst
runtime and runs against a live `ProjectContext`.

```bash
pip install dead-cst[build-plugin]          # pulls the prebuilt runtime
PLUGIN=$(dead-cst build-plugin my_plugin.rs)  # compile against it
```

```python
from dead_cst import Analysis, _native as native
plugins = native.load_native_plugins(PLUGIN)
Analysis(root, plugins=plugins).materialize_all()
```

This is a preview (macOS and Linux today, recompile-per-release). See
**[`NATIVE_PLUGINS.md`](NATIVE_PLUGINS.md)** for the full guide and the worked
`examples/main_block_plugin/`.

## Development

```bash
git clone https://github.com/lpetre/dead-cst
cd dead-cst
uv sync
uv run pytest
uv run prek run --all-files
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a walkthrough of the analyzer pipeline, [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full dev guide, [`CHANGELOG.md`](CHANGELOG.md) for release notes, and [`ROADMAP.md`](ROADMAP.md) for the stack-ranked plan toward 1.0.

## TODO

- Host API documentation on Read the Docs.

## License

MIT — see [`LICENSE`](LICENSE).
