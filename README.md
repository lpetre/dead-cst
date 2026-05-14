# dead-cst

[![PyPI](https://img.shields.io/pypi/v/dead-cst.svg)](https://pypi.org/project/dead-cst/)
[![Python](https://img.shields.io/pypi/pyversions/dead-cst.svg)](https://pypi.org/project/dead-cst/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/lpetre/dead-cst/actions/workflows/ci.yml/badge.svg)](https://github.com/lpetre/dead-cst/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lpetre/dead-cst/branch/main/graph/badge.svg)](https://codecov.io/gh/lpetre/dead-cst)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Python dead code analysis using [libcst](https://github.com/Instagram/LibCST).

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
dead-cst analyze ./src -e "re:.*__main__\.py"

# See why a symbol is kept alive
dead-cst why-alive ./src mypackage.some_module.some_function

# Generate a patch that removes dead code, then apply it
dead-cst remove ./src -e "re:.*__main__\.py" | git apply

# List third-party dependencies imported by the codebase
dead-cst dependencies ./src
```

## CLI reference

### `dead-cst analyze`

Analyze a Python codebase for dead code.

```
dead-cst analyze ROOT -e ENTRYPOINT [OPTIONS]
```

| Option | Description |
|---|---|
| `-e, --entrypoint` | Entrypoint: file path, FQN, or `re:pattern` for regex (repeatable) |
| `-p, --path` | Search path spec: `package:dep1,dep2` or `package` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` (mutually exclusive with `-p`) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `--format` | Output format: `text` or `json` |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |
| `-j, --workers` | Run cache-miss visitor passes in this many worker processes (`>=2` enables it) |

Exit code 1 if dead code is found, 0 otherwise.

### `dead-cst why-alive`

Show why a symbol is considered alive by printing its predecessor chain.

```
dead-cst why-alive ROOT FQNAME [OPTIONS]
```

| Option | Description |
|---|---|
| `-p, --path` | Search path spec: `package:dep1,dep2` or `package` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` (mutually exclusive with `-p`) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |
| `-j, --workers` | Run cache-miss visitor passes in this many worker processes (`>=2` enables it) |

### `dead-cst unused-exports`

Report `__all__` entries whose targets are only alive because of `__all__`. Useful in closed-world / monorepo settings to prune the public surface.

```
dead-cst unused-exports ROOT -e ENTRYPOINT [OPTIONS]
```

| Option | Description |
|---|---|
| `-e, --entrypoint` | Entrypoint: file path, FQN, or `re:pattern` for regex (repeatable) |
| `-p, --path` | Search path spec: `package:dep1,dep2` or `package` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` (mutually exclusive with `-p`) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |
| `-j, --workers` | Run cache-miss visitor passes in this many worker processes (`>=2` enables it) |

### `dead-cst dependencies`

List third-party dependencies imported by the codebase. Each package
gets its own section. Distributions are reported as `[external dist] <name>`; files
resolved inside `site-packages` without a matching distribution are reported as
`[external file] <name>`.

```
dead-cst dependencies ROOT [OPTIONS]
```

| Option | Description |
|---|---|
| `-p, --path` | Search path spec: `package:dep1,dep2` or `package` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` (mutually exclusive with `-p`) |
| `--format` | Output format: `text` or `json` |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |
| `-j, --workers` | Run cache-miss visitor passes in this many worker processes (`>=2` enables it) |

### `dead-cst remove`

Emit a `git apply`-compatible unified diff that removes the dead code. The
command never touches source files itself — pipe the patch into `git apply`
(or write it to a file with `-o` and apply later).

```
dead-cst remove ROOT -e ENTRYPOINT [OPTIONS]
```

```bash
dead-cst remove ./src -e mypkg.__main__ | git apply
# or
dead-cst remove ./src -e mypkg.__main__ -o dead.patch && git apply dead.patch
```

| Option | Description |
|---|---|
| `-e, --entrypoint` | Entrypoint: file path, FQN, or `re:pattern` for regex (repeatable) |
| `-p, --path` | Search path spec: `package:dep1,dep2` or `package` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `uv` (mutually exclusive with `-p`) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `-v, --verbose` | Enable verbose logging |
| `-o, --output` | Write the patch to this file instead of stdout |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |
| `-j, --workers` | Run cache-miss visitor passes in this many worker processes (`>=2` enables it) |

### `dead-cst cache clear`

Delete the on-disk `VisitorPayload` cache (`<root>/.dead-cst-cache/`) for a project. Each row is keyed by a per-package fingerprint over the visitor / plugin / unreachable-region-detector `(name, version)` triple, schema version, Python version, and the package's `exported` subdirs (which feed `NodeFlags.EXPORTED` into every node from a file under them), so most analyzer-version changes invalidate it automatically; this command is for force-clearing when needed. Resolvers, search paths, and `Package.path` / `name` / `deps` deliberately do *not* enter the fingerprint — import resolution runs unconditionally on every analysis, so resolver / search-path / package-name swaps re-stitch edges without re-running the visitor. Editing `Package.exported` invalidates only the affected package's rows.

```
dead-cst cache clear [ROOT]
```

`ROOT` defaults to the current directory.

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
    print(f"dead: {node.fqname} ({node.type}) at {node.path}")

# Per-package queries scope work to the smallest package set that gives
# correct reachability answers. Local queries (modules, declarations)
# never materialize cross-package state.
pkg = analysis.package(root)
print(sum(1 for _ in pkg.modules()), "modules")
pkg.remove_dead_code()  # codemod, scoped to this package
```

For non-destructive review, `dead_cst.codemod.generate_patch(G, root)` returns the same removal as a `git apply`-compatible unified diff. Selection is driven entirely by `G.nodes`, so you can pass any subgraph slice — e.g. one [strongly-connected component](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.strongly_connected_components.html) at a time — to review a large codebase as a series of focused patches.

`Analysis(...).materialize_all()` returns the full `networkx.MultiDiGraph` if you need raw access; `analysis.package(path).graph()` returns the closure-scoped subgraph for one package.

Edge plugins and the unreachable-region detector share a single `Cacheable` protocol (`name: str`, `version: int`). The core `SymbolVisitor` carries the same pair, so visitor-level changes get an explicit knob too. The visitor / plugin / detector triple plus the package's `exported` setting feeds the per-package cache fingerprint — bumping any of those `version`s invalidates stale payloads automatically, and editing `Package.exported` invalidates only the affected package. Path resolvers do **not** implement `Cacheable`: most of their output (`Package.path` / `name` / `deps`, `resolve_import`) flows through the (uncached) edge-stitching pass, and the one piece that feeds the cache (`exported`) is captured by value in the fingerprint, so a resolver swap that produces the same exports keeps cache hits intact. The package `__version__` is intentionally *not* in the fingerprint: every component whose output can shift between releases owns a dedicated `version`, and folding in `__version__` would let unbumped components ride for free on a release bump.

Entrypoint detection is fully plugin-driven. Builtins:

| Plugin | Purpose |
|---|---|
| `MainBlockPlugin` | Mark modules containing `if __name__ == "__main__":` as entrypoints |
| `ProjectScriptsPlugin` | Read `pyproject.toml [project.scripts]` and mark each target as an entrypoint |
| `ExplicitEntrypointPlugin` | Match user-supplied file paths / FQNs / regexes (powers the `-e` flag) |
| `ModuleDundersPlugin` | Keep top-level dunder variables (`__all__`, `__version__`, etc.) alive (always on) |
| `PytestPlugin` | Keep pytest-discovered tests, `conftest.py` decls, and `@pytest.fixture` functions alive (`--plugin pytest`) |
| `UnittestPlugin` | Keep stdlib `unittest.TestCase` / `IsolatedAsyncioTestCase` subclasses and `setUpModule` / `tearDownModule` / `load_tests` hooks alive. Transitive: a class extending a project-local `TestCase` mixin or a re-exported `TestCase` is detected (`--plugin unittest`) |
| `MockPatchPlugin` | Resolve string-fqname patch targets so symbols whose only consumers are tests stay alive. Recognizes `unittest.mock.patch` / `mock.patch` (decorator and context-manager forms, plus aliased imports), pytest-mock's `mocker.patch`, and pytest's `monkeypatch.setattr("X.Y", v)` / `monkeypatch.delattr("X.Y")` (`--plugin mock_patch`) |
| `ServerConfigPlugin` | Mark Gunicorn / Hypercorn config modules (`gunicorn.conf.py`, `hypercorn.conf.py`, and the `*_conf.py` variants) as entrypoints. The server loads these files by path at startup (Docker, Cloud Run, systemd, ...) -- nothing imports them statically, so the whole top-level surface (hooks, settings, imports) would otherwise look dead. Override the `filenames` tuple for non-standard layouts (`--plugin server_config`) |
| `FastAPIPlugin` | Detect top-level `FastAPI()` / `APIRouter()` instances; mark `FastAPI` apps as entrypoints and add `instance -> handler` edges for every `@app.get(...)`-style decorator (HTTP methods, websockets, middleware, exception handlers, `on_event`). Routers stay pass-through, so an `APIRouter` that's never `include_router`'d remains dead (`--plugin fastapi`) |
| `FlaskPlugin` | Detect top-level `Flask()` / `Blueprint()` instances; mark `Flask` apps as entrypoints and add `instance -> handler` edges for every `@app.route(...)` / `@app.get(...)` / lifecycle / errorhandler / template-helper / URL-processor decorator. Blueprints stay pass-through, so a `Blueprint` that's never `register_blueprint`'d remains dead (`--plugin flask`) |
| `TyperPlugin` | Detect top-level `Typer()` instances and add `instance -> handler` edges for every `@app.command(...)` / `@app.callback(...)` decorator. Typer apps are pass-through (reach them via `[project.scripts]` or `if __name__ == "__main__": app()`), so a sub-typer that's never `add_typer`'d stays dead (`--plugin typer`) |
| `ClickPlugin` | Detect top-level Click `Group` instances (functions decorated `@click.group(...)` or `X = click.Group(...)`) and add `instance -> handler` edges for every `@cli.command(...)` / `@cli.group(...)` / `@cli.result_callback(...)` decorator. Groups are pass-through (reach them via `[project.scripts]` or a `__main__` block), so a sub-group that's never `add_command`'d stays dead (`--plugin click`) |
| `CycloptsPlugin` | Detect top-level `cyclopts.App()` instances and add `instance -> handler` edges for every `@app.command(...)` / `@app.default(...)` decorator. Apps are pass-through (reach them via `[project.scripts]` or a `__main__` block), mirroring `TyperPlugin` (`--plugin cyclopts`) |
| `DiscordPyPlugin` | Detect top-level `commands.Bot()` / `discord.Client()` (and `AutoSharded*`) instances, mark them as entrypoints, and add `instance -> handler` edges for `@bot.command()` / `event` / `listen()` / `group()` / `hybrid_command()` / `before_invoke` / `after_invoke` / `check` decorators plus the two-level `@bot.tree.command()` / `@bot.tree.context_menu()` slash-command form. Anchors every `commands.Cog` / `GroupCog` subclass and the file's module-level `setup` / `teardown` hooks to a per-file synthetic, and folds `<expr>.load_extension("dotted.path")` / `load_extensions([...])` string-literal targets into the captured module's surface (`--plugin discordpy`) |
| `FastMCPPlugin` | Detect top-level `FastMCP()` server instances, mark them as entrypoints (the `fastmcp` CLI loads `module:mcp` by import path the same way `uvicorn` loads a FastAPI `module:app`), and add `instance -> handler` edges for every `@mcp.tool` / `@mcp.resource` / `@mcp.prompt` / `@mcp.completion` decorator. Factory-aware: `def create_server() -> FastMCP: ...` chains classify across packages via the shared `DispatchAppPlugin` factory marker. Only the standalone `fastmcp` import path is recognized (`--plugin fastmcp`) |
| `InitSubclassPlugin` | Detect classes that define `__init_subclass__` and add `parent -> subclass` edges for every (transitive) first-party subclass. Parents stay pass-through, so a registry base class only keeps subclasses alive once something else (an entrypoint, an import) keeps the parent alive (`--plugin init_subclass`) |

For project-specific dynamic-import patterns, three abstract bases ship as scaffolding that subclasses configure in 4-5 lines:

| Abstract base | Use it for |
|---|---|
| `DecoratedDeclPlugin` | "Find decorated decls in files matching a search path." Subclass with `package_prefix`, `decorator_module`, `decorator_names`, `constructor_names`. Pure observe-time. |
| `LiteralListPlugin` | "Read `<owner>.<var> = ['fqn', ...]` and treat each entry as alive." Subclass with `owner_fqname`, `variable_name`. observe parses and caches; finalize only does graph lookups. |
| `DispatchAppPlugin` | "Wire `@<instance>.<reg>(...)` handlers to an app instance." Subclass with `app_module`, `registration_decorators`, plus either `constructor_targets` (pure-dispatch — powers `TyperPlugin` / `CycloptsPlugin`, app instances stay pass-through) or `instance_kinds: Mapping[str, bool]` (factory-aware — powers `FlaskPlugin` / `FastAPIPlugin` / `CeleryPlugin`; emits `<{name}-app>` / `<{name}-pending>` / `<{name}-factory>` synthetics and runs a per-package finalize walk that classifies factory chains across files and promotes auto-entrypoint kinds). |

All three bases require subclasses to set `name` (a unique identifier for the cache namespace) and `version` (a Unix epoch int — bump it to the current epoch when the subclass's config changes). For example:

```python
from dataclasses import dataclass
from dead_cst.plugins import LiteralListPlugin

@dataclass(kw_only=True)
class MyInternalModulesPlugin(LiteralListPlugin):
    owner_fqname: str = "myapp.config"
    variable_name: str = "INTERNAL_MODULES"
    name: str = "my_internal_modules"
    version: int = 1700000000
```

Write your own from scratch by implementing the `EdgePlugin` protocol (`name`, `version`, `observe`, `finalize`); register under the `dead_cst.plugins` entry-point group for CLI discovery.

Path resolution is similarly pluggable. `PathResolver` implementations return a tuple of `Package` records (`path`, `name`, `exported`, `deps`) to feed `Analysis`. Builtins: `ManualResolver` (explicit `package:dep` specs from `-p`) and `UvResolver` (parses `uv.lock` to discover workspace members and their inter-member dep edges). Third-party resolvers register under `dead_cst.resolvers`.

Unreachable-code detection is pluggable through the `UnreachableRegionDetector` protocol. `Analysis` accepts an `unreachable_detector` whose `find_regions(wrapper) -> list[CodeRange]` is invoked once per file. The built-in `DefaultUnreachableRegionDetector` covers three things out of the box:

- **Literal truthiness** on every `if` / `while` test (e.g. `if False:` always-dead body, `if True: ... else: ...` always-dead else).
- **Flow-sensitive name resolution** over simple `Name = literal` (and `Name: T = literal`) assignments. Chains like `foo = False; bar = foo or False; if bar: ...` resolve to dead because the goal-directed `TruthinessResolver` recursively evaluates each binding's RHS on demand and memoizes the result. Mutable container literals (`x = []`, `x = {}`, comprehensions) are deliberately *not* folded — `.append` / item assignment between binding and use is invisible to the binding-only flow walk, so their truthiness stays unknown.
- **Post-terminator regions** inside every suite. Statements after an unconditional `return` / `raise` / `break` / `continue` / `assert <statically-falsy>` are marked dead, and so are the statements after a compound `if` / `with` / `try` whose every reachable branch itself terminates — so `if True: return` (and constant-folded variants like `if FLAG: return` with `FLAG = True`) kills the rest of its enclosing suite. Suite-relative, so a `raise` in a `try` body kills only the rest of the try body — the `except` handler still runs on its own path.

To layer in domain knowledge — e.g. config flags whose values are fixed in production — subclass and override `resolve(self, expr) -> bool | None`. The override gets first crack at every non-keyword expression routed through the resolver chain; returning `None` defers to the built-in literal handling and name lookup. Constants resolved this way compose with name resolution, so a single high-level decision propagates through chains:

```python
from dataclasses import dataclass

import libcst as cst
from dead_cst import Analysis
from dead_cst.branches import DefaultUnreachableRegionDetector

@dataclass(frozen=True)
class FlagAwareDetector(DefaultUnreachableRegionDetector):
    # name/version satisfy the Cacheable contract -- bump version when
    # the override's logic changes so stale per-file payloads rebuild
    # automatically.
    name: str = "flag_aware"
    version: int = 1700000000

    def resolve(self, expr: cst.BaseExpression) -> bool | None:
        # The override is consulted recursively, so guard with an early
        # isinstance check to keep it cheap.
        if (
            isinstance(expr, cst.Call)
            and isinstance(expr.func, cst.Name)
            and expr.func.value == "check_flag"
            and expr.args
            and isinstance(expr.args[0].value, cst.SimpleString)
        ):
            return MIGRATIONS[expr.args[0].value.evaluated_value]
        return None

graph = Analysis(
    root,
    resolver=ManualResolver(specs=["."]),
    unreachable_detector=FlagAwareDetector(),
).materialize_all()
```

With the override above, `if check_flag("migration-abc"): ...` and `flag = check_flag("migration-abc"); if flag: ...` both resolve to a known truthiness, and the unreachable suite is flagged just like a literal `if False:` would be.

For detectors that don't fit the constant-folding model at all, write a fresh class that implements `find_regions(wrapper) -> list[CodeRange]` directly — the protocol requires nothing else beyond the `Cacheable` `(name, version)` pair.

## Graph model

The graph has one node per top-level declaration plus a synthetic module node per file. Edges run from a declaration to each symbol it references, and from every submodule to its parent package so `__init__.py` stays alive as long as anything in the package does. Entrypoints seed the reachability walk; every node not reached is reported as dead.

A module-level `import` / `from ... import ...` is itself a declaration of type `"import"` in the current module. Uses of the imported name inside the file are wired through that local import node, and the import node in turn points at the upstream module (and, when applicable, at the specific imported symbol). Removing the last local use therefore makes the import itself dead, which is how `dead-cst remove` knows to drop now-unused import lines.

Imports whose source line carries a ruff/pyflakes `# noqa` directive that silences F401 (`# noqa`, `# noqa: F401`, multi-rule `# noqa: E501, F401`, case-variant `# NOQA`) are pinned alive. File-level `# ruff: noqa` and `# flake8: noqa` directives (`ruff:` / `flake8:` is matched case-sensitively per ruff; `noqa` is not) pin every import in the file. This matches ruff's own semantics: an import you have explicitly preserved (re-exports, side-effect imports, `TYPE_CHECKING` shims guarded by F401) is not surfaced as dead and `dead-cst remove` will not drop it.

Pinned imports are tagged with `NodeFlags.NOQA` (in addition to `NodeFlags.ENTRYPOINT`). The opt-in `Analysis.kept_alive_by_flags_only(flags)` / `PackageView.kept_alive_by_flags_only(flags)` query takes any `NodeFlags` combination and returns the "blast radius" of dropping every entrypoint with those flag bits. Pass `NodeFlags.NOQA` to audit stale F401 pins ("if I removed every `# noqa: F401`, what would actually become dead?"), `NodeFlags.TESTCASE` for "what would die if you dropped the test suite", or `NodeFlags.TESTCASE | NodeFlags.NOQA` to combine in one pass.

## Scope

`dead-cst` tracks top-level declarations only -- module-level functions, classes, and variables. Nested definitions (inner functions, methods, nested classes) are deliberately not given their own nodes; references made from inside those nested scopes are attributed to the enclosing top-level declaration. Keeping the containing top-level symbol alive keeps its nested source alive with it.

`.pyi` stub files are ingested for the **compiled-extension** layout: a binary like `mypkg/_native.so` shipping next to `mypkg/_native.pyi`. With no matching `.py`, the stub is the only Python-discoverable description of those names, so the analyzer parses it under its natural module FQN (`mypkg._native`) and `from mypkg._native import compute` resolves to the stub's declaration through the normal import path. A `.pyi` shipped alongside a real `.py` is dropped during file enumeration -- the runtime always wins, and the peer stub is invisible to dead-cst.

Jupyter notebooks (`.ipynb`) are ingested too: code cells are concatenated in document order into a single libcst-parseable module, IPython line / cell magics, shell escapes (`!ls`), and trailing-help forms (`obj?`, `obj.attr??`) are neutralized line-for-line into `pass  # <line>` comments, and every emitted `SymbolNode` is flagged `NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT`. Notebooks aren't importable, so their decls stay out of the cross-module lookup trie — but `.py` code a notebook imports stays alive, which makes notebooks effective reachability seeds. `dead-cst remove` skips `.ipynb` paths (cell-aware writeback into the notebook JSON envelope is out of scope today).

`@typing.overload`-decorated functions in `.py` files are flagged so a dead implementation drags its overloads along during `dead-cst remove` instead of leaving them behind as orphans.

## Limitations

- `from X import *` is treated pessimistically (every top-level declaration in the target stays alive) **and** materialized: each name the target exposes becomes a synthetic re-export in the importing module's trie, so cross-module `from <importer> import <name>` resolves through to `<name>`'s real source — even across chained `__init__.py` re-exports. When two stars in the same module export the same name, "first writer wins"; Python's runtime "last star wins" semantics is not implemented (see `tests/test_limitations.py::last-star-wins-not-implemented`). Bare-name references inside a function body that rely on a star binding (`from x import *; def a(): g()`) still aren't resolved per-access — LibCST's `ScopeProvider` can't bind them back to the star.
- `__import__('pkg.mod')` and `importlib.import_module('pkg.mod')` are treated the same way when the module name is a string literal -- the call fans out to every top-level decl in the target module so `getattr(__import__('pkg.mod'), 'name')` keeps `pkg.mod.name` reachable. Relative names follow the same rules as `from .x import *`: `importlib.import_module('.sub')` (or `__import__('sub', ..., level=1)`) resolves against the file's enclosing package, and an explicit `package=` literal overrides the anchor. `__import__(name, fromlist=[...])` with a literal list/tuple resolves each entry as a possible submodule and fans those out too. Non-literal arguments (name, `level`, `package`, `fromlist`) are skipped with a warning.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- Only first-party code is analysed; third-party dependencies are treated as opaque (they appear as synthetic nodes — see `dead-cst dependencies`).
- `__all__` is followed only when assigned a list/tuple of string literals; dynamic mutation (`__all__.append`, comprehensions, etc.) is not tracked.
- Files `libcst` cannot parse are not fatal: the analyser logs a warning and stands in a `[unparseable] <module>` placeholder so the file stays alive in reachability and importers can still target the module. Declarations *inside* the file are invisible until parsing succeeds.

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
