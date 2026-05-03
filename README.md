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

> **Pre-release software.** `dead-cst` is in early alpha. APIs, CLI flags, and output formats may change without notice, and bugs are expected. Do not run `dead-cst remove` against code that isn't committed to version control.

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

# Remove dead code (interactive confirmation)
dead-cst remove ./src -e "re:.*__main__\.py"

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
| `-p, --path` | Search path spec: `base:dep1,dep2` or `base` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `venv`, `pyproject` (repeatable) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `--format` | Output format: `text` or `json` |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |

Exit code 1 if dead code is found, 0 otherwise.

### `dead-cst why-alive`

Show why a symbol is considered alive by printing its predecessor chain.

```
dead-cst why-alive ROOT FQNAME [OPTIONS]
```

| Option | Description |
|---|---|
| `-p, --path` | Search path spec: `base:dep1,dep2` or `base` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `venv`, `pyproject` (repeatable) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |

### `dead-cst unused-exports`

Report `__all__` entries whose targets are only alive because of `__all__`. Useful in closed-world / monorepo settings to prune the public surface.

```
dead-cst unused-exports ROOT -e ENTRYPOINT [OPTIONS]
```

| Option | Description |
|---|---|
| `-e, --entrypoint` | Entrypoint: file path, FQN, or `re:pattern` for regex (repeatable) |
| `-p, --path` | Search path spec: `base:dep1,dep2` or `base` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `venv`, `pyproject` (repeatable) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |

### `dead-cst dependencies`

List third-party dependencies imported by the codebase. Each base path gets its
own section. Distributions are reported as `[external dist] <name>`; files
resolved inside `site-packages` without a matching distribution are reported as
`[external file] <name>`.

```
dead-cst dependencies ROOT [OPTIONS]
```

| Option | Description |
|---|---|
| `-p, --path` | Search path spec: `base:dep1,dep2` or `base` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `venv`, `pyproject` (repeatable) |
| `--format` | Output format: `text` or `json` |
| `-v, --verbose` | Enable verbose logging |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |

### `dead-cst remove`

Remove dead code from a Python codebase. Prompts for confirmation before modifying files.

```
dead-cst remove ROOT -e ENTRYPOINT [OPTIONS]
```

| Option | Description |
|---|---|
| `-e, --entrypoint` | Entrypoint: file path, FQN, or `re:pattern` for regex (repeatable) |
| `-p, --path` | Search path spec: `base:dep1,dep2` or `base` (repeatable) |
| `--resolver` | Path resolver to run, e.g. `venv`, `pyproject` (repeatable) |
| `--plugin` | Edge plugin to run, e.g. `main_block`, `project_scripts` (repeatable) |
| `-v, --verbose` | Enable verbose logging |
| `--dry-run` | Show what would be removed without making changes |
| `--no-cache` | Bypass the per-file `VisitorPayload` cache |

### `dead-cst cache clear`

Delete the on-disk `VisitorPayload` cache (`<root>/.dead-cst-cache/`) for a project. The cache is keyed by a fingerprint over the `PathMap`, resolver chain, and plugin set, so most layout changes invalidate it automatically; this command is for force-clearing when needed.

```
dead-cst cache clear [ROOT]
```

`ROOT` defaults to the current directory.

## Python API

```python
import re
from pathlib import Path
from dead_cst import (
    build_symbol_graph,
    ExplicitEntrypointPlugin,
    MainBlockPlugin,
    find_reachable,
    remove_code,
)

root = Path("./src")
graph = build_symbol_graph(
    {root: []},
    plugins=[
        MainBlockPlugin(),
        ExplicitEntrypointPlugin(specs=[re.compile(r".*__main__\.py")]),
    ],
    project_root=root,
)
reachable = find_reachable(graph)

unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable])
# Inspect unreachable nodes, or remove them:
remove_code(unreachable, root)
```

All three extension points — edge plugins, path resolvers, and the unreachable-region detector — share a single `Cacheable` protocol (`name: str`, `version: int`) that feeds the per-file cache fingerprint. Bumping a component's epoch `version` invalidates stale payloads automatically, so swapping or upgrading any of them is safe by default.

Entrypoint detection is fully plugin-driven. Builtins:

| Plugin | Purpose |
|---|---|
| `MainBlockPlugin` | Mark modules containing `if __name__ == "__main__":` as entrypoints |
| `ProjectScriptsPlugin` | Read `pyproject.toml [project.scripts]` and mark each target as an entrypoint |
| `ExplicitEntrypointPlugin` | Match user-supplied file paths / FQNs / regexes (powers the `-e` flag) |
| `ModuleDundersPlugin` | Keep top-level dunder variables (`__all__`, `__version__`, etc.) alive (always on) |
| `PytestPlugin` | Keep pytest-discovered tests, `conftest.py` decls, and `@pytest.fixture` functions alive (`--plugin pytest`) |
| `UnittestPlugin` | Keep stdlib `unittest.TestCase` / `IsolatedAsyncioTestCase` subclasses and `setUpModule` / `tearDownModule` / `load_tests` hooks alive (`--plugin unittest`) |
| `FastAPIPlugin` | Detect top-level `FastAPI()` / `APIRouter()` instances; mark `FastAPI` apps as entrypoints and add `instance -> handler` edges for every `@app.get(...)`-style decorator (HTTP methods, websockets, middleware, exception handlers, `on_event`). Routers stay pass-through, so an `APIRouter` that's never `include_router`'d remains dead (`--plugin fastapi`) |
| `FlaskPlugin` | Detect top-level `Flask()` / `Blueprint()` instances; mark `Flask` apps as entrypoints and add `instance -> handler` edges for every `@app.route(...)` / `@app.get(...)` / lifecycle / errorhandler / template-helper / URL-processor decorator. Blueprints stay pass-through, so a `Blueprint` that's never `register_blueprint`'d remains dead (`--plugin flask`) |
| `TyperPlugin` | Detect top-level `Typer()` instances and add `instance -> handler` edges for every `@app.command(...)` / `@app.callback(...)` decorator. Typer apps are pass-through (reach them via `[project.scripts]` or `if __name__ == "__main__": app()`), so a sub-typer that's never `add_typer`'d stays dead (`--plugin typer`) |
| `ClickPlugin` | Detect top-level Click `Group` instances (functions decorated `@click.group(...)` or `X = click.Group(...)`) and add `instance -> handler` edges for every `@cli.command(...)` / `@cli.group(...)` / `@cli.result_callback(...)` decorator. Groups are pass-through (reach them via `[project.scripts]` or a `__main__` block), so a sub-group that's never `add_command`'d stays dead (`--plugin click`) |
| `InitSubclassPlugin` | Detect classes that define `__init_subclass__` and add `parent -> subclass` edges for every (transitive) first-party subclass. Parents stay pass-through, so a registry base class only keeps subclasses alive once something else (an entrypoint, an import) keeps the parent alive (`--plugin init_subclass`) |

For project-specific dynamic-import patterns, two abstract bases ship as scaffolding that subclasses configure in 4-5 lines:

| Abstract base | Use it for |
|---|---|
| `DecoratedDeclPlugin` | "Find decorated decls in files matching a search path." Subclass with `package_prefix`, `decorator_module`, `decorator_names`, `constructor_names`. Pure observe-time. |
| `LiteralListPlugin` | "Read `<owner>.<var> = ['fqn', ...]` and treat each entry as alive." Subclass with `owner_fqname`, `variable_name`. observe parses and caches; finalize only does graph lookups. |

Both bases require subclasses to set `name` (a unique identifier for the cache namespace) and `version` (a Unix epoch int — bump it to the current epoch when the subclass's config changes). For example:

```python
from dataclasses import dataclass
from dead_cst import LiteralListPlugin

@dataclass(kw_only=True)
class MyInternalModulesPlugin(LiteralListPlugin):
    owner_fqname: str = "myapp.config"
    variable_name: str = "INTERNAL_MODULES"
    name: str = "my_internal_modules"
    version: int = 1700000000
```

Write your own from scratch by implementing the `EdgePlugin` protocol (`name`, `version`, `observe`, `finalize`); register under the `dead_cst.plugins` entry-point group for CLI discovery.

Path resolution is similarly pluggable. `PathResolver` implementations return a `{base: [dep_paths]}` map to feed `build_symbol_graph`. Builtins: `VenvResolver`, `PyprojectResolver`, `UvWorkspaceResolver` (parses `uv.lock` to discover workspace members and their inter-member dep edges). Third-party resolvers register under `dead_cst.resolvers`.

Unreachable-code detection is pluggable through the `UnreachableRegionDetector` protocol. `build_symbol_graph` accepts an `unreachable_detector` whose `find_regions(wrapper) -> list[CodeRange]` is invoked once per file. The built-in `DefaultUnreachableRegionDetector` covers three things out of the box:

- **Literal truthiness** on every `if` / `while` test (e.g. `if False:` always-dead body, `if True: ... else: ...` always-dead else).
- **Fixpoint constant folding** over simple `Name = literal` (and `Name: T = literal`) assignments. Chains like `foo = False; bar = foo or False; if bar: ...` resolve to dead because each fixpoint pass propagates one more level of indirection.
- **Post-terminator regions** inside every suite. Statements after an unconditional `return` / `raise` / `break` / `continue` / `assert <statically-falsy>` in the same suite are marked dead. Suite-relative, so a `raise` in a `try` body kills only the rest of the try body — the `except` handler still runs on its own path.

To layer in domain knowledge — e.g. config flags whose values are fixed in production — subclass and override `resolve(self, expr) -> bool | None`. The override gets first crack at every non-keyword expression in every `if` / `while` / `assert` test and every foldable assignment RHS; returning `None` defers to the built-in literal handling. Constants resolved this way flow through the same fixpoint loop as `Name = literal` bindings, so a single high-level decision propagates through chains:

```python
from dataclasses import dataclass

import libcst as cst
from dead_cst import build_symbol_graph
from dead_cst._branches import DefaultUnreachableRegionDetector

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

graph = build_symbol_graph({root: []}, unreachable_detector=FlagAwareDetector())
```

With the override above, `if check_flag("migration-abc"): ...` and `flag = check_flag("migration-abc"); if flag: ...` both resolve to a known truthiness, and the unreachable suite is flagged just like a literal `if False:` would be.

For detectors that don't fit the constant-folding model at all, write a fresh class that implements `find_regions(wrapper) -> list[CodeRange]` directly — the protocol requires nothing else beyond the `Cacheable` `(name, version)` pair.

## Graph model

The graph has one node per top-level declaration plus a synthetic module node per file. Edges run from a declaration to each symbol it references, and from every submodule to its parent package so `__init__.py` stays alive as long as anything in the package does. Entrypoints seed the reachability walk; every node not reached is reported as dead.

A module-level `import` / `from ... import ...` is itself a declaration of type `"import"` in the current module. Uses of the imported name inside the file are wired through that local import node, and the import node in turn points at the upstream module (and, when applicable, at the specific imported symbol). Removing the last local use therefore makes the import itself dead, which is how `dead-cst remove` knows to drop now-unused import lines.

## Scope

`dead-cst` tracks top-level declarations only -- module-level functions, classes, and variables. Nested definitions (inner functions, methods, nested classes) are deliberately not given their own nodes; references made from inside those nested scopes are attributed to the enclosing top-level declaration. Keeping the containing top-level symbol alive keeps its nested source alive with it.

## Limitations

- `import *` is treated pessimistically: every top-level declaration in the target module is considered used by the importing module.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- Only first-party code is analysed; third-party dependencies are treated as opaque (they appear as synthetic nodes — see `dead-cst dependencies`).
- PEP 695 `type` statements are not tracked.
- `__all__` is followed only when assigned a list/tuple of string literals; dynamic mutation (`__all__.append`, comprehensions, etc.) is not tracked.
- Walrus expressions (PEP 572) at module scope — including walruses leaked from a module-level comprehension — bind a name in the module namespace at runtime but are not surfaced as top-level declarations, so cross-decl references to those names are missed. The default unreachable-region detector likewise ignores walrus: `(FLAG := False)` and `if (FLAG := False):` don't enter the constant-folding table even when the RHS is a literal.
- PEP 750 template strings (`t"..."`, 3.14+) cannot be parsed by the pinned `libcst`, so any file containing one aborts the analysis with a `ParserSyntaxError`.

## Development

```bash
git clone https://github.com/lpetre/dead-cst
cd dead-cst
uv sync
uv run pytest
uv run prek run --all-files
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full dev guide, [`CHANGELOG.md`](CHANGELOG.md) for release notes, and [`ROADMAP.md`](ROADMAP.md) for the stack-ranked plan toward 1.0.

## TODO

- Host API documentation on Read the Docs.

## License

MIT — see [`LICENSE`](LICENSE).
