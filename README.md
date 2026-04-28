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

Exit code 1 if dead code is found, 0 otherwise.

### `dead-cst why-alive`

Show why a symbol is considered alive by printing its predecessor chain.

```
dead-cst why-alive ROOT FQNAME [OPTIONS]
```

### `dead-cst unused-exports`

Report `__all__` entries whose targets would be dead without `__all__`. Useful in closed-world / monorepo settings to prune the public surface.

```
dead-cst unused-exports ROOT -e ENTRYPOINT [OPTIONS]
```

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

### `dead-cst remove`

Remove dead code from a Python codebase. Prompts for confirmation before modifying files.

```
dead-cst remove ROOT -e ENTRYPOINT [OPTIONS]
```

| Option | Description |
|---|---|
| `--dry-run` | Show what would be removed without making changes |

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

Entrypoint detection is now fully plugin-driven. Builtins:

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

Write your own by implementing the `EdgePlugin` or `CSTAwareEdgePlugin` protocol; register under the `dead_cst.plugins` entry-point group for CLI discovery.

Path resolution is similarly pluggable. `PathResolver` implementations return a `{base: [dep_paths]}` map to feed `build_symbol_graph`. Builtins: `VenvResolver`, `PyprojectResolver`, `UvWorkspaceResolver` (parses `uv.lock` to discover workspace members and their inter-member dep edges). Third-party resolvers register under `dead_cst.resolvers`.

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
