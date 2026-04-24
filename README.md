# dead-cst

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
| `--preserve-dunder-all / --no-preserve-dunder-all` | Keep `__all__` variables alive (default: true) |
| `--format` | Output format: `text` or `json` |
| `-v, --verbose` | Enable verbose logging |

Exit code 1 if dead code is found, 0 otherwise.

### `dead-cst why-alive`

Show why a symbol is considered alive by printing its predecessor chain.

```
dead-cst why-alive ROOT FQNAME [OPTIONS]
```

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
from pathlib import Path
from dead_cst import build_symbol_graph, find_reachable, remove_code

root = Path("./src")
graph = build_symbol_graph({root: []})
reachable = find_reachable(graph, root, ["re:.*__main__\\.py"])

unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable])
# Inspect unreachable nodes, or remove them:
remove_code(unreachable, root)
```

## Graph model

The graph has one node per top-level declaration plus a synthetic module node per file. Edges run from a declaration to each symbol it references, and from every submodule to its parent package so `__init__.py` stays alive as long as anything in the package does. Entrypoints seed the reachability walk; every node not reached is reported as dead.

A module-level `import` / `from ... import ...` is itself a declaration of type `"import"` in the current module. Uses of the imported name inside the file are wired through that local import node, and the import node in turn points at the upstream module (and, when applicable, at the specific imported symbol). Removing the last local use therefore makes the import itself dead, which is how `dead-cst remove` knows to drop now-unused import lines.

## Scope

`dead-cst` tracks top-level declarations only -- module-level functions, classes, and variables. Nested definitions (inner functions, methods, nested classes) are deliberately not given their own nodes; references made from inside those nested scopes are attributed to the enclosing top-level declaration. Keeping the containing top-level symbol alive keeps its nested source alive with it.

## Limitations

- `import *` is not resolved.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- Only first-party code is analysed; third-party dependencies are treated as opaque.
- PEP 695 `type` statements are not tracked.
- String names in `__all__` are not followed to their declarations (but `--preserve-dunder-all` keeps the `__all__` variable itself alive).

## TODO

- Host API documentation on Read the Docs.
