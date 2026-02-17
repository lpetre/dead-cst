# dead-cst

Python dead code analysis using [libcst](https://github.com/Instagram/LibCST).

`dead-cst` builds a full symbol graph of your Python codebase, walks from your entrypoints, and reports (or removes) anything unreachable.

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

## Limitations

- Only top-level declarations (functions, classes, variables) are tracked; nested definitions are not individually reported.
- `import *` is not resolved.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- Only first-party code is analysed; third-party dependencies are treated as opaque.
