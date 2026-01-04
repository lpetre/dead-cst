# dead-cst

Python dead code analysis using [libcst](https://github.com/Instagram/LibCST).

## Features

- **Graph-based analysis**: Builds a dependency graph of all symbols (modules, classes, functions, variables, imports) and finds unreachable code from your entrypoints.
- **Cross-module tracking**: Follows imports across your codebase to accurately detect unused code.
- **Multiple entrypoint types**: Supports file paths, fully qualified names, and regex patterns.
- **Monorepo support**: Handle complex projects with multiple packages and search paths.
- **Code removal**: Optionally remove dead code automatically.

## Installation

```bash
pip install dead-cst
```

Or with uv:

```bash
uv add dead-cst
```

## Quick Start

Analyze a project for dead code:

```bash
dead-cst analyze ./src --entrypoint main.py
```

With multiple entrypoints:

```bash
dead-cst analyze ./src \
    --entrypoint main.py \
    --entrypoint "re:tests/.*\.py" \
    --entrypoint "mypackage.api.public_function"
```

## CLI Reference

### `dead-cst analyze`

Analyze a Python codebase for dead code.

```bash
dead-cst analyze ROOT [OPTIONS]
```

**Arguments:**
- `ROOT`: Root directory to analyze

**Options:**
- `-p, --path SPEC`: Search path specification (see below). Can be repeated.
- `-e, --entrypoint EP`: Entrypoint specification. Can be repeated. Required.
- `--preserve-dunder-all/--no-preserve-dunder-all`: Keep `__all__` variables alive (default: true)
- `-v, --verbose`: Enable verbose output
- `--format [text|json]`: Output format (default: text)

**Entrypoint formats:**
- File path: `main.py`, `src/app.py`
- Fully qualified name: `mypackage.module.function`
- Regex pattern: `re:tests/.*\.py`

**Path specification format:**

For monorepos with multiple packages that depend on each other:

```bash
# Package with no dependencies
--path "libs/core"

# Package that depends on libs/core
--path "libs/utils:libs/core"

# Package that depends on multiple others
--path "src:libs/core,libs/utils"
```

### `dead-cst why-alive`

Explain why a symbol is considered reachable.

```bash
dead-cst why-alive ROOT FQNAME [OPTIONS]
```

**Arguments:**
- `ROOT`: Root directory
- `FQNAME`: Fully qualified name of the symbol to check

**Example:**

```bash
dead-cst why-alive ./src "mypackage.utils.helper_function"
```

### `dead-cst remove`

Remove dead code from the codebase.

```bash
dead-cst remove ROOT [OPTIONS]
```

**Options:**
- `--dry-run`: Show what would be removed without making changes
- Same options as `analyze`

## Python API

```python
from pathlib import Path
from dead_cst import build_symbol_graph, find_reachable, count_nodes

# Define your project structure
root = Path("./src")
paths = {
    root: [],  # No dependencies
    # Or for monorepos:
    # Path("./libs/core"): [],
    # Path("./src"): [Path("./libs/core")],
}

# Build the symbol graph
graph = build_symbol_graph(paths)

# Define entrypoints
entrypoints = [
    "main.py",
    "mypackage.api.public_function",
]

# Find reachable symbols
reachable = find_reachable(graph, root, entrypoints)

# Get unreachable (dead) symbols
dead = [n for n in graph.nodes if n not in reachable]

# Count by type
print(count_nodes(graph, root))
```

## Limitations

- **`import *`**: Currently not fully supported. Symbols imported via `from module import *` may be incorrectly marked as dead.
- **Dynamic imports**: Code loaded via `importlib.import_module()`, `__import__()`, or similar dynamic mechanisms is not tracked.
- **String references**: References to symbols via strings (e.g., `getattr(obj, "method_name")`) are not detected.
- **Framework magic**: Some frameworks use decorators or metaclasses that create implicit references. You may need to add additional entrypoints for:
  - FastAPI/Flask route handlers
  - pytest fixtures
  - Celery tasks
  - Django views/models

## How It Works

1. **Parse**: Uses libcst to parse all Python files and extract symbol definitions and references.
2. **Build graph**: Creates a directed graph where nodes are symbols and edges represent "uses" relationships.
3. **Find reachable**: Starting from entrypoints, traverses the graph to find all reachable symbols.
4. **Report**: Symbols not in the reachable set are reported as dead code.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

MIT License - see [LICENSE](LICENSE) for details.
