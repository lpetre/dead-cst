---
name: dead-cst
description: Use when running, configuring, or scripting `dead-cst` — the libcst-based dead-code analyzer. Covers entrypoint specs, path layouts, resolver/plugin pairings (especially `uv_workspace` + `uv sync --all-packages`), the dead-symbols vs. unreachable-branches output split, the safety contract around `dead-cst remove`, and the workflow for pruning unused `[project.dependencies]` from `pyproject.toml` via `dead-cst dependencies`. Trigger on `dead-cst analyze|why-alive|remove|dependencies|unused-exports`, on edits to `[tool.dead-cst]` / `[project.dependencies]` in `pyproject.toml`, or when `from dead_cst import ...` appears.
---

# dead-cst

Static dead-code analyzer for Python. Builds a symbol graph (one node per top-level decl + one per module), walks from entrypoints, reports/removes whatever isn't reachable.

## Mental model

- **Nodes**: top-level functions, classes, variables, and imports — *plus* a synthetic module node per file. Nested defs (inner functions, methods) are **not** separate nodes; references inside them attribute to the enclosing top-level symbol.
- **Edges**: declaration → each name it references; submodule → parent package (so `__init__.py` stays alive while the package does).
- **Imports are declarations**: a module-level `import foo` is itself a node. The last in-file use disappearing makes the import dead, which is how `remove` knows to drop unused import lines.
- **Reachability**: entrypoints (contributed by plugins) seed the walk. Anything not reached is dead.

## Two kinds of output: dead symbols vs. unreachable branches

`analyze` (and the JSON output) reports **two separate categories**:

1. **Dead symbols** — top-level decls / imports the reachability walk never reaches. These are what `remove` deletes.
2. **Unreachable branches** — suites inside `if` / `while` whose test is statically known to never fire (or always fire, making `else` / `while-else` dead). Reported as `path:line:col-line:col` ranges.

`remove` **does not delete unreachable branches** — the codemod only handles whole top-level decls, not arbitrary suites. They're report-only; rip them out by hand.

What counts as "statically known" is a deliberately small whitelist (see `_branches.py`):

- The keywords `True` / `False` / `None`
- Integer and string literals
- Empty vs. non-empty `list` / `tuple` / `set` / `dict` literals (no `*splat` / `**splat`)
- `not` / `and` / `or` composed over the above

Anything else — name lookups (other than the three keywords), attribute access, function calls, comparisons (`x == 0`), `sys.version_info`, `TYPE_CHECKING`, env-var checks — returns "unknown" and the branch stays live. This is intentional: false positives here would silently drop real code.

Concretely, dead-cst flags:

```python
if False: ...                    # body dead
if True: ... else: ...           # else dead
if 0 or "": ...                  # body dead
while False: ...                 # body dead
while True: ...; else: ...       # else dead (while-True exits via break/return/exc)
if cond: ... elif True: ... else: ...  # else dead (a True branch fires first)
```

It does **not** flag `if TYPE_CHECKING:`, `if sys.version_info >= (3, 12):`, `if DEBUG:`, etc. — by design.

Internally, each dead suite becomes a synthetic node (`type="synthetic"`, fqname prefixed `<unreachable>:`); use `dead_cst._branches.is_unreachable_node` to identify them rather than string-matching the prefix.

## CLI shape

All commands take a positional `ROOT`. Common options:

| Flag | Form | Notes |
|---|---|---|
| `-e, --entrypoint` | `path/to/file.py` \| `pkg.module.func` \| `re:<pattern>` | Repeatable. `re:` = regex. **Required for `analyze`/`remove`/`unused-exports`** unless your plugins already seed entrypoints (e.g. `project_scripts`, `main_block`). |
| `-p, --path` | `base` \| `base:dep1,dep2` | Repeatable. `base` is analyzed; `dep`s are import-resolution-only (not reported as dead). |
| `--resolver` | `venv` \| `pyproject` \| `uv_workspace` \| third-party | Repeatable. Outputs are merged with `-p` specs. |
| `--plugin` | see plugin table | Repeatable. Order doesn't matter; user plugins run before `ModuleDundersPlugin` and `-e` (which always run last). |
| `--format` | `text` \| `json` |  |

`analyze` exits 1 if dead code is found, 0 otherwise — useful in CI.

## Picking resolvers

Resolvers produce the `{base: [deps]}` map. **Pick one resolver per project shape**; they compose if needed.

| Shape | Resolver | What you also need |
|---|---|---|
| Single package, `src/` layout, deps in a venv | `--resolver pyproject --resolver venv` | A populated `.venv` (run `uv sync` or equivalent first). |
| Single package with custom paths | `--resolver pyproject` + `[tool.dead-cst].paths` | See config below. |
| uv workspace (multi-package, shared `.venv`) | `--resolver uv_workspace` | **`uv sync --all-packages`** — workspace members must be installed into the shared `.venv` or the resolver raises `MissingVenvError`. |
| Anything ad-hoc | just `-p base:dep1,dep2` | Skip resolvers entirely. |

`pyproject` config:

```toml
[tool.dead-cst]
paths = [
  { base = "src", deps = ["tests"] },
  { base = "scripts" },
]
```

Without `[tool.dead-cst]`, `pyproject` falls back to `src/` if it exists.

`uv_workspace` parses `uv.lock`, treating each `editable`/`virtual` package source as a member, wires direct workspace deps from the lockfile's `dependencies` array, and appends the shared `site-packages` so third-party imports resolve to `[external dist] <pkg>` instead of `[unresolved]`. The workspace root (`virtual = "."`) is intentionally skipped.

## Picking plugins

Two categories: **entrypoint plugins** (seed reachability) and **edge plugins** (add `parent → child` edges so a framework's "magically called" handlers stay alive when the framework instance does). `ModuleDundersPlugin` is always on (keeps `__all__`, `__version__`, etc. alive).

| Plugin | When to add it |
|---|---|
| `main_block` | Codebase has `if __name__ == "__main__":` blocks you want treated as entrypoints. |
| `project_scripts` | `pyproject.toml` declares `[project.scripts]` console scripts. (Often the primary entrypoint source — pairs naturally with `--resolver pyproject`.) |
| `pytest` | Project has a pytest suite. Keeps test files, `conftest.py` decls, and `@pytest.fixture` functions alive. |
| `unittest` | Stdlib unittest (`TestCase`, `IsolatedAsyncioTestCase`, `setUpModule`/`tearDownModule`/`load_tests`). |
| `fastapi` | FastAPI app. Marks `FastAPI()` instances as entrypoints; `APIRouter` is pass-through (an unused router stays dead until something `include_router`s it). |
| `flask` | Flask app. Same shape — `Flask()` is entrypoint, `Blueprint` is pass-through. |
| `typer` | Typer CLI. `Typer()` is pass-through; reach it via `[project.scripts]` or a `__main__` block. |
| `click` | Click CLI. Top-level `Group` instances; sub-groups stay dead unless `add_command`'d. |
| `init_subclass` | Registry pattern via `__init_subclass__`. Parent stays pass-through; subclasses light up once anything keeps the parent alive. |

## Pruning `pyproject.toml` dependencies

`dead-cst dependencies` lists every third-party module imported under each base. Use it to find dependencies declared in `[project.dependencies]` (or `[dependency-groups]`) that the codebase never actually imports.

```bash
# Text output
dead-cst dependencies . --resolver pyproject --resolver venv

# JSON for scripting (one list per base path)
dead-cst dependencies . --resolver pyproject --resolver venv --format json
```

Each entry is one of:

- `[external dist] <name>` — resolver matched the import to an installed distribution (the common case; `<name>` is the **distribution** name, e.g. `pyyaml`, not the import name `yaml`).
- `[external file] <name>` — resolved to a file inside `site-packages` but no matching dist metadata. Treat the same as `external dist` for pruning purposes.
- `[unresolved] <name>` — visible only in the graph, not in `dependencies` output. Means a venv resolver wasn't run, or the package isn't installed. Fix the venv before pruning.

### Workflow for "remove unused deps from pyproject.toml"

1. Run `dead-cst dependencies . --resolver pyproject --resolver venv --format json` (or `--resolver uv_workspace` for workspaces — remember `uv sync --all-packages` first).
2. Read `[project.dependencies]` (and any `[project.optional-dependencies]` / `[dependency-groups]` you're pruning).
3. For each declared dep, check whether its imported name appears in the dependencies output. **Distribution name ≠ import name** — common mismatches: `pyyaml`→`yaml`, `pillow`→`PIL`, `beautifulsoup4`→`bs4`, `python-dateutil`→`dateutil`, `protobuf`→`google.protobuf`, `opencv-python`→`cv2`, `scikit-learn`→`sklearn`. Hyphens vs. underscores are normalized in dist names but **not** in module names. When unsure, look at the dist's top-level packages on PyPI / in the installed `*.dist-info/RECORD`.
4. Before flagging a dep as unused, rule out:
   - **CLI-only tools** (`pytest`, `ruff`, `mypy`, `prek`, `coverage`) — invoked as a binary, never imported. They legitimately don't appear in `dependencies` output. They typically belong in `[dependency-groups]` / `[project.optional-dependencies]`, not runtime deps; if found in runtime deps, that's a different cleanup.
   - **Plugin / entry-point packages** loaded by `importlib.metadata.entry_points` (e.g. pytest plugins, dead-cst plugins) — never imported by name, but required at runtime.
   - **Build / packaging deps** (`hatchling`, `setuptools`, `hatch-vcs`) — declared in `[build-system].requires`, not `[project.dependencies]`, and never imported.
   - **Optional / extras-only deps** the user wires up conditionally.
5. **`dependencies` reports imports across the whole base, including dead modules.** A dep imported only by a file `analyze` reports as dead is still listed. To get a tighter "deps required by reachable code" view, run `analyze` first, exclude reported dead files from the source tree (or remove them), then re-run `dependencies`.
6. Propose pyproject edits and let the user confirm. Don't bulk-delete.

The reverse check — "imports that aren't declared in `pyproject.toml`" — is also useful: every name in `dependencies` output should appear (under its dist name) in `[project.dependencies]` or be a stdlib module. Anything left over is an undeclared transitive import (a latent bug).

## Recipes

```bash
# Single package, console-script entrypoint, pytest suite, deps in .venv
dead-cst analyze . \
  --resolver pyproject --resolver venv \
  --plugin project_scripts --plugin pytest

# uv workspace — run uv sync --all-packages FIRST
uv sync --all-packages
dead-cst analyze . --resolver uv_workspace --plugin project_scripts

# FastAPI service — entrypoint is the app module
dead-cst analyze ./src --resolver venv \
  --plugin fastapi -e myapp.main

# Ad-hoc, no resolver
dead-cst analyze ./src -p src:tests -e "re:.*__main__\.py"

# Why is X kept alive? (no -e needed; just inspects predecessors)
dead-cst why-alive ./src mypackage.foo.bar

# What does __all__ uselessly re-export?
dead-cst unused-exports ./src --plugin project_scripts

# What third-party deps does the codebase actually import?
dead-cst dependencies . --resolver pyproject --resolver venv --format json
```

## `dead-cst remove` — safety contract

- **Only run against a clean working tree.** The README says it explicitly: "Do not run `dead-cst remove` against code that isn't committed to version control." Recommend `git status` clean + a new branch before invoking.
- Use `--dry-run` first to preview.
- The codemod removes whole top-level decls and now-unused module-level imports. It does **not** rewrite arbitrary unreachable suites (those are reported as "Unreachable branches" by `analyze` but skipped by `remove`).
- After removal, re-run `analyze` — removing one symbol can make others newly dead.

## Common failure modes

- **`MissingVenvError`** from `venv` or `uv_workspace`: no `.venv` / no populated workspace venv. Fix: `uv sync` (single project) or `uv sync --all-packages` (workspace). Don't try to silence by dropping the resolver — framework plugins need `site-packages` to identify external imports as external rather than `[unresolved]`.
- **Everything reported dead**: usually missing entrypoints. Add `--plugin project_scripts` / `--plugin main_block` or pass `-e`.
- **Framework handlers reported dead**: missing the framework plugin (`fastapi`/`flask`/`typer`/`click`).
- **Subclasses reported dead even though a registry uses them**: add `--plugin init_subclass`.
- **`import *`** is treated pessimistically — every top-level decl in the target module is considered used. Don't expect dead-cst to flag dead names re-exported via `*`.
- **Dynamic access** (`getattr`, runtime symbol creation, PEP 695 `type` statements, dynamic `__all__` mutation) is invisible.

## Python API sketch

```python
from pathlib import Path
from dead_cst import build_symbol_graph, find_reachable, remove_code
from dead_cst._plugins import MainBlockPlugin, ExplicitEntrypointPlugin
import re

root = Path("./src")
graph = build_symbol_graph(
    {root: []},
    plugins=[MainBlockPlugin(), ExplicitEntrypointPlugin(specs=[re.compile(r".*__main__\.py")])],
    project_root=root,
)
reachable = find_reachable(graph)
unreachable = graph.subgraph([n for n in graph.nodes if n not in reachable])
remove_code(unreachable, root)
```

Custom plugins implement `EdgePlugin` / `CSTAwareEdgePlugin` and register under the `dead_cst.plugins` entry-point group. Custom resolvers register under `dead_cst.resolvers`.
