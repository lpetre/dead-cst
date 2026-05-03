# Contributing to dead-cst

Thanks for your interest. `dead-cst` is small and pre-release — bug reports with minimal repros, real-world test cases, and PRs are all welcome. Expect APIs and CLI flags to keep moving until the first stable release. See [`ROADMAP.md`](ROADMAP.md) for the planned trajectory.

## Development setup

`dead-cst` uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone https://github.com/lpetre/dead-cst
cd dead-cst
uv sync
```

That installs the package in editable mode along with the `dev` group: `pytest`, `ruff`, `prek`, and `ty`.

## Running tests

```bash
uv run pytest
```

For a tight inner loop while editing the visitor, resolver, or a plugin, `pytest-watcher` is in the dev group:

```bash
uv run ptw
```

To collect coverage locally:

```bash
uv run pytest --cov=dead_cst --cov-branch --cov-report=term-missing
```

CI uploads branch coverage from the 3.13 matrix entry to Codecov on every push and PR.

## Coverage policy

Per-component thresholds reflect blast radius, not total LOC. The full
configuration lives in `codecov.yml`; the targets are:

| Component | Target | Why |
|---|---|---|
| `_codemod.py` | 95% | Rewrites user files. Regressions corrupt source. |
| `cli.py` | 80% | User-visible trust surface. |
| `_plugins/**` | 85% | Each plugin is a framework promise. |
| `_resolvers/**` | 85% | Path resolution drives every analysis. |
| `_analyze.py`, `_visitor.py`, `_resolve.py`, `_flow.py`, `_branches.py`, `_symbols.py`, `_fqn.py` | 80% | Analytical core. |

Per-component statuses are **informational** until Tier 1 of `ROADMAP.md`
(CLI integration tests, remaining framework presets) lands and the numbers
clear the bars above. The patch-coverage gate on new code is **enforcing**
at 90% — regressions are caught at the diff, not in aggregate.

`if TYPE_CHECKING:` blocks, `@overload` stubs, and `_version.py` are
excluded; see `[tool.coverage.*]` in `pyproject.toml`.

## Linting and formatting

Ruff (lint + format), `ty` (type check), and a small set of hooks run on every commit via [`prek`](https://github.com/j178/prek), a fast Rust-based drop-in replacement for `pre-commit` that reads the same `.pre-commit-config.yaml`.

```bash
uv run prek install        # one-time, sets up the git hook
uv run prek run --all-files
```

CI runs `prek run --all-files` on every push and pull request, so running it locally before committing avoids round-trips.

## Project layout

```
dead_cst/
  __init__.py            # public API surface
  _analyze.py            # build_symbol_graph, find_reachable, count_nodes, order_paths
  _codemod.py            # remove_code -- LibCST transformer + import pruner
  _resolve.py            # import resolution (stdlib / first-party / third-party)
  _resolvers/            # path resolvers:
    venv.py              #   sibling .venv -> site-packages
    pyproject.py         #   [tool.dead-cst] paths or src/ fallback
    uv_workspace.py      #   uv.lock workspace members + inter-member deps
    _exports.py          #   exported_roots: hide internal dirs from consumers
  _symbols.py            # SymbolNode, SymbolTrie data classes
  _visitor.py            # SymbolVisitor -- walks each file and emits symbols + edges
  _flow.py               # flow-sensitive live-at-exit analysis for shadowing
  _branches.py           # statically-dead suite detection
  _fqn.py                # FullyQualifiedNameProvider patches / wrappers
  _plugins/              # edge plugins:
    main_block.py        #   if __name__ == "__main__"
    project_scripts.py   #   pyproject.toml [project.scripts]
    explicit.py          #   user-supplied -e specs
    module_dunders.py    #   __all__, __version__, ...
    pytest.py            #   conftest, test_*.py, @pytest.fixture
    fastapi.py           #   FastAPI / APIRouter route handlers
  cli.py                 # Typer entrypoints: analyze, why-alive, unused-exports,
                         #                    dependencies, remove
tests/                   # pytest suite, fixture-driven from inline source snippets
```

Modules prefixed with `_` are internal; only the names re-exported from `dead_cst/__init__.py` are part of the supported API. Within the package, plugins and resolvers are stable extension points — see below.

## Adding a plugin

A plugin is any class that satisfies `EdgePlugin` (or `CSTAwareEdgePlugin` if it needs LibCST metadata). Implement `name: str` and a `contribute(ctx)` method that yields `AddNode`, `AddEdge`, or `RemoveEdge` ops.

```python
from dead_cst import AddEdge, AddNode, GraphOp, PluginContext, synthetic_node

class FlaskRoutesPlugin:
    name = "flask_routes"

    def contribute(self, ctx: PluginContext):
        for node in ctx.graph.nodes:
            ...  # detect @app.route handlers and emit AddNode(..., entrypoint=True)
```

Built-in plugins live in `dead_cst/_plugins/<name>.py`; out-of-tree plugins register under the `dead_cst.plugins` entry-point group:

```toml
[project.entry-points."dead_cst.plugins"]
flask_routes = "myproj.plugins:FlaskRoutesPlugin"
```

`FastAPIPlugin` is a good full-featured reference — it walks each module's CST, recognises the `FastAPI()` / `APIRouter()` instance pattern, and wires `instance -> handler` edges through decorator detection.

## Adding a resolver

A resolver implements `PathResolver`: a `name: str` attribute and a `resolve(project_root)` method returning a `{base: [dep_paths]}` dict. Built-in resolvers live in `_resolvers.py`; third-party resolvers register under `dead_cst.resolvers`.

## Adding an unreachable-region detector

A detector implements `UnreachableRegionDetector`: `find_regions(wrapper) -> list[CodeRange]` plus the standard `Cacheable` `(name, version)` pair. The built-in `DefaultUnreachableRegionDetector` covers literal-only truthiness on `if` / `while` tests, fixpoint constant-folding over simple `Name = literal` assignments, and post-terminator regions inside every suite. To layer in domain knowledge — e.g. "`check_flag("migration-abc")` is always `True` in production" — subclass it and override `resolve(self, expr) -> bool | None`. The override is consulted recursively for every non-keyword expression in every `if` / `while` / `assert` test and every foldable assignment RHS, so guard with an early `isinstance` check to keep it cheap. Returning `None` defers to the built-in literal handling. For detectors that don't fit the constant-folding model at all, write a fresh class that implements `find_regions` directly. Pass an instance via `build_symbol_graph(unreachable_detector=...)`. The detector's `(name, version)` participates in the per-file cache fingerprint, so bumping `version` (epoch-int by convention) after a logic change invalidates stale `VisitorPayload` blobs automatically.

## Adding a test

Tests use the `build_decl_graph` fixture (in `tests/conftest.py`), which writes a dict of `{filename: source}` to a tmpdir, runs `build_symbol_graph` on it, and returns the resulting graph. The `assert_edges` fixture compares edges as `"src.fqname -> dst.fqname"` strings.

```python
def test_something(build_decl_graph, assert_edges):
    graph = build_decl_graph({"mod.py": "def a(): pass\na()"})
    assert_edges(graph, {"mod.a -> mod", "mod -> mod.a"})
```

Plugin tests follow the same pattern in `tests/test_plugins/`. For codemod regressions, `tests/test_codemod.py` writes a source snippet, runs `remove_code`, and asserts on the rewritten text.

## Reporting bugs

A good bug report contains a minimal `.py` file (or pair of files) and the entrypoint flag you ran, plus the actual vs. expected dead-symbol output. The smaller the repro, the faster it gets fixed.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Add or update tests for behaviour changes.
- Run `prek run --all-files` and `pytest` before pushing.
- If your change is user-visible, add an entry to `CHANGELOG.md` under `[Unreleased]`.

## Releasing

Releases are tag-driven. Pushing a `vX.Y.Z` tag triggers `.github/workflows/publish.yml`, which builds the package with `uv build` and publishes to PyPI via OIDC. The version is read from the tag by `hatch-vcs`, so no manual version bumps in source files are needed.
