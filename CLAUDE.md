# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

This project uses [uv](https://github.com/astral-sh/uv); always run tools through `uv run` so the locked dev environment is used.

```bash
uv sync                                # install package + dev group in editable mode
uv run pytest                          # full test suite
uv run pytest tests/test_branches.py   # one file
uv run pytest -k name_substring        # one test by name
uv run pytest --cov=dead_cst --cov-branch --cov-report=term-missing
uv run ptw                             # pytest-watcher: tight inner loop
uv run prek run --all-files            # ruff (lint+format), ty type-check, pre-commit hooks
```

CI runs `uv run prek run --all-files` and the matrix `uv run pytest` (Python 3.11–3.14) on every push/PR. Run prek locally before committing to avoid round-trips. Coverage is uploaded only from the 3.13 matrix entry.

A `PostToolUse` hook in `.claude/settings.json` automatically runs `ruff format`, `ruff check --fix` (preserving unused imports — F401 is unfixable), and `ty check` after every Write/Edit on a `.py` file. There is no need to invoke these manually after editing.

## Architecture

`dead-cst` builds a symbol-level reachability graph of a Python codebase using [libcst](https://github.com/Instagram/LibCST) and [networkx](https://networkx.org/), walks from configured entrypoints, and reports (or removes) anything unreachable. The flow is staged so each stage has one job; understanding the staging is the key to navigating the code.

### Pipeline (top-down)

1. **Path resolution** (`dead_cst/_resolvers/`). A `PathResolver` returns a `PathMap` (`{base: [dep_paths]}`) plus an import resolver. Builtins: `VenvResolver` (sibling `.venv`), `PyprojectResolver` (`[tool.dead-cst]` paths or `src/` fallback), `UvWorkspaceResolver` (parses `uv.lock` to discover workspace members + inter-member dep edges), `ManualResolver`. Multiple resolvers chain via `_chain_resolvers` — first non-`None` wins. `_resolvers/_imports.py` does name→path lookup; `_resolvers/_exports.py` computes which subdirs are exposed to consumers.
2. **Per-file visitor** (`dead_cst/_visitor.py`). `SymbolVisitor` walks each `.py` file, returning a `VisitorPayload` (`nodes`, `edges`, `imports`, `module_node`). Top-level decls become `SymbolNode`s; references and module-level imports are recorded for later edge stitching. Heavy lifting: LibCST `ScopeProvider`, `FullyQualifiedNameProvider` (wrapped via `_fqn.FixedFullyQualifiedNameProvider`), `_flow.py` for flow-sensitive shadowing analysis, `_branches.py` for statically-dead suite detection (refs from those suites are tagged `EdgeFlags.DEAD_BRANCH`).
3. **Visitor cache** (`dead_cst/_cache.py`). SQLite-backed `GraphCache` at `<root>/.dead-cst-cache/` stores pickled `VisitorPayload`s keyed by file SHA-256. A `meta.fingerprint` row covers the dead-cst version, Python version, search paths, and resolver chain — mismatch wipes the cache. **Plugins are intentionally not part of the fingerprint**; they run after payload application, so swapping plugins reuses cached visitor work. Bypass with `--no-cache`.
4. **Edge stitching** (`dead_cst/_edges.py`). `resolve_edges` runs unconditionally (not cached). It takes `(src, Import, flags)` triples from each payload, walks the per-base `SymbolTrie`, and yields concrete `(src_symbol, dst_symbol, flags)` edges — following re-exports, fanning star imports out to every top-level decl, emitting synthetic nodes for stdlib/external/unresolved targets. Re-running this every analysis is what makes single-file edits cheap (only the edited file's payload is recomputed; all importers re-stitch for free).
5. **Plugins** (`dead_cst/_plugins/`). Run once per base in topological order, after that base's edges are resolved. They emit `GraphOp` values (`AddNode`, `AddEdge`, `RemoveEdge`) — never mutate the graph directly. `PluginContext` provides helpers (`parse`, `importers`, `base_modules`, …). Plugins encode knowledge static analysis can't infer: entrypoints, framework conventions (FastAPI/Flask/Click/Typer route handlers), `__init_subclass__` registries, pytest/unittest discovery, `if __name__ == "__main__":`, `[project.scripts]`, `__all__`/`__version__` dunders. Builtins are registered in `BUILTIN_PLUGINS`; out-of-tree plugins use the `dead_cst.plugins` entry-point group.
6. **Reachability** (`_analyze.find_reachable`). BFS successors from every node with `entrypoint=True`. Default traversal **does** follow `DEAD_BRANCH` edges (preserving today's behavior). `find_kept_alive_by_dead_branches` is the opt-in inverse: returns the blast radius of removing every dead suite by skipping those edges.
7. **Codemod** (`dead_cst/_codemod.py`). `remove_code` runs a LibCST `RemoveDeadSymbols` transformer keyed on `(fqname, CodeRange)` pairs (position disambiguates shadowed decls), then prunes now-unused imports via `RemoveImportsVisitor`. Position keying is critical — losing it conflates a dead decl with a live shadow.

### Graph model invariants

- One node per top-level declaration plus one synthetic module node per file. **Nested defs (inner functions, methods, nested classes) are deliberately not given their own nodes** — refs from inside them are attributed to the enclosing top-level decl.
- A module-level `import` / `from ... import ...` is itself a node of type `"import"`. Local uses of an imported name go through the import node, which points at the upstream module/symbol. This is how `dead-cst remove` knows to drop now-unused imports.
- Submodules edge to their parent package, so `__init__.py` stays alive as long as anything in the package does.
- `EdgeFlags.DEAD_BRANCH` is metadata only; tests using the `assert_edges` fixture see those edges, while `assert_dead_branch_edges` filters to just them.
- `NodeFlags.SHADOWED` decls are emitted into the graph (with their parent module edge) but excluded from the trie, so cross-module imports never resolve to them.

### Public API surface

Only names re-exported from `dead_cst/__init__.py` are supported. Modules prefixed with `_` are internal. Stable extension points are the `EdgePlugin` and `PathResolver` protocols (and their `dead_cst.plugins` / `dead_cst.resolvers` entry-point groups).

## Tests

Tests are fixture-driven from inline source snippets — no checked-in `.py` fixture files under `tests/`. Two patterns dominate (defined in `tests/conftest.py`):

```python
def test_something(build_decl_graph, assert_edges):
    graph = build_decl_graph({"mod.py": "def a(): pass\na()"})
    assert_edges(graph, {"mod.a -> mod", "mod -> mod.a"})
```

- `build_decl_graph({filename: source})` — writes dedented sources to `tmp_path` and runs `build_symbol_graph`.
- `assert_edges` / `assert_positional_edges` / `assert_dead_branch_edges` — compare edges as `"src.fqname -> dst.fqname"` strings (positional variant uses `fqname@line:col` to disambiguate redeclarations).
- Plugin tests live in `tests/test_plugins/` with their own `conftest.py`; resolver tests in `tests/test_resolvers/`.
- Codemod tests (`tests/test_codemod.py`) write a snippet, call `remove_code`, and assert on the rewritten text.
- `[tool.pytest.ini_options] testpaths = ["tests"]` is required — without it pytest collects the `examples/uv-workspace/packages/*/tests/` packages and fails with `ImportPathMismatchError` (which is the very layout that example exists to demonstrate).

## Coverage policy (informational, except patch gate)

Per-component thresholds in `codecov.yml` reflect blast radius, not LOC:

| Component | Target |
|---|---|
| `_codemod.py` | 95% (rewrites user files) |
| `cli.py` | 80% |
| `_plugins/**`, `_resolvers/**` | 85% |
| `_analyze.py`, `_visitor.py`, `_flow.py`, `_branches.py`, `_symbols.py`, `_fqn.py` | 80% |

Per-component statuses are informational until Tier 1 of `ROADMAP.md` lands. The **patch-coverage gate on new code is enforcing at 90%**.

## Adding a plugin

1. Implement `EdgePlugin` (or `CSTAwareEdgePlugin` for LibCST metadata): `name: str` and `contribute(ctx) -> Iterable[GraphOp]`.
2. Drop it in `dead_cst/_plugins/<name>.py` and register in `BUILTIN_PLUGINS` (in `_plugins/__init__.py`).
3. `FastAPIPlugin` is the full-featured reference: walks each module's CST, recognises `FastAPI()`/`APIRouter()` instances, wires `instance -> handler` edges through decorator detection, keeps routers pass-through (an `APIRouter` never `include_router`'d stays dead).
4. Out-of-tree plugins register under the `dead_cst.plugins` entry-point group; `load_plugin` checks builtins first, then entry points.

## Adding a resolver

Implement `PathResolver`: `name: str` plus `resolve(project_root) -> PathMap` and `resolve_import(name, search_paths) -> str | Path | None`. Drop in `dead_cst/_resolvers/<name>.py`. Third-party resolvers register under `dead_cst.resolvers`.

## Conventions

- **Python 3.11+**. Ruff line length 100. `known-first-party = ["dead_cst"]`. `from __future__ import annotations` is used throughout the package.
- **`ty` type-checks `dead_cst/` only** — tests, examples, and workspace fixtures intentionally exercise untyped LibCST internals and the unresolved-import case the analyzer detects, so they aren't meaningful targets.
- **Versioning is tag-driven via `hatch-vcs`** — never hand-edit a version. `local_scheme = "no-local-version"` is required because PyPI rejects PEP 440 local-version identifiers; `SETUPTOOLS_SCM_OVERRIDES_FOR_*` does not work here (only `Configuration.from_file()` reads it).
- **PRs**: one logical change per PR, add/update tests, add a `[Unreleased]` `CHANGELOG.md` entry for user-visible changes.
- **Releases**: pushing a `vX.Y.Z` tag triggers `.github/workflows/publish.yml` (uv build + PyPI OIDC publish).

## Pre-release status

`dead-cst` is alpha. APIs, CLI flags, and output formats may change without notice. Never run `dead-cst remove` against uncommitted code. See `ROADMAP.md` for the trajectory toward 1.0.
