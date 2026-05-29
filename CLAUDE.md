# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

This project uses [uv](https://github.com/astral-sh/uv); always run tools through `uv run` so the locked dev environment is used.

```bash
uv sync                                # install package + dev group; triggers maturin
                                       # to compile src/ → python/dead_cst/_native.*.so
uv run pytest                          # full suite (e2e is deselected by default)
uv run pytest tests/test_imports.py    # one file
uv run pytest -k name_substring        # one test by name
uv run pytest -m e2e                   # opt-in e2e suite (clones real GitHub repos)
uv run pytest --cov=dead_cst --cov-branch --cov-report=term-missing
uv run ptw                             # pytest-watcher: tight inner loop
uv run prek run --all-files            # ruff (lint+format), ty type-check, pre-commit hooks
```

`pyproject.toml` pins `addopts = "-m 'not e2e'"`, so the standard `uv run pytest` is hermetic.

CI runs `prek run --all-files` and the matrix `pytest` (Python 3.11–3.14) on every push/PR.

## Architecture

`dead-cst` builds a symbol-level reachability graph of a Python codebase via the rust-backed `dead_cst._native` extension (which uses [ty's](https://github.com/astral-sh/ruff) `SemanticIndex`), walks from configured entrypoints, and reports (or removes) anything unreachable.

The native code is split: **`dead-cst-runtime`** (`runtime/`) holds the whole implementation, built as both an `rlib` and a `dylib`; **`dead-cst-native`** (`src/lib.rs`) is a thin pyo3 `#[pymodule]` shim. The **dev build** (`uv sync` / `maturin develop`) and the **Windows wheel** statically link the runtime `rlib` (one self-contained `_native.{abi3.so,pyd}`, no plugin loading). The **shipped macOS/Linux wheel** instead links the runtime `dylib`: the `_native` shim + `libdead_cst_runtime` + `libstd` ride in the package (resolved via `$ORIGIN`/`@loader_path`), so the host runs the shared runtime and external plugins load with no `_native` swap. The publish workflow builds the dynamic wheel by repacking maturin's static wheel (`scripts/repack_dynamic_wheel.py`). See [`NATIVE_PLUGINS.md`](NATIVE_PLUGINS.md) and `runtime/src/CLAUDE.md`.

### Pipeline (top-down)

1. **Path resolution** (`dead_cst/resolvers/`). A `PathResolver` returns a `tuple[Package, ...]` (`Package(path, name, deps)`; deps reference other packages by name). Builtins: `ManualResolver` (explicit `package:dep1,dep2` specs from the CLI's `-p`) and `UvResolver` (parses `uv.lock` to discover workspace members + inter-member dep edges; lives in `dead_cst/contrib/uv.py`). `Analysis(project_root, resolver=...)` takes exactly one resolver. Construction calls `resolver.resolve(project_root)` and validates the output via `_validate_packages`.
2. **Graph materialization** (rust crate `dead_cst._native`). `Analysis.materialize_all()` instantiates a `native.ProjectContext` rooted at the project root (each package's path is passed as a `src_root`), registers each plugin's `run(ctx)` callback, and calls `materialize()`. The rust backend uses ty's `SemanticIndex` to:
   - Parse every `.py` file under the project root and per-package `src_roots`.
   - Resolve every cross-file reference through ty's module resolver and use-def chains.
   - Emit one `kind="import"` node per import alias (and one per name pulled in by `from X import *`).
   - Mark dead-suite branches (`if False:`, code after `return`/`raise`, etc.) with `EdgeFlags.DEAD_BRANCH`.
   - Surface non-first-party imports as `[external dist] X` / `[external file] X` / `[unresolved] X` / `[stdlib] X` synthetic nodes.
3. **Plugin pass** (rust-side, during `materialize()`). Each registered plugin's `run(ctx)` is invoked with the in-progress `ProjectContext`. Plugins yield `AddNode` / `AddEdge` / `AddEntrypoint` ops; the rust apply pass folds them into the graph in one atomic step. After `materialize()` returns, `Analysis` holds the live `ProjectContext` — there is no Python-side adjacency copy. `SymbolNode`, `Import`, `NodeFlags`, and `EdgeFlags` are re-exports of the rust pyclasses. The compiled rust extension lives at `python/dead_cst/_native.{abi3.so,pyd}` (built by maturin from `src/`) and is imported as `from dead_cst import _native as native`.
4. **Reachability** (`Analysis.reachable` / `Analysis.dead` / `Analysis.descendants` / `Analysis.ancestors`). Bulk queries delegate to `ctx.reachable(...)` / `ctx.descendants(...)` / `ctx.ancestors(...)` — one FFI call per query, no per-node Python ↔ rust round-trips. Default traversal **does** follow `DEAD_BRANCH` edges (preserving today's behavior); `Analysis.kept_alive_by_dead_branches` is the opt-in inverse. `Analysis.kept_alive_by_flags_only(flags)` is the blast-radius query for any `NodeFlags` combination (`TESTCASE`, `NOQA`, or both). Path scoping is the caller's job: filter `analysis.dead()` by `Path(n.path).is_relative_to(pkg)` when you need a per-package slice.
5. **Codemod** (`dead_cst/codemod.py`). `remove_code(dead_nodes, package_path)` runs a LibCST `RemoveDeadSymbols` transformer keyed on `(fqname, start_line)` pairs (line disambiguates shadowed decls), then prunes now-unused imports via `RemoveImportsVisitor`. The codemod is the only remaining libcst-using stage — it's a pure source rewriter, not a graph builder. `generate_patch(dead_nodes, root)` is the non-destructive twin: returns a `git apply`-compatible unified diff. The `dead-cst remove` CLI uses `generate_patch` exclusively. Both functions take an iterable of dead `SymbolNode`s — edges are ignored, so callers can slice the unreachable set however they like (per SCC, per file, …) without rebuilding a graph.

### Graph model invariants

- One node per top-level declaration plus one synthetic module node per file. **Nested defs (inner functions, methods, nested classes) are deliberately not given their own nodes** — refs from inside them are attributed to the enclosing top-level decl.
- A module-level `import` / `from ... import ...` is itself a node of type `"import"`. Local uses of an imported name go through the import node, which points at the upstream module/symbol. This is how `dead-cst remove` knows to drop now-unused imports.
- Every import binds a local `kind="import"` node. A use of an imported name always emits an edge to its local alias (codemod invariant: an unused import has zero in-edges). When ty's module resolver / global-scope lookup can pin the use to specific upstream targets, the use *also* emits direct edges to each of them (reachability edges). See `src/CLAUDE.md` for the full contract.
- Imports whose source line carries a ruff/pyflakes `# noqa` directive that silences F401 are flagged `NodeFlags.ENTRYPOINT | NodeFlags.NOQA` so reachability keeps them alive. File-level `# ruff: noqa` and `# flake8: noqa` directives pin every import in the file.
- Submodules edge to their parent package, so `__init__.py` stays alive as long as anything in the package does.
- `EdgeFlags.DEAD_BRANCH` is metadata only; tests using the `assert_edges` fixture see those edges, while `assert_dead_branch_edges` filters to just them.
- `NodeFlags.SHADOWED` decls are emitted into the graph (with their parent module edge) but are flagged so cross-module imports never resolve to them. Two `def f` at different lines stay as distinct nodes.
- `.pyi` stub files are ingested only for the compiled-extension layout (`_native.so` + `_native.pyi`, no `.py` twin).
- `NodeFlags.NOTEBOOK` tags every node sourced from a Jupyter `.ipynb` file. The codemod skips notebook nodes (cell-aware writeback is out of scope today).

### Public API surface

The supported surface is whatever is re-exported from a module named without a leading `_`. The top-level `dead_cst/__init__.py` re-exports the highlights (`Analysis` plus the graph data types). Deeper symbols live in focused public submodules:

- `dead_cst.graph` — `SymbolNode`, `Import`, `NodeFlags`, `EdgeFlags`.
- `dead_cst.analyze` — `Analysis`.
- `dead_cst.codemod` — `remove_code` and `generate_patch`.
- `dead_cst.plugins` — the synthetic-node prefix constants and every built-in plugin.
- `dead_cst.resolvers` — `PathResolver`, `ManualResolver`. (`UvResolver` lives under `dead_cst.contrib`.)
- `dead_cst.contrib` — third-party-aware extensions.

`tests/test_public_api.py` pins each public module's `__all__` against a snapshot.

## Tests

Tests are fixture-driven from inline source snippets — no checked-in `.py` fixture files under `tests/`. Two patterns dominate (defined in `tests/conftest.py`):

```python
def test_something(build_decl_graph, assert_edges):
    graph = build_decl_graph({"mod.py": "def a(): pass\na()"})
    assert_edges(graph, {"mod.a -> mod", "mod -> mod.a"})
```

- `build_decl_graph({filename: source})` — writes dedented sources to `tmp_path` and runs `Analysis(...).materialize_all()`.
- `make_analysis(specs=["."], **kwargs)` — lower-level fixture for tests that need extra `Analysis` kwargs.
- `assert_edges` / `assert_positional_edges` / `assert_dead_branch_edges` / `assert_dynamic_import_edges` — compare edges as `"src.fqname -> dst.fqname"` strings.
- Plugin tests live in `tests/test_plugins/` with their own `conftest.py`; resolver tests in `tests/test_resolvers/`.
- Codemod tests (`tests/test_codemod.py`) write a snippet, call `remove_code`, and assert on the rewritten text.
- E2E tests (`tests/e2e/`) shallow-clone real repos at pinned SHAs. Behind the `e2e` marker, excluded by default.

## Adding a plugin

1. Define a class with `name: str`, `version: int` (epoch), and `run(ctx: native.ProjectContext) -> Iterable[native.GraphOp]`. The plugin yields `AddNode` / `AddEdge` / `AddEntrypoint` ops the rust backend applies.
2. For common shapes, subclass `DecoratedDeclPlugin`, `DispatchAppPlugin`, or `LiteralListPlugin` from `plugins/decl_shapes.py` and configure with class-level attributes (`app_module`, `decorator_names`, etc.).
3. Drop generic-Python plugins in `dead_cst/plugins/<name>.py` (re-exported from `dead_cst/plugins/__init__.py`). Drop framework / third-party-aware plugins in `dead_cst/contrib/<name>.py` (re-exported from `dead_cst/contrib/__init__.py`). Register the CLI key in `_BUILTIN_PLUGINS` (`cli.py`), importing the plugin directly.
4. Out-of-tree plugins register under the `dead_cst.plugins` entry-point group; the CLI's `_load_plugin` checks `_BUILTIN_PLUGINS` first.

The plugin queries live on `native.ProjectContext` — `find_subclasses`, `find_module`, `find_declarations`, `module_for`, `find_main_blocks`, etc. — plus the chainable `query(ctx).decorators()...` / `.constructions()...` / `.calls()...` builder. See `python/dead_cst/_native.pyi` for the full surface.

For **native (Rust) plugins** — in-tree ones (e.g. `NativePlugin.main_block()`) and external ones compiled against the runtime `dylib` and loaded via `native.load_native_plugins(...)` — see [`NATIVE_PLUGINS.md`](NATIVE_PLUGINS.md). The author-facing flow is `pip install dead-cst[build-plugin]` (the shipped macOS/Linux wheel + the `dead-cst-plugin-host` rlib compile closure) then `dead-cst build-plugin <PLUGIN.rs>`, which compiles against the in-package runtime dylib + that closure — no swap. Maintainers populate the closure with `dead-cst bundle-plugin-host` (a prefer-dynamic build; the same build's runtime dylib is repacked into the base wheel). macOS + Linux only; needs the pinned toolchain. Prefer Python plugins unless the work needs native speed or plugin-defined salsa-cached queries. The `build-plugin`/`bundle-plugin-host` flow can't run from a dev/static checkout — it needs the dynamic wheel installed.

## Adding a resolver

Implement `PathResolver`: `resolve(project_root) -> tuple[Package, ...]`. Drop generic resolvers in `dead_cst/resolvers/<name>.py` (re-exported from `dead_cst/resolvers/__init__.py`); resolvers that target a specific external tool (like `UvResolver` for `uv.lock`) belong in `dead_cst/contrib/<name>.py` (re-exported from `dead_cst/contrib/__init__.py`). Register the CLI key in `_BUILTIN_RESOLVERS` (`cli.py`), importing the resolver directly.

## Conventions

- **Python 3.11+**. Ruff line length 100. `known-first-party = ["dead_cst"]`. `from __future__ import annotations` is used throughout the package.
- **`ty` type-checks `dead_cst/` only** — tests, examples, and workspace fixtures intentionally exercise untyped third-party internals.
- **Versioning is mostly manual**. `pyproject.toml` is `dynamic = ["version"]`; maturin reads the version from `Cargo.toml`'s `[package].version`. Bump that field by hand before tagging a release. On every push to `main`, the publish workflow rewrites it to `<base>-dev${{ github.run_number }}` (in-CI only, never committed) so TestPyPI gets a unique version per commit. The SemVer pre-release suffix (`-dev`, not `.dev`) is required so Cargo accepts the version; maturin normalizes it to PEP 440 `<base>.dev<N>` for the wheel. Never hand-edit a version on a non-release commit.
- **No optional arguments without a real reason.** When a parameter would be supplied by every caller anyway, make it required. The project is pre-1.0 and breaking the public API is fine.
- **PRs**: one logical change per PR, add/update tests, add a `[Unreleased]` `CHANGELOG.md` entry for user-visible changes.

## Known limitations to keep in mind

- `import *` is treated pessimistically (every top-level decl in the target is considered used) by default. `__import__('m')` and `importlib.import_module('m')` calls with a string-literal name emit `EdgeFlags.DYNAMIC_IMPORT` edges; opt into per-name fan-out with `DynamicImportFallbackPlugin`.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- `__all__` is followed only when assigned a list/tuple of string literals.
- ty has a `TODO walrus in comprehensions is implicitly nonlocal` so walrus-in-comprehension target bindings don't leak to the enclosing module scope yet (see `tests/test_limitations.py`).

## Pre-release status

`dead-cst` is alpha. APIs, CLI flags, and output formats may change without notice. Never run `dead-cst remove` against uncommitted code.
