# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

This project uses [uv](https://github.com/astral-sh/uv); always run tools through `uv run` so the locked dev environment is used.

```bash
uv sync                                # install package + dev group in editable mode
uv run pytest                          # full suite (e2e is deselected by default)
uv run pytest tests/test_branches.py   # one file
uv run pytest -k name_substring        # one test by name
uv run pytest -m e2e                   # opt-in e2e suite (clones real GitHub repos)
uv run pytest --cov=dead_cst --cov-branch --cov-report=term-missing
uv run ptw                             # pytest-watcher: tight inner loop
uv run prek run --all-files            # ruff (lint+format), ty type-check, pre-commit hooks
```

`pyproject.toml` pins `addopts = "-m 'not e2e'"`, so the standard `uv run pytest` is hermetic. The `tests/e2e/` suite shallow-clones third-party repos at pinned SHAs into pytest's cache (override with `DEAD_CST_E2E_CACHE`); skips on missing `git` or network errors.

CI runs `prek run --all-files` and the matrix `pytest` (Python 3.11–3.14) on every push/PR. Coverage is uploaded only from the 3.13 matrix entry.

A `PostToolUse` hook in `.claude/settings.json` automatically runs `ruff format`, `ruff check --fix` (preserving unused imports — F401 is unfixable), and `ty check` after every Write/Edit on a `.py` file. There is no need to invoke these manually after editing.

## Architecture

`dead-cst` builds a symbol-level reachability graph of a Python codebase using [libcst](https://github.com/Instagram/LibCST) and [networkx](https://networkx.org/), walks from configured entrypoints, and reports (or removes) anything unreachable. The flow is staged so each stage has one job; understanding the staging is the key to navigating the code.

### The unified `Cacheable` contract

Four moving pieces — `SymbolVisitor` itself plus the three extension points (`EdgePlugin`, `PathResolver`, `UnreachableRegionDetector`) — all satisfy `Cacheable` (`dead_cst/_cacheable.py`), which is just `name: str` + `version: int`. The `(name, version)` pair feeds `compute_fingerprint` in `_cache.py`, so swapping or reconfiguring any of them invalidates the per-file SQLite cache. `version` is a Unix epoch int **by convention** — concurrent bumps on different branches merge with `max()`-wins semantics rather than colliding on a re-used label. Bump it any time a component's *output* would change for the same input. The package `__version__` is **not** in the fingerprint: each component carries its own knob, and folding `__version__` in on top would let lazily-unbumped components ride for free on a release bump. Schema version and Python version still participate.

### Pipeline (top-down)

1. **Path resolution** (`dead_cst/resolvers/`). A `PathResolver` returns `list[Package]` plus an import resolver. Each `Package` is a directory `path` + unique `name` + `exported` subdirs (whose files ship in the wheel) + `deps` (other package names — the production-only DAG). Files under `path` but NOT under any `exported` subdir are *internal* (tests, scripts, app entrypoints); they participate in the analysis but don't appear in the package's consumer-visible export trie. `validate_packages` enforces non-empty unique names, unique paths, exported subdirs under `path`, no self-deps, and acyclic `deps`. The analyzer parses each package in two phases (see Pipeline step 5); only `deps` participates in the topological order, which is what allows internal files to have apparent cross-package cycles (`A.tests` ↔ `B.lib`, `B.tests` ↔ `A.lib`) without violating the production DAG. Builtins: `UvResolver` (parses `uv.lock` to discover workspace members + inter-member dep edges) and `ManualResolver` (`-p` CLI specs; deps in spec are package names, not paths). Multiple resolvers chain via `_chain_resolvers` — first non-`None` wins. Run `dead-cst` with the project's venv active so `default_resolve_import` finds third-party dists via the running Python's `sys.path`. `resolvers/_imports.py` does name→path lookup; `resolvers/_core.py` ships the `is_exported_file` and `export_search_root` helpers custom resolvers can reuse.
2. **Per-file visitor** (`dead_cst/_visitor.py`). `SymbolVisitor` walks each `.py` file, returning a `VisitorPayload` (defined in `dead_cst/graph.py`) with four fields: `nodes` (every real `SymbolNode`, including `SHADOWED`-flagged ones), `edges` (`(src, dst, access_pos)` triples for resolved decl-to-decl refs), `imports` (`(src, Import, access_pos)` triples for cross-file refs), and `dead_suites` (`CodeRange`s of every statically-dead suite). The apply step compares each `access_pos` against `dead_suites` to flag `EdgeFlags.DEAD_BRANCH`. Heavy lifting: LibCST `ScopeProvider`, `FullyQualifiedNameProvider` (wrapped via `_fqn.FixedFullyQualifiedNameProvider`), `_flow.py` for flow-sensitive shadowing.
3. **Unreachable-region detection** (`dead_cst/branches.py`, `dead_cst/_const_fold.py`). `UnreachableRegionDetector` is a Protocol with `find_regions(wrapper) -> list[CodeRange]`; `SymbolVisitor.visit_Module` invokes it once per file using the analyzer's already-resolved `PositionProvider`. The shipped `DefaultUnreachableRegionDetector` runs three passes: the literal-truthiness `unreachable_suites` walk over `if`/`while` tests, a fixpoint `fold_constants` pre-pass (`_const_fold.fold_constants` propagates simple `Name = literal` bindings through chained `or`/`and`/`not` until quiescent), and a post-terminator scan that marks statements after an unconditional `return`/`raise`/`break`/`continue`/`assert <falsy>` as dead within the same suite (suite-relative — a `raise` in a `try` body doesn't kill the matching `except`). Subclass and override `resolve(self, expr) -> bool | None` to fold non-literal expressions (e.g. `check_flag("migration-abc")` is always `True` in production); the override is consulted recursively for every non-keyword expression in `if`/`while`/`assert` tests and every foldable RHS, and resolved values flow through the same fixpoint loop. Pass an instance via `Analysis(unreachable_detector=...)`. `branches.py` re-exports `fold_constants` so detector authors writing `find_regions` from scratch don't have to reach into `_const_fold`.
4. **Visitor + observe cache** (`dead_cst/cache.py`). SQLite-backed `GraphCache` at `<root>/.dead-cst-cache/cache.db` stores pickled per-file payloads keyed by file SHA-256. Each row carries a per-(package, phase) fingerprint covering the package path, the phase's `sys.path` entries, and every visitor / resolver / plugin / detector `(name, version)` pair via `compute_fingerprint`. The cache's schema version lives in `meta`; a mismatch wipes `file_cache`. Each cache entry covers both the visitor's payload **and** every plugin's `observe()` output for that file — warm runs skip both. Bypass with `--no-cache`. Force-clear with `dead-cst cache clear`.
5. **Two-phase composition + edge stitching** (`Analysis._materialize_full` + `dead_cst/_edges.py`). The analyzer composes the graph in two passes. **Phase 1 (exports)**: walks every package's exported files in topological order over `deps`. For each package's exports contribution, builds a per-package export trie and stitches cross-package imports against (own export trie under construction + each dep's already-built export trie). **Phase 2 (internals)**: walks every package's internal files in any order. Stitches against (own combined trie = exports + internals) + (every other package's export trie). Phase 2 reads tries that phase 1 already finalized, so apparent cross-package cycles between non-exported code (`A.tests` ↔ `B.lib`, `B.tests` ↔ `A.lib`) resolve cleanly. `resolve_edges` runs unconditionally per (package, phase) (not cached). It walks the per-phase `SymbolTrie` against the `(src, Import, flags)` triples from each payload and yields concrete `(src_symbol, dst_symbol, flags)` edges — following re-exports, fanning star imports out to every top-level decl, emitting synthetic nodes (`[external dist]`, `[external file]`, `[unresolved]`) for non-first-party targets. Re-running this every analysis is what makes single-file edits cheap (only the edited file's payload is recomputed; importers re-stitch for free).
6. **Plugins** (`dead_cst/plugins/`). Two phases:
   - **`observe(ctx: ObserveContext) -> VisitorPayload | None`** runs once per file inside the visitor loop, with the parsed `cst.Module` and the visitor's just-built payload. Returns a new payload (or `None`) whose `nodes`/`edges` extend the file's contribution; the result is cached alongside the visitor's output. Cross-file work does **not** belong here.
   - **`finalize(ctx: PluginContext) -> Iterable[GraphOp]`** runs once per base after `resolve_edges` has stitched cross-file imports. It operates purely on the assembled graph — no CST access — and emits `AddNode` / `AddEdge` / `RemoveEdge` ops. This is where transitive subclass closures, factory walks, and `[project.scripts]` lookups live.

   Plugins emit `GraphOp` values rather than mutating the graph directly. `PluginContext` provides helpers (`find_module`, `find_declarations`, `module_surface`, `base_modules`, `importers`, …). Builtins ship in `BUILTIN_PLUGINS`. Extensions targeting specific third-party tools live under `dead_cst/contrib/` — both framework plugins (`FastAPIPlugin`, `FlaskPlugin`, `ClickPlugin`, `TyperPlugin`, `PytestPlugin`, `UnittestPlugin`, `MockPatchPlugin`) and the `UvResolver` (which knows about `uv.lock`). Generic-Python plugins (`MainBlockPlugin`, `ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`, `ModuleDundersPlugin`, `InitSubclassPlugin`) live as siblings of `plugins/__init__.py`. Contrib classes are re-exported from `dead_cst.plugins` and `dead_cst.resolvers` for ergonomics. Scaffolds for the common dynamic-import idioms live in `plugins/decl_shapes.py`: `DecoratedDeclPlugin` ("decorated decls in files matching a prefix" — `ClickPlugin` is itself a subclass) and `LiteralListPlugin` ("read `<owner>.<var> = ['fqn', ...]` and treat each entry as alive"). Both are pure `observe`-time so their work rides the cache. The contrib/resolvers cycle (the `UvResolver` re-export) is broken with a lazy `__getattr__` in `dead_cst/contrib/__init__.py` — see the comment there.
7. **Reachability** (`Analysis.reachable` / `PackageView.reachable`). BFS successors from every node with `entrypoint=True`. Default traversal **does** follow `DEAD_BRANCH` edges (preserving today's behavior). `Analysis.kept_alive_by_dead_branches` (and the per-base `PackageView` version) is the opt-in inverse: returns the blast radius of removing every dead suite by skipping those edges.
8. **Codemod** (`dead_cst/codemod.py`). `remove_code` runs a LibCST `RemoveDeadSymbols` transformer keyed on `(fqname, CodeRange)` pairs (position disambiguates shadowed decls), then prunes now-unused imports via `RemoveImportsVisitor`. Position keying is critical — losing it conflates a dead decl with a live shadow. The high-level entry point is `PackageView.remove_dead_code()`, which materializes the full graph, computes reachability, and feeds the unreachable subgraph for that package into `remove_code`.

### Graph model invariants

- One node per top-level declaration plus one synthetic module node per file. **Nested defs (inner functions, methods, nested classes) are deliberately not given their own nodes** — refs from inside them are attributed to the enclosing top-level decl.
- A module-level `import` / `from ... import ...` is itself a node of type `"import"`. Local uses of an imported name go through the import node, which points at the upstream module/symbol. This is how `dead-cst remove` knows to drop now-unused imports.
- Submodules edge to their parent package, so `__init__.py` stays alive as long as anything in the package does.
- `EdgeFlags.DEAD_BRANCH` is metadata only; tests using the `assert_edges` fixture see those edges, while `assert_dead_branch_edges` filters to just them.
- `NodeFlags.SHADOWED` decls are emitted into the graph (with their parent module edge) but excluded from the trie, so cross-module imports never resolve to them.

### Public API surface

The supported surface is whatever is re-exported from a module named without a leading `_`. The top-level `dead_cst/__init__.py` re-exports the highlights (`Analysis`, `PackageView`, `Package`, `Cacheable`, the graph data types). Deeper symbols live in focused public submodules:

- `dead_cst.graph` — `SymbolNode`, `Import`, `NodeFlags`, `EdgeFlags`, `VisitorPayload` (the data classes the analyzer emits and consumes).
- `dead_cst.analyze` — `Analysis` and `PackageView`, the lazy entry-point classes.
- `dead_cst.codemod` — the LibCST source rewriter.
- `dead_cst.cache` — `GraphCache`, `compute_fingerprint`, schema/dirname constants.
- `dead_cst.branches` — `UnreachableRegionDetector`, `DefaultUnreachableRegionDetector`, plus the truthiness/fold helpers a from-scratch detector needs.
- `dead_cst.plugins` — the `EdgePlugin` protocol, `PluginContext` / `ObserveContext`, `GraphOp` value objects, the synthetic-node prefix constants, the generic-Python builtin plugins, and (re-exported) every contrib plugin.
- `dead_cst.resolvers` — the `PathResolver` protocol, the `Package` data type and `validate_packages` / `assign_file_to_package` / `is_exported_file` / `export_search_root` helpers, generic builtin resolvers, and (re-exported) `UvResolver` from contrib.
- `dead_cst.contrib` — the canonical home for third-party-aware extensions: framework plugins (FastAPI, Flask, Click, Typer, pytest, unittest, mock-patch) and the `UvResolver`.

Modules prefixed with `_` are internal — the visitor (`_visitor.py`), edge stitcher (`_edges.py`), flow analysis (`_flow.py`), libcst FQN patch (`_fqn.py`), const-fold helper (`_const_fold.py`), and the `_cacheable.py` protocol module. `tests/test_public_api.py` pins each public module's `__all__` against a snapshot, so accidentally dropping a public name is caught in CI; intentional changes require updating the snapshot.

The stable extension points are the `EdgePlugin`, `PathResolver`, and `UnreachableRegionDetector` protocols (and the `dead_cst.plugins` / `dead_cst.resolvers` entry-point groups).

## Tests

Tests are fixture-driven from inline source snippets — no checked-in `.py` fixture files under `tests/`. Two patterns dominate (defined in `tests/conftest.py`):

```python
def test_something(build_decl_graph, assert_edges):
    graph = build_decl_graph({"mod.py": "def a(): pass\na()"})
    assert_edges(graph, {"mod.a -> mod", "mod -> mod.a"})
```

- `build_decl_graph({filename: source})` — writes dedented sources to `tmp_path` and runs `Analysis(...).materialize_all()`.
- `assert_edges` / `assert_positional_edges` / `assert_dead_branch_edges` — compare edges as `"src.fqname -> dst.fqname"` strings (positional variant uses `fqname@line:col` to disambiguate redeclarations).
- Plugin tests live in `tests/test_plugins/` with their own `conftest.py`; resolver tests in `tests/test_resolvers/`.
- Codemod tests (`tests/test_codemod.py`) write a snippet, call `remove_code`, and assert on the rewritten text.
- E2E tests (`tests/e2e/`) shallow-clone real repos (e.g. `flux0`) at pinned SHAs, run `dead-cst` against them, and assert on real-world breakage. They're behind the `e2e` marker and excluded by default; opt in with `-m e2e`. They skip (rather than fail) on missing `git` or network errors.
- `[tool.pytest.ini_options] testpaths = ["tests"]` is required — without it pytest also collects `examples/uv-workspace/packages/*/tests/`, where two `tests/` packages share `tests.conftest` and fail with `ImportPathMismatchError` (which is the very layout that example exists to demonstrate).

## Coverage policy (informational, except patch gate)

Per-component thresholds in `codecov.yml` reflect blast radius, not LOC:

| Component | Target |
|---|---|
| `codemod.py` | 95% (rewrites user files) |
| `cli.py` | 80% |
| `plugins/**`, `resolvers/**`, `contrib/**` | 85% |
| `analyze.py`, `_visitor.py`, `_flow.py`, `branches.py`, `graph.py`, `_fqn.py` | 80% |

Per-component statuses are informational until Tier 1 of `ROADMAP.md` lands. The **patch-coverage gate on new code is enforcing at 90%**.

## Adding a plugin

1. Implement `EdgePlugin`: `name: str`, `version: int` (epoch), `observe(ctx) -> VisitorPayload | None`, `finalize(ctx) -> Iterable[GraphOp]`. Either method may be a no-op (`return None` / `return ()`).
2. Keep `observe` file-local (it rides the per-file cache) and put cross-file work in `finalize` (which only sees the assembled graph).
3. For the common shapes, subclass `DecoratedDeclPlugin` or `LiteralListPlugin` from `plugins/decl_shapes.py` instead of starting from scratch — set `package_prefix`, `decorator_module`, etc., plus a unique `name` and current-epoch `version`.
4. Drop generic-Python builtins in `dead_cst/plugins/<name>.py`. Drop framework / third-party-aware builtins in `dead_cst/contrib/<name>.py` and re-export from `dead_cst/plugins/__init__.py` for ergonomics. Either way, register in `BUILTIN_PLUGINS` (`plugins/__init__.py`).
5. `FastAPIPlugin` is the full-featured reference for two-phase plugins; `ClickPlugin` for the `DecoratedDeclPlugin` subclass shape.
6. Out-of-tree plugins register under the `dead_cst.plugins` entry-point group; `load_plugin` checks builtins first, then entry points.

## Adding a resolver

Implement `PathResolver`: `name`, `version`, `resolve(project_root) -> list[Package]`, and `resolve_import(name, search_paths) -> str | Path | None`. Drop generic resolvers in `dead_cst/resolvers/<name>.py`; resolvers that target a specific external tool (like `UvResolver` for `uv.lock`) belong in `dead_cst/contrib/<name>.py`, with a re-export in `dead_cst/resolvers/__init__.py`. Third-party resolvers register under the `dead_cst.resolvers` entry-point group.

## Adding an unreachable-region detector

Implement `UnreachableRegionDetector`: `name`, `version`, `find_regions(wrapper) -> list[CodeRange]`. To layer domain knowledge over the default behavior, subclass `DefaultUnreachableRegionDetector` and override `resolve(self, expr) -> bool | None` — the override is consulted recursively for every non-keyword expression in `if`/`while`/`assert` tests and every foldable RHS, so guard with an early `isinstance` check to keep it cheap. Returning `None` defers to the literal handling. Pass the instance via `Analysis(unreachable_detector=...)`. Bump `version` (epoch-int) when the override's logic changes so cached `VisitorPayload` blobs rebuild automatically.

## Conventions

- **Python 3.11+**. Ruff line length 100. `known-first-party = ["dead_cst"]`. `from __future__ import annotations` is used throughout the package.
- **`ty` type-checks `dead_cst/` only** — tests, examples, and workspace fixtures intentionally exercise untyped LibCST internals and the unresolved-import case the analyzer detects, so they aren't meaningful targets.
- **Versioning is tag-driven via `hatch-vcs`** — never hand-edit a version. `local_scheme = "no-local-version"` is required because PyPI rejects PEP 440 local-version identifiers; `SETUPTOOLS_SCM_OVERRIDES_FOR_*` does not work here (only `Configuration.from_file()` reads it).
- **Cache invalidation discipline**: any change to the visitor / a plugin / resolver / detector's *output* for the same input requires bumping its `version` to a fresh epoch, or stale cache entries will be served. `__version__` deliberately is not a backstop. The schema-level `SCHEMA_VERSION` in `_cache.py` is for `VisitorPayload` shape breaks the unpickler can't handle.
- **PRs**: one logical change per PR, add/update tests, add a `[Unreleased]` `CHANGELOG.md` entry for user-visible changes.
- **Releases**: pushing a `vX.Y.Z` tag triggers `.github/workflows/publish.yml` (uv build + PyPI OIDC publish).

## Known limitations to keep in mind

- `import *` is treated pessimistically (every top-level decl in the target is considered used). `__import__('m')` and `importlib.import_module('m')` calls with a string-literal name are folded into the same star-import path; relative names (`'.sub'`, or `__import__('sub', ..., level=1)`) resolve against the file's enclosing package the same way `from .sub import *` does. Non-literal names / levels / packages warn. `__import__(name, fromlist=[...])` literal entries that resolve as submodules are fanned out as well.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- `__all__` is followed only when assigned a list/tuple of string literals.
- PEP 750 template strings (`t"..."`, 3.14+) cannot be parsed by the pinned `libcst` and abort analysis with `ParserSyntaxError`.

## Pre-release status

`dead-cst` is alpha. APIs, CLI flags, and output formats may change without notice. Never run `dead-cst remove` against uncommitted code. See `ROADMAP.md` for the trajectory toward 1.0.
