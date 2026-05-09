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

Three moving pieces — `SymbolVisitor` itself plus two of the three extension points (`EdgePlugin`, `UnreachableRegionDetector`) — satisfy `Cacheable` (`dead_cst/_cacheable.py`), which is just `name: str` + `version: int`. Their `(name, version)` pairs feed `compute_fingerprint` in `_cache.py`. `PathResolver` deliberately does **not** carry a `Cacheable` knob: import resolution moved out of the visitor (stage 5 below) and runs unconditionally on every analysis, so swapping a resolver re-stitches edges without invalidating any per-file SQLite cache entry — there is nothing to bump. `version` is a Unix epoch int **by convention** — concurrent bumps on different branches merge with `max()`-wins semantics rather than colliding on a re-used label. Bump it any time a `Cacheable` component's *output* would change for the same input. The package `__version__` is **not** in the fingerprint: each component carries its own knob, and folding `__version__` in on top would let lazily-unbumped components ride for free on a release bump. Schema version and Python version still participate.

### Pipeline (top-down)

1. **Path resolution** (`dead_cst/resolvers/`). A `PathResolver` returns a `tuple[Package, ...]` (`Package(path, name, exported, deps)`; deps reference other packages by name) plus an import resolver. Builtins: `ManualResolver` (explicit `package:dep1,dep2` specs from the CLI's `-p`; auto-promotes inline dep paths to their own packages) and `UvResolver` (parses `uv.lock` to discover workspace members + inter-member dep edges; lives in `dead_cst/contrib/uv.py` and is re-exported from `dead_cst.resolvers`). `Analysis(project_root, resolver=...)` takes exactly one resolver — there is no chaining today, and the CLI's `-p` and `--resolver` flags are mutually exclusive. Construction calls `resolver.resolve(project_root)` and validates the output via the private `_validate_packages` helper (name uniqueness, dep references, `exported` entries under their package's `path`). Non-package search paths (workspace `.venv/site-packages`, vendored bundles) are not represented in `Package.deps` — `UvResolver` splices its workspace venv onto `sys.path` lazily inside its own `resolve_import`. `resolvers/_imports.py` does name→path lookup (and exposes `clear_path_caches()` for resolvers that mutate `sys.path`); `resolvers/_exports.py` computes which subdirs each package exposes to consumers.
2. **Per-file visitor** (`dead_cst/_visitor.py`). `SymbolVisitor` walks each `.py` file, returning a `VisitorPayload` (defined in `dead_cst/graph.py`) with four fields: `nodes` (every real `SymbolNode`, including `SHADOWED`-flagged ones), `edges` (`(src, dst, access_pos)` triples for resolved decl-to-decl refs), `imports` (`(src, Import, access_pos)` triples for cross-file refs — **raw**: `Import` carries only the dotted names written in source, no path classification), and `dead_suites` (`CodeRange`s of every statically-dead suite). The apply step compares each `access_pos` against `dead_suites` to flag `EdgeFlags.DEAD_BRANCH`. The visitor never calls a resolver and never reads `sys.path`, so its output is purely a function of the file's source — that's what lets the per-file cache survive `search_paths` / resolver changes. Heavy lifting: LibCST `ScopeProvider`, `FullyQualifiedNameProvider` (wrapped via `_fqn.FixedFullyQualifiedNameProvider`), `_flow.py` for flow-sensitive shadowing.
3. **Unreachable-region detection** (`dead_cst/branches.py`). `UnreachableRegionDetector` is a Protocol with `find_regions(wrapper) -> list[CodeRange]`; `SymbolVisitor.visit_Module` invokes it once per file using the analyzer's already-resolved `PositionProvider`. The shipped `DefaultUnreachableRegionDetector` is a two-pass design: a single `cst.CSTVisitor` walk collects every `If`/`While` plus every statement-bearing suite, and a `TruthinessResolver` (also in `branches.py`) answers truthiness queries on demand — `unreachable_suites` for `if`/`while` tests, plus a per-suite scan that marks statements after an unconditional `return`/`raise`/`break`/`continue`/`assert <falsy>` as dead (suite-relative — a `raise` in a `try` body doesn't kill the matching `except`). The resolver is goal-directed: it lazily resolves `ScopeProvider`/`ParentNodeProvider`, walks `live_referents` only for names that actually feed a query, and memoizes by access node id with a cycle sentinel for `a = b; b = a` patterns. Subclass and override `resolve(self, expr) -> bool | None` to fold non-literal expressions (e.g. `check_flag("migration-abc")` is always `True` in production); the override is consulted recursively for every non-keyword expression encountered. Pass an instance via `Analysis(unreachable_detector=...)`. From-scratch detector authors instantiate `TruthinessResolver(wrapper, resolve_expr=...)` and pass `resolver.evaluate` as the `resolve_expr` callback to `unreachable_suites` / `evaluate_truthiness`.
4. **Visitor + observe cache** (`dead_cst/cache.py`). SQLite-backed `GraphCache` at `<root>/.dead-cst-cache/cache.db` stores pickled per-file payloads keyed by file SHA-256. Each row carries a single analysis-wide fingerprint covering Python version, schema version, and every visitor / plugin / detector `(name, version)` pair via `compute_fingerprint`. **The package layout, `search_paths`, and the resolver are deliberately not in the fingerprint** — the visitor's output is purely a function of the file's source plus the plugin / detector chain, and import resolution moved to stage 5 (which runs unconditionally), so resolver / search-path / package-layout swaps re-stitch edges without invalidating cached payloads. Mismatch wipes `file_cache`. Each cache entry covers both the visitor's payload **and** every plugin's `observe()` output for that file — warm runs skip both. Bypass with `--no-cache`. Force-clear with `dead-cst cache clear`.
5. **Edge stitching** (`dead_cst/_edges.py`). `resolve_edges` runs unconditionally (not cached). Walks the raw `(src, Import, flags)` triples against the per-package `SymbolTrie`, **canonicalizing** each `Import` first by pushing decl parts into `module` while they resolve as submodules in the trie (`from p import functions` -> `module="p.functions", decl=None` once `p.functions` is found). Trie misses fall back to the `PathResolver` chain for stdlib / external dist / external file / unresolved classification, which is the single place that reads `sys.path` and the resolver LRU caches — `Analysis._materialize` rebinds `sys.path` to each package's `(path, *deps)` view before composing it and clears the resolver caches at every transition, restoring the original `sys.path` on the way out. Star imports follow the same path; `Import.speculative` (set on `__import__` fromlist synthesis) silences the warning when the trie + resolver both miss. Re-running this every analysis is what makes single-file edits cheap (only the edited file's payload is recomputed; importers re-stitch for free). The per-file work that feeds this stage (file enumeration, stale detection, parallel visitor pool, payload application into the per-package contribution) lives in `dead_cst/_refresh.py` so `analyze.py` can stay focused on cross-package composition.
6. **Plugins** (`dead_cst/plugins/`). Two phases:
   - **`observe(ctx: ObserveContext) -> VisitorPayload | None`** runs once per file inside the visitor loop, with the parsed `cst.Module` and the visitor's just-built payload. Returns a new payload (or `None`) whose `nodes`/`edges` extend the file's contribution; the result is cached alongside the visitor's output. Cross-file work does **not** belong here.
   - **`finalize(ctx: PluginContext) -> Iterable[GraphOp]`** runs once per package after `resolve_edges` has stitched cross-file imports. It operates purely on the assembled graph — no CST access — and emits `AddNode` / `AddEdge` / `RemoveEdge` ops. This is where transitive subclass closures, factory walks, and `[project.scripts]` lookups live.

   Plugins emit `GraphOp` values rather than mutating the graph directly. `PluginContext` provides helpers (`find_module`, `find_declarations`, `module_surface`, `package_modules`, `package_nodes`, `importers`, …) and exposes the current `Package` via `ctx.package`. Builtins ship in `BUILTIN_PLUGINS`. Extensions targeting specific third-party tools live under `dead_cst/contrib/` — both framework plugins (`FastAPIPlugin`, `FlaskPlugin`, `ClickPlugin`, `TyperPlugin`, `PytestPlugin`, `UnittestPlugin`, `MockPatchPlugin`) and the `UvResolver` (which knows about `uv.lock`). Generic-Python plugins (`MainBlockPlugin`, `ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`, `ModuleDundersPlugin`, `InitSubclassPlugin`) live as siblings of `plugins/__init__.py`. Contrib classes are re-exported from `dead_cst.plugins` and `dead_cst.resolvers` for ergonomics. Scaffolds for the common dynamic-import idioms live in `plugins/decl_shapes.py`: `DecoratedDeclPlugin` ("decorated decls in files matching a prefix" — `ClickPlugin` is itself a subclass) and `LiteralListPlugin` ("read `<owner>.<var> = ['fqn', ...]` and treat each entry as alive"). Both are pure `observe`-time so their work rides the cache. The contrib/resolvers cycle (the `UvResolver` re-export) is broken with a lazy `__getattr__` in `dead_cst/contrib/__init__.py` — see the comment there.
7. **Reachability** (`Analysis.reachable` / `PackageView.reachable`). BFS successors from every node with `entrypoint=True`. Default traversal **does** follow `DEAD_BRANCH` edges (preserving today's behavior). `Analysis.kept_alive_by_dead_branches` (and the per-package `PackageView` version) is the opt-in inverse: returns the blast radius of removing every dead suite by skipping those edges.
8. **Codemod** (`dead_cst/codemod.py`). `remove_code` runs a LibCST `RemoveDeadSymbols` transformer keyed on `(fqname, CodeRange)` pairs (position disambiguates shadowed decls), then prunes now-unused imports via `RemoveImportsVisitor`. Position keying is critical — losing it conflates a dead decl with a live shadow. The high-level entry point is `PackageView.remove_dead_code()`, which materializes the closure, computes reachability, and feeds the unreachable subgraph for that package into `remove_code`. `generate_patch(G, root)` is the non-destructive twin: same selection logic, but returns a `git apply`-compatible unified diff (`diff --git` headers; `deleted file mode 100644` for module-node deletions) instead of writing back. Selection is driven entirely by `G.nodes`, so callers can slice the unreachable graph (e.g. `G.subgraph(scc)` for one SCC at a time) to review a big codebase as a series of focused patches. The `dead-cst remove` CLI uses `generate_patch` exclusively — it emits a patch to stdout (or `--output PATH`) for the user to pipe into `git apply`, and never mutates source.

### Graph model invariants

- One node per top-level declaration plus one synthetic module node per file. **Nested defs (inner functions, methods, nested classes) are deliberately not given their own nodes** — refs from inside them are attributed to the enclosing top-level decl.
- A module-level `import` / `from ... import ...` is itself a node of type `"import"`. Local uses of an imported name go through the import node, which points at the upstream module/symbol. This is how `dead-cst remove` knows to drop now-unused imports.
- Imports whose source line carries a ruff/pyflakes `# noqa` directive that silences F401 (bare `# noqa`, `# noqa: F401`, multi-rule `# noqa: E501, F401`, case-variant `# NOQA`) are flagged `NodeFlags.ENTRYPOINT | NodeFlags.NOQA` so reachability keeps them alive — matching ruff's own semantics for explicitly-preserved imports. File-level `# ruff: noqa` and `# flake8: noqa` directives (`ruff:` / `flake8:` is matched case-sensitively per ruff; `noqa` is not) pin every import in the file. The `NOQA` bit powers the opt-in `Analysis.kept_alive_by_noqa_only` / `PackageView.kept_alive_by_noqa_only` queries (parallel to `TESTCASE` / `kept_alive_by_tests_only`), which return the blast radius of removing every F401 pin. The shared `_find_reachable_excluding(graph, flags)` and `_find_kept_alive_by_flags_only(graph, flags)` helpers take any `NodeFlags` combination.
- Submodules edge to their parent package, so `__init__.py` stays alive as long as anything in the package does.
- `EdgeFlags.DEAD_BRANCH` is metadata only; tests using the `assert_edges` fixture see those edges, while `assert_dead_branch_edges` filters to just them.
- `NodeFlags.SHADOWED` decls are emitted into the graph (with their parent module edge) but excluded from the trie, so cross-module imports never resolve to them. `NodeFlags.OVERLOAD` follows the same trie-exclusion rule but its lifetime is anchored to the matching same-file impl via explicit `impl -> overload` edges emitted by the visitor.
- `.pyi` stub files are ingested only for the compiled-extension layout (`_native.so` + `_native.pyi`, no `.py` twin). The stub is parsed under its natural module FQN (`mypkg._native`), so `from mypkg._native import X` resolves through the normal trie path. Peer-mode `.pyi` (alongside a real `.py`) is dropped at `enumerate_files` time — the runtime always wins, and ingesting both would collide on the same FQN in the trie.

### Public API surface

The supported surface is whatever is re-exported from a module named without a leading `_`. The top-level `dead_cst/__init__.py` re-exports the highlights (`Analysis`, `PackageView`, `Cacheable`, the graph data types). Deeper symbols live in focused public submodules:

- `dead_cst.graph` — `SymbolNode`, `Import`, `NodeFlags`, `EdgeFlags`, `VisitorPayload` (the data classes the analyzer emits and consumes).
- `dead_cst.analyze` — `Analysis` and `PackageView`, the lazy entry-point classes.
- `dead_cst.codemod` — the LibCST source rewriter.
- `dead_cst.cache` — `GraphCache`, `compute_fingerprint`, schema/dirname constants.
- `dead_cst.branches` — `UnreachableRegionDetector`, `DefaultUnreachableRegionDetector`, `TruthinessResolver`, plus the truthiness helpers (`evaluate_truthiness`, `unreachable_suites`, `unreachable_bodies`) a from-scratch detector needs.
- `dead_cst.plugins` — the `EdgePlugin` protocol, `PluginContext` / `ObserveContext`, `GraphOp` value objects, the synthetic-node prefix constants, the generic-Python builtin plugins, and (re-exported) every contrib plugin.
- `dead_cst.resolvers` — the `PathResolver` protocol, the `ManualResolver` builtin, and (re-exported) `UvResolver` from contrib.
- `dead_cst.contrib` — the canonical home for third-party-aware extensions: framework plugins (FastAPI, Flask, Click, Typer, pytest, unittest, mock-patch) and the `UvResolver`.

Modules prefixed with `_` are internal — the visitor (`_visitor.py`), edge stitcher (`_edges.py`), flow analysis (`_flow.py`), libcst FQN patch (`_fqn.py`), and the `_cacheable.py` protocol module. `tests/test_public_api.py` pins each public module's `__all__` against a snapshot, so accidentally dropping a public name is caught in CI; intentional changes require updating the snapshot.

The stable extension points are the `EdgePlugin`, `PathResolver`, and `UnreachableRegionDetector` protocols (and the `dead_cst.plugins` / `dead_cst.resolvers` entry-point groups).

## Tests

Tests are fixture-driven from inline source snippets — no checked-in `.py` fixture files under `tests/`. Two patterns dominate (defined in `tests/conftest.py`):

```python
def test_something(build_decl_graph, assert_edges):
    graph = build_decl_graph({"mod.py": "def a(): pass\na()"})
    assert_edges(graph, {"mod.a -> mod", "mod -> mod.a"})
```

- `build_decl_graph({filename: source})` — writes dedented sources to `tmp_path` and runs `Analysis(...).materialize_all()` (under the hood: a `ManualResolver(specs=["."])` wired through the `make_analysis` fixture).
- `make_analysis(specs=["."], **kwargs)` — lower-level fixture for tests that need to pass extra `Analysis` kwargs (plugins, cache, detector, workers). Defaults the resolver to `ManualResolver(specs=["."])` so `make_analysis(plugins=[X()])` is the canonical single-package shape.
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

Implement `PathResolver`: `resolve(project_root) -> tuple[Package, ...]` and `resolve_import(name, search_paths) -> str | Path | None`. Resolvers do **not** carry a `Cacheable` `(name, version)` pair — their output flows through the (uncached) edge-stitching pass, so a swap re-stitches edges on the next run with nothing to invalidate. Drop generic resolvers in `dead_cst/resolvers/<name>.py`; resolvers that target a specific external tool (like `UvResolver` for `uv.lock`) belong in `dead_cst/contrib/<name>.py`, with a re-export in `dead_cst/resolvers/__init__.py`. Built-in resolvers register in `BUILTIN_RESOLVERS` (`dead_cst/resolvers/__init__.py`); third-party resolvers register under the `dead_cst.resolvers` entry-point group.

## Adding an unreachable-region detector

Implement `UnreachableRegionDetector`: `name`, `version`, `find_regions(wrapper) -> list[CodeRange]`. To layer domain knowledge over the default behavior, subclass `DefaultUnreachableRegionDetector` and override `resolve(self, expr) -> bool | None` — the override is consulted recursively for every non-keyword expression in `if`/`while`/`assert` tests and every foldable RHS, so guard with an early `isinstance` check to keep it cheap. Returning `None` defers to the literal handling. Pass the instance via `Analysis(unreachable_detector=...)`. Bump `version` (epoch-int) when the override's logic changes so cached `VisitorPayload` blobs rebuild automatically. From-scratch implementations that need name-aware truthiness should construct `TruthinessResolver(wrapper, resolve_expr=...)` once per file and call `resolver.evaluate(expr)` (or pass `resolver.evaluate` to `unreachable_suites`) — the resolver is goal-directed, so files that never query a Name pay only construction cost.

## Conventions

- **Python 3.11+**. Ruff line length 100. `known-first-party = ["dead_cst"]`. `from __future__ import annotations` is used throughout the package.
- **`ty` type-checks `dead_cst/` only** — tests, examples, and workspace fixtures intentionally exercise untyped LibCST internals and the unresolved-import case the analyzer detects, so they aren't meaningful targets.
- **Versioning is tag-driven via `hatch-vcs`** — never hand-edit a version. `local_scheme = "no-local-version"` is required because PyPI rejects PEP 440 local-version identifiers; `SETUPTOOLS_SCM_OVERRIDES_FOR_*` does not work here (only `Configuration.from_file()` reads it).
- **Cache invalidation discipline**: any change to the visitor / plugin / detector's *output* for the same input requires bumping its `version` to a fresh epoch, or stale cache entries will be served. Resolvers and the package layout are *not* in the fingerprint — their effect flows through the (uncached) edge-stitching pass — so a resolver swap or update doesn't invalidate cached payloads (and `PathResolver` carries no `Cacheable` knob to bump). `__version__` deliberately is not a backstop. The schema-level `SCHEMA_VERSION` in `_cache.py` is for `VisitorPayload` shape breaks the unpickler can't handle.
- **PRs**: one logical change per PR, add/update tests, add a `[Unreleased]` `CHANGELOG.md` entry for user-visible changes.
- **Releases**: pushing a `vX.Y.Z` tag triggers `.github/workflows/publish.yml` (uv build + PyPI OIDC publish).

## Known limitations to keep in mind

- `import *` is treated pessimistically (every top-level decl in the target is considered used). `__import__('m')` and `importlib.import_module('m')` calls with a string-literal name are folded into the same star-import path; relative names (`'.sub'`, or `__import__('sub', ..., level=1)`) resolve against the file's enclosing package the same way `from .sub import *` does. Non-literal names / levels / packages warn. `__import__(name, fromlist=[...])` literal entries that resolve as submodules are fanned out as well.
- Dynamic attribute access (`getattr`) and runtime-generated symbols are invisible to static analysis.
- `__all__` is followed only when assigned a list/tuple of string literals.
- When `libcst` rejects a file's syntax, the analyser logs a warning and substitutes a placeholder payload (the real module node plus a `[unparseable] <module>` synthetic flagged `ENTRYPOINT`), so the file stays alive in reachability but its decls are invisible until parsing succeeds. The placeholder rides the per-file cache; a fresh source SHA invalidates it automatically.

## Pre-release status

`dead-cst` is alpha. APIs, CLI flags, and output formats may change without notice. Never run `dead-cst remove` against uncommitted code. See `ROADMAP.md` for the trajectory toward 1.0.
