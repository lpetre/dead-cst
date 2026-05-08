# Changelog

All notable changes to `dead-cst` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first stable release the public API and CLI may change between any
two versions.

## [Unreleased]

### Changed
- **Breaking:** the analyzer's "base" terminology has been renamed to
  "package" everywhere it referred to a `Package` (the unit of
  workspace membership). On `Analysis`, `bases` is now `packages`
  (returning `tuple[Package, ...]` in BFS order; the previous
  `packages` resolver-order tuple is gone), `refresh(bases=)` is now
  `refresh(packages=)`, `reverse_closure(base)` /
  `materialize_closure(base)` rename their parameter to `package`,
  and `Analysis.package(base)` takes `path`. `PackageView.base` is
  now `PackageView.package` (a `Package`) with a `.path` convenience
  property. On `PluginContext` and `ObserveContext`, the `base: Path`
  field is now `package: Package` (use `ctx.package.path` for the
  directory); `PluginContext.base_modules()` /
  `PluginContext.base_nodes()` are now `package_modules()` /
  `package_nodes()`. `remove_code(G, base)` is now
  `remove_code(G, package_path)`, and
  `dead_cst.resolvers.exported_roots(base)` is now
  `exported_roots(package_path)`. Out-of-tree plugins and resolver
  consumers must update accordingly.
- The CLI's `-p` / `--path` spec is now described as
  `'package:dep1,dep2' or 'package'` (formerly
  `'base:dep1,dep2' or 'base'`). The parsing rules are unchanged --
  same syntax, clearer name -- so existing scripts keep working.
- Per-file refresh logic moved from `dead_cst/analyze.py` into a new
  `dead_cst/_refresh.py` (file enumeration, stale detection, the
  worker pool, payload application, and per-package contribution
  build). `analyze.py` keeps cross-package composition, reachability,
  and the public `Analysis` / `PackageView` classes. Tests that
  monkey-patched `analyze.SymbolVisitor` /
  `analyze.ProcessPoolExecutor` should now patch the same names on
  `dead_cst._refresh`.

### Added
- `.pyi` stub files are now ingested by the analyzer and rewritten by
  the codemod, targeting the **compiled-extension layout**: a binary
  module (e.g. `mypkg/_native.so`) shipping next to its hand-written
  type stub (`mypkg/_native.pyi`) with no `.py` twin. Stubs are
  discovered alongside `.py` siblings and parsed through the same
  visitor (so import edges, type-alias edges, etc. show up normally),
  given a synthetic `__pyi__` FQN segment (e.g. `mod.pyi` becomes
  `mod.__pyi__`) so they don't collide with a same-named `.py`, and --
  in the orphan case where no `.py` twin exists -- the per-package
  contribution rebinds the runtime FQN in the trie to point at the
  stub's module + decls. The result: `from mypkg._native import
  compute` resolves to the stub's `compute` decl through the normal
  cross-module import path, and reachability + the codemod work the
  same as for any first-party module. Peer stubs (a `.pyi` shipping
  alongside an actual `.py`) are intentionally left orphaned -- the
  runtime module wins, and the unused stub is treated as dead code
  unless something imports it.
- New `NodeFlags.OVERLOAD` flag plus visitor support for
  `typing.overload`. `@overload`-decorated functions (recognized
  syntactically -- bare `overload`, `typing.overload`, and
  `typing_extensions.overload`) are flagged and excluded from the
  cross-module lookup trie just like `SHADOWED` decls, so `from mod
  import f` continues to resolve to the impl rather than a typing
  stub. The visitor also wires `impl -> overload` edges so an
  overload's lifetime is anchored to its same-name impl: the codemod
  removes the overloads alongside the impl when the impl is dead, and
  preserves them as long as the impl is alive.
- Progress reporting around the per-file visitor pass ("Parsing
  files") and the cross-package composition pass ("Reconciling
  packages")
  in `Analysis.refresh` / `Analysis._materialize`. On a TTY the user
  sees a live `tqdm` bar; off a TTY (pytest capture, pipes, agent
  harnesses) the same wrapper emits one newline-terminated checkpoint
  at 0%, every ~10%, and at 100%, so CI logs and LLM tool consumers
  can track long runs without `tqdm`'s `\r`-overwriting frames going
  to mush. `tqdm>=4.66` is now a hard runtime dependency.
- New plugin helpers re-exported from `dead_cst.plugins`: `module_node`,
  `dotted_parts`, `dotted_name`, `string_value`,
  `payload_imports_module`. They consolidate boilerplate every contrib
  plugin used to inline (lookup the per-file module node, walk
  attribute chains, evaluate string literals, scan payload imports).
- New `DispatchAppPlugin` base in `dead_cst.plugins.decl_shapes` for
  CLI-style dispatch apps (`X = App(); @X.command(...)`). `TyperPlugin`
  and `CycloptsPlugin` are now thin subclasses configuring the module,
  constructor, and registration-decorator names.

### Fixed
- `dead-cst unused-exports` no longer matches variables whose names
  merely end with the literal `__all__` (e.g. `pkg.foo__all__`); only
  variables actually named `__all__` are considered.

### Changed
- **Breaking:** `compute_fingerprint` no longer takes a `base: Path`
  argument. The visitor's output is purely a function of the file's
  source plus the plugin / detector chain, so the analysis fingerprint
  is now a single value shared across every base. Callers should drop
  the `base=` keyword from any `compute_fingerprint(...)` invocation;
  per-file cache rows continue to gate on the analysis-wide fingerprint
  the same way they previously gated on the per-base one.
- File parsing is now flow-based rather than partitioned per base.
  `Analysis.refresh` walks each requested base's tree, collapses every
  base's cache misses into one global stale-file list, and runs the
  visitor + observe pass once across the whole batch. Multi-base
  refreshes that previously paid for one worker pool startup per base
  now pay for one total.
- The package dependency graph is no longer represented as a
  `networkx.DiGraph`, and `Package.deps` may now contain cycles.
  `Analysis.bases` BFS-walks forward from no-dep packages through
  the precomputed consumer reverse map (so dependencies precede
  their consumers when the graph is acyclic, and any cycle-trapped
  packages get appended in path order); `reverse_closure` walks the
  same consumer map from a single seed; `_interesting_set` walks
  the dep map from `reverse_closure`'s result. All three share one
  `_bfs_order` helper whose visited-set guard makes them cycle-safe,
  and their results memoize on `Analysis` so repeated `PackageView`
  queries share the cost. Tolerating cycles lets a package with an
  acyclic exported subset list cyclic dev/test deps without
  hand-splitting them out.
- **Breaking:** `Analysis` no longer accepts a pre-built `paths` mapping.
  The constructor now takes `project_root` as the first argument and a
  required `resolver=` keyword argument (singular -- there is no
  resolver chain). Callers that used to write
  `Analysis({base: deps}, ...)` should switch to
  `Analysis(base, resolver=ManualResolver(specs=["."]), ...)` (or
  whichever resolver describes their layout). The CLI's `-p` / `--path`
  and `--resolver` flags are mutually exclusive, and `--resolver` takes
  a single value.
- **Breaking:** `PathResolver.resolve()` now returns
  `tuple[Package, ...]` instead of a `dict[Path, list[Path]]`
  (`PathMap`). `Package` is a frozen dataclass carrying `path`,
  `name` (unique within an analysis), `exported` (subdirs visible to
  consumers; empty means "no restriction"), and `deps` (other
  packages by name). The `PathMap` type alias and `merge_paths`
  re-export are gone; `Analysis` validates a single resolver's output
  internally. Resolvers no longer represent non-package search paths
  (workspace `.venv/site-packages`, vendored bundles) in
  `Package.deps`; `UvResolver` splices the workspace venv onto
  `sys.path` lazily inside its own `resolve_import` instead.
- **Breaking:** `Analysis.paths` is replaced by `Analysis.packages`,
  which returns the `tuple[Package, ...]` the analysis was built
  with. The previous `Analysis.packages()` iterator (yielding one
  `PackageView` per base) is renamed to `Analysis.views()` to free
  the name for the new data attribute.
- **Breaking:** `UvWorkspaceResolver` is renamed to `UvResolver`, the
  CLI resolver name `uv_workspace` is renamed to `uv`, and the module
  moved from `dead_cst/contrib/uv_workspace.py` to
  `dead_cst/contrib/uv.py`. The `name` field bumps to `"uv"` so cached
  per-base fingerprints rebuild automatically.
- **Breaking:** the CLI helper `resolve_paths` is replaced by
  `build_resolver`, which returns a single `PathResolver`. Callers
  (and the CLI itself) read the package list back from
  `Analysis.packages` after construction.

### Removed
- **Breaking:** `VenvResolver` and `PyprojectResolver` are removed,
  along with the `--resolver venv` and `--resolver pyproject` CLI
  names. `MissingVenvError` is also no longer part of the resolver
  surface; `UvResolver` raises its own `dead_cst.contrib.uv.MissingVenvError`
  when the workspace's shared `.venv` is missing. Use `-p src` (the
  CLI's `ManualResolver`) for the old `pyproject` `src/` shortcut.

### Added
- `dead_cst.resolvers.Package`, the frozen dataclass every resolver
  now emits.
- `dead_cst.resolvers.clear_path_caches`, a one-call helper that
  drops the three `sys.path`-derived LRU caches
  (`safe_resolve_module`, `distribution_lookup`,
  `editable_distribution_roots`). Custom resolvers that mutate
  `sys.path` should call it instead of clearing each cache by hand.
- New primary API: `dead_cst.Analysis` and `dead_cst.PackageView`,
  the lazy entry point that callers should reach for on large repos.
  Construction is cheap (no filesystem walk, no parsing). `refresh()`
  is base-scoped and idempotent. `package(base)` returns a
  `PackageView` whose `modules` / `declarations` / `count_nodes`
  queries are local to that base, while `dead` / `reachable` /
  `kept_alive_by_dead_branches` / `importers_of` / `graph` /
  `remove_dead_code` materialize only the "interesting set" -- the
  forward closure of the base's reverse (consumer) closure -- which
  is the smallest scope that gives correct reachability answers for
  decls in that base.
- `CycloptsPlugin` (`dead_cst.contrib.cyclopts`, re-exported from
  `dead_cst.plugins` and `dead_cst.contrib`) wires
  `@<app>.command` and `@<app>.default` handlers through their owning
  `cyclopts.App` instance, mirroring the Typer/Click plugins.
  Registered in `BUILTIN_PLUGINS` under the name `cyclopts` and
  loadable via `--plugin cyclopts`.
- `MockPatchPlugin` (`dead_cst.contrib.mock_patch`, re-exported from
  `dead_cst.plugins` and `dead_cst.contrib`) resolves string-fqname
  targets passed to `unittest.mock.patch` / `mock.patch` /
  `mocker.patch` (decorator and context-manager forms) plus pytest's
  `monkeypatch.setattr("X.Y", value)` and `monkeypatch.delattr("X.Y")`
  so symbols whose only consumers are tests patching them by string
  aren't flagged as dead. `patch.object`, `patch.dict`,
  `patch.multiple`, and the object form of `monkeypatch.setattr` /
  `delattr` are intentionally not handled -- their targets are real
  references the analyzer already sees. `monkeypatch.setitem` /
  `setenv` / `syspath_prepend` / etc. are not handled either -- their
  string args name dict keys, env vars, or paths, not symbols.
  Registered in `BUILTIN_PLUGINS` under the name `mock_patch` and
  loadable via `--plugin mock_patch`.

### Changed
- **Breaking (visitor / cache / `Import` shape):** Cross-file import
  resolution moved out of the per-file visitor and into the edge
  stitcher (`dead_cst._edges.resolve_edges`). Consequences:
  - `dead_cst.graph.Import` drops `path: Path | str` and gains
    `speculative: bool = False` (set on the synthetic star imports
    the visitor produces for `__import__(name, fromlist=[...])`
    fromlist entries; the stitcher silently drops a speculative
    entry when neither the trie nor the resolver places it).
    Plugins that consumed `Import.path` should switch to
    `Import.module` (and pair it with `ctx.find_module` /
    `ctx.importers` if they need a resolved target).
  - `compute_fingerprint` no longer takes `search_paths=` /
    `resolvers=`. The per-file fingerprint covers only the visitor /
    plugin / detector versions plus the base path; resolver swaps
    and search-path changes re-stitch edges on the next analysis
    without invalidating any cached `VisitorPayload` blobs.
  - `SymbolVisitor.__init__` no longer takes `search_paths` or
    `import_resolver`; the visitor is purely a function of the
    file's source. `resolve_edges` gained
    `import_resolver=` / `search_paths=` keyword arguments for the
    trie-miss classification path.
  - `Analysis._materialize` now rebinds `sys.path` to each base's
    `(base, *deps)` view before composing it (and clears the
    resolver LRUs at every transition), restoring the original
    `sys.path` on the way out. Workers in the parallel visitor pass
    no longer touch `sys.path` at all -- the cost moves from O(files)
    to O(bases).
  - `from p import functions` (where `functions` is a submodule of
    `p`) now produces edges to `p.functions` only when the access
    path canonicalizes to that module; the previous shape sometimes
    pointed at the parent package instead. Reachability is
    unchanged; the only observable difference is which intermediate
    module appears in the edge set.
  - Visitor `version` bumped (cache wipes on first run).
- **Breaking (cache API):** `compute_fingerprint` is now per-base
  (`base=`) rather than per-project (`paths=`). Each cache row
  carries its own fingerprint, so changing one base's plugins no
  longer invalidates sibling bases' cached payloads.
  `GraphCache(db_path)` no longer takes a fingerprint at open time;
  `get` and `put` take it per call. Schema version bumped to 2;
  older databases auto-wipe on first open. The CLI is unaffected --
  `Analysis` computes per-base fingerprints internally.
- `_order_packages` (formerly `_order_paths`) returns only the
  paths of `Package` records the resolvers emit. Search paths that
  aren't themselves packages (e.g. a workspace's
  `.venv/site-packages`) are never walked; resolvers that need them
  during classification handle that inside their `resolve_import`.
- `PluginContext`, `ObserveContext`, and the `AddNode` / `AddEdge` /
  `RemoveEdge` graph-op value objects now use `__slots__`, as do the
  analyzer-internal `_BaseSpec` / `_Task` records, shaving a
  per-instance `__dict__` off objects allocated per file and per
  emitted op.

### Removed
- **Breaking (top-level API):** `build_symbol_graph`, `find_reachable`,
  `find_kept_alive_by_dead_branches`, `count_nodes`, `order_paths`,
  and the top-level `remove_code` re-export are gone. Replace with
  the equivalent `Analysis` / `PackageView` methods:
  - `build_symbol_graph(...)` -> `Analysis(...).materialize_all()`
    (returns the same `nx.MultiDiGraph` for callers that want raw
    access).
  - `find_reachable(graph)` -> `Analysis.reachable()` /
    `PackageView.reachable()`.
  - `find_kept_alive_by_dead_branches(graph)` ->
    `Analysis.kept_alive_by_dead_branches()` /
    `PackageView.kept_alive_by_dead_branches()`.
  - `count_nodes(graph, prefix)` -> `Analysis.count_nodes(prefix)` /
    `PackageView.count_nodes()`.
  - `order_paths(paths)` -> `Analysis(...).bases`.
  - `remove_code(graph, base)` -> `PackageView.remove_dead_code()`
    for the high-level entry point. The standalone function is still
    available at `dead_cst.codemod.remove_code` for power users.
- `PluginContext.prime_module`, the public method for inserting an
  already-parsed `cst.Module` into the request-scope `parse` memo.
  The analyzer never called it (warm cache hits skip parsing entirely
  and the visitor's parsed module is consumed in-process), so it was
  dead code in the public surface. Plugins that need a `cst.Module`
  during `observe` already get it for free as `ObserveContext.module`;
  during `finalize` they go through `ctx.parse(path)` and that path
  still memoizes within the analysis.

### Fixed
- Documentation no longer claims that `PluginContext.parse` is primed
  with the modules the analyzer walked during the visitor pass: that
  hasn't been true since per-file results moved into the SQLite cache
  (warm hits skip parsing entirely). The `parse` cache is now
  documented as a request-scope memo, populated lazily on first
  access.
- `from mod import MyAlias` where `MyAlias` is a PEP 695 `type` statement
  no longer raises `AssertionError` during edge resolution. `type_alias`
  declarations are now treated as concrete termination points in the
  re-export follower, the same as `function`, `class`, and `variable`.

## [0.5.0] - 2026-05-05

### Added
- New public surface: `GraphCache`, `compute_fingerprint`,
  `clear_cache`, `default_cache_path`, and `SCHEMA_VERSION` from
  `dead_cst.cache`; `VisitorPayload` from `dead_cst.graph`;
  `evaluate_truthiness`, `unreachable_suites`, `unreachable_bodies`,
  `ResolveExpr`, and `fold_constants` from `dead_cst.branches`;
  the synthetic-node prefix constants (`STDLIB_PREFIX`,
  `EXTERNAL_DIST_PREFIX`, `EXTERNAL_FILE_PREFIX`, `UNRESOLVED_PREFIX`,
  `EXTERNAL_PREFIXES`, `SYNTHETIC_PATH_PREFIXES`, `SYNTHETIC_POSITION`)
  from `dead_cst.plugins`; `safe_resolve_module`,
  `distribution_lookup`, `editable_distribution_roots`, `STDLIB`,
  `SITE_PACKAGES_MARKERS`, and `load_toml` from `dead_cst.resolvers`.
  All previously reachable only via private (underscored) modules.
- `tests/test_public_api.py` pins each public module's `__all__`
  against a snapshot so accidental drops fail loudly in CI.

- Dynamic-import calls with a string-literal argument
  (`__import__('pkg.mod')` and `importlib.import_module('pkg.mod')`)
  are now treated like `from pkg.mod import *`: every top-level decl
  in the target module is fanned out as a successor of the enclosing
  top-level decl, so `getattr(__import__('pkg.mod'), 'name')()` keeps
  `pkg.mod.name` reachable instead of being silently dropped.
  Relative names are resolved against the file's enclosing package
  the same way `from .x import *` is:
  `importlib.import_module('.sub')` from `pkg/x.py` resolves to
  `pkg.sub`, and `__import__('sub', ..., level=1)` does the same;
  an explicit `package=` literal overrides the inferred anchor.
  `__import__(name, fromlist=[...])` with a literal list/tuple is
  parsed: every entry that resolves as a submodule of `name`
  (e.g. `__import__('pkg', fromlist=['mod'])` imports `pkg.mod`
  as a side effect) is fanned out the same way, while non-resolving
  entries are silently treated as plain attributes (already covered
  by the fan-out from `name`). Non-literal arguments (name,
  `level`, `package`, `fromlist`) skip with a warning. Bumps
  `SymbolVisitor.version` to invalidate cached payloads.
- `build_symbol_graph(workers=N)` (and matching `--workers` / `-j` CLI
  flag) dispatches per-file visitor + observe passes to a
  `ProcessPoolExecutor` when at least two cache-miss files exist
  across all bases. Workers return `VisitorPayload` blobs to the
  main process, which still owns SQLite cache writes, trie
  stitching, and edge resolution; serial behaviour and graph output
  are unchanged. A single persistent pool spans the whole run with
  tasks sorted by `search_paths`, so any one worker tends to see
  contiguous miss runs from the same base; on each transition the
  worker rebinds `sys.path` and clears `safe_resolve_module` plus
  `distribution_lookup` so cross-venv uv-workspace members don't
  inherit a sibling base's resolution state. The FQN provider's
  per-base cache is now built once in the parent over miss files
  only and shipped per-task to workers, so workers no longer rebuild
  a `FullRepoManager` and the analyzer skips FQN computation for
  cache-hit files entirely. `workers=None` (default) and
  `workers=1` keep the in-process path.

### Changed
- `build_symbol_graph` runs as three phases: collect per-base specs
  (cache hits + miss files + per-base FQN cache), compute every miss
  payload (in-process or via the pool), then per-base apply + edge
  stitch + plugin finalize. The graph and cache contents are
  unchanged. The in-process and worker paths share a single
  `_process_task` body — the only difference is whether the runner
  state lives on the main process or in worker globals.

### Removed
- The internal `temp_sys_path` context manager
  (`dead_cst.resolvers._imports`). The runner now manages
  `sys.path` directly, restoring it from a baseline snapshot when
  the in-process path finishes. Not part of the public API.

### Changed (breaking)
- Public modules dropped their leading underscore: `_analyze` →
  `analyze`, `_branches` → `branches`, `_cache` → `cache`, `_codemod`
  → `codemod`, `_plugins` → `plugins`, `_resolvers` → `resolvers`,
  and `_symbols` was renamed to `graph` (with `VisitorPayload` moved
  in from `_visitor`). The `explicit` plugin module was renamed to
  `explicit_entrypoint` to match its class name.
- New top-level `dead_cst.contrib` package collects every
  third-party-aware extension: framework plugins (`FastAPIPlugin`,
  `FlaskPlugin`, `ClickPlugin`, `TyperPlugin`, `PytestPlugin`,
  `UnittestPlugin`) and `UvWorkspaceResolver`. They are re-exported
  from `dead_cst.plugins` and `dead_cst.resolvers` for ergonomics, so
  `from dead_cst.plugins import FastAPIPlugin` and
  `from dead_cst.resolvers import UvWorkspaceResolver` keep working.
- The top-level `dead_cst` package no longer re-exports every plugin
  and resolver class. The curated highlights remain importable from
  `dead_cst` directly (`build_symbol_graph`, `find_reachable`,
  `find_kept_alive_by_dead_branches`, `count_nodes`, `order_paths`,
  `remove_code`, `Cacheable`, `SymbolNode`, `Import`, `NodeFlags`,
  `EdgeFlags`, `__version__`). Plugin and resolver classes must now
  be imported from `dead_cst.plugins`, `dead_cst.resolvers`, or
  `dead_cst.contrib`.
- Modules still prefixed with `_` (`_visitor`, `_edges`, `_flow`,
  `_fqn`, `_const_fold`, `_cacheable`, `_version`) are internal and
  not part of the supported surface.

### Fixed
- Path classification in `default_resolve_import` no longer
  misclassifies third-party packages as stdlib when running against
  a Python install whose `site-packages` is nested *inside* the
  stdlib root (the typical layout for a system Python with no venv,
  e.g. `/usr/local/lib/python3.13/site-packages` under
  `/usr/local/lib/python3.13`). The stdlib check now excludes paths
  under `purelib` / `platlib` and any directory named
  `site-packages` / `dist-packages`.
- Editably-installed third-party packages (`pip install -e`,
  `uv pip install -e`) are now resolved to `[external dist] <name>`
  instead of raising `Module ... resolved to an unexpected path`.
  Distribution metadata is consulted via PEP 610
  `direct_url.json` and any `.pth` shims in the dist's
  `RECORD`, so editable source dirs that live outside the project's
  search paths still get attributed to their owning distribution.
  The new cache (`editable_distribution_roots`) is cleared alongside
  `distribution_lookup` on worker venv transitions. All four
  shipping resolvers (`venv`, `pyproject`, `uv_workspace`,
  `manual`) bump their `version` so cached `VisitorPayload` blobs
  rebuild against the corrected classification.
- First-party search paths now win over editable distribution roots
  in `default_resolve_import`. Previously a project whose source
  happened to live inside another editable install's root (e.g. an
  e2e fixture cloned into `.pytest_cache/` of an editable
  `dead-cst` checkout) had every first-party file misclassified as
  `[external dist] <host-pkg>`, severing cross-module edges and
  reporting the entire surface as dead. The four shipping resolvers
  bump their `version` again so cached payloads rebuild.

## [0.4.0] - 2026-05-03

### Added
- PEP 695 `type` statements are now tracked. `type Foo = list[int]`
  surfaces `mod.Foo` as a top-level declaration of kind `"type_alias"`,
  and refs in the RHS are attributed to the alias so removing a dead
  alias releases its references rather than holding them through the
  enclosing module. The generic form (`type Pair[T] = tuple[T, T]`)
  is captured the same way. Cross-module users of the alias (e.g.
  `def f(x: Foo) -> Foo`) get a normal edge into the alias decl.
  `dead-cst remove` deletes unreachable aliases through a new
  `leave_TypeAlias` pass on `RemoveDeadSymbols`. Bumps
  `SymbolVisitor.version` to invalidate cached payloads.

## [0.3.0] - 2026-05-03

### Added
- PEP 572 walrus (`:=`) bindings at module scope are now surfaced as
  top-level declarations. `if (Y := src()): ...` registers `mod.Y`
  with an outgoing edge to whatever `src` resolves to, and downstream
  references like `def use(): return Y` get a `mod.use -> mod.Y`
  edge. Walruses leaked from a module-level comprehension (e.g.
  `result = [last := n for n in nums]`) are also captured: libcst's
  `ScopeProvider` keeps the binding inside the comprehension scope,
  so `SymbolVisitor` patches the gap by routing any unresolved Name
  access whose `.value` matches a leaked walrus target back to the
  matching decl. The default unreachable-region detector folds
  walrus bindings the same way it folds `Assign` / `AnnAssign` --
  `(DEBUG := False)` and `if (DEBUG := False):` both flag dead
  branches. Walruses inside a function / class / lambda body still
  bind locally, matching Python's runtime semantics.
- `SymbolVisitor` now satisfies the `Cacheable` protocol with
  class-level `name: str = "default"` and an epoch `version: int`,
  and `compute_fingerprint` includes the pair in the cache key.
  Bump `SymbolVisitor.version` on any change to the visitor's
  per-file output (new node kinds surfaced as decls, edge-attribution
  rules, flow-analysis fixes, etc.) so stale `VisitorPayload` blobs
  invalidate even between releases. Concurrent bumps on different
  branches merge with `max()` semantics. The walrus-support change
  bumps the visitor version accordingly.
- `UnreachableRegionDetector`: pluggable Protocol for module-level
  dead-region detection. Implementers provide
  `find_regions(wrapper) -> list[CodeRange]` and a `(name, version)`
  pair; consumers pass an instance via the new
  `build_symbol_graph(unreachable_detector=...)` keyword. Lets a
  company fold domain knowledge (e.g. "`settings.IS_PROD` is always
  `True` in production") into the analysis without forking the
  package. The shipped `DefaultUnreachableRegionDetector` preserves
  existing behavior — literal-only truthiness on `if` / `while`
  tests. The detector runs from inside `SymbolVisitor.visit_Module`
  reusing the analyzer's already-resolved `PositionProvider`, so the
  abstraction is free for the default path.
- `dead_cst._const_fold.fold_constants(wrapper, resolve_expr=None)`:
  fixpoint constant-folding pass that returns a `dict[id(Name), bool]`
  of every access whose binding ties back to a simple `Name = literal`
  (or `Name: T = literal`) assignment. Iteration is the point: chained
  forms like `foo = False; bar = foo or False; if bar: ...` resolve
  fully because each pass propagates one more level of indirection.
  Flow-sensitive (a later rebinding shadows an earlier one) and
  conservative (mixed-value bindings, non-literal RHS, unsupported
  shapes, and cycles all stay unknown). The optional `resolve_expr`
  callback gets first crack at any expression encountered during RHS
  evaluation, so domain-specific truthiness composes with the literal
  fold automatically — `flag = check_flag("x"); if flag:` resolves
  when the resolver answers for the call.
- `DefaultUnreachableRegionDetector` now runs three passes per file:
  the literal-only `unreachable_suites` walk, the new `fold_constants`
  pre-pass, and a post-terminator scan over every statement-bearing
  suite. Patterns like `DEBUG = False; if DEBUG: ...`, `return`
  followed by dead code, and `assert False` followed by dead code are
  all flagged out of the box. Post-terminator detection is purely
  suite-relative, so a `raise` inside a `try` body kills the rest of
  the try body without touching the `except` handler, which runs on
  its own path.
- `DefaultUnreachableRegionDetector.resolve(self, expr) -> bool | None`:
  overridable hook for subclasses to fold non-literal expressions to a
  known truthiness — e.g. `check_flag("migration-abc")` is always
  `True` in production. The default returns `None` (defer to literal
  handling). The override is consulted recursively for every
  non-keyword expression in every `if` / `while` / `assert` test and
  every foldable assignment RHS; folded values flow through the
  fixpoint loop alongside `Name = literal` bindings, so a single
  high-level decision propagates through chains. Subclasses bump
  `version` (epoch-int) for cache invalidation.
- `evaluate_truthiness` and `unreachable_suites` / `unreachable_bodies`
  now accept `resolve_expr: Callable[[cst.BaseExpression], bool | None]`
  in place of the previous `Name`-only `resolve_name` callback. The
  resolver is consulted for any non-keyword expression and short-
  circuits the built-in literal handling when it returns a `bool`;
  language keywords (`True` / `False` / `None`) always resolve to
  their language-defined truthiness and are never passed through.
  Detector `version` set to the current Unix epoch (`1777795837`),
  matching the convention used by every other shipped `Cacheable`,
  so any cached `VisitorPayload` from the prior detector is
  automatically invalidated and concurrent bumps merge with `max()`
  semantics.
- `Cacheable` Protocol (`name: str`, `version: int`): the shared
  cache-fingerprint contract that `EdgePlugin`, `PathResolver`, and
  `UnreachableRegionDetector` now all inherit from. Bumping a
  component's epoch `version` invalidates the per-file cache the same
  way it does for plugins. `compute_fingerprint` reads the attributes
  directly instead of falling back to `getattr` defaults.
- `DecoratedDeclPlugin`: abstract `EdgePlugin` base for the "find decls
  decorated by `@<module>.<name>(...)` or assigned via
  `X = <module>.<ctor>(...)` in files matching a search path" idiom.
  Subclasses set `package_prefix`, `decorator_module`, `decorator_names`,
  `constructor_names`. Pure observe-time, so matches turn directly into
  cached entrypoint payloads. `ClickPlugin` is now a subclass that adds
  the nested-group fixpoint pass and overrides `observe` to emit
  `instance -> handler` edges instead of seeding entrypoints.
- `LiteralListPlugin`: abstract `EdgePlugin` base for the "read
  `<owner>.<var> = ["a.b.c", ...]` and revive every fqname inside" idiom.
  observe parses the literal once and emits ENTRYPOINT-flagged synthetic
  decls (one per entry, positioned at the literal's site); finalize is a
  graph-only pass that adds the cross-file edges. Owner-file-only CST
  work is cached alongside the per-file payload, so warm runs do zero
  parsing for this plugin. Both bases are abstract -- subclasses must
  set `name` and `version`.
- `PluginContext.module_surface(fqname)`: returns the module's
  `SymbolNode` plus every top-level decl plus every transitive
  submodule's surface, walked via the symbol trie in
  `O(decls_in_subtree)`. Replaces hand-rolled scans of
  `ctx.base_nodes()` for the common "this module is loaded
  dynamically -- keep its surface alive" pattern.
- `synthetic_node(..., position=...)`: optional `CodeRange` to anchor
  a synthetic at a specific source location so `why-alive` and the
  codemod report the right line. Defaults to `SYNTHETIC_POSITION` for
  backwards compatibility.
- New public re-exports under `dead_cst._plugins` (and the most-used
  ones at the top level): `ObserveContext`, `make_payload` (renamed
  from the private `_payload_from`), `mark_entrypoints`,
  `decls_by_simple_name`, `simple_name`, `collect_module_imports`,
  `matched_attr_call`, `single_target_assignment`, `find_handlers`,
  `find_call_assignments`, `decorator_owner`, `is_name`,
  `is_from_module`. These are the helpers user-written plugins reach
  for; previously they lived in `_core` and required private imports.
- E2E test suite under `tests/e2e/`, deselected by default
  (`addopts = "-m 'not e2e'"`); run with `uv run pytest -m e2e`. The
  first target is `flux0-ai/flux0` pinned at SHA `8d04176`. Tests
  cover three levels: analyze runs to completion, `why-alive` chains
  for known-alive symbols, and project-specific plugins that close
  the repo's `importlib`-driven blind spots. `tests/e2e/conftest.py`
  exposes a `clone_repo(name, url, sha)` fixture that shallow-clones
  into pytest's cache dir (or `DEAD_CST_E2E_CACHE`) with a SHA
  marker for idempotent reuse.

### Changed
- The package `__version__` is no longer folded into the cache
  fingerprint. Every component whose output can shift between
  releases (visitor, resolvers, plugins, detector) already carries
  its own `Cacheable` `(name, version)` knob; mixing `__version__`
  in on top let unbumped components ride for free on a release bump
  and masked cases where the granular versions weren't being
  maintained. The discipline now is to bump the relevant component's
  `version`. Schema version and Python version still participate.
- `PathResolver` is now a `Cacheable` Protocol: shipped resolvers
  (`ManualResolver`, `PyprojectResolver`, `UvWorkspaceResolver`,
  `VenvResolver`) all carry an epoch `version: int` matching the
  plugin convention, and the per-file cache fingerprint includes
  each resolver's `(name, version)` pair. Bump a resolver's
  `version` when its layout-discovery or `resolve_import` logic
  changes; stale `VisitorPayload` blobs rebuild automatically.
- `EdgePlugin.version` is now `int` (Unix epoch by convention) rather
  than `str`. Bump `version` to the current epoch on any change to a
  plugin's `observe` shape that should not be served from older
  caches; concurrent bumps on different branches merge with `max()`
  semantics rather than colliding on a re-used label like `"2"`. All
  builtin plugins migrated; the cache fingerprint format follows.
- `_decls_by_simple_name` -- a four-line helper duplicated across
  `ClickPlugin`, `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin` -- is
  hoisted into `dead_cst._plugins._core.decls_by_simple_name` and
  re-exported. Same behaviour, one definition.
- `ClickPlugin` now subclasses `DecoratedDeclPlugin`, dropping ~70
  lines of duplicated decorator-finding code. Behaviour is preserved
  -- the plugin still emits `instance -> handler` edges and does not
  seed Click groups as entrypoints.

## [0.2.0] - 2026-05-01

### Added
- SQLite-backed `GraphCache` keyed by per-file SHA-256 hashes,
  storing pickled `VisitorPayload` blobs under
  `<root>/.dead-cst-cache/cache.db`. Cache hits skip the per-file
  visitor pass (the dominant cost in `build_symbol_graph`); the
  per-base `resolve_edges` step and plugin pass run unconditionally,
  so a graph built from a warm cache is identical to one built from
  scratch. The cache is keyed by a fingerprint over
  `(__version__, python version, PathMap, resolver chain)`; a
  fingerprint mismatch wipes `file_cache` and rebuilds. Plugins are
  intentionally **not** part of the fingerprint -- swapping plugins
  reuses cached payloads. New `--no-cache` flag on `analyze`,
  `why-alive`, `unused-exports`, `dependencies`, and `remove`; new
  `dead-cst cache clear` subcommand. `build_symbol_graph` accepts a
  new `cache=` keyword.
- `ManualResolver`: a `PathResolver` built from explicit
  ``base:dep1,dep2`` specs. The CLI's ``-p`` flag now flows through
  this resolver, so explicit specs sit in the same chain as named
  resolvers and participate in `resolve_import` lookups too.
- `NodeFlags.ENTRYPOINT`: a node flag that `_apply_payload` reads to
  set `graph.nodes[node]["entrypoint"] = True`. Plugin observe passes
  emit synthetic nodes flagged `ENTRYPOINT` to declare reachability
  seeds without a separate API surface.

### Changed
- `EdgePlugin` is now a two-pass protocol:
  - `observe(ctx) -> VisitorPayload | None` runs in the per-file
    analyzer loop with the file's parsed CST and just-built
    `VisitorPayload`. The plugin returns the same payload shape
    (`nodes` + `edges`) and the analyzer concatenates that with the
    visitor's payload before applying. Plugin contributions are
    cached alongside the visitor's output, so a warm cache hit
    returns the combined payload in one read.
  - `finalize(ctx) -> Iterable[GraphOp]` runs once per base after
    `resolve_edges`. It is graph-only -- no CST access -- and is
    where cross-file work belongs (FastAPI's factory walk,
    InitSubclass's transitive subclass closure, `[project.scripts]`
    lookups). Plugins use synthetic markers from their `observe`
    pass to communicate state forward into `finalize` (e.g.
    `<fastapi-pending>:<X.fqname>` for variables that need a graph
    walk to classify).
  - `EdgePlugin.contribute` is replaced by `observe` + `finalize`.
    Every builtin plugin -- `MainBlockPlugin`,
    `ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`,
    `ModuleDundersPlugin`, `PytestPlugin`, `UnittestPlugin`,
    `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin`, `ClickPlugin`,
    `InitSubclassPlugin` -- migrates accordingly.
  - Plugins declare a `version: str` attribute. The cache
    fingerprint includes each plugin's `(name, version)` pair, so
    bumping a plugin's version invalidates the file_cache (its
    observe contributions are baked into cached payloads).
- Warm cache runs with the full builtin plugin set now parse **zero**
  files: the visitor and every plugin's observe contributions are
  baked into the cached payloads, and `finalize` runs purely off the
  graph. Pinned by `test_warm_run_with_plugins_parses_zero_files`.
- `PathResolver` protocol now includes a `resolve_import(name, search_paths)`
  method, folding `name -> path` lookup into the resolver alongside
  search-path discovery. The shipped resolvers (`ManualResolver`,
  `PyprojectResolver`, `UvWorkspaceResolver`, `VenvResolver`) all
  delegate to the new `dead_cst._resolvers.default_resolve_import`, the
  `sys.path` + `importlib` implementation. Custom resolvers can now
  override import lookups for their own layout (vendored deps, `.pyi`
  siblings, ...) without monkey-patching internals.
- `build_symbol_graph` accepts a new `resolvers=` keyword whose entries'
  `resolve_import` methods are tried in order. With no resolvers it
  falls back to `default_resolve_import`, preserving today's behavior.
- The CLI threads loaded resolvers through to the analyzer so
  `--resolver` (and ``-p``) selections govern import lookup, not just
  search paths.
- Renamed `dead_cst._resolve` to `dead_cst._edges`. The remaining
  module is purely about edge construction in the symbol trie;
  resolution now lives under `dead_cst._resolvers`.

### Removed
- `dead_cst.cli.parse_paths` -- callers should construct a
  `ManualResolver` and call `.resolve(root)`.

### Fixed
- `MainBlockPlugin` now keeps decls bound inside the
  `if __name__ == "__main__":` block alive, not just the containing
  module. Previously a top-level decl introduced by an assignment in
  the block (e.g. `app = Foo(fn=main).cli()`) had no incoming edge --
  the visitor's value-frame produced `app -> Foo` / `app -> main`, but
  nothing pointed at `app` itself -- so the chain was unreachable and
  `Foo` / `main` were reported dead. The plugin now resolves the
  block's `CodeRange` via `PositionProvider` and emits `synth -> decl`
  edges for every top-level decl whose binding site falls inside the
  block.

## [0.1.0] - 2026-04-28

Initial alpha release. `dead-cst` is pre-1.0 software: the public Python API,
CLI flags, and output formats may change without notice between any two
versions until the first stable release.

### Added
- Symbol-level reachability analysis built on LibCST's
  `FullyQualifiedNameProvider` and `ScopeProvider`.
- Resolution of relative imports, aliased imports, and re-export chains
  through `__init__.py`.
- `dead-cst analyze` CLI for reporting unreachable symbols and unreachable
  branches (`if False:`, raise-only suites, etc.), with `text` and `json`
  output formats.
- `dead-cst why-alive` CLI for explaining why a symbol is kept alive.
- `dead-cst remove` CLI that rewrites files in place via a LibCST codemod,
  with import pruning when the last local user of an import is deleted and
  position-aware shadowing so a shadowed dead binding no longer drags its
  live sibling out with it.
- `dead-cst unused-exports` CLI command: report `__all__` entries whose
  targets are kept alive only because they are listed in `__all__`.
- `dead-cst dependencies` CLI command: list third-party distributions and
  files imported by the codebase, surfaced as synthetic
  `[external dist] <name>` / `[external file] <name>` graph nodes.
- Multi-package / monorepo support via the `-p base:dep1,dep2` search-path
  spec, with topological ordering of bases.
- Edge plugin architecture (`EdgePlugin`, `CSTAwareEdgePlugin`,
  `PluginContext`, `GraphOp`/`AddNode`/`AddEdge`/`RemoveEdge`, `apply_ops`,
  `synthetic_node`). Built-in plugins: `MainBlockPlugin`,
  `ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`, `ModuleDundersPlugin`,
  `PytestPlugin`, `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin`,
  `ClickPlugin`, `UnittestPlugin`, `InitSubclassPlugin`. Third-party
  plugins register under the `dead_cst.plugins` entry-point group and load
  via `load_plugin`.
- `PytestPlugin` (`--plugin pytest`): keep pytest-discovered tests,
  `conftest.py` decls, and `@pytest.fixture` functions alive.
- `FastAPIPlugin` (`--plugin fastapi`): detect top-level `FastAPI()` and
  `APIRouter()` instances (including factory-style apps), mark `FastAPI`
  apps as entrypoints, and emit `instance -> handler` edges for
  `@app.get(...)`-style decorators (HTTP methods, websockets, middleware,
  exception handlers, `on_event`). Routers stay pass-through, so an
  `APIRouter` that's never `include_router`'d remains dead.
- `FlaskPlugin` (`--plugin flask`): detect top-level `Flask()` /
  `Blueprint()` instances (including factory-style apps) and emit
  `instance -> handler` edges for `@app.route(...)`, HTTP-verb shortcuts,
  request-lifecycle hooks (`before_request`, `after_request`,
  `teardown_*`), error handlers, template helpers (`context_processor`,
  `template_filter`, ...), and URL processors. `Flask` apps are seeded as
  entrypoints (WSGI servers load `module:app`); `Blueprint`s stay
  pass-through, so a blueprint never `register_blueprint`'d remains dead,
  mirroring the `APIRouter` behavior in `FastAPIPlugin`.
- `TyperPlugin` (`--plugin typer`): detect top-level `Typer()` instances
  and emit `instance -> handler` edges for `@app.command(...)` and
  `@app.callback(...)` decorators. Typer apps are pass-through;
  reachability is expected through `[project.scripts]` or a `__main__`
  block, after which every registered command and callback stays alive.
  Sub-typers that are never `add_typer`'d remain dead.
- `ClickPlugin` (`--plugin click`): detect top-level Click `Group`
  instances (functions decorated `@click.group(...)` / `@click.Group(...)`,
  `X = click.Group(...)` constructor calls, and inline sub-groups
  registered via `@<group>.group(...)`, all resolved via fixpoint so a
  chain of nested groups is fully discovered) and emit
  `instance -> handler` edges for `@<group>.command(...)`,
  `@<group>.group(...)`, and `@<group>.result_callback(...)` decorators.
  Click groups stay pass-through; reachability is expected through
  `[project.scripts]` or a `__main__` block, mirroring `TyperPlugin`.
- `UnittestPlugin` (`--plugin unittest`): mark stdlib `unittest.TestCase`
  and `unittest.IsolatedAsyncioTestCase` subclasses, plus module-level
  `setUpModule` / `tearDownModule` / `load_tests` hooks, as entrypoints.
  Discovery is CST-based and prefiltered to files whose import nodes
  reference `unittest`. Supports `import unittest` (with alias),
  `from unittest import TestCase` (with alias), and module-prefixed base
  references. Only direct base-class matches are recognised; transitive
  subclasses through a project-local mixin need an explicit `-e`
  entrypoint or coverage from `PytestPlugin`'s filename heuristics.
  `from unittest import *`-only files are skipped (the resolver doesn't
  surface stdlib star imports as graph nodes); use a non-star import.
- `InitSubclassPlugin` (`--plugin init_subclass`): detect classes that
  define `__init_subclass__` and route reachability through a synthetic
  marker node `<__init_subclass__>:<parent.fqname>` with edges
  `parent -> marker -> subclass` for every transitive first-party
  subclass. Registry-pattern subclasses stay alive whenever the parent
  class does; the marker shows up in `why-alive` chains as a labeled
  breadcrumb. Parents are pass-through, so a registry base nobody else
  uses still surfaces as dead code.
- `ModuleDundersPlugin`: keep module-level dunder variables (`__all__`,
  `__version__`, `__future__` imports, etc.) alive. Always registered by
  the CLI.
- Path resolver architecture (`PathResolver`, `merge_paths`). Built-in
  resolvers: `VenvResolver`, `PyprojectResolver`, `UvWorkspaceResolver`
  (parses `uv.lock` to discover workspace members and inter-member
  edges, including virtual workspace members that don't ship as wheels).
  Third-party resolvers register under `dead_cst.resolvers` and load via
  `load_resolver`.
- `exported_roots(base)` in `dead_cst._resolvers`: inspect a base's
  `pyproject.toml` (src-layout, hatchling/setuptools/poetry/pdm/flit
  backends, name-match fallback) to determine which subdirs the build
  backend would actually ship, so internal dirs like `tests/` stay scoped
  to their owning workspace member during cross-member import resolution.
- `--resolver` and `--plugin` flags on `analyze`, `why-alive`,
  `unused-exports`, and `remove` for selecting path resolvers and edge
  plugins.
- Public Python API: `build_symbol_graph`, `find_reachable`,
  `count_nodes`, `order_paths`, `remove_code`, plus a `position` field
  on `SymbolNode`.
- `py.typed` marker for downstream type-checking.
- `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and `ROADMAP.md` with a
  stack-ranked plan from alpha to 1.0.

[Unreleased]: https://github.com/lpetre/dead-cst/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/lpetre/dead-cst/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/lpetre/dead-cst/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/lpetre/dead-cst/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lpetre/dead-cst/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lpetre/dead-cst/releases/tag/v0.1.0
