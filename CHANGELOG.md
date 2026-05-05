# Changelog

All notable changes to `dead-cst` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first stable release the public API and CLI may change between any
two versions.

## [Unreleased]

### Added
- Dynamic-import calls with a string-literal argument
  (`__import__('pkg.mod')` and `importlib.import_module('pkg.mod')`)
  are now treated like `from pkg.mod import *`: every top-level decl
  in the target module is fanned out as a successor of the enclosing
  top-level decl, so `getattr(__import__('pkg.mod'), 'name')()` keeps
  `pkg.mod.name` reachable instead of being silently dropped.
  Non-literal arguments (`__import__(var)`) are skipped with a
  warning; relative-name literals (`'.sub'`) are likewise skipped
  rather than misresolved. `__import__(name, fromlist=[...])` with
  a literal list/tuple is parsed: every entry that resolves as a
  submodule of `name` (e.g. `__import__('pkg', fromlist=['mod'])`
  imports `pkg.mod` as a side effect) is fanned out the same way,
  while non-resolving entries are silently treated as plain
  attributes (already covered by the fan-out from `name`).
  Non-literal fromlists warn. Bumps `SymbolVisitor.version` to
  invalidate cached payloads.

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

[Unreleased]: https://github.com/lpetre/dead-cst/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/lpetre/dead-cst/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/lpetre/dead-cst/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lpetre/dead-cst/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lpetre/dead-cst/releases/tag/v0.1.0
