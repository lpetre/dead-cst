# Changelog

All notable changes to `dead-cst` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first stable release the public API and CLI may change between any
two versions.

## [Unreleased]

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
- `NodeFlags.ENTRYPOINT`: a node flag that `_apply_payload` reads to
  set `graph.nodes[node]["entrypoint"] = True`. Plugin observe passes
  emit synthetic nodes flagged `ENTRYPOINT` to declare reachability
  seeds without a separate API surface.
- Warm cache runs with the full builtin plugin set now parse **zero**
  files: the visitor and every plugin's observe contributions are
  baked into the cached payloads, and `finalize` runs purely off the
  graph. Pinned by `test_warm_run_with_plugins_parses_zero_files`.

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

### Changed
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

[Unreleased]: https://github.com/lpetre/dead-cst/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lpetre/dead-cst/releases/tag/v0.1.0
