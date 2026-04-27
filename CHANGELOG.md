# Changelog

All notable changes to `dead-cst` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first stable release the public API and CLI may change between any
two versions.

## [Unreleased]

### Added
- `InitSubclassPlugin` (`--plugin init_subclass`): detect classes that define
  `__init_subclass__` and route reachability through a synthetic marker node
  `<__init_subclass__>:<parent.fqname>` with edges
  `parent -> marker -> subclass` for every transitive first-party subclass.
  Registry-pattern subclasses stay alive whenever the parent class does;
  the marker shows up in `why-alive` chains as a labeled breadcrumb.
  Parents are pass-through, so a registry base nobody else uses still
  surfaces as dead code.
- Edge plugin architecture (`EdgePlugin`, `CSTAwareEdgePlugin`, `PluginContext`,
  `GraphOp`/`AddNode`/`AddEdge`/`RemoveEdge`, `apply_ops`, `synthetic_node`).
  Built-in plugins: `MainBlockPlugin`, `ProjectScriptsPlugin`,
  `ExplicitEntrypointPlugin`, `ModuleDundersPlugin`, `PytestPlugin`,
  `FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin`, `ClickPlugin`,
  `InitSubclassPlugin`. Third-party plugins register under the
  `dead_cst.plugins` entry-point group and load via `load_plugin`.
- Path resolver architecture (`PathResolver`, `merge_paths`). Built-in
  resolvers: `VenvResolver`, `PyprojectResolver`, `UvWorkspaceResolver`. Third-
  party resolvers register under `dead_cst.resolvers` and load via
  `load_resolver`.
- `exported_roots(base)` in `dead_cst._resolvers`: inspect a base's
  `pyproject.toml` (src-layout, hatchling/setuptools/poetry/pdm/flit
  backends, name-match fallback) to determine which subdirs the build
  backend would actually ship, so internal dirs like `tests/` stay scoped
  to their owning workspace member during cross-member import resolution.
- `TyperPlugin` (`--plugin typer`): detect top-level `Typer()` instances and
  emit `instance -> handler` edges for `@app.command(...)` and
  `@app.callback(...)` decorators. Typer apps are pass-through; reachability
  is expected through `[project.scripts]` or a `__main__` block, after which
  every registered command and callback stays alive. Sub-typers that are
  never `add_typer`'d remain dead.
- `FlaskPlugin` (`--plugin flask`): detect top-level `Flask()` / `Blueprint()`
  instances and emit `instance -> handler` edges for `@app.route(...)`,
  HTTP-verb shortcuts (`@app.get(...)`, ...), request-lifecycle hooks
  (`before_request`, `after_request`, `teardown_*`), error handlers,
  template helpers (`context_processor`, `template_filter`, ...), and URL
  processors. `Flask` apps are seeded as entrypoints (WSGI servers load
  `module:app`); `Blueprint`s stay pass-through, so a blueprint never
  `register_blueprint`'d remains dead, mirroring the `APIRouter` behavior
  in `FastAPIPlugin`.
- `ClickPlugin` (`--plugin click`): detect top-level Click `Group` instances
  (functions decorated `@click.group(...)` / `@click.Group(...)`,
  `X = click.Group(...)` constructor calls, and inline sub-groups
  registered via `@<group>.group(...)`, all resolved via fixpoint so a
  chain of nested groups is fully discovered) and emit `instance ->
  handler` edges for `@<group>.command(...)`, `@<group>.group(...)`, and
  `@<group>.result_callback(...)` decorators. Click groups stay
  pass-through; reachability is expected through `[project.scripts]` or a
  `__main__` block, mirroring `TyperPlugin`.
- `dead-cst unused-exports` CLI command: report `__all__` entries whose targets
  are kept alive only because they are listed in `__all__`.
- `dead-cst dependencies` CLI command: list third-party distributions and
  files imported by the codebase, surfaced as synthetic
  `[external dist] <name>` / `[external file] <name>` graph nodes.
- `dead-cst analyze` now reports unreachable branches (`if False:`,
  raise-only suites, etc.) alongside dead symbols, in both the text and JSON
  output formats.
- `--resolver` and `--plugin` flags on `analyze`, `why-alive`, `unused-exports`,
  and `remove` for selecting path resolvers and edge plugins.
- Position-aware shadowing in the codemod: same-name decls in different
  branches are tracked by `(fqname, position)` so a shadowed dead binding no
  longer drags its live sibling out with it.
- Import pruning in `dead-cst remove`: when the last local user of an import
  is deleted, the import line is removed via libcst's `RemoveImportsVisitor`.
- Pre-release notice in the README and a `position` field on `SymbolNode`.
- `ROADMAP.md` with a stack-ranked plan from alpha to 1.0.
- Public-API docstrings across `build_symbol_graph`, `find_reachable`,
  `count_nodes`, `order_paths`, `remove_code`, and the package `__init__`.
- `CONTRIBUTING.md` and `CHANGELOG.md`.

### Changed
- `find_reachable(graph)` no longer takes an entrypoints argument. Entrypoints
  are seeded by plugins via `graph.nodes[node]["entrypoint"] = True`, so the
  reachability walk is purely graph-driven.
- `build_symbol_graph` now accepts `plugins=` and `project_root=` keyword
  arguments and runs the plugin pipeline before returning. Per-consumer
  lookup tries are now scoped via `exported_roots`, so a workspace member's
  internal `tests/` package is no longer visible to its dependents.
- `_resolvers.py` was split into a `_resolvers/` package mirroring
  `_plugins/` (one submodule per built-in resolver, plus `_core.py` and
  `_exports.py`). Public imports from `dead_cst` and `dead_cst._resolvers`
  are unchanged.
- `--preserve-dunder-all` removed; dunder preservation is provided by
  `ModuleDundersPlugin`, which is always registered by the CLI and covers
  every module-level `__xxx__` variable, not just `__all__`.
- `SymbolTrie.merge` softened from a hard assertion to a first-wins-with-
  warning when two members declare the same module, so the analyzer logs
  a clear message instead of crashing on residual collisions.

### Fixed
- `try/finally` no longer raises `TypeError` in the flow-sensitive filter
  (`_flow.py`): `Try.finalbody` is a `cst.Finally` whose `.body` is an
  `IndentedBlock`, so the recursive walk now drills one level further to
  match every other branch.
- `UvWorkspaceResolver` now picks up workspace members whose lockfile entry
  uses `source = { virtual = "..." }` (uv's marker for runnable apps/services
  that don't ship as wheels), not just `editable` ones. The workspace root
  itself (`virtual = "."`) is still skipped.

## [0.1.0] - Initial release

### Added
- Symbol-level reachability analysis built on LibCST's
  `FullyQualifiedNameProvider` and `ScopeProvider`.
- Resolution of relative imports, aliased imports, and re-export chains
  through `__init__.py`.
- `dead-cst analyze` CLI for reporting unreachable symbols, with `text` and
  `json` output formats.
- `dead-cst why-alive` CLI for explaining why a symbol is kept alive.
- `dead-cst remove` CLI that rewrites files in place via a LibCST codemod.
- Multi-package / monorepo support via the `-p base:dep1,dep2` search-path
  spec, with topological ordering of bases.
- Public Python API: `build_symbol_graph`, `find_reachable`, `count_nodes`,
  `order_paths`, `remove_code`.
- `py.typed` marker for downstream type-checking.

[Unreleased]: https://github.com/lpetre/dead-cst/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lpetre/dead-cst/releases/tag/v0.1.0
