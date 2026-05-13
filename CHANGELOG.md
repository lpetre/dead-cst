# Changelog

All notable changes to `dead-cst` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until the first stable release the public API and CLI may change between any
two versions.

## [Unreleased]

### Changed
- `PackageContribution` no longer carries an `nx.MultiDiGraph`. The
  per-package intermediate now exposes raw `frozenset[SymbolNode]` /
  `frozenset[(src, dst, EdgeFlags)]` / `Mapping[Path, tuple[CodeRange, ...]]`
  fields plus the trie and import-edges; the target graph is built
  once at compose time via `target_graph.add_nodes_from` /
  `add_edges_from`. `module_nodes` is dropped — `PluginContext.package_modules`
  filters `package_nodes` by `type == "module"` on demand.

- `PluginContext.package_graph` and `PluginContext.module_nodes` are
  **removed**. `package_nodes` is now a `frozenset[SymbolNode]` field
  (was an iterator method with internal caching). Plugin call sites
  change from `ctx.package_nodes()` to `ctx.package_nodes`. Third-party
  plugins that reached `ctx.package_graph` for graph queries need to
  call `ctx.graph` (the full target graph) instead.

- `AddNode` drops its `entrypoint: bool` / `testcase: bool` fields.
  Plugins that need an entrypoint synthetic construct the node with
  `synthetic_node(..., flags=NodeFlags.ENTRYPOINT)` directly; the bit
  on `SymbolNode.flags` is now the only source of truth.

- `_find_reachable` reads `n.flags & NodeFlags.ENTRYPOINT` instead of
  `graph.nodes[n].get("entrypoint")`. The attr-dict mirror in
  `_apply_payload` and `apply_ops` is gone — same change for
  `TESTCASE`. Existing graph node attrs (`entrypoint`, `testcase`) are
  no longer set or read anywhere.

- `SymbolTrie.add_module_hierarchy_edges(graph)` (mutator) renamed to
  `SymbolTrie.module_hierarchy_edges()` (iterator yielding
  `(child_module, parent_module)` pairs). Callers append to their own
  edge accumulators.

- `Package.exported` now enters the per-package cache fingerprint, so
  editing the exported subdirs invalidates that package's cache (siblings
  are unaffected). The fingerprint is computed per-package via the new
  `package=` argument on `compute_fingerprint`. `Package.path` / `name` /
  `deps`, the resolver, and `search_paths` remain outside the fingerprint.
  Schema bumped to 4; existing caches will rebuild on first run.

- Two trie fields on `PackageContribution` collapse into one. Previously
  each package kept a `current_trie` (everything) plus an `export_trie`
  (only decls from files under `Package.exported`); they now share a
  single `trie`, with each entry's exported-ness carried by the new
  `NodeFlags.EXPORTED` flag. `_build_symbol_lookup` calls
  `SymbolTrie.merge` for the package's own trie and the new
  `SymbolTrie.merge_exported` for each dep's trie, which filters to
  EXPORTED-flagged entries while still walking through unexported
  intermediate modules so exported descendants stay reachable.
  `EXPORTED` is set via the visitor's `default_flags` mechanism (same
  pattern as `NOTEBOOK`).

- The "shadowed by sibling package" file-vs-package precedence case is
  now called **eclipsed** to disambiguate from `NodeFlags.SHADOWED`
  (intra-file decl rebinding, unchanged). The function is
  `eclipsed_paths`, the warning text says "eclipsed by sibling package",
  and `_apply_payload` takes `eclipsed: bool`. The standard linter
  meaning of `SHADOWED` is preserved.

### Refactored
- The per-package apply layer (`PackageContribution`, `build_contribution`,
  `_apply_payload`, `eclipsed_paths`) moved from `_refresh.py` into a new
  `_package.py` module. `_refresh.py` now hosts the per-file pipeline
  exclusively (enumerate, parse, observe, cache); the per-package step
  consumes those payloads. No public API change.

- `Analysis._materialize` renamed its `scope` parameter to `included` and
  dropped the `None`-means-everything case (`materialize_all` now passes
  the full package set explicitly). `Analysis._build_symbol_lookup` lost
  its `scope` parameter entirely — the `_interesting_set` is closed under
  transitive deps by construction, so the filter could never trigger.

### Removed
- The overlay / what-if API on `Analysis` is gone (breaking):
  `Analysis.preview_payloads`, `Analysis.materialize_with`,
  `Analysis.preview`, and the `GraphView` class (with its
  `dead_cst.analyze` re-export) have been deleted. The design
  didn't pay its keep — callers comparing baseline vs. perturbed
  reachability can construct a second `Analysis` with a substitute
  detector or modified sources and diff `dead()` directly.
  `TruthinessResolver.resolve_constant` and the
  `DefaultUnreachableRegionDetector.resolve(expr, resolver)` hook
  stay — they're independently useful for custom detectors.

### Changed
- `DispatchAppPlugin` (in `dead_cst.plugins.decl_shapes`) is now the
  shared base for `FlaskPlugin`, `FastAPIPlugin`, and `CeleryPlugin` in
  addition to `TyperPlugin` / `CycloptsPlugin`. The base learned an
  opt-in factory-aware mode driven by a new
  `instance_kinds: Mapping[str, bool]` field: when set, the plugin
  emits `<{name}-app>:` / `<{name}-pending>:` / `<{name}-factory>:`
  synthetics and runs a per-package finalize pass that walks pending
  variables forward to a discriminating import node or factory marker
  before promoting the matching kinds to entrypoints. When
  `instance_kinds` is empty (Typer / Cyclopts) the plugin behaves as
  before: pure observe, no entrypoint promotion. Flask / FastAPI /
  Celery now consist almost entirely of their handler / kind config,
  with Celery's `@shared_task` channel as its only override. No
  user-visible behavior change.

### Added
- `FastMCPPlugin` ships in `dead_cst.contrib.fastmcp` and is registered
  under the `fastmcp` builtin name. Marks top-level
  `X = FastMCP(...)` server instances as entrypoints (the `fastmcp` CLI
  loads `module:mcp` by import path the same way `uvicorn` loads a
  FastAPI `module:app`, so every FastMCP server is framework-visible
  the moment it's constructed) and wires `@mcp.tool` / `@mcp.resource`
  / `@mcp.prompt` / `@mcp.completion` decorators on top-level functions
  through the owning server. Supports the
  `def create_server() -> FastMCP: ...` factory shape across packages
  via the shared `DispatchAppPlugin` factory-marker mechanism. Only the
  `fastmcp` import path is recognized; the Anthropic MCP SDK's
  compatibility re-export (`mcp.server.fastmcp.FastMCP`) is not
  detected -- users on that path can keep their handlers alive with
  explicit `-e` entrypoints or a project-local plugin.
- Jupyter notebook (`.ipynb`) support. Every `.ipynb` file under a
  package root is ingested by concatenating its code cells into a
  single libcst-parseable module; IPython line magics (`%foo`),
  cell magics (`%%bash`), shell escapes (`!ls`), and trailing-help
  forms (`obj?`, `obj.attr??`) are neutralized to `pass  # <line>`
  so libcst accepts the source. Notebooks aren't importable, so their
  decls are deliberately kept out of the cross-module lookup trie.
- `NodeFlags.NOTEBOOK` stamps every `SymbolNode` sourced from a
  notebook. `SymbolVisitor` now takes a `default_flags` kwarg that
  `_refresh._process_one_file` sets to `NOTEBOOK | ENTRYPOINT` for
  notebooks, so a notebook's contents are reachability seeds and any
  `.py` code the notebook imports stays alive. Malformed notebooks
  fall through to the same `[unparseable] <module>` placeholder used
  for `.py` files that fail to parse.
- The codemod (`generate_patch` / `remove_code`) skips
  `NodeFlags.NOTEBOOK` nodes; cell-aware writeback into the notebook
  JSON envelope is out of scope.
- `CeleryPlugin` ships in `dead_cst.contrib.celery` and is registered
  under the `celery` builtin name. Marks top-level `X = Celery(...)`
  app instances as entrypoints (the Celery worker process loads
  `module:app` by import path via `celery -A`, mirroring how
  `FastAPIPlugin` / `FlaskPlugin` treat their app instances), wires
  `@app.task` / `@app.task(...)` decorators on top-level functions
  through the owning app (so a task callable lives as long as the app
  does), supports the `def make_celery(): return Celery(...)` factory
  shape across packages via a factory marker, and seeds module-level
  `@shared_task` / `@shared_task(...)` decorated functions (with
  `shared_task` imported from `celery`) as entrypoints directly --
  `shared_task` registers into Celery's global registry and is invoked
  by name with no owning app variable to wire through.

## [0.9.4] - 2026-05-12

### Changed
- `PluginContext` now requires `package_graph` (the per-package
  contribution graph) and `module_nodes` (its module-typed entries).
  `package_nodes` and `package_modules` source from these directly,
  dropping the `Path.is_relative_to` filter and `type == "module"`
  scan they used to run on the merged cross-package `graph`. The
  analyzer wires both through automatically; custom callers that
  construct `PluginContext` directly must pass them.
- `PackageView.modules()` now reads `PackageContribution.module_nodes`
  instead of refiltering the contribution graph.

## [0.9.3] - 2026-05-12

### Added
- `ServerConfigPlugin` ships in `dead_cst.contrib.server_config` and
  is registered under the `server_config` builtin name. Matches
  Gunicorn / Hypercorn config files by basename (`gunicorn.conf.py`,
  `gunicorn_conf.py`, `hypercorn.conf.py`, `hypercorn_conf.py` by
  default; override the `filenames` tuple for non-standard layouts)
  and marks the module plus every top-level decl (functions for hook
  callbacks like `on_starting` / `post_fork` / `when_ready`,
  variables for settings like `bind` / `workers`, classes for inline
  custom logger / worker definitions, imports for helpers used only
  to build config values) as an entrypoint. These files are loaded
  by the server process at startup (Docker, Cloud Run, systemd) and
  not imported anywhere in the project, so without this plugin their
  whole surface looks dead.
- `find_factory_decls(module, imports, valid_targets)` is exported from
  `dead_cst.plugins`. Third-party framework plugins that follow the
  same instance-construction shape can use it together with
  `walk_to_instance_kind(..., factory_marker_prefix=...)` to get
  cross-package factory support for free.

### Fixed
- `FastAPIPlugin` and `FlaskPlugin` now classify the factory pattern
  across packages when the factory uses `import fastapi; fastapi.FastAPI()`
  / `import flask; flask.Flask()` (module-prefixed form). The
  external-edge classifier drops the `decl=` half of the access, so the
  downstream walk had no discriminator to tell `FastAPI` from
  `APIRouter` (or `Flask` from `Blueprint`). `observe` now tags every
  top-level decl whose body constructs one of those classes with a
  `<{fastapi,flask}-factory>:<kind>:<owner>` synthetic, and
  `walk_to_instance_kind` accepts a `factory_marker_prefix=` kwarg so
  the per-package finalize walk picks the marker up regardless of which
  file the factory lives in. The named-import shape
  (`from fastapi import FastAPI`) was already covered by the
  import-node discriminator and is unaffected.
- Attribute access on a runtime module dunder (`some_pkg.__file__`,
  `some_pkg.__name__`, `some_pkg.__spec__`, etc.) no longer surfaces a
  "Failed to resolve import edge" warning. The import machinery injects
  these attributes on every module object at runtime, so the chain past
  them is a path / string op, not a symbol reference. The visitor now
  truncates the access chain at the dunder and emits a clean
  `Import(module=X, decl=None)` instead of a speculative
  `Import(module=X, decl="__file__")`. Reachability is unchanged -- the
  module-level edge that previously rode alongside the failed lookup is
  the same edge that's now emitted directly. Recognised dunders:
  `__file__`, `__name__`, `__doc__`, `__loader__`, `__spec__`,
  `__package__`, `__path__`, `__builtins__`, `__cached__`. Visitor
  `version` is bumped so cached payloads rebuild.

## [0.9.2] - 2026-05-12

### Added
- `DiscordPyPlugin` ships in `dead_cst.contrib.discordpy` and is
  registered under the `discordpy` builtin name. The plugin recognizes
  top-level `commands.Bot` / `discord.Client` (and the `AutoSharded*`
  variants) constructions and seeds them as entrypoints, wires
  `@bot.command()` / `@bot.event` / `@bot.listen()` / `@bot.tree.command()`
  / `@bot.tree.context_menu()` decorators (and their group / hybrid /
  invoke-hook siblings) to their bot variable, marks any module that
  defines a `commands.Cog` subclass as alive together with its module-
  level `setup` / `teardown` hooks, and resolves
  `<expr>.load_extension("dotted.path")` / `load_extensions([...])`
  string-literal targets onto the captured module's surface
  (matching `importlib.import_module` semantics).

### Changed
- `distribution_lookup` and `editable_distribution_roots` are now keyed
  on the dist-bearing slice of `sys.path` (site-packages /
  dist-packages / purelib / platlib entries) instead of an empty
  tuple, so they survive the analyzer's per-package `sys.path` rebind
  for free -- only the first-party prefix moves during a transition,
  and that prefix never enters the key. `Analysis._materialize` now
  uses the narrower `clear_module_specs_cache()` (also newly exported
  from `dead_cst.resolvers`) on every package transition instead of
  the full `clear_path_caches()`, dropping a ~10s/package
  `importlib.metadata` walk that dominated large-workspace runs. A
  real venv change (uv splicing in a workspace `.venv`) still flips
  the key automatically and triggers a single rebuild. Pure
  performance change.
- `_count_nodes_by_prefix` batches the per-package node-counting that
  the CLI text/JSON report does. Previously
  `_output_text` / `_output_json` called `_count_nodes(graph, prefix)`
  twice per package, each walking the entire graph; on large
  workspaces this dominated report formatting. The bucketed helper
  walks the graph once and aggregates by prefix in a second pass keyed
  on unique file paths.
- `is_from_module` (exported from `dead_cst.plugins`) now recognizes
  dotted module names — `is_from_module(node, "discord.ext.commands")`
  matches `from discord.ext.commands import ...`. Previously only
  single-segment module names worked, because the helper bottomed out
  in `is_name` (bare `cst.Name` only). Backward-compatible: every
  existing single-segment caller still matches. `collect_module_imports`
  inherits the change, so plugins can now scan dotted source modules
  without rolling their own import-walker.
- `SymbolVisitor` now hoists the `_descendant_ids` cache used by
  `live_referents` / `live_at_exit` onto the visitor instance, so a
  single shared cache covers every flow-analysis call the visitor
  makes for a file. Previously each multi-referent access in
  `on_leave` triggered a fresh cache allocation, so large files with
  many reassignments re-walked the same statement subtrees from
  scratch on every access. Pure performance change — output and
  payload-cache fingerprint are unchanged.

### Fixed
- A `foo.py` sibling of a `foo/__init__.py` package no longer asserts
  out of `SymbolTrie.add_declaration`. The new
  `dead_cst._refresh.shadowed_paths` pre-pass mirrors CPython's
  `FileFinder` precedence (regular package wins over a same-named
  module file), so the trie holds the package and cross-module imports
  of `pkg.foo` route there. The shadowed `.py` is still parsed and its
  nodes still appear in the package graph -- observe-time entrypoints
  (`__main__` blocks, plugin synthetics) keep working -- but consumer
  imports never see its decls. A WARNING is logged per shadowed file
  so the layout (almost always a bug) surfaces during analysis.

## [0.9.1] - 2026-05-11

### Fixed
- `resolve_edges` no longer spins forever on cyclic re-exports. The
  worklist DFS now carries a per-walk `visited` set keyed on
  `(id(SymbolTrie), parts_tail)`, so a pathological pair like
  `A.x: from B import x` / `B.x: from A import x` terminates after one
  trip around the cycle instead of repeatedly chaining back to its
  starting state. The decls actually encountered along the cycle are
  still emitted, so first-party reachability through the chain is
  preserved.

### Changed
- `resolve_edges` now memoizes the full per-import resolution at three
  layers: ``_resolve_targets`` keyed by ``Import`` value (so equal
  spellings across files share the precomputed dst list — the visitor
  builds fresh ``Import`` objects per file, but they hash equal because
  ``Import`` is frozen with an eager ``__hash__``); ``_walk`` keyed by
  ``(start_node, decl_parts)`` (so different ``Import`` shapes that
  canonicalize to the same trie state share the re-export DFS); and
  ``_classify`` keyed by ``(import.module, import.speculative)`` (so
  the resolver runs once per unique external name). The per-src loop
  collapses to ``for dst in cached_targets: emit(...)``. On large
  multi-package workspaces where one big package's edge contribution
  dominates composition (the case `compute_fingerprint` / hash
  precomputation helped on but did not change the algorithmic shape
  of), this turns the per-package compose loop's growth in importer
  count from multiplicative to additive.

## [0.9.0] - 2026-05-11

### Changed
- The parallel refresh pool (``--workers >= 2``) now drains worker
  results via ``concurrent.futures.as_completed`` instead of
  ``pool.map``. Cache writes and progress ticks land in completion
  order, so a single slow file no longer blocks the cache from warming
  with the fast files behind it. Tasks are still submitted in
  ``(package_path, file)`` order so same-package work stays contiguous.
- Refresh now collects per-task failures and raises a single
  ``ExceptionGroup`` after every other task has finished, instead of
  aborting on the first error. Successfully-parsed files are still
  cache-warmed before the group is raised, so a re-run after fixing
  the bad file only re-parses what failed.
- The pool branch installs SIGTERM / SIGINT handlers for the lifetime
  of the run; on signal it cancels every still-pending future, calls
  ``pool.shutdown(wait=False, cancel_futures=True)``, restores the
  prior handlers, and raises ``KeyboardInterrupt``. Files that
  completed before the signal stay cache-warmed.
- `SymbolNode` and `Import` now pre-compute their hash in
  `__post_init__` and store it in a private `_hash` slot; `__hash__`
  becomes a single attribute read. The instances are frozen so the
  result is stable. Cuts edge-stitching time on large multi-package
  workspaces where `resolve_edges._emit` re-hashes the same
  `(src, dst, flags)` tuples into its dedup set, and pays off again
  every time a `SymbolNode` is hashed by networkx (graph insertion,
  BFS traversal). `SCHEMA_VERSION` is bumped to 3 so cache rows
  pickled before the slot existed are invalidated on first use.
- Progress reporting is fully logger-driven and controlled by the root
  logger level. Per-file refresh status ``[i/N] ok|FAILED <file>`` goes
  through ``logger.debug`` on ``dead_cst._refresh``; off-TTY decile
  checkpoints go through ``logger.info`` on ``dead_cst._progress``.
  The on-TTY tqdm bar is preserved and wraps its iteration in
  ``logging_redirect_tqdm`` so concurrent log records print above the
  bar without shattering it. ``dead-cst -v`` keeps its meaning (the
  CLI's ``setup_logging`` flips the root level to ``DEBUG``); library
  users get the same firehose by configuring their root logger.

### Fixed
- Stdlib imports no longer emit a spurious ``Failed to resolve import
  module: <name>`` warning during edge stitching. The orphaned warning
  in ``_edges._emit_external`` fired for every successfully-classified
  stdlib import (``import datetime`` / ``from pathlib import Path`` / …)
  because the silent-drop and speculative-miss branches both returned
  ``None``; the warning is removed since no legitimate path reached it
  -- truly-unresolved non-speculative imports already surface as
  ``[unresolved] <top-level>`` synthetic nodes.
- ``default_resolve_import`` now falls back to the parent module when a
  dotted name can't be resolved directly, so ``collections.abc``,
  ``importlib.resources.abc``, and similar synthesized-in-``__init__``
  submodules classify as ``[stdlib] <name>`` instead of being misfiled
  as ``[unresolved] <top>``.

## [0.8.0] - 2026-05-09

### Added
- The visitor now honors ruff/pyflakes ``# noqa`` directives that
  silence F401 (unused-import). A per-line ``# noqa``,
  ``# noqa: F401``, multi-rule ``# noqa: E501, F401``, or case-variant
  ``# NOQA`` on the same source line as an import alias pins the
  resulting import node alive (``NodeFlags.ENTRYPOINT |
  NodeFlags.NOQA``) so it is no longer reported as dead. File-level
  directives -- ``# ruff: noqa``, ``# ruff: noqa: F401``, and the
  ``# flake8: noqa`` aliases -- pin every import in the file. The
  ``ruff:`` / ``flake8:`` prefix is matched case-sensitively per
  ruff's documented behavior; the ``noqa`` keyword is case-insensitive.
  Per-alias directives inside a parenthesized ``from x import (a, b)``
  pin only the alias on that line. This brings dead-cst's unused-import
  semantics in line with ruff: an import you have explicitly marked as
  intentionally preserved (re-exports, side-effect imports,
  ``TYPE_CHECKING`` shims guarded by F401) is no longer surfaced or
  removed.
- New ``NodeFlags.NOQA`` flag, layered on ``NodeFlags.ENTRYPOINT``
  (parallel to ``NodeFlags.TESTCASE``). Read ``n.flags & NodeFlags.NOQA``
  off the ``SymbolNode`` directly; there is no graph attr-dict mirror.
- `dead_cst.codemod.generate_patch(G, root)` returns the same removal
  as `remove_code` as a `git apply`-compatible unified diff (with
  `diff --git` headers and `deleted file mode 100644` for module
  deletions) instead of writing in place. Selection is driven entirely
  by `G.nodes`, so callers can slice the unreachable graph (e.g.
  `G.subgraph(scc)` for one SCC at a time) to review a big codebase as
  a series of focused patches.
- `TruthinessResolver.resolve_constant(expr) -> Const | None` returns
  the underlying literal *value* (`str` / `int` / `bool` / `None`,
  wrapped in the new `Const` dataclass) over the same flow-sensitive
  Name walk `evaluate` uses for truthiness. Intended for custom
  detectors that pattern-match against a flag *name* -- e.g.
  `check_flag(FEATURE_A)` where `FEATURE_A = "feature_a"` is bound at
  module level. `Const(None)` (the `None` literal was proved) stays
  distinct from a bare `None` return ("unknown"). Re-exported from
  `dead_cst.branches`.
- `Analysis.preview_payloads(files, *, detector=None) -> dict[Path, VisitorPayload]`
  regenerates per-file payloads for a hand-picked file set, bypassing
  the on-disk cache (no read, no write) and accepting a one-shot
  `UnreachableRegionDetector` override. Pairs with the new
  `Analysis.materialize_with(payloads)` to splice the substitute
  payloads into a fresh graph without mutating any baseline state.
  `Analysis.preview(files, *, detector=None)` chains the two and
  returns a new `GraphView` (also new) wrapping the overlay graph
  with the same reachability surface as `Analysis` (`reachable`,
  `dead`, `kept_alive_by_dead_branches`, `kept_alive_by_flags_only`,
  `count_nodes`). The combined API enables what-if graph surgery:
  bake a flag's truthiness into a small set of files and ask "what
  becomes dead if we land this rollout?" without forking the
  analysis-wide fingerprint or polluting the cache.

### Changed
- **Breaking:** The two ``kept_alive_by_*_only`` methods on
  ``Analysis`` and ``PackageView`` have been collapsed into a single
  ``kept_alive_by_flags_only(flags: NodeFlags)``. Pass
  ``NodeFlags.TESTCASE`` for the old ``kept_alive_by_tests_only``
  behavior ("blast radius of dropping the test suite"),
  ``NodeFlags.NOQA`` for "blast radius of removing every F401 pin",
  or both ORed together. ``dead_cst.analyze._find_reachable(graph,
  exclude_flags=NodeFlags.NONE)`` is the matching private helper
  shape; ``_find_reachable_excluding_tests`` and
  ``_find_kept_alive_by_tests_only`` are gone.
- **Breaking:** `dead-cst remove` no longer modifies files in place. It
  now emits a unified diff to stdout (or to `--output PATH`) and exits;
  apply with `dead-cst remove . | git apply` (or `git apply <path>` if
  you used `--output`). The `--dry-run` flag and the
  `Proceed with removal? [y/N]` confirmation prompt are gone -- the
  command is non-destructive by construction. The "Building symbol
  graph" banner, the "No dead code found." message, and the apply hint
  go to stderr so the patch on stdout stays clean.
- **Breaking:** `DefaultUnreachableRegionDetector.resolve(expr)` is now
  `resolve(expr, resolver)`. The active `TruthinessResolver` is the
  second argument, so overrides can call
  `resolver.resolve_constant(...)` to fold a flag-name `Name` to its
  literal value before pattern-matching. Subclasses with a one-arg
  `def resolve(self, expr): ...` need to add the `resolver` parameter;
  ignore it if you don't use it.

### Fixed
- `DefaultUnreachableRegionDetector` now recognizes compound statements
  as terminators when every reachable branch itself terminates. An
  ``if`` whose taken branch always ``return``s (e.g.
  ``if True: return``, ``if FLAG: return`` with ``FLAG = True``, or
  ``if cond: return; else: return``) kills statements that follow it
  in the enclosing suite, so constant-folded early returns now flag
  trailing dead code as expected. Same handling applies to ``with``
  whose body terminates, and to ``try``/``except``/``finally`` where
  every path terminates (or a ``finally`` itself terminates).
- `TruthinessResolver` no longer folds a ``Name`` whose binding's RHS
  is a mutable container literal (``[]``, ``{}``, ``set()``-shaped
  comprehensions, etc.). The binding-only flow walk is invisible to
  ``.append`` / item assignment / ``.update`` mutations, so an
  ``edges = []; edges.append(x); if not edges:`` chain used to fold
  ``edges`` to ``False`` and incorrectly mark the trailing code dead.
  Tuples and primitive literals are immutable and stay safe to fold.
  Bumps the detector's `version` so cached payloads rebuild.

## [0.7.0] - 2026-05-09

### Added
- New `[unparseable] <module>` synthetic node, exposed via the
  `UNPARSEABLE_PREFIX` constant re-exported from `dead_cst.plugins`.
  When `libcst` cannot parse a file, the analyser now logs a warning
  and emits a minimal payload pairing the real module node with one of
  these synthetics flagged `ENTRYPOINT`, instead of aborting the whole
  run with `ParserSyntaxError`. The file stays alive in reachability
  and importers can still target the module; decls inside the file
  remain invisible until parsing succeeds. The placeholder rides the
  per-file cache, so a fresh source SHA (i.e. the user fixing the
  syntax) invalidates the entry automatically.

### Changed
- **Breaking:** `PathResolver` no longer satisfies the `Cacheable`
  protocol -- the `name` / `version` attributes have been removed from
  the protocol and from the shipped `ManualResolver` and `UvResolver`.
  Resolver output flows through the (uncached) edge-stitching pass, so
  swapping a resolver re-stitches edges without invalidating any
  cached `VisitorPayload` blob; there was nothing for the
  `(name, version)` pair to gate. Out-of-tree resolvers should drop
  their `name` and `version` fields. `BUILTIN_RESOLVERS` now keys the
  builtin entry by the literal string `"uv"` rather than reading
  `UvResolver.name`.
- **Breaking:** `dead_cst.branches.fold_constants` is gone, replaced
  by `dead_cst.branches.TruthinessResolver`, a goal-directed
  flow-sensitive truthiness object. From-scratch
  `UnreachableRegionDetector` implementations should construct one
  resolver per file and pass `resolver.evaluate` as the `resolve_expr`
  argument to `unreachable_suites` / `evaluate_truthiness`.
  `DefaultUnreachableRegionDetector`'s `resolve(self, expr)` subclass
  hook is unchanged. The internal `_const_fold` module is gone.
- `DefaultUnreachableRegionDetector.find_regions` is now a two-pass
  design: one CST visit collects every `If` / `While` / suite-bearing
  site, and the resolver answers truthiness queries on demand. Files
  with no conditional tests now skip `ScopeProvider` /
  `ParentNodeProvider` resolution entirely. Self-analysis benchmark
  drops `find_regions` from 24.2 s to 1.8 s over `dead_cst/` (13×).
- Bumped the `libcst` floor to `>=1.8.6` so PEP 750 template strings
  (`t"..."`, Python 3.14+) parse rather than aborting with
  `ParserSyntaxError`. Names referenced inside t-string interpolations
  and format specs flow through the visitor's existing scope
  resolution -- a `t"hello {NAME}"` inside a function now produces
  the same `func -> NAME` edge an f-string would, and the
  documented limitation has been removed.

### Fixed
- `enumerate_files` now skips directories whose name happens to end in
  `.py` / `.pyi`. `Path.rglob` matches by name only, so an oddly-named
  directory (e.g. `something.py/`) used to slip through and crash the
  visitor on `read_text`.
- Fixed an `AttributeError` crash in the visitor when a lambda body
  contained an access with multiple potential referents (e.g. a
  walrus that rebinds the lambda's own parameter). The flow-analysis
  helper now skips lambdas, whose body is a single expression rather
  than an `IndentedBlock`.

## [0.6.0] - 2026-05-08

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
  type stub (`mypkg/_native.pyi`) with no `.py` twin. The stub is
  parsed through the same visitor as a regular module under its
  natural FQN (`mypkg._native`), so `from mypkg._native import
  compute` resolves to the stub's `compute` decl through the normal
  cross-module import path, and reachability + the codemod work the
  same as for any first-party module. A `.pyi` shipped alongside a
  real `.py` is dropped during file enumeration -- the runtime
  module is the canonical declaration of those names, and ingesting
  the stub on top would collide with it in the symbol trie. Peer-mode
  stubs are therefore invisible to dead-cst.
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
- New `NodeFlags.TESTCASE` flag tags entrypoints created by test
  plugins (pytest / unittest discovery, fixture seeds). The default
  `Analysis.reachable()` traversal still treats those seeds as
  ordinary entrypoints. The opt-in `Analysis.kept_alive_by_tests_only()`
  / `PackageView.kept_alive_by_tests_only()` returns the "blast
  radius" of dropping the test suite -- production code currently
  kept alive only because tests still touch it. `PytestPlugin` and
  `UnittestPlugin` stamp `ENTRYPOINT | TESTCASE` on every synthetic
  seed they create; their `version` epochs were bumped accordingly.
  `AddNode` plugin ops grew an analogous `testcase: bool = False`
  field for plugins that emit entrypoints from `finalize`.
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
- `UnittestPlugin` now resolves transitive `TestCase` subclasses.
  Refactored into a two-phase plugin: `observe` emits
  `<unittest:base-of>:<base_fqname>` bucket markers for every class
  def, and `finalize` walks the bucket chain from `unittest.TestCase`
  / `IsolatedAsyncioTestCase` (plus every import alias of those) to
  find the transitive subclass closure. This closes two real-world
  gaps the previous single-pass implementation missed: test classes
  that inherit from a project-local `TestCase` mixin (`class
  ProjectTC(unittest.TestCase); class MyTest(ProjectTC)`), and test
  classes that inherit from a re-exported `TestCase` (`from
  pkg.bases import TestCase` where `pkg.bases` does `from unittest
  import TestCase`). Alias expansion uses each import node's raw
  `Import` metadata rather than graph successors, so it works
  across the stdlib boundary that `unittest.TestCase` sits on. The
  plugin's `version` was bumped, invalidating cached payloads.
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

[Unreleased]: https://github.com/lpetre/dead-cst/compare/v0.9.3...HEAD
[0.9.3]: https://github.com/lpetre/dead-cst/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/lpetre/dead-cst/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/lpetre/dead-cst/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/lpetre/dead-cst/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/lpetre/dead-cst/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/lpetre/dead-cst/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/lpetre/dead-cst/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/lpetre/dead-cst/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/lpetre/dead-cst/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/lpetre/dead-cst/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lpetre/dead-cst/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lpetre/dead-cst/releases/tag/v0.1.0
