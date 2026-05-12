# dead-cst Roadmap

A stack-ranked plan for moving `dead-cst` from alpha to a tool maintainers
trust enough to wire into CI. Ordering reflects leverage, not effort.

The central question driving this ordering: **what makes a maintainer trust a
dead-code tool enough to run it on their codebase?** The architecture is in
good shape; the gap to wide adoption is (a) trust that it won't flag legitimate
code and (b) trust that the codemod won't break files. Tier 1 buys both.

Items marked _shipped_ are kept here briefly for context — see `CHANGELOG.md`
for the full record.

---

## Tier 1 — Trust and correctness

### 1. Finish the framework-aware plugin presets

`PytestPlugin`, `UnittestPlugin` (with transitive `TestCase` discovery),
`FastAPIPlugin`, `FlaskPlugin`, `TyperPlugin`, `ClickPlugin`,
`CycloptsPlugin`, `DiscordPyPlugin`, `MockPatchPlugin`,
`ServerConfigPlugin` (gunicorn / hypercorn config files), and
`InitSubclassPlugin` shipped, but the existential risk is still
"I tried it and it flagged half my codebase." The remaining common
offenders:

- Django URLConf, admin registration, signal handlers, management commands
- Pydantic validators and field serializers
- Descriptor-style hooks: `__set_name__`, dataclass `__post_init__`
- SQLAlchemy declarative models / event listeners; Celery tasks / signals

Surface a `--preset pytest,fastapi,django` shortcut that expands to the
existing `--plugin` wiring, and document the entry-point group so third
parties can publish their own.

### 2. Function-call folding (same-file, caller-capped)

`def is_new_auth(): return False; if is_new_auth(): X = 1` should mark `X` as
dead, but doesn't yet — `evaluate_truthiness` doesn't fold through calls. Have
`TruthinessResolver` recognize trivial single-return `FunctionDef` shapes and
fold a bare `Call` to one of them by recursing into the function's return
expression. Both sync `def` (folded via bare `Call`) and async `def` (folded
via `Await(Call)` only — a bare async call returns a coroutine, always
truthy). Cap by caller count (default 3) so that configuration helpers used
in 50 places don't get folded into noise. Same-file only; cross-file is
item 8.

### 3. Suite-removal codemod and expanded dead-set helper

`dead-cst remove` only deletes decls today; the `If` / `While` / post-
terminator suites that `analyze` flags can't be removed automatically. Add a
`RemoveDeadSuites` LibCST transformer covering each parent shape (`If` /
`While` / `elif` chains / post-terminator) with empty-suite guards inserting
`pass` where needed, and a helper that returns the union of dead decls + dead
suite ranges + `Analysis.kept_alive_by_dead_branches()` so removal expands
to the blast radius. Foundational for items 5 and 7 — invisible by itself,
shipped together with whichever consumer lands first.

---

## Tier 2 — Adoption surface

### 4. "New dead code only" / diff mode

Most teams cannot fix all existing dead code in one PR. A `--since <ref>` flag
that filters to symbols introduced or last touched after a git ref makes the
tool drop-in for CI on existing codebases without a big-bang cleanup. This
pattern is what drove adoption for tools like Knip in the JS ecosystem.

### 5. `remove --include-dead-branches`

Wire item 3's suite-removal codemod into `dead-cst remove` behind an opt-in
flag. Default off — existing decl-only behavior keeps backward-compatibility
while the new transformer bakes. Once warm in real-world use, flip the
default to on and rename the opt-out to `--no-dead-branches`.

### 6. `remove --inline-folds`

Wire item 2's function-call folding into `dead-cst remove` behind an opt-in
flag, mirroring item 5's shape for suites. For every trivial-return function
the fold pass classifies as a constant, rewrite each call site with that
constant and (when the function has no surviving consumers) delete the
function itself. Default off; flip to default-on once warm in real-world
use. Composes with item 5: once a fold is inlined, `if is_new_auth():`
becomes `if False:`, which the suite-removal pass then collapses in the
same run.

### 7. `dead-cst preview` (graph-clustered patches)

`dead-cst remove` already emits a `git apply`-compatible unified diff
(see "Recently shipped"), and `dead_cst.codemod.generate_patch(G, root)`
slices on whatever subgraph you hand it — so per-cluster patches are
already a one-liner from the Python API. The remaining work for the
peer subcommand is automatic clustering (weakly-connected components of
the unreachable subgraph, each the maximal unit applicable atomically
without leaving dangling references), cluster headers carrying
blast-radius metadata (symbols, LOC, files affected), and folding in
item 3's suite-removal output once that lands. Composes with `git apply`
(whole cluster) and `git add -p` (per-line review) without needing a
custom TUI; see item 11 for the deferred TUI shape.

### 8. Cross-file trivial-return folding

Item 2 only folds within a single file. Cross-module folding
(`from flags import is_new_auth; if is_new_auth():` resolving via the import)
requires plumbing fold state through `resolve_edges` and bumping the cache
contract. Defer until item 2 ships and demand is real.

---

## Tier 3 — Polish and ecosystem

### 9. Read the Docs site with plugin/resolver tutorials

The `EdgePlugin` and `PathResolver` protocols are well-designed but
undiscovered. A short Sphinx site with one tutorial each ("write a custom
`EdgePlugin`", "write a custom `PathResolver`") activates the extensibility
that's already built. The docstring pass already in place gives the API
reference for free.

### 10. `examples/flag_audit/` recipe

A working example, not a CLI command. Ships a `flags.toml` mapping flag-name
→ fixed truthiness, a `FlagAuditDetector(DefaultUnreachableRegionDetector)`
whose `resolve()` answers `check_flag("foo")` calls per the toml config, and
a `main()` that builds the analysis, calls `Analysis.kept_alive_by_dead_branches()`,
and prints "removing flag X would delete N symbols / M LOC". Item 2 makes
this much cleaner: a one-line wrapper `def is_new_auth(): return
check_flag("new-auth")` folds without the detector having to recognize the
wrapper directly.

### 11. Interactive TUI (`dead-cst review`)

Speculative; only promote if `git add -p` over item 7's output proves
insufficient. Design direction when promoted: walk the cluster condensation
in topological order, asking accept/reject for each cluster, stopping on the
first rejection (or letting the user mark "skip" to keep going). Equivalent
to a DAG-walk where rejecting a parent means we don't bother asking about
its successors. Pairs directly with item 7's WCC clustering — the data
shape carries over.

---

## Tier 4 — Speculative, wait for signal

### 12. Multiple reachability frontiers

Splitting "reachable from tests" vs. "reachable from production entrypoints"
is interesting and would enable rules like "no production code reachable only
from tests." But it's solving a problem nobody has reported yet. Wait for
demand.

---

## Recently shipped

Folded down from earlier tiers as they landed:

- **v0.9.4**: ``PluginContext`` now requires a ``package_graph`` field
  (the per-package contribution graph, passed by the analyzer);
  ``package_nodes`` and ``package_modules`` snapshot from it directly
  instead of filtering the merged cross-package ``graph`` by
  ``Path.is_relative_to``. The filter was an O(N_total) walk that
  dominated the first ``package_nodes`` call in every finalize pass
  (~9 ms → ~40 µs on the dead-cst self-analysis). Plugin behavior is
  unchanged; the helpers expose the same node sets they always did.
  Custom callers that construct ``PluginContext`` directly (tests,
  out-of-tree pipelines) must pass ``package_graph=`` -- a hard break
  rather than an optional shim, matching the pre-1.0 API churn budget.
- **v0.9.3**: ``ServerConfigPlugin`` (``dead_cst.contrib.server_config``,
  registered as the ``server_config`` builtin) marks Gunicorn / Hypercorn
  config files (``gunicorn.conf.py``, ``gunicorn_conf.py``,
  ``hypercorn.conf.py``, ``hypercorn_conf.py`` by default; override the
  ``filenames`` tuple for non-standard layouts) and every top-level decl
  inside them (hook callbacks like ``on_starting`` / ``post_fork`` /
  ``when_ready``, settings like ``bind`` / ``workers``, inline custom
  logger / worker classes, helper imports) as entrypoints. These files
  are loaded by the server process at startup (Docker, Cloud Run,
  systemd) and not imported anywhere in the project, so without this
  plugin their whole surface looked dead. Closes the
  ServerConfigPlugin checkbox in Tier 1 #1.
- **v0.9.3**: ``FastAPIPlugin`` and ``FlaskPlugin`` now classify the
  factory pattern across packages when the factory uses the
  module-prefixed form (``import fastapi; fastapi.FastAPI()`` /
  ``import flask; flask.Flask()``). The external-edge classifier
  drops the ``decl=`` half of the access, so the downstream walk had
  no discriminator to tell ``FastAPI`` from ``APIRouter`` (or ``Flask``
  from ``Blueprint``). ``observe`` now tags every top-level decl whose
  body constructs one of those classes with a
  ``<{fastapi,flask}-factory>:<kind>:<owner>`` synthetic, and
  ``walk_to_instance_kind`` accepts a ``factory_marker_prefix=`` kwarg
  so the per-package finalize walk picks the marker up regardless of
  which file the factory lives in. ``find_factory_decls`` is exported
  from ``dead_cst.plugins`` so third-party framework plugins with the
  same instance-construction shape get cross-package factory support
  for free. The named-import shape (``from fastapi import FastAPI``)
  was already covered by the import-node discriminator and is
  unaffected.
- **v0.9.3**: Attribute access on a runtime module dunder
  (``some_pkg.__file__``, ``some_pkg.__name__``, ``some_pkg.__spec__``,
  etc.) no longer surfaces a "Failed to resolve import edge" warning.
  The import machinery injects these attributes on every module object
  at runtime, so the chain past them is a path / string op, not a
  symbol reference. The visitor now truncates the access chain at the
  dunder and emits a clean ``Import(module=X, decl=None)`` instead of a
  speculative ``Import(module=X, decl="__file__")``. Reachability is
  unchanged. Recognised dunders: ``__file__``, ``__name__``, ``__doc__``,
  ``__loader__``, ``__spec__``, ``__package__``, ``__path__``,
  ``__builtins__``, ``__cached__``. Visitor ``version`` bumped so cached
  payloads rebuild.
- **v0.9.2**: ``DiscordPyPlugin`` (``dead_cst.contrib.discordpy``,
  registered as the ``discordpy`` builtin) recognizes top-level
  ``commands.Bot`` / ``discord.Client`` (and the ``AutoSharded*``
  variants) constructions and seeds them as entrypoints, wires
  ``@bot.command()`` / ``@bot.event`` / ``@bot.listen()`` /
  ``@bot.tree.command()`` / ``@bot.tree.context_menu()`` decorators
  (and their group / hybrid / invoke-hook siblings) to their bot
  variable, marks any module that defines a ``commands.Cog`` subclass
  as alive together with its module-level ``setup`` / ``teardown``
  hooks, and resolves ``<expr>.load_extension("dotted.path")`` /
  ``load_extensions([...])`` string-literal targets onto the captured
  module's surface (matching ``importlib.import_module`` semantics).
  Closes the DiscordPyPlugin checkbox in Tier 1 #1.
- **v0.9.2**: ``distribution_lookup`` and ``editable_distribution_roots``
  are now keyed on the dist-bearing slice of ``sys.path``
  (site-packages / dist-packages / purelib / platlib entries) instead
  of an empty tuple, so they survive the analyzer's per-package
  ``sys.path`` rebind for free — only the first-party prefix moves
  during a transition, and that prefix never enters the key.
  ``Analysis._materialize`` now uses the narrower
  ``clear_module_specs_cache()`` (newly exported from
  ``dead_cst.resolvers``) on every package transition instead of the
  full ``clear_path_caches()``, dropping a ~10 s/package
  ``importlib.metadata`` walk that dominated large-workspace runs
  (a 110-package analysis was spending ~19 minutes inside the inner
  dist scan alone). A real venv change (uv splicing in a workspace
  ``.venv``) still flips the key automatically and triggers a single
  rebuild. The CLI text/JSON report also gained a
  ``_count_nodes_by_prefix`` batching helper that walks the graph
  once per report instead of twice per package.
- **v0.9.2**: ``SymbolVisitor`` hoists the ``_descendant_ids`` cache
  used by ``live_referents`` / ``live_at_exit`` onto the visitor
  instance, so a single shared cache covers every flow-analysis call
  the visitor makes for a file. Previously each multi-referent access
  in ``on_leave`` triggered a fresh cache allocation, so large files
  with many reassignments re-walked the same statement subtrees from
  scratch on every access. Pure performance change — output and
  payload-cache fingerprint are unchanged.
- **v0.9.2**: A ``foo.py`` sibling of a ``foo/__init__.py`` package no
  longer asserts out of ``SymbolTrie.add_declaration``. The new
  ``dead_cst._refresh.shadowed_paths`` pre-pass mirrors CPython's
  ``FileFinder`` precedence (regular package wins over a same-named
  module file), so the trie holds the package and cross-module imports
  of ``pkg.foo`` route there. The shadowed ``.py`` is still parsed and
  its nodes still appear in the package graph — observe-time
  entrypoints (``__main__`` blocks, plugin synthetics) keep working —
  but consumer imports never see its decls. A WARNING is logged per
  shadowed file so the layout (almost always a bug) surfaces during
  analysis.
- **v0.9.1**: ``resolve_edges`` no longer spins forever on cyclic
  re-exports. The worklist DFS now carries a per-walk ``visited`` set
  keyed on ``(id(SymbolTrie), parts_tail)``, so a pathological pair
  like ``A.x: from B import x`` / ``B.x: from A import x`` terminates
  after one trip around the cycle instead of repeatedly chaining back
  to its starting state. On a 1470-file monorepo this manifested as a
  22-minute hang at 100% CPU stuck inside the per-package compose
  loop; the decls actually encountered along the cycle are still
  emitted, so first-party reachability through the chain is preserved.
  ``resolve_edges`` also memoizes the full per-import resolution at
  three layers: ``_resolve_targets`` keyed by ``Import`` value (so
  equal spellings across files share the precomputed dst list — the
  visitor builds fresh ``Import`` objects per file, but they hash
  equal because ``Import`` is frozen with an eager ``__hash__``);
  ``_walk`` keyed by ``(start_node, decl_parts)`` (so different
  ``Import`` shapes that canonicalize to the same trie state share the
  re-export DFS); and ``_classify`` keyed by
  ``(import.module, import.speculative)`` (so the resolver runs once
  per unique external name). The per-src loop collapses to
  ``for dst in cached_targets: emit(...)``, turning the per-package
  compose loop's growth in importer count from multiplicative to
  additive.
- **v0.9.0**: Parallel refresh (``--workers >= 2``) is now resilient and
  observable. Worker results stream via ``concurrent.futures.as_completed``
  instead of ``pool.map``, so cache writes and progress ticks land in
  completion order and a single slow file no longer blocks the cache from
  warming with the fast files behind it. Per-task failures are collected
  and re-raised as one ``ExceptionGroup`` after every other task finishes
  (successfully-parsed files are still cache-warmed before the group is
  raised, so a re-run after fixing the bad file only re-parses what
  failed). The pool installs SIGTERM/SIGINT handlers for the lifetime of
  the run; on signal it cancels every pending future and raises
  ``KeyboardInterrupt``, with completed files still cache-warmed.
- **v0.9.0**: Progress reporting is fully logger-driven and controlled by
  the root logger level. Per-file refresh status goes through
  ``logger.debug`` on ``dead_cst._refresh``; off-TTY decile checkpoints
  go through ``logger.info`` on ``dead_cst._progress``; the on-TTY tqdm
  bar wraps its iteration in ``logging_redirect_tqdm`` so concurrent log
  records print above the bar without shattering it. ``dead-cst -v``
  keeps its meaning; library users get the same firehose by configuring
  their root logger.
- **v0.9.0**: ``SymbolNode`` and ``Import`` pre-compute their hash in
  ``__post_init__`` and store it in a private ``_hash`` slot, so
  ``__hash__`` is a single attribute read. Cuts edge-stitching time on
  large multi-package workspaces where ``resolve_edges._emit`` re-hashes
  the same ``(src, dst, flags)`` tuples into its dedup set, and pays off
  again every time a ``SymbolNode`` is hashed by networkx (graph
  insertion, BFS traversal). ``SCHEMA_VERSION`` bumped to 3 so cache
  rows pickled before the slot existed are invalidated on first use.
- **v0.9.0**: Edge stitching no longer emits a spurious "Failed to
  resolve import module: <name>" warning for stdlib imports
  (``import datetime``, ``from pathlib import Path``); the orphaned
  warning fired for every successfully-classified stdlib import because
  the silent-drop and speculative-miss branches both returned ``None``.
  Truly-unresolved non-speculative imports already surface as
  ``[unresolved] <top-level>`` synthetic nodes.
  ``default_resolve_import`` also now falls back to the parent module
  when a dotted name can't be resolved directly, so
  ``collections.abc``, ``importlib.resources.abc``, and similar
  synthesized-in-``__init__`` submodules classify as ``[stdlib] <name>``
  instead of being misfiled as ``[unresolved] <top>``.
- **v0.8.0**: `dead-cst remove` is now non-destructive — it emits a
  `git apply`-compatible unified diff to stdout (or `--output PATH`)
  and never touches source files (breaking; `--dry-run` and the
  confirmation prompt are gone). New public function
  `dead_cst.codemod.generate_patch(G, root)` returns the same diff for
  any subgraph slice, so callers can render per-SCC (or per-WCC)
  patches by passing `G.subgraph(scc)` for incremental review of large
  codebases. This lays the foundation for the `dead-cst preview`
  subcommand in tier 7 — automatic WCC clustering and blast-radius
  headers are the remaining work.
- **v0.8.0**: The visitor now honors ruff/pyflakes ``# noqa`` directives
  that silence F401. Per-line variants (bare ``# noqa``,
  ``# noqa: F401``, multi-rule ``# noqa: E501, F401``, case-variant
  ``# NOQA``) and file-level pins (``# ruff: noqa``, ``# flake8: noqa``)
  flag the resulting import nodes ``ENTRYPOINT | NOQA`` so reachability
  keeps them alive — bringing dead-cst's unused-import semantics in
  line with ruff's. The new ``NodeFlags.NOQA`` flag layers on
  ``NodeFlags.ENTRYPOINT`` parallel to ``NodeFlags.TESTCASE``, and the
  two ``kept_alive_by_*_only`` methods collapse into a single
  ``kept_alive_by_flags_only(flags: NodeFlags)`` (breaking) that takes
  any flag combination — pass ``NodeFlags.TESTCASE`` for the old
  test-blast-radius behavior, ``NodeFlags.NOQA`` for the F401 pin
  blast radius, or both ORed together.
- **v0.8.0**: What-if graph surgery via
  ``Analysis.preview(files, *, detector=None)`` — regenerates per-file
  payloads for a hand-picked file set (bypassing the on-disk cache,
  no read or write), splices them into a fresh overlay graph leaving
  the baseline untouched, and returns a ``GraphView`` exposing the
  same ``reachable`` / ``dead`` / ``kept_alive_by_dead_branches`` /
  ``kept_alive_by_flags_only`` / ``count_nodes`` surface as
  ``Analysis``. Pairs with the new
  ``TruthinessResolver.resolve_constant(expr) -> Const | None`` (the
  literal-value sibling of ``evaluate``, wrapped in ``Const`` so a
  proved-``None`` literal stays distinct from a bare ``None`` "unknown"
  return) so a custom detector can fold a flag-name ``Name``
  (``check_flag(FEATURE_A)`` where ``FEATURE_A = "feature_a"``) before
  pattern-matching. ``DefaultUnreachableRegionDetector.resolve(expr)``
  is now ``resolve(expr, resolver)`` (breaking — subclasses with a
  one-arg ``resolve`` need to add the parameter) so overrides can call
  ``resolver.resolve_constant`` recursively.
- **v0.8.0**: ``DefaultUnreachableRegionDetector`` recognizes compound
  statements as terminators when every reachable branch terminates —
  ``if True: return``, ``if FLAG: return`` with ``FLAG = True``,
  ``if cond: return; else: return``, ``with`` whose body terminates,
  and ``try``/``except``/``finally`` where every path terminates all
  kill statements that follow them in the enclosing suite. Constant-
  folded early returns now flag trailing dead code as expected. The
  resolver also no longer folds a ``Name`` whose binding's RHS is a
  mutable container literal (``[]``, ``{}``, comprehension RHS): the
  binding-only flow walk is invisible to ``.append`` / item assignment
  / ``.update``, so an ``edges = []; edges.append(x); if not edges:``
  chain used to fold incorrectly. Tuples and primitives stay safe to
  fold. Detector ``version`` bumped so cached payloads rebuild.
- **v0.7.0**: `DefaultUnreachableRegionDetector` rewrite — the
  `fold_constants` fixpoint pre-pass is replaced by the goal-directed
  `TruthinessResolver`, which lazily walks only the binding slices a
  query touches and memoizes by access node id. Self-analysis benchmark
  drops `find_regions` from 24.2 s to 1.5 s over `dead_cst/` (~16×).
  `dead_cst.branches.fold_constants` and the internal `_const_fold`
  module are gone (breaking); from-scratch detector authors construct
  `TruthinessResolver(wrapper, resolve_expr=...)` and pass
  `resolver.evaluate` to `unreachable_suites` / `evaluate_truthiness`.
- **v0.7.0**: `libcst >= 1.8.6` floor; PEP 750 t-strings (`t"..."`,
  Python 3.14+) parse cleanly and route through the visitor's existing
  scope resolution, so `t"hello {NAME}"` produces the same edge an
  f-string would. The "t-strings unsupported" limitation is gone.
- **v0.7.0**: Files `libcst` cannot parse no longer abort the run. The
  analyser logs a warning and substitutes a placeholder payload (the
  real module node plus an `[unparseable] <module>` synthetic flagged
  `ENTRYPOINT`), so the file stays alive in reachability and importers
  still resolve. Decls inside the file are invisible until parsing
  succeeds; the placeholder rides the per-file cache and a fresh source
  SHA invalidates it automatically. `enumerate_files` also skips
  directories whose names happen to end in `.py` / `.pyi`.
- **v0.6.0**: Compiled-extension `.pyi` stub ingestion (`mypkg/_native.so`
  + `mypkg/_native.pyi`); peer-mode stubs alongside a real `.py` are
  intentionally dropped. `@typing.overload`-decorated decls flagged with
  `NodeFlags.OVERLOAD`, excluded from the cross-module trie, and anchored
  to their same-file impl via explicit `impl -> overload` edges so the
  codemod removes overloads with their impl.
- **v0.6.0**: `NodeFlags.TESTCASE` plus the per-package
  blast-radius query for "what would die if the test suite were
  dropped" (later generalized into `kept_alive_by_flags_only`).
  `PytestPlugin` and `UnittestPlugin` stamp `ENTRYPOINT | TESTCASE` on
  their synthetic seeds.
- **v0.6.0**: `UnittestPlugin` resolves transitive `TestCase` subclasses
  through bucket markers in `observe` + a `finalize` walk from
  `unittest.TestCase` / `IsolatedAsyncioTestCase` (and every alias) so
  project-local mixins and re-exported `TestCase` bases keep their
  subclasses alive.
- **v0.6.0**: Per-file refresh logic extracted into
  `dead_cst/_refresh.py` (file enumeration, stale detection, worker
  pool, payload application). `analyze.py` keeps cross-package
  composition only. "Base" -> "package" rename across the public API.
- **v0.6.0**: `tqdm` progress reporting around the parse and reconcile
  passes; off-TTY consumers get newline-terminated checkpoints instead
  of `\r`-overwriting frames.
- **v0.6.0**: `CycloptsPlugin` plus a generalized `DispatchAppPlugin`
  base for `X = App(); @X.command(...)` shapes (Typer migrated onto
  it). `MockPatchPlugin` resolves string-fqname targets for
  `mock.patch` / `mocker.patch` / `monkeypatch.setattr`.
- **v0.5.0**: Cross-file import resolution moved out of `SymbolVisitor`
  and into `_edges.resolve_edges`. `Import` is now raw (just the
  written-down dotted name); the per-file cache survives `search_paths`
  / resolver / package-layout swaps. Single resolver per `Analysis`
  (no chain). `PathResolver.resolve` returns `tuple[Package, ...]`
  with explicit `name` / `exported` / `deps`. `VenvResolver` and
  `PyprojectResolver` retired (use `-p` / `ManualResolver`);
  `UvWorkspaceResolver` renamed to `UvResolver`.
- **v0.5.0**: Parallel visitor pass via `--workers` / `-j`. Workers
  return `VisitorPayload` blobs; cache writes, trie stitching, and
  edge resolution stay in the parent. FQN cache built once over miss
  files only and shipped per-task.
- **v0.5.0**: Public API split into focused submodules
  (`dead_cst.graph`, `dead_cst.analyze`, `dead_cst.codemod`,
  `dead_cst.cache`, `dead_cst.branches`, `dead_cst.plugins`,
  `dead_cst.resolvers`, `dead_cst.contrib`).
  `tests/test_public_api.py` pins each module's `__all__`. The lazy
  `Analysis` / `PackageView` shape replaces the `build_symbol_graph` /
  `find_reachable` / `count_nodes` / `order_paths` /
  `find_kept_alive_by_dead_branches` / top-level `remove_code` API.
- **v0.5.0**: Path-classification fixes for system-Python layouts
  (site-packages nested inside the stdlib root) and editable installs
  (`pip install -e`); first-party search paths win over editable dist
  roots so e2e fixtures cloned inside another project don't blow away
  reachability. `__import__` / `importlib.import_module` with a
  string-literal name (including relative names and `fromlist=[...]`
  literals) fanned out as star imports.
- PEP 695 `type` statements: `type Foo = list[int]` (and the generic
  `type Pair[T] = tuple[T, T]` form) now surface as top-level
  `"type_alias"` decls. RHS references attribute to the alias, so
  removing a dead alias releases its references; users that reference
  the alias get an edge into it. The codemod's `RemoveDeadSymbols`
  pass deletes unreachable aliases.
- PEP 572 walrus (`:=`) bindings at module scope are surfaced as top-level
  decls and folded by the unreachable-region detector the same way
  `Assign` / `AnnAssign` are. Walruses leaked from module-level
  comprehensions are captured by patching `ScopeProvider`'s comprehension-
  scoped binding.
- `UnreachableRegionDetector` Protocol with a shipped
  `DefaultUnreachableRegionDetector`: a single CST visit collects every
  `If` / `While` and statement-bearing suite, and a goal-directed
  `TruthinessResolver` answers truthiness queries on demand (literal +
  flow-sensitive `Name` lookup over `Name = literal` chains).
  Post-terminator scan over every collected suite, including compound
  `if` / `with` / `try` whose every reachable branch terminates (so a
  constant-folded `if True: return` kills the rest of its enclosing
  suite). Subclasses override `resolve(self, expr) -> bool | None` to
  fold domain-specific expressions; resolved values compose with the
  resolver's name lookup.
- `Cacheable` Protocol unifying `(name, version)` across visitor, resolvers,
  plugins, and detectors. Package `__version__` removed from the cache
  fingerprint — each component carries its own knob, and concurrent bumps
  on different branches merge with `max()` semantics.
- `DecoratedDeclPlugin` and `LiteralListPlugin` abstract bases for the two
  most common plugin idioms (decorator-driven decls and string-literal-list
  registries). `ClickPlugin` migrated to the former.
- E2E test suite at `tests/e2e/` (`-m e2e`, deselected by default) clones
  real repos at pinned SHAs and exercises analyze + why-alive + project-
  specific plugins.
- CLI integration tests at `tests/test_cli.py` covering analyze, remove,
  why-alive, unused-exports, and dependencies via `typer.testing.CliRunner`
  (Tier 1).
- Coverage tracking in CI: Codecov upload from the 3.13 matrix entry with
  per-component thresholds in `codecov.yml` (Tier 2).
- SQLite-cached graph with partial rebuilds: `GraphCache` stores
  pickled `VisitorPayload` blobs keyed by per-file content hash under
  `<root>/.dead-cst-cache/cache.db`. Cache hits skip the per-file
  visitor pass; edge resolution and plugin `finalize` run every
  analysis. The fingerprint covers Python version, schema version,
  and each visitor / plugin / detector `(name, version)` pair --
  resolver, `search_paths`, and the package layout deliberately do
  not enter it (their effect flows through the uncached edge stitcher).
  `--no-cache` flag and `dead-cst cache clear` subcommand.
- Resolver logic as a protocol: `PathResolver.resolve_import` folds
  `name -> path` lookup into the resolver, so custom resolvers can
  override import resolution for their own layouts. `_resolve.py`
  renamed to `_edges.py` since resolution now lives under
  `_resolvers/`.
- Codemod test coverage and import pruning (Tier 1).
- `from X import *` resolution, pessimistic by default (Tier 1).
- `PytestPlugin`, `UnittestPlugin`, `FastAPIPlugin`, `FlaskPlugin`,
  `TyperPlugin`, `ClickPlugin`, and `InitSubclassPlugin` (Tier 1,
  partial — see item 1).
- `unused-exports` and `dependencies` CLI commands.
- Unreachable-branch detection surfaced as synthetic graph nodes.
- Workspace-aware cross-member import scoping via `exported_roots`.
- Position-aware shadowing in the codemod.
- `ModuleDundersPlugin` replacing `--preserve-dunder-all`.
- Public-API docstring pass across the package.
