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

`PytestPlugin`, `UnittestPlugin`, `FastAPIPlugin`, `FlaskPlugin`,
`TyperPlugin`, `ClickPlugin`, and `InitSubclassPlugin` shipped, but the
existential risk is still "I tried it and it flagged half my codebase."
The remaining common offenders:

- Django URLConf, admin registration, signal handlers, management commands
- Pydantic validators and field serializers
- Descriptor-style hooks: `__set_name__`, dataclass `__post_init__`

Surface a `--preset pytest,fastapi,django` shortcut that expands to the
existing `--plugin` wiring, and document the entry-point group so third
parties can publish their own.

### 2. Function-call folding (same-file, caller-capped)

`def is_new_auth(): return False; if is_new_auth(): X = 1` should mark `X` as
dead, but doesn't yet — `evaluate_truthiness` doesn't fold through calls. Add
a pre-pass in `fold_constants` that walks top-level `FunctionDef` bodies for
trivial single-return shapes and lets the resolver recurse through them. Both
sync `def` (folded via bare `Call`) and async `def` (folded via `Await(Call)`
only — a bare async call returns a coroutine, always truthy). Cap by caller
count (default 3) so that configuration helpers used in 50 places don't get
folded into noise. Same-file only; cross-file is item 8.

### 3. Suite-removal codemod and expanded dead-set helper

`dead-cst remove` only deletes decls today; the `If` / `While` / post-
terminator suites that `analyze` flags can't be removed automatically. Add a
`RemoveDeadSuites` LibCST transformer covering each parent shape (`If` /
`While` / `elif` chains / post-terminator) with empty-suite guards inserting
`pass` where needed, and a helper that returns the union of dead decls + dead
suite ranges + `find_kept_alive_by_dead_branches(graph)` so removal expands
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

Peer subcommand to `analyze` / `remove` that renders item 3's codemod as
per-cluster unified diffs instead of editing files. Clusters are weakly-
connected components of the unreachable subgraph: each WCC is the maximal
unit that can be applied atomically without leaving dangling references.
Cluster headers carry blast-radius metadata (symbols, LOC, files affected).
Composes with `git apply` (whole cluster) and `git add -p` (per-line review)
without needing a custom TUI; see item 11 for the deferred TUI shape.

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
a `main()` that builds the graph, calls `find_kept_alive_by_dead_branches`,
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
  `DefaultUnreachableRegionDetector` running three passes per file: the
  literal-only `unreachable_suites` walk, a `fold_constants` fixpoint pre-
  pass (chained `Name = literal` propagation), and a post-terminator scan
  over every statement-bearing suite. Subclasses override
  `resolve(self, expr) -> bool | None` to fold domain-specific expressions;
  folded values flow through the same fixpoint loop.
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
  visitor pass; per-base edge resolution and the plugin pass run
  every analysis. Fingerprint over `(__version__, python version,
  PathMap, resolver chain)` invalidates the whole file_cache on
  layout / version change. `--no-cache` flag and
  `dead-cst cache clear` subcommand.
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
