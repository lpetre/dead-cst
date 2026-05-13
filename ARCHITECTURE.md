# Architecture / program flow

This doc walks the code path a single `dead-cst` invocation takes, from CLI
arguments to a written file. It's the developer-facing companion to
[`README.md`](README.md) (user-facing) and [`CLAUDE.md`](CLAUDE.md) (LLM-
oriented summary). Read this before adding a plugin, resolver, or detector,
or before touching the visitor.

## At a glance

```
   CLI / Python API
        │
        ▼
┌──────────────────┐
│   PathResolver   │  resolve(project_root) -> tuple[Package, ...]
└─────────┬────────┘
          │ Packages (path, name, exported, deps)
          ▼
┌──────────────────┐
│     Analysis     │  cheap construction; everything below is lazy
└─────────┬────────┘
          │ refresh(packages=…)
          ▼
┌──────────────────────────────────────────────────────────────┐
│  per-file pipeline (cached in SQLite by file SHA-256)        │
│                                                              │
│   source ─► SymbolVisitor ─► VisitorPayload                  │
│                  │                                           │
│                  └─► UnreachableRegionDetector.find_regions  │
│                                                              │
│   each EdgePlugin.observe(ctx) -> VisitorPayload | None      │
│   ────────────────────────────────────────────────────────   │
│   combined payload pickled into GraphCache                   │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  per-package contribution                                    │
│  (SymbolTrie + package-local graph slice + unresolved imps)  │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  cross-package composition (uncached)                        │
│                                                              │
│   merge contributions ─► resolve_edges (against merged trie  │
│                          + PathResolver fallback)            │
│                                                              │
│   each EdgePlugin.finalize(ctx) -> Iterable[GraphOp]         │
└────────────────────────────┬─────────────────────────────────┘
                             │ MultiDiGraph
                             ▼
                ┌────────────┴────────────┐
                ▼                         ▼
        reachability BFS         codemod (remove_code /
        from entrypoint=True       generate_patch)
```

The cache boundary is the line that matters most when reasoning about
performance: stages 1–4 ride the per-file SQLite cache; stage 5 onward
runs every invocation. That's why import classification deliberately
moved out of the visitor — keeping the cache survival surface as small
as possible was worth re-stitching edges every run.

## The `Cacheable` contract

Four moving parts satisfy `Cacheable` (`dead_cst/_cacheable.py`) — just
`name: str` + `version: int`:

| Component                    | Where                       | In the fingerprint? |
| ---------------------------- | --------------------------- | ------------------- |
| `SymbolVisitor`              | `dead_cst/_visitor.py`      | yes                 |
| `EdgePlugin`                 | `dead_cst/plugins/`         | yes                 |
| `UnreachableRegionDetector`  | `dead_cst/branches.py`      | yes                 |
| `PathResolver`               | `dead_cst/resolvers/`       | **no**              |

Only the visitor / plugin / detector triple feeds `compute_fingerprint`
(`dead_cst/cache.py`); resolver and package-layout swaps re-stitch edges
through the (uncached) stage 5 pass instead of invalidating the per-file
cache. `version` is a Unix epoch int *by convention* — concurrent bumps
on different branches merge with `max()`-wins semantics rather than
colliding on a re-used label. The package `__version__` is **not** in
the fingerprint: every component whose output can shift between releases
owns its own knob.

## The pipeline, top-down

### 1. Path resolution — `dead_cst/resolvers/`

A `PathResolver` answers two questions about the project layout:

* `resolve(project_root) -> tuple[Package, ...]` — what packages exist,
  what subdirs do they export to consumers, and which other packages
  do they depend on (referenced by `name`).
* `resolve_import(name, search_paths) -> str | Path | None` — fallback
  classifier when an import misses the per-package symbol trie.

Builtins:

* `ManualResolver` (`dead_cst/resolvers/manual.py`) — explicit
  `package:dep1,dep2` specs from the CLI's `-p` flag. Auto-promotes
  inline dep paths to their own `Package` records.
* `UvResolver` (`dead_cst/contrib/uv.py`, re-exported from
  `dead_cst.resolvers`) — parses `uv.lock` to discover workspace
  members and inter-member edges; lazily splices the workspace
  `.venv/site-packages` onto `sys.path` inside its own
  `resolve_import`.

`Analysis` takes exactly one resolver — no chain. CLI flags `-p` and
`--resolver` are mutually exclusive. Construction validates the
resolver's output (name uniqueness, dep references, `exported` entries
under `path`) via `_validate_packages`.

### 2. Per-file visitor — `dead_cst/_visitor.py`

`SymbolVisitor` walks one `.py` (or `.pyi`) file and returns a
`VisitorPayload` (defined in `dead_cst/graph.py`) with four fields:

| Field          | Shape                                          | Notes                                   |
| -------------- | ---------------------------------------------- | --------------------------------------- |
| `nodes`        | `tuple[SymbolNode, ...]`                       | every real decl, including `SHADOWED`   |
| `edges`        | `tuple[(src, dst, EdgeFlags), ...]`            | resolved decl-to-decl refs in this file |
| `imports`      | `tuple[(src, Import, EdgeFlags), ...]`         | **raw** — just the dotted name written  |
| `dead_suites`  | `tuple[CodeRange, ...]`                        | every statically-dead suite             |

Crucially the visitor never calls a resolver and never reads `sys.path` —
its output is purely a function of the file's source plus the plugin /
detector chain. That's what lets the per-file cache survive
`search_paths` / resolver swaps.

Heavy lifting comes from LibCST `ScopeProvider`,
`FullyQualifiedNameProvider` (wrapped via `_fqn.FixedFullyQualifiedNameProvider`),
and the flow-sensitive shadowing in `_flow.py`.

### 3. Unreachable-region detection — `dead_cst/branches.py`

Invoked once per file from inside `SymbolVisitor.visit_Module` reusing
the analyzer's resolved `PositionProvider`. The shipped
`DefaultUnreachableRegionDetector` is a two-pass design:

1. A single `cst.CSTVisitor` walk collects every `If` / `While` plus
   every statement-bearing suite (module body and every
   `IndentedBlock`).
2. A `TruthinessResolver` (also in `branches.py`) answers truthiness
   queries on demand: `unreachable_suites` for the conditional sites,
   plus a per-suite scan that marks statements after an unconditional
   `return` / `raise` / `break` / `continue` / `assert <falsy>` as
   dead *within the same suite* — a `raise` in a `try` body doesn't
   kill the matching `except`. Compound statements (`if` / `with` /
   `try`) themselves count as terminators when every reachable path
   inside them terminates, so a constant-folded `if True: return`
   (and its `FLAG = True` / `if cond: return; else: return` cousins)
   kills the rest of the enclosing suite. The resolver is goal-directed: it
   lazily resolves `ScopeProvider` / `ParentNodeProvider` only when
   the first `Name` query lands, walks `live_referents` only for the
   names that actually feed a query, memoizes by access node id, and
   uses a `_PENDING` sentinel to bottom out cyclic references like
   `a = b; b = a`.

The apply step compares each `access_pos` in `edges` against
`dead_suites` to flag `EdgeFlags.DEAD_BRANCH`. Reachability still
follows those edges by default; `Analysis.kept_alive_by_dead_branches`
is the opt-in inverse.

To layer in domain knowledge, subclass `DefaultUnreachableRegionDetector`
and override `resolve(self, expr) -> bool | None` — it gets first crack
at every non-keyword expression routed through the resolver chain. Pass
an instance via `Analysis(unreachable_detector=...)`. Bump `version`
whenever the override's logic changes. From-scratch detectors that
need name-aware truthiness construct `TruthinessResolver(wrapper,
resolve_expr=...)` once per file and pass `resolver.evaluate` as the
`resolve_expr` callback to `unreachable_suites` / `evaluate_truthiness`.

### 4. Visitor + observe cache — `dead_cst/cache.py`

SQLite-backed `GraphCache` at `<root>/.dead-cst-cache/cache.db` stores
pickled per-file payloads keyed by file SHA-256. Each row carries a
single analysis-wide fingerprint over Python version, schema version,
and every visitor / plugin / detector `(name, version)` pair.

A cache row covers both the visitor's payload **and** every plugin's
`observe()` output for that file — warm runs skip both. Bypass with
`--no-cache`; force-clear with `dead-cst cache clear`.

The orchestration around this stage — file enumeration, stale detection,
the parallel worker pool, payload application into the per-package
contribution — lives in `dead_cst/_refresh.py` so `analyze.py` can stay
focused on cross-package composition. File enumeration walks the tree
once via `Path.rglob("*")` and dispatches by suffix into `.py` /
`.pyi` / `.ipynb` buckets — directory I/O dominates the per-name
fnmatch cost on large repos, so one walk beats three suffix-specific
globs. Jupyter notebooks are converted to a single Python source
string by `_notebooks.notebook_to_module` before the visitor sees
them, and the visitor is constructed with
`default_flags=NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT` so every
emitted node carries those flags from the start. When `libcst`
rejects a file's syntax (or a notebook's JSON is malformed), the
per-file work logs a warning and substitutes a placeholder payload
pairing the real module node with a `[unparseable] <module>`
synthetic flagged `ENTRYPOINT` — the file stays alive in reachability
and rides the per-file cache like any other miss, so a fresh source
SHA picks up the fix automatically. The pool consumes worker results via
`concurrent.futures.as_completed`, so cache writes land in completion
order — a single slow file does not block the cache from warming with
the fast files behind it. Per-task failures other than the in-band
`OSError` / parse cases are collected and re-raised as a single
`ExceptionGroup` after the run drains, so one bad file does not waste
the rest of the work; successful payloads are cache-warmed before the
group is raised. The pool installs SIGTERM/SIGINT handlers for its
lifetime, so a signal cancels pending futures and re-raises
`KeyboardInterrupt`; files that completed before the signal stay
cache-warmed.

### 5. Edge stitching — `dead_cst/_edges.py`

`resolve_edges` runs **unconditionally** every analysis (it isn't
cached). It walks the raw `(src, Import, flags)` triples against the
per-package `SymbolTrie`, **canonicalizing** each `Import` first by
pushing decl parts into `module` while they resolve as submodules in
the trie:

```
from p import functions
   trie has p.functions as a submodule?
   -> module="p.functions", decl=None
```

Trie misses fall back to the `PathResolver` chain for stdlib /
external-dist / external-file / unresolved classification — the *only*
place that reads `sys.path` and the resolver LRU caches.
`Analysis._materialize` rebinds `sys.path` to each package's
`(path, *deps)` view before composing it and calls
`clear_module_specs_cache()` at every transition, restoring `sys.path`
on the way out. That narrow clear drops only the fullname-keyed
`safe_resolve_module` cache; `distribution_lookup` and
`editable_distribution_roots` are keyed on the dist-bearing slice of
`sys.path` (site-packages / dist-packages / purelib / platlib entries)
so they survive the rebind for free — only the first-party prefix
moves, and that prefix never enters the key. A real venv change (uv
splicing in a workspace `.venv`) flips the key and triggers a single
rebuild. `clear_path_caches()` is still available as the heavy-hammer
full reset for callers that mutate the venv slice themselves.

Stdlib imports drop silently — they aren't surfaced as graph nodes.
External-dist / external-file / unresolved misses produce one synthetic
node per group so plugins can still answer "which files imported X?".
A dotted name whose own `find_spec` returns nothing (`collections.abc`
and friends, synthesized in their parent's `__init__`) falls back to
the parent's classification, so the child inherits stdlib / dist
attribution instead of being misfiled as `[unresolved]`.

Star imports follow the same path; `Import.speculative` (set on
`__import__` fromlist synthesis) drops a trie+resolver miss without
emitting an `[unresolved]` node.

Re-running this every analysis is what makes single-file edits cheap:
only the edited file's payload is recomputed; importers re-stitch for
free.

Resolution is memoized at three nested layers per call so the per-
package compose loop stays additive in importer count: equal-spelling
`Import` instances (the visitor builds fresh objects per file but they
hash equal because `Import` is frozen) share one `_resolve_targets`
entry; different `Import` shapes that canonicalize to the same trie
state share one `_walk` entry; and the resolver fallback runs once per
unique `(module, speculative)` external. The walk's worklist DFS is
cycle-protected via a per-walk `visited` set keyed on
`(id(SymbolTrie), parts_tail)`, so a pathological re-export cycle
(`A.x: from B import x` / `B.x: from A import x`) terminates after one
trip with the encountered decls still emitted.

### 6. Plugins — `dead_cst/plugins/`

Two phases per `EdgePlugin`:

* `observe(ctx: ObserveContext) -> VisitorPayload | None` — runs once
  per file inside the visitor loop, with the parsed `cst.Module` and
  the visitor's just-built payload. Returns a new payload (or `None`)
  whose `nodes` / `edges` extend the file's contribution; the result
  is cached alongside the visitor's output. **Cross-file work does
  not belong here.**
* `finalize(ctx: PluginContext) -> Iterable[GraphOp]` — runs once per
  package after `resolve_edges` has stitched cross-file imports.
  Operates purely on the assembled graph (no CST access) and emits
  `AddNode` / `AddEdge` / `RemoveEdge` ops.

`PluginContext` provides helpers (`find_module`, `find_declarations`,
`module_surface`, `package_modules`, `package_nodes`, `importers`, …)
and exposes the current `Package` via `ctx.package`.

Builtins ship in `BUILTIN_PLUGINS`. Generic-Python plugins live as
siblings of `plugins/__init__.py` (`MainBlockPlugin`,
`ProjectScriptsPlugin`, `ExplicitEntrypointPlugin`,
`ModuleDundersPlugin`, `InitSubclassPlugin`). Third-party-aware
plugins live under `dead_cst/contrib/` and are re-exported from
`dead_cst.plugins`. `FastAPIPlugin` is the full-featured reference for
two-phase plugins; `ClickPlugin` for the `DecoratedDeclPlugin`
subclass shape; `TyperPlugin` / `CycloptsPlugin` for the
`DispatchAppPlugin` shape.

For the common dynamic-import idioms, three abstract bases ship as
scaffolding (`plugins/decl_shapes.py`):

* `DecoratedDeclPlugin` — "decorated decls in files matching a search path."
* `LiteralListPlugin` — "read `<owner>.<var> = ['fqn', ...]` and revive each entry."
* `DispatchAppPlugin` — "wire `@<instance>.<reg>(...)` handlers to a CLI app."

### 7. Reachability — `Analysis.reachable` / `PackageView.reachable`

BFS from every node with `entrypoint=True`. Default traversal **does**
follow `DEAD_BRANCH` edges (preserving today's behavior).
`Analysis.kept_alive_by_dead_branches()` is the opt-in inverse, returning
the blast radius of removing every dead suite by skipping those edges.
`Analysis.kept_alive_by_flags_only(flags)` / the per-package
`PackageView` twin returns the blast radius of dropping every
entrypoint carrying any of those flag bits. Pass `NodeFlags.TESTCASE`
for "production code currently kept alive only because tests still
touch it" (`PytestPlugin` / `UnittestPlugin` stamp `ENTRYPOINT |
TESTCASE`); pass `NodeFlags.NOQA` for "decls kept alive only by an
F401 pin" (the visitor stamps `ENTRYPOINT | NOQA` on imports
preserved by a per-line or file-level ruff/pyflakes directive); OR
the bits to combine.

### 8. Codemod — `dead_cst/codemod.py`

`remove_code` runs a LibCST `RemoveDeadSymbols` transformer keyed on
`(fqname, CodeRange)` pairs (position disambiguates shadowed decls),
then prunes now-unused imports via `RemoveImportsVisitor`. Position
keying is critical — losing it conflates a dead decl with a live shadow.

The high-level entry point is `PackageView.remove_dead_code()`, which
materializes the package's interesting-set closure, computes
reachability, and feeds the unreachable subgraph into `remove_code`.

`generate_patch(G, root)` is the non-destructive twin: same selection
logic, same two-pass LibCST pipeline (a private `_rewrite_one` helper
keeps the two functions from drifting), but instead of writing back it
emits a `git apply`-compatible unified diff with `diff --git` headers
and a `deleted file mode 100644` extended header for module-node
deletions. Selection is driven entirely by `G.nodes`, so callers can
slice the unreachable graph however they like (e.g. one SCC at a time)
to review a big codebase as a series of focused patches. The
`dead-cst remove` CLI uses `generate_patch` exclusively — it emits a
patch to stdout (or `--output PATH`) and never mutates source.

## Lazy materialization on `Analysis`

`Analysis` is cheap to construct — the resolver runs once at `__init__`,
but no source files are read or parsed until you ask. Three coarse
stages happen on demand:

1. **File enumeration + visitor pass** — `refresh()`. Walks each
   requested package's tree, collapses every package's cache misses
   into one global stale-file list, and runs the visitor + observe
   pass once across the whole batch.
2. **Per-package contribution build** — the per-package
   `SymbolTrie` + a package-local graph slice + the unresolved
   cross-file import set. Built once per package from the payloads
   above, memoized for the lifetime of the `Analysis`.
3. **Cross-package composition** — `materialize_all()` (every package)
   or `materialize_closure(package)` (the "interesting set" of one
   package — the forward dep closure of that package's reverse-consumer
   closure). Composing a graph is much cheaper than recomputing
   payloads, so warm per-package queries stay fast.

Per-package queries that don't need the assembled graph
(`PackageView.modules` / `PackageView.declarations`) skip stage 3
entirely.

## Graph model invariants

* One node per top-level declaration plus one synthetic module node per
  file. Nested defs (inner functions, methods, nested classes) are
  deliberately not given their own nodes — refs from inside them
  attribute to the enclosing top-level decl.
* A module-level `import` / `from ... import ...` is itself a node of
  type `"import"`. Local uses of an imported name go through the import
  node, which points at the upstream module / symbol. This is how
  `dead-cst remove` knows to drop now-unused imports.
* Imports whose source line carries a ruff/pyflakes `# noqa` directive
  that silences F401 (bare `# noqa`, `# noqa: F401`, multi-rule
  `# noqa: E501, F401`, case-variant `# NOQA`) are flagged
  `NodeFlags.ENTRYPOINT | NodeFlags.NOQA` so reachability keeps them
  alive — matching ruff's own semantics. File-level `# ruff: noqa` and
  `# flake8: noqa` directives (`ruff:` / `flake8:` matched
  case-sensitively per ruff; `noqa` is not) pin every import in the
  file. The visitor scans for these in `visit_Comment` plus a
  per-import-statement `SimpleStatementLine` walk, so per-alias
  comments inside a parenthesized `from x import (a, b)` are honored.
  The `NOQA` bit is metadata layered on `ENTRYPOINT` (parallel to
  `TESTCASE`); the single `kept_alive_by_flags_only(flags)` method on
  `Analysis` / `PackageView` takes either flag (or both ORed) to
  return the blast radius of dropping the matching entrypoints.
* Submodules edge to their parent package, so `__init__.py` stays alive
  as long as anything in the package does.
* `EdgeFlags.DEAD_BRANCH` is metadata only.
* `NodeFlags.SHADOWED` decls are emitted into the graph but excluded
  from the trie, so cross-module imports never resolve to them.
  `NodeFlags.OVERLOAD` follows the same trie-exclusion rule but its
  lifetime is anchored to the matching same-file impl via explicit
  `impl -> overload` edges.
* `.pyi` stubs are ingested only for the compiled-extension layout
  (`_native.so` + `_native.pyi`, no `.py` twin). Peer-mode `.pyi` is
  dropped at file-enumeration time — the runtime always wins.
* `.ipynb` (Jupyter) files are concatenated cell-by-cell into one
  parseable Python module by `_notebooks.notebook_to_module`. IPython
  magics, shell escapes, and trailing-help forms are rewritten to
  `pass  # <line>` so libcst accepts the source. The visitor is
  constructed with `default_flags=NodeFlags.NOTEBOOK | NodeFlags.ENTRYPOINT`
  so every node carries those flags from the start; `NOTEBOOK` also
  keeps the decl out of the cross-module lookup trie alongside
  `SHADOWED` / `OVERLOAD`. The codemod skips any node flagged
  `NOTEBOOK`.
* A `foo.py` next to a `foo/__init__.py` is shadowed by the package
  (mirroring CPython's `FileFinder`). `_refresh.shadowed_paths` flags
  the `.py` before `_apply_payload` runs, and the apply pass skips its
  trie additions only — its nodes still land in the graph so
  observe-time entrypoints (`__main__`, plugin synthetics) keep
  working, but cross-module imports of `pkg.foo` route to the
  package's `__init__.py` alone.
* `[unparseable] <module>` synthetics stand in for files `libcst`
  cannot parse. They carry `NodeFlags.ENTRYPOINT` and edge at the real
  module node, so the file stays alive even though its decls are
  invisible.

## Where to make changes

| If you want to…                                          | Touch                                          |
| -------------------------------------------------------- | ---------------------------------------------- |
| Recognize a new decl shape                               | `_visitor.py` (bump `SymbolVisitor.version`)   |
| Fold a domain-specific expression to a known truthiness  | subclass `DefaultUnreachableRegionDetector`    |
| Keep alive symbols a framework registers dynamically     | new `EdgePlugin` under `contrib/`              |
| Support a new project layout / lockfile                  | new `PathResolver` under `contrib/`            |
| Change how cross-file imports get classified             | `_edges.resolve_edges` + the resolver fallback |
| Change codemod output shape                              | `codemod.py` (`RemoveDeadSymbols` / `_rewrite_one`) |
| Change patch format / per-SCC patch slicing              | `codemod.generate_patch`                       |

See `CLAUDE.md` for the per-stage cache-invalidation discipline.
