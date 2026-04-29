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

## Next release — Quality of life

Two committed items for the upcoming version. Both are internal-architecture
changes that unblock work elsewhere in the roadmap.

### Resolver logic as a protocol

`PathResolver` already abstracts how source roots are discovered, but the
edge-resolution layer in `_resolve.py` (`resolve_import`, `resolve_edges`,
`safe_resolve_module`) is still a flat module of functions hard-coded to
`sys.path` and `importlib`. Lift it behind an `EdgeResolver` protocol so the
analyzer can swap strategies — sys.path-based, project-vendored, stub-aware,
or test fakes — the same way `EdgePlugin` and `PathResolver` already work.
Concrete wins:

- Testability: `resolve_edges` is currently exercised only end-to-end; a
  protocol lets us inject deterministic fakes.
- Extensibility: third parties can publish resolvers for monorepo layouts
  (Bazel, Pants) without patching internals.
- Sets up the partial-rebuild work below, which needs a stable interface to
  invalidate against.

### SQLite-cached graph with partial rebuilds

Cache the symbol graph in a SQLite database keyed by file content hash, and
rebuild only the modules whose hashes changed (plus their reverse
dependencies). This is the "Incremental analysis" item previously parked in
Tier 4 — promoting it because the resolver-protocol work makes the
invalidation surface tractable, and because cold-start latency is the
remaining friction users hit before CI integration.

Scope for the first cut:

- Schema for nodes, edges, and per-file content hashes; bump on schema
  changes.
- Invalidate a file when its hash changes; recompute its declared symbols
  and outgoing edges; recompute incoming edges from any file that imports
  it.
- Fall back to a full rebuild when the resolver, plugin set, or workspace
  layout changes (cache key includes those).
- `--no-cache` escape hatch and a `dead-cst cache clear` subcommand.

---

## Tier 1 — Trust and correctness

### 1. CLI integration tests

`dead_cst/cli.py` is now 571 LOC and still untested. It is the largest
untested surface in the package and the most user-visible. Cover it with
`typer.testing.CliRunner`:

- `analyze` text and JSON output (including unreachable branches and the
  `--resolver` / `--plugin` flags)
- `remove` with `--dry-run` and actual writes, including import pruning
- `why-alive` predecessor tracing
- `unused-exports` and `dependencies` output
- Exit codes and stderr on parse errors and missing inputs

Cheap to write, catches the regressions users notice first. Highest-leverage
trust work remaining.

### 2. Finish the framework-aware plugin presets

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

### 3. `if TYPE_CHECKING:` awareness

Currently treated as live, which is safe but pessimistic. Tag imports inside
`TYPE_CHECKING` blocks as type-only so removal is safe when the only
references are themselves in type-only contexts (annotations, other
`TYPE_CHECKING` blocks). The flow-sensitive filter (`_flow.py`) already
distinguishes branches; this is mostly an edge-tagging change in
`_resolve.py`. Low effort, visible win for typed codebases.

### 4. `--continue-on-parse-error`

A single broken file currently aborts the whole analysis, which is a
non-starter for running in CI on large repos. Skip the file, log it, mark its
declared symbols as live conservatively, and exit non-zero only at the end.

---

## Tier 2 — Adoption surface

### 5. "New dead code only" / diff mode

Most teams cannot fix all existing dead code in one PR. A `--since <ref>` flag
that filters to symbols introduced or last touched after a git ref makes the
tool drop-in for CI on existing codebases without a big-bang cleanup. This
pattern is what drove adoption for tools like Knip in the JS ecosystem.

### 6. Public reachability-explanation API

The `why-alive` predecessor walk is locked inside CLI code. Lift it to a
top-level `explain_reachability(graph, symbol)` that returns the path from an
entrypoint to the symbol. Enables IDE plugins, custom dashboards, and richer
error messages without re-implementing the BFS.

### 7. Coverage tracking in CI

Add codecov (or coveralls) to `.github/workflows/ci.yml` to establish a
baseline and prevent silent regressions. Small, but the value compounds once
Tier 1 lands and we want to defend the new test coverage.

---

## Tier 3 — Polish and ecosystem

### 8. Read the Docs site with plugin/resolver tutorials

The `EdgePlugin` and `PathResolver` protocols are well-designed but
undiscovered. A short Sphinx site with one tutorial each ("write a custom
`EdgePlugin`", "write a custom `PathResolver`") activates the extensibility
that's already built. The docstring pass already in place gives the API
reference for free.

### 9. PEP 695 `type` statements and `del` modeling

Documented limitations in `tests/test_limitations.py`. Low real-world impact;
tackle once Tier 1–2 has shipped and the protocol surface is stable.

---

## Tier 4 — Speculative, wait for signal

### 10. Multiple reachability frontiers

Splitting "reachable from tests" vs. "reachable from production entrypoints"
is interesting and would enable rules like "no production code reachable only
from tests." But it's solving a problem nobody has reported yet. Wait for
demand.

---

## Recently shipped

Folded down from earlier tiers as they landed:

- Codemod test coverage and import pruning (Tier 1).
- `from X import *` resolution, pessimistic by default (Tier 1).
- `PytestPlugin`, `UnittestPlugin`, `FastAPIPlugin`, `FlaskPlugin`,
  `TyperPlugin`, `ClickPlugin`, and `InitSubclassPlugin` (Tier 1,
  partial — see item 2).
- `unused-exports` and `dependencies` CLI commands.
- Unreachable-branch detection surfaced as synthetic graph nodes.
- Workspace-aware cross-member import scoping via `exported_roots`.
- Position-aware shadowing in the codemod.
- `ModuleDundersPlugin` replacing `--preserve-dunder-all`.
- Public-API docstring pass across the package.

---

## Out of scope (for now)

- Per-symbol weighting / decay heuristics
- Stub file (`.pyi`) ingestion
- Cross-language analysis
- Language Server Protocol implementation

These may be worth revisiting after a stable 1.0, but each adds significant
surface area without addressing current adoption blockers.
