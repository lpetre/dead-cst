# dead-cst Roadmap

A stack-ranked plan for moving `dead-cst` from alpha to a tool maintainers
trust enough to wire into CI. Ordering reflects leverage, not effort.

The central question driving this ordering: **what makes a maintainer trust a
dead-code tool enough to run it on their codebase?** The architecture is in
good shape; the gap to wide adoption is (a) trust that it won't flag legitimate
code and (b) trust that the codemod won't break files. Tier 1 buys both.

---

## Tier 1 — Trust and correctness

### 1. Framework-aware plugin presets

The existential risk for any dead-code tool is "I tried it and it flagged half
my codebase." `dead_cst/_plugins.py` already exposes the right protocol; ship
opt-in presets for the common offenders:

- pytest fixtures, marks, and `conftest.py` discovery
- Click / Typer commands
- FastAPI routes and dependencies
- Django URLConf, admin registration, signal handlers
- Pydantic validators and field serializers
- `__init_subclass__`, `__set_name__`, dataclass `__post_init__`

Surface as `--preset pytest,fastapi` and via entry points so third parties can
publish their own. Highest-leverage adoption work.

### 2. Codemod test coverage

`dead_cst/_codemod.py` (81 LOC) modifies user files and is currently only
exercised end-to-end. Add a focused test file covering:

- Trailing-comma cleanup after partial removal
- Decorator stripping
- Leading / trailing blank-line normalization
- Multi-line `import` and `from … import a, b, c` partial removal
- Class body collapse when the last member is removed

This is the highest-stakes module per LOC in the package.

### 3. CLI integration tests

`dead_cst/cli.py` is 443 LOC and untested. Cover the user-facing surface with
`typer.testing.CliRunner`:

- `analyze` text and JSON output
- `remove` with `--dry-run` and actual writes
- `why-alive` predecessor tracing
- Exit codes and stderr on parse errors

Cheap to write, catches the regressions users notice first.

### 4. Resolve `from X import *`

Tracked as a FIXME at `dead_cst/_visitor.py:334`. Star imports are silently
skipped today, which causes false positives in any codebase that uses them.
The symbol trie in `dead_cst/_symbols.py` already has the data needed —
extend `dead_cst/_resolve.py` to expand stars at resolution time.

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

### 7. `if TYPE_CHECKING:` awareness

Currently treated as live, which is safe but pessimistic. Tag imports inside
`TYPE_CHECKING` blocks as type-only so removal is safe when the only
references are themselves in type-only contexts (annotations, other
`TYPE_CHECKING` blocks). Low effort, visible win for typed codebases.

---

## Tier 3 — Polish and ecosystem

### 8. Read the Docs site with plugin/resolver tutorials

The `EdgePlugin` and `PathResolver` protocols are well-designed but
undiscovered. A short Sphinx site with one tutorial each ("write a custom
`EdgePlugin`", "write a custom `PathResolver`") activates the extensibility
that's already built. Replaces the existing single TODO line in `README.md`.

### 9. Coverage tracking + parser-error recovery

Pair these:

- Add codecov (or coveralls) to `.github/workflows/ci.yml` to establish a
  baseline and prevent silent regressions.
- Add `--continue-on-parse-error` so a single broken file doesn't abort the
  whole analysis.

Both are small; both pay off long-term.

### 10. PEP 695 `type` statements and `del` modeling

Documented limitations in `tests/test_limitations.py`. Low real-world impact;
tackle once Tier 1–2 has shipped and the protocol surface is stable.

---

## Tier 4 — Speculative, wait for signal

### 11. Incremental analysis

A real performance problem at 100k+ LOC, but the project isn't there yet. Adds
substantial complexity (file-content hashing, partial graph rebuild,
invalidation). Defer until a user reports it.

### 12. Multiple reachability frontiers

Splitting "reachable from tests" vs. "reachable from production entrypoints"
is interesting and would enable rules like "no production code reachable only
from tests." But it's solving a problem nobody has reported yet. Wait for
demand.

---

## Out of scope (for now)

- Per-symbol weighting / decay heuristics
- Stub file (`.pyi`) ingestion
- Cross-language analysis
- Language Server Protocol implementation

These may be worth revisiting after a stable 1.0, but each adds significant
surface area without addressing current adoption blockers.
