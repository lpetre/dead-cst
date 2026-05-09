# Performance analysis: unreachable-region detection and flow tracking

Profiling artefact for the work on `claude/analyze-unreachable-regions-EdRNR`. Not user-facing; this is an engineering note. Numbers come from profiling `dead_cst` against itself (39 files, 7,895 module-level Name accesses, 514 scopes) on Python 3.13.12 with libcst 1.x.

## TL;DR

The current `DefaultUnreachableRegionDetector` is the dominant cost of `dead-cst`'s cold per-file pass. cProfile of `find_regions` over the 39 files of `dead_cst/`:

| phase                                     | cum time | share |
|-------------------------------------------|---------:|------:|
| `find_regions`                            | 103.1 s  | 100%  |
| └─ `fold_constants`                       |  97.5 s  |  95%  |
| &nbsp;&nbsp; └─ `live_referents`          |  82.2 s  |  80%  |
| &nbsp;&nbsp;&nbsp;&nbsp; └─ `_walk_flow`  |  82.1 s  |  80%  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └─ `_descendant_ids` |  81.4 s |  79%  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └─ libcst `.children` | 75.2 s | 73% |

Call counts per 39-file run: 4,334 `live_referents` invocations, 39,614 `_walk_flow` activations, 103,368 `_observe` callbacks, **3,974,073 `_descendant_ids` calls**, 20.5M `isinstance` checks. The hot path is `fold_constants` rebuilding a fresh `_descendant_ids` cache for every access, then walking the entire enclosing scope body for each one.

## Pipeline as it stands

`SymbolVisitor.visit_Module` runs the detector once per file inside the visitor loop (`_visitor.py:380`). `DefaultUnreachableRegionDetector.find_regions` is three passes (`branches.py:277-358`):

1. `fold_constants(wrapper, resolve_expr=self.resolve)` — fixpoint constant propagation over `Name = literal` assignments.
2. A `cst.CSTVisitor` (`_Collector`) walks every `If`/`While`/`IndentedBlock`/`Module`, calling `unreachable_suites` and `is_terminator`.
3. `evaluate_truthiness` is recursively dispatched against the fold table for each `if`/`while`/`assert` test.

`fold_constants` (`_const_fold.py:53`) drives the cost. Its inner loop, **per access**:

```python
body = _scope_body(referents[0].scope, module)
live_ids = {id(n) for n in live_referents(body, access.node, [r.node for r in referents])}
```

`live_referents` (`_flow.py:131`) walks the binding scope's statement list via `_walk_flow`, observing each statement's pre-state and asking `_descendant_ids(stmt) ∋ access_id?` to find the access's enclosing statements.

## Bottlenecks

### 1. `_descendant_ids` cache is per-call, not per-file (the big one)

`live_referents` allocates `cache: dict[int, set[int]] = {}` on every call (`_flow.py:143`). The same statement subtrees get re-traversed thousands of times across `fold_constants`'s fixpoint loop. 3.97M calls and 75s in libcst's `.children` for a 39-file project.

**Fix**: hoist the cache. Since the wrapper holds all CST nodes alive for the duration of `find_regions`, ids are stable. A single `dict[int, set[int]]` (or better, a precomputed `stmt_id -> frozenset[access_id]` map) shared across the file's `live_referents` calls collapses 3.97M calls to ~scope_count.

Even better: invert it. We don't actually need "what ids are inside stmt X"; we need "what stmt(s) in this body contain access Y". Building one `access_id -> innermost_stmt` map per scope body in O(scope_size) amortizes to O(1) lookup per access.

### 2. `fold_constants` runs even on files with no reachable use of the result

Every Name access in every scope is walked in the fixpoint loop, but only accesses that appear inside an `if`/`while`/`assert` test (or transitively feed an assignment that does) ever influence dead-region output. Of `dead_cst/`'s 7,895 accesses, the vast majority are RHS expressions inside function bodies that the `_Collector` pass never consults.

The pessimal cases are:

- 5/39 files in `dead_cst/` have **zero** `if`/`while`/`assert` and still pay full `fold_constants` cost (~10–20ms each). For larger projects with many `__init__.py` / data modules this is dominant overhead.
- A 1,000-line file with 100 functions × 10 accesses each but only 5 module-level `if` tests still pays `O(1,000 accesses × scope_walk)` instead of `O(5 + transitive bindings)`.

**Fix**: lazy / demand-driven fold.
- Pre-pass: collect the set of `Name` accesses that appear inside `if`/`while`/`assert` tests (or are RHS of an assignment whose target is itself in that set, transitively).
- Only fold those accesses.

For the typical case where most accesses don't feed conditionals, this turns the dominant `fold_constants × live_referents × _walk_flow` factor into something close to "number of test-Names plus their transitive assignments."

### 3. `live_referents` recomputes the result on every fixpoint iteration

```python
while True:
    progressed = False
    for access in accesses:
        if id(access.node) in truthy:
            continue
        ...
        live_ids = {id(n) for n in live_referents(body, access.node, [r.node for r in referents])}  # ← stable across iterations
        ...
```

`live_referents`' output depends only on `(body, access, referents)` — none of which change between fixpoint iterations. Yet the call repeats every loop. For accesses that never fold (most of them), this is K×N wasted work where K is the iteration count.

**Fix**: memoize per-access. Compute the live referent set once outside the loop; the loop only re-evaluates `evaluate_truthiness(rhs, resolve)` against an updated `truthy` table.

### 4. `_walk_flow` rebuilds `bindings_here` per statement

```python
bindings_here = _referents_in(stmt)
if bindings_here:
    state = bindings_here
```

`_referents_in` filters the *full* `referent_set` against the stmt's descendants for every stmt in every walk. For a body of N statements and R referents this is O(N × R) per walk, plus the `_descendant_ids` recursion.

**Fix**: precompute a `stmt -> {referents inside stmt}` map once before the walk. Lookup in O(1) per stmt.

### 5. `live_referents` over-approximates and silently weakens fold precision

When an access is nested inside an `if`/`for`/`while`/`try`, `_observe` is called *both* on the enclosing compound statement (with pre-block state) *and* on the inner statement holding the access (with the in-block state, which may include rebindings). The two states are then unioned (`live_referents` end of body):

```
x = 1            # ass1
if cond:
    x = 2        # ass2
    f(x)         # access — observed with state {ass1} from the outer if AND state {ass2} from the inner stmt
```

Result: `{ass1, ass2}` is returned as live, even though `ass2` strictly dominates `ass1` at the access. Visible behaviour:

- Visitor: extra (live, shadowed) edges in `on_leave`. Pessimistic but harmless for correctness.
- `fold_constants`: when the dominant binding has known truthiness but the shadowed one disagrees, `len(values) != 1` triggers and the access stays unfolded. **This hides a class of foldable cases.**

There's no test in `tests/test_flow_sensitive_filter.py` that pins the access-inside-shadowed-block case. The "within-branch-shadowing" test (`test_flow_sensitive_filter.py:302`) only checks the *post-block* access, where the union semantics happen to give the right answer.

**Fix**: keep only the innermost state (track the deepest observation, not the union). Improves both precision and a small slice of folding power.

### 6. `_Collector` is a separate full-tree walk

`_Collector` is its own `cst.CSTVisitor` traversal (`branches.py:334`). The main `SymbolVisitor` is already walking the same module milliseconds later. Two CST walks per file is unnecessary — but folding into the visitor would tangle responsibilities (the detector doesn't need a per-decl frame).

**Mitigations** (in order of effort):
- Cheap: make `find_regions` short-circuit when the tree contains no `If`/`While`/`Assert`/`Return`/`Raise`/`Break`/`Continue` (one-pass scan, then skip everything).
- Medium: keep `_Collector` but stop walking subtrees that are themselves wholly inside an already-identified dead region.
- Big: merge `_Collector` into `SymbolVisitor` (but the visitor's frame stack is already complex enough).

### 7. `evaluate_truthiness` recurses through the resolver chain

Every recursive descent into a `BooleanOperation` calls `resolve_expr` on each sub-expression before falling through to literal handling. For `if a or b or c or d:` and a custom `resolve()` that's a non-trivial isinstance ladder, this is `O(depth × ladder_cost)`. Within the default detector this is cheap (`resolve` returns `None` immediately), but the docstring's warning to subclasses ("guard with an early-return on the wrong type") understates how hot the call site is.

**Fix**: pass a precomputed truthiness *table* keyed by node id rather than a callback, and only consult the user's resolver lazily on cache miss. Lets `DefaultUnreachableRegionDetector` populate the table in one pass instead of dispatching per-recursion.

### 8. Metadata resolution overhead

ScopeProvider and ParentNodeProvider take ~3.9s combined across the 39 files (vs. ~0.5s for parsing). They're cached on the wrapper, so the second consumer is free. The visitor passes its wrapper into the detector (`_visitor.py:141, 380`), so they're computed once per file. ✅ already correct.

### 9. Bytes-level memory pressure

`_descendant_ids` returns `set[int]` containing every descendant id in a subtree. For a module with M nodes the per-stmt sets sum to roughly M × avg_depth (~5 in practice). For a 10k-node module that's ~50k ints. Per-call. Multiplied across `fold_constants`'s fixpoint loop, GC pressure is real (though not visible in the profile we ran).

**Fix**: same hoisting from #1; one cache per file means one allocation.

### 10. Process pool overhead on small files

`_refresh.process_stale_files` opts into `ProcessPoolExecutor` at `workers >= 2 AND tasks >= 2`. For every miss, the pool pickles `(detector, plugins)` once at startup and the per-task `StaleFile`. For the per-file work-units we have here (single-digit ms after the optimizations above), serialization could dominate. Worth re-benchmarking after #1–#3 land.

## Estimated wins

The optimizations stack mostly multiplicatively for the dominant path:

- **#1 (per-file `_descendant_ids` cache)**: collapses 3.97M calls to ~scope_count. Easily 5–10× on `fold_constants`.
- **#2 (lazy fold)**: cuts work proportional to `(test-relevant accesses) / (total accesses)`. For typical Python codebases this is well under 10%, so another 5–20× on top of #1.
- **#3 (memoize `live_referents`)**: removes the redundant K-iteration multiplier — typically K=2–4 in practice. ~2–4× on top of #1+#2.
- **#7 (table-based resolver)**: smaller, ~1.5×.

Realistic projection: **`find_regions` from 103s → 1–3s** for the 39-file self-analysis, with the metadata resolution (~3.9s) becoming the new floor unless we also short-circuit ScopeProvider for files with no relevant accesses.

## Risk surface

- **Cache invalidation discipline**: any of #1, #3, #5 changes the function's *output* under the same input. The detector's `version` field (`branches.py:261`) MUST be bumped to a fresh epoch when these land, or warm caches will serve stale `dead_suites`. The `Cacheable` contract makes this a one-line bump per change.
- **#5 changes observable behaviour** in a case that's currently silently pessimistic. Need a regression test for "access inside a same-name rebinding" (the example above) before flipping the union semantics.
- **#2's transitive closure** must be sound: an assignment whose RHS is a Call returning a side-effecting flag would be missed. The conservative guard is "fold transitively from the test set; if any binding's RHS contains a Name, recurse into that Name." Cycles already terminate (the existing `truthy` cache), so it's safe.

## What I'd ship first

Order of impact-vs-risk:

1. **#1 hoist `_descendant_ids` cache** — pure refactor, no behaviour change, no version bump. Biggest single win.
2. **#3 memoize `live_referents` per access** — same. No behaviour change.
3. **#2 lazy fold** — bigger win but introduces conditional logic; behaviour-equivalent if the closure is correctly computed. Worth a careful diff against the current full-fold output on a real codebase.
4. **#5 innermost-state semantics** — small behaviour change, needs new tests, version bump required.
5. **#7 table-based resolver** — API tweak (`resolve_expr` → table), nice ergonomics for subclassers.

Items #4, #6, #8, #9, #10 are smaller wins or already mostly handled.

## Measured outcome

Levers #1, #2, and #3 (and a piece of #7 — the resolver is now an object
instead of a callback) shipped together as the
:class:`TruthinessResolver` + two-pass detector restructure. Same
benchmark, same ``dead_cst/`` self-analysis, same ``total_regions``
output:

| metric                     | before    | after    | speedup |
|----------------------------|----------:|---------:|--------:|
| ``find_regions`` total     | 24,215 ms | 1,816 ms | **13.3×** |
| ``find_regions`` per file  |    621 ms |    48 ms | **13.0×** |
| total dead regions found   |         1 |        1 | (unchanged) |

cProfile of the post-refactor run no longer shows ``_descendant_ids``,
``_walk_flow``, or ``live_referents`` in the top 20. The new dominant
cost is libcst's ``PositionProvider`` codegen (~2 s under cProfile,
~1 s wall clock) — work both implementations have to do — followed by
the single ``_SiteCollector`` CST walk per file (~0.4 s).

The full pytest suite (775 tests) passes; one test had to be updated
because the new resolver correctly returns ``True`` for the ``True``
keyword, where the old ``fold_constants`` table simply omitted it.
