# Investigation: how ty resolves names in two import-reference scenarios

You are investigating **ty** (the Rust type checker in this repo's vendored
fork of ruff, under `vendor/ruff/crates/`). The goal is to determine whether
ty's name-resolution / use-def behavior in the two scenarios below is
**correct and intended**, and if not, whether there is a bug or a reasonable
fix/API addition in the ty source.

A downstream consumer queries ty's semantic model to find, for a given name
use, the set of definitions that reach it. It does this through ty's
public-ish semantic APIs (all in `ty_python_semantic`):

- `SemanticModel::scope(name)` — the scope a name reference lives in.
- `SemanticIndex::visible_ancestor_scopes(scope)` — scope chain to walk.
- `SemanticIndex::place_table(scope)` + `PlaceTable::symbol_id(name)`.
- `SemanticIndex::use_def_map(scope)`.
- `ExprName::scoped_use_id(db, scope)` then
  `UseDefMap::bindings_at_use(use_id)` — the **flow-sensitive reaching
  bindings** at that specific use position.
- `UseDefMap::end_of_scope_symbol_bindings(symbol_id)` — bindings live at end
  of scope (used as a fallback for enclosing scopes).
- `Binding::definition()` to get the `Definition` behind each binding.

Start by reading `vendor/ruff/crates/ty_python_semantic/src/semantic_index/`
(especially `use_def.rs`, the builder, and the narrowing/visibility-constraint
code). The two scenarios below each raise a question about what
`bindings_at_use` returns and why.

The consumer needs the **runtime-reachable** set of definitions for a name:
every definition that could be the binding in effect when the code actually
runs. That is subtly different from the **type-checker's flow-narrowed** view,
and the two scenarios probe exactly where they diverge. Part of the task is to
decide, for each, whether ty's current answer is right for a type checker, and
whether ty exposes (or could cheaply expose) the runtime/union view.

---

## Scenario A — two `import a.X` statements binding the same root name

```python
import a.foo
import a.bar
a.foo.x()
a.bar.z()
```

(`a/__init__.py` empty; `a/foo.py` defines `x`; `a/bar.py` defines `z`.)

Both `import a.foo` and `import a.bar` bind the **same** local name `a`. After
both execute, the module object `a` has both submodules attached, so both
`a.foo` and `a.bar` are valid at runtime.

**Observed:** at the use of `a` in `a.foo.x()` (and again in `a.bar.z()`),
`bindings_at_use` returns only the binding from the **last** statement
(`import a.bar`) — straight-line flow treats the second `import a.*` as
overwriting the first's binding of `a`. The definition introduced by
`import a.foo` is not reported as reaching either use.

**Questions to answer from the ty source:**

1. How does ty model an `import a.foo` statement's binding of the name `a` —
   as an ordinary store that kills the previous binding of `a`, or as
   something submodule-aware? Find the definition/binding construction for
   dotted `import` statements (look for how `Stmt::Import` with a dotted name
   is lowered into definitions and place bindings).
2. Is "the last `import a.*` overwrites the earlier one's binding of `a`"
   correct for a type checker? Consider that `a.foo` and `a.bar` are both
   still valid member accesses afterwards. Does ty type-check `a.foo.x()`
   correctly *despite* the name `a` resolving only to the `import a.bar`
   statement (i.e. is submodule member resolution independent of which
   statement bound `a`)?
3. Is there any existing mechanism (or a sensible place to add one) that links
   a submodule member access `a.foo` back to the specific `import a.foo`
   statement that made that submodule available — e.g. via
   `definitions_for_name`, a submodule-import definition kind, or the module
   resolver? The consumer wants to know *which import statement* a given
   submodule access depends on.

Frame your conclusion as: is this a ty bug, intended type-checker behavior
with a missing query, or correct-as-is (and the consumer should derive the
link itself)?

---

## Scenario B — `if TYPE_CHECKING / else` binding the same name in both branches

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a import SomeClass
else:
    from b import SomeClass
SomeClass()
```

(`a.py` and `b.py` each define `SomeClass`.)

**Observed:** at the use `SomeClass()`, `bindings_at_use` returns only the
**if-branch** binding (`from a import SomeClass`). The **else-branch** binding
(`from b import SomeClass`) is not reported as reaching the use. ty appears to
narrow `TYPE_CHECKING` to statically `True`, making the `else` branch
unreachable from the type-checker's perspective.

**Control:** replacing the guard with a genuine runtime condition makes ty
return **both** bindings:

```python
import random
if random.random() > 0.5:
    from a import SomeClass
else:
    from b import SomeClass
SomeClass()   # bindings_at_use returns BOTH the `a` and `b` imports
```

So the divergence is specifically caused by ty's `TYPE_CHECKING` handling, not
by branch/visibility logic in general.

**Questions to answer from the ty source:**

1. Where and how does ty treat `typing.TYPE_CHECKING` (and
   `typing.TYPE_CHECKING`-equivalent constants) as statically true? Find the
   constant evaluation / reachability-constraint code (search the crates for
   `TYPE_CHECKING`). Is the `else` branch marked statically unreachable, or
   just narrowed to a lower-priority binding?
2. Is treating `TYPE_CHECKING` as `True` the intended and correct behavior for
   ty as a type checker? (It very likely is — that is the whole point of the
   constant.) Confirm this is deliberate and document where.
3. Does ty expose any way to recover the **runtime** picture — the union of
   bindings across both branches as if `TYPE_CHECKING` were not narrowed?
   E.g. an unnarrowed binding iterator, a way to enumerate definitions for a
   symbol ignoring reachability constraints, or per-branch scope bindings the
   consumer could union itself. If not, assess how hard it would be to add
   such a query without disturbing normal type-checking.

Frame your conclusion as: confirm ty's narrowing is correct for type-checking,
then report what API (if any) lets a consumer obtain the runtime-union of
bindings, or what minimal addition to ty would provide it.

---

## Deliverable

For each scenario, report:

- The exact ty code path that produces the observed `bindings_at_use` result
  (file + function in `vendor/ruff/crates/...`).
- Whether ty's behavior is a bug or correct-as-intended for a type checker.
- If correct-as-intended: what existing ty API yields the runtime/union view
  the consumer wants, or the smallest change/addition that would.
- If a bug: a concrete description of the defect and a proposed fix.

You can verify observed behavior by writing a small Rust test against the ty
crates, or by exercising ty's semantic model directly on the snippets above.
