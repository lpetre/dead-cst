# Briefing: two ty-flow bugs in dead-cst's reference resolution

## Project context

`dead-cst` builds a symbol-level reachability graph of a Python codebase to
find/remove dead code. It's rust-backed (`src/`, a crate using a vendored fork
of ty/ruff at `vendor/ruff`) with a Python wrapper (`python/dead_cst/`). Read
`CLAUDE.md` and `src/CLAUDE.md` first — the latter documents the edge-model
invariants.

**Key invariant (codemod safety):** every use of an imported name emits an edge
to its local `kind="import"` alias node. *An unused import has zero in-edges and
is safe to delete.* So if a use site fails to emit an alias edge to an import
that's actually needed, the codemod will delete that import and break the code
at runtime. That's the failure mode for both bugs below.

## Build / test loop

```bash
git submodule update --init --recursive   # vendored ruff fork
uv sync                                    # compiles src/ via maturin
# after editing src/*.rs, rebuild the extension:
uv sync --reinstall-package dead-cst       # ~1 min rust build
uv run pytest tests/test_limitations.py -k positional   # the two failing-when-fixed tests
```

Both bugs are already pinned as limitation tests in
`tests/test_limitations.py::test_limitation_positional` (they assert the
*current buggy* behavior with a comment describing the ideal; they flip to
failing when fixed). IDs: `sibling-submodule-imports-share-root-binding` and
`type-checking-else-branch-import-dropped`.

## Shared root cause

Reference resolution lives in `src/file_ref_edges.rs`:

- `find_local_bindings` (~line 262) resolves a use's name to its local
  binding(s) via ty's use-def map: `bindings_at_use(use_id)` for the use's own
  scope (~line 299), `end_of_scope_symbol_bindings` for enclosing scopes
  (~line 301).
- `emit_name_use` (~line 360) emits the alias edge to each resolved binding,
  plus parallel upstream edges via `emit_upstream` (~line 678).

`bindings_at_use` returns ty's **flow-sensitive reaching binding(s)** — the
type-checker's view. dead-cst instead needs the **runtime view**: every import
statement that contributes the accessed name/submodule must keep an in-edge.
The two bugs are two ways this diverges.

## Bug A — sibling submodule imports sharing a root binding

```python
import a.foo
import a.bar
a.foo.x()
a.bar.z()
```

(scaffold: `a/__init__.py`, `a/foo.py` defines `x`, `a/bar.py` defines `z`)

Both `import a.foo` and `import a.bar` bind the local name `a`. In straight-line
code ty's use-def map says the reaching binding of `a` at *both* use sites is
the **last** one (`import a.bar`). So:

- `mod.a@2:7` (import a.bar) gets both alias in-edges.
- `mod.a@1:7` (import a.foo) gets **zero in-edges** → codemod deletes
  `import a.foo` → `a.foo.x()` breaks (importing `a.bar` does not import the
  `a.foo` submodule).

The *upstream* parallel edges are already correct (`emit_upstream` walks the
access chain `.foo`/`.bar` and emits `mod → a.foo`, `mod → a.foo.x`,
`mod → a.bar`, `mod → a.bar.z`). Only the **alias** edge lands on the wrong
statement.

**Ideal:** `a.foo.x()` should also emit `mod → mod.a@1:7`. Likely fix: when the
use has an attribute chain (`a.foo.x`), don't rely only on the single reaching
binding — inspect all module-scope `import a.*` bindings and emit an alias edge
to the one whose dotted submodule path matches the chain prefix (`.foo` → the
`import a.foo` statement).

### Current (buggy) positional edges

```
"a.bar -> a",
"a.bar.z@1:0 -> a.bar",
"a.foo -> a",
"a.foo.x@1:0 -> a.foo",
"mod -> a.bar",
"mod -> a.bar.z@1:0",
"mod -> a.foo",
"mod -> a.foo.x@1:0",
"mod -> mod.a@2:7",
"mod.a@1:7 -> a.foo",
"mod.a@1:7 -> mod",
"mod.a@2:7 -> a.bar",
"mod.a@2:7 -> mod",
```

Missing (the fix should add): `"mod -> mod.a@1:7"`.

## Bug B — `if TYPE_CHECKING / else` drops the live branch

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from a import SomeClass
else:
    from b import SomeClass
SomeClass()
```

(scaffold: `a.py` and `b.py` each define `SomeClass`)

ty narrows `TYPE_CHECKING` to `True`, so its use-def map says only the
**if-branch** import (`from a`, `mod.SomeClass@3:18`) reaches `SomeClass()`. The
**else-branch** import (`from b`, `mod.SomeClass@5:18`) gets **zero in-edges**.
But at *runtime* `TYPE_CHECKING` is `False`, so the else branch is the live one
— the codemod would delete the import that actually runs.

**Control proving it's TYPE_CHECKING-specific:** replace the guard with a
runtime condition (`if random() > 0.5:` / `else:`) and ty returns *both*
reaching bindings correctly — the use links both. So the divergence is entirely
ty's TYPE_CHECKING narrowing.

**Ideal:** the use should reach **both** bindings: `mod → mod.SomeClass@3:18`,
`mod → mod.SomeClass@5:18`, plus upstreams to both `a.SomeClass` and
`b.SomeClass`. Likely fix: detect `if TYPE_CHECKING` blocks and treat both
branches' bindings as reaching (union the if- and else-branch bindings), since
dead-cst wants runtime-union semantics, not type-checker narrowing. Investigate
whether ty exposes the unnarrowed bindings, or whether dead-cst must collect
bindings from both branch scopes itself.

### Current (buggy) positional edges

```
"a.SomeClass@1:0 -> a",
"b.SomeClass@1:0 -> b",
"mod -> a",
"mod -> a.SomeClass@1:0",
"mod -> mod.SomeClass@3:18",
"mod -> mod.TYPE_CHECKING@1:19",
"mod.SomeClass@3:18 -> a",
"mod.SomeClass@3:18 -> a.SomeClass@1:0",
"mod.SomeClass@3:18 -> mod",
"mod.SomeClass@5:18 -> b",
"mod.SomeClass@5:18 -> b.SomeClass@1:0",
"mod.SomeClass@5:18 -> mod",
"mod.TYPE_CHECKING@1:19 -> mod",
```

Missing (the fix should add): `"mod -> mod.SomeClass@5:18"`, `"mod -> b"`,
`"mod -> b.SomeClass@1:0"`.

## What's already been done

A *third*, unrelated bug in the same area (subscript/slice-assignment targets
like `os.environ["k"]=v` dropping all uses) was already root-caused and fixed in
`src/ingest.rs` (`stmt_creates_top_level_definition`) — that's a separate
AST-walk issue, not a ty-flow issue, so don't conflate it. The two bugs above
are specifically about ty's use-def chain returning a single/narrowed reaching
binding where dead-cst needs the runtime union.
