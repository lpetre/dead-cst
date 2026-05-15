# CLAUDE.md (crates/dead-cst-ty-native)

Scoped to the ty-backed native crate. Reads in addition to the
top-level `CLAUDE.md`. Where the two disagree, this file wins inside
this crate.

## Guiding principles

### 1. ty for everything; ruff only when ty hasn't exposed it yet

`ty_python_semantic` is the source of truth for every piece of Python
semantics: scope resolution, use-def chains, module resolution,
star-import expansion, stdlib / first-party / third-party
classification. Reach for `SemanticModel`, `definitions_for_name`,
`SemanticIndex`, the module resolver, and `KnownModule` before
writing anything that re-implements scoping rules. If a behavior
needs flow analysis, name binding, or "where does this name come
from" — that's ty's job, not ours.

`ruff_python_ast` / `ruff_python_parser` are used only for raw AST
access where ty hasn't surfaced the structure we need — today the
walks over `Stmt::{Import, ImportFrom, FunctionDef, ClassDef, Assign,
AnnAssign, For, With, Try, Match}` to enumerate decl sites. Whenever
ty grows an API that covers one of those walks (global-scope binding
enumeration via `SemanticIndex` is the obvious next candidate),
prefer it and delete the AST walk.

**No per-file cache.** The libcst pipeline has one because resolver
swaps would otherwise cost a full re-parse; here, ty's Salsa database
already gives us incremental re-analysis. Don't reintroduce a
parallel `VisitorPayload`-style cache, a per-file pickled blob, or
any other "snapshot the visitor output" layer. Trust the Salsa db.

### 2. Every import binds a local declaration

Every import materializes a `kind="import"` node in the *importing*
file. A use of an imported name always emits an edge to its local
alias — that's the codemod invariant: the codemod removes an import
node iff its in-edge count is zero, so the use must show up in those
in-edges. **It also makes "what does this file consume" a faithful
per-file query** without having to walk cross-file edges.

```
mod.f  ─uses→  mod.bar  ─aliases→  foo.bar
                (local)             (cross-file)
```

The alias edge is required; it is **not exclusive**. A single use
may carry additional parallel edges directly to the upstream module
and decl. These extra edges exist because static reachability needs
the granularity:

* For `import foo.bar; foo.bar.f()`, the local alias binds *only*
  the name `foo` — the runtime `foo` module object. Without parallel
  edges, the function `foo.bar.f` is never reached from the use, and
  reachability marks it dead. The use therefore emits `use → alias`
  **and** `use → foo.bar` **and** `use → foo.bar.f`.
* For `from foo import bar; bar()`, the use emits `use → alias`
  **and** `use → foo.bar` (the upstream decl) **and** `use → foo`
  (the upstream module).

Concrete contract:

* **Codemod invariant.** Every use of an imported name emits exactly
  one edge to the local alias. An unused import has zero in-edges
  and is safe to drop.
* **Reachability edges (parallel).** When ty's module resolver /
  global-scope lookup can pin the use to specific upstream targets,
  the use also emits direct edges to each of them. These edges
  bypass the alias for reachability purposes but do not replace it.
* **Attribute chains on aliased modules.** A chain `name.seg1.seg2`
  rooted at an aliased module is walked segment by segment. Each
  segment that resolves as a submodule or as a global-scope decl
  emits a parallel `use → target` edge. The walk stops on the first
  segment that resolves as neither (or at the chain's end).

This applies equally to:

* **Explicit** imports — `from foo import bar`, `import a.b`,
  `from a.b import c as d`. One local node per alias.
* **Implicit** imports — every exported name brought in by
  `from foo import *`. Enumerate ty's `StarImport` bindings in the
  importing file's global scope and mint one local node per imported
  name. The libcst pipeline labels these `NodeFlags.STAR_REEXPORT`;
  in the rust path the `kind == "import"` plus a missing source
  position (it's synthetic) is enough.

### 3. Shadowed declarations are first-class graph nodes

`def f` followed by another `def f` mints two nodes, distinguished by
`target_range`. ty's use-def chain resolves each use site to the
reaching definition; in-edges accumulate on the live one, the
shadowed one stays in the graph with no in-edges. The codemod can
then drop the shadowed copy without touching the live one — and the
"useless redefinition" detector finds dead shadows even when the
live decl is itself reachable.

Concrete rules:

* Never dedup decls by FQN. Only by `(fqname, kind, path, position)`
  — which is what `NodeKey` already enforces. Two `def f` at
  different lines stay distinct.
* Never compute "the" binding for a name. Ask ty at each use site
  via `definitions_for_name`; let its flow-sensitive use-def chain
  pick the reaching def. A use that has two reaching defs (e.g.
  `try: import X; except: import Y as X`) produces edges to both.
* The same rule applies to shadowed *imports* (`import X; import X`
  rebinds, `try/except` rebinds, conditional rebinds): both nodes
  exist, only the live ones accumulate uses.

A dead `def f` followed by a live-but-unused `def f` should produce
**two** unreachable nodes, not one. That distinction is what lets the
codemod report "useless redefinition" separately from "unused
function."
