# Architecture / program flow

This doc walks the code path a single `dead-cst` invocation takes, from
CLI arguments to a written patch. It's the developer-facing companion
to [`README.md`](README.md) (user-facing) and [`CLAUDE.md`](CLAUDE.md)
(LLM-oriented summary). Read this before adding a plugin, resolver, or
touching the rust crate.

## At a glance

```
   CLI / Python API
        │
        ▼
┌──────────────────┐
│   PathResolver   │  resolve(project_root) -> tuple[Package, ...]
└─────────┬────────┘
          │ Packages (path, name, deps)
          ▼
┌──────────────────┐
│     Analysis     │  cheap construction; everything below is lazy
└─────────┬────────┘
          │ materialize_all()
          ▼
┌──────────────────────────────────────────────────────────────┐
│  rust crate: native.ProjectContext.materialize()             │
│                                                              │
│   Phase 1 — decls       ty SemanticIndex + ruff AST          │
│   Phase 2 — chain       parent-module + import edges         │
│   Phase 3 — references  ty use-def chains for every Name     │
│   Plugins               NativePlugin ops → Node/Edge/Entry   │
└────────────────────────────┬─────────────────────────────────┘
                             │ live ProjectContext
                             ▼
              ┌──────────────┴──────────────┐
              ▼                             ▼
       Analysis queries              codemod (libcst)
       (reachable / dead /           remove_code,
        descendants / ancestors      generate_patch
        — all run in rust)
              │
              ▼
       graph persistence
       write_graph / read_graph
       (bincode)
```

The entire graph build runs inside the rust extension. There is no
per-file Python cache; ty's Salsa database already provides incremental
re-analysis. The only stage still in Python is the LibCST codemod, and
the only stage still in pure Python is `Analysis`'s thin wrapper around
the rust BFS queries.

## The rust crate

The native code is split in two:

- **`dead-cst-runtime`** (`runtime/`) — the whole implementation: the build
  pipeline, the query surface, the pyclasses, the native-plugin API. Built as
  **both** an `rlib` and a `dylib`.
- **`dead-cst-native`** (`src/lib.rs`) — a thin pyo3 `#[pymodule]` shim that
  ships as `python/dead_cst/_native.{abi3.so,pyd}`.

The default wheel links the runtime `rlib` *statically* into the shim, so it's
a single self-contained extension (~10 MB) that needs no rust at install time.
The runtime `dylib` is what **external native plugins** link against to share
one salsa/ty instance — see [`NATIVE_PLUGINS.md`](NATIVE_PLUGINS.md). Type
stubs live at `python/dead_cst/_native.pyi`.

Module layout (`runtime/src/lib.rs`'s `register()` registers the pymodule; the
`src/lib.rs` shim forwards to it):

| File | Purpose |
| ---- | ------- |
| `src/lib.rs`                    | the cdylib shim: pyo3 `#[pymodule]` forwarding to `runtime::register` |
| `runtime/src/lib.rs`            | `register()` — the pyo3 module registration |
| `runtime/src/project.rs`        | `Project`, `ProjectContext`, the `build()` pipeline |
| `runtime/src/builder.rs`        | `NodeKey`, `GraphBuilder`, `PreparedOp` (`Node` / `Edge` / `Entrypoint`), generic BFS |
| `runtime/src/graph.rs`          | `SymbolNode`, `Import`, `NativeGraph`, `NodeFlags`, `EdgeFlags` |
| `runtime/src/ingest.rs`         | the three build phases (decls / chain / references) |
| `runtime/src/query.rs`          | shared per-file scan helpers (identifier prefilter, path-regex, `par_scan_files`) |
| `runtime/src/native_plugins.rs` | in-tree + external native plugins (the `plugin_api`, the ABI airlock) |
| `runtime/src/helpers.rs`        | shared utilities (noqa parser, notebook decoder, dist-info lookup, …) |
| `runtime/src/io.rs`             | `write_graph` / `read_graph` (bincode + a hard-versioned header) |

The deeper crate-level rules live in `runtime/src/CLAUDE.md` — ty is the source
of truth for every piece of Python semantics, every import binds a
local declaration, and shadowed declarations are first-class graph
nodes.

## The pipeline, top-down

### 1. Path resolution — `dead_cst/resolvers/`

A `PathResolver` answers one question: what packages exist in this
project, and which other packages does each depend on?

```python
resolve(project_root) -> tuple[Package, ...]
```

Each `Package` carries `path`, `name`, and `deps: tuple[str, ...]`
(referencing other packages by name). Builtins:

* `ManualResolver` (`dead_cst/resolvers/manual.py`) — explicit
  `package:dep1,dep2` specs from the CLI's `-p` flag.
* `UvResolver` (`dead_cst/contrib/uv.py`) — parses `uv.lock` to
  discover workspace members and inter-member edges.

`Analysis` takes exactly one resolver — no chain. CLI flags `-p` and
`--resolver` are mutually exclusive. Construction validates the
resolver's output (name uniqueness, dep references resolve) via
`_validate_packages`.

### 2. `Analysis` — `dead_cst/analyze.py`

Cheap to construct: the resolver runs once at `__init__` and packages
are sorted into a dep-first BFS order, but nothing else happens until
you ask a question (`materialize_all`, `reachable`, `dead`,
`descendants`, `ancestors`, `kept_alive_by_flags_only`,
`kept_alive_by_dead_branches`).

`Analysis._ctx` holds the live `native.ProjectContext` after the first
`materialize_all()` call, so subsequent BFS queries run on the already-
built graph — one FFI hop per query, no rebuild.

`Analysis.re_materialize(events)` rebuilds the graph against the
same `ProjectContext` without tearing down the salsa db. The caller
supplies the change set: typically `ctx.detect_changes()` (which
today returns a single `ChangeEvent.rescan()`), or an explicit
`list[native.ChangeEvent]` built via the `.changed(path)` /
`.created(path)` / `.deleted(path)` / `.rescan()` classmethods for
LSP / file-watcher integrations. `ctx.apply_changes(events)` forwards
into ty's `ProjectDatabase::apply_changes`, which bumps file
revisions only when mtime / size differ, registers `Created` paths so
new files are visible on the next walk, drops `Deleted` paths, and
triggers a full rescan + project reload for the `Rescan` sentinel.
Cross-file importers invalidate transitively through salsa's
auto-tracked `file_to_nodes` reads. Plugin `prepare(project_root)` is
one-shot (owned by `materialize_all`); `re_materialize` only
re-drives the build + plugin pass.

### 3. Graph materialization — `native.ProjectContext.materialize()`

Driven from Python by `Analysis.materialize_all()`:

```python
ctx = native.ProjectContext(project_root, src_roots=[pkg.path, ...])
for plugin in self._plugins:
    ctx.add_plugin(plugin)
ctx.materialize()
```

Each package's path is registered as a `src_root` so the rust backend
mounts files at the right module FQN (`pkg_a/A/__init__.py` → `A`, not
`pkg_a.A`). Inside `materialize()` the rust pipeline runs in three
phases (see `src/lib.rs` for the canonical comment):

**Phase 1 — decls.** For every project file, iterate every binding in
the file's global scope via `UseDefMap::all_definitions_with_usage`,
minting a `SymbolNode` per binding (including each name brought in by
`from foo import *`). Each node lands in a global
`(File, target_range) → node_idx` index so cross-file edges can find
it later. Shadowed declarations are first-class — two `def f` at
different lines stay as distinct nodes, distinguished by their
positional `NodeKey`.

**Phase 2 — chain.** For every module node, emit the parent-module
edge so `__init__.py` stays alive as long as anything in the package
does. For every import-kind binding, resolve the upstream target via
`ty_module_resolver::resolve_module` and emit `alias_node →
upstream_node`. Targets outside the project (stdlib, site-packages)
get lazily minted module-only nodes under the
`[stdlib] X` / `[external dist] X` / `[external file] X` /
`[unresolved] X` synthetic prefixes.

**Phase 3 — references.** For every `Definition` that owns an
expression (function body, class body, assignment RHS, annotation),
walk the contained `Name`s and resolve each to its reaching def via
ty's `visible_ancestor_scopes` + `end_of_scope_symbol_bindings`. The
local alias is the target, not the upstream definition. Module-level
non-definition statements attribute to the module node itself.
Statically-dead regions identified by ty's reachability constraints
get their outgoing edges flagged `EdgeFlags.DEAD_BRANCH`.

### 4. Plugins — `runtime/src/native_plugins.rs`

After the three phases, the registered plugins contribute graph ops.
Every plugin is a native (Rust) `NativePlugin`; there is no Python
`Plugin` protocol. Project-wide plugins fan out across a GIL-free
`rayon` scope, each running against a `FrozenView` (a `Send` snapshot
of the build outputs) and collecting `PreparedOp` rows (`Node` / `Edge`
/ `Entrypoint`); per-file plugins are salsa-cached during the build. The
rust apply pass folds every plugin's ops into the graph atomically
(`apply_prepared_batch`), in registration order, before `materialize()`
returns. There is no two-phase observe/finalize split — one collect, one
apply.

A plugin builds against the query surface on `native.ProjectContext`:
the `*_indices` queries (`find_declarations_indices`,
`module_surface_indices`, `decls_matching_indices`, `indices_where`, …)
return positional indices into `ctx.nodes()`, materialized in bulk via
`ctx.nodes_at(idxs)` / `ctx.node_attrs(idxs)`. Direct accessors
(`find_module_idx`, `module_for_indices`, `find_main_blocks_indices`,
`find_factory_decls`, …) cover the rest. The decorator / construction /
call walks the dispatch-app plugins drive are rust-internal. See
`python/dead_cst/_native.pyi` for the full surface.

Every built-in is resolved through `native._builtin_native_plugin`
(consulted first by the CLI's `_load_plugin`, which otherwise falls
through to the `dead_cst.plugins` entry-point group). The dispatch-app
frameworks (Flask / FastAPI / Typer / Cyclopts / Slack Bolt / FastMCP /
Celery), Click, pytest, mock_patch, discordpy, and the rest are all
`*PluginImpl`s in `runtime/src/native_plugins.rs` (see `ClickPluginImpl`,
`DispatchAppPluginImpl`, and friends). Out-of-tree plugins are external
native plugins compiled against the runtime dylib (see
[`NATIVE_PLUGINS.md`](NATIVE_PLUGINS.md)).

### 5. Reachability — `Analysis.reachable` and friends

BFS from every node whose flags overlap `seed_flags`. Default seed
mask is `KEEPALIVE_DEFAULT = ENTRYPOINT | TESTCASE | NOQA | NOTEBOOK`.
Default traversal **does** follow `DEAD_BRANCH` edges (preserving
historical behavior). Three queries cover the standard slicing:

* `Analysis.dead(seed_flags=...)` — every decl not reached from any
  seed in the mask.
* `Analysis.kept_alive_by_dead_branches()` — diff of the default
  closure against the strict closure that skips dead-branch edges.
* `Analysis.kept_alive_by_flags_only(flags, ...)` — diff of
  `reachable(seed_flags)` against `reachable(seed_flags & ~flags)`.
  Pass `NodeFlags.TESTCASE` for the "what dies if the test suite goes
  away" question, `NodeFlags.NOQA` for the stale-F401 audit, or OR
  them together.

All four delegate to a rust BFS in one FFI call — there is no Python
adjacency walk on the hot path.

### 6. Graph persistence — `dead_cst.graph.write_graph` / `read_graph`

`dead-cst build ROOT -o PATH` materializes the graph once and writes it
to disk as a bincode blob prefixed with the magic bytes `DEADCSTG` and
a `u32` format version. `dead-cst analyze --graph PATH` /
`dead-cst remove --graph PATH` skip the build entirely and load the
file instead.

The library-level twin:

```python
from dead_cst.graph import write_graph, read_graph

graph = Analysis(root, resolver=...).materialize_all()
write_graph("graph.bin", graph, meta=[("branch", "main")])

loaded, metadata = read_graph("graph.bin")
loaded.reachable(seed_flags=NodeFlags.ENTRYPOINT)  # pure-Python BFS
```

Loaders are strict — a version mismatch is a fatal error with no
migration path (rebuilding is cheap). The metadata block records
`created_at`, node / edge / file / line counts, and any
user-supplied `(key, value)` pairs the CLI's `--meta` flag threads
through.

Plugins deliberately do **not** round-trip. Loading a graph gives you
a `LoadedGraph` (not a `ProjectContext`), and only the materialized
adjacency is captured — a project that needs plugin-emitted edges has
to rebuild rather than load.

### 7. Codemod — `dead_cst/codemod.py`

`remove_code(dead_nodes, package_path)` runs a LibCST
`RemoveDeadSymbols` transformer keyed on `(fqname, start_line)` pairs
(line disambiguates shadowed decls), then prunes now-unused imports
via libcst's `RemoveImportsVisitor`. Position keying is critical —
losing it conflates a dead decl with a live shadow.

`generate_patch(dead_nodes, root)` is the non-destructive twin: same
selection logic, same two-pass LibCST pipeline, but emits a
`git apply`-compatible unified diff with `diff --git` headers (and
`deleted file mode 100644` for module-node deletions) instead of
writing back. Selection is driven entirely by the input iterable, so
callers can slice the unreachable set however they like (e.g. one SCC
at a time) to review a big codebase as a series of focused patches.
`dead-cst remove` uses `generate_patch` exclusively — it emits the
patch to stdout (or `--output PATH`) and never mutates source.

The codemod is the only stage that still uses LibCST.

## Graph model invariants

* One node per top-level declaration plus one synthetic module node
  per file. Nested defs (inner functions, methods, nested classes) are
  deliberately not given their own nodes — refs from inside them
  attribute to the enclosing top-level decl.
* A module-level `import` / `from ... import ...` is itself a node of
  type `"import"`. Local uses of an imported name always edge to the
  local alias (the codemod invariant: an unused import has zero
  in-edges). When ty's module resolver / global-scope lookup can pin
  the use to specific upstream targets, the use *also* emits direct
  edges to each of them.
* Submodules edge to their parent package so `__init__.py` stays alive
  as long as anything in the package does.
* `EdgeFlags.DEAD_BRANCH` is metadata only; default reachability still
  follows the edge.
* `NodeFlags.SHADOWED` decls (a `def f` rebound later in the same
  file) are kept in the graph but excluded from the cross-module
  lookup so consumer imports route to the live binding.
* `NodeFlags.OVERLOAD` follows the same trie-exclusion rule as
  `SHADOWED`; lifetime is anchored to the matching same-file impl via
  explicit `impl -> overload` edges.
* `NodeFlags.NOQA` flags import aliases preserved by a user
  ruff/pyflakes noqa directive (per-line `# noqa`, `# noqa: F401`,
  `# noqa: E501, F401`, or file-level `# ruff: noqa` / `# flake8:
  noqa`). Layered on `ENTRYPOINT`.
* `NodeFlags.NOTEBOOK` tags every node sourced from a Jupyter
  `.ipynb` file. Notebook cells run top-to-bottom rather than being
  imported, so the bit alone keeps the node alive (no `ENTRYPOINT`
  overlay needed — `NOTEBOOK` is in `KEEPALIVE_DEFAULT`). The codemod
  skips notebook nodes (cell-aware writeback into the JSON envelope is
  out of scope today).
* `NodeFlags.STAR_REEXPORT` tags synthetic `kind="import"` nodes
  minted for each name brought in by `from X import *`. They live in
  the cross-module lookup like real re-export imports; the codemod
  skips them (there is no per-name source line to remove).
* `.pyi` stubs are ingested only for the **compiled-extension** layout
  (`_native.so` + `_native.pyi`, no `.py` twin). Peer-mode stubs
  alongside a real `.py` are dropped — the runtime always wins.

## Where to make changes

| If you want to…                                                | Touch                                              |
| -------------------------------------------------------------- | -------------------------------------------------- |
| Recognize a new decl shape                                     | `src/ingest.rs` (Phase 1)                          |
| Tweak how imports resolve                                      | `src/ingest.rs` (Phase 2, `emit_import_edges`)     |
| Tweak how references attribute                                 | `src/ingest.rs` (Phase 3, `emit_reference_edges`)  |
| Add a new plugin query                                         | a `*_indices` pymethod on `ProjectContext` (`src/project.rs`) |
| Keep alive symbols a framework registers dynamically           | new `NativePlugin` impl in `src/native_plugins.rs` |
| Support a new project layout / lockfile                        | new `PathResolver` under `dead_cst/contrib/`       |
| Change graph persistence format                                | `src/io.rs` (bump the format version)              |
| Change codemod output shape                                    | `dead_cst/codemod.py` (`RemoveDeadSymbols` / `_rewrite_one`) |
| Change patch format / per-SCC patch slicing                    | `dead_cst/codemod.py` (`generate_patch`)           |

See `CLAUDE.md` for the top-level invariants and `src/CLAUDE.md` for
the rust crate's principles.
