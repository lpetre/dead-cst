# Native plugins

> **Status: experimental, macOS + Linux.** The runtime split and in-tree native
> plugins are stable; *external* native plugin authoring is a preview — the
> API, the build flow, and the distribution are still moving, and only
> macOS (arm64) and Linux (x86_64 / aarch64) are wired today. See
> [Limitations](#limitations).

dead-cst extends the reachability graph through **native (Rust) plugins** —
there is no Python plugin protocol. They come in two flavors:

| | Built-in plugins | External plugins |
|---|---|---|
| Where they live | `runtime/src/native_plugins.rs`, compiled into the shipped extension | your own crate, compiled against the runtime `dylib` |
| How you get one | `NativePlugin.main_block()` … `NativePlugin.celery()`, or `--plugin <name>` | `native.load_native_plugins(<path>)` |
| Toolchain to author | n/a (in-tree) | rust, pinned to the build version |
| View of the graph | the full `ProjectContext` **by rust reference** | same |
| Own salsa-cached queries | yes | yes |

Built-ins ship in the default wheel and need no toolchain — author one by adding
an impl to the runtime crate (see
[CONTRIBUTING.md → Adding a plugin](CONTRIBUTING.md#adding-a-plugin)). This
document is about **external** native plugins: ones you compile in your own
crate and load at runtime, without forking dead-cst. The `[build-plugin]` extra
(`pip install dead-cst[build-plugin]`) provides the compile closure.

---

## How it works: one shared runtime

The native crate is split in two:

- **`dead-cst-runtime`** — the entire implementation (the graph builder, the
  query surface, the pyclasses, the plugin API). Built as both an **`rlib`** and a
  **`dylib`**.
- **`dead-cst-native`** — a thin pyo3 `#[pymodule]` shim that becomes
  `dead_cst._native`.

The runtime is linked one of two ways:

- The **dev build** (`maturin develop` / `uv sync`) and the **Windows wheel**
  link the runtime `rlib` *statically* into `_native` — one self-contained `.so`
  / `.pyd` that needs no rust at install time. This is the simplest, most
  robust layout, and it's what contributors and Windows users get. It cannot
  load external plugins (the runtime isn't shared).
- The **shipped macOS + Linux wheel** links the runtime **`dylib`** instead: a
  thin `_native` shim + `libdead_cst_runtime` + `libstd` ride in the package,
  resolving each other via `$ORIGIN` / `@loader_path`. The host therefore runs
  the *shared* runtime out of the box.

An **external plugin** is a separate `cdylib` that links the runtime **`dylib`**
(under `-C prefer-dynamic`). When both the extension module and the plugin
dynamically link the *same* `libdead_cst_runtime`, they share **one runtime
instance** — one salsa database, one `ProjectContext`, one set of types. That's
what lets a plugin receive a live `&ProjectContext` and define its own
salsa-tracked queries that cache against ty's database. (This is the
[dylint](https://github.com/trailofbits/dylint) model: link the host's runtime
dynamically, pinned to the exact compiler.) Because the shipped wheel already
runs the dylib, loading a plugin needs **no `_native` swap** — just the plugin's
compile closure, which the `[build-plugin]` extra provides.

The price of that fidelity is that a plugin is **ABI-coupled to the exact
dead-cst build** it was compiled against (Rust has no stable ABI). dead-cst
makes that safe rather than crashy — see [The ABI airlock](#the-abi-airlock).

---

## Writing a plugin

A plugin is a Rust `cdylib` that depends on `dead-cst-runtime` and implements
the [`ExternalPlugin`](runtime/src/native_plugins.rs) trait from the curated
`plugin_api`:

```rust
use dead_cst_runtime::native_plugins::plugin_api::{
    ExternalPlugin, PluginCtx, PluginError, PluginOps,
};

struct KeepMainBlocksAlive;

impl ExternalPlugin for KeepMainBlocksAlive {
    fn name(&self) -> &str {
        "KeepMainBlocksAlive"
    }

    fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps) -> Result<(), PluginError> {
        // `ctx` is a restricted, mostly-stable view of the frozen graph.
        for (module_idx, decl_indices) in ctx.main_blocks() {
            // Emit ops the host folds in after every plugin has run.
            ops.keep_alive(module_idx);
            for decl_idx in decl_indices {
                ops.keep_alive(decl_idx);
            }
        }
        Ok(())
    }
}
```

`PluginCtx` deliberately exposes a small, stable, **index-based** surface
rather than the whole internal `ProjectContext`, and `PluginOps` emits through
named methods instead of internal op types. The contract is a frozen-graph
one: the plugin observes the base graph only; its emissions are applied in a
single batch after every plugin returns. No
`Python<'_>` token is ever exposed — `PluginCtx` reads `#[pyclass(frozen)]`
data directly. This is the *same* trait every in-tree built-in implements — no
separate internal plugin surface exists, so anything a shipped plugin does is
expressible here.

**`run` is fallible.** It returns `Result<(), PluginError>`; a returned `Err`
aborts the materialize and surfaces to Python as an exception (`PluginError`'s
kind picks the type — `PluginError::value(..)` → `ValueError`,
`PluginError::runtime(..)` → `RuntimeError`). `PluginError` is pyo3-free, so
the curated surface stays GIL-free; the conversion happens once, at the host
boundary. A plugin that can't fail just ends with `Ok(())`. (Per-file
`run_on_file` stays infallible — see below.)

### The `PluginCtx` / `PluginOps` surface

Every query returns positional **indices** into the frozen node list — the
same index space `PluginOps` emits against. `ctx.node(idx)` turns an index
into an owned `NodeView { idx, fqname, kind, path, start_line, end_line,
flags }`.

`PluginCtx<'_>` — read the frozen graph:

| method | returns | what |
|---|---|---|
| `node_count()` | `usize` | number of nodes; valid indices are `0..node_count()` |
| `node(idx)` | `Option<NodeView>` | owned snapshot of one node |
| `find_module(fqname)` | `Option<usize>` | module node by dotted fqname |
| `find_declarations(fqname)` | `Vec<usize>` | every decl with this fqname (>1 when shadowed) |
| `module_for(path)` | `Option<usize>` | module node by source path |
| `resolve(fqname)` | `Option<usize>` | decl-or-module, walking back dotted segments |
| `decls_under(path_prefix)` | `Vec<usize>` | every node under a path prefix |
| `find_subclasses_of(class_idx)` | `Vec<usize>` | transitive subclasses of the class at `class_idx` |
| `find_subclasses_of_fqn(fqn)` | `Vec<usize>` | transitive subclasses of the class named by dotted `fqn` (fqn twin of `find_subclasses_of`) |
| `descendants(root_idx)` | `Vec<usize>` | forward reachability closure |
| `ancestors(decl_idx)` | `Vec<usize>` | reverse reachability closure |
| `direct_predecessors(idx)` | `Vec<usize>` | one-hop reverse step |
| `main_blocks()` | `Vec<(usize, Vec<usize>)>` | each `if __name__` block as `(module, [decls])` |
| `decorated_decls(fqnames)` | `Vec<usize>` | decls decorated by one of the decorators named by dotted `fqnames` (`"pytest.fixture"`), resolved like `find_subclasses_of_fqn` resolves a base class — a direct import, an alias, a re-export, a sibling-module spelling, `from X import *`, or a same-file definition all match; the fqname must name a module member resolvable from the analysis environment (an unresolvable one, or a non-module attribute like `pytest.mark.parametrize`, matches nothing); a builder suffix (`@launchable().cpus(16)`) is classified by its head call |
| `constructions(modules, names)` | `Vec<usize>` | `X = Ctor(...)` decls whose `Ctor` is one of `names` imported from one of `modules` |
| `handler_decorators(attrs)` | `Vec<(String, usize)>` | top-level fns decorated `@<owner>.<attr>(...)` for `attr` in `attrs`; returns `(owner_name, decl_idx)` — the raw textual owner, unresolved |
| `calls_with_string_arg(modules, name, arg_index)` | `Vec<(usize, String)>` | calls to `name` (imported from one of `modules`) whose arg at `arg_index` is a string literal; returns `(owning_decl_idx, literal)` |
| `calls_on_attr(attr, arg_index)` | `Vec<(usize, String)>` | `<recv>.<attr>(...)` calls (any receiver shape) whose arg at `arg_index` is a string literal; returns `(owning_decl_idx, literal)` |
| `calls_on_var(owner, attr, arg_index, required_positional)` | `Vec<(usize, String)>` | `<owner>.<attr>(...)` calls where `owner` is a bare variable (`mocker`, `monkeypatch`) and the arg at `arg_index` is a string literal; `required_positional` (if set) restricts to calls with exactly that many positional args |
| `handler_decorators_via(via_attr, attrs)` | `Vec<(String, usize)>` | two-level twin of `handler_decorators`: `@<owner>.<via_attr>.<attr>(...)` (e.g. `via_attr="tree"` for `@bot.tree.command`) |
| `decorated_decls_with_args(fqnames)` | `Vec<(usize, CallArgs)>` | args-capturing twin of `decorated_decls`: each match paired with its decorator's head-call kwargs (read via `CallArgs`) |
| `factory_decls(modules, ctor_names)` | `Vec<(usize, Vec<String>)>` | fns/classes that *return* an instance of one of `ctor_names` (app-factory shape); `(decl_idx, return_kinds)` |
| `classes_defining_method(method_name)` | `Vec<usize>` | classes with a top-level `def <method_name>` |
| `module_surface(module_fqn)` | `Vec<usize>` | the names `from module_fqn import *` would bind |
| `module_surfaces(module_fqns)` | `Vec<Vec<usize>>` | bulk `module_surface`, one row per input fqn |
| `module_top_level_decls(module_fqn)` | `Vec<usize>` | every top-level decl in a module (not just the `*`-exported surface) |
| `dunder_all_exports(module_fqn)` | `Option<Vec<usize>>` | the decls named by `module_fqn`'s `__all__`, or `None` |
| `literal_list_entries(var_fqn)` | `Option<Vec<String>>` | the string entries of a `X = ["a", "b"]` literal-list assignment |
| `decls_matching_name(pattern)` | `Vec<usize>` | top-level decls whose simple name matches the regex `pattern` |
| `nodes_matching(&filter)` | `Vec<usize>` | nodes matching a `NodeFilter` (kinds / simple names / paths / fqname prefix / flags-all / flags-any) |
| `has_imports_of(module)` | `bool` | does any file import `module`? |
| `imports_of(module)` | `Vec<usize>` | the `import` nodes binding names from `module` |
| `module_for(path)` / `modules_for_paths(paths)` | `Option<usize>` / `Vec<Option<usize>>` | module node(s) by source path, one-shot or bulk |
| `nodes_at(idxs)` | `Vec<NodeView>` | bulk `node()` — owned snapshots, out-of-range indices skipped |
| `node_paths(idxs)` | `Vec<String>` | source path per index (cheaper than `nodes_at` when only paths are needed) |
| `function_parameters(idxs)` / `class_method_parameters(idxs)` | `Vec<Vec<String>>` | parameter-name lists for the function (or `__init__`) at each index |
| `nodes()` | `impl Iterator<Item = NodeRef<'_>>` | borrowing walk over every node (no per-node `String` clone — use over `nodes_at(0..n)` for a full scan) |
| `edges()` | `impl Iterator<Item = EdgeRef>` | borrowing walk over every `(src, dst, flags)` edge |

The decorator / construction / call / subclass-by-fqn / name-pattern matchers
delegate to the same rust query cores the in-tree native plugins use; they're
the project-wide analogue of the per-file
[`PluginFileCtx`](#per-file-plugins-optional-salsa-cached) helpers.

`NodeView` (owned, from `node`/`nodes_at`) vs. `NodeRef<'_>` / `EdgeRef`
(borrowing, from `nodes()`/`edges()`): reach for the borrowing iterators when
scanning the whole graph and the owned `NodeView` for a handful of specific
indices. `CallArgs` / `ArgValue` (the captured decorator/constructor kwargs
returned by `decorated_decls_with_args`) are re-exported from `plugin_api`;
read a keyword via `CallArgs::str_value(name)`.

Every string these queries hand back (positional string args, `CallArgs`
kwargs, literal-list elements) is **constant-folded** at extraction time: a
bare literal, implicit concatenation, `"a" + "b"`, an f-string whose
interpolations fold, the file's own `__name__`, and top-level names bound
exactly once to a foldable string in the same file all read as the resulting
string. Anything else (a call, an imported name, `!r`, a format spec, a name
shadowed in the enclosing `def` / `class`) is unknown, exactly as a non-literal
always was. The rules live in `runtime/src/string_fold.rs`.

`PluginOps` — emit ops (each maps to one host `PreparedOp`):

| method | host op | what |
|---|---|---|
| `keep_alive(decl_idx)` | `PreparedOp::Entrypoint` | flag a node `ENTRYPOINT` so it seeds reachability directly |
| `flag_decl(decl_idx, flags)` | `PreparedOp::FlagDecl` | OR `flags` onto an existing decl — e.g. a registered node flag resolved via `ctx.node_flag(name)` — without minting a marker node |
| `add_edge(src_idx, dst_idx, flags)` | `PreparedOp::Edge` | add `src -> dst` between existing nodes; `flags` is `0` or one of `plugin_api::FLAG_DEAD_BRANCH` / `FLAG_DYNAMIC_IMPORT` / `FLAG_INIT_SUBCLASS` |

Endpoint indices are bounds-checked by the host at apply time — a dangling
index is rejected cleanly.

### Declaring flags

Node and edge flags are described by a single `FlagSpec` vocabulary (re-exported
from `plugin_api`) and live in two separate registries — a 32-bit node space and
an 8-bit edge space. The engine registers its built-ins first (`engine/…`);
plugins contribute their own through two optional `ExternalPlugin` hooks:

```rust
use dead_cst_runtime::native_plugins::plugin_api::FlagSpec;

impl ExternalPlugin for MyPlugin {
    fn name(&self) -> &str { "MyPlugin" }

    fn declare_node_flags(&self) -> Vec<FlagSpec> {
        vec![FlagSpec {
            name: "acme/handler".to_string(),   // owner/name; "engine/" is reserved
            seed: true,         // contributes to the default reachability seed mask…
            default_on: true,   // …when also default_on
            description: "Kept alive because acme's router registers it.".to_string(),
        }]
    }

    fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps) -> Result<(), PluginError> {
        let handler = ctx
            .node_flag("acme/handler")
            .expect("declared in declare_node_flags");
        // …stamp `handler` directly on discovered decls via
        // `ops.flag_decl(idx, handler)`.
        Ok(())
    }
}
```

`FlagSpec` carries **no bit value**. The host allocates each plugin flag a bit
**above the engine masks, in plugin registration order** — so two builds with
the same plugin set produce identical bits. Read the assigned bit back at `run`
time with `ctx.node_flag(name) -> Option<u32>` / `ctx.edge_flag(name) ->
Option<u8>` (`None` if no plugin declared it). `declare_edge_flags` is the
edge-space twin; edge flags share the 8-bit width and `seed`/`default_on` are
node-reachability concepts (recorded but unused for edges today).

Registration rules — all enforced once, under the GIL, before the parallel run:

- **`engine/` is protected.** A plugin declaring an `engine/…` flag fails the
  whole materialize.
- **Idempotent on an identical spec.** Two plugins declaring the *same*
  `owner/name` with the same `seed`/`default_on`/`description` collapse to one
  shared bit — this is exactly how the built-in `pytest` and `unittest` plugins
  both declare `test/testcase` and share it. A re-declaration with a
  *conflicting* spec (same name, different fields) fails loudly.
- **Width is capped.** Exhausting the node width (bit 31) or edge width (bit 7)
  fails with a clear error naming the flag and the cap.

Both registries are serialized into the `.dcg` graph file (`FORMAT_VERSION` 2),
so an external reader can decode any bit — engine or plugin — back to its name.
The Python layer derives its default keepalive mask from the registry
(`ctx.default_seed_mask()`, the OR of every `seed && default_on` bit) rather than
a hand-maintained constant, so a plugin's seed flag is kept alive by default iff
that plugin is registered.

### Per-file plugins (optional, salsa-cached)

By default an `ExternalPlugin` is **project-wide**: its `run(ctx, ops)` is
called once against the whole frozen graph. A plugin can instead opt into
**per-file** dispatch — invoked once per project file through a salsa-cached
query, so an unchanged file's ops are reused across a `re_materialize` with
zero re-run (the same fast-path the in-tree `MainBlockPlugin` rides).

Opt in by also implementing `PerFilePlugin` and returning `Some(self)` from
`per_file()`:

```rust
impl ExternalPlugin for MyPlugin {
    fn name(&self) -> &str { "MyPlugin" }
    fn per_file(&self) -> Option<&dyn PerFilePlugin> { Some(self) }   // opt in
}

impl PerFilePlugin for MyPlugin {
    fn run_on_file(&self, file: &PluginFileCtx<'_>, ops: &mut FileOps) {
        if let Some(block) = file.main_block_range() {
            let (start, end) = file.line_span(block);
            // Seed the module node directly, then keep alive every decl that
            // lives inside the `if __name__ == "__main__"` block — no marker node.
            ops.keep_alive(file.module_local_idx());
            for n in file.nodes() {
                if n.local_idx != 0 && n.start_line >= start && n.end_line <= end {
                    ops.keep_alive(n.local_idx);
                }
            }
        }
    }
}
```

When `per_file()` returns `Some`, the host ignores `run` entirely. Whether a
plugin is per-file is read **once, at load**.

`PluginFileCtx<'_>` is a restricted, single-file view — `module_fqname()`,
`module_local_idx()`, `node_count()`, `node(local_idx)`, `nodes()`,
`line_span(range)`, `parsed()` (the raw AST), the `main_block_range()`
convenience, and the ready-made file-local matchers `imports_any_module(modules)`,
`decorated_decls(fqnames)` (file-local: matches a decorator by its spelling after
this file's import / alias following or a same-file definition — re-exports through
*other* files need the project-wide query), `constructions(modules, names)`, and
`calls(modules, names)` (file-local twins of the project-wide `PluginCtx`
matchers above). `FileOps` mirrors every `PluginOps` emitter — `keep_alive`,
`flag_decl(decl_idx, flags)`, and `add_edge(src, dst, flags)` — but in the
file's **local** index space (the index args are positions in `file.nodes()`,
which the host maps to global indices at apply time).

**Purity is the contract that buys the cache.** `run_on_file` must be a pure
function of its `file` — it may read only what `PluginFileCtx` exposes and must
not consult project-wide state, other files, globals, the clock, or the
filesystem, and it may only reference nodes *in this file*. An impure
`run_on_file` would serve stale ops after an unrelated edit.

This is also why a per-file plugin **cannot** look flags up through the registry
(`ctx.node_flag` / `ctx.edge_flag` exist only on the project-wide `PluginCtx`):
the flag registry is derived from the *whole* plugin set, which isn't a file
input, so reading it from `run_on_file` would serve stale bits after a
plugin-set change. Per-file plugins emit the built-in engine flag constants
(`plugin_api::FLAG_*`) instead; declaring custom flags is a project-wide-plugin
capability today.

### Topics: per-file facts → project-wide reader

A per-file plugin can't see project-wide state — but it often discovers
something one file at a time that its project-wide half needs to act on. The
**topic/fact** channel carries that across: the per-file run *publishes facts*
under a topic, and the project-wide `run` *reads them all back*.

Declare topics the same way you declare flags, via `declare_topics`:

```rust
use dead_cst_runtime::native_plugins::plugin_api::TopicSpec;

impl ExternalPlugin for MyPlugin {
    fn name(&self) -> &str { "MyPlugin" }
    fn per_file(&self) -> Option<&dyn PerFilePlugin> { Some(self) }

    fn declare_topics(&self) -> Vec<TopicSpec> {
        vec![TopicSpec {
            name: "acme/handlers".to_string(),   // owner/name; "engine/" is reserved
            description: "decls that look like request handlers".to_string(),
        }]
    }
}

impl PerFilePlugin for MyPlugin {
    fn run_on_file(&self, file: &PluginFileCtx<'_>, ops: &mut FileOps) {
        for n in file.nodes() {
            if n.local_idx != 0 && n.fqname.ends_with("_handler") {
                // emit by topic *name*, optionally pinned to a file-local decl
                ops.emit_fact("acme/handlers", Some(n.local_idx), n.fqname.to_string());
            }
        }
    }
}
```

Then read them in the project-wide `run`:

```rust
fn run(&self, ctx: &PluginCtx<'_>, ops: &mut PluginOps) -> Result<(), PluginError> {
    let handle = ctx.topic("acme/handlers").expect("declared in declare_topics");
    for fact in ctx.facts_for_topic(handle) {
        // fact: { path: String, decl_idx: Option<usize>, value: String }
        if let Some(idx) = fact.decl_idx {
            ops.keep_alive(idx);
        }
    }
    Ok(())
}
```

Key points:

- **Facts are emitted by topic *name*, not handle.** Same reason a per-file
  plugin can't resolve a flag bit: the registry handle is allocated from the
  whole plugin set, which isn't a file input, so keying the salsa cache on it
  would serve stale handles after a plugin-set change. The topic *name* is
  salsa-stable, so the per-file output stays cacheable; the host resolves
  name → handle on the project-wide side.
- **`decl_local_idx` is optional and translated.** `Some(local)` pins the fact
  to a position in `file.nodes()`, which the host maps to a **global** node
  index at assemble time; a fact whose decl doesn't resolve is dropped (so a
  `Some` `decl_idx` you read back always names a live node). `None` leaves the
  fact file-scoped (`path` only).
- **`engine/` is reserved and registration is idempotent**, exactly like flags:
  two plugins declaring the same `owner/name` topic share one handle; a
  conflicting re-declaration fails loudly.
- **Facts don't mutate the graph and aren't serialized.** They're an in-memory
  build-time side channel, queried on the live `ProjectContext`
  (`ctx.facts_for_topic` in Rust; `ProjectContext.topic_registry()` /
  `ProjectContext.facts_for_topic(name)` from Python). Nothing is written to the
  `.dcg` file.

Each plugin `cdylib` also exports a **manifest** — a self-contained
`#[repr(C)]` table listing the plugins it provides and the ABI fingerprint it
was built against. It's deliberately boring boilerplate; copy it from a worked
example:

- **[`examples/main_block_plugin/`](examples/main_block_plugin/)** — a complete
  **project-wide** external plugin (a `MainBlockPlugin` equivalent) including
  the manifest. Start here.
- **[`examples/per_file_main_block/`](examples/per_file_main_block/)** — the
  same behaviour as a **per-file** plugin (`per_file()` + `run_on_file`).
- **[`examples/per_file_decorated/`](examples/per_file_decorated/)** — a
  **two-plugin** cdylib wiring the per-file → project-wide channel end to end:
  a per-file scanner emits a fact per Click-decorated command (via the
  ready-made `decorated_decls` query), and a project-wide reader resolves the
  shared topic, reads the facts, and stamps a declared `click/command` node
  flag on each matched decl. The worked example for `declare_node_flags` /
  `declare_topics` / `emit_fact` / `facts_for_topic` and for a manifest that
  exports more than one plugin.

---

## Building a plugin

You need (on macOS or Linux — the shipped dynamic wheel, not a dev checkout):

1. **The compile closure**, installed via the extra:

   ```bash
   pip install dead-cst[build-plugin]
   ```

   This pulls a separate `dead-cst-plugin-host` package (large — see
   [Distribution](#distribution)) carrying the `.rlib` dependency closure +
   proc-macro dylibs `rustc` needs to compile against the runtime. (The runtime
   dylib itself already ships inside the `dead_cst` wheel.)

2. **The Rust toolchain at dead-cst's pinned version** (see
   [`rust-toolchain.toml`](rust-toolchain.toml)). A different `rustc` produces a
   different ABI fingerprint and the plugin will be rejected at load (cleanly).

Then build:

```bash
# Compiles plugin.rs against the in-package runtime dylib (via `rustc --extern`)
# + the extra's rlib closure — no Cargo project, no dead-cst source, no swap —
# and prints the built plugin path on stdout.
PLUGIN=$(dead-cst build-plugin path/to/plugin.rs)
```

Load it from Python and use it like any other plugin:

```python
from dead_cst import Analysis, _native as native

plugins = native.load_native_plugins(PLUGIN)
ctx = Analysis(root, plugins=plugins).materialize_all()
```

The shipped wheel already runs the shared runtime, so the built plugin loads
straight away. `build-plugin` finds the runtime dylib inside the installed
`dead_cst` package and the rlib closure via `import dead_cst_plugin_host`;
`--runtime-dir` overrides both. Run `dead-cst build-plugin --help` for the
options. (In a dev/static checkout there's no in-package runtime dylib, so
`build-plugin` needs a dynamic wheel installed first.)

### Dependencies available to a plugin

`build-plugin` wires `--extern` for a small, curated allowlist of runtime
dependencies, so a plugin can pull them in directly: `serde_json` (e.g. `use
serde_json::Value;`) to parse config or metadata, and `regex` (`use
regex::Regex;`) for pattern matching. Both are pinned into the compile closure
for exactly this. The rest of the runtime's transitive dependency tree is *not*
exposed as a stable surface: it ships in the closure (so the runtime itself
links), but its crates and versions are an implementation detail and may change
between releases. Build on `dead_cst_runtime`, `serde_json`, and `regex` only.

---

## The ABI airlock

A plugin is compiled against one exact dead-cst build, so it must be
**recompiled for each release**. dead-cst enforces that without ever crashing:

- Every plugin bakes an **ABI fingerprint** at compile time —
  `epoch | api<plugin_api epoch> | rustc commit | runtime version | target` —
  taken from the runtime it built against. The `api<N>` segment is a dedicated
  epoch for the curated `plugin_api` surface: bumping it (in `runtime/build.rs`)
  rejects plugins compiled against an older API even when nothing else changed.
- `native.load_native_plugins(path)` opens the dylib lazily and reads a
  **self-contained manifest** (plain data + the baked fingerprint) *before*
  touching any version-hashed runtime symbol.
- If the manifest is missing, the magic is wrong, or the fingerprint doesn't
  match the running runtime, the load is **rejected with a clear error**:

  ```
  ABI mismatch — plugin built against '1|api1|commit-hash: …|v0.13.0|aarch64-apple-darwin',
  this runtime is '1|api2|…|v0.14.0|…'. Rebuild the plugin against this release.
  ```

So a stale `.so` is refused, not segfaulted. Rebuild with `build-plugin` and
you're back in business.

---

## Distribution

The split keeps the cost where it belongs:

| install | download | installed | for |
|---|---|---|---|
| `dead-cst` | ~25 MB | ~60 MB | everyone — analysis, the CLI, the built-in plugins |
| `dead-cst[build-plugin]` | +~70 MB | +~70 MB | authoring native plugins |

The base `dead-cst` macOS/Linux wheel carries the shared runtime dylib + libstd
alongside the `_native` shim (so it's a bit larger than a pure static build, and
the host runs the shared runtime). The `[build-plugin]` extra then pulls
**`dead-cst-plugin-host`**, a data-only, platform-specific package shipping only
the **compile closure**: the `.rlib` dependency archives + proc-macro dylibs
`rustc` needs to validate the crate graph. `build-plugin` finds it via `import
dead_cst_plugin_host`. Each artifact ships **xz-compressed** — the raw closure is
~320 MB and zip's deflate only gets it to ~107 MB (over PyPI's 100 MB/file cap),
while xz packs it to ~70 MB; `build-plugin` decompresses to a temp dir (~320 MB,
freed on exit) before invoking `rustc`. It's large because `rustc` needs the full
closure to compile against the runtime; nothing in it is needed at *runtime*.

Maintainers produce both halves from **one** prefer-dynamic build, so the
runtime dylib's SVH matches the rlib closure plugins compile against. **`dead-cst
bundle-plugin-host`** does that build and gathers the rlib closure into the
`dead_cst_plugin_host` package; the publish workflow then repacks the runtime
dylib + libstd into the base wheel (`$ORIGIN` / `@loader_path`, stripped) and
builds one `py3-none-<plat>` `dead-cst-plugin-host` wheel per target (macOS
arm64 + Linux x86_64/aarch64). Both wheels are version-locked by
`scripts/stamp_version.py` and shipped to TestPyPI on every push to `main`, to
PyPI on release.

---

## Limitations

This is a preview. Known gaps:

- **macOS (arm64) and Linux only.** The Windows wheel is static — it runs
  analysis fine but can't load native plugins (no shared runtime); the loader
  plumbing isn't wired there. Plugins need the shipped *dynamic* macOS/Linux
  wheel: a dev/static checkout has no in-package runtime dylib, so
  `build-plugin` there asks you to install a dynamic wheel first.
- **Single-`.rs` plugins.** `build-plugin` compiles one source file; multi-crate
  plugins with their own dependencies are a follow-up.
- **`dead-cst-plugin-host` wheels are published to TestPyPI** (per push to
  `main`) **and PyPI** (per release), one per macOS arm64 / Linux x86_64 /
  Linux aarch64; their version is kept in lockstep with `dead-cst` by
  `scripts/stamp_version.py`. The base wheel's dynamic-runtime repack uses
  [`patchelf`](https://github.com/NixOS/patchelf) on Linux (present in the
  manylinux build image).
- **Recompile per release**, by design (full Rust fidelity has no stable ABI).
  The airlock makes a mismatch a clean error, not a crash.
- **External native plugins are the only out-of-tree extension path** — there
  is no Python plugin protocol — so extending dead-cst without forking means
  either opting into this preview or contributing a built-in to the runtime
  crate upstream.
