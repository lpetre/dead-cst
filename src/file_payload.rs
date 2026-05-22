//! Per-file payload returned by the salsa-tracked `file_to_nodes` query.
//!
//! This is the first step of the three-phase → fan-out/fan-in
//! refactor described in the project conversation. `file_to_nodes` is
//! a `#[salsa::tracked]` function so that:
//!
//! * the per-file walk runs in parallel via salsa's lock-free worker
//!   coordination (the same machinery that already gives `prewarm_file`
//!   16k files/sec);
//! * cross-file calls — e.g. `file_to_edges(A)` looking up a target
//!   `Definition` in `B` via `file_to_nodes(B)` — are memoized for free
//!   (`returns(ref)` returns a borrow straight into salsa's storage,
//!   no clone).
//!
//! Pattern cribbed from `function_known_decorators` in
//! `vendor/ruff/crates/ty_python_semantic/src/types/infer.rs`, which is
//! the closest analogue (tracked fn, `'db`-lifetimed return, salsa::Update
//! derive on the carrier struct).
//!
//! This module DOES NOT yet replace `ingest_decls` — the existing
//! pipeline still runs. The function is wired into the crate but has
//! no callers; landing it first lets us validate the salsa macro
//! pattern + the `NodeData` shape before the driver swap.

use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::{line_index, source_text};
use ruff_text_size::TextRange;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::HashMap;
use ty_module_resolver::{file_to_module, resolve_module, ModuleName};
use ty_project::Db as ProjectDb;
use ty_python_core::definition::{Definition, DefinitionKind, DefinitionState};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;

use crate::helpers::{
    file_default_flags, file_path_string, module_fqname_for_file, position, scan_noqa_directives,
    NODE_FLAGS_NOQA_PIN,
};
use crate::ingest::{decl_kind_str, from_module_string};

/// Stable handle for a graph node, used as the public identity across
/// the fan-out pipeline. Edge endpoints in `file_to_edges` are
/// `(NodeRef, NodeRef, flags)`; the assembly pass translates each
/// `NodeRef` to its final `u32` graph index after all per-file
/// payloads are collected.
///
/// `NodeRef::Def` covers every binding ty enumerates as a `Definition`.
/// `NodeRef::Module` covers the synthetic `kind="module"` node per
/// file — `File` is itself a salsa input ingredient, so its handle is
/// stable and `Hash + Eq + Copy` just like `Definition`.
/// `NodeRef::External` covers the synthetic `[external] X` /
/// `[stdlib] X` / `[unresolved] X` nodes via a globally-interned
/// `ExternalKey` — same fqname produces the same key, dedup is free.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) enum NodeRef<'db> {
    Def(Definition<'db>),
    Module(File),
    External(ExternalKey<'db>),
}

/// Globally-interned synthetic external node identity. Same `fqname`
/// → same key → same final graph node, regardless of which file
/// emitted the edge.
///
/// Today's fqname conventions (simplified — full dist classification
/// is a separate follow-up):
///
/// * `[stdlib] X` — module X resolves to ty's standard library
///   search path.
/// * `[external] X` — module X resolves to a non-stdlib non-first-
///   party search path (site-packages). `X` is the *top-level*
///   module name; canonicalising to a PEP 503 dist name needs a
///   project-wide dist_lookup which lands later.
/// * `[unresolved] X` — module X doesn't resolve at all.
#[salsa::interned(debug)]
pub(crate) struct ExternalKey<'db> {
    pub(crate) fqname: String,
}

// Salsa tracks the interned heap separately; report 0 from GetSize.
impl get_size2::GetSize for ExternalKey<'_> {}

/// One graph node's data without the `Py<SymbolNode>` envelope.
///
/// Pure rust so it can live inside a salsa-tracked function's return
/// value. The assembly pass converts these to `Py<SymbolNode>` in a
/// single GIL acquisition at the end of the build.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct NodeData {
    pub(crate) fqname: String,
    pub(crate) kind: NodeKind,
    pub(crate) path: String,
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
    pub(crate) flags: u32,
    pub(crate) imports: Option<ImportPayload>,
}

/// Compact 1-byte discriminant for the graph's `kind` field. Mirrors
/// `graph::VALID_KINDS` exactly and converts to the static `&str` at
/// `Py<SymbolNode>` materialization time.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, salsa::Update, get_size2::GetSize)]
#[repr(u8)]
pub(crate) enum NodeKind {
    Function = 0,
    Class = 1,
    Variable = 2,
    Import = 3,
    TypeAlias = 4,
    Module = 5,
    Synthetic = 6,
}

impl NodeKind {
    // Used by the assembly pass that converts `NodeData → Py<SymbolNode>`
    // once `file_to_edges` lands and the driver is swapped.
    #[allow(dead_code)]
    pub(crate) fn as_static_str(self) -> &'static str {
        match self {
            NodeKind::Function => "function",
            NodeKind::Class => "class",
            NodeKind::Variable => "variable",
            NodeKind::Import => "import",
            NodeKind::TypeAlias => "type_alias",
            NodeKind::Module => "module",
            NodeKind::Synthetic => "synthetic",
        }
    }

    fn from_decl_kind_str(s: &str) -> Option<Self> {
        Some(match s {
            "function" => NodeKind::Function,
            "class" => NodeKind::Class,
            "variable" => NodeKind::Variable,
            "import" => NodeKind::Import,
            "type_alias" => NodeKind::TypeAlias,
            "module" => NodeKind::Module,
            "synthetic" => NodeKind::Synthetic,
            _ => return None,
        })
    }
}

/// Per-alias import metadata. Same fields as the python-facing
/// `Import` pyclass, but without the `Py` envelope so it can live
/// inside `NodeData`.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct ImportPayload {
    pub(crate) module: String,
    pub(crate) decl: Option<String>,
    pub(crate) star: bool,
}

/// Salsa-tracked output of [`file_to_nodes`]. Parallel arrays plus a
/// reverse lookup map make every per-file node addressable in O(1)
/// either by index (assembly pass) or by `NodeRef` identity
/// (cross-file edge resolution via [`ref_to_node`]).
///
/// Indexing convention: `nodes[0]` is the synthetic `kind="module"`
/// node for the file; `nodes[1..]` are global-scope definitions in
/// `use_def_map.all_definitions_with_usage()` iteration order.
/// `refs[i]` identifies `nodes[i]` (`Module(file)` at index 0,
/// `Def(definition)` at indices 1..).
///
/// * `exports_by_name` — per-name → live local indices into `nodes`.
///   Populated from ty's `all_end_of_scope_symbol_bindings()`; collapses
///   sequential rebinds while preserving branch-bound multi-binding
///   cases (try/except, if/else where each branch assigns).
///   This is the per-file slice of today's `globals_by_name`
///   shortcut a consumer's `from <thisfile> import x` uses.
/// * `star_reexports` — per-name → upstream absolute module name when
///   `name` is bound in this file by `from <upstream> import *`. Lets
///   downstream chain walks hop through this file.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileNodes<'db> {
    pub(crate) nodes: Box<[NodeData]>,
    pub(crate) refs: Box<[NodeRef<'db>]>,
    pub(crate) ref_to_local: FxHashMap<NodeRef<'db>, u32>,
    pub(crate) exports_by_name: FxHashMap<String, SmallVec<[u32; 2]>>,
    pub(crate) star_reexports: FxHashMap<String, String>,
}

/// Resolve a `NodeRef` to its `NodeData` payload.
///
/// Thin accessor (not salsa-tracked — see PR discussion): one
/// `file_to_nodes` lookup (memoized, ~ns) plus one HashMap probe.
/// Keeps the public identity model (callers traffic in `NodeRef`,
/// fetch `NodeData` on demand) without paying the ~200–400MB of
/// salsa ingredient overhead a `#[salsa::tracked] ref_to_node` would
/// cost at 4M-node scale.
///
/// Panics if the `NodeRef` doesn't belong to the project (file not
/// in the project, or `Def(d)` whose `d.file(db)` doesn't enumerate
/// `d` at module scope). Both indicate a contract violation upstream;
/// surface the bug loudly.
#[allow(dead_code)]
pub(crate) fn ref_to_node<'db>(db: &'db dyn ProjectDb, r: NodeRef<'db>) -> NodeData {
    match r {
        NodeRef::Def(d) => {
            let file = d.file(db);
            let payload = file_to_nodes(db, file);
            let idx = *payload
                .ref_to_local
                .get(&r)
                .expect("NodeRef::Def does not belong to its claimed file's payload")
                as usize;
            payload.nodes[idx].clone()
        }
        NodeRef::Module(f) => {
            let payload = file_to_nodes(db, f);
            payload.nodes[0].clone()
        }
        NodeRef::External(k) => NodeData {
            fqname: k.fqname(db).clone(),
            kind: NodeKind::Synthetic,
            path: String::new(),
            start_line: 0,
            start_column: 0,
            end_line: 0,
            end_column: 0,
            flags: 0,
            imports: None,
        },
    }
}

/// Build the per-file node payload. Salsa-tracked so it's memoized and
/// safe to call concurrently from rayon workers; `returns(ref)` so
/// callers get a borrow into salsa's storage instead of a clone.
///
/// Stub flag handling (`is_stub` / `stub_py_twin`) is intentionally
/// omitted here — it depends on the project-wide `peer_pyi_to_py` map,
/// which we'll feed into the assembly pass once the file-level work is
/// in place. The flag set is otherwise the same as today's
/// `ingest_decls`.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_nodes<'db>(db: &'db dyn ProjectDb, file: File) -> FileNodes<'db> {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let path_str = file_path_string(db, file);
    let module_fqname = module_fqname_for_file(db, file);
    let default_flags = file_default_flags(db, file);
    let (file_pinned_by_noqa, per_line_noqa_pins) =
        scan_noqa_directives(&parsed, &source, &line_index);

    let (msl, msc, mel, mec) = position(&line_index, &source, parsed.syntax().range);

    let mut nodes: Vec<NodeData> = Vec::new();
    let mut refs: Vec<NodeRef<'db>> = Vec::new();
    let mut ref_to_local: FxHashMap<NodeRef<'db>, u32> = FxHashMap::default();

    // Index 0: the synthetic module node. Anchors reachability for
    // every per-file decl via the `decl → module` edge `file_to_edges`
    // will emit.
    nodes.push(NodeData {
        fqname: module_fqname.clone(),
        kind: NodeKind::Module,
        path: path_str.clone(),
        start_line: msl,
        start_column: msc,
        end_line: mel,
        end_column: mec,
        flags: default_flags,
        imports: None,
    });
    refs.push(NodeRef::Module(file));
    ref_to_local.insert(NodeRef::Module(file), 0);

    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);

    // Star imports: ty mints one Definition per name brought in by
    // `from X import *`, but every per-name def from the same statement
    // shares the `*` token's range. Cache the synthesized `*<src>`
    // local name so we don't re-format / re-resolve it N times for
    // a star with N names. Identical names → identical NodeData →
    // dedup happens naturally at assembly time.
    let mut star_local_name_cache: HashMap<TextRange, String> = HashMap::new();

    let mut star_reexports: FxHashMap<String, String> = FxHashMap::default();

    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }

        let kind = def.kind(db);
        let Some(node_kind_str) = decl_kind_str(kind) else {
            continue;
        };
        let Some(node_kind) = NodeKind::from_decl_kind_str(node_kind_str) else {
            continue;
        };

        let place_id = def.place(db);
        let PlaceExprRef::Symbol(symbol) = place_table.place(place_id) else {
            continue;
        };
        let per_name = symbol.name().as_str().to_string();

        let target_range = kind.target_range(&parsed);
        let local_name = match kind {
            DefinitionKind::StarImport(k) => star_local_name_cache
                .entry(target_range)
                .or_insert_with(|| {
                    let src = from_module_string(db, file, k.import(&parsed));
                    format!("*{src}")
                })
                .clone(),
            _ => per_name.clone(),
        };

        let (mut sl, mut sc, el, ec) = position(&line_index, &source, target_range);

        // Snap `def`/`class`/`type` decls to the keyword position so
        // positions line up with the libcst pipeline (see comment in
        // `ingest_decls` for the rationale).
        if matches!(
            kind,
            DefinitionKind::Function(_) | DefinitionKind::Class(_) | DefinitionKind::TypeAlias(_)
        ) {
            let name_start = line_index.line_column(target_range.start(), &source);
            let line_text = &source[line_index.line_range(name_start.line, &source)];
            let indent = line_text
                .bytes()
                .take_while(|b| matches!(*b, b' ' | b'\t'))
                .count();
            sl = name_start.line.get();
            sc = indent;
        }

        let import_spec: Option<ImportPayload> = if matches!(node_kind, NodeKind::Import) {
            Some(import_payload_for_pure(kind, db, file, &parsed))
        } else {
            None
        };

        let mut flags: u32 = default_flags;
        if matches!(node_kind, NodeKind::Import)
            && (file_pinned_by_noqa || per_line_noqa_pins.contains(&sl))
        {
            flags |= NODE_FLAGS_NOQA_PIN;
        }
        // Stub-only ENTRYPOINT flagging is deferred to the assembly
        // pass; it needs the project-wide peer-stub map and the .py
        // twin's `exports_by_name`, which aren't visible to this
        // per-file query.

        let node = NodeData {
            fqname: format!("{module_fqname}.{local_name}"),
            kind: node_kind,
            path: path_str.clone(),
            start_line: sl,
            start_column: sc,
            end_line: el,
            end_column: ec,
            flags,
            imports: import_spec.clone(),
        };
        let local_idx = nodes.len() as u32;
        nodes.push(node);
        refs.push(NodeRef::Def(def));
        ref_to_local.insert(NodeRef::Def(def), local_idx);

        // Star-reexport tracking: mirror `ingest_decls`'s logic — a
        // later non-star binding for the same name clears the star
        // entry, so a shadowing import correctly disables chain walks
        // through this file.
        if let Some(spec) = &import_spec {
            if spec.star {
                star_reexports.insert(per_name.clone(), spec.module.clone());
            } else {
                star_reexports.remove(&per_name);
            }
        } else {
            star_reexports.remove(&per_name);
        }
    }

    // Post-pass: populate `exports_by_name` from ty's end-of-scope
    // symbol bindings. This collapses sequential rebinds to the latest
    // while preserving branch-bound multi-binding cases (try/except,
    // if/else where each branch assigns). Mirrors `globals_by_name` in
    // today's pipeline. Values are indices into `nodes` (≥ 1, since
    // index 0 is the module node which can't be bound to a name).
    let mut exports_by_name: FxHashMap<String, SmallVec<[u32; 2]>> = FxHashMap::default();
    for (symbol_id, bindings) in use_def_map.all_end_of_scope_symbol_bindings() {
        let PlaceExprRef::Symbol(sym) = place_table.place(ScopedPlaceId::Symbol(symbol_id)) else {
            continue;
        };
        let name = sym.name().as_str().to_string();
        let mut live: SmallVec<[u32; 2]> = SmallVec::new();
        for binding in bindings {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != file || def.file_scope(db) != global {
                continue;
            }
            if let Some(&local_idx) = ref_to_local.get(&NodeRef::Def(def)) {
                live.push(local_idx);
            }
        }
        if !live.is_empty() {
            exports_by_name.insert(name, live);
        }
    }

    FileNodes {
        nodes: nodes.into_boxed_slice(),
        refs: refs.into_boxed_slice(),
        ref_to_local,
        exports_by_name,
        star_reexports,
    }
}

/// Pure-rust variant of `ingest::import_payload_for`. Returns the same
/// three fields but as the per-file payload's `ImportPayload`, no
/// `ImportSpec` allocation.
pub(crate) fn import_payload_for_pure<'db>(
    kind: &DefinitionKind<'db>,
    db: &'db dyn ty_python_semantic::Db,
    file: File,
    parsed: &ruff_db::parsed::ParsedModuleRef,
) -> ImportPayload {
    match kind {
        DefinitionKind::Import(k) => {
            let alias = k.alias(parsed);
            ImportPayload {
                module: alias.name.id.as_str().to_string(),
                decl: None,
                star: false,
            }
        }
        DefinitionKind::ImportFrom(k) => {
            let alias = k.alias(parsed);
            ImportPayload {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(alias.name.id.as_str().to_string()),
                star: false,
            }
        }
        DefinitionKind::ImportFromSubmodule(k) => ImportPayload {
            module: from_module_string(db, file, k.import(parsed)),
            decl: Some(k.module(parsed).id.as_str().to_string()),
            star: false,
        },
        DefinitionKind::StarImport(k) => ImportPayload {
            module: from_module_string(db, file, k.import(parsed)),
            decl: None,
            star: true,
        },
        _ => unreachable!("import_payload_for_pure called with non-import kind"),
    }
}

// ---------------------------------------------------------------------------
// file_to_edges
// ---------------------------------------------------------------------------

/// Salsa-tracked output of [`file_to_edges`]. One entry per
/// `(src, dst, flags)` edge originating in this file. `FxHashSet` so
/// duplicate emissions within the file (a name used twice resolving
/// to the same target) collapse to a single entry — matching today's
/// `builder.edge_set` behavior.
///
/// Edges in this set are *unresolved-to-graph-idx*: endpoints are
/// `NodeRef<'db>` handles, not `u32` graph indices. The assembly pass
/// translates each `NodeRef` to its final `u32` after all per-file
/// payloads are collected.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileEdges<'db> {
    pub(crate) edges: FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)>,
}

/// Per-file edge emission. Salsa-tracked so the per-file walk
/// parallelizes via salsa's worker coordination, and cross-file
/// lookups (`file_to_nodes(target_file)`) are memoized for free.
///
/// Emits today:
///
/// * `Def → Module(file)` for every per-file decl — the reachability
///   anchor that wires a file's decls to its module node.
/// * `Module(file) → Module(parent_file)` when the file has a parent
///   package whose `__init__.py` resolves into the project. Mirrors
///   today's `emit_module_hierarchy`.
/// * `Def(alias) → Module(target_file)` and (for from-imports)
///   `Def(alias) → Def(target_decl)` for each import binding whose
///   upstream resolves to a project file. The `target_decl` lookup
///   goes through `file_to_nodes(target_file).exports_by_name`,
///   which is the "for free" salsa memoization this whole
///   refactor is built around.
///
/// Deliberately deferred (each is a self-contained follow-up):
///
/// * **External nodes.** `[external dist] X` / `[stdlib] X` /
///   `[unresolved] X` need `NodeRef::External` + a `#[salsa::interned]`
///   key. Imports that resolve outside the project are currently
///   *skipped*, not redirected to external nodes.
/// * **Attribute-chain edges.** `import a.b.c; a.b.c.f()` today
///   emits parallel edges to every chain segment. This cut emits
///   only the alias → `a` edge.
/// * **Per-statement sibling-to-submodule edges.** `from .submod
///   import X` in an `__init__.py` mints both an `X` alias and a
///   `submod` submodule alias; today the X alias gets a sibling
///   edge to the submodule alias so they stay alive together.
/// * **Reference edges (Phase 3's `RefCollector` work).** `use →
///   alias` and `use → upstream` from every Name reference in every
///   owned expression. This is the biggest deferred chunk — the
///   refactor of the ~1k-line `RefCollector` into a per-file
///   producer is its own follow-up.
/// * **Dynamic import edges** (`__import__('x')` /
///   `importlib.import_module('x')`).
/// * **Dead-branch edge flag.** Edges originating inside `if False:`
///   / post-`return` blocks need `EdgeFlags::DEAD_BRANCH` set; needs
///   `dead_ranges` on the payload.
#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn file_to_edges<'db>(db: &'db dyn ProjectDb, file: File) -> FileEdges<'db> {
    let self_nodes = file_to_nodes(db, file);
    let module_ref = NodeRef::Module(file);
    let mut edges: FxHashSet<(NodeRef<'db>, NodeRef<'db>, u32)> = FxHashSet::default();

    // 1. Decl → module anchor edges. Every per-file def points at the
    //    module node so reachability seeded from the module covers all
    //    its decls. `refs[0]` is the module itself; skip it.
    for &node_ref in self_nodes.refs.iter().skip(1) {
        edges.insert((node_ref, module_ref, 0));
    }

    // 2. Submodule → parent module hierarchy edge. Mirrors today's
    //    `emit_module_hierarchy`. Skipped when the parent doesn't
    //    resolve into the project (top-level package, or sibling
    //    outside the project root).
    if let Some(parent_file) = parent_module_file(db, file) {
        edges.insert((module_ref, NodeRef::Module(parent_file), 0));
    }

    // 3. Import alias edges. Walk each kind="import" Definition in
    //    this file's global scope, resolve its upstream module via
    //    `resolve_module`, and emit:
    //
    //    * `Def(alias) → Module(target_file)` always (for `import M`
    //       and for the module-side reachability anchor of `from M
    //       import x`).
    //    * `Def(alias) → Def(target_decl)` for from-imports whose
    //      `decl_name` resolves in the target's `exports_by_name`.
    //
    //    Targets outside the project (stdlib / dist / unresolved)
    //    are SKIPPED in this cut — the External NodeRef variant
    //    lands later.
    let index = semantic_index(db, file);
    let global = FileScopeId::global();
    let use_def_map = index.use_def_map(global);
    let parsed = parsed_module(db, file).load(db);
    let place_table = index.place_table(global);

    for (_def_id, state, _used) in use_def_map.all_definitions_with_usage() {
        let DefinitionState::Defined(def) = state else {
            continue;
        };
        if def.file(db) != file || def.file_scope(db) != global {
            continue;
        }
        let alias_ref = NodeRef::Def(def);
        // Skip Definitions that file_to_nodes didn't mint (non-symbol
        // places, unsupported kinds). Keeps the edge set internally
        // consistent with the node set.
        if !self_nodes.ref_to_local.contains_key(&alias_ref) {
            continue;
        }

        let kind = def.kind(db);
        if !kind.is_import() {
            continue;
        }

        let (target_module_str, decl_name): (String, Option<String>) = match kind {
            DefinitionKind::Import(k) => {
                // `import a.b.c` binds `a` locally. Resolving `a.b.c`
                // here would target the deepest module; ty's runtime
                // semantics bind only `a`. We resolve to `a` for the
                // first-cut module-level edge. Full chain edges
                // (parallel edges to `a.b` and `a.b.c`) are deferred.
                let alias = k.alias(&parsed);
                (alias.name.id.as_str().to_string(), None)
            }
            DefinitionKind::ImportFrom(k) => {
                let module = from_module_string(db, file, k.import(&parsed));
                let alias = k.alias(&parsed);
                (module, Some(alias.name.id.as_str().to_string()))
            }
            DefinitionKind::ImportFromSubmodule(k) => {
                // `from <parent_pkg> import <submod>` — the binding
                // attribute is `<submod>` on the parent package. The
                // upstream FILE is `<parent_pkg>.<submod>`'s file.
                let parent = from_module_string(db, file, k.import(&parsed));
                let segment = k.module(&parsed).id.as_str().to_string();
                let full = if parent.is_empty() {
                    segment
                } else {
                    format!("{parent}.{segment}")
                };
                (full, None)
            }
            DefinitionKind::StarImport(k) => {
                // ty mints one StarImport Definition per name the
                // star brought in. The per-name lives in the symbol;
                // the upstream module lives on the statement.
                let module = from_module_string(db, file, k.import(&parsed));
                let place_id = def.place(db);
                let PlaceExprRef::Symbol(symbol) = place_table.place(place_id) else {
                    continue;
                };
                (module, Some(symbol.name().as_str().to_string()))
            }
            _ => continue,
        };

        if target_module_str.is_empty() {
            continue;
        }

        // Submodule disambiguation: `from p import functions` where
        // `p.functions` is itself a submodule should point at the
        // submodule file, not at `p` with a decl lookup. Mirror
        // emit_upstream's `module_name_resolves` check.
        let (target_module_str, decl_name) = if let Some(decl) = &decl_name {
            let candidate = format!("{target_module_str}.{decl}");
            if crate::ingest::module_name_resolves(&candidate, file, db) {
                (candidate, None)
            } else {
                (target_module_str, Some(decl.clone()))
            }
        } else {
            (target_module_str, decl_name)
        };
        let Some(target_module_name) = ModuleName::new(&target_module_str) else {
            // Bad module name (relative-dots overflow, etc.). Route
            // to a synthetic `[unresolved] X` external node.
            let top_level = target_module_str
                .split('.')
                .next()
                .unwrap_or(&target_module_str);
            let key = ExternalKey::new(db, format!("[unresolved] {top_level}"));
            edges.insert((alias_ref, NodeRef::External(key), 0));
            continue;
        };
        let Some(target_module) = resolve_module(db, file, &target_module_name) else {
            let top_level = target_module_str
                .split('.')
                .next()
                .unwrap_or(&target_module_str);
            let key = ExternalKey::new(db, format!("[unresolved] {top_level}"));
            edges.insert((alias_ref, NodeRef::External(key), 0));
            continue;
        };
        // Stdlib targets drop silently — matches existing pipeline.
        if target_module
            .search_path(db)
            .is_some_and(|sp| sp.is_standard_library())
        {
            continue;
        }
        let Some(target_file) = target_module.file(db) else {
            let top_level = target_module_str
                .split('.')
                .next()
                .unwrap_or(&target_module_str);
            let key = ExternalKey::new(db, format!("[unresolved] {top_level}"));
            edges.insert((alias_ref, NodeRef::External(key), 0));
            continue;
        };
        // Non-first-party (site-packages) → External node keyed on
        // top-level module name. TODO: dist_lookup canonicalisation.
        if target_module
            .search_path(db)
            .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
        {
            let top_level = target_module_str
                .split('.')
                .next()
                .unwrap_or(&target_module_str);
            let key = ExternalKey::new(db, format!("[external] {top_level}"));
            edges.insert((alias_ref, NodeRef::External(key), 0));
            continue;
        }
        // Self-imports (re-importing from the same file) don't
        // produce meaningful cross-file edges — skip to avoid the
        // trivial alias → self-module cycle.
        if target_file == file {
            continue;
        }

        // Module-level edge.
        edges.insert((alias_ref, NodeRef::Module(target_file), 0));

        // Named-decl edge for from-imports. Goes through
        // `file_to_nodes(target_file).exports_by_name` — salsa
        // memoizes the target's payload, so this is the cross-file
        // call that's "free" if the target's worker already ran.
        if let Some(decl_name) = decl_name {
            let target_nodes = file_to_nodes(db, target_file);
            if let Some(target_locals) = target_nodes.exports_by_name.get(&decl_name) {
                for &local_idx in target_locals {
                    let target_ref = target_nodes.refs[local_idx as usize];
                    edges.insert((alias_ref, target_ref, 0));
                }
            }
        }
    }

    FileEdges { edges }
}

/// Resolve a file's parent module to its `File` handle, when the
/// parent's `__init__.py` is itself in the project. Mirrors today's
/// `emit_module_hierarchy::parent_module_edge`. Returns `None` for
/// top-level packages (no parent) and for parents that live outside
/// the project (the cross-file hierarchy edge has nowhere to land).
fn parent_module_file(db: &dyn ProjectDb, file: File) -> Option<File> {
    let module = file_to_module(db, file)?;
    let parent_name = module.name(db).parent()?;
    let parent_module = resolve_module(db, file, &parent_name)?;
    parent_module.file(db)
}
