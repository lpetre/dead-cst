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
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::HashMap;
use ty_module_resolver::{resolve_module, ModuleName};
use ty_project::Db as ProjectDb;
use ty_python_core::definition::{Definition, DefinitionKind, DefinitionState};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;

use crate::helpers::{
    collect_all_imports_local, file_default_flags, file_path_string, iter_top_level_classes,
    module_fqname_for_file, position, range_key, resolve_base_fqn, scan_noqa_directives,
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

/// Project-wide PEP 503 distribution lookup, salsa-tracked so it's
/// memoized + accessible from per-file salsa-tracked queries. Wraps
/// `crate::ingest::DistLookup` (HashMap<PathBuf, String>) with the
/// derives needed for salsa-tracked return.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct ProjectDistLookup {
    pub(crate) map: std::collections::HashMap<std::path::PathBuf, String>,
}

#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn project_dist_lookup(db: &dyn ProjectDb) -> ProjectDistLookup {
    let site_packages = crate::ingest::site_packages_roots(db);
    ProjectDistLookup {
        map: crate::ingest::build_dist_lookup(&site_packages),
    }
}

/// Compute the canonical synthetic-node fqname for a non-first-party
/// target: `[external dist] {dist_name}` if dist_lookup knows the
/// file (PEP 503 canonical name), else `[external file] {top_level}`
/// for orphan site-packages files or editable installs.
pub(crate) fn external_fqname_for(
    db: &dyn ProjectDb,
    target_file: ruff_db::files::File,
    fallback_top_level: &str,
) -> String {
    use ruff_db::files::FilePath;
    let path_str = match target_file.path(db) {
        FilePath::System(p) => p.to_string(),
        _ => return format!("[external file] {fallback_top_level}"),
    };
    let canonical =
        std::fs::canonicalize(&path_str).unwrap_or_else(|_| std::path::PathBuf::from(&path_str));
    let lookup = project_dist_lookup(db);
    match lookup.map.get(&canonical) {
        Some(dist_name) => format!("[external dist] {dist_name}"),
        None => format!("[external file] {fallback_top_level}"),
    }
}

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
/// * `class_bases` — per top-level class declaration, the resolved
///   base list. Each entry is `(name_range_key, bases)` where
///   `name_range_key` matches the `(start, end)` `u32` pair the
///   assemble pass stores in `class_by_selection`, and `bases` is a
///   small-vec of [`ResolvedBase`] entries. Drives the project-wide
///   class-hierarchy fan-in at assemble time without a second AST
///   walk. Same-file `LocalSameFileClass(local_idx)` bases reference
///   the file's own `refs[local_idx]`; assemble translates them via
///   the same `NodeRef -> global_idx` map used for edges.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileNodes<'db> {
    pub(crate) nodes: Box<[NodeData]>,
    pub(crate) refs: Box<[NodeRef<'db>]>,
    pub(crate) ref_to_local: FxHashMap<NodeRef<'db>, u32>,
    pub(crate) exports_by_name: FxHashMap<String, SmallVec<[u32; 2]>>,
    pub(crate) star_reexports: FxHashMap<String, String>,
    pub(crate) class_bases: Vec<ClassBaseEntry>,
}

/// `(name_range_key, resolved_bases)` pair stored in
/// [`FileNodes::class_bases`]. The range key matches what the
/// assemble pass writes into `class_by_selection`, so the project-wide
/// fan-in is a hashmap probe.
pub(crate) type ClassBaseEntry = ((u32, u32), SmallVec<[ResolvedBase; 2]>);

/// One base-class expression resolved against a single file's
/// imports + same-file class bindings. Computed purely from per-file
/// state so it lives inside the salsa-tracked [`FileNodes`] payload —
/// project-wide fan-in happens at assemble time.
///
/// Four shapes cover today's class headers:
///
/// * `LocalSameFileClass(local_idx)` — `class Sub(Base): ...` where
///   `Base` is a top-level class earlier in the same file. The
///   `local_idx` indexes into the per-file `refs`/`nodes` arrays;
///   assemble translates it to a global graph index via the same
///   `ref_to_global` map used for edges.
/// * `ImportedFqn(fqn)` — `from <module> import Base` (or aliased)
///   followed by `class Sub(Base): ...`. The fqn is fully qualified
///   after relative-import resolution. Matches both project and
///   external bases by string equality with no per-file state.
/// * `Attribute { module_fqn, attr_name }` — `import <module> [as M];
///   class Sub(M.Base): ...`. We resolve `M` to its absolute module
///   name via the file's imports and keep `attr_name` separate so the
///   assemble pass can probe the target module's `exports_by_name`
///   (project-side) and synthesize the canonical `<module>.<attr>`
///   fqn (external-side) without re-parsing.
/// * `Unresolvable` — `Generic[T]`, dynamic constructions, any base
///   shape outside the static-resolution contract. Kept as an entry
///   so the parent class still appears in the index.
#[derive(Debug, Clone, Eq, PartialEq, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) enum ResolvedBase {
    LocalSameFileClass(u32),
    ImportedFqn(String),
    Attribute {
        module_fqn: String,
        attr_name: String,
    },
    Unresolvable,
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

    // Build per-file `class_bases`: walk every top-level ClassDef and
    // resolve each base expression against the file's imports + the
    // same-file class name table. No project-wide state needed; the
    // assemble pass turns these into the global `children_by_node` /
    // `children_by_fqn` indices.
    let class_bases = build_class_bases(&parsed, &exports_by_name, &nodes);

    FileNodes {
        nodes: nodes.into_boxed_slice(),
        refs: refs.into_boxed_slice(),
        ref_to_local,
        exports_by_name,
        star_reexports,
        class_bases,
    }
}

/// Per-file class-hierarchy producer used by [`file_to_nodes`].
///
/// For each top-level `ClassDef`, locate its corresponding graph node
/// (via `exports_by_name` + `nodes[..]` kind probe) and resolve each
/// base expression to a [`ResolvedBase`]. The result is a flat list of
/// `(local_class_idx, bases)` pairs the assemble pass folds into the
/// project-wide hierarchy without re-parsing.
///
/// Resolution mirrors today's `build_class_children` + `resolve_base_fqn`
/// helpers but reads everything from the per-file state:
///
/// * `Name(Foo)` where `Foo` is in `file_imports` → `ImportedFqn`.
/// * `Name(Foo)` where `Foo` is a top-level class earlier in this
///   file → `LocalSameFileClass(local_idx)`.
/// * `Attribute(Name(M).N)` where `M` is in `file_imports` →
///   `Attribute { module_fqn, attr_name: N }`.
/// * Deeper `a.b.c.N` chains rooted at an import → flattened into
///   `ImportedFqn("<a-upstream>.b.c.N")` (matches today's
///   `resolve_base_fqn` behaviour for the `find_subclasses_via_bases`
///   path).
/// * Anything else → `ResolvedBase::Unresolvable`.
fn build_class_bases(
    parsed: &ruff_db::parsed::ParsedModuleRef,
    exports_by_name: &FxHashMap<String, SmallVec<[u32; 2]>>,
    nodes: &[NodeData],
) -> Vec<ClassBaseEntry> {
    use ruff_python_ast::Expr;
    let file_imports = collect_all_imports_local(parsed);

    // Build name → local class idx for same-file class references.
    // Multiple top-level `class Foo` rebinds at the same name keep
    // the *first* (matching the historical libcst behaviour where a
    // syntactic base reference resolves to the lexically-earliest
    // class with that name). The cross-module case is unaffected —
    // `subclasses_of_fqn` indexes by fqname, so shadowed declarations
    // resolve through the import path, not this map.
    let mut same_file_classes: FxHashMap<&str, u32> = FxHashMap::default();
    for (name, locals) in exports_by_name {
        for &local_idx in locals {
            if matches!(nodes[local_idx as usize].kind, NodeKind::Class) {
                same_file_classes.entry(name.as_str()).or_insert(local_idx);
                break;
            }
        }
    }

    let mut out: Vec<ClassBaseEntry> = Vec::new();
    for cls in iter_top_level_classes(parsed) {
        // Key on the class name's `(start, end)` byte range — same
        // tuple the assemble pass uses to populate `class_by_selection`,
        // so the assemble-side fan-in is an O(1) hashmap probe.
        let cls_rk = range_key(cls.name.range());

        let Some(args) = &cls.arguments else {
            out.push((cls_rk, SmallVec::new()));
            continue;
        };
        let mut bases: SmallVec<[ResolvedBase; 2]> = SmallVec::new();
        for raw_base in &args.args {
            // Unwrap `Foo[T]`, `Generic[T]`, `Protocol[T]`, etc.
            // The original `build_class_children` matched these
            // before the fqn-aware resolver ran; preserve that
            // behaviour so subscripted generic bases still feed
            // the class hierarchy.
            let base = match raw_base {
                Expr::Subscript(s) => s.value.as_ref(),
                other => other,
            };
            // First-pass: try the fqn-aware resolver. It returns an
            // upstream fqname for Name/Attribute chains rooted at an
            // imported name. For a single Attribute(Name(M).N) we
            // override to the `Attribute` variant so the assemble pass
            // can probe the target module's `exports_by_name` for the
            // project-side class lookup.
            if let Some(fqn) = resolve_base_fqn(base, &file_imports) {
                match base {
                    Expr::Attribute(a) => {
                        if let Expr::Name(root) = a.value.as_ref() {
                            if let Some(module_fqn) = file_imports.get(root.id.as_str()) {
                                bases.push(ResolvedBase::Attribute {
                                    module_fqn: module_fqn.clone(),
                                    attr_name: a.attr.as_str().to_string(),
                                });
                                continue;
                            }
                        }
                        // Deeper chains: keep the flat fqn — used by
                        // the fqn-keyed lookup.
                        bases.push(ResolvedBase::ImportedFqn(fqn));
                    }
                    _ => {
                        bases.push(ResolvedBase::ImportedFqn(fqn));
                    }
                }
                continue;
            }
            // Fallback: same-file bare-name class reference.
            if let Expr::Name(name) = base {
                if let Some(&local_idx) = same_file_classes.get(name.id.as_str()) {
                    bases.push(ResolvedBase::LocalSameFileClass(local_idx));
                    continue;
                }
            }
            bases.push(ResolvedBase::Unresolvable);
        }
        out.push((cls_rk, bases));
    }
    out
}

/// Pure-rust variant of `ingest::import_payload_for`. Returns the same
/// three fields but as the per-file payload's `ImportPayload`, no
/// `ImportSpec` allocation.
/// Walk `(file, name)` through star-reexport chains. Returns every
/// non-star-reexport NodeRef reachable from `(target_file,
/// symbol_name)` — a decl, a non-star import alias, or several of
/// either when the name has multiple live bindings. Mirrors today's
/// `ingest::walk_globals_chain`.
///
/// `from A import g` where A has `from B import *` lands on A's
/// star alias for `g`; we resolve B → file, look up `g` there, and
/// recurse. Stops on a decl, on a non-star import, on a missed
/// lookup (yields nothing past it), or on a cycle.
pub(crate) fn walk_exports_chain<'db>(
    db: &'db dyn ProjectDb,
    target_file: File,
    symbol_name: &str,
) -> Vec<NodeRef<'db>> {
    let mut seen: std::collections::HashSet<(File, String)> = std::collections::HashSet::new();
    let mut out: Vec<NodeRef<'db>> = Vec::new();
    let mut stack: Vec<(File, String)> = vec![(target_file, symbol_name.to_string())];
    while let Some(key) = stack.pop() {
        if !seen.insert(key.clone()) {
            continue;
        }
        let target_nodes = file_to_nodes(db, key.0);
        // If `key` is a `from <upstream> import *` reexport, step
        // into the upstream file's same-name lookup. The star alias
        // itself isn't useful as a target — uses should land on the
        // upstream decl. Skip emitting it here.
        if let Some(upstream_module) = target_nodes.star_reexports.get(&key.1) {
            if let Some(mn) = ty_module_resolver::ModuleName::new(upstream_module) {
                if let Some(upstream) = ty_module_resolver::resolve_module(db, key.0, &mn) {
                    if let Some(upstream_file) = upstream.file(db) {
                        stack.push((upstream_file, key.1.clone()));
                        continue;
                    }
                }
            }
        }
        let Some(locals) = target_nodes.exports_by_name.get(&key.1) else {
            continue;
        };
        for &local_idx in locals {
            out.push(target_nodes.refs[local_idx as usize]);
        }
    }
    out
}

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
                // ImportFromSubmodule binds a submodule attribute on
                // the containing package as a side effect of a
                // `from X import …` statement inside __init__.py.
                // The bound NAME (k.module) is the attribute, but the
                // TARGET module is the from-clause itself — for
                // `from pkg._internal import *` in pkg/__init__.py
                // the side-effect attribute `_internal` on pkg
                // points at the file `pkg._internal`, not at
                // `pkg._internal._internal`. The from-clause module
                // is what resolves.
                let target = from_module_string(db, file, k.import(&parsed));
                (target, None)
            }
            DefinitionKind::StarImport(k) => {
                // `from X import *` — emit only the module-level
                // edge (`alias → Module(X)`); no per-name decl
                // fan-out. We collapse the per-name aliases to one
                // node per statement (see file_to_nodes), and the
                // existing pipeline matches this by returning
                // Vec::new() in resolve_from_imported's StarImport
                // branch. Uses of star-bound names still get
                // resolved via walk_exports_chain on the consumer's
                // from-import lookup; the chain walk takes care of
                // them.
                let module = from_module_string(db, file, k.import(&parsed));
                (module, None)
            }
            _ => continue,
        };

        if target_module_str.is_empty() {
            continue;
        }

        // Submodule disambiguation: `from p import functions` where
        // `p.functions` is itself a submodule should point at the
        // submodule file, not at `p` with a decl lookup. BUT: per
        // CPython's `_handle_fromlist`, namespace bindings WIN over
        // submodule lookups when both exist (e.g. `q = 42` in
        // p/__init__.py with a sibling p/q.py — the int binds, not
        // the submodule). Check namespace first; only switch to
        // submodule when the namespace doesn't already export decl.
        let (target_module_str, decl_name) = if let Some(decl) = &decl_name {
            let candidate = format!("{target_module_str}.{decl}");
            if crate::ingest::module_name_resolves(&candidate, file, db) {
                let namespace_has_decl = ModuleName::new(&target_module_str)
                    .and_then(|mn| resolve_module(db, file, &mn))
                    .and_then(|m| m.file(db))
                    .map(|f| {
                        let nodes = file_to_nodes(db, f);
                        nodes.exports_by_name.contains_key(decl)
                    })
                    .unwrap_or(false);
                if namespace_has_decl {
                    (target_module_str, Some(decl.clone()))
                } else {
                    (candidate, None)
                }
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
            let fqname = external_fqname_for(db, target_file, top_level);
            let key = ExternalKey::new(db, fqname);
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

        // Named-decl edge for from-imports. Walks the star-reexport
        // chain so `from A import g` where A re-exports from B lands
        // on B's real `g` rather than A's star alias node. salsa
        // memoizes each file's payload on the way through, so the
        // chain walk is hashmap lookups + a `seen` set.
        if let Some(decl_name) = decl_name {
            for target_ref in walk_exports_chain(db, target_file, &decl_name) {
                edges.insert((alias_ref, target_ref, 0));
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
    let module = crate::helpers::canonical_module_for_file(db, file)?;
    let parent_name = module.name(db).parent()?;
    let parent_module = resolve_module(db, file, &parent_name)?;
    parent_module.file(db)
}
