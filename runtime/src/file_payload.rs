//! Per-file payload returned by the salsa-tracked `file_to_nodes` query.
//!
//! This is the first step of the three-phase → fan-out/fan-in
//! refactor described in the project conversation. `file_to_nodes` is
//! a `#[salsa::tracked]` function so that:
//!
//! * the per-file walk runs in parallel via salsa's lock-free worker
//!   coordination (the same machinery that already gives `prewarm_file`
//!   16k files/sec);
//! * cross-file calls — e.g. the assembly pass's `resolve_member`
//!   looking up a target `Definition` in `B` via `file_to_nodes(B)` —
//!   are memoized for free (`returns(ref)` returns a borrow straight
//!   into salsa's storage, no clone).
//!
//! Pattern cribbed from `function_known_decorators` in
//! `vendor/ruff/crates/ty_python_semantic/src/types/infer.rs`, which is
//! the closest analogue (tracked fn, salsa::Update derive on the
//! carrier struct).

use compact_str::{CompactString, ToCompactString};
use ruff_db::files::File;
use ruff_db::parsed::parsed_module;
use ruff_db::source::{line_index, source_text};
use ruff_python_ast::Stmt;
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::HashMap;
use ty_module_resolver::resolve_module;
use ty_project::Db as ProjectDb;
use ty_python_core::definition::{Definition, DefinitionKind, DefinitionState};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::scope::FileScopeId;
use ty_python_core::semantic_index;

use crate::graph::{DeclKey, NodeFlags};
use crate::helpers::{
    classify_base, collect_all_imports_local, detect_dead_ranges, file_default_flags,
    iter_top_level_classes, module_fqname_for_file, position, range_key, scan_noqa_directives,
    NODE_FLAGS_NOQA_PIN,
};
use crate::ingest::{decl_kind_str, file_package_name, from_module_string};

/// Stable handle for a graph node, used as the public identity across
/// the fan-out pipeline. Cross-file resolution outputs
/// ([`crate::refspec::ResolvedNode`], `walk_exports_chain`) name nodes
/// by `NodeRef`; the assembly pass translates each `NodeRef` to its
/// final graph index via `ref_to_global` after all per-file payloads
/// are collected.
///
/// `NodeRef::Def` covers every binding ty enumerates as a `Definition`,
/// keyed by an owned [`DeclKey`] (`(File, ScopedPlaceId, name_range)`)
/// rather than the `Definition<'db>` itself. The triple is the same
/// cross-file identity the assembly pass interns into `global_index`,
/// and — unlike `Definition` — it carries no `semantic_index` lifetime,
/// so storing it lets salsa evict the per-file `SemanticIndex` once the
/// payloads are built. `def_key` mints it from a `Definition`.
/// `NodeRef::Module` covers the synthetic `kind="module"` node per
/// file — `File` is itself a salsa input ingredient, so its handle is
/// stable and `Hash + Eq + Copy`.
///
/// Synthetic `[external dist] X` / `[external file] X` nodes have no
/// `NodeRef`: they only ever appear as resolution *outputs*
/// ([`crate::refspec::ResolvedNode::External`]) and are interned
/// directly into the builder by the assembly pass.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) enum NodeRef {
    Def(DeclKey),
    Module(File),
    /// The kept `*<module>` node for a `from <module> import *` statement.
    /// Keyed on `(file, <star-token range>)` rather than a ty def: the
    /// per-name `StarImport` defs back the individual `STAR_REEXPORT`
    /// nodes now, so the statement-level node it folds into needs its own
    /// identity. Carries the import payload + the module-level upstream
    /// edge, and stays the unit the codemod removes when the whole star
    /// import is unused.
    StarStmt(File, (u32, u32)),
}

/// Mint the owned [`DeclKey`] for a global-scope `Definition`: the
/// `(file, bound place, name target_range)` triple. Deterministic in
/// `def`, so a construction site and a later lookup that pass the same
/// `Definition` agree by construction — which is what lets
/// `ref_to_local` and `global_index` use the key interchangeably.
pub(crate) fn def_key<'db>(
    db: &'db dyn ProjectDb,
    def: Definition<'db>,
    parsed: &ruff_db::parsed::ParsedModuleRef,
) -> DeclKey {
    (
        def.file(db),
        def.place(db),
        range_key(def.kind(db).target_range(parsed)),
    )
}

/// Project-wide PEP 503 distribution lookup, salsa-tracked so it's
/// memoized + accessible from per-file salsa-tracked queries. Wraps
/// `crate::ingest::DistLookup` (HashMap<PathBuf, String>) with the
/// derives needed for salsa-tracked return.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct ProjectDistLookup {
    pub(crate) map: std::collections::HashMap<std::path::PathBuf, CompactString>,
}

#[salsa::tracked(returns(ref), heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn project_dist_lookup(db: &dyn ProjectDb) -> ProjectDistLookup {
    let site_packages = crate::ingest::site_packages_roots(db);
    ProjectDistLookup {
        map: crate::ingest::build_dist_lookup(&site_packages),
    }
}

/// Order-independent fingerprint of the slice of a file's payload that
/// cross-file resolution can observe: every `exports_by_name` entry
/// (the name plus the [`NodeRef`]s — including positions — it maps to)
/// and every `star_reexports` entry. Two payloads with equal
/// fingerprints give byte-identical answers to every resolution that
/// reads them (`resolve_member` / `resolve_dynamic` /
/// `walk_exports_chain` only ever look at these two maps).
///
/// Salsa-tracked, so its **invalidation is the dependency analysis**:
/// the query depends on [`file_to_nodes`], which depends on ty's
/// semantic index, which for `from X import *` reads X's exported
/// names — so editing X recomputes the fingerprint of every transitive
/// star importer automatically, with no hand-built dependency walk.
/// [`crate::project::ResolveCache`] compares each file's fingerprint
/// against the previous build's to find the *effectively changed* file
/// set that drives read-set eviction.
#[salsa::tracked(heap_size = ruff_memory_usage::heap_size)]
pub(crate) fn resolution_surface_fp(db: &dyn ProjectDb, file: File) -> u64 {
    use std::hash::{Hash, Hasher};
    let payload = file_to_nodes(db, file);
    // XOR-fold per-entry hashes so the FxHashMap iteration order
    // (which is not stable run-to-run) can't leak into the value.
    let mut acc: u64 = 0;
    for (name, locals) in &payload.exports_by_name {
        let mut h = rustc_hash::FxHasher::default();
        0u8.hash(&mut h);
        name.hash(&mut h);
        for &local in locals {
            payload.refs[local as usize].hash(&mut h);
        }
        acc ^= h.finish();
    }
    for (name, upstream) in &payload.star_reexports {
        let mut h = rustc_hash::FxHasher::default();
        1u8.hash(&mut h);
        name.hash(&mut h);
        upstream.hash(&mut h);
        acc ^= h.finish();
    }
    acc
}

/// Compute the canonical synthetic-node fqname for a non-first-party
/// target: `[external dist] {dist_name}` if dist_lookup knows the
/// file (PEP 503 canonical name), else `[external file] {top_level}`
/// for orphan site-packages files or editable installs.
pub(crate) fn external_fqname_for(
    db: &dyn ProjectDb,
    target_file: ruff_db::files::File,
    fallback_top_level: &str,
) -> CompactString {
    use ruff_db::files::FilePath;
    let path_str = match target_file.path(db) {
        FilePath::System(p) => p.to_string(),
        _ => return compact_str::format_compact!("[external file] {fallback_top_level}"),
    };
    let canonical =
        std::fs::canonicalize(&path_str).unwrap_or_else(|_| std::path::PathBuf::from(&path_str));
    let lookup = project_dist_lookup(db);
    match lookup.map.get(&canonical) {
        Some(dist_name) => compact_str::format_compact!("[external dist] {dist_name}"),
        None => compact_str::format_compact!("[external file] {fallback_top_level}"),
    }
}

/// One graph node's data without the `Py<SymbolNode>` envelope.
///
/// Pure rust so it can live inside a salsa-tracked function's return
/// value. The assembly pass converts these to `Py<SymbolNode>` in a
/// single GIL acquisition at the end of the build.
#[derive(Debug, Clone, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct NodeData {
    pub(crate) fqname: CompactString,
    pub(crate) kind: NodeKind,
    /// The file this node belongs to. Every node in a `file_to_nodes`
    /// payload shares it; the assembly pass re-derives the path string
    /// from it (one allocation per file instead of one per node).
    pub(crate) file: File,
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
    pub(crate) flags: u32,
    /// Whether this decl is an `@overload` stub. Build-time only: drives the
    /// `impl → stub` anchor-edge pass and the export-binding / fqname-trie
    /// exclusions, then is dropped — overload status is neither serialized nor
    /// Python-visible (a loaded graph behaves correctly off the anchor edges).
    pub(crate) is_overload: bool,
    pub(crate) imports: Option<ImportPayload>,
    /// Byte range of the bound *name* (ty's `DefinitionKind::target_range`),
    /// stashed so the assembly pass can rebuild its `(File, place_id,
    /// range)` decl index without re-parsing the module. `(0, 0)` for
    /// module and synthetic nodes, which the assemble pass never keys on.
    pub(crate) name_range: (u32, u32),
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
    External = 7,
}

impl NodeKind {
    // Used by the assembly pass that converts `NodeData → Py<SymbolNode>`.
    #[allow(dead_code)]
    pub(crate) fn as_static_str(self) -> &'static str {
        match self {
            NodeKind::Function => "function",
            NodeKind::Class => "class",
            NodeKind::Variable => "variable",
            NodeKind::Import => "import",
            NodeKind::TypeAlias => "type_alias",
            NodeKind::Module => "module",
            NodeKind::External => "external",
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
            "external" => NodeKind::External,
            _ => return None,
        })
    }
}

/// Per-alias import metadata. Same fields as the python-facing
/// `Import` pyclass, but without the `Py` envelope so it can live
/// inside `NodeData`.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) struct ImportPayload {
    pub(crate) module: CompactString,
    pub(crate) decl: Option<CompactString>,
    pub(crate) star: bool,
}

/// Salsa-tracked output of [`file_to_nodes`]. Parallel arrays plus a
/// reverse lookup map make every per-file node addressable in O(1)
/// either by index (assembly pass) or by `NodeRef` identity via
/// `ref_to_local` (cross-file edge resolution).
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
/// * `class_bases` — per top-level class declaration with at least one
///   statically-describable base, the symbolic base list. Each entry is
///   `(name_range_key, specs)` where `name_range_key` matches the
///   `(start, end)` `u32` pair the assemble pass stores in
///   `class_by_selection`, and `specs` is a small-vec of [`ClassBaseSpec`]
///   first-hop descriptors derived from each base expression via ty's
///   use-def chain ([`crate::helpers::classify_base`]). The spec carries
///   only this file's view — a same-file class name range, or an
///   `(absolute module, member)` pair — never a resolved cross-file
///   target, so editing the base's defining file doesn't dirty this
///   payload. Assemble resolves each spec through
///   [`crate::helpers::resolve_member_def`] and dispatches the resulting
///   `(File, name_range)` against `class_by_selection`; sibling spellings
///   collapse to one key because resolution lands them on the same
///   canonical definition. Classes with no describable base are omitted.
/// * `overload_anchors` — `(impl_local_idx, stub_local_idx)` pairs for
///   in-file `@typing.overload` groups. The assembly pass emits one
///   `impl → stub` edge per pair so reachability propagates from a
///   live impl to its stubs (and a dead impl drags its stubs along
///   for the codemod). Stubs also carry `NodeData::is_overload`
///   in `nodes` and are excluded from `exports_by_name` so cross-module
///   `from mod import f` resolves to the impl only.
#[derive(Debug, Eq, PartialEq, salsa::Update, get_size2::GetSize)]
pub(crate) struct FileNodes {
    pub(crate) nodes: Box<[NodeData]>,
    pub(crate) refs: Box<[NodeRef]>,
    pub(crate) ref_to_local: FxHashMap<NodeRef, u32>,
    pub(crate) exports_by_name: FxHashMap<CompactString, SmallVec<[u32; 2]>>,
    pub(crate) star_reexports: FxHashMap<CompactString, CompactString>,
    pub(crate) class_bases: Vec<ClassBaseEntry>,
    pub(crate) overload_anchors: Box<[(u32, u32)]>,
}

/// `(name_range_key, base_specs)` pair stored in
/// [`FileNodes::class_bases`]. The range key matches what the
/// assemble pass writes into `class_by_selection`, so the project-wide
/// fan-in is a hashmap probe once each [`ClassBaseSpec`] is resolved.
pub(crate) type ClassBaseEntry = ((u32, u32), SmallVec<[ClassBaseSpec; 2]>);

/// A symbolic, per-file description of one class base — the first hop
/// only, with no cross-file resolution, so a file's payload stays
/// invalidation-local (editing the base's defining file never dirties
/// this one). [`crate::helpers::classify_base`] derives it from the base
/// expression via ty's use-def chain; both the assemble pass and the
/// query side then resolve it through
/// [`crate::helpers::resolve_member_def`], so every spelling of one base
/// (direct, aliased, re-exported, sibling-module) lands on the same
/// `(File, name_range)` key by construction.
///
/// * `LocalClass` — the base name binds a class defined in *this* file;
///   the `(start, end)` is that class's name range, byte-identical to the
///   `class_by_selection` key the assemble pass mints for it.
/// * `ModuleMember` — the base is a member `name` of absolute module
///   `module` (a `from`-import, an attribute access on an imported
///   module, or a same-file alias of one). Resolution maps it to the
///   member's canonical definition range.
#[derive(Debug, Clone, PartialEq, Eq, salsa::Update, get_size2::GetSize)]
pub(crate) enum ClassBaseSpec {
    LocalClass((u32, u32)),
    ModuleMember {
        module: CompactString,
        name: CompactString,
    },
}

/// Modules whose `.overload` attribute marks a function decl as a stub.
const OVERLOAD_MODULES: &[&str] = &["typing", "typing_extensions"];

/// Return `true` if `dec` references `typing.overload` /
/// `typing_extensions.overload` in any of its bound forms:
///
/// * `@overload` / `@overload(...)` when `imports["overload"]` resolves
///   to `typing.overload` (or `typing_extensions.overload`).
/// * `@typing.overload` / `@typing.overload(...)` when `imports["typing"]`
///   is bound as a module (the `<module>` marker used by
///   `collect_all_imports_local` for plain `import typing`).
/// * Aliased forms such as `from typing import overload as ovl` (the
///   `imports` map already records `"ovl" -> "typing.overload"`).
fn is_overload_decorator(
    dec: &ruff_python_ast::Decorator,
    imports: &crate::helpers::LocalImports,
) -> bool {
    // Unwrap call form (`@overload()`) so we look at the callee.
    let mut expr = &dec.expression;
    while let ruff_python_ast::Expr::Call(call) = expr {
        expr = &call.func;
    }
    match expr {
        ruff_python_ast::Expr::Name(n) => imports
            .get(n.id.as_str())
            .map(|target| {
                OVERLOAD_MODULES
                    .iter()
                    .any(|m| target.as_str() == format!("{m}.overload"))
            })
            .unwrap_or(false),
        ruff_python_ast::Expr::Attribute(attr) => {
            if attr.attr.as_str() != "overload" {
                return false;
            }
            let ruff_python_ast::Expr::Name(prefix) = attr.value.as_ref() else {
                return false;
            };
            imports
                .get(prefix.id.as_str())
                .map(|target| OVERLOAD_MODULES.iter().any(|m| target == m))
                .unwrap_or(false)
        }
        _ => false,
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
pub(crate) fn file_to_nodes(db: &dyn ProjectDb, file: File) -> FileNodes {
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    let module_fqname = module_fqname_for_file(db, file);
    let default_flags = file_default_flags(db, file);
    let (file_pinned_by_noqa, per_line_noqa_pins) =
        scan_noqa_directives(&parsed, &source, &line_index);

    let (msl, msc, mel, mec) = position(&line_index, &source, parsed.syntax().range);

    let mut nodes: Vec<NodeData> = Vec::new();
    let mut refs: Vec<NodeRef> = Vec::new();
    let mut ref_to_local: FxHashMap<NodeRef, u32> = FxHashMap::default();

    // Index 0: the synthetic module node. Anchors reachability for
    // every per-file decl via the `decl → module` anchor edge the
    // assembly pass synthesizes.
    nodes.push(NodeData {
        fqname: module_fqname.clone(),
        kind: NodeKind::Module,
        file,
        start_line: msl,
        start_column: msc,
        end_line: mel,
        end_column: mec,
        flags: default_flags,
        is_overload: false,
        imports: None,
        name_range: (0, 0),
    });
    refs.push(NodeRef::Module(file));
    ref_to_local.insert(NodeRef::Module(file), 0);

    let index = semantic_index(db, file).load(db);
    let global = FileScopeId::global();
    let place_table = index.place_table(global);
    let use_def_map = index.use_def_map(global);

    // Star imports: ty mints one Definition per name brought in by
    // `from X import *`, each kept as its own `STAR_REEXPORT` node. Every
    // per-name def from the same statement shares the `*` token's range,
    // so this map (range → `*<src>` fqname) doubles as the "already
    // minted the statement-level `*<src>` node?" guard — `insert`
    // returning `None` means this is the first name of the statement.
    let mut star_local_name_cache: HashMap<TextRange, String> = HashMap::new();

    let mut star_reexports: FxHashMap<CompactString, CompactString> = FxHashMap::default();

    // Pre-scan top-level `Stmt::FunctionDef` decorator lists to identify
    // `@typing.overload`-decorated function defs. The set is keyed on the
    // function's name TextRange (which is what `DefinitionKind::Function`
    // reports as its `target_range`), so we can flag the matching decl
    // when ty enumerates it below.
    let file_package = file_package_name(db, file);
    let local_imports = collect_all_imports_local(&parsed, file_package.as_deref());
    let mut overload_decorated: FxHashSet<TextRange> = FxHashSet::default();
    for stmt in &parsed.syntax().body {
        if let Stmt::FunctionDef(func) = stmt {
            if func
                .decorator_list
                .iter()
                .any(|d| is_overload_decorator(d, &local_imports))
            {
                overload_decorated.insert(func.name.range());
            }
        }
    }

    // Statically-dead statement ranges (`if False:` bodies, post-`return`
    // suites, …). A decl whose name sits inside one is stamped
    // `NodeFlags::DEAD_BRANCH` — the node-level companion to the edge flag
    // of the same name. Metadata only (not a seed); recorded for the
    // explorer / blast-radius queries.
    let dead_ranges = detect_dead_ranges(&parsed);

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
        let per_name = symbol.name().as_str().to_compact_string();

        let target_range = kind.target_range(&parsed);
        let is_star = matches!(kind, DefinitionKind::StarImport(_));
        // Per-name star bindings keep their real name now (one
        // `STAR_REEXPORT` node per exported name); the statement-level
        // `*<src>` node is minted separately, once, below.
        let local_name = per_name.clone();

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
        let import_noqa_pin = matches!(node_kind, NodeKind::Import)
            && (file_pinned_by_noqa || per_line_noqa_pins.contains(&sl));
        if import_noqa_pin {
            flags |= NODE_FLAGS_NOQA_PIN;
        }
        // `@typing.overload`-decorated function defs: record now so the
        // post-pass can partition same-name groups into impl / stubs.
        // The set is keyed on the function's name TextRange, which is
        // exactly what `DefinitionKind::Function::target_range` returns.
        let is_overload =
            matches!(node_kind, NodeKind::Function) && overload_decorated.contains(&target_range);
        // Per-name implicit star bindings: flag so the codemod skips
        // them. They share the `*` token's range and can't be removed
        // individually; the `*<src>` statement node (minted below, no
        // `STAR_REEXPORT` flag) stays the removable unit.
        if is_star {
            flags |= NodeFlags::STAR_REEXPORT;
        }
        // Decl sitting in a statically-dead region — node-level companion
        // to `EdgeFlags::DEAD_BRANCH`.
        if dead_ranges.iter().any(|r| r.contains_range(target_range)) {
            flags |= NodeFlags::DEAD_BRANCH;
        }
        // Stub-only ENTRYPOINT flagging is deferred to the assembly
        // pass; it needs the project-wide peer-stub map and the .py
        // twin's `exports_by_name`, which aren't visible to this
        // per-file query.

        let node = NodeData {
            fqname: compact_str::format_compact!("{module_fqname}.{local_name}"),
            kind: node_kind,
            file,
            start_line: sl,
            start_column: sc,
            end_line: el,
            end_column: ec,
            flags,
            is_overload,
            imports: import_spec.clone(),
            name_range: (target_range.start().to_u32(), target_range.end().to_u32()),
        };
        // Same triple `def_key` would mint, reusing the `place_id` /
        // `target_range` already computed above.
        let key = NodeRef::Def((file, place_id, range_key(target_range)));
        let local_idx = nodes.len() as u32;
        nodes.push(node);
        refs.push(key);
        ref_to_local.insert(key, local_idx);

        // Mint the statement-level `*<src>` node once per `from <src>
        // import *` (keyed on the shared `*` token range). It carries the
        // import payload + the module-level upstream edge, and stays the
        // codemod's removable unit — so it gets the import flags but NOT
        // `STAR_REEXPORT`. The per-name nodes above edge into it.
        if is_star {
            if let Some(spec) = &import_spec {
                let star_fqname = format!("{module_fqname}.*{}", spec.module);
                if star_local_name_cache
                    .insert(target_range, star_fqname.clone())
                    .is_none()
                {
                    let mut star_flags: u32 = default_flags;
                    if import_noqa_pin {
                        star_flags |= NODE_FLAGS_NOQA_PIN;
                    }
                    let star_node = NodeData {
                        fqname: star_fqname.into(),
                        kind: NodeKind::Import,
                        file,
                        start_line: sl,
                        start_column: sc,
                        end_line: el,
                        end_column: ec,
                        flags: star_flags,
                        is_overload: false,
                        imports: import_spec.clone(),
                        name_range: (target_range.start().to_u32(), target_range.end().to_u32()),
                    };
                    let star_key = NodeRef::StarStmt(file, range_key(target_range));
                    let star_idx = nodes.len() as u32;
                    nodes.push(star_node);
                    refs.push(star_key);
                    ref_to_local.insert(star_key, star_idx);
                }
            }
        }

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

    // Overload-anchor post-pass: for each same-name group of top-level
    // function defs containing both `@overload`-decorated stubs AND a
    // non-overload impl, pair each stub with the *last* non-overload
    // decl (CPython's runtime semantics — the impl is the final one).
    // Stubs that aren't anchored to an impl (e.g. a .pyi stub file
    // where every `def f` is `@overload`) stay `is_overload` but
    // emit no anchor; reachability falls back to the module-anchor
    // edge.
    let mut overload_anchors: Vec<(u32, u32)> = Vec::new();
    let mut by_name: FxHashMap<&str, Vec<u32>> = FxHashMap::default();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.kind, NodeKind::Function) {
            continue;
        }
        let simple = node
            .fqname
            .rsplit_once('.')
            .map(|p| p.1)
            .unwrap_or(&node.fqname);
        by_name.entry(simple).or_default().push(i as u32);
    }
    for group in by_name.values() {
        // Find the last non-overload entry — that's the impl. Groups
        // without any non-overload decl emit no anchors (e.g. a .pyi
        // stub file whose every `def f` is `@overload`).
        let Some(&impl_idx) = group
            .iter()
            .rev()
            .find(|&&i| !nodes[i as usize].is_overload)
        else {
            continue;
        };
        for &i in group {
            if nodes[i as usize].is_overload {
                overload_anchors.push((impl_idx, i));
            }
        }
    }

    // Post-pass: populate `exports_by_name` from ty's end-of-scope
    // symbol bindings. This collapses sequential rebinds to the latest
    // while preserving branch-bound multi-binding cases (try/except,
    // if/else where each branch assigns). Mirrors `globals_by_name` in
    // today's pipeline. Values are indices into `nodes` (≥ 1, since
    // index 0 is the module node which can't be bound to a name).
    // Overload-flagged decls are excluded so `from mod import f`
    // resolves to the impl only — the impl `def f` shadows them via
    // ty's end-of-scope binding anyway, but pyi stub files (every
    // `def f` decorated `@overload`) need the explicit filter.
    let mut exports_by_name: FxHashMap<CompactString, SmallVec<[u32; 2]>> = FxHashMap::default();
    for (symbol_id, bindings) in use_def_map.all_end_of_scope_symbol_bindings() {
        let PlaceExprRef::Symbol(sym) = place_table.place(ScopedPlaceId::Symbol(symbol_id)) else {
            continue;
        };
        let name = sym.name().as_str().to_compact_string();
        let mut live: SmallVec<[u32; 2]> = SmallVec::new();
        for binding in bindings {
            let Some(def) = binding.binding.definition() else {
                continue;
            };
            if def.file(db) != file || def.file_scope(db) != global {
                continue;
            }
            if let Some(&local_idx) = ref_to_local.get(&NodeRef::Def(def_key(db, def, &parsed))) {
                if nodes[local_idx as usize].is_overload {
                    continue;
                }
                live.push(local_idx);
            }
        }
        if !live.is_empty() {
            exports_by_name.insert(name, live);
        }
    }

    // Describe every top-level class base symbolically via ty's use-def
    // chain: each base expression becomes a first-hop `ClassBaseSpec`
    // (this file's view only — a same-file class range or an
    // `(absolute module, member)` pair), never an eagerly-resolved
    // cross-file target. Classes with no describable base contribute
    // nothing; resolution is deferred to the assemble/query side, which
    // keeps this payload invalidation-local.
    let class_bases = build_class_bases(db, file, &parsed);

    FileNodes {
        nodes: nodes.into_boxed_slice(),
        refs: refs.into_boxed_slice(),
        ref_to_local,
        exports_by_name,
        star_reexports,
        class_bases,
        overload_anchors: overload_anchors.into_boxed_slice(),
    }
}

/// Per-file class-hierarchy producer used by [`file_to_nodes`].
///
/// For each top-level `ClassDef` with at least one statically-describable
/// base, key on its name range and describe every base expression
/// symbolically via [`crate::helpers::classify_base`] — ty's use-def
/// chain, this file's scope only. Each base becomes a [`ClassBaseSpec`]:
///
/// * a same-file class name → `LocalClass(range)` (the
///   `class_by_selection` key the assemble pass mints for that class);
/// * a `from`-import, an attribute access on an imported module
///   (`mod.Base`, `a.b.Base`), or a same-file alias chain that bottoms
///   out at one → `ModuleMember { module, name }` with an *absolute*
///   module string.
///
/// `Base[T]` is unwrapped to `Base` inside `classify_base`. Anything that
/// doesn't bottom out at a class name or an importable member (a dynamic
/// base, a bare imported module, a union) yields no spec and is dropped;
/// a class left with no specs is omitted entirely. No cross-file
/// resolution happens here — that's deferred to
/// [`crate::helpers::resolve_member_def`] at assemble/query time.
fn build_class_bases(
    db: &dyn ProjectDb,
    file: File,
    parsed: &ruff_db::parsed::ParsedModuleRef,
) -> Vec<ClassBaseEntry> {
    let mut out: Vec<ClassBaseEntry> = Vec::new();
    for cls in iter_top_level_classes(parsed) {
        let Some(args) = &cls.arguments else {
            continue;
        };
        let mut bases: SmallVec<[ClassBaseSpec; 2]> = SmallVec::new();
        for raw_base in &args.args {
            if let Some(spec) = classify_base(db, file, parsed, raw_base, 0) {
                bases.push(spec);
            }
        }
        if !bases.is_empty() {
            // Key on the class name's `(start, end)` byte range — same
            // tuple the assemble pass uses to populate `class_by_selection`,
            // so the assemble-side fan-in is an O(1) hashmap probe.
            out.push((range_key(cls.name.range()), bases));
        }
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
pub(crate) fn walk_exports_chain(
    db: &dyn ProjectDb,
    target_file: File,
    symbol_name: &str,
    touched: &mut crate::refspec::Touched,
) -> Vec<NodeRef> {
    let mut seen: std::collections::HashSet<(File, String)> = std::collections::HashSet::new();
    let mut out: Vec<NodeRef> = Vec::new();
    let mut stack: Vec<(File, String)> = vec![(target_file, symbol_name.to_string())];
    while let Some(key) = stack.pop() {
        if !seen.insert(key.clone()) {
            continue;
        }
        // Every chain hop is a content read — record it so the resolve
        // cache's read sets cover intermediary star-reexport files, not
        // just the final targets.
        touched.record(key.0);
        let target_nodes = file_to_nodes(db, key.0);
        // If `key` is a `from <upstream> import *` reexport, step
        // into the upstream file's same-name lookup. The star alias
        // itself isn't useful as a target — uses should land on the
        // upstream decl. Skip emitting it here.
        if let Some(upstream_module) = target_nodes.star_reexports.get(key.1.as_str()) {
            if let Some(mn) = ty_module_resolver::ModuleName::new(upstream_module) {
                if let Some(upstream) = ty_module_resolver::resolve_module(db, key.0, &mn) {
                    if let Some(upstream_file) = upstream.file(db) {
                        stack.push((upstream_file, key.1.clone()));
                        continue;
                    }
                }
            }
        }
        let Some(locals) = target_nodes.exports_by_name.get(key.1.as_str()) else {
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
                module: alias.name.id.as_str().to_compact_string(),
                decl: None,
                star: false,
            }
        }
        DefinitionKind::ImportFrom(k) => {
            let alias = k.alias(parsed);
            ImportPayload {
                module: from_module_string(db, file, k.import(parsed)),
                decl: Some(alias.name.id.as_str().to_compact_string()),
                star: false,
            }
        }
        DefinitionKind::ImportFromSubmodule(k) => ImportPayload {
            module: from_module_string(db, file, k.import(parsed)),
            decl: Some(k.module(parsed).id.as_str().to_compact_string()),
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
// (file_to_edges removed)
//
// The old `file_to_edges` query — import-alias upstream edges, decl →
// module anchors, overload anchors, the module-hierarchy edge — was
// folded into the combined `file_to_refspecs` walk
// (`crate::file_ref_edges`) and the assembly pass:
//
// * import-alias upstream resolution is now a `Member(Binding)`
//   [`crate::refspec::RefSpec`] resolved project-wide at assembly
//   (`crate::refspec::resolve_binding` carries the `_handle_fromlist`
//   namespace-vs-submodule disambiguation and the unresolved-alias
//   flagging that lived here);
// * decl → module anchors, overload anchors, and the submodule →
//   parent-package hierarchy edge are synthesized by the assembly
//   pass straight from `FileNodes` / [`parent_module_file`], since
//   they're fully derivable and would only bloat the payload.
// ---------------------------------------------------------------------------

/// Resolve a file's parent module to its `File` handle, when the
/// parent's `__init__.py` is itself in the project. Mirrors today's
/// `emit_module_hierarchy::parent_module_edge`. Returns `None` for
/// top-level packages (no parent) and for parents that live outside
/// the project (the cross-file hierarchy edge has nowhere to land).
pub(crate) fn parent_module_file(db: &dyn ProjectDb, file: File) -> Option<File> {
    let module = crate::helpers::canonical_module_for_file(db, file)?;
    let parent_name = module.name(db).parent()?;
    let parent_module = resolve_module(db, file, &parent_name)?;
    parent_module.file(db)
}
