//! File-local reference specs and their assembly-side resolution.
//!
//! [`RefSpec`] is the unit the combined per-file walk
//! ([`crate::file_ref_edges::file_to_refspecs`]) emits: one unresolved
//! reference edge, expressed entirely in the owning file's local terms.
//! Crucially the type is `'static` plain data — no interned keys —
//! so the walk query has **zero cross-file salsa
//! dependencies**: editing an imported file's contents never
//! invalidates an importer's walk. All cross-file work (ty's module
//! resolver, upstream `exports_by_name` lookups, star-reexport chain
//! walks, external classification) happens in the assembly pass via
//! [`resolve_member`] / [`resolve_dynamic`], memoized project-wide on
//! `(anchor directory, ref)` keys.
//!
//! This mirrors the existing [`ClassBaseSpec`](crate::file_payload::ClassBaseSpec)
//! pattern — a symbolic per-file descriptor resolved at fan-in — and
//! generalizes it to every reference edge.

use ruff_db::files::File;
use smallvec::SmallVec;
use ty_module_resolver::{resolve_module, ModuleName};
use ty_project::Db as ProjectDb;

use crate::file_payload::{
    external_fqname_for, file_to_nodes, walk_exports_chain, ImportPayload, NodeRef,
};
use crate::graph::DeclKey;
use crate::ingest::module_name_resolves;

/// One unresolved reference edge, expressed entirely in this file's
/// local terms. `src` is the local index (into
/// [`FileNodes::refs`](crate::file_payload::FileNodes::refs)) of the
/// owning node; `flags` are the edge flags (dead-branch /
/// dynamic-import) computed locally. The `target` is resolved by the
/// assembly pass — see [`Target`].
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) struct RefSpec {
    pub(crate) src: u32,
    pub(crate) target: Target,
    pub(crate) flags: u8,
}

/// A [`RefSpec`]'s destination.
///
/// * `Local` — already resolved: a local index in *this* file (a
///   use→alias edge, a module→dunder edge, an `__all__` edge, a
///   per-name star alias → star statement edge). The assembly pass
///   maps both endpoints through the file's node-offset base.
/// * `Member` — an unresolved cross-file member/module access. The
///   assembly pass runs [`resolve_member`] to turn it into
///   zero-or-more upstream nodes.
/// * `Dynamic` — an unresolved `__import__` / `importlib.import_module`
///   target. The assembly pass runs [`resolve_dynamic`].
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) enum Target {
    Local(u32),
    Member(MemberRef),
    Dynamic(DynamicRef),
}

/// Which side of the import-edge contract a [`MemberRef`] encodes.
/// The two roles share the module-resolution front-end but
/// deliberately land differently (see `CLAUDE.md` principle 2):
///
/// * `Binding` — the import alias's own upstream edge (`alias →
///   Module(target)` + `alias → target decl`). Resolution walks
///   star-reexport chains past intermediary star aliases
///   ([`walk_exports_chain`]) so the alias reaches the *real* decl,
///   applies CPython's `_handle_fromlist` namespace-vs-submodule
///   disambiguation, and reports resolution failure so the assembly
///   pass can stamp `NodeFlags::UNRESOLVED` on the alias.
/// * `Use` — a use site flowing through an import alias (or a
///   nested-context import). Resolution lands *on* whatever is in the
///   upstream namespace — including star-reexport aliases, which the
///   use keeps alive — and silently drops failures.
#[derive(
    Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, salsa::Update, get_size2::GetSize,
)]
pub(crate) enum MemberRole {
    Binding,
    Use,
}

/// Unresolved descriptor for a reference that flows through an import.
/// For `Use` it mirrors the `(spec, bound_name, extra_chain)` arguments
/// the old `emit_upstream` took; for `Binding` it carries the per-kind
/// `(module, decl)` resolution input the old `file_to_edges` alias loop
/// computed (with `bound_name` empty and `chain` unused).
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) struct MemberRef {
    pub(crate) role: MemberRole,
    pub(crate) spec: ImportPayload,
    pub(crate) bound_name: String,
    pub(crate) chain: Vec<String>,
}

/// Unresolved descriptor for a dynamic import. `target` is already
/// absolutized at walk time via `resolve_dynamic_target` (which needs
/// only the owning file's package name — a self-property), so
/// resolution here is anchor-light: module probes + upstream
/// `exports_by_name` lookups.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, salsa::Update, get_size2::GetSize)]
pub(crate) struct DynamicRef {
    pub(crate) target: String,
    pub(crate) fromlist: Vec<String>,
}

/// A resolved cross-file target, in `'static` terms, so resolution can
/// run on rayon workers against per-worker db snapshots and hand
/// results back to the main thread.
///
/// `External` defers the graph-node interning to the serial
/// translation step — workers only read.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum ResolvedNode {
    Module(File),
    Def(DeclKey),
    StarStmt(File, (u32, u32)),
    External { fqname: String, file: File },
}

impl ResolvedNode {
    fn from_node_ref(r: NodeRef) -> Self {
        match r {
            NodeRef::Module(f) => ResolvedNode::Module(f),
            NodeRef::Def(k) => ResolvedNode::Def(k),
            NodeRef::StarStmt(f, rk) => ResolvedNode::StarStmt(f, rk),
        }
    }
}

/// Output of [`resolve_member`].
///
/// * `targets` — the upstream nodes the ref reaches.
/// * `unresolved` — `Binding` only: the upstream module didn't resolve;
///   the assembly pass ORs `NodeFlags::UNRESOLVED` into the alias.
/// * `start_file` — the resolved upstream module's file, when it
///   resolved to a first-party file. The assembly pass uses it to
///   suppress self-imports (`Binding` refs whose target is the
///   importing file itself emit nothing — mirrors the old
///   `file_to_edges` `target_file == file` skip, which must be applied
///   per importing file, not per memo entry).
#[derive(Debug, Clone, Default)]
pub(crate) struct ResolvedMember {
    pub(crate) targets: SmallVec<[ResolvedNode; 4]>,
    pub(crate) unresolved: bool,
    pub(crate) start_file: Option<File>,
}

/// Resolve a [`MemberRef`] from `anchor`. `anchor` only feeds ty's
/// module resolver, whose anchor-sensitivity is limited to the
/// desperate-resolution fallback (ancestor-directory search) — which is
/// why the assembly pass may memoize results per *directory* rather
/// than per file.
pub(crate) fn resolve_member(
    db: &dyn ProjectDb,
    anchor: File,
    member: &MemberRef,
) -> ResolvedMember {
    match member.role {
        MemberRole::Binding => resolve_binding(db, anchor, &member.spec),
        MemberRole::Use => resolve_use(db, anchor, member),
    }
}

/// `Binding` arm: port of the old `file_to_edges` import-alias
/// resolution. Emits `Module(target)` + (for from-imports) the decl
/// reached through [`walk_exports_chain`]; classifies stdlib (silent),
/// site-packages (External), and unresolved (flag) targets.
fn resolve_binding(db: &dyn ProjectDb, anchor: File, spec: &ImportPayload) -> ResolvedMember {
    let mut out = ResolvedMember::default();
    if spec.module.is_empty() {
        return out;
    }

    // Submodule disambiguation: `from p import functions` where
    // `p.functions` is itself a submodule should point at the
    // submodule file, not at `p` with a decl lookup. BUT: per
    // CPython's `_handle_fromlist`, namespace bindings WIN over
    // submodule lookups when both exist (e.g. `q = 42` in
    // p/__init__.py with a sibling p/q.py — the int binds, not
    // the submodule). Check namespace first; only switch to
    // submodule when the namespace doesn't already export decl.
    let (target_module_str, decl_name): (String, Option<String>) = match &spec.decl {
        Some(decl) => {
            let candidate = format!("{}.{}", spec.module, decl);
            if module_name_resolves(&candidate, anchor, db) {
                let namespace_has_decl = ModuleName::new(&spec.module)
                    .and_then(|mn| resolve_module(db, anchor, &mn))
                    .and_then(|m| m.file(db))
                    .map(|f| file_to_nodes(db, f).exports_by_name.contains_key(decl))
                    .unwrap_or(false);
                if namespace_has_decl {
                    (spec.module.clone(), Some(decl.clone()))
                } else {
                    (candidate, None)
                }
            } else {
                (spec.module.clone(), Some(decl.clone()))
            }
        }
        None => (spec.module.clone(), None),
    };

    // Bad module name (relative-dots overflow, etc.), a module that
    // doesn't resolve, or a resolved module with no backing file:
    // report `unresolved` so the assembly pass flags the alias.
    let Some(target_module_name) = ModuleName::new(&target_module_str) else {
        out.unresolved = true;
        return out;
    };
    let Some(target_module) = resolve_module(db, anchor, &target_module_name) else {
        out.unresolved = true;
        return out;
    };
    let Some(target_file) = target_module.file(db) else {
        out.unresolved = true;
        return out;
    };
    let top_level = target_module_str
        .split('.')
        .next()
        .unwrap_or(&target_module_str);
    // Stdlib targets emit no node: an unused stdlib import is still
    // caught dead via its alias's zero in-edge count, so a stdlib
    // endpoint would be pure noise.
    if target_module
        .search_path(db)
        .is_some_and(|sp| sp.is_standard_library())
    {
        return out;
    }
    // Non-first-party (site-packages) → External node keyed on the
    // PEP 503 dist name (or top-level module name) with the resolved
    // target file (its path is re-derived at assembly).
    if target_module
        .search_path(db)
        .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
    {
        let fqname = external_fqname_for(db, target_file, top_level);
        out.targets.push(ResolvedNode::External {
            fqname,
            file: target_file,
        });
        return out;
    }
    // First-party target: record `start_file` so the assembly pass can
    // suppress self-imports (the whole emission, decl edges included —
    // mirroring the old `target_file == file { continue; }`).
    out.start_file = Some(target_file);

    // Module-level edge.
    out.targets.push(ResolvedNode::Module(target_file));

    // Named-decl edge for from-imports. Walks the star-reexport
    // chain so `from A import g` where A re-exports from B lands
    // on B's real `g` rather than A's star alias node. salsa
    // memoizes each file's payload on the way through, so the
    // chain walk is hashmap lookups + a `seen` set.
    if let Some(decl_name) = decl_name {
        for target_ref in walk_exports_chain(db, target_file, &decl_name) {
            out.targets.push(ResolvedNode::from_node_ref(target_ref));
        }
    }
    out
}

/// `Use` arm: port of the old `RefWalker::emit_upstream`. Classifies
/// the loading target, walks the submodule chain, lands on the deepest
/// module reached plus any terminal decl — *without* skipping past
/// star-reexport aliases (the use side keeps the intermediary star
/// import alive; see [`MemberRole`]).
fn resolve_use(db: &dyn ProjectDb, anchor: File, member: &MemberRef) -> ResolvedMember {
    let mut out = ResolvedMember::default();
    let spec = &member.spec;
    let bound_name = member.bound_name.as_str();
    if spec.module.is_empty() {
        return out;
    }
    let module_first_seg = spec.module.split('.').next().unwrap_or("").to_string();

    let mut adjusted_chain: Vec<&str> = member.chain.iter().map(String::as_str).collect();
    let loading_target: String;
    let mut decl_tail: Option<String> = None;

    if spec.star {
        let candidate = format!("{}.{}", spec.module, bound_name);
        if module_name_resolves(&candidate, anchor, db) {
            loading_target = candidate;
        } else {
            loading_target = spec.module.clone();
            decl_tail = Some(bound_name.to_string());
        }
    } else {
        match &spec.decl {
            Some(decl) => {
                let candidate = format!("{}.{}", spec.module, decl);
                if module_name_resolves(&candidate, anchor, db) {
                    loading_target = candidate;
                } else {
                    loading_target = spec.module.clone();
                    decl_tail = Some(decl.clone());
                }
            }
            None => {
                let no_asname = bound_name == module_first_seg;
                if no_asname && spec.module != module_first_seg {
                    let loading_extras: Vec<&str> = spec.module.split('.').skip(1).collect();
                    let n = loading_extras.len();
                    let prefix_matches = adjusted_chain.len() >= n
                        && adjusted_chain
                            .iter()
                            .take(n)
                            .zip(&loading_extras)
                            .all(|(a, b)| *a == *b);
                    if prefix_matches {
                        adjusted_chain.drain(..n);
                        loading_target = spec.module.clone();
                    } else {
                        loading_target = module_first_seg;
                    }
                } else {
                    loading_target = spec.module.clone();
                }
            }
        }
    }

    // Classify the loading target. Site-packages targets land on a
    // single External node carrying the upstream path and stop (no
    // submodule chain walk through them, since they don't have
    // file_to_nodes payloads to look up against). Stdlib and
    // genuinely-unresolved targets emit nothing.
    let top_level = loading_target.split('.').next().unwrap_or(&loading_target);
    let Some(start_mn) = ModuleName::new(&loading_target) else {
        return out;
    };
    let Some(start_module) = resolve_module(db, anchor, &start_mn) else {
        return out;
    };
    let Some(start_file) = start_module.file(db) else {
        return out;
    };
    if start_module
        .search_path(db)
        .is_some_and(|sp| sp.is_standard_library())
    {
        return out;
    }
    if start_module
        .search_path(db)
        .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
    {
        let fqname = external_fqname_for(db, start_file, top_level);
        out.targets.push(ResolvedNode::External {
            fqname,
            file: start_file,
        });
        return out;
    }

    // Decl-style alias: land on the upstream module and the decl
    // inside it. Attribute access past a decl is field access on the
    // decl's value, which we don't model.
    //
    // Use sites land on whatever's in exports_by_name (including
    // star-reexport aliases). Don't walk through star aliases here —
    // the from-import binding side uses walk_exports_chain to skip
    // past stars; this is the *use* side, which should reach the star
    // alias itself.
    if let Some(decl_name) = decl_tail {
        out.targets.push(ResolvedNode::Module(start_file));
        let target_nodes = file_to_nodes(db, start_file);
        if let Some(locals) = target_nodes.exports_by_name.get(&decl_name) {
            for &local_idx in locals {
                out.targets.push(ResolvedNode::from_node_ref(
                    target_nodes.refs[local_idx as usize],
                ));
            }
        }
        return out;
    }

    // Module-style alias: walk the chain submodule-by-submodule,
    // land on one module (deepest reached) plus at most one
    // terminal decl.
    let mut current_file = start_file;
    let mut current_path = loading_target.clone();
    let mut terminal_decl_refs: Vec<ResolvedNode> = Vec::new();
    for seg in &adjusted_chain {
        let candidate = format!("{current_path}.{seg}");
        let submodule_file = ModuleName::new(&candidate)
            .and_then(|mn| resolve_module(db, anchor, &mn))
            .and_then(|m| m.file(db));
        if let Some(sub_file) = submodule_file {
            current_file = sub_file;
            current_path = candidate;
            continue;
        }
        // Not a submodule — check for decl in current_file's exports.
        let target_nodes = file_to_nodes(db, current_file);
        if let Some(locals) = target_nodes.exports_by_name.get(*seg) {
            for &local_idx in locals {
                terminal_decl_refs.push(ResolvedNode::from_node_ref(
                    target_nodes.refs[local_idx as usize],
                ));
            }
        }
        break;
    }
    out.targets.push(ResolvedNode::Module(current_file));
    out.targets.extend(terminal_decl_refs);
    out
}

/// Resolve a [`DynamicRef`] from `anchor`: the target module itself
/// plus per-fromlist-entry edges (submodule-or-decl, mirroring
/// CPython's `_handle_fromlist`). Port of the old
/// `RefWalker::emit_dynamic_edges` + `emit_resolved_module`.
pub(crate) fn resolve_dynamic(
    db: &dyn ProjectDb,
    anchor: File,
    dynamic: &DynamicRef,
) -> SmallVec<[ResolvedNode; 4]> {
    let mut out: SmallVec<[ResolvedNode; 4]> = SmallVec::new();
    resolve_module_target(db, anchor, &dynamic.target, &mut out);
    for entry in &dynamic.fromlist {
        if entry.is_empty() {
            continue;
        }
        let candidate = format!("{}.{entry}", dynamic.target);
        if module_name_resolves(&candidate, anchor, db) {
            resolve_module_target(db, anchor, &candidate, &mut out);
            continue;
        }
        // Treat as decl-style: resolve target to file, look up
        // entry in its exports_by_name.
        let target_file = ModuleName::new(&dynamic.target)
            .and_then(|n| resolve_module(db, anchor, &n))
            .and_then(|m| m.file(db));
        if let Some(target_file) = target_file {
            let target_nodes = file_to_nodes(db, target_file);
            if let Some(locals) = target_nodes.exports_by_name.get(entry.as_str()) {
                for &local_idx in locals {
                    out.push(ResolvedNode::from_node_ref(
                        target_nodes.refs[local_idx as usize],
                    ));
                }
            }
        }
    }
    out
}

/// Resolve one dotted module path. First-party modules land on a
/// `Module` node; site-packages modules become External nodes carrying
/// the upstream path; stdlib and genuinely-unresolved targets land
/// nothing.
fn resolve_module_target(
    db: &dyn ProjectDb,
    anchor: File,
    dotted: &str,
    out: &mut SmallVec<[ResolvedNode; 4]>,
) {
    let top_level = dotted.split('.').next().unwrap_or(dotted);
    let Some(mn) = ModuleName::new(dotted) else {
        return;
    };
    let Some(module) = resolve_module(db, anchor, &mn) else {
        return;
    };
    let Some(target_file) = module.file(db) else {
        return;
    };
    if module
        .search_path(db)
        .is_some_and(|sp| sp.is_standard_library())
    {
        return;
    }
    if module
        .search_path(db)
        .is_some_and(|sp| !sp.is_first_party() && sp.is_site_packages())
    {
        let fqname = external_fqname_for(db, target_file, top_level);
        out.push(ResolvedNode::External {
            fqname,
            file: target_file,
        });
        return;
    }
    out.push(ResolvedNode::Module(target_file));
}
