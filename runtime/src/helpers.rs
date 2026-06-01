//! Leftover utilities shared across modules: AST navigation helpers,
//! call-args extraction used by the native plugin query helpers,
//! dead-region detection, noqa scanning, position/file helpers, and the
//! shared `NodeFlags`/`EdgeFlags` constant aliases.

use pyo3::prelude::*;
use ruff_db::files::{File, FilePath};
use ruff_db::parsed::{parsed_module, ParsedModuleRef};
use ruff_db::source::{line_index, source_text};
use ruff_db::system::SystemPath;
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::visitor::{walk_expr, Visitor};
use ruff_python_ast::{Expr, Stmt, StmtClassDef};
use ruff_source_file::LineIndex;
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use ty_module_resolver::Module;
use ty_module_resolver::{
    file_to_module, resolve_module, search_paths, ModuleName, ModuleResolveMode,
};
use ty_project::metadata::value::RelativePathBuf;
use ty_project::{Db as ProjectDb, ProjectDatabase};

use crate::builder::GraphNode;
use crate::graph::{EdgeFlags, NodeFlags};
use crate::ingest::collapse_attribute_chain;
use crate::project::BuildOutputs;

/// Sentinel value stored in a file's local imports map (the value half
/// of ``{local_name: target}`` returned by
/// [`collect_module_imports_local`]) when the local binding is the
/// module object itself (`import flask` / `import flask as f`), not a
/// specific name from inside it. Matchers test against this exact
/// string when classifying attribute-style decorator / call references.
pub(crate) const MODULE_ALIAS_MARKER: &str = "<module>";

pub(crate) fn is_dunder_name(fqname: &str) -> bool {
    let name = fqname.rsplit('.').next().unwrap_or("");
    name.len() > 4 && name.starts_with("__") && name.ends_with("__")
}

pub(crate) fn class_body_defines_method(class_def: &StmtClassDef, method_name: &str) -> bool {
    class_def.body.iter().any(|stmt| match stmt {
        Stmt::FunctionDef(f) => f.name.as_str() == method_name,
        _ => false,
    })
}

/// Return ``true`` if any decorator in ``decorators`` matches
/// ``@<module>.<name>`` (literal module name) or ``@<name>`` for
/// any ``name`` in ``names``. Trailing ``(...)`` is unwrapped before
/// the pattern check.
/// Resolves decorator references through the file's local imports map
/// (built by :func:`collect_module_imports_local`) — mirrors the libcst
/// helpers' ``matched_attr_call`` shape. Recognized forms:
///
/// * ``@<name>(...)`` / ``@<name>`` where ``imports[name]`` is in ``names``
///   (covers ``from module import name`` and aliased variants).
/// * ``@<alias>.<attr>(...)`` / ``@<alias>.<attr>`` where
///   ``imports[alias] == MODULE_ALIAS_MARKER`` and ``attr`` is in ``names``
///   (covers ``import module`` and ``import module as alias``).
pub(crate) fn decorators_match_imports<'ast>(
    decorators: &'ast [ruff_python_ast::Decorator],
    imports: &FxHashMap<String, String>,
    names: &FxHashSet<&str>,
) -> Option<Option<&'ast ruff_python_ast::ExprCall>> {
    for dec in decorators {
        let (root_expr, call_form) = match &dec.expression {
            Expr::Call(call) => (&*call.func, Some(call)),
            other => (other, None),
        };
        // Unwrap subscripted-generic callees so ``@route[T]`` /
        // ``@route[T]()`` are classified the same as ``@route`` /
        // ``@route()``.
        let root_expr = unwrap_subscripted_callee(root_expr);
        match root_expr {
            Expr::Name(n) => {
                if let Some(target) = imports.get(n.id.as_str()) {
                    if names.contains(target.as_str()) {
                        return Some(call_form);
                    }
                }
            }
            Expr::Attribute(attr) => {
                if let Expr::Name(prefix) = attr.value.as_ref() {
                    if imports.get(prefix.id.as_str()).map(String::as_str)
                        == Some(MODULE_ALIAS_MARKER)
                        && names.contains(attr.attr.as_str())
                    {
                        return Some(call_form);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn iter_top_level_classes(
    parsed: &ParsedModuleRef,
) -> impl Iterator<Item = &StmtClassDef> {
    parsed.syntax().body.iter().filter_map(|stmt| match stmt {
        Stmt::ClassDef(cls) => Some(cls),
        _ => None,
    })
}

/// Locate a class's File + name TextRange from its SymbolNode positions.
///
/// We don't store ty `Definition<'db>` references across plugin calls
/// (the `'db` lifetime is tied to the active borrow), so this re-walks
/// the matching file's top-level classes for one whose name lands on
/// the node's start line. Match-by-line (not line+column) because
/// Function / Class / TypeAlias node columns are snapped to the line's
/// indent — not the bound name's column — to align with libcst, and
/// two top-level classes can't share a source line.
pub(crate) fn locate_class_def(
    db: &ProjectDatabase,
    path_to_file: &FxHashMap<String, File>,
    path: &str,
    class_node: &GraphNode,
) -> Option<(File, TextRange)> {
    let &file = path_to_file.get(path)?;
    let parsed = parsed_module(db, file).load(db);
    let source = source_text(db, file);
    let line_index = line_index(db, file);
    for cls in iter_top_level_classes(&parsed) {
        let name_range = cls.name.range();
        let (sl, _, _, _) = position(&line_index, &source, name_range);
        if sl == class_node.start_line {
            return Some((file, name_range));
        }
    }
    None
}

/// Find a top-level `StmtClassDef` whose name range equals `selection_range`.
/// Return the top-level ``StmtClassDef`` (if any) that uses
/// ``ref_range`` as one of its direct base-list arguments.
///
/// Used by the find_references-based subclass walk: each
/// :func:`ty_ide::find_references` hit gives us a ``(File, range)``
/// for a use of the seed class; if that range falls inside a top-level
/// class's ``arguments.args`` list, the surrounding class is a direct
/// subclass.
///
/// Note: this is a syntactic match (range containment), so a use of
/// ``TestCase`` inside a parameterized generic base
/// (``class X(SomeGeneric[TestCase]):``) will falsely flag ``X`` as a
/// subclass of ``TestCase``. ty's :func:`type_hierarchy_subtypes`
/// avoids that via real type inference; we trade that accuracy for
/// the prefilter+rayon scan ty_ide's find_references provides.
pub(crate) fn class_base_arg_owner(
    parsed: &ParsedModuleRef,
    ref_range: TextRange,
) -> Option<&StmtClassDef> {
    for stmt in &parsed.syntax().body {
        let Stmt::ClassDef(class_def) = stmt else {
            continue;
        };
        let Some(arguments) = &class_def.arguments else {
            continue;
        };
        for arg in &arguments.args {
            if arg.range().contains_range(ref_range) {
                return Some(class_def);
            }
        }
    }
    None
}

/// If ``ref_range`` covers the "original name" of an aliased
/// ``from M import Name as Alias`` (or ``import M as Alias``), return
/// the alias's local-binding name range so the BFS can follow the
/// re-binding to its uses.
///
/// ty_ide's :func:`find_references` is designed for IDE rename
/// semantics: it does NOT follow aliased imports across the
/// re-binding (renaming ``TestCase`` should not rename uses of
/// ``TC`` from ``from unittest import TestCase as TC``). For our
/// subclass walk we DO want the alias's uses — they're the
/// subclasses we'd otherwise miss — so we seed the BFS with the
/// alias's local name range and call find_references on that.
pub(crate) fn aliased_import_local_name_range(
    parsed: &ParsedModuleRef,
    ref_range: TextRange,
) -> Option<TextRange> {
    for stmt in &parsed.syntax().body {
        let aliases: &[ruff_python_ast::Alias] = match stmt {
            Stmt::ImportFrom(import_from) => &import_from.names,
            Stmt::Import(import) => &import.names,
            _ => continue,
        };
        for alias in aliases {
            let Some(asname) = &alias.asname else {
                continue;
            };
            if alias.name.range() == ref_range {
                return Some(asname.range());
            }
        }
    }
    None
}

/// Walk transitive subclasses of the seed class via
/// :func:`ty_ide::find_references`, which carries an
/// identifier-aware text prefilter and per-file rayon parallelism for
/// free. Each find_references hit is filtered to "syntactically in a
/// class base list"; matched classes seed the next round when
/// ``transitive`` is set.
///
/// Returns ``Vec<usize>`` of indices into
/// ``BuildOutputs::builder.nodes`` (only project classes — typeshed
/// / external matches are dropped because they don't have a
/// :attr:`class_by_selection` entry).
#[allow(dead_code)]
pub(crate) fn find_subclass_indices_via_refs(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    seed_file: File,
    seed_name_range: TextRange,
    transitive: bool,
) -> Vec<usize> {
    find_subclass_indices_via_refs_with_queue(
        db,
        &outputs.class_by_selection,
        vec![(seed_file, seed_name_range)],
        &[],
        transitive,
    )
}

/// Lower-level entry point for [`find_subclass_indices_via_refs`] that
/// takes a pre-built initial BFS queue + a "pre-found" set of direct
/// subclasses and just the ``class_by_selection`` index. Used by
/// [`find_subclasses`](crate::project::ProjectContext::find_subclasses)
/// after the fast-path parallel scan has produced the first-hop
/// frontier from an external seed.
///
/// ``prefound_direct_subclasses`` is the list of
/// ``(File, class_name_range)`` pairs the fast path already classified
/// as direct subclasses; they're recorded as hits up-front *and*
/// seeded into the BFS so transitive subclasses + alias chains are
/// still walked through find_references. ``initial_queue`` is the
/// remaining seeds to run find_references on (typically the original
/// external seed range, so re-exports through project modules / star
/// imports are still caught).
pub(crate) fn find_subclass_indices_via_refs_with_queue(
    db: &dyn ProjectDb,
    class_by_selection: &FxHashMap<(File, (u32, u32)), usize>,
    initial_queue: Vec<(File, TextRange)>,
    prefound_direct_subclasses: &[(File, TextRange)],
    transitive: bool,
) -> Vec<usize> {
    let mut out_idx: FxHashSet<usize> = FxHashSet::default();
    let mut visited_seeds: FxHashSet<(File, (u32, u32))> = FxHashSet::default();
    let mut queue: Vec<(File, TextRange)> = initial_queue;

    // Record the fast-path direct subclasses as hits up-front. When
    // transitive=true also seed the BFS with them so subclasses of
    // these local classes get walked through find_references.
    for &(f, r) in prefound_direct_subclasses {
        let key = (f, range_key(r));
        if let Some(&idx) = class_by_selection.get(&key) {
            if out_idx.insert(idx) && transitive {
                queue.push((f, r));
            }
        }
    }

    while let Some((cur_file, cur_range)) = queue.pop() {
        if !visited_seeds.insert((cur_file, range_key(cur_range))) {
            continue;
        }
        // Cursor offset inside the identifier (start can sit on the
        // token boundary and ty_ide's tokens.at_offset has edge-cases
        // there). Anywhere inside the identifier resolves the same
        // goto target.
        let offset =
            cur_range.start() + ruff_text_size::TextSize::from(u32::from(cur_range.len()) / 2);
        // Pass include_declaration=true: skip-declaration mode filters
        // out re-binding declarations like `from M import Name as Alias`
        // alongside the seed itself. We need that import-line entry to
        // detect the alias and recurse, so we take the full list and
        // drop the seed in class_base_arg_owner / via the visited_seeds
        // dedup.
        let Some(refs) = ty_ide::find_references(db, cur_file, offset, true) else {
            continue;
        };
        for r in refs {
            let r_file = r.file();
            let r_range = r.range();
            let parsed = parsed_module(db, r_file).load(db);
            if let Some(class_def) = class_base_arg_owner(&parsed, r_range) {
                let class_name_range = class_def.name.range();
                let key = (r_file, range_key(class_name_range));
                let Some(&idx) = class_by_selection.get(&key) else {
                    continue;
                };
                if out_idx.insert(idx) && transitive {
                    queue.push((r_file, class_name_range));
                }
            } else if let Some(alias_range) = aliased_import_local_name_range(&parsed, r_range) {
                queue.push((r_file, alias_range));
            }
        }
    }

    out_idx.into_iter().collect()
}

/// Fast first-hop scan for the external-seed branch of
/// [`crate::project::ProjectContext::find_subclasses`]. Returns the
/// ``(File, class_name_range)`` of every top-level project class whose
/// base-arg list resolves (through the file's local imports) to
/// ``<seed_module>.<seed_simple_name>``.
///
/// This intentionally bypasses [`ty_ide::find_references`] (which is
/// the dominant cost when the seed is in an external package — e.g.
/// ``typer.Typer`` or ``fastapi.FastAPI``) and instead does a parallel
/// per-file text-prefilter + AST walk over project files. The result
/// seeds the BFS so transitive subclasses + alias chains are still
/// walked through the existing find_references path.
///
/// Recognized base shapes (mirroring
/// [`decorators_match_imports`]'s import-aware matcher):
///
/// * ``class X(SimpleName): ...`` where the file has
///   ``from <seed_module> import <seed_simple_name>`` (with or without
///   an ``as`` alias, so ``SimpleName`` may be the alias).
/// * ``class X(<alias>.<simple_name>): ...`` where ``<alias>`` is
///   bound to ``seed_module`` via ``import <seed_module>`` or
///   ``import <seed_module> as <alias>``.
///
/// Subscripted generics (``class X(SomeGeneric[<base>]): ...``) and
/// nested attribute chains are NOT matched — these are the rare
/// shapes the upstream ``find_references`` walk would catch via
/// semantic resolution. Callers run the historic find_references
/// path as a fallback when the fast path returns empty.
///
/// Trade-off vs the find_references walk: this is a *syntactic*
/// match against the file's import map. It can miss a subclass
/// whose base is bound via a re-export through an intermediate
/// project module (``from my_lib import Typer`` where ``my_lib``
/// re-exports ``typer.Typer``). Callers wishing to keep that path
/// must compose with find_references.
///
/// Caller must wrap in [`pyo3::Python::allow_threads`] — the inner
/// per-file scan uses rayon and is GIL-free.
/// Per-file fast-path result. ``direct_classes`` is the list of
/// class-name ranges identified as direct subclasses of the external
/// seed. ``has_module_import`` is ``true`` if this file imports the
/// seed module in any form (``from <module> import ...``,
/// ``import <module>``, ``from <module> import *``) — used by the
/// caller as a presence signal: if no file in the project has a
/// module import, the find_references fallback can be skipped (a
/// re-export chain through a project module would still surface here).
pub(crate) struct ExternalSeedFastPathResult {
    pub(crate) direct_classes: Vec<(File, TextRange)>,
    pub(crate) any_project_file_imports_seed_module: bool,
}

pub(crate) fn find_external_seed_direct_subclasses_par(
    db: &dyn ProjectDb,
    project_files: &[File],
    seed_module: &str,
    seed_simple_name: &str,
) -> ExternalSeedFastPathResult {
    use crate::query::{_contains_identifier, par_scan_files};
    let db_handle: Box<dyn ProjectDb> = ProjectDb::dyn_clone(db);
    let needle: &str = seed_simple_name;
    let module: &str = seed_module;
    // Per-file output: (class subclass ranges, file-imports-seed-module).
    let per_file: Vec<(Vec<(File, TextRange)>, bool)> =
        par_scan_files(db_handle, project_files, &None, |db, file| {
            // Cheap text prefilter on the seed module's leftmost
            // segment — every form of seed-module import (``import
            // <module>``, ``from <module> import …``, ``from <module>
            // import *``) mentions that segment in the source.
            let source = source_text(db, file);
            let module_root = module.split('.').next().unwrap_or(module);
            if !_contains_identifier(&source, module_root) {
                return Vec::new();
            }
            let parsed = parsed_module(db, file).load(db);
            let has_module_import = file_imports_module(&parsed, module);
            let mentions_name = _contains_identifier(&source, needle);
            // Only build the imports map + walk class bases when the
            // file textually mentions the simple name — the typical
            // shortcut.
            let direct: Vec<(File, TextRange)> = if mentions_name {
                let allowed: FxHashSet<&str> = std::iter::once(needle).collect();
                // ``file_package`` is `None`: relative imports of
                // external framework names (``from .foo import
                // Typer``) are vanishingly rare; the
                // ``find_references`` fallback covers them when
                // ``any_project_file_imports_seed_module`` flips on.
                let imports = collect_module_imports_local(&parsed, module, &allowed, None);
                if imports.is_empty() {
                    Vec::new()
                } else {
                    let mut out: Vec<(File, TextRange)> = Vec::new();
                    for cls in iter_top_level_classes(&parsed) {
                        let Some(arguments) = &cls.arguments else {
                            continue;
                        };
                        for arg in &arguments.args {
                            if base_arg_resolves_to_seed(arg, &imports, needle) {
                                out.push((file, cls.name.range()));
                                break;
                            }
                        }
                    }
                    out
                }
            } else {
                Vec::new()
            };
            vec![(direct, has_module_import)]
        });
    let mut direct_classes: Vec<(File, TextRange)> = Vec::new();
    let mut any_project_file_imports_seed_module = false;
    for (per, imports_seed) in per_file {
        direct_classes.extend(per);
        any_project_file_imports_seed_module |= imports_seed;
    }
    ExternalSeedFastPathResult {
        direct_classes,
        any_project_file_imports_seed_module,
    }
}

/// Returns ``true`` if ``parsed`` imports ``module`` in any of:
/// ``from <module> import …``, ``from <module> import *``,
/// ``import <module>``, or ``import <module> as …``. Used as the
/// presence-signal for the fast-path "no project file imports the
/// seed module" shortcut.
fn file_imports_module(parsed: &ParsedModuleRef, module: &str) -> bool {
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                if let Some(name) = &im.module {
                    if name.as_str() == module {
                        return true;
                    }
                }
            }
            Stmt::Import(im) => {
                for alias in &im.names {
                    if alias.name.as_str() == module {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// Return ``true`` if ``arg`` is a base-list reference that, after
/// resolving through the file's local imports, names the seed class.
/// Recognized shapes (matching what
/// [`find_external_seed_direct_subclasses_par`] documents):
///
/// * ``Name(local)`` where ``imports[local] == seed_simple_name``
///   (the local binding is the imported class).
/// * ``Attribute(Name(local).attr)`` where ``imports[local] ==
///   MODULE_ALIAS_MARKER`` and ``attr == seed_simple_name`` (the
///   local binding is the module alias and we're accessing the class
///   through it).
fn base_arg_resolves_to_seed(
    arg: &Expr,
    imports: &FxHashMap<String, String>,
    seed_simple_name: &str,
) -> bool {
    match arg {
        Expr::Name(n) => imports
            .get(n.id.as_str())
            .is_some_and(|target| target == seed_simple_name),
        Expr::Attribute(attr) => {
            if attr.attr.as_str() != seed_simple_name {
                return false;
            }
            let Expr::Name(prefix) = attr.value.as_ref() else {
                return false;
            };
            imports
                .get(prefix.id.as_str())
                .is_some_and(|t| t == MODULE_ALIAS_MARKER)
        }
        _ => false,
    }
}

/// Find a top-level ``StmtClassDef`` by its bound name.
pub(crate) fn class_def_named<'a>(
    parsed: &'a ParsedModuleRef,
    name: &str,
) -> Option<&'a StmtClassDef> {
    iter_top_level_classes(parsed).find(|cls| cls.name.as_str() == name)
}

/// Resolve a dotted class fqn to ``(File, name_range)`` so the caller
/// can fetch the corresponding ``Type`` via ``class_def_at`` +
/// ``inferred_type``. Handles both project classes (looked up via
/// ``decl_by_fqname``) and external classes (ty's ``resolve_module``
/// + AST scan by name in the resolved module).
pub(crate) fn locate_class_seed(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    fqn: &str,
) -> Option<(File, TextRange)> {
    // Project class: cheap path through the existing indices.
    if let Some(idxs) = outputs.decl_by_fqname.get(fqn) {
        for &idx in idxs {
            let node = &outputs.builder.nodes[idx];
            if node.kind != "class" {
                continue;
            }
            let path = node.path.clone();
            if let Some(seed) = locate_class_def(db, &outputs.path_to_file, &path, node) {
                return Some(seed);
            }
        }
    }
    // External class: resolve the module via ty, then follow any
    // re-export chains until we find the actual class def.
    let (module_str, class_name) = fqn.rsplit_once('.')?;
    let module_name = ModuleName::new(module_str)?;
    let anchor = *outputs.project_files.first()?;
    let module = resolve_module(db, anchor, &module_name)?;
    let module_file = module.file(db)?;
    let mut visited: FxHashSet<File> = FxHashSet::default();
    follow_class_through_module(db, module_file, class_name, &mut visited)
}

/// Walk through a module looking for ``class_name``. Handles three
/// re-export shapes: a direct ``class class_name: ...`` definition,
/// an explicit ``from .other import class_name [as class_name]``
/// re-export, and ``from .other import *`` (which exposes everything
/// the source module defines at the top level). Bounded by a
/// visited-file set so cycles can't loop forever.
pub(crate) fn follow_class_through_module(
    db: &ProjectDatabase,
    start_file: File,
    class_name: &str,
    visited: &mut FxHashSet<File>,
) -> Option<(File, TextRange)> {
    if !visited.insert(start_file) {
        return None;
    }
    let parsed = parsed_module(db, start_file).load(db);
    if let Some(cls) = class_def_named(&parsed, class_name) {
        return Some((start_file, cls.name.range()));
    }
    for stmt in &parsed.syntax().body {
        let Stmt::ImportFrom(im) = stmt else {
            continue;
        };
        let Ok(module_name) = ModuleName::from_import_statement(db, start_file, im) else {
            continue;
        };
        let Some(resolved) = resolve_module(db, start_file, &module_name) else {
            continue;
        };
        let Some(source_file) = resolved.file(db) else {
            continue;
        };
        for alias in &im.names {
            let imported_name = alias.name.as_str();
            if imported_name == "*" {
                if let Some(seed) =
                    follow_class_through_module(db, source_file, class_name, visited)
                {
                    return Some(seed);
                }
                continue;
            }
            let local_name = alias
                .asname
                .as_ref()
                .map(|n| n.as_str())
                .unwrap_or(imported_name);
            if local_name != class_name {
                continue;
            }
            if let Some(seed) = follow_class_through_module(db, source_file, imported_name, visited)
            {
                return Some(seed);
            }
        }
    }
    None
}

/// Locate the top-level ``if __name__ == "__main__":`` block in a
/// parsed module and return its source range. Both orderings
/// (``__name__ == "__main__"`` and ``"__main__" == __name__``) are
/// recognized; the ``elif`` / ``else`` branches are not matched.
pub(crate) fn find_main_block_range(parsed: &ParsedModuleRef) -> Option<TextRange> {
    for stmt in &parsed.syntax().body {
        let Stmt::If(if_stmt) = stmt else {
            continue;
        };
        if is_name_eq_main(&if_stmt.test) {
            return Some(if_stmt.range);
        }
    }
    None
}

/// Match ``__name__ == "__main__"`` or its reverse.
pub(crate) fn is_name_eq_main(expr: &Expr) -> bool {
    let Expr::Compare(cmp) = expr else {
        return false;
    };
    if cmp.ops.len() != 1 {
        return false;
    }
    if !matches!(cmp.ops[0], ruff_python_ast::CmpOp::Eq) {
        return false;
    }
    let comparators = &cmp.comparators;
    if comparators.len() != 1 {
        return false;
    }
    let left = &*cmp.left;
    let right = &comparators[0];
    (is_name(left, "__name__") && is_string_literal(right, "__main__"))
        || (is_string_literal(left, "__main__") && is_name(right, "__name__"))
}

pub(crate) fn is_name(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Name(n) if n.id.as_str() == value)
}

pub(crate) fn is_string_literal(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::StringLiteral(s) if s.value.to_str() == value)
}

/// Resolve a class-base expression to an upstream fqname using the
/// file's full imports map (as produced by
/// :func:`collect_all_imports_local`).
///
/// Supported expression shapes:
/// * ``Name(X)`` — looked up in ``file_imports``; returns the upstream
///   fqname when ``X`` is an imported name.
/// * ``Attribute(Name(M).N)`` — when ``M`` is bound to a module via
///   ``import <module> [as M]``, returns ``<module>.N``.
/// * Deeper attribute chain ``a.b.c.N`` rooted at an imported name
///   ``a`` — appends segments to reach ``<a-upstream>.b.c.N``.
///
/// Returns ``None`` when the expression doesn't have a single
/// recognized identifier shape, the root isn't imported, or the chain
/// can't be resolved. Generics (``Base[T]``) are not supported (the
/// caller is matching against `class C(Base[T]): ...` only to the
/// extent ``ruff_python_ast`` exposes ``Base[T]`` as an
/// ``ExprSubscript`` — those drop here, which is the correct
/// behavior).
pub(crate) fn resolve_base_fqn(
    expr: &Expr,
    file_imports: &FxHashMap<String, String>,
) -> Option<String> {
    match expr {
        Expr::Name(name) => file_imports.get(name.id.as_str()).cloned(),
        Expr::Attribute(_) => {
            let (root, segs) = collapse_attribute_chain(expr)?;
            let root_target = file_imports.get(root.id.as_str())?;
            let mut out = String::with_capacity(
                root_target.len() + segs.iter().map(|s| s.len() + 1).sum::<usize>(),
            );
            out.push_str(root_target);
            for seg in &segs {
                out.push('.');
                out.push_str(seg);
            }
            Some(out)
        }
        _ => None,
    }
}

/// Walk every `Stmt::Import` / `Stmt::ImportFrom` in `parsed` and
/// produce `local_name -> upstream_fqname` for every alias.
///
/// Unlike :fn:`collect_module_imports_local` (which filters to a single
/// `module` + `allowed` name set), this captures every top-level
/// import so callers can resolve arbitrary names referenced in
/// arg / kwarg positions.
///
/// Mappings produced:
/// * ``from foo.bar import Baz`` → ``"Baz" -> "foo.bar.Baz"``
/// * ``from foo.bar import Baz as B`` → ``"B" -> "foo.bar.Baz"``
/// * ``import foo`` → ``"foo" -> "foo"``
/// * ``import foo.bar as fb`` → ``"fb" -> "foo.bar"``
/// * ``import foo.bar`` → ``"foo" -> "foo"`` (Python binds the
///   leftmost segment; the runtime module ``foo`` is what the name
///   resolves to).
pub(crate) fn collect_all_imports_local(parsed: &ParsedModuleRef) -> FxHashMap<String, String> {
    let mut out: FxHashMap<String, String> = FxHashMap::default();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                let Some(mod_name) = &im.module else { continue };
                let module = mod_name.as_str();
                for alias in &im.names {
                    let target = alias.name.as_str();
                    let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                    let upstream = if module.is_empty() {
                        target.to_string()
                    } else {
                        format!("{module}.{target}")
                    };
                    out.insert(local.to_string(), upstream);
                }
            }
            Stmt::Import(im) => {
                for alias in &im.names {
                    let name = alias.name.as_str();
                    if let Some(asname) = alias.asname.as_ref() {
                        // ``import foo.bar as fb`` binds ``fb`` to the
                        // ``foo.bar`` module object.
                        out.insert(asname.as_str().to_string(), name.to_string());
                    } else {
                        // ``import foo`` binds ``foo``.
                        // ``import foo.bar`` binds ``foo`` (Python
                        // binds the leftmost segment).
                        let leftmost = name.split('.').next().unwrap_or(name);
                        out.insert(leftmost.to_string(), leftmost.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Convenience: build the file-local imports map for names imported
/// from any of ``modules`` (OR semantics — merged into a single map).
/// Later modules win on local-name collisions, which doesn't matter in
/// practice since callers route the matched ``target`` (the upstream
/// constructor name) against an ``allowed`` set; identical targets
/// across modules collapse to the same value.
///
/// ``file_package`` is the importing file's enclosing package (e.g.
/// ``Some("pkg")`` for ``pkg/handlers.py``, ``Some("pkg")`` for
/// ``pkg/__init__.py``, ``None`` for a top-level ``mod.py``). It is
/// used to resolve ``from .foo import N`` / ``from ..bar import N``
/// relative imports against ``modules``.
pub(crate) fn collect_modules_imports_local(
    parsed: &ParsedModuleRef,
    modules: &[String],
    allowed: &FxHashSet<&str>,
    file_package: Option<&str>,
) -> FxHashMap<String, String> {
    let mut out: FxHashMap<String, String> = FxHashMap::default();
    for module in modules {
        let one = collect_module_imports_local(parsed, module, allowed, file_package);
        out.extend(one);
    }
    out
}

/// Compute the absolute module path for a relative ``from`` import.
///
/// ``level`` is the count of leading dots on the ``from`` statement
/// (1 for ``from .x``, 2 for ``from ..x``, ...); ``tail`` is the
/// dotted suffix after the dots (``"x"`` for ``from .x``,
/// ``"x.y"`` for ``from .x.y``, empty for ``from . import name``).
/// ``file_package`` is the importing file's enclosing package.
///
/// Mirrors CPython's relative-import resolution: ``level=1`` anchors
/// at ``file_package``, each additional level strips one trailing
/// segment. Returns ``None`` when the resolution would walk past the
/// top of ``file_package`` (or when ``file_package`` is ``None`` but
/// ``level > 0``).
pub(crate) fn resolve_relative_import(
    level: u32,
    tail: &str,
    file_package: Option<&str>,
) -> Option<String> {
    if level == 0 {
        return None;
    }
    let pkg = file_package?;
    let segments: Vec<&str> = pkg.split('.').filter(|s| !s.is_empty()).collect();
    let levels_up = (level - 1) as usize;
    if levels_up > segments.len() {
        return None;
    }
    let base = &segments[..segments.len() - levels_up];
    let mut parts: Vec<&str> = base.to_vec();
    if !tail.is_empty() {
        parts.push(tail);
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("."))
}

/// Build the file-local imports map ``{local_name: target}`` for
/// names imported from ``module``. ``target`` is the upstream
/// constructor / decl name (e.g. ``"Flask"`` when bound via
/// ``from flask import Flask``) or [`MODULE_ALIAS_MARKER`] when bound
/// via ``import flask`` / ``import flask as f``.
///
/// Only entries whose target is in ``allowed`` survive — keeps the
/// map small and lets call-site matchers do a cheap second check.
pub(crate) fn collect_module_imports_local(
    parsed: &ParsedModuleRef,
    module: &str,
    allowed: &FxHashSet<&str>,
    file_package: Option<&str>,
) -> FxHashMap<String, String> {
    // Submodule binding: ``from <parent> import <last_seg>`` makes
    // ``last_seg`` a local alias for the queried module (e.g.
    // ``from unittest import mock`` for module ``unittest.mock``).
    let parent_last = module.rsplit_once('.');
    let mut out = FxHashMap::default();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                // Compute the import's effective absolute module path,
                // resolving relative dots against the importing file's
                // package. ``mod_name`` is just the dotted suffix; ty
                // exposes the dot count via ``im.level``.
                let tail = im.module.as_ref().map(|n| n.as_str()).unwrap_or("");
                let absolute: Option<String> = if im.level == 0 {
                    if tail.is_empty() {
                        None
                    } else {
                        Some(tail.to_string())
                    }
                } else {
                    resolve_relative_import(im.level, tail, file_package)
                };
                let Some(absolute) = absolute else { continue };
                if absolute == module {
                    for alias in &im.names {
                        let target = alias.name.as_str();
                        if !allowed.contains(target) {
                            continue;
                        }
                        let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                        out.insert(local.to_string(), target.to_string());
                    }
                } else if let Some((parent, last)) = parent_last {
                    if absolute == parent {
                        for alias in &im.names {
                            if alias.name.as_str() != last {
                                continue;
                            }
                            let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(last);
                            out.insert(local.to_string(), MODULE_ALIAS_MARKER.to_string());
                        }
                    }
                }
            }
            Stmt::Import(im) => {
                for alias in &im.names {
                    if alias.name.as_str() != module {
                        continue;
                    }
                    let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(module);
                    out.insert(local.to_string(), MODULE_ALIAS_MARKER.to_string());
                }
            }
            _ => {}
        }
    }
    out
}

/// Unwrap a leading ``Expr::Subscript`` once so subscripted-generic
/// callees (``Worker[int]``, ``app.route[Self]``) classify the same as
/// their bare form. Returns the inner expression for any other shape.
pub(crate) fn unwrap_subscripted_callee(expr: &Expr) -> &Expr {
    match expr {
        Expr::Subscript(sub) => sub.value.as_ref(),
        other => other,
    }
}

/// Multi-module variant of [`matched_call_target`]: returns the matched
/// upstream name as soon as any of ``modules`` produces a hit. Used by
/// the query forms that accept a list of modules to match against.
pub(crate) fn matched_call_target_any(
    call: &ruff_python_ast::ExprCall,
    imports: &FxHashMap<String, String>,
    modules: &[String],
    allowed: &FxHashSet<&str>,
) -> Option<String> {
    for module in modules {
        if let Some(name) = matched_call_target(call, imports, module, allowed) {
            return Some(name);
        }
    }
    None
}

/// Match an expression against a callable-from-module pattern. Three
/// shapes resolve through ``imports`` to the configured ``module``:
///
/// * ``<imported_local>(...)`` — local was bound via ``from <module>
///   import <name>``;
/// * ``<module_alias>.<allowed_name>(...)`` — alias bound via
///   ``import <module> [as <alias>]``;
/// * ``<m>.<n>.…<allowed_name>(...)`` — literal dotted access of a
///   multi-segment ``<module>`` (e.g. ``import unittest.mock;
///   unittest.mock.patch(...)``).
///
/// Returns the matched upstream name (``"Flask"``) on hit, else ``None``.
pub(crate) fn matched_call_target(
    call: &ruff_python_ast::ExprCall,
    imports: &FxHashMap<String, String>,
    module: &str,
    allowed: &FxHashSet<&str>,
) -> Option<String> {
    // Unwrap a leading ``Expr::Subscript`` once so subscripted-generic
    // constructors (``Worker[int](...)``, ``Logger[Self]("foo")``)
    // classify the same as their bare ``Worker(...)`` form. Generics
    // applied to imported classes are syntactically a ``Subscript``
    // whose ``value`` is the bare ``Name`` / ``Attribute`` we already
    // know how to resolve.
    let callee = unwrap_subscripted_callee(call.func.as_ref());
    match callee {
        Expr::Name(name) => {
            let target = imports.get(name.id.as_str())?;
            allowed.contains(target.as_str()).then(|| target.clone())
        }
        Expr::Attribute(attr) => {
            let attr_name = attr.attr.as_str();
            if !allowed.contains(attr_name) {
                return None;
            }
            match attr.value.as_ref() {
                Expr::Name(prefix) => (imports.get(prefix.id.as_str()).map(String::as_str)
                    == Some(MODULE_ALIAS_MARKER))
                .then(|| attr_name.to_string()),
                _ => {
                    let (root, segs) = collapse_attribute_chain(attr.value.as_ref())?;
                    let mut dotted = String::with_capacity(module.len());
                    dotted.push_str(root.id.as_str());
                    for seg in &segs {
                        dotted.push('.');
                        dotted.push_str(seg);
                    }
                    (dotted == module).then(|| attr_name.to_string())
                }
            }
        }
        _ => None,
    }
}

/// Top-level ``X = <expr>`` / ``X: T = <expr>`` where the target is a
/// bare ``Name``. Returns ``(name_range, &expr)`` so the caller can
/// look up ``X``'s node and inspect the RHS.
pub(crate) fn top_level_assign_to_name(stmt: &Stmt) -> Option<(TextRange, &Expr)> {
    match stmt {
        Stmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return None;
            }
            let Expr::Name(name) = &assign.targets[0] else {
                return None;
            };
            Some((name.range, assign.value.as_ref()))
        }
        Stmt::AnnAssign(assign) => {
            let value = assign.value.as_deref()?;
            let Expr::Name(name) = assign.target.as_ref() else {
                return None;
            };
            Some((name.range, value))
        }
        _ => None,
    }
}

/// Recursive visitor: walk a function / class body collecting the set
/// of constructor names called anywhere inside it. Matches against any
/// of ``modules`` (OR semantics — for single-module callers pass a
/// one-element slice).
pub(crate) struct FactoryCallFinder<'a> {
    pub(crate) imports: &'a FxHashMap<String, String>,
    pub(crate) modules: &'a [String],
    pub(crate) allowed: &'a FxHashSet<&'a str>,
    pub(crate) kinds: FxHashSet<String>,
}

impl<'ast, 'a> Visitor<'ast> for FactoryCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Some(name) =
                matched_call_target_any(call, self.imports, self.modules, self.allowed)
            {
                self.kinds.insert(name);
            }
        }
        walk_expr(self, expr);
    }
}

/// Owner-resolution helper shared by the call-finder queries.
/// Top-level ``FunctionDef`` / ``ClassDef`` own calls inside their
/// subtree (decorators included via the walk); other top-level stmts
/// attribute their calls to the module node.
/// Look up the owning decl index for a top-level statement. Takes
/// the two hashmaps directly (rather than ``&BuildOutputs``) because
/// ``BuildOutputs`` carries ``Vec<Py<SymbolNode>>`` and is therefore
/// ``!Sync`` — these maps are ``Sync`` on their own, which lets the
/// callers borrow them across rayon thread boundaries.
pub(crate) fn owner_idx_for_stmt_with(
    decl_by_name_range: &FxHashMap<(File, (u32, u32)), usize>,
    module_nodes_by_file: &FxHashMap<File, usize>,
    file: File,
    stmt: &Stmt,
) -> Option<usize> {
    let module_idx = module_nodes_by_file.get(&file).copied();
    let name_range = match stmt {
        Stmt::FunctionDef(f) => f.name.range(),
        Stmt::ClassDef(c) => c.name.range(),
        _ => return module_idx,
    };
    decl_by_name_range
        .get(&(file, range_key(name_range)))
        .copied()
        .or(module_idx)
}

/// Extract the string-literal value of a call's ``args[arg_index]``
/// positional argument. ``None`` when out of range or not a single
/// string literal — concatenated / f-string / b-string forms don't
/// project to a static fqname and are deliberately rejected.
pub(crate) fn nth_positional_string(
    call: &ruff_python_ast::ExprCall,
    arg_index: usize,
) -> Option<String> {
    match call.arguments.args.get(arg_index)? {
        Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
        _ => None,
    }
}

/// A string literal extracted from a call keyword argument, or
/// ``Unknown`` for anything else. The only consumer today is the
/// pytest plugin, which reads the ``name=`` alias on ``@pytest.fixture``;
/// every non-string expression collapses to ``Unknown``.
#[derive(Clone, Debug)]
pub(crate) enum ArgValue {
    Str(String),
    Unknown,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CallArgs {
    pub(crate) kwargs: FxHashMap<String, ArgValue>,
}

/// Tuple-like result row returned by
/// :meth:`ProjectContext.node_attrs` and the per-query
/// ``.attrs()`` terminals.
///
/// Supports both attribute access (``attr.fqname``) and tuple
/// semantics (``kind, path, fqname, flags = attr``; ``attr[2]``;
/// ``len(attr) == 4``). Not a `typing.NamedTuple` instance, but
/// drop-in compatible for unpacking and subscript access. Frozen —
/// fields are immutable once constructed.
#[pyclass(frozen, get_all)]
pub(crate) struct NodeAttrs {
    pub(crate) kind: String,
    pub(crate) path: String,
    pub(crate) fqname: String,
    pub(crate) flags: u32,
}

#[pymethods]
impl NodeAttrs {
    fn __len__(&self) -> usize {
        4
    }

    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n: isize = 4;
        let i = if idx < 0 { idx + n } else { idx };
        if !(0..n).contains(&i) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "NodeAttrs index {idx} out of range (len=4)"
            )));
        }
        Ok(match i {
            0 => self.kind.clone().into_py(py),
            1 => self.path.clone().into_py(py),
            2 => self.fqname.clone().into_py(py),
            3 => self.flags.into_py(py),
            _ => unreachable!(),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tup = pyo3::types::PyTuple::new_bound(
            py,
            &[
                slf.kind.clone().into_py(py),
                slf.path.clone().into_py(py),
                slf.fqname.clone().into_py(py),
                slf.flags.into_py(py),
            ],
        );
        Ok(tup.call_method0("__iter__")?.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeAttrs(kind={:?}, path={:?}, fqname={:?}, flags={})",
            self.kind, self.path, self.fqname, self.flags
        )
    }
}

/// Convert one AST argument expression to an ``ArgValue``. Only string
/// literals are captured; every other expression is ``Unknown``.
pub(crate) fn extract_arg_value(expr: &Expr) -> ArgValue {
    match expr {
        Expr::StringLiteral(s) => ArgValue::Str(s.value.to_str().to_string()),
        _ => ArgValue::Unknown,
    }
}

/// Extract keyword arguments from a call. Positional args are not
/// captured — no consumer reads them.
pub(crate) fn extract_call_kwargs(call: &ruff_python_ast::ExprCall) -> CallArgs {
    let mut kwargs: FxHashMap<String, ArgValue> = FxHashMap::default();
    for kw in &call.arguments.keywords {
        let Some(name) = kw.arg.as_ref() else {
            continue;
        };
        kwargs.insert(name.as_str().to_string(), extract_arg_value(&kw.value));
    }
    CallArgs { kwargs }
}

/// Match ``<owner>.<attr>(...)`` where ``<owner>`` is a bare ``Name``
/// equal to the given owner string and ``<attr>`` matches. No import
/// resolution — meant for pytest fixture conventions (``mocker``,
/// ``monkeypatch``) whose names come from function parameters.
pub(crate) fn call_callee_matches_var(
    call: &ruff_python_ast::ExprCall,
    owner: &str,
    attr: &str,
    required_positional: Option<usize>,
) -> bool {
    let Expr::Attribute(attribute) = unwrap_subscripted_callee(call.func.as_ref()) else {
        return false;
    };
    if attribute.attr.as_str() != attr {
        return false;
    }
    let Expr::Name(name) = attribute.value.as_ref() else {
        return false;
    };
    if name.id.as_str() != owner {
        return false;
    }
    if let Some(expected) = required_positional {
        if call.arguments.args.len() != expected {
            return false;
        }
    }
    true
}

/// Visit every Call expression in a subtree, push
/// ``(string_arg_at_index, CallArgs)`` whenever ``predicate(call)``
/// returns ``true``. Backs both call-finder queries.
pub(crate) struct StringArgCallFinder<F>
where
    F: FnMut(&ruff_python_ast::ExprCall) -> bool,
{
    pub(crate) predicate: F,
    pub(crate) arg_index: usize,
    /// When ``false``, every result row gets a default-constructed
    /// (empty) :struct:`CallArgs`. The caller pays for the rust-side
    /// kwarg walk only when ``extract_args = true``.
    pub(crate) extract_args: bool,
    pub(crate) results: Vec<(String, CallArgs)>,
}

impl<'ast, F> Visitor<'ast> for StringArgCallFinder<F>
where
    F: FnMut(&ruff_python_ast::ExprCall) -> bool,
{
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if (self.predicate)(call) {
                if let Some(value) = nth_positional_string(call, self.arg_index) {
                    let call_args = if self.extract_args {
                        extract_call_kwargs(call)
                    } else {
                        CallArgs::default()
                    };
                    self.results.push((value, call_args));
                }
            }
        }
        walk_expr(self, expr);
    }
}

/// Visit every Call expression in a subtree, capture string-literal
/// args at ``arg_index`` for calls whose callee is ``<expr>.<attr>(...)``
/// regardless of receiver shape. The captured arg can be a single
/// string literal **or** a list/tuple of string literals (the latter
/// produces multiple results for one call). Backs ``find_calls_on_attr``.
///
/// Each emitted row is ``(captured_string, CallArgs)``. Multiple rows
/// from one call (list/tuple shape) share the same ``CallArgs``.
pub(crate) struct AttrCallFinder<'a> {
    pub(crate) attr: &'a str,
    pub(crate) arg_index: usize,
    /// See :struct:`StringArgCallFinder` — same gate.
    pub(crate) extract_args: bool,
    pub(crate) results: Vec<(String, CallArgs)>,
}

impl<'ast, 'a> Visitor<'ast> for AttrCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Expr::Attribute(attribute) = unwrap_subscripted_callee(call.func.as_ref()) {
                if attribute.attr.as_str() == self.attr {
                    if let Some(arg) = call.arguments.args.get(self.arg_index) {
                        let hits = string_or_string_collection(arg);
                        if !hits.is_empty() {
                            let call_args = if self.extract_args {
                                extract_call_kwargs(call)
                            } else {
                                CallArgs::default()
                            };
                            for s in hits {
                                self.results.push((s, call_args.clone()));
                            }
                        }
                    }
                }
            }
        }
        walk_expr(self, expr);
    }
}

/// Pull string-literal values out of either a single ``"..."`` or a
/// homogeneous list/tuple of string literals. Non-string elements in a
/// list/tuple are silently dropped; non-matching expressions yield an
/// empty vec.
fn string_or_string_collection(arg: &Expr) -> Vec<String> {
    let lit = |e: &Expr| match e {
        Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
        _ => None,
    };
    match arg {
        Expr::StringLiteral(s) => vec![s.value.to_str().to_string()],
        Expr::List(list) => list.elts.iter().filter_map(lit).collect(),
        Expr::Tuple(tup) => tup.elts.iter().filter_map(lit).collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn rel_path<P: AsRef<str>>(path: P) -> RelativePathBuf {
    RelativePathBuf::cli(SystemPath::new(path.as_ref()))
}

/// Per-source-type default flags for ``.ipynb``: notebook decls are
/// always alive (cells run top-to-bottom, not imported) and the
/// codemod must skip them (it can't rewrite the cell JSON envelope).
/// The `NOTEBOOK` bit alone is enough — the Python-side
/// `KEEPALIVE_DEFAULT` mask includes `NOTEBOOK`, so reachability seeds
/// from notebook nodes without needing the `ENTRYPOINT` overlay.
pub(crate) const NODE_FLAGS_NOTEBOOK_DEFAULT: u32 = NodeFlags::NOTEBOOK;

/// Bits stamped on every import alias pinned by a noqa directive. The
/// `NOQA` bit alone is enough — the Python-side `KEEPALIVE_DEFAULT`
/// mask includes `NOQA`, so reachability seeds from noqa-pinned
/// aliases and the `kept_alive_by_flags_only(NOQA)` blast-radius query
/// isolates them.
pub(crate) const NODE_FLAGS_NOQA_PIN: u32 = NodeFlags::NOQA;

/// Internal aliases for the pyclass classattrs used by the call sites
/// scattered through this file. Read as bare constants rather than
/// `NodeFlags::ENTRYPOINT` (which would force every reader to chase the
/// `pyclass` macro to decide whether it's a runtime lookup or a const).
pub(crate) const NODE_FLAG_ENTRYPOINT: u32 = NodeFlags::ENTRYPOINT;
pub(crate) const EDGE_FLAG_DEAD_BRANCH: u32 = EdgeFlags::DEAD_BRANCH;
pub(crate) const EDGE_FLAG_DYNAMIC_IMPORT: u32 = EdgeFlags::DYNAMIC_IMPORT;

/// Collect the source ranges of every statically-dead statement in
/// `parsed` — bodies of `if False` / `else` of `if True` / `while False`
/// / `while True else` branches, statements after an unconditional
/// terminator (`return` / `raise` / `break` / `continue` /
/// `assert <falsy>`), and the suites of compound terminators (an
/// `if/elif/else` whose every branch terminates, a `try` whose body
/// and every handler terminate, a `with` whose body terminates).
///
/// The result is a flat `Vec<TextRange>`; callers use
/// `TextRange::contains_range` to test whether a use site sits
/// inside any of them.
/// Per-scope name → known-truthiness map. Populated by
/// `build_scope_table` from the scope's literal assignments,
/// annotated assignments, and walrus expressions. Function and class
/// bodies inherit and override their enclosing scope's table.
pub(crate) type NameTable = FxHashMap<String, bool>;

pub(crate) fn detect_dead_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut dead = Vec::new();
    let empty: NameTable = FxHashMap::default();
    let module_table = build_scope_table(&parsed.syntax().body, &empty);
    walk_suite_for_dead(&parsed.syntax().body, &module_table, &mut dead);
    dead
}

pub(crate) fn walk_suite_for_dead(stmts: &[Stmt], table: &NameTable, dead: &mut Vec<TextRange>) {
    let mut killed = false;
    for stmt in stmts {
        if killed {
            // Past an in-suite terminator. Mark this stmt's range as
            // dead and don't recurse — descendant uses are covered by
            // the outer range.
            dead.push(stmt.range());
            continue;
        }
        walk_compound_for_dead(stmt, table, dead);
        if stmt_is_terminator(stmt, table) {
            killed = true;
        }
    }
}

pub(crate) fn walk_compound_for_dead(stmt: &Stmt, table: &NameTable, dead: &mut Vec<TextRange>) {
    match stmt {
        Stmt::If(if_stmt) => {
            let mut taken = false;
            match evaluate_truthiness(&if_stmt.test, table) {
                Some(true) => {
                    walk_suite_for_dead(&if_stmt.body, table, dead);
                    taken = true;
                }
                Some(false) => {
                    for s in &if_stmt.body {
                        dead.push(s.range());
                    }
                }
                None => {
                    walk_suite_for_dead(&if_stmt.body, table, dead);
                }
            }
            for clause in &if_stmt.elif_else_clauses {
                if taken {
                    dead.push(clause.range);
                    continue;
                }
                match &clause.test {
                    Some(test) => match evaluate_truthiness(test, table) {
                        Some(true) => {
                            walk_suite_for_dead(&clause.body, table, dead);
                            taken = true;
                        }
                        Some(false) => {
                            for s in &clause.body {
                                dead.push(s.range());
                            }
                        }
                        None => walk_suite_for_dead(&clause.body, table, dead),
                    },
                    None => walk_suite_for_dead(&clause.body, table, dead),
                }
            }
        }
        Stmt::While(w) => match evaluate_truthiness(&w.test, table) {
            Some(false) => {
                for s in &w.body {
                    dead.push(s.range());
                }
                walk_suite_for_dead(&w.orelse, table, dead);
            }
            Some(true) => {
                walk_suite_for_dead(&w.body, table, dead);
                for s in &w.orelse {
                    dead.push(s.range());
                }
            }
            None => {
                walk_suite_for_dead(&w.body, table, dead);
                walk_suite_for_dead(&w.orelse, table, dead);
            }
        },
        Stmt::For(f) => {
            walk_suite_for_dead(&f.body, table, dead);
            walk_suite_for_dead(&f.orelse, table, dead);
        }
        Stmt::With(w) => walk_suite_for_dead(&w.body, table, dead),
        Stmt::Try(t) => {
            walk_suite_for_dead(&t.body, table, dead);
            for handler in &t.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
                walk_suite_for_dead(&h.body, table, dead);
            }
            walk_suite_for_dead(&t.orelse, table, dead);
            walk_suite_for_dead(&t.finalbody, table, dead);
        }
        Stmt::FunctionDef(f) => {
            let nested = build_scope_table(&f.body, table);
            walk_suite_for_dead(&f.body, &nested, dead);
        }
        Stmt::ClassDef(c) => {
            let nested = build_scope_table(&c.body, table);
            walk_suite_for_dead(&c.body, &nested, dead);
        }
        Stmt::Match(m) => {
            for case in &m.cases {
                walk_suite_for_dead(&case.body, table, dead);
            }
        }
        _ => {}
    }
}

/// Source ranges of the statements inside an `if TYPE_CHECKING:` block
/// (the guard spelled bare as `TYPE_CHECKING` or as an attribute like
/// `typing.TYPE_CHECKING`). ty narrows `TYPE_CHECKING` to `True`, so a
/// use whose flow-resolved binding lives in one of these ranges has had
/// its *runtime* binding (in the `else` clause, or after the block)
/// narrowed away. The ref walker uses these ranges to recover the
/// runtime binding. The `else`/`elif` clauses are deliberately not
/// recorded — those are what actually run at runtime.
pub(crate) fn detect_type_checking_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut ranges = Vec::new();
    collect_type_checking_ranges(&parsed.syntax().body, &mut ranges);
    ranges
}

fn collect_type_checking_ranges(stmts: &[Stmt], ranges: &mut Vec<TextRange>) {
    for stmt in stmts {
        match stmt {
            Stmt::If(if_stmt) => {
                if expr_is_type_checking(&if_stmt.test) {
                    for s in &if_stmt.body {
                        ranges.push(s.range());
                    }
                }
                collect_type_checking_ranges(&if_stmt.body, ranges);
                for clause in &if_stmt.elif_else_clauses {
                    collect_type_checking_ranges(&clause.body, ranges);
                }
            }
            Stmt::While(w) => {
                collect_type_checking_ranges(&w.body, ranges);
                collect_type_checking_ranges(&w.orelse, ranges);
            }
            Stmt::For(f) => {
                collect_type_checking_ranges(&f.body, ranges);
                collect_type_checking_ranges(&f.orelse, ranges);
            }
            Stmt::With(w) => collect_type_checking_ranges(&w.body, ranges),
            Stmt::Try(t) => {
                collect_type_checking_ranges(&t.body, ranges);
                for handler in &t.handlers {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
                    collect_type_checking_ranges(&h.body, ranges);
                }
                collect_type_checking_ranges(&t.orelse, ranges);
                collect_type_checking_ranges(&t.finalbody, ranges);
            }
            Stmt::FunctionDef(f) => collect_type_checking_ranges(&f.body, ranges),
            Stmt::ClassDef(c) => collect_type_checking_ranges(&c.body, ranges),
            Stmt::Match(m) => {
                for case in &m.cases {
                    collect_type_checking_ranges(&case.body, ranges);
                }
            }
            _ => {}
        }
    }
}

/// True for the two spellings of the type-checking guard: a bare
/// `TYPE_CHECKING` name or any `<…>.TYPE_CHECKING` attribute access
/// (e.g. `typing.TYPE_CHECKING`).
fn expr_is_type_checking(expr: &Expr) -> bool {
    match expr {
        Expr::Name(n) => n.id.as_str() == "TYPE_CHECKING",
        Expr::Attribute(a) => a.attr.as_str() == "TYPE_CHECKING",
        _ => false,
    }
}

/// Does `stmt` unconditionally hand off control out of the enclosing
/// suite? Statements *after* a terminator in the same suite are dead.
///
/// Bare terminators are `return` / `raise` / `break` / `continue`
/// plus `assert <falsy>`. Compound forms (`if/elif/else`, `try`,
/// `with`, `match`) are terminators when every reachable path inside
/// them ends in one.
pub(crate) fn stmt_is_terminator(stmt: &Stmt, table: &NameTable) -> bool {
    match stmt {
        Stmt::Return(_) | Stmt::Raise(_) | Stmt::Break(_) | Stmt::Continue(_) => true,
        Stmt::Assert(a) => evaluate_truthiness(&a.test, table) == Some(false),
        Stmt::If(if_stmt) => {
            // A constant-truthy `if` is unconditional: it's a
            // terminator iff its body terminates, with or without
            // any else (`if True: return; …` kills the rest of
            // the enclosing suite).
            if evaluate_truthiness(&if_stmt.test, table) == Some(true) {
                return suite_terminates(&if_stmt.body, table);
            }
            // Otherwise we need an else for completeness — without
            // one, the `if` (or every `elif`) might not fire and the
            // suite continues normally.
            let has_else = if_stmt
                .elif_else_clauses
                .last()
                .is_some_and(|c| c.test.is_none());
            if !has_else {
                return false;
            }
            let test = evaluate_truthiness(&if_stmt.test, table);
            let body_term = suite_terminates(&if_stmt.body, table) || test == Some(false);
            let clauses_term = if_stmt
                .elif_else_clauses
                .iter()
                .all(|c| suite_terminates(&c.body, table));
            body_term && clauses_term
        }
        Stmt::With(w) => suite_terminates(&w.body, table),
        Stmt::Try(t) => {
            if suite_terminates(&t.finalbody, table) {
                return true;
            }
            suite_terminates(&t.body, table)
                && t.handlers.iter().all(|h| {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(eh) = h;
                    suite_terminates(&eh.body, table)
                })
        }
        _ => false,
    }
}

pub(crate) fn suite_terminates(stmts: &[Stmt], table: &NameTable) -> bool {
    stmts.iter().any(|s| stmt_is_terminator(s, table))
}

/// Constant-fold the truthiness of `expr`. Returns `None` when the
/// expression doesn't reduce to a known value — names not in `table`,
/// calls, attribute access, comparisons, etc.
///
/// `BoolOp` (`and` / `or`) short-circuits the way Python does:
/// `True or anything` is `True` even when "anything" is unknown.
/// `not` flips a known truth value through `UnaryOp`.
pub(crate) fn evaluate_truthiness(expr: &Expr, table: &NameTable) -> Option<bool> {
    match expr {
        Expr::BooleanLiteral(b) => Some(b.value),
        Expr::NoneLiteral(_) => Some(false),
        Expr::EllipsisLiteral(_) => Some(true),
        Expr::NumberLiteral(n) => match &n.value {
            ruff_python_ast::Number::Int(i) => i.as_i64().map(|x| x != 0),
            ruff_python_ast::Number::Float(f) => Some(*f != 0.0),
            ruff_python_ast::Number::Complex { real, imag } => Some(*real != 0.0 || *imag != 0.0),
        },
        Expr::StringLiteral(s) => Some(!s.value.is_empty()),
        Expr::BytesLiteral(b) => Some(!b.value.is_empty()),
        Expr::List(l) => Some(!l.elts.is_empty()),
        Expr::Tuple(t) => Some(!t.elts.is_empty()),
        Expr::Set(s) => Some(!s.elts.is_empty()),
        Expr::Dict(d) => Some(!d.items.is_empty()),
        Expr::Name(n) => table.get(n.id.as_str()).copied(),
        Expr::Named(named) => evaluate_truthiness(&named.value, table),
        Expr::BoolOp(b) => match b.op {
            ruff_python_ast::BoolOp::Or => {
                let mut any_unknown = false;
                for v in &b.values {
                    match evaluate_truthiness(v, table) {
                        Some(true) => return Some(true),
                        Some(false) => continue,
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(false)
                }
            }
            ruff_python_ast::BoolOp::And => {
                let mut any_unknown = false;
                for v in &b.values {
                    match evaluate_truthiness(v, table) {
                        Some(false) => return Some(false),
                        Some(true) => continue,
                        None => any_unknown = true,
                    }
                }
                if any_unknown {
                    None
                } else {
                    Some(true)
                }
            }
        },
        Expr::UnaryOp(u) => {
            if matches!(u.op, ruff_python_ast::UnaryOp::Not) {
                evaluate_truthiness(&u.operand, table).map(|v| !v)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build the scope's name→truthiness table.
///
/// Collects all unconditional assignments at the scope's top level
/// (`X = literal`, `X: T = literal`) plus every walrus binding the
/// scope's expressions contain (PEP 572 leaks walrus bindings to the
/// enclosing function/module). Iterates to fixed point so that
/// chained constants — `foo = False; bar = foo or False; baz = bar`
/// — fold through the chain.
///
/// Conservative on conflicts: a name with multiple disagreeing
/// bindings (`X = True` and `X = False` both at top level), or any
/// binding whose RHS we can't fold, drops out of the table. That
/// matches the libcst pipeline's "only fold when every binding
/// agrees" rule and is what keeps the `does_not_fold_*` tests honest.
///
/// Non-monotonic flips poison the name. A binding like `foo = not foo`
/// in a scope that inherits `foo = False` from the enclosing scope
/// would otherwise oscillate the table between true/false forever
/// (`not false → true → not true → false → ...`). The poison set is
/// the convergence guarantee: once a name produces a different
/// resolved value than the table already holds, we drop it and
/// refuse to re-fold it for the rest of this scope.
pub(crate) fn build_scope_table(stmts: &[Stmt], enclosing: &NameTable) -> NameTable {
    let mut table = enclosing.clone();
    let mut bindings: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
    collect_scope_bindings(stmts, &mut bindings);

    let mut poisoned: FxHashSet<String> = FxHashSet::default();
    let mut changed = true;
    while changed {
        changed = false;
        for (name, exprs) in &bindings {
            if poisoned.contains(name) {
                continue;
            }
            let values: Vec<Option<bool>> = exprs
                .iter()
                .map(|e| evaluate_truthiness(e, &table))
                .collect();
            let resolved = if let Some(first) = values.first().copied() {
                if values.iter().all(|v| *v == first) {
                    first
                } else {
                    None
                }
            } else {
                None
            };
            match (resolved, table.get(name).copied()) {
                (Some(v), None) => {
                    table.insert(name.clone(), v);
                    changed = true;
                }
                (Some(new), Some(prev)) if prev != new => {
                    table.remove(name);
                    poisoned.insert(name.clone());
                    changed = true;
                }
                (None, Some(_)) => {
                    table.remove(name);
                    changed = true;
                }
                _ => {}
            }
        }
    }
    table
}

/// Top-level `name → rhs-expression` bindings for the scope, plus
/// any walrus bindings the scope's *expressions* contain (PEP 572
/// leaks walruses to the enclosing function/module). Does NOT
/// recurse into nested function/class bodies — those are separate
/// scopes with their own tables.
pub(crate) fn collect_scope_bindings<'a>(
    stmts: &'a [Stmt],
    out: &mut FxHashMap<String, Vec<&'a Expr>>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign(a) => {
                if let [Expr::Name(target)] = a.targets.as_slice() {
                    out.entry(target.id.as_str().to_string())
                        .or_default()
                        .push(&a.value);
                }
                collect_walrus_in_expr(&a.value, out);
            }
            Stmt::AnnAssign(a) => {
                if let (Expr::Name(target), Some(value)) = (&*a.target, &a.value) {
                    out.entry(target.id.as_str().to_string())
                        .or_default()
                        .push(value);
                }
                if let Some(value) = &a.value {
                    collect_walrus_in_expr(value, out);
                }
                collect_walrus_in_expr(&a.annotation, out);
            }
            Stmt::Expr(e) => {
                collect_walrus_in_expr(&e.value, out);
            }
            Stmt::If(if_stmt) => {
                collect_walrus_in_expr(&if_stmt.test, out);
            }
            Stmt::While(w) => {
                collect_walrus_in_expr(&w.test, out);
            }
            Stmt::Return(r) => {
                if let Some(value) = &r.value {
                    collect_walrus_in_expr(value, out);
                }
            }
            // Nested function/class bodies are separate scopes —
            // don't drag their bindings into this table.
            Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
            _ => {}
        }
    }
}

/// Scan `expr` for walrus (`:=`) targets that bind names in the
/// enclosing scope.
pub(crate) fn collect_walrus_in_expr<'a>(
    expr: &'a Expr,
    out: &mut FxHashMap<String, Vec<&'a Expr>>,
) {
    match expr {
        Expr::Named(named) => {
            if let Expr::Name(target) = &*named.target {
                out.entry(target.id.as_str().to_string())
                    .or_default()
                    .push(&named.value);
            }
            collect_walrus_in_expr(&named.value, out);
        }
        Expr::BoolOp(b) => {
            for v in &b.values {
                collect_walrus_in_expr(v, out);
            }
        }
        Expr::UnaryOp(u) => collect_walrus_in_expr(&u.operand, out),
        Expr::BinOp(b) => {
            collect_walrus_in_expr(&b.left, out);
            collect_walrus_in_expr(&b.right, out);
        }
        Expr::Compare(c) => {
            collect_walrus_in_expr(&c.left, out);
            for v in &c.comparators {
                collect_walrus_in_expr(v, out);
            }
        }
        _ => {}
    }
}

/// Result of parsing the `noqa[: codes…]` tail of a comment.
///
/// `Bare` means the directive carried no code list and so suppresses
/// every rule (including F401); `F401Present` means F401 is in the
/// comma-separated rule list; `OtherOnly` means a rule list was given
/// but F401 was absent. The first two pin the import alive; the third
/// doesn't.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoqaKind {
    Bare,
    F401Present,
    OtherOnly,
}

impl NoqaKind {
    fn pins_f401(self) -> bool {
        matches!(self, NoqaKind::Bare | NoqaKind::F401Present)
    }
}

/// Parse a `noqa[: codes…]` tail from `content`. Returns `None` if
/// `content` doesn't start with `noqa` (case-insensitive) after
/// optional whitespace.
pub(crate) fn parse_noqa_tail(content: &str) -> Option<NoqaKind> {
    let trimmed = content.trim_start();
    // `str::get` short-circuits when byte 4 isn't a char boundary —
    // e.g. a comment that starts with the 3-byte `─` box-drawing char
    // (common in section banners). Direct slicing would panic.
    let prefix = trimmed.get(..4)?;
    if !prefix.eq_ignore_ascii_case("noqa") {
        return None;
    }
    let after_noqa = &trimmed[4..];
    let after_colon = after_noqa.trim_start();
    let Some(rest) = after_colon.strip_prefix(':') else {
        // Bare `noqa` with no code list.
        return Some(NoqaKind::Bare);
    };
    let rules = rest.trim();
    if rules.is_empty() {
        return Some(NoqaKind::Bare);
    }
    let has_f401 = rules
        .split(',')
        .any(|code| code.trim().eq_ignore_ascii_case("F401"));
    if has_f401 {
        Some(NoqaKind::F401Present)
    } else {
        Some(NoqaKind::OtherOnly)
    }
}

/// Returns `true` when `comment_body` (the text *after* the leading
/// `#`) is a per-line `# noqa` directive that silences F401.
pub(crate) fn is_per_line_pin(comment_body: &str) -> bool {
    parse_noqa_tail(comment_body)
        .map(NoqaKind::pins_f401)
        .unwrap_or(false)
}

/// Returns `true` when `comment_body` is a *file-level*
/// `# ruff: noqa` / `# flake8: noqa` directive that silences F401.
///
/// The `ruff:` / `flake8:` prefix is matched case-sensitively (per
/// ruff's documented behavior); the `noqa` keyword itself is
/// case-insensitive.
pub(crate) fn is_file_pin(comment_body: &str) -> bool {
    let trimmed = comment_body.trim_start();
    let after_prefix = trimmed
        .strip_prefix("ruff:")
        .or_else(|| trimmed.strip_prefix("flake8:"));
    let Some(after_prefix) = after_prefix else {
        return false;
    };
    parse_noqa_tail(after_prefix)
        .map(NoqaKind::pins_f401)
        .unwrap_or(false)
}

/// Scan every Comment token in `parsed` and partition the file's
/// noqa directives into `(file_pinned, per_line_pins)`.
///
/// `file_pinned` is true when *any* comment in the file is a
/// file-level `# ruff: noqa` / `# flake8: noqa` that silences F401 —
/// ruff scans the whole source, not just the header.
/// `per_line_pins` collects the (1-indexed) line numbers carrying a
/// `# noqa[: …F401…]` per-line directive; an import alias on one of
/// those lines is pinned individually.
pub(crate) fn scan_noqa_directives(
    parsed: &ParsedModuleRef,
    source: &str,
    line_index: &LineIndex,
) -> (bool, FxHashSet<usize>) {
    let mut file_pinned = false;
    let mut per_line_pins: FxHashSet<usize> = FxHashSet::default();
    for token in parsed.tokens().iter() {
        if token.kind() != TokenKind::Comment {
            continue;
        }
        let range = token.range();
        let text = &source[range];
        let Some(body) = text.strip_prefix('#') else {
            continue;
        };
        if is_file_pin(body) {
            file_pinned = true;
        }
        if is_per_line_pin(body) {
            let line = line_index.line_column(range.start(), source).line.get();
            per_line_pins.insert(line);
        }
    }
    (file_pinned, per_line_pins)
}

pub(crate) fn position(
    index: &LineIndex,
    source: &str,
    range: TextRange,
) -> (usize, usize, usize, usize) {
    let start = index.line_column(range.start(), source);
    let end = index.line_column(range.end(), source);
    (
        start.line.get(),
        start.column.get() - 1,
        end.line.get(),
        end.column.get() - 1,
    )
}

pub(crate) fn range_key(range: TextRange) -> (u32, u32) {
    (range.start().to_u32(), range.end().to_u32())
}

pub(crate) fn file_path_string(db: &dyn ProjectDb, file: File) -> String {
    match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    }
}

/// Per-source-type default flag mask ORed into every node minted from
/// this file. Only ``.ipynb`` opts in today — cells are inherently
/// entrypoints (run top-to-bottom, not imported) and the codemod must
/// skip the JSON envelope. ``.pyi`` decls get per-decl flagging in
/// ``ingest_decls`` (the flag depends on whether a matching ``.py``
/// runtime decl exists), so the file-level helper returns ``0`` for
/// them.
pub(crate) fn file_default_flags(db: &dyn ProjectDb, file: File) -> u32 {
    if file_path_string(db, file).ends_with(".ipynb") {
        NODE_FLAGS_NOTEBOOK_DEFAULT
    } else {
        0
    }
}

/// Convert a search-path-relative path into a dotted module name.
///
/// `rel` is the file's path relative to a search root (e.g.
/// `"lib_a/__init__.py"` or `"handlers/job.py"`). Strips `.py` / `.pyi`
/// extensions and any trailing `__init__` segment so a package's
/// `__init__.py` collapses to the package's dotted name.
///
/// Returns `None` for paths that aren't valid module containers — most
/// commonly a bare `__init__.py` whose search path *is* the package
/// directory itself (a layout no `.pth` install would produce, but we
/// reject it for safety: there's no name to bind).
fn relative_to_module_name(rel: &str) -> Option<String> {
    let stem = rel
        .strip_suffix(".pyi")
        .or_else(|| rel.strip_suffix(".py"))
        .unwrap_or(rel);
    if stem == "__init__" {
        return None;
    }
    let stem = stem
        .strip_suffix("/__init__")
        .or_else(|| stem.strip_suffix("\\__init__"))
        .unwrap_or(stem);
    if stem.is_empty() {
        return None;
    }
    Some(stem.replace(['/', '\\'], "."))
}

/// Specificity-aware reverse module lookup: which fqname would Python
/// import to reach this file?
///
/// `ty_module_resolver::file_to_module` takes the *first* search path
/// that contains the file. That's correct when search paths reflect
/// Python's resolution order — but when project_root is on the list
/// alongside a `.pth`-derived path inside it, the first match yields a
/// workspace-relative dotted name (`libs.lib_a.src.lib_a`) instead of
/// the import-time name (`lib_a`). The forward resolver still picks
/// the right file via the `.pth`, so cross-file edges built against
/// the workspace-relative fqname never connect.
///
/// This walks every containing search path, builds a candidate
/// module name from each, round-trips it through `resolve_module`, and
/// returns the candidate whose search path is *most* specific (largest
/// component count). The specificity tiebreak picks `lib_a` over
/// `libs.lib_a.src.lib_a` whenever both round-trip — i.e. when both a
/// `.pth` entry and the project root cover the file.
///
/// Returns `None` when no candidate round-trips (the file is shadowed
/// by another module of the same name in a higher-priority search
/// path) or when the file isn't on any search path at all. Callers
/// should fall back to a path-mangle in that case.
pub(crate) fn canonical_module_for_file<'db>(
    db: &'db dyn ty_module_resolver::Db,
    file: File,
) -> Option<Module<'db>> {
    let name = canonical_module_name_for_file(db, file)?;
    resolve_module(db, file, &name)
}

/// Same logic as :func:`canonical_module_for_file` but stops at the
/// ``ModuleName``. Saves one ``resolve_module`` call per file in the
/// common single-package case where ``module_fqname_for_file`` only
/// needs the dotted name. Callers that go on to walk the parent module
/// (``parent_module_file``, ``file_package_name``) still go through
/// the ``Module``-returning wrapper above.
pub(crate) fn canonical_module_name_for_file(
    db: &dyn ty_module_resolver::Db,
    file: File,
) -> Option<ModuleName> {
    let file_path = match file.path(db) {
        FilePath::System(p) => p,
        _ => return file_to_module(db, file).map(|m| m.name(db).clone()),
    };

    // Collect every search path that physically contains the file.
    // Typical projects have ≤ 5 search paths and a file usually lives
    // under 1–2 of them, so the SmallVec stays inline.
    let mut candidates: smallvec::SmallVec<[(usize, ModuleName); 4]> = smallvec::SmallVec::new();
    for sp in search_paths(db, ModuleResolveMode::StubsAllowed) {
        let Some(sp_path) = sp.as_system_path() else {
            continue;
        };
        let Ok(rel) = file_path.strip_prefix(sp_path) else {
            continue;
        };
        let Some(name_str) = relative_to_module_name(rel.as_str()) else {
            continue;
        };
        let Some(name) = ModuleName::new(&name_str) else {
            continue;
        };
        candidates.push((sp_path.components().count(), name));
    }

    // Fast path: a single containing search path means no ambiguity —
    // skip the ``resolve_module`` round-trip entirely. This is the
    // single-package shape (just ``project_root`` on the search list)
    // and accounts for the bulk of first-party files in typical
    // codebases. Without this short-circuit, the round-trip would cost
    // one ``resolve_module`` call per project file at cold start.
    //
    // Correctness note: ``file_to_module`` round-trips to reject files
    // shadowed by a same-named package (``foo.py`` next to
    // ``foo/__init__.py``). We deliberately don't reject here — the
    // current code path falls back to a path-mangle for those files
    // anyway, returning the same fqname we would. Shadowing precedence
    // for import-edge routing lives in ``file_payload`` / the trie,
    // not here.
    if candidates.len() == 1 {
        // No ambiguity, no round-trip needed.
        return candidates.into_iter().next().map(|(_, name)| name);
    }

    // Multiple containing paths -- specificity tiebreak with round-trip.
    candidates.sort_by_key(|c| std::cmp::Reverse(c.0));
    for (_, name) in candidates {
        let Some(resolved) = resolve_module(db, file, &name) else {
            continue;
        };
        let Some(resolved_file) = resolved.file(db) else {
            continue;
        };
        if resolved_file == file {
            return Some(name);
        }
    }
    None
}

pub(crate) fn module_fqname_for_file(db: &dyn ProjectDb, file: File) -> String {
    if let Some(name) = canonical_module_name_for_file(db, file) {
        return name.as_str().to_string();
    }
    // Fallback for files ty doesn't classify (e.g. ``gunicorn.conf.py`` at
    // the project root — there's no ``__init__.py`` and the stem contains
    // a dot, so ty's module resolver returns ``None``). Mirror the libcst
    // pipeline's ``strip .py / replace / with .`` derivation so plugins
    // like ``ServerConfigPlugin`` can address the file by its conventional
    // dotted name.
    let path_str = file_path_string(db, file);
    let root_str = db.project().root(db).as_str();
    let rel = path_str
        .strip_prefix(root_str)
        .unwrap_or(&path_str)
        .trim_start_matches('/')
        .trim_start_matches('\\');
    let stem = rel
        .strip_suffix(".pyi")
        .or_else(|| rel.strip_suffix(".py"))
        .unwrap_or(rel);
    stem.replace(['/', '\\'], ".")
}

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the AST / string helpers.
    //!
    //! Anything that takes a `Python<'_>` GIL token, a `Py<SymbolNode>`,
    //! a `Bound<'_, PyAny>`, a Salsa `ProjectDatabase`, a `File`, or a
    //! `ParsedModuleRef` (which is also DB-backed) is intentionally NOT
    //! covered here — the FFI surface is exercised end-to-end by the
    //! python suite. We focus on pure functions over primitives + raw
    //! AST nodes built from `parse_module(...)` / `parse_expression(...)`.
    use super::*;
    use ruff_python_parser::{parse_expression, parse_module};

    fn parse_expr(source: &str) -> Box<Expr> {
        parse_expression(source).unwrap().into_syntax().body
    }

    fn parse_stmts(source: &str) -> Vec<Stmt> {
        parse_module(source).unwrap().into_syntax().body
    }

    // -- is_dunder_name ----------------------------------------------------

    #[test]
    fn is_dunder_name_true_for_typical_dunders() {
        assert!(is_dunder_name("__init__"));
        assert!(is_dunder_name("__repr__"));
        assert!(is_dunder_name("mod.cls.__init__"));
    }

    #[test]
    fn is_dunder_name_false_for_single_underscore() {
        assert!(!is_dunder_name("_x"));
        assert!(!is_dunder_name("__x"));
        assert!(!is_dunder_name("x__"));
    }

    #[test]
    fn is_dunder_name_false_for_short_dunder() {
        // ``__`` and ``____`` are <= 4 chars after stripping `__` —
        // the helper requires len() > 4 so the dunder body is at least 1 char.
        assert!(!is_dunder_name("__"));
        assert!(!is_dunder_name("____"));
    }

    #[test]
    fn is_dunder_name_empty_string_ok() {
        assert!(!is_dunder_name(""));
        assert!(!is_dunder_name("foo.bar.baz"));
    }

    // -- rel_path ----------------------------------------------------------

    #[test]
    fn rel_path_roundtrips_unix_path() {
        let p = rel_path("foo/bar.py");
        assert_eq!(p.path().as_str(), "foo/bar.py");
    }

    #[test]
    fn rel_path_handles_empty_input() {
        // Should not panic on empty input — the `SystemPath::new` path
        // accepts zero-length strings.
        let p = rel_path("");
        assert_eq!(p.path().as_str(), "");
    }

    // -- is_name / is_string_literal --------------------------------------

    #[test]
    fn is_name_matches_bare_name() {
        let expr = parse_expr("foo");
        assert!(is_name(&expr, "foo"));
        assert!(!is_name(&expr, "bar"));
    }

    #[test]
    fn is_name_rejects_non_name_exprs() {
        let expr = parse_expr("foo.bar");
        assert!(!is_name(&expr, "foo"));
        assert!(!is_name(&expr, "bar"));
        let expr = parse_expr("123");
        assert!(!is_name(&expr, "123"));
    }

    #[test]
    fn is_string_literal_matches_exact_value() {
        let expr = parse_expr("'__main__'");
        assert!(is_string_literal(&expr, "__main__"));
        assert!(!is_string_literal(&expr, "other"));
    }

    #[test]
    fn is_string_literal_rejects_non_string() {
        let expr = parse_expr("123");
        assert!(!is_string_literal(&expr, "123"));
        let expr = parse_expr("foo");
        assert!(!is_string_literal(&expr, "foo"));
    }

    // -- is_name_eq_main --------------------------------------------------

    #[test]
    fn is_name_eq_main_matches_both_orderings() {
        let a = parse_expr("__name__ == '__main__'");
        let b = parse_expr("'__main__' == __name__");
        assert!(is_name_eq_main(&a));
        assert!(is_name_eq_main(&b));
    }

    #[test]
    fn is_name_eq_main_rejects_wrong_constant() {
        let expr = parse_expr("__name__ == 'main'");
        assert!(!is_name_eq_main(&expr));
    }

    #[test]
    fn is_name_eq_main_rejects_other_comparisons() {
        // !=, multiple comparators, non-name LHS, non-string RHS, etc.
        assert!(!is_name_eq_main(&parse_expr("__name__ != '__main__'")));
        assert!(!is_name_eq_main(&parse_expr("x == y == z")));
        assert!(!is_name_eq_main(&parse_expr("foo == '__main__'")));
        assert!(!is_name_eq_main(&parse_expr("123 == 456")));
    }

    // -- top_level_assign_to_name -----------------------------------------

    #[test]
    fn top_level_assign_to_name_matches_simple_assignment() {
        let stmts = parse_stmts("x = 42\n");
        let got = top_level_assign_to_name(&stmts[0]);
        assert!(got.is_some());
        let (_, expr) = got.unwrap();
        assert!(matches!(expr, Expr::NumberLiteral(_)));
    }

    #[test]
    fn top_level_assign_to_name_matches_ann_assignment() {
        let stmts = parse_stmts("x: int = 42\n");
        let got = top_level_assign_to_name(&stmts[0]);
        assert!(got.is_some());
    }

    #[test]
    fn top_level_assign_to_name_rejects_tuple_target() {
        let stmts = parse_stmts("x, y = 1, 2\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    #[test]
    fn top_level_assign_to_name_rejects_attribute_target() {
        let stmts = parse_stmts("obj.attr = 1\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    #[test]
    fn top_level_assign_to_name_rejects_ann_without_value() {
        let stmts = parse_stmts("x: int\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    #[test]
    fn top_level_assign_to_name_rejects_non_assign() {
        let stmts = parse_stmts("def foo(): pass\n");
        assert!(top_level_assign_to_name(&stmts[0]).is_none());
    }

    // -- class_body_defines_method -----------------------------------------

    #[test]
    fn class_body_defines_method_finds_named_method() {
        let stmts = parse_stmts("class C:\n    def m(self): pass\n    def n(self): pass\n");
        let cls = match &stmts[0] {
            Stmt::ClassDef(c) => c,
            _ => unreachable!(),
        };
        assert!(class_body_defines_method(cls, "m"));
        assert!(class_body_defines_method(cls, "n"));
        assert!(!class_body_defines_method(cls, "missing"));
    }

    #[test]
    fn class_body_defines_method_ignores_non_function_members() {
        let stmts = parse_stmts("class C:\n    x = 1\n    class Inner: pass\n");
        let cls = match &stmts[0] {
            Stmt::ClassDef(c) => c,
            _ => unreachable!(),
        };
        assert!(!class_body_defines_method(cls, "x"));
        assert!(!class_body_defines_method(cls, "Inner"));
    }

    // -- nth_positional_string --------------------------------------------

    fn first_call(source: &str) -> ruff_python_ast::ExprCall {
        let expr = parse_expr(source);
        match *expr {
            Expr::Call(c) => c,
            _ => panic!("expected a call expression"),
        }
    }

    #[test]
    fn nth_positional_string_returns_literal_value() {
        let call = first_call("f('hello', 'world')");
        assert_eq!(nth_positional_string(&call, 0), Some("hello".to_string()));
        assert_eq!(nth_positional_string(&call, 1), Some("world".to_string()));
    }

    #[test]
    fn nth_positional_string_returns_none_for_non_string() {
        let call = first_call("f(42, foo)");
        assert_eq!(nth_positional_string(&call, 0), None);
        assert_eq!(nth_positional_string(&call, 1), None);
    }

    #[test]
    fn nth_positional_string_out_of_range_returns_none() {
        let call = first_call("f('a')");
        assert_eq!(nth_positional_string(&call, 5), None);
    }

    #[test]
    fn nth_positional_string_rejects_f_string() {
        // f-strings are not StringLiteral nodes — should be rejected.
        let call = first_call("f(f'value-{x}')");
        assert_eq!(nth_positional_string(&call, 0), None);
    }

    // -- call_callee_matches_var ------------------------------------------

    #[test]
    fn call_callee_matches_var_matches_owner_attr() {
        let call = first_call("mocker.patch('x')");
        assert!(call_callee_matches_var(&call, "mocker", "patch", None));
        assert!(call_callee_matches_var(&call, "mocker", "patch", Some(1)));
        assert!(!call_callee_matches_var(&call, "mocker", "patch", Some(2)));
    }

    #[test]
    fn call_callee_matches_var_rejects_wrong_owner_or_attr() {
        let call = first_call("mocker.patch('x')");
        assert!(!call_callee_matches_var(&call, "other", "patch", None));
        assert!(!call_callee_matches_var(&call, "mocker", "spy", None));
    }

    #[test]
    fn call_callee_matches_var_rejects_non_attribute_callee() {
        let call = first_call("f('x')");
        assert!(!call_callee_matches_var(&call, "f", "patch", None));
    }

    #[test]
    fn call_callee_matches_var_rejects_chained_receiver() {
        // ``a.b.patch(...)`` has ``a.b`` (Attribute) as the receiver, not
        // a bare Name — should not match.
        let call = first_call("a.b.patch('x')");
        assert!(!call_callee_matches_var(&call, "b", "patch", None));
    }

    // -- range_key ---------------------------------------------------------

    #[test]
    fn range_key_packs_start_end() {
        use ruff_text_size::TextSize;
        let range = TextRange::new(TextSize::from(5), TextSize::from(12));
        assert_eq!(range_key(range), (5, 12));
    }

    // -- evaluate_truthiness ----------------------------------------------

    #[test]
    fn evaluate_truthiness_literals() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(evaluate_truthiness(&parse_expr("True"), &empty), Some(true));
        assert_eq!(
            evaluate_truthiness(&parse_expr("False"), &empty),
            Some(false)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("None"), &empty),
            Some(false)
        );
        assert_eq!(evaluate_truthiness(&parse_expr("..."), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("0"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("1"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("0.0"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("3.14"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("''"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("'x'"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("[]"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("[1]"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("()"), &empty), Some(false));
        assert_eq!(evaluate_truthiness(&parse_expr("(1,)"), &empty), Some(true));
        assert_eq!(evaluate_truthiness(&parse_expr("{}"), &empty), Some(false));
        assert_eq!(
            evaluate_truthiness(&parse_expr("{'a': 1}"), &empty),
            Some(true)
        );
    }

    #[test]
    fn evaluate_truthiness_names_via_table() {
        let mut table: NameTable = FxHashMap::default();
        table.insert("DEBUG".into(), true);
        table.insert("RELEASE".into(), false);
        assert_eq!(
            evaluate_truthiness(&parse_expr("DEBUG"), &table),
            Some(true)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("RELEASE"), &table),
            Some(false)
        );
        assert_eq!(evaluate_truthiness(&parse_expr("OTHER"), &table), None);
    }

    #[test]
    fn evaluate_truthiness_unary_not() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(
            evaluate_truthiness(&parse_expr("not True"), &empty),
            Some(false)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("not False"), &empty),
            Some(true)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("not unknown_name"), &empty),
            None
        );
        // Other unary ops aren't folded.
        assert_eq!(evaluate_truthiness(&parse_expr("-1"), &empty), None);
    }

    #[test]
    fn evaluate_truthiness_bool_ops_short_circuit_or() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(
            evaluate_truthiness(&parse_expr("True or unknown"), &empty),
            Some(true)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("False or unknown"), &empty),
            None
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("False or False"), &empty),
            Some(false)
        );
    }

    #[test]
    fn evaluate_truthiness_bool_ops_short_circuit_and() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(
            evaluate_truthiness(&parse_expr("False and unknown"), &empty),
            Some(false)
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("True and unknown"), &empty),
            None
        );
        assert_eq!(
            evaluate_truthiness(&parse_expr("True and True"), &empty),
            Some(true)
        );
    }

    #[test]
    fn evaluate_truthiness_returns_none_for_unsupported() {
        let empty: NameTable = FxHashMap::default();
        assert_eq!(evaluate_truthiness(&parse_expr("f()"), &empty), None);
        assert_eq!(evaluate_truthiness(&parse_expr("a.b"), &empty), None);
        assert_eq!(evaluate_truthiness(&parse_expr("a == b"), &empty), None);
    }

    #[test]
    fn evaluate_truthiness_walrus_unwraps_value() {
        let empty: NameTable = FxHashMap::default();
        // ``(x := True)`` should fold to True (the walrus's value).
        assert_eq!(
            evaluate_truthiness(&parse_expr("(x := True)"), &empty),
            Some(true)
        );
    }

    // -- stmt_is_terminator / suite_terminates ----------------------------

    #[test]
    fn stmt_is_terminator_basic_terminators() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    return 1\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));

        let stmts = parse_stmts("def f():\n    raise X\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_break_continue_inside_loop() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("for _ in []:\n    break\n");
        let body = match &stmts[0] {
            Stmt::For(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));

        let stmts = parse_stmts("for _ in []:\n    continue\n");
        let body = match &stmts[0] {
            Stmt::For(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(stmt_is_terminator(&body[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_assert_falsy() {
        let empty: NameTable = FxHashMap::default();
        // `assert False` is a terminator; `assert True` is not.
        let stmts = parse_stmts("assert False\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("assert True\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("assert 0\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("assert unknown_var\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_constant_truthy_body_terminates() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if True:\n    return\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_without_else_not_terminator() {
        let empty: NameTable = FxHashMap::default();
        // No else clause + a non-constant test means the if might be
        // skipped — not a terminator.
        let stmts = parse_stmts("if cond:\n    return\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_else_both_terminate() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if cond:\n    return\nelse:\n    raise X\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_if_else_one_falls_through() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if cond:\n    return\nelse:\n    pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_try_finally_terminates() {
        let empty: NameTable = FxHashMap::default();
        let stmts =
            parse_stmts("try:\n    x = 1\nexcept Exception:\n    pass\nfinally:\n    return\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_try_body_and_all_handlers_terminate() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("try:\n    return\nexcept Exception:\n    raise\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_try_handler_falls_through_not_terminator() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("try:\n    return\nexcept Exception:\n    pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_with_body_terminates() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("with cm as c:\n    return\n");
        assert!(stmt_is_terminator(&stmts[0], &empty));
        let stmts = parse_stmts("with cm as c:\n    pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn stmt_is_terminator_pass_is_not() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("pass\n");
        assert!(!stmt_is_terminator(&stmts[0], &empty));
    }

    #[test]
    fn suite_terminates_finds_any_terminator() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    x = 1\n    return 2\n    y = 3\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(suite_terminates(body, &empty));

        let stmts = parse_stmts("def f():\n    x = 1\n    y = 2\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.body,
            _ => unreachable!(),
        };
        assert!(!suite_terminates(body, &empty));
    }

    // -- walk_suite_for_dead / detect_dead_ranges (via module body) -------

    #[test]
    fn walk_suite_for_dead_marks_post_terminator_statements() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    return 1\n    x = 2\n    y = 3\n");
        let body = match &stmts[0] {
            Stmt::FunctionDef(f) => f.body.clone(),
            _ => unreachable!(),
        };
        let mut dead = Vec::new();
        walk_suite_for_dead(&body, &empty, &mut dead);
        // x = 2 and y = 3 should both be marked.
        assert_eq!(dead.len(), 2);
    }

    #[test]
    fn walk_suite_for_dead_handles_if_false() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if False:\n    x = 1\n    y = 2\n");
        let mut dead = Vec::new();
        walk_suite_for_dead(&stmts, &empty, &mut dead);
        // Both statements in the False body are marked individually.
        assert_eq!(dead.len(), 2);
    }

    #[test]
    fn walk_suite_for_dead_handles_if_true_drops_else() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("if True:\n    x = 1\nelse:\n    y = 2\n");
        let mut dead = Vec::new();
        walk_suite_for_dead(&stmts, &empty, &mut dead);
        // The else clause is dead.
        assert_eq!(dead.len(), 1);
    }

    #[test]
    fn walk_suite_for_dead_recurses_into_function_body() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def outer():\n    return\n    nested = 1\n");
        let mut dead = Vec::new();
        walk_suite_for_dead(&stmts, &empty, &mut dead);
        assert_eq!(dead.len(), 1);
    }

    #[test]
    fn walk_suite_for_dead_empty_input_yields_nothing() {
        let empty: NameTable = FxHashMap::default();
        let mut dead = Vec::new();
        walk_suite_for_dead(&[], &empty, &mut dead);
        assert!(dead.is_empty());
    }

    // -- build_scope_table / collect_scope_bindings -----------------------

    #[test]
    fn build_scope_table_folds_simple_constant() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("DEBUG = True\nX = False\n");
        let table = build_scope_table(&stmts, &empty);
        assert_eq!(table.get("DEBUG"), Some(&true));
        assert_eq!(table.get("X"), Some(&false));
    }

    #[test]
    fn build_scope_table_handles_chained_constants() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("A = False\nB = A or False\nC = B\n");
        let table = build_scope_table(&stmts, &empty);
        assert_eq!(table.get("A"), Some(&false));
        assert_eq!(table.get("B"), Some(&false));
        assert_eq!(table.get("C"), Some(&false));
    }

    #[test]
    fn build_scope_table_drops_conflicting_bindings() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("X = True\nX = False\n");
        let table = build_scope_table(&stmts, &empty);
        assert!(!table.contains_key("X"));
    }

    #[test]
    fn build_scope_table_inherits_enclosing_table() {
        let mut enclosing: NameTable = FxHashMap::default();
        enclosing.insert("OUTER".into(), true);
        let stmts = parse_stmts("INNER = False\n");
        let table = build_scope_table(&stmts, &enclosing);
        assert_eq!(table.get("OUTER"), Some(&true));
        assert_eq!(table.get("INNER"), Some(&false));
    }

    #[test]
    fn build_scope_table_self_negation_does_not_loop() {
        // Inheriting ``foo`` from the enclosing scope and then evaluating
        // ``foo = not foo`` would oscillate the table between true/false
        // forever before the poison set was added.
        let mut enclosing: NameTable = FxHashMap::default();
        enclosing.insert("foo".into(), false);
        let stmts = parse_stmts("foo = not foo\n");
        let table = build_scope_table(&stmts, &enclosing);
        assert!(!table.contains_key("foo"));
    }

    #[test]
    fn build_scope_table_detect_dead_ranges_terminates_for_global_flipper() {
        // End-to-end: module-level ``foo = False`` plus a function that
        // does ``global foo; foo = not foo``. ``detect_dead_ranges``
        // recurses into the function body via ``build_scope_table``,
        // which used to spin forever on the ``not foo`` binding.
        let parsed = ruff_python_parser::parse_module(
            "foo = False\ndef flip():\n    global foo\n    foo = not foo\n",
        )
        .unwrap();
        let mut bindings: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        let body = parsed.syntax().body.clone();
        collect_scope_bindings(&body, &mut bindings);
        // Sanity: module sees ``foo = False`` only, not the in-function rebind.
        assert!(bindings.contains_key("foo"));

        // The inner ``flip`` body inherits ``foo = false``; building its
        // table must terminate and drop ``foo`` rather than fold it.
        let mut enclosing: NameTable = FxHashMap::default();
        enclosing.insert("foo".into(), false);
        if let Stmt::FunctionDef(f) = &body[1] {
            let nested = build_scope_table(&f.body, &enclosing);
            assert!(!nested.contains_key("foo"));
        } else {
            panic!("second stmt should be a FunctionDef");
        }
    }

    #[test]
    fn build_scope_table_ignores_nested_function_bindings() {
        let empty: NameTable = FxHashMap::default();
        let stmts = parse_stmts("def f():\n    X = True\n");
        let table = build_scope_table(&stmts, &empty);
        // ``X`` is defined inside ``f`` — separate scope.
        assert!(!table.contains_key("X"));
    }

    // -- collect_walrus_in_expr -------------------------------------------

    #[test]
    fn collect_walrus_in_expr_picks_up_assignment_target() {
        let expr = parse_expr("(x := 5)");
        let mut out: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        collect_walrus_in_expr(&expr, &mut out);
        assert!(out.contains_key("x"));
    }

    #[test]
    fn collect_walrus_in_expr_descends_compound_exprs() {
        let expr = parse_expr("(a := 1) or (b := 2) and (c := 3)");
        let mut out: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        collect_walrus_in_expr(&expr, &mut out);
        assert!(out.contains_key("a"));
        assert!(out.contains_key("b"));
        assert!(out.contains_key("c"));
    }

    #[test]
    fn collect_walrus_in_expr_no_walrus_yields_empty() {
        let expr = parse_expr("a + b");
        let mut out: FxHashMap<String, Vec<&Expr>> = FxHashMap::default();
        collect_walrus_in_expr(&expr, &mut out);
        assert!(out.is_empty());
    }

    // -- noqa helpers (extending the existing parse_noqa_tail tests) ------

    #[test]
    fn parse_noqa_tail_case_insensitive_prefix() {
        assert_eq!(parse_noqa_tail("noqa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("NoQa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("nOqA"), Some(NoqaKind::Bare));
    }

    #[test]
    fn parse_noqa_tail_rejects_non_noqa_comment() {
        assert_eq!(parse_noqa_tail("type: ignore"), None);
        assert_eq!(parse_noqa_tail("comment about noqa"), None);
    }

    #[test]
    fn parse_noqa_tail_codes_are_trimmed_and_case_insensitive() {
        assert_eq!(parse_noqa_tail("noqa: f401"), Some(NoqaKind::F401Present));
        assert_eq!(
            parse_noqa_tail("noqa:  F401  ,  E501"),
            Some(NoqaKind::F401Present)
        );
        assert_eq!(
            parse_noqa_tail("noqa: E501,  E502"),
            Some(NoqaKind::OtherOnly)
        );
    }

    #[test]
    fn parse_noqa_tail_handles_leading_whitespace() {
        assert_eq!(
            parse_noqa_tail("    noqa: F401"),
            Some(NoqaKind::F401Present)
        );
    }

    #[test]
    fn is_per_line_pin_basic() {
        assert!(is_per_line_pin(" noqa"));
        assert!(is_per_line_pin(" noqa: F401"));
        assert!(is_per_line_pin(" noqa: E501, F401"));
        assert!(!is_per_line_pin(" noqa: E501"));
        assert!(!is_per_line_pin(" type: ignore"));
        assert!(!is_per_line_pin(""));
    }

    #[test]
    fn is_file_pin_recognizes_ruff_and_flake8_prefixes() {
        assert!(is_file_pin(" ruff: noqa"));
        assert!(is_file_pin(" ruff: noqa: F401"));
        assert!(is_file_pin(" flake8: noqa"));
        assert!(is_file_pin(" flake8: noqa: F401"));
    }

    #[test]
    fn is_file_pin_other_codes_dont_pin_f401() {
        assert!(!is_file_pin(" ruff: noqa: E501"));
        assert!(!is_file_pin(" flake8: noqa: W292, E303"));
    }

    #[test]
    fn is_file_pin_unrelated_comments_rejected() {
        assert!(!is_file_pin(" pyright: ignore"));
        assert!(!is_file_pin(" type: ignore"));
        assert!(!is_file_pin(" noqa"));
        assert!(!is_file_pin(""));
    }

    #[test]
    fn is_file_pin_prefix_is_case_sensitive() {
        // ``ruff:`` / ``flake8:`` are matched case-sensitively per ruff's docs.
        assert!(!is_file_pin(" Ruff: noqa"));
        assert!(!is_file_pin(" FLAKE8: noqa"));
    }

    // -- decorators_match_imports -----------------------------------------

    #[test]
    fn decorators_match_imports_via_local_alias() {
        let stmts = parse_stmts("@register\ndef f(): pass\n");
        let decorators = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.decorator_list,
            _ => unreachable!(),
        };
        let mut imports: FxHashMap<String, String> = FxHashMap::default();
        imports.insert("register".into(), "register".into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("register");
        let got = decorators_match_imports(decorators, &imports, &allowed);
        assert!(got.is_some());
    }

    #[test]
    fn decorators_match_imports_via_module_attr() {
        let stmts = parse_stmts("@flask.route('/x')\ndef f(): pass\n");
        let decorators = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.decorator_list,
            _ => unreachable!(),
        };
        let mut imports: FxHashMap<String, String> = FxHashMap::default();
        imports.insert("flask".into(), MODULE_ALIAS_MARKER.into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("route");
        let got = decorators_match_imports(decorators, &imports, &allowed);
        // Has a call form.
        assert!(matches!(got, Some(Some(_))));
    }

    #[test]
    fn decorators_match_imports_bare_attribute() {
        let stmts = parse_stmts("@app.task\ndef f(): pass\n");
        let decorators = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.decorator_list,
            _ => unreachable!(),
        };
        let mut imports: FxHashMap<String, String> = FxHashMap::default();
        imports.insert("app".into(), MODULE_ALIAS_MARKER.into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("task");
        let got = decorators_match_imports(decorators, &imports, &allowed);
        // Bare (no call) — outer Some, inner None.
        assert!(matches!(got, Some(None)));
    }

    #[test]
    fn decorators_match_imports_returns_none_when_no_match() {
        let stmts = parse_stmts("@other\ndef f(): pass\n");
        let decorators = match &stmts[0] {
            Stmt::FunctionDef(f) => &f.decorator_list,
            _ => unreachable!(),
        };
        let imports: FxHashMap<String, String> = FxHashMap::default();
        let allowed: FxHashSet<&str> = FxHashSet::default();
        assert!(decorators_match_imports(decorators, &imports, &allowed).is_none());
    }

    // -- matched_call_target ----------------------------------------------

    #[test]
    fn matched_call_target_via_local_name() {
        let call = first_call("Flask(__name__)");
        let mut imports: FxHashMap<String, String> = FxHashMap::default();
        imports.insert("Flask".into(), "Flask".into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("Flask");
        assert_eq!(
            matched_call_target(&call, &imports, "flask", &allowed),
            Some("Flask".into())
        );
    }

    #[test]
    fn matched_call_target_via_module_alias_attr() {
        let call = first_call("flask.Flask(__name__)");
        let mut imports: FxHashMap<String, String> = FxHashMap::default();
        imports.insert("flask".into(), MODULE_ALIAS_MARKER.into());
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("Flask");
        assert_eq!(
            matched_call_target(&call, &imports, "flask", &allowed),
            Some("Flask".into())
        );
    }

    #[test]
    fn matched_call_target_via_multi_seg_module() {
        let call = first_call("unittest.mock.patch('x')");
        // No file imports entry for the leftmost name — fallback to the
        // literal dotted match against ``module``.
        let imports: FxHashMap<String, String> = FxHashMap::default();
        let mut allowed: FxHashSet<&str> = FxHashSet::default();
        allowed.insert("patch");
        assert_eq!(
            matched_call_target(&call, &imports, "unittest.mock", &allowed),
            Some("patch".into())
        );
    }

    #[test]
    fn matched_call_target_rejects_unknown_name() {
        let call = first_call("unrelated()");
        let imports: FxHashMap<String, String> = FxHashMap::default();
        let allowed: FxHashSet<&str> = FxHashSet::default();
        assert!(matched_call_target(&call, &imports, "x", &allowed).is_none());
    }

    #[test]
    fn matched_call_target_rejects_disallowed_name() {
        let call = first_call("Flask()");
        let mut imports: FxHashMap<String, String> = FxHashMap::default();
        imports.insert("Flask".into(), "Flask".into());
        // Empty allowed set: target is present but not allowed.
        let allowed: FxHashSet<&str> = FxHashSet::default();
        assert!(matched_call_target(&call, &imports, "flask", &allowed).is_none());
    }

    // -- NoqaKind helper --------------------------------------------------

    #[test]
    fn noqakind_pins_f401() {
        assert!(NoqaKind::Bare.pins_f401());
        assert!(NoqaKind::F401Present.pins_f401());
        assert!(!NoqaKind::OtherOnly.pins_f401());
    }
}
