//! Leftover utilities shared across modules: AST navigation helpers,
//! call-args/kwargs extraction for the chainable query payloads,
//! dead-region detection, noqa scanning, position/file helpers, and the
//! shared `NodeFlags`/`EdgeFlags` constant aliases.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
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
use ty_module_resolver::{file_to_module, resolve_module, ModuleName};
use ty_project::metadata::value::RelativePathBuf;
use ty_project::{Db as ProjectDb, ProjectDatabase};

use crate::graph::{DeclIndex, EdgeFlags, SymbolNode, NodeFlags};
use crate::ingest::collapse_attribute_chain;
use crate::project::BuildOutputs;

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
///   ``imports[alias] == "<module>"`` and ``attr`` is in ``names``
///   (covers ``import module`` and ``import module as alias``).
pub(crate) fn decorators_match_imports<'ast>(
    decorators: &'ast [ruff_python_ast::Decorator],
    imports: &HashMap<String, String>,
    names: &HashSet<&str>,
) -> Option<Option<&'ast ruff_python_ast::ExprCall>> {
    for dec in decorators {
        let (root_expr, call_form) = match &dec.expression {
            Expr::Call(call) => (&*call.func, Some(call)),
            other => (other, None),
        };
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
                    if imports.get(prefix.id.as_str()).map(String::as_str) == Some("<module>")
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
    path_to_file: &HashMap<String, File>,
    path: &str,
    class_node: &SymbolNode,
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
pub(crate) fn find_subclass_indices_via_refs(
    db: &ProjectDatabase,
    outputs: &BuildOutputs,
    seed_file: File,
    seed_name_range: TextRange,
    transitive: bool,
) -> Vec<usize> {
    let mut out_idx: HashSet<usize> = HashSet::new();
    let mut visited_seeds: HashSet<(File, (u32, u32))> = HashSet::new();
    let mut queue: Vec<(File, TextRange)> = vec![(seed_file, seed_name_range)];

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
                let Some(&idx) = outputs.class_by_selection.get(&key) else {
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
    py: Python<'_>,
    fqn: &str,
) -> Option<(File, TextRange)> {
    // Project class: cheap path through the existing indices.
    if let Some(idxs) = outputs.decl_by_fqname.get(fqn) {
        for &idx in idxs {
            let node = outputs.builder.nodes[idx].borrow(py);
            if node.kind != "class" {
                continue;
            }
            let path = node.path.clone();
            if let Some(seed) = locate_class_def(db, &outputs.path_to_file, &path, &node) {
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
    let mut visited: HashSet<File> = HashSet::new();
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
    visited: &mut HashSet<File>,
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
pub(crate) fn collect_all_imports_local(parsed: &ParsedModuleRef) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
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

/// Build the file-local imports map ``{local_name: target}`` for
/// names imported from ``module``. ``target`` is the upstream
/// constructor / decl name (e.g. ``"Flask"`` when bound via
/// ``from flask import Flask``) or the sentinel ``"<module>"``
/// when bound via ``import flask`` / ``import flask as f``.
///
/// Only entries whose target is in ``allowed`` survive — keeps the
/// map small and lets call-site matchers do a cheap second check.
pub(crate) fn collect_module_imports_local(
    parsed: &ParsedModuleRef,
    module: &str,
    allowed: &HashSet<&str>,
) -> HashMap<String, String> {
    // Submodule binding: ``from <parent> import <last_seg>`` makes
    // ``last_seg`` a local alias for the queried module (e.g.
    // ``from unittest import mock`` for module ``unittest.mock``).
    let parent_last = module.rsplit_once('.');
    let mut out = HashMap::new();
    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::ImportFrom(im) => {
                let Some(mod_name) = &im.module else { continue };
                if mod_name.as_str() == module {
                    for alias in &im.names {
                        let target = alias.name.as_str();
                        if !allowed.contains(target) {
                            continue;
                        }
                        let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(target);
                        out.insert(local.to_string(), target.to_string());
                    }
                } else if let Some((parent, last)) = parent_last {
                    if mod_name.as_str() == parent {
                        for alias in &im.names {
                            if alias.name.as_str() != last {
                                continue;
                            }
                            let local = alias.asname.as_ref().map(|n| n.as_str()).unwrap_or(last);
                            out.insert(local.to_string(), "<module>".to_string());
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
                    out.insert(local.to_string(), "<module>".to_string());
                }
            }
            _ => {}
        }
    }
    out
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
    imports: &HashMap<String, String>,
    module: &str,
    allowed: &HashSet<&str>,
) -> Option<String> {
    match call.func.as_ref() {
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
                    == Some("<module>"))
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
/// of constructor names called anywhere inside it.
pub(crate) struct FactoryCallFinder<'a> {
    pub(crate) imports: &'a HashMap<String, String>,
    pub(crate) module: &'a str,
    pub(crate) allowed: &'a HashSet<&'a str>,
    pub(crate) kinds: HashSet<String>,
}

impl<'ast, 'a> Visitor<'ast> for FactoryCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Some(name) = matched_call_target(call, self.imports, self.module, self.allowed) {
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
    decl_by_name_range: &HashMap<(File, (u32, u32)), usize>,
    module_nodes_by_file: &HashMap<File, usize>,
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

/// Send-able value extracted from an AST expression.
///
/// ``DeclRef(idx)`` holds an index into ``BuildOutputs.builder.nodes``.
/// The ``Py<SymbolNode>`` materialization runs in the GIL-holding
/// caller after :fn:`par_scan_files` returns (the rust extractor must
/// stay ``Send``).
///
/// ``Unknown`` is reported when the expression is neither a recognized
/// literal nor a name/attribute that resolves through the file's
/// imports to a project decl. It surfaces to Python as ``None`` —
/// callers who care about the difference between "literal None" and
/// "unresolvable" should also inspect the original source.
#[derive(Clone, Debug)]
pub(crate) enum ArgValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<ArgValue>),
    Tuple(Vec<ArgValue>),
    DeclRef(usize),
    Unknown,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CallArgs {
    pub(crate) args: Vec<ArgValue>,
    pub(crate) kwargs: HashMap<String, ArgValue>,
}

/// Resolve a dotted Name / Attribute chain to a project decl index
/// using the file's imports map. Returns the matching node index when
/// the leftmost ``Name`` is a known local import and the resulting
/// dotted upstream fqname (``upstream + remaining attrs``) is present
/// in ``decl_by_fqname``. Picks the first index when multiple bindings
/// exist.
pub(crate) fn resolve_dotted_name_to_decl(
    root: &str,
    segs: &[&str],
    file_imports: &HashMap<String, String>,
    decl_by_fqname: &HashMap<String, Vec<usize>>,
) -> Option<usize> {
    let upstream = file_imports.get(root)?;
    let mut fqn =
        String::with_capacity(upstream.len() + segs.iter().map(|s| s.len() + 1).sum::<usize>());
    fqn.push_str(upstream);
    for seg in segs {
        fqn.push('.');
        fqn.push_str(seg);
    }
    let idxs = decl_by_fqname.get(&fqn)?;
    idxs.first().copied()
}

/// Convert one AST argument expression to an ``ArgValue``.
pub(crate) fn extract_arg_value(
    expr: &Expr,
    file_imports: &HashMap<String, String>,
    decl_by_fqname: &HashMap<String, Vec<usize>>,
) -> ArgValue {
    match expr {
        Expr::NoneLiteral(_) => ArgValue::None,
        Expr::BooleanLiteral(b) => ArgValue::Bool(b.value),
        Expr::StringLiteral(s) => ArgValue::Str(s.value.to_str().to_string()),
        Expr::NumberLiteral(n) => match &n.value {
            ruff_python_ast::Number::Int(i) => match i.as_i64() {
                Some(v) => ArgValue::Int(v),
                None => ArgValue::Unknown,
            },
            ruff_python_ast::Number::Float(f) => ArgValue::Float(*f),
            ruff_python_ast::Number::Complex { .. } => ArgValue::Unknown,
        },
        Expr::List(list) => ArgValue::List(
            list.elts
                .iter()
                .map(|e| extract_arg_value(e, file_imports, decl_by_fqname))
                .collect(),
        ),
        Expr::Tuple(tup) => ArgValue::Tuple(
            tup.elts
                .iter()
                .map(|e| extract_arg_value(e, file_imports, decl_by_fqname))
                .collect(),
        ),
        Expr::Name(n) => {
            if let Some(idx) =
                resolve_dotted_name_to_decl(n.id.as_str(), &[], file_imports, decl_by_fqname)
            {
                ArgValue::DeclRef(idx)
            } else {
                ArgValue::Unknown
            }
        }
        Expr::Attribute(_) => {
            if let Some((root, segs)) = collapse_attribute_chain(expr) {
                if let Some(idx) = resolve_dotted_name_to_decl(
                    root.id.as_str(),
                    &segs,
                    file_imports,
                    decl_by_fqname,
                ) {
                    return ArgValue::DeclRef(idx);
                }
            }
            ArgValue::Unknown
        }
        _ => ArgValue::Unknown,
    }
}

/// Extract positional + keyword arguments from a call.
pub(crate) fn extract_call_args_kwargs(
    call: &ruff_python_ast::ExprCall,
    file_imports: &HashMap<String, String>,
    decl_by_fqname: &HashMap<String, Vec<usize>>,
) -> CallArgs {
    let args: Vec<ArgValue> = call
        .arguments
        .args
        .iter()
        .map(|a| extract_arg_value(a, file_imports, decl_by_fqname))
        .collect();
    let mut kwargs: HashMap<String, ArgValue> = HashMap::new();
    for kw in &call.arguments.keywords {
        let Some(name) = kw.arg.as_ref() else {
            continue;
        };
        kwargs.insert(
            name.as_str().to_string(),
            extract_arg_value(&kw.value, file_imports, decl_by_fqname),
        );
    }
    CallArgs { args, kwargs }
}

/// Materialize one ``ArgValue`` into a Python object. ``DeclRef``
/// resolves through the build's ``Py<SymbolNode>`` pool.
pub(crate) fn arg_value_to_py(py: Python<'_>, v: &ArgValue, nodes: &[Py<SymbolNode>]) -> PyObject {
    match v {
        ArgValue::None => py.None(),
        ArgValue::Bool(b) => b.into_py(py),
        ArgValue::Int(i) => i.into_py(py),
        ArgValue::Float(f) => f.into_py(py),
        ArgValue::Str(s) => s.into_py(py),
        ArgValue::List(items) => {
            let py_items: Vec<PyObject> = items
                .iter()
                .map(|v| arg_value_to_py(py, v, nodes))
                .collect();
            pyo3::types::PyList::new_bound(py, py_items).into_py(py)
        }
        ArgValue::Tuple(items) => {
            let py_items: Vec<PyObject> = items
                .iter()
                .map(|v| arg_value_to_py(py, v, nodes))
                .collect();
            pyo3::types::PyTuple::new_bound(py, py_items).into_py(py)
        }
        ArgValue::DeclRef(idx) => match nodes.get(*idx) {
            Some(node) => node.clone_ref(py).into_py(py),
            None => py.None(),
        },
        ArgValue::Unknown => py.None(),
    }
}

/// Convert a list of ``ArgValue`` to a `Vec<Py<PyAny>>` ready for a
/// frozen pyclass field.
pub(crate) fn args_to_py_vec(
    py: Python<'_>,
    args: &[ArgValue],
    nodes: &[Py<SymbolNode>],
) -> Vec<Py<PyAny>> {
    args.iter().map(|v| arg_value_to_py(py, v, nodes)).collect()
}

/// Convert a kwargs map to ``HashMap<String, Py<PyAny>>``.
pub(crate) fn kwargs_to_py_map(
    py: Python<'_>,
    kwargs: &HashMap<String, ArgValue>,
    nodes: &[Py<SymbolNode>],
) -> HashMap<String, Py<PyAny>> {
    kwargs
        .iter()
        .map(|(k, v)| (k.clone(), arg_value_to_py(py, v, nodes)))
        .collect()
}

/// A user-supplied kwarg matcher. Only literal-value equality is
/// supported; ``SymbolNode``-valued matchers are rejected at
/// ``where_kwarg`` call time.
#[derive(Clone, Debug)]
pub(crate) enum KwargMatcher {
    Literal(ArgValue),
}

/// True iff two ``ArgValue``s are equal as literals. ``Unknown`` never
/// compares equal to anything.
pub(crate) fn arg_value_eq_literal(a: &ArgValue, b: &ArgValue) -> bool {
    match (a, b) {
        (ArgValue::None, ArgValue::None) => true,
        (ArgValue::Bool(x), ArgValue::Bool(y)) => x == y,
        (ArgValue::Int(x), ArgValue::Int(y)) => x == y,
        (ArgValue::Float(x), ArgValue::Float(y)) => x == y,
        // Allow mixed int/float comparison for ergonomic matching.
        (ArgValue::Int(x), ArgValue::Float(y)) => (*x as f64) == *y,
        (ArgValue::Float(x), ArgValue::Int(y)) => *x == (*y as f64),
        (ArgValue::Str(x), ArgValue::Str(y)) => x == y,
        (ArgValue::List(xs), ArgValue::List(ys))
        | (ArgValue::Tuple(xs), ArgValue::Tuple(ys))
        | (ArgValue::List(xs), ArgValue::Tuple(ys))
        | (ArgValue::Tuple(xs), ArgValue::List(ys)) => {
            xs.len() == ys.len()
                && xs
                    .iter()
                    .zip(ys.iter())
                    .all(|(a, b)| arg_value_eq_literal(a, b))
        }
        _ => false,
    }
}

/// Extract a ``KwargMatcher`` from a Python value supplied at
/// ``.where_kwarg(name, value)`` call time. Accepts only Python
/// literals (``None``, ``bool``, ``int``, ``float``, ``str``,
/// ``list[...]``, ``tuple[...]``); passing a ``SymbolNode`` (or any
/// other unsupported type) raises ``PyValueError``.
pub(crate) fn kwarg_matcher_from_py(py: Python<'_>, value: &Py<PyAny>) -> PyResult<KwargMatcher> {
    let bound = value.bind(py);
    // Reject SymbolNode explicitly with a targeted error so callers
    // see the dropped feature rather than the generic "unknown type"
    // message from `py_to_arg_value`.
    if bound.extract::<PyRef<'_, SymbolNode>>().is_ok() {
        return Err(PyValueError::new_err(
            "where_kwarg value must be a Python literal (str/int/float/bool/None/list/tuple), got SymbolNode",
        ));
    }
    Ok(KwargMatcher::Literal(py_to_arg_value(bound)?))
}

/// Convert a Python value to an ``ArgValue`` literal. Errors when the
/// value isn't a recognized literal shape (so the matcher never
/// silently matches "anything").
pub(crate) fn py_to_arg_value(value: &Bound<'_, PyAny>) -> PyResult<ArgValue> {
    if value.is_none() {
        return Ok(ArgValue::None);
    }
    // Order matters: bool is a subclass of int in Python, so we must
    // check bool BEFORE int. ``extract::<bool>()`` succeeds for both
    // True/False and the literal ints 0/1 — the strict ``is_instance_of``
    // guard distinguishes a real bool from a coerced int.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(ArgValue::Bool(value.extract::<bool>()?));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(ArgValue::Int(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(ArgValue::Float(f));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(ArgValue::Str(s));
    }
    if let Ok(list) = value.downcast::<pyo3::types::PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_arg_value(&item)?);
        }
        return Ok(ArgValue::List(out));
    }
    if let Ok(tup) = value.downcast::<pyo3::types::PyTuple>() {
        let mut out = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            out.push(py_to_arg_value(&item)?);
        }
        return Ok(ArgValue::Tuple(out));
    }
    Err(PyValueError::new_err(format!(
        "where_kwarg value must be a literal (None / bool / int / float / str / list / tuple); got {}",
        value.get_type().name()?,
    )))
}

/// True iff every ``(name, matcher)`` pair holds against the call's
/// ``kwargs``. A missing kwarg never matches.
pub(crate) fn call_args_match_kwargs(
    call_args: &CallArgs,
    matchers: &[(String, KwargMatcher)],
) -> bool {
    for (name, matcher) in matchers {
        let Some(av) = call_args.kwargs.get(name) else {
            return false;
        };
        match matcher {
            KwargMatcher::Literal(expected) => {
                if !arg_value_eq_literal(av, expected) {
                    return false;
                }
            }
        }
    }
    true
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
    let Expr::Attribute(attribute) = call.func.as_ref() else {
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
pub(crate) struct StringArgCallFinder<'a, F>
where
    F: FnMut(&ruff_python_ast::ExprCall) -> bool,
{
    pub(crate) predicate: F,
    pub(crate) arg_index: usize,
    pub(crate) file_imports: &'a HashMap<String, String>,
    pub(crate) decl_by_fqname: &'a HashMap<String, Vec<usize>>,
    pub(crate) results: Vec<(String, CallArgs)>,
}

impl<'ast, 'a, F> Visitor<'ast> for StringArgCallFinder<'a, F>
where
    F: FnMut(&ruff_python_ast::ExprCall) -> bool,
{
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if (self.predicate)(call) {
                if let Some(value) = nth_positional_string(call, self.arg_index) {
                    let call_args =
                        extract_call_args_kwargs(call, self.file_imports, self.decl_by_fqname);
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
    pub(crate) file_imports: &'a HashMap<String, String>,
    pub(crate) decl_by_fqname: &'a HashMap<String, Vec<usize>>,
    pub(crate) results: Vec<(String, CallArgs)>,
}

impl<'ast, 'a> Visitor<'ast> for AttrCallFinder<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Call(call) = expr {
            if let Expr::Attribute(attribute) = call.func.as_ref() {
                if attribute.attr.as_str() == self.attr {
                    if let Some(arg) = call.arguments.args.get(self.arg_index) {
                        let mut hits: Vec<String> = Vec::new();
                        match arg {
                            Expr::StringLiteral(s) => {
                                hits.push(s.value.to_str().to_string());
                            }
                            Expr::List(list) => {
                                for elt in &list.elts {
                                    if let Expr::StringLiteral(s) = elt {
                                        hits.push(s.value.to_str().to_string());
                                    }
                                }
                            }
                            Expr::Tuple(tup) => {
                                for elt in &tup.elts {
                                    if let Expr::StringLiteral(s) = elt {
                                        hits.push(s.value.to_str().to_string());
                                    }
                                }
                            }
                            _ => {}
                        }
                        if !hits.is_empty() {
                            let call_args = extract_call_args_kwargs(
                                call,
                                self.file_imports,
                                self.decl_by_fqname,
                            );
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

/// Per-file sorted list of `(target_start_offset, node_idx)` for the
/// file's top-level decls. Sorted ascending so callers can binary-search
/// for the next decl after a comment.
///
/// Reads straight from `global_index` — every key in there already
/// carries the decl's `(start, end)` range tuple, and `ingest_decls`
/// populated it from the same `all_definitions_with_usage()` walk.
pub(crate) fn file_decl_sites(file: File, global_index: &DeclIndex) -> Vec<(u32, usize)> {
    let mut out: Vec<(u32, usize)> = global_index
        .iter()
        .filter(|((f, _, _), _)| *f == file)
        .map(|((_, _, (start, _)), idx)| (*start, *idx))
        .collect();
    out.sort_by_key(|(start, _)| *start);
    out
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
pub(crate) type NameTable = HashMap<String, bool>;

pub(crate) fn detect_dead_ranges(parsed: &ParsedModuleRef) -> Vec<TextRange> {
    let mut dead = Vec::new();
    let empty: NameTable = HashMap::new();
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
pub(crate) fn build_scope_table(stmts: &[Stmt], enclosing: &NameTable) -> NameTable {
    let mut table = enclosing.clone();
    let mut bindings: HashMap<String, Vec<&Expr>> = HashMap::new();
    collect_scope_bindings(stmts, &mut bindings);

    let mut changed = true;
    while changed {
        changed = false;
        for (name, exprs) in &bindings {
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
                (Some(v), prev) if prev != Some(v) => {
                    table.insert(name.clone(), v);
                    changed = true;
                }
                (None, Some(_)) => {
                    // Previously folded, now conflicting / unknown:
                    // drop it. Don't fall back to the enclosing
                    // scope's value because a same-name binding in
                    // this scope shadows it.
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
    out: &mut HashMap<String, Vec<&'a Expr>>,
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
pub(crate) fn collect_walrus_in_expr<'a>(expr: &'a Expr, out: &mut HashMap<String, Vec<&'a Expr>>) {
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
) -> (bool, HashSet<usize>) {
    let mut file_pinned = false;
    let mut per_line_pins: HashSet<usize> = HashSet::new();
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

pub(crate) fn module_fqname_for_file(db: &dyn ProjectDb, file: File) -> String {
    if let Some(module) = file_to_module(db, file) {
        return module.name(db).as_str().to_string();
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
