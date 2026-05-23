//! Helpers shared across the salsa-tracked per-file graph queries.
//!
//! After the fan-out refactor, this module's role is just a grab-bag
//! of pure-function helpers the per-file salsa-tracked queries in
//! `file_payload.rs` and `file_ref_edges.rs` call into:
//!
//! * `decl_kind_str`, `from_module_string` — small classifiers used
//!   when building per-file node payloads.
//! * `stmt_creates_top_level_definition`, `target_is_dunder_all`,
//!   `paired_unpack_rhs`, `collapse_attribute_chain` — AST shape
//!   helpers for the per-file ref walk.
//! * `detect_dynamic_call` / `parse_dynamic_args` /
//!   `resolve_dynamic_target` + the small literal-extraction helpers
//!   — dynamic-import (`__import__` / `importlib.import_module`)
//!   detection.
//! * `file_package_name`, `module_name_resolves`,
//!   `emit_visitor_warning` — misc per-file utilities.
//! * `build_dist_lookup` / `site_packages_roots` /
//!   `pep503_canonicalize` / `DistLookup` — driven by
//!   `file_payload::project_dist_lookup` for `[external dist] X`
//!   classification.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use ruff_db::files::{File, FilePath};
use ruff_python_ast::{Expr, ExprName, Stmt};
use ruff_text_size::Ranged;
use ty_module_resolver::{resolve_module, search_paths, ModuleName, ModuleResolveMode};
use ty_python_core::definition::DefinitionKind;

pub(crate) fn decl_kind_str(kind: &DefinitionKind<'_>) -> Option<&'static str> {
    if kind.is_import() {
        return Some("import");
    }
    Some(match kind {
        DefinitionKind::Function(_) => "function",
        DefinitionKind::Class(_) => "class",
        DefinitionKind::TypeAlias(_) => "type_alias",
        // Walrus's bound name (PEP 572) leaks to the enclosing scope
        // — at module level that's a top-level decl. The other
        // binding-statement targets (`for`, `with ... as`, `except as`,
        // structural-`match` captures) are local loop / context /
        // pattern bindings that the libcst pipeline does *not* model
        // as top-level nodes; skip them so we don't mint phantom
        // module-scope decls.
        DefinitionKind::Assignment(_)
        | DefinitionKind::AnnotatedAssignment(_)
        | DefinitionKind::NamedExpression(_) => "variable",
        _ => return None,
    })
}

/// Resolved absolute module name for a `from ... import ...` clause.
///
/// Returns the empty string when ty's `from_import_statement` fails to
/// resolve (invalid syntax or too many leading dots) — downstream
/// classification can treat that as an unresolved target.
pub(crate) fn from_module_string(
    db: &dyn ty_python_semantic::Db,
    file: File,
    stmt: &ruff_python_ast::StmtImportFrom,
) -> String {
    ModuleName::from_import_statement(db, file, stmt)
        .map(|n| n.as_str().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Phase 2: module-hierarchy + cross-file alias edges
// ---------------------------------------------------------------------------

/// ``abs_file_path -> PEP 503-canonical dist name``. Populated once
/// per ``materialize`` call by walking every ``*.dist-info/`` directory
/// under each ``is_site_packages()`` search path that ty knows about
/// and indexing the files listed in the distribution's ``RECORD``.
///
/// Mirrors ``dead_cst.resolvers._imports.distribution_lookup`` from
/// the libcst pipeline but reads dist-info directly off the
/// filesystem — no Python callback, so the same lookup works in
/// pyo3 contexts where issuing ``importlib.metadata.distributions()``
/// would round-trip through the runtime.
pub(crate) type DistLookup = HashMap<PathBuf, String>;

/// PEP 503 normalization. Replaces every run of ``[-_.]`` with a
/// single ``-`` and lowercases the rest. Equivalent to Python's
/// ``re.sub(r"[-_.]+", "-", name).lower()`` from libcst's
/// ``_canonical_dist_name``.
pub(crate) fn pep503_canonicalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for ch in name.chars() {
        if matches!(ch, '-' | '_' | '.') {
            if !prev_sep {
                out.push('-');
                prev_sep = true;
            }
        } else {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        }
    }
    out
}

/// Site-packages roots ty's resolver is configured with, canonicalised
/// upfront so the worker thread that runs ``build_dist_lookup`` doesn't
/// need to borrow ``db`` (Salsa's ``ProjectDatabase`` is !Sync).
pub(crate) fn site_packages_roots(db: &dyn ty_python_semantic::Db) -> Vec<PathBuf> {
    search_paths(db, ModuleResolveMode::StubsAllowed)
        .filter(|sp| sp.is_site_packages())
        .filter_map(|sp| sp.as_system_path().map(|p| p.as_str()))
        .map(|s| std::fs::canonicalize(s).unwrap_or_else(|_| PathBuf::from(s)))
        .collect()
}

/// Build the dist-file lookup by walking ``*.dist-info/`` under every
/// site-packages search path.
///
/// For each dist-info directory:
///
/// 1. Read ``METADATA`` for the ``Name:`` header (the project's PyPI
///    name pre-canonicalization). Stop at the blank line that
///    separates headers from the long description.
/// 2. Apply PEP 503 normalization so casing / separators don't matter
///    at lookup time.
/// 3. Read ``RECORD`` for the comma-separated ``path,hash,size``
///    entries, take the path (relative to the site-packages dir),
///    join + canonicalize against the dist-info parent, and map each
///    resolved absolute path to the canonical name.
///
/// Failures (missing METADATA, malformed RECORD, unreadable site-
/// packages) are silently skipped — the file just won't appear in
/// the lookup and the caller falls through to ``[external file]``.
///
/// Per-dist work is dispatched across a rayon pool: a typical venv
/// has dozens of dists and the per-RECORD ``canonicalize`` syscalls
/// (~2.5k in dead-cst's own dev venv) dominate everything else.
pub(crate) fn build_dist_lookup(site_packages: &[PathBuf]) -> DistLookup {
    use rayon::prelude::*;

    site_packages
        .iter()
        .flat_map(|sp_root| {
            std::fs::read_dir(sp_root)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "dist-info"))
                .map(move |dist_info| (dist_info, sp_root.as_path()))
        })
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|(dist_info, sp_root)| process_dist_info(dist_info, sp_root))
        .flatten()
        .collect()
}

/// One ``*.dist-info/`` directory's ``(absolute file path, canonical
/// dist name)`` pairs. ``None`` when METADATA / RECORD are missing or
/// the ``Name:`` header isn't found.
fn process_dist_info(dist_info: &Path, sp_root: &Path) -> Option<Vec<(PathBuf, String)>> {
    let metadata = std::fs::read_to_string(dist_info.join("METADATA")).ok()?;
    let canonical = metadata
        .lines()
        .take_while(|l| !l.is_empty())
        .find_map(|line| {
            line.strip_prefix("Name:")
                .map(|v| pep503_canonicalize(v.trim()))
        })?;
    let record = std::fs::read_to_string(dist_info.join("RECORD")).ok()?;
    Some(
        record
            .lines()
            .filter_map(|line| {
                let (rel, _) = line.split_once(',')?;
                let joined = sp_root.join(rel);
                let abs = std::fs::canonicalize(&joined).unwrap_or(joined);
                Some((abs, canonical.clone()))
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Phase 3: same-file Name→decl reference edges
// ---------------------------------------------------------------------------

/// True iff this top-level statement is a binding form whose Names
/// have already been attributed by the per-definition walk in (a).
///
/// Compound non-scope statements (``if`` / ``while`` / ``for`` / ...)
/// return ``false`` here: their *bodies* contain definitions that (a)
/// covers, but their *test/iter/etc. expressions* belong to the module
/// and (b) needs to walk them.
pub(crate) fn stmt_creates_top_level_definition(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::FunctionDef(_)
            | Stmt::ClassDef(_)
            | Stmt::Assign(_)
            | Stmt::AnnAssign(_)
            | Stmt::TypeAlias(_)
            | Stmt::Import(_)
            | Stmt::ImportFrom(_)
    )
}

/// Pair a tuple-unpack target with the RHS element at the same
/// position, so each LHS name only collects edges from the value it
/// actually receives.
///
/// Returns the paired RHS element when:
/// * the full LHS (the parent unpack target) is a flat `Tuple` /
///   `List` literal with no starred element, and
/// * the RHS is a flat `Tuple` / `List` literal with the same arity,
///   and
/// * the bare `target` name appears as one of the LHS elements.
///
/// Returns `None` otherwise so the caller falls back to walking the
/// whole RHS (the safe over-approximation for non-literal RHS values
/// like `c, d = f()` and starred-target patterns).
pub(crate) fn paired_unpack_rhs<'ast>(
    lhs: &Expr,
    target: &Expr,
    rhs: &'ast Expr,
) -> Option<&'ast Expr> {
    let lhs_elts = match lhs {
        Expr::Tuple(t) => &t.elts,
        Expr::List(l) => &l.elts,
        _ => return None,
    };
    let rhs_elts = match rhs {
        Expr::Tuple(t) => &t.elts,
        Expr::List(l) => &l.elts,
        _ => return None,
    };
    if lhs_elts.len() != rhs_elts.len() {
        return None;
    }
    if lhs_elts.iter().any(|e| matches!(e, Expr::Starred(_))) {
        return None;
    }
    let target_range = target.range();
    let pos = lhs_elts.iter().position(|e| e.range() == target_range)?;
    Some(&rhs_elts[pos])
}

/// True iff `target` is the bare `Name("__all__")` LHS of an
/// assignment (`__all__ = ...` or `__all__: list[str] = ...`).
///
/// Subscript / attribute / tuple-unpack targets (e.g.
/// `__all__[0] = "f"`, `mod.__all__ = [...]`) are excluded — those
/// don't redefine the binding in a way the libcst pipeline picks up,
/// and treating them as `__all__` would emit edges from unrelated
/// nodes.
pub(crate) fn target_is_dunder_all(target: &Expr) -> bool {
    matches!(target, Expr::Name(n) if n.id.as_str() == "__all__")
}

// ---------------------------------------------------------------------------
// Reference collector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) enum DynamicKind {
    DunderImport,
    ImportlibImportModule,
}

impl DynamicKind {
    fn label(self) -> &'static str {
        match self {
            DynamicKind::DunderImport => "__import__(...)",
            DynamicKind::ImportlibImportModule => "importlib.import_module(...)",
        }
    }
}

/// Match `call.func` against the two dynamic-import call shapes we
/// recognize: a bare `Name("__import__")` or
/// `Attribute(Name("importlib"), "import_module")`. The `importlib`
/// receiver is matched textually — a local `class importlib` would
/// be a false positive, but that pattern is vanishingly rare in
/// practice and the libcst pipeline does the same syntactic match.
pub(crate) fn detect_dynamic_call(func: &Expr) -> Option<DynamicKind> {
    match func {
        Expr::Name(n) if n.id.as_str() == "__import__" => Some(DynamicKind::DunderImport),
        Expr::Attribute(attr) => {
            if let Expr::Name(receiver) = &*attr.value {
                if receiver.id.as_str() == "importlib" && attr.attr.as_str() == "import_module" {
                    return Some(DynamicKind::ImportlibImportModule);
                }
            }
            None
        }
        _ => None,
    }
}

pub(crate) enum DynamicParseResult<'a> {
    /// Successfully parsed.
    Ok {
        name: String,
        fromlist: Vec<&'a str>,
        explicit_package: Option<&'a str>,
        explicit_level: Option<u32>,
    },
    /// Recognized as a dynamic import but rejected with this warning
    /// message. The visitor still skips the call's argument walk —
    /// the call site itself doesn't carry normal name references
    /// once we've identified it as a dynamic import.
    Warn(String),
    /// Not a dynamic import (e.g. the first argument is missing
    /// entirely, so it isn't really one of these call shapes). Caller
    /// should fall through to a normal expression walk.
    NotApplicable,
}

pub(crate) fn parse_dynamic_args<'a>(
    kind: DynamicKind,
    call: &'a ruff_python_ast::ExprCall,
) -> DynamicParseResult<'a> {
    let Some(first_arg) = first_positional(call) else {
        return DynamicParseResult::NotApplicable;
    };
    let Some(name) = string_literal(first_arg) else {
        return DynamicParseResult::Warn(format!(
            "Skipping dynamic import '{}': name is not a string literal",
            kind.label()
        ));
    };

    match kind {
        DynamicKind::DunderImport => {
            // `__import__(name, globals=None, locals=None, fromlist=[], level=0)`
            let fromlist_expr = positional_or_kwarg(call, 3, "fromlist");
            let level_expr = positional_or_kwarg(call, 4, "level");
            let fromlist = match fromlist_expr {
                None => Vec::new(),
                Some(expr) => match string_literal_list(expr) {
                    Some(list) => list,
                    None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': fromlist is not a literal \
                             list/tuple of strings (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            let explicit_level = match level_expr {
                None => None,
                Some(expr) => match int_literal(expr) {
                    Some(n) if n >= 0 => Some(n as u32),
                    Some(_) | None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': level is not an int literal \
                             (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            DynamicParseResult::Ok {
                name: name.to_string(),
                fromlist,
                explicit_package: None,
                explicit_level,
            }
        }
        DynamicKind::ImportlibImportModule => {
            // `importlib.import_module(name, package=None)`
            let pkg_expr = positional_or_kwarg(call, 1, "package");
            let explicit_package = match pkg_expr {
                None => None,
                Some(expr) => match string_literal(expr) {
                    Some(s) => Some(s),
                    None => {
                        return DynamicParseResult::Warn(format!(
                            "Skipping dynamic import '{}': package is not a string literal \
                             (name: '{name}')",
                            kind.label(),
                        ));
                    }
                },
            };
            DynamicParseResult::Ok {
                name: name.to_string(),
                fromlist: Vec::new(),
                explicit_package,
                explicit_level: None,
            }
        }
    }
}

pub(crate) fn first_positional(call: &ruff_python_ast::ExprCall) -> Option<&Expr> {
    call.arguments.args.first()
}

pub(crate) fn positional_or_kwarg<'a>(
    call: &'a ruff_python_ast::ExprCall,
    idx: usize,
    kwarg_name: &str,
) -> Option<&'a Expr> {
    if let Some(expr) = call.arguments.args.get(idx) {
        return Some(expr);
    }
    call.arguments
        .keywords
        .iter()
        .find(|k| k.arg.as_ref().is_some_and(|id| id.as_str() == kwarg_name))
        .map(|k| &k.value)
}

pub(crate) fn string_literal(expr: &Expr) -> Option<&str> {
    if let Expr::StringLiteral(s) = expr {
        Some(s.value.to_str())
    } else {
        None
    }
}

pub(crate) fn int_literal(expr: &Expr) -> Option<i64> {
    let Expr::NumberLiteral(n) = expr else {
        return None;
    };
    if let ruff_python_ast::Number::Int(i) = &n.value {
        i.as_i64()
    } else {
        None
    }
}

pub(crate) fn string_literal_list(expr: &Expr) -> Option<Vec<&str>> {
    let elements: &[Expr] = match expr {
        Expr::List(l) => &l.elts,
        Expr::Tuple(t) => &t.elts,
        _ => return None,
    };
    let mut out = Vec::with_capacity(elements.len());
    for elem in elements {
        out.push(string_literal(elem)?);
    }
    Some(out)
}

/// Resolve a dynamic-import `name` against its level / package
/// context to an absolute module name, or return an error message
/// suitable for logging.
///
/// Per CPython semantics:
/// * `__import__`: leading dots in `name` are invalid (level is the
///   `level=` keyword/positional arg); `level=N` means "start from
///   the current package and go up N-1 levels."
/// * `importlib.import_module`: leading dots in `name` are the level
///   (count them); `level=1` means current package, `level=2` parent,
///   etc. `package=` is optional and defaults to the caller's
///   package — that's what `pkg` is here.
pub(crate) fn resolve_dynamic_target(
    kind: DynamicKind,
    name: &str,
    explicit_level: Option<u32>,
    pkg: Option<&str>,
) -> Result<String, String> {
    let (name_no_dots, level): (&str, u32) = match kind {
        DynamicKind::DunderImport => {
            if name.starts_with('.') {
                return Err(format!(
                    "Skipping dynamic import '{}': leading dots are invalid for \
                     __import__ (name: '{name}')",
                    kind.label()
                ));
            }
            (name, explicit_level.unwrap_or(0))
        }
        DynamicKind::ImportlibImportModule => {
            let dots = name.chars().take_while(|c| *c == '.').count();
            (&name[dots..], dots as u32)
        }
    };

    if level == 0 {
        return Ok(name_no_dots.to_string());
    }
    let Some(pkg) = pkg else {
        return Err(format!(
            "Skipping dynamic import '{}': relative import '{name}' has no package context",
            kind.label()
        ));
    };
    let segments: Vec<&str> = pkg.split('.').filter(|s| !s.is_empty()).collect();
    let levels_up = (level - 1) as usize;
    if levels_up > segments.len() {
        return Err(format!(
            "Skipping dynamic import '{}': relative import '{name}' goes beyond top-level \
             package '{pkg}'",
            kind.label()
        ));
    }
    let base = &segments[..segments.len() - levels_up];
    let mut parts: Vec<&str> = base.to_vec();
    if !name_no_dots.is_empty() {
        parts.push(name_no_dots);
    }
    Ok(parts.join("."))
}

/// Package name (i.e. enclosing package) of `file`. For
/// `pkg/__init__.py` this is `"pkg"`; for `pkg/sub.py` this is
/// `"pkg"`; for a top-level `mod.py` this is `None`.
pub(crate) fn file_package_name(db: &dyn ty_python_semantic::Db, file: File) -> Option<String> {
    let module = crate::helpers::canonical_module_for_file(db, file)?;
    let name = module.name(db);
    let path_str = match file.path(db) {
        FilePath::System(p) => p.to_string(),
        FilePath::SystemVirtual(p) => p.to_string(),
        FilePath::Vendored(p) => p.to_string(),
    };
    let is_init = path_str.ends_with("/__init__.py") || path_str.ends_with("\\__init__.py");
    if is_init {
        Some(name.as_str().to_string())
    } else {
        name.parent().map(|n| n.as_str().to_string())
    }
}

/// Send a warning to the `dead_cst._visitor` logger so the
/// `visitor_warnings` test fixture (pytest caplog scoped to that
/// logger) can observe it.
pub(crate) fn emit_visitor_warning(py: Python<'_>, message: &str) {
    let _ = (|| -> PyResult<()> {
        let logging = py.import_bound("logging")?;
        let logger = logging.call_method1("getLogger", ("dead_cst._visitor",))?;
        logger.call_method1("warning", (message,))?;
        Ok(())
    })();
}

/// Peel an attribute chain back to its `Name` root.
///
/// Returns `Some((root, segments))` when `expr` is `Name`, `Name.s1`,
/// `Name.s1.s2`, ... — i.e. one or more attribute accesses on a bare
/// name. Returns `None` when the chain bottoms out at anything else
/// (a call result, subscript, attribute of attribute of a non-Name, …).
pub(crate) fn collapse_attribute_chain(expr: &Expr) -> Option<(&ExprName, Vec<&str>)> {
    let mut segments: Vec<&str> = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Attribute(attr) => {
                segments.push(attr.attr.as_str());
                current = &attr.value;
            }
            Expr::Name(n) => {
                segments.reverse();
                return Some((n, segments));
            }
            _ => return None,
        }
    }
}

/// True iff `dotted` resolves to *some* module (project, stdlib, or
/// third-party) as seen from `anchor`. Used to disambiguate
/// "submodule" vs "decl in module" for `from X import Y`.
pub(crate) fn module_name_resolves(
    dotted: &str,
    anchor: File,
    db: &dyn ty_python_semantic::Db,
) -> bool {
    ModuleName::new(dotted)
        .and_then(|n| resolve_module(db, anchor, &n))
        .is_some()
}
