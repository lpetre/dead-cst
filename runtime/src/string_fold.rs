//! Static folding of string-valued expressions for the plugin-query facts.
//!
//! Every string a plugin query hands back (`mock.patch` targets, fixture
//! `name=` kwargs, `__all__`-style literal lists, …) is captured at
//! extraction time by [`crate::file_extraction`]. This module decides what
//! counts as "a string we know statically": beyond a plain literal it
//! folds implicit concatenation, `+` of foldable strings, f-strings whose
//! interpolations are themselves foldable, the module's own `__name__`,
//! and top-level names bound exactly once to a foldable string in the
//! same file. Anything else stays unknown, exactly as before.
//!
//! The rules are deliberately conservative — a wrong fold would send a
//! plugin chasing a fake fqname, while a refused fold merely leaves today's
//! behaviour in place:
//!
//! * An interpolation folds only when it has no `=` debug text, no
//!   conversion other than `!s`, and no (or an empty) format spec.
//! * A top-level constant is used only when its name is bound exactly
//!   once at module level (assignment, def, class, import, loop target …
//!   all count as bindings) and no `global` statement anywhere in the
//!   file names it.
//! * Inside a `def` / `class`, any name bound anywhere in that subtree
//!   (parameters included) shadows the module constant and is never
//!   folded. The guard is purely syntactic and errs on the side of
//!   refusing.

use compact_str::{CompactString, ToCompactString};
use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
use ruff_python_ast::{
    ConversionFlag, Expr, FStringPart, InterpolatedElement, InterpolatedStringElement, Operator,
    Parameters, Stmt,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::helpers::top_level_assign_to_name;

/// Names a string fold may substitute. Built once per file (module
/// constants) and narrowed per `def` / `class` subtree (shadow set).
#[derive(Clone, Copy)]
pub(crate) struct StringFoldCtx<'a> {
    /// The value of `__name__` in this file.
    pub(crate) module_name: Option<&'a str>,
    /// Top-level names bound exactly once to a foldable string.
    pub(crate) constants: Option<&'a FxHashMap<CompactString, CompactString>>,
    /// Names bound in the enclosing `def` / `class` subtree — never folded.
    pub(crate) shadowed: Option<&'a FxHashSet<CompactString>>,
}

impl<'a> StringFoldCtx<'a> {
    /// Literal-only folding: no `__name__`, no constants.
    pub(crate) const LITERAL_ONLY: StringFoldCtx<'static> = StringFoldCtx {
        module_name: None,
        constants: None,
        shadowed: None,
    };

    /// The same context with `shadowed` applied.
    pub(crate) fn with_shadowed(self, shadowed: &'a FxHashSet<CompactString>) -> Self {
        StringFoldCtx {
            shadowed: Some(shadowed),
            ..self
        }
    }

    fn lookup(&self, name: &str) -> Option<&'a str> {
        if self.shadowed.is_some_and(|s| s.contains(name)) {
            return None;
        }
        if name == "__name__" {
            return self.module_name;
        }
        self.constants?.get(name).map(CompactString::as_str)
    }
}

/// The static string value of `expr`, or `None` when any part of it is
/// unknown.
pub(crate) fn fold_string_expr(expr: &Expr, ctx: &StringFoldCtx<'_>) -> Option<CompactString> {
    let mut out = CompactString::default();
    fold_into(expr, ctx, &mut out).then_some(out)
}

/// The static string values of a list / tuple whose every element folds,
/// or `None` (not a list / tuple, or any element unknown).
pub(crate) fn fold_string_list(expr: &Expr, ctx: &StringFoldCtx<'_>) -> Option<Vec<CompactString>> {
    let elements: &[Expr] = match expr {
        Expr::List(l) => &l.elts,
        Expr::Tuple(t) => &t.elts,
        _ => return None,
    };
    elements.iter().map(|e| fold_string_expr(e, ctx)).collect()
}

fn fold_into(expr: &Expr, ctx: &StringFoldCtx<'_>, out: &mut CompactString) -> bool {
    match expr {
        // `StringLiteralValue::to_str` already joins implicit concatenation.
        Expr::StringLiteral(s) => {
            out.push_str(s.value.to_str());
            true
        }
        Expr::FString(f) => {
            for part in f.value.iter() {
                match part {
                    FStringPart::Literal(s) => out.push_str(&s.value),
                    FStringPart::FString(fs) => {
                        for element in fs.elements.iter() {
                            match element {
                                InterpolatedStringElement::Literal(l) => out.push_str(&l.value),
                                InterpolatedStringElement::Interpolation(i) => {
                                    if !fold_interpolation(i, ctx, out) {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            true
        }
        Expr::BinOp(b) if b.op == Operator::Add => {
            fold_into(&b.left, ctx, out) && fold_into(&b.right, ctx, out)
        }
        Expr::Name(n) => match ctx.lookup(n.id.as_str()) {
            Some(value) => {
                out.push_str(value);
                true
            }
            None => false,
        },
        _ => false,
    }
}

fn fold_interpolation(
    element: &InterpolatedElement,
    ctx: &StringFoldCtx<'_>,
    out: &mut CompactString,
) -> bool {
    if element.debug_text.is_some() {
        return false;
    }
    if !matches!(
        element.conversion,
        ConversionFlag::None | ConversionFlag::Str
    ) {
        return false;
    }
    if element
        .format_spec
        .as_ref()
        .is_some_and(|spec| !spec.elements.is_empty())
    {
        return false;
    }
    fold_into(&element.expression, ctx, out)
}

/// Top-level names of `body` bound exactly once, to a string that folds
/// (given `module_name` as `__name__` and the other constants — a
/// constant may be built from constants declared before or after it).
pub(crate) fn top_level_string_constants(
    body: &[Stmt],
    module_name: Option<&str>,
) -> FxHashMap<CompactString, CompactString> {
    let mut binding_count: FxHashMap<&str, usize> = FxHashMap::default();
    for stmt in body {
        let mut binder = TopLevelBinder { names: Vec::new() };
        binder.visit_stmt(stmt);
        for name in binder.names {
            *binding_count.entry(name).or_insert(0) += 1;
        }
    }
    let mut globals = GlobalCollector {
        names: FxHashSet::default(),
    };
    for stmt in body {
        globals.visit_stmt(stmt);
    }

    let mut pending: Vec<(&str, &Expr)> = body
        .iter()
        .filter_map(top_level_assign_to_name)
        .filter_map(|(range, value)| {
            let name = name_at(body, range)?;
            (binding_count.get(name) == Some(&1) && !globals.names.contains(name))
                .then_some((name, value))
        })
        .collect();

    // Fixpoint: each pass folds every pending constant against what has
    // resolved so far, so `B = f"{A}.x"` resolves once `A` has, whichever
    // order they were declared in. A cycle simply never resolves.
    let mut resolved: FxHashMap<CompactString, CompactString> = FxHashMap::default();
    loop {
        let ctx = StringFoldCtx {
            module_name,
            constants: Some(&resolved),
            shadowed: None,
        };
        let mut newly: Vec<(CompactString, CompactString)> = Vec::new();
        let mut still_pending = Vec::with_capacity(pending.len());
        for (name, value) in pending {
            match fold_string_expr(value, &ctx) {
                Some(folded) => newly.push((name.to_compact_string(), folded)),
                None => still_pending.push((name, value)),
            }
        }
        pending = still_pending;
        let progressed = !newly.is_empty();
        resolved.extend(newly);
        if !progressed || pending.is_empty() {
            break;
        }
    }
    resolved
}

/// The identifier text of the single-`Name` assignment target whose range
/// is `range` (as returned by `top_level_assign_to_name`).
fn name_at(body: &[Stmt], range: ruff_text_size::TextRange) -> Option<&str> {
    use ruff_text_size::Ranged;
    for stmt in body {
        let target = match stmt {
            Stmt::Assign(a) => a.targets.first(),
            Stmt::AnnAssign(a) => Some(a.target.as_ref()),
            _ => None,
        };
        if let Some(Expr::Name(n)) = target {
            if n.range() == range {
                return Some(n.id.as_str());
            }
        }
    }
    None
}

/// Every name bound in a `def` / `class` subtree (parameters, assignment
/// and loop targets, nested defs, imports, `except … as`, walrus …). Used
/// as the shadow set for string folds inside that subtree.
pub(crate) fn scope_bound_names(stmt: &Stmt) -> FxHashSet<CompactString> {
    let mut collector = ScopeBinder {
        names: FxHashSet::default(),
    };
    collector.visit_stmt(stmt);
    collector.names
}

/// Names bound by one top-level statement, *not* descending into `def` /
/// `class` bodies (those bind their own scope).
struct TopLevelBinder<'a> {
    names: Vec<&'a str>,
}

impl<'a> Visitor<'a> for TopLevelBinder<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(f) => self.names.push(f.name.as_str()),
            Stmt::ClassDef(c) => self.names.push(c.name.as_str()),
            Stmt::Import(im) => self.names.extend(im.names.iter().map(alias_local)),
            Stmt::ImportFrom(im) => self.names.extend(im.names.iter().map(alias_local)),
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Name(n) if n.ctx.is_store() => self.names.push(n.id.as_str()),
            // A lambda's parameters bind its own scope.
            Expr::Lambda(_) => {}
            _ => walk_expr(self, expr),
        }
    }
}

struct GlobalCollector {
    names: FxHashSet<CompactString>,
}

impl<'a> Visitor<'a> for GlobalCollector {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if let Stmt::Global(g) = stmt {
            self.names
                .extend(g.names.iter().map(|n| n.as_str().to_compact_string()));
        }
        walk_stmt(self, stmt);
    }
}

struct ScopeBinder {
    names: FxHashSet<CompactString>,
}

impl ScopeBinder {
    fn bind_params(&mut self, params: &Parameters) {
        for p in params.iter() {
            self.names.insert(p.name().as_str().to_compact_string());
        }
    }
}

impl<'a> Visitor<'a> for ScopeBinder {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(f) => {
                self.names.insert(f.name.as_str().to_compact_string());
                self.bind_params(&f.parameters);
            }
            Stmt::ClassDef(c) => {
                self.names.insert(c.name.as_str().to_compact_string());
            }
            Stmt::Import(im) => self
                .names
                .extend(im.names.iter().map(|a| alias_local(a).to_compact_string())),
            Stmt::ImportFrom(im) => self
                .names
                .extend(im.names.iter().map(|a| alias_local(a).to_compact_string())),
            _ => {}
        }
        walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::Name(n) if n.ctx.is_store() => {
                self.names.insert(n.id.as_str().to_compact_string());
            }
            Expr::Lambda(l) => {
                if let Some(params) = &l.parameters {
                    self.bind_params(params);
                }
            }
            _ => {}
        }
        walk_expr(self, expr);
    }

    fn visit_except_handler(&mut self, handler: &'a ruff_python_ast::ExceptHandler) {
        let ruff_python_ast::ExceptHandler::ExceptHandler(h) = handler;
        if let Some(name) = &h.name {
            self.names.insert(name.as_str().to_compact_string());
        }
        ruff_python_ast::visitor::walk_except_handler(self, handler);
    }
}

fn alias_local(alias: &ruff_python_ast::Alias) -> &str {
    alias
        .asname
        .as_ref()
        .map(|n| n.as_str())
        .unwrap_or(alias.name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruff_python_parser::parse_module;

    fn parse(source: &str) -> Vec<Stmt> {
        parse_module(source)
            .expect("parse")
            .into_syntax()
            .body
            .into_iter()
            .collect()
    }

    fn first_expr(source: &str) -> Expr {
        match parse(source).remove(0) {
            Stmt::Expr(e) => *e.value,
            other => panic!("expected expression statement, got {other:?}"),
        }
    }

    fn fold(source: &str, ctx: &StringFoldCtx<'_>) -> Option<String> {
        fold_string_expr(&first_expr(source), ctx).map(|s| s.to_string())
    }

    fn consts(pairs: &[(&str, &str)]) -> FxHashMap<CompactString, CompactString> {
        pairs
            .iter()
            .map(|(k, v)| (CompactString::from(*k), CompactString::from(*v)))
            .collect()
    }

    #[test]
    fn folds_literal_forms() {
        let ctx = StringFoldCtx::LITERAL_ONLY;
        assert_eq!(fold("'a.b'", &ctx).as_deref(), Some("a.b"));
        assert_eq!(fold("'a.' 'b'", &ctx).as_deref(), Some("a.b"));
        assert_eq!(fold("'a.' + 'b'", &ctx).as_deref(), Some("a.b"));
        assert_eq!(fold("f'a.b'", &ctx).as_deref(), Some("a.b"));
        assert_eq!(fold("'a.' f'b'", &ctx).as_deref(), Some("a.b"));
        assert_eq!(fold("f'{\"a\"}.b'", &ctx).as_deref(), Some("a.b"));
    }

    #[test]
    fn refuses_unknown_forms() {
        let ctx = StringFoldCtx::LITERAL_ONLY;
        assert_eq!(fold("x", &ctx), None);
        assert_eq!(fold("f'{x}.b'", &ctx), None);
        assert_eq!(fold("'a' + x", &ctx), None);
        assert_eq!(fold("'a' * 2", &ctx), None);
        assert_eq!(fold("'a'.upper()", &ctx), None);
        assert_eq!(fold("42", &ctx), None);
    }

    #[test]
    fn folds_dunder_name_and_constants() {
        let constants = consts(&[("MOD", "pkg.lib")]);
        let ctx = StringFoldCtx {
            module_name: Some("tests.test_lib"),
            constants: Some(&constants),
            shadowed: None,
        };
        assert_eq!(
            fold("f'{__name__}.helper'", &ctx).as_deref(),
            Some("tests.test_lib.helper")
        );
        assert_eq!(
            fold("f'{MOD}.helper'", &ctx).as_deref(),
            Some("pkg.lib.helper")
        );
        assert_eq!(
            fold("MOD + '.helper'", &ctx).as_deref(),
            Some("pkg.lib.helper")
        );
        assert_eq!(
            fold("f'{MOD!s}.helper'", &ctx).as_deref(),
            Some("pkg.lib.helper")
        );
        assert_eq!(
            fold("f'{MOD:}.helper'", &ctx).as_deref(),
            Some("pkg.lib.helper")
        );
    }

    #[test]
    fn refuses_conversions_specs_and_debug_text() {
        let constants = consts(&[("MOD", "pkg.lib")]);
        let ctx = StringFoldCtx {
            module_name: None,
            constants: Some(&constants),
            shadowed: None,
        };
        assert_eq!(fold("f'{MOD!r}.helper'", &ctx), None);
        assert_eq!(fold("f'{MOD!a}.helper'", &ctx), None);
        assert_eq!(fold("f'{MOD:>10}.helper'", &ctx), None);
        assert_eq!(fold("f'{MOD=}.helper'", &ctx), None);
        assert_eq!(fold("f'{__name__}.helper'", &ctx), None);
    }

    #[test]
    fn shadowed_names_never_fold() {
        let constants = consts(&[("MOD", "pkg.lib")]);
        let shadowed: FxHashSet<CompactString> = ["MOD", "__name__"]
            .into_iter()
            .map(CompactString::from)
            .collect();
        let ctx = StringFoldCtx {
            module_name: Some("m"),
            constants: Some(&constants),
            shadowed: None,
        }
        .with_shadowed(&shadowed);
        assert_eq!(fold("f'{MOD}.helper'", &ctx), None);
        assert_eq!(fold("f'{__name__}.helper'", &ctx), None);
    }

    #[test]
    fn top_level_constants_single_binding_only() {
        let body = parse(
            "A = 'pkg'\nB = f'{A}.lib'\nC = B + '.x'\nD = 'once'\nD = 'twice'\n\
             E = 'e'\ndef f():\n    global E\n    E = 'z'\nF = g()\nfor G in x: pass\n\
             H = 'h'\nimport H\n",
        );
        let got = top_level_string_constants(&body, Some("m"));
        assert_eq!(got.get("A").map(|s| s.as_str()), Some("pkg"));
        assert_eq!(got.get("B").map(|s| s.as_str()), Some("pkg.lib"));
        assert_eq!(got.get("C").map(|s| s.as_str()), Some("pkg.lib.x"));
        assert!(!got.contains_key("D"), "reassigned");
        assert!(!got.contains_key("E"), "declared global in a function");
        assert!(!got.contains_key("F"), "not a string");
        assert!(!got.contains_key("G"), "loop target");
        assert!(!got.contains_key("H"), "also bound by an import");
    }

    #[test]
    fn top_level_constants_resolve_regardless_of_order_and_stop_on_cycles() {
        let body =
            parse("B = f'{A}.lib'\nA = 'pkg'\nX = f'{Y}'\nY = f'{X}'\nN = f'{__name__}.z'\n");
        let got = top_level_string_constants(&body, Some("mod"));
        assert_eq!(got.get("B").map(|s| s.as_str()), Some("pkg.lib"));
        assert!(!got.contains_key("X"));
        assert!(!got.contains_key("Y"));
        assert_eq!(got.get("N").map(|s| s.as_str()), Some("mod.z"));
    }

    #[test]
    fn scope_bound_names_cover_params_targets_and_nested_bindings() {
        let body = parse(
            "def f(a, /, b, *args, c=1, **kw):\n    d = 1\n    for e in x: pass\n\
             \n    with y as g: pass\n    try: pass\n    except E as h: pass\n\
             \n    import i\n    from j import k as l\n    def m(n): pass\n\
             \n    o = lambda p: p\n    [q for q in x]\n    (r := 3)\n",
        );
        let got = scope_bound_names(&body[0]);
        for name in [
            "f", "a", "b", "args", "c", "kw", "d", "e", "g", "h", "i", "l", "m", "n", "o", "p",
            "q", "r",
        ] {
            assert!(got.contains(name), "{name} should be bound");
        }
        assert!(!got.contains("x"));
        assert!(!got.contains("E"));
    }
}
