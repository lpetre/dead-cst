//! Experimental ruff-backed parser for dead-cst.
//!
//! Exposes one function today, `extract_top_level_decls(source)`, that
//! parses a Python module via `ruff_python_parser` and returns the
//! top-level def / async def / class / assignment-target names with
//! source positions. The shape mirrors what `SymbolVisitor` produces
//! for the same constructs, minus flow/shadowing/decorators/imports.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ruff_python_ast::{Expr, Stmt};
use ruff_source_file::LineIndex;
use ruff_text_size::TextRange;

/// One top-level declaration with libcst-compatible position info.
///
/// `line` is 1-based, `column` is 0-based (matches libcst's
/// `CodePosition`). `kind` is one of `"function"`, `"class"`,
/// `"variable"`.
#[pyclass(get_all, frozen)]
struct Decl {
    name: String,
    kind: &'static str,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

#[pymethods]
impl Decl {
    fn __repr__(&self) -> String {
        format!(
            "Decl(name={:?}, kind={:?}, start=({}, {}), end=({}, {}))",
            self.name,
            self.kind,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column,
        )
    }
}

fn position(index: &LineIndex, source: &str, range: TextRange) -> (usize, usize, usize, usize) {
    let start = index.line_column(range.start(), source);
    let end = index.line_column(range.end(), source);
    // ruff's `column` is 1-based; libcst's is 0-based. Subtract one.
    (
        start.line.get(),
        start.column.get() - 1,
        end.line.get(),
        end.column.get() - 1,
    )
}

fn collect_assign_targets(target: &Expr, out: &mut Vec<(String, TextRange)>) {
    match target {
        Expr::Name(n) => out.push((n.id.to_string(), n.range)),
        Expr::Tuple(t) => {
            for e in &t.elts {
                collect_assign_targets(e, out);
            }
        }
        Expr::List(l) => {
            for e in &l.elts {
                collect_assign_targets(e, out);
            }
        }
        _ => {}
    }
}

/// Parse `source` and return its top-level declarations.
#[pyfunction]
fn extract_top_level_decls(source: &str) -> PyResult<Vec<Decl>> {
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| PyValueError::new_err(format!("parse error: {e}")))?;
    let index = LineIndex::from_source_text(source);
    let mut decls: Vec<Decl> = Vec::new();

    for stmt in &parsed.syntax().body {
        match stmt {
            Stmt::FunctionDef(f) => {
                let (sl, sc, el, ec) = position(&index, source, f.range);
                decls.push(Decl {
                    name: f.name.to_string(),
                    kind: "function",
                    start_line: sl,
                    start_column: sc,
                    end_line: el,
                    end_column: ec,
                });
            }
            Stmt::ClassDef(c) => {
                let (sl, sc, el, ec) = position(&index, source, c.range);
                decls.push(Decl {
                    name: c.name.to_string(),
                    kind: "class",
                    start_line: sl,
                    start_column: sc,
                    end_line: el,
                    end_column: ec,
                });
            }
            Stmt::Assign(a) => {
                let mut targets = Vec::new();
                for t in &a.targets {
                    collect_assign_targets(t, &mut targets);
                }
                for (name, range) in targets {
                    let (sl, sc, el, ec) = position(&index, source, range);
                    decls.push(Decl {
                        name,
                        kind: "variable",
                        start_line: sl,
                        start_column: sc,
                        end_line: el,
                        end_column: ec,
                    });
                }
            }
            Stmt::AnnAssign(a) => {
                if let Expr::Name(n) = a.target.as_ref() {
                    let (sl, sc, el, ec) = position(&index, source, n.range);
                    decls.push(Decl {
                        name: n.id.to_string(),
                        kind: "variable",
                        start_line: sl,
                        start_column: sc,
                        end_line: el,
                        end_column: ec,
                    });
                }
            }
            _ => {}
        }
    }

    Ok(decls)
}

#[pymodule]
fn dead_cst_ty_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Decl>()?;
    m.add_function(wrap_pyfunction!(extract_top_level_decls, m)?)?;
    Ok(())
}
