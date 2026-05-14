//! Experimental ty-backed native parser for dead-cst.
//!
//! Exposes a `Project` class that owns a ty `ProjectDatabase`. Unlike
//! `ty check`, this prototype does NOT discover a `pyproject.toml`:
//! the resolver -- whatever produces dead-cst's `Package` tuples
//! today -- is responsible for feeding the project layout in
//! directly. This mirrors dead-cst's `Analysis(project_root,
//! resolver=...)` shape: callers retain control of which paths are
//! first-party, where the venv lives, and which Python version to
//! target.
//!
//! One query today, `extract_top_level_decls(path)`. The Salsa db is
//! re-used across calls so warm queries are free.

use std::str::FromStr;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use ruff_db::files::system_path_to_file;
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use ruff_db::system::{OsSystem, SystemPath, SystemPathBuf};
use ruff_python_ast::{Expr, Stmt};
use ruff_source_file::LineIndex;
use ruff_text_size::TextRange;
use ty_project::metadata::options::{EnvironmentOptions, Options};
use ty_project::metadata::python_version::SupportedPythonVersion;
use ty_project::metadata::value::{RangedValue, RelativePathBuf};
use ty_project::{ProjectDatabase, ProjectMetadata};
use ty_python_core::program::UseDefaultStrategy;

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

/// A ty-backed analysis project with explicitly-injected configuration.
///
/// `root` is the directory all relative paths are resolved against.
/// `src_roots` lists first-party search roots in priority order
/// (analogous to the union of `Package.path` values produced by a
/// `PathResolver`). `extra_paths` covers non-package roots that still
/// need to be importable (vendored bundles, etc.). `python_env`
/// points at a venv / `sys.prefix` so site-packages becomes
/// available. `python_version` is the language version ty assumes
/// when targeting `sys.version_info` conditionals. `typeshed`
/// overrides the bundled stdlib stubs.
///
/// No on-disk config (`pyproject.toml`, `ty.toml`) is read.
#[pyclass(unsendable)]
struct Project {
    db: ProjectDatabase,
}

#[pymethods]
impl Project {
    #[new]
    #[pyo3(signature = (
        root,
        *,
        src_roots = None,
        extra_paths = None,
        python_env = None,
        python_version = None,
        typeshed = None,
    ))]
    fn new(
        root: &str,
        src_roots: Option<Vec<String>>,
        extra_paths: Option<Vec<String>>,
        python_env: Option<&str>,
        python_version: Option<&str>,
        typeshed: Option<&str>,
    ) -> PyResult<Self> {
        let root = SystemPathBuf::from(root);

        let env = EnvironmentOptions {
            root: src_roots.map(|paths| paths.into_iter().map(rel_path).collect()),
            extra_paths: extra_paths.map(|paths| paths.into_iter().map(rel_path).collect()),
            python: python_env.map(rel_path),
            python_version: python_version
                .map(|v| {
                    SupportedPythonVersion::from_str(v).map_err(|e| {
                        PyValueError::new_err(format!("invalid python_version {v:?}: {e}"))
                    })
                })
                .transpose()?
                .map(RangedValue::cli),
            typeshed: typeshed.map(rel_path),
            ..EnvironmentOptions::default()
        };

        let options = Options {
            environment: Some(env),
            ..Options::default()
        };

        let metadata = ProjectMetadata::from_options(options, root, None, &UseDefaultStrategy)
            .map_err(|e| {
                PyValueError::new_err(format!("invalid configuration: {e:?}"))
            })?;

        let cwd = std::env::current_dir()
            .map_err(|e| PyOSError::new_err(format!("cwd unavailable: {e}")))?;
        let cwd = SystemPathBuf::from_path_buf(cwd.try_into().map_err(|_| {
            PyValueError::new_err("current working directory is not valid UTF-8")
        })?)
        .map_err(|_| PyValueError::new_err("current working directory is not absolute"))?;
        let system = OsSystem::new(cwd);

        let db = ProjectDatabase::use_defaults(metadata, system);
        Ok(Self { db })
    }

    /// Return the top-level declarations of the given file.
    ///
    /// `path` may be absolute or relative to the project root.
    fn extract_top_level_decls(&self, path: &str) -> PyResult<Vec<Decl>> {
        let path = SystemPath::new(path);
        let file = system_path_to_file(&self.db, path)
            .map_err(|e| PyOSError::new_err(format!("cannot open {path}: {e:?}")))?;
        let parsed = parsed_module(&self.db, file).load(&self.db);
        let source = source_text(&self.db, file);
        let index = LineIndex::from_source_text(&source);
        let mut decls: Vec<Decl> = Vec::new();
        for stmt in &parsed.syntax().body {
            collect_top_level(stmt, &source, &index, &mut decls);
        }
        Ok(decls)
    }
}

fn rel_path<P: AsRef<str>>(path: P) -> RelativePathBuf {
    RelativePathBuf::cli(SystemPath::new(path.as_ref()))
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

fn push_decl(
    name: String,
    kind: &'static str,
    range: TextRange,
    source: &str,
    index: &LineIndex,
    out: &mut Vec<Decl>,
) {
    let (sl, sc, el, ec) = position(index, source, range);
    out.push(Decl {
        name,
        kind,
        start_line: sl,
        start_column: sc,
        end_line: el,
        end_column: ec,
    });
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

fn collect_top_level(stmt: &Stmt, source: &str, index: &LineIndex, out: &mut Vec<Decl>) {
    match stmt {
        Stmt::FunctionDef(f) => {
            push_decl(f.name.to_string(), "function", f.range, source, index, out);
        }
        Stmt::ClassDef(c) => {
            push_decl(c.name.to_string(), "class", c.range, source, index, out);
        }
        Stmt::Assign(a) => {
            let mut targets = Vec::new();
            for t in &a.targets {
                collect_assign_targets(t, &mut targets);
            }
            for (name, range) in targets {
                push_decl(name, "variable", range, source, index, out);
            }
        }
        Stmt::AnnAssign(a) => {
            if let Expr::Name(n) = a.target.as_ref() {
                push_decl(n.id.to_string(), "variable", n.range, source, index, out);
            }
        }
        _ => {}
    }
}

#[pymodule]
fn dead_cst_ty_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Decl>()?;
    m.add_class::<Project>()?;
    Ok(())
}
