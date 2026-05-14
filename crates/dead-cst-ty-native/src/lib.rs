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
//! Two queries today:
//!   * `extract_top_level_decls(path)` returns a flat list of decls
//!     (development/debug shape).
//!   * `build_file_graph(path, module_fqname)` returns a
//!     `NativeGraph` envelope (unique nodes + unique edge triples) that
//!     Python materializes into a `SymbolGraph` (rustworkx PyDiGraph +
//!     SymbolNode ↔ int map). The boundary crosses once; rustworkx
//!     construction stays Python-side because rustworkx is itself a
//!     pyo3 type and per-node FFI hops would defeat the speed win.

use std::collections::{HashMap, HashSet};
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

/// Top-level declaration with libcst-compatible position info.
///
/// `line` is 1-based, `column` is 0-based. `kind` is one of
/// `"function"`, `"class"`, `"variable"`. Used by the
/// `extract_top_level_decls` debug query.
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

/// A single node in a `NativeGraph`.
///
/// Mirrors the primitive fields of `dead_cst.graph.SymbolNode`.
/// `fqname` is the fully-qualified dotted name (composed in Rust
/// from the caller-supplied `module_fqname` plus the source-level
/// local name); `path` is the absolute filesystem path the node
/// belongs to. Identity for deduplication is
/// `(fqname, kind, path, position, flags)` -- matching how
/// `SymbolNode.__hash__` works in Python.
#[pyclass(get_all, frozen)]
struct NativeNode {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    flags: u32,
}

#[pymethods]
impl NativeNode {
    fn __repr__(&self) -> String {
        format!(
            "NativeNode(fqname={:?}, kind={:?}, path={:?}, start=({}, {}), end=({}, {}), flags={})",
            self.fqname,
            self.kind,
            self.path,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column,
            self.flags,
        )
    }
}

/// One file's worth of graph contribution, packed for one FFI hop.
///
/// `nodes` is the unique node list in insertion order; `nodes[0]` is
/// always the synthetic module node and later entries are top-level
/// decls in source order. `edges` is the unique
/// `(src_idx, dst_idx, flags)` triple list, also in insertion order.
/// Both lists are deduplicated at build time -- callers (and the
/// Python materializer) get to assume uniqueness without re-checking.
///
/// `file_path` and `module_fqname` are echoed back so the Python
/// materializer doesn't need to re-derive them.
#[pyclass(get_all, frozen)]
struct NativeGraph {
    file_path: String,
    module_fqname: String,
    nodes: Vec<Py<NativeNode>>,
    edges: Vec<(usize, usize, u32)>,
}

#[pymethods]
impl NativeGraph {
    fn __repr__(&self) -> String {
        format!(
            "NativeGraph(file_path={:?}, module_fqname={:?}, nodes={}, edges={})",
            self.file_path,
            self.module_fqname,
            self.nodes.len(),
            self.edges.len(),
        )
    }
}

/// Dedup-aware builder for a `NativeGraph`'s nodes and edges.
///
/// `intern_node` returns the existing index for any node already
/// present (matched on the full identity tuple), inserting only if
/// it's new. `add_edge` is a no-op on a repeated `(src, dst, flags)`
/// triple. Both methods preserve insertion order in the returned
/// vectors so tests can assert deterministic shapes.
#[derive(Hash, Eq, PartialEq, Clone)]
struct NodeKey {
    fqname: String,
    kind: &'static str,
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    flags: u32,
}

struct GraphBuilder {
    nodes: Vec<Py<NativeNode>>,
    node_index: HashMap<NodeKey, usize>,
    edges: Vec<(usize, usize, u32)>,
    edge_set: HashSet<(usize, usize, u32)>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            edges: Vec::new(),
            edge_set: HashSet::new(),
        }
    }

    fn intern_node(&mut self, py: Python<'_>, node: NativeNode) -> PyResult<usize> {
        let key = NodeKey {
            fqname: node.fqname.clone(),
            kind: node.kind,
            path: node.path.clone(),
            start_line: node.start_line,
            start_column: node.start_column,
            end_line: node.end_line,
            end_column: node.end_column,
            flags: node.flags,
        };
        if let Some(&idx) = self.node_index.get(&key) {
            return Ok(idx);
        }
        let idx = self.nodes.len();
        self.nodes.push(Py::new(py, node)?);
        self.node_index.insert(key, idx);
        Ok(idx)
    }

    fn add_edge(&mut self, src: usize, dst: usize, flags: u32) {
        let triple = (src, dst, flags);
        if self.edge_set.insert(triple) {
            self.edges.push(triple);
        }
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
            .map_err(|e| PyValueError::new_err(format!("invalid configuration: {e:?}")))?;

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

    /// Return the top-level declarations of the given file (debug shape).
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

    /// Build a per-file graph contribution as a transferable envelope.
    ///
    /// Emits one synthetic module node first, then one node per
    /// top-level decl, then one `decl -> module` edge per decl.
    /// Nodes and edges are deduplicated -- repeated calls into the
    /// builder with identical inputs collapse to a single entry.
    fn build_file_graph(
        &self,
        py: Python<'_>,
        path: &str,
        module_fqname: &str,
    ) -> PyResult<NativeGraph> {
        let sys_path = SystemPath::new(path);
        let file = system_path_to_file(&self.db, sys_path)
            .map_err(|e| PyOSError::new_err(format!("cannot open {path}: {e:?}")))?;
        let parsed = parsed_module(&self.db, file).load(&self.db);
        let source = source_text(&self.db, file);
        let index = LineIndex::from_source_text(&source);

        // Module node spans the whole file. The module node's
        // position is informational -- libcst's PositionProvider over
        // the module spans (1,0)..(<last>, 0) and we don't try to
        // match it bit-for-bit.
        let module_range = parsed.syntax().range;
        let (msl, msc, mel, mec) = position(&index, &source, module_range);

        let mut builder = GraphBuilder::new();
        let module_idx = builder.intern_node(
            py,
            NativeNode {
                fqname: module_fqname.to_string(),
                kind: "module",
                path: path.to_string(),
                start_line: msl,
                start_column: msc,
                end_line: mel,
                end_column: mec,
                flags: 0,
            },
        )?;

        let mut decls: Vec<Decl> = Vec::new();
        for stmt in &parsed.syntax().body {
            collect_top_level(stmt, &source, &index, &mut decls);
        }
        for decl in decls {
            let decl_idx = builder.intern_node(
                py,
                NativeNode {
                    fqname: format!("{module_fqname}.{}", decl.name),
                    kind: decl.kind,
                    path: path.to_string(),
                    start_line: decl.start_line,
                    start_column: decl.start_column,
                    end_line: decl.end_line,
                    end_column: decl.end_column,
                    flags: 0,
                },
            )?;
            builder.add_edge(decl_idx, module_idx, 0);
        }

        Ok(NativeGraph {
            file_path: path.to_string(),
            module_fqname: module_fqname.to_string(),
            nodes: builder.nodes,
            edges: builder.edges,
        })
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
    m.add_class::<NativeNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<Project>()?;
    Ok(())
}
