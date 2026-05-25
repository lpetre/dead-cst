//! The chainable query DSL exposed to Python via `ProjectContext.query()`.
//! Result types (`DecoratorRef`/`ConstructionRef`/`CallRef`/`FactoryRef`),
//! the `QueryBuilder` entry point, and the per-stream `Query` builders.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyClass;
use ruff_db::files::File;
use rustc_hash::{FxHashMap, FxHashSet};
use ty_project::Db as ProjectDb;

use crate::graph::SymbolNode;
use crate::helpers::{
    args_to_py_vec, call_args_match_kwargs, file_path_string, kwarg_matcher_from_py,
    kwargs_to_py_map, KwargMatcher,
};
use crate::project::ProjectContext;

/// One decorator application on a top-level function or class.
///
/// Field nullability follows the query shape that produced the ref:
/// * ``where_module + where_name`` populates ``decorated`` only.
/// * ``where_owner_attr`` fills ``decorator_owner`` (the textual
///   ``@<owner>.<attr>`` prefix).
/// * ``where_owner_attr_via`` additionally fills ``decorator_via``
///   with the middle attribute name.
#[pyclass(frozen, get_all)]
pub(crate) struct DecoratorRef {
    pub(crate) decorated: Py<SymbolNode>,
    pub(crate) decorator_name: Option<String>,
    pub(crate) decorator_owner: Option<String>,
    pub(crate) decorator_via: Option<String>,
    /// Positional arguments of the decorator's ``Call`` form. Empty
    /// for bare attribute decorators (``@app.route`` without ``()``).
    /// Each entry is a Python literal, a :class:`SymbolNode` (when the
    /// expression resolves to a project decl), or ``None``.
    pub(crate) args: Vec<Py<PyAny>>,
    /// Keyword arguments of the decorator's ``Call`` form. Same value
    /// shape as ``args``.
    pub(crate) kwargs: FxHashMap<String, Py<PyAny>>,
}

#[pymethods]
impl DecoratorRef {
    /// File path of the decorated decl. Read off ``decorated.path`` —
    /// surfaced as a top-level attribute for ergonomics in path-keyed
    /// dispatch.
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.decorated.borrow(py).path.clone()
    }
}

/// One ``<var> = <Ctor>(...)`` construction at module scope.
///
/// ``class_name`` is the upstream constructor's bare name
/// (``"Flask"`` even when imported as ``F``).
///
/// ``args`` and ``kwargs`` carry the construction's full positional /
/// keyword argument shape (same Python-side value shape as
/// :class:`CallRef`).
#[pyclass(frozen, get_all)]
pub(crate) struct ConstructionRef {
    pub(crate) var: Py<SymbolNode>,
    pub(crate) class_name: String,
    /// Positional arguments of the constructor call. Each entry is a
    /// Python literal, a :class:`SymbolNode` (when the expression
    /// resolves to a project decl), or ``None``.
    pub(crate) args: Vec<Py<PyAny>>,
    /// Keyword arguments of the constructor call. Same value shape as
    /// ``args``.
    pub(crate) kwargs: FxHashMap<String, Py<PyAny>>,
}

#[pymethods]
impl ConstructionRef {
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.var.borrow(py).path.clone()
    }
}

/// One matched call site. ``string_arg`` is the literal at the
/// positional index passed to :meth:`CallQuery.string_arg_at`.
#[pyclass(frozen, get_all)]
pub(crate) struct CallRef {
    pub(crate) owner: Py<SymbolNode>,
    pub(crate) string_arg: String,
    /// Positional arguments of the matched call, one entry per source
    /// positional arg. Each entry is a Python literal, a
    /// :class:`SymbolNode` (when the expression resolves to a project
    /// decl), or ``None``.
    pub(crate) args: Vec<Py<PyAny>>,
    /// Keyword arguments of the matched call. Same value shape as
    /// ``args``.
    pub(crate) kwargs: FxHashMap<String, Py<PyAny>>,
}

#[pymethods]
impl CallRef {
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.owner.borrow(py).path.clone()
    }
}

// ---------------------------------------------------------------------------
// Index-form result rows
//
// Lean siblings of the ref types: same dispatch, but every
// ``SymbolNode`` field is replaced with a positional index into
// ``ctx.nodes()`` and the costly ``args`` / ``kwargs`` row payloads
// are dropped. Returned by the ``.row_indices()`` terminals on
// :class:`DecoratorQuery` / :class:`ConstructionQuery` /
// :class:`CallQuery` / :class:`FactoryQuery`. Pair with
// :meth:`ProjectContext.node_attrs` to batch-fetch any
// ``(kind, path, fqname, flags)`` the plugin still needs without
// per-row ``borrow`` ping-pong.
// ---------------------------------------------------------------------------

/// Index-form sibling of :class:`DecoratorRef`. Same metadata fields,
/// minus ``args`` / ``kwargs``; ``decorated_idx`` indexes into
/// ``ctx.nodes()``.
#[pyclass(frozen, get_all)]
pub(crate) struct DecoratorIdxRef {
    pub(crate) decorated_idx: usize,
    pub(crate) decorator_name: Option<String>,
    pub(crate) decorator_owner: Option<String>,
    pub(crate) decorator_via: Option<String>,
}

/// Index-form sibling of :class:`ConstructionRef`. Same metadata
/// fields, minus ``args`` / ``kwargs``; ``var_idx`` indexes into
/// ``ctx.nodes()``.
#[pyclass(frozen, get_all)]
pub(crate) struct ConstructionIdxRef {
    pub(crate) var_idx: usize,
    pub(crate) class_name: String,
}

/// Index-form sibling of :class:`CallRef`. Same metadata fields,
/// minus ``args`` / ``kwargs``; ``owner_idx`` indexes into
/// ``ctx.nodes()``.
#[pyclass(frozen, get_all)]
pub(crate) struct CallIdxRef {
    pub(crate) owner_idx: usize,
    pub(crate) string_arg: String,
}

/// Index-form sibling of :class:`FactoryRef`. ``decl_idx`` indexes
/// into ``ctx.nodes()``; ``kinds`` is the same sorted set of matched
/// constructor bare-names the node-returning terminal emits.
#[pyclass(frozen, get_all)]
pub(crate) struct FactoryIdxRef {
    pub(crate) decl_idx: usize,
    pub(crate) kinds: Vec<String>,
}

// ---------------------------------------------------------------------------
// Builder query API: builders
// ---------------------------------------------------------------------------

/// Entry point for the chainable query API. Returned by
/// :meth:`ProjectContext.query`. Pick a stream type:
/// :meth:`decorators`, :meth:`constructions`, :meth:`calls`,
/// :meth:`subclasses`, :meth:`imports`, :meth:`classes`,
/// :meth:`factories`, or :meth:`edges`. Point lookups (``module``,
/// ``declarations``, ``main_blocks``, etc.) live directly on
/// :class:`ProjectContext`.
#[pyclass(unsendable)]
pub(crate) struct QueryBuilder {
    pub(crate) ctx: Py<ProjectContext>,
}

#[pymethods]
impl QueryBuilder {
    fn decorators(&self, py: Python<'_>) -> DecoratorQuery {
        DecoratorQuery::new(self.ctx.clone_ref(py))
    }
    fn constructions(&self, py: Python<'_>) -> ConstructionQuery {
        ConstructionQuery::new(self.ctx.clone_ref(py))
    }
    fn calls(&self, py: Python<'_>) -> CallQuery {
        CallQuery::new(self.ctx.clone_ref(py))
    }
    fn subclasses(&self, py: Python<'_>) -> SubclassQuery {
        SubclassQuery::new(self.ctx.clone_ref(py))
    }
    fn imports(&self, py: Python<'_>) -> ImportQuery {
        ImportQuery::new(self.ctx.clone_ref(py))
    }
    fn classes(&self, py: Python<'_>) -> ClassQuery {
        ClassQuery::new(self.ctx.clone_ref(py))
    }
    fn factories(&self, py: Python<'_>) -> FactoryQuery {
        FactoryQuery::new(self.ctx.clone_ref(py))
    }
    fn edges(&self, py: Python<'_>) -> EdgeQuery {
        EdgeQuery::new(self.ctx.clone_ref(py))
    }
    fn decls(&self, py: Python<'_>) -> DeclQuery {
        DeclQuery::new(self.ctx.clone_ref(py))
    }
}

pub(crate) fn _extract_str_or_list(py: Python<'_>, obj: PyObject) -> PyResult<Vec<String>> {
    let bound = obj.bind(py);
    if let Ok(s) = bound.extract::<String>() {
        Ok(vec![s])
    } else {
        bound.extract::<Vec<String>>()
    }
}

/// Extract either a single ``str``, a single ``re.Pattern``, a
/// ``list[str]``, a ``list[re.Pattern]``, or a mixed sequence of both,
/// returning ``(literals, compiled_regexes)``. Empty lists yield two
/// empty vecs. Anything else raises ``TypeError``; invalid regex
/// patterns raise ``ValueError`` with the pattern in the message.
///
/// Used by ``DeclQuery::where_fqname`` so the predicate accepts the
/// same four input shapes ``re.fullmatch`` callers reach for.
pub(crate) fn _extract_str_or_regex_or_list(
    py: Python<'_>,
    obj: PyObject,
) -> PyResult<(Vec<String>, Vec<regex::Regex>)> {
    let bound = obj.bind(py);
    let mut literals: Vec<String> = Vec::new();
    let mut regexes: Vec<regex::Regex> = Vec::new();
    let pattern_type = py.import_bound("re")?.getattr("Pattern")?;

    // Single str.
    if let Ok(s) = bound.extract::<String>() {
        literals.push(s);
        return Ok((literals, regexes));
    }
    // Single ``re.Pattern``.
    if bound.is_instance(&pattern_type)? {
        let pat: String = bound.getattr("pattern")?.extract()?;
        regexes.push(
            regex::Regex::new(&pat)
                .map_err(|e| PyValueError::new_err(format!("invalid regex {pat:?}: {e}")))?,
        );
        return Ok((literals, regexes));
    }
    // List / tuple / any iterable of str | re.Pattern.
    if let Ok(seq) = bound.iter() {
        for item in seq {
            let item = item?;
            if let Ok(s) = item.extract::<String>() {
                literals.push(s);
            } else if item.is_instance(&pattern_type)? {
                let pat: String = item.getattr("pattern")?.extract()?;
                regexes.push(
                    regex::Regex::new(&pat).map_err(|e| {
                        PyValueError::new_err(format!("invalid regex {pat:?}: {e}"))
                    })?,
                );
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "expected str | re.Pattern in sequence",
                ));
            }
        }
        return Ok((literals, regexes));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected str | list[str] | re.Pattern | list[re.Pattern]",
    ))
}

/// Compile an optional path regex once for a query's file-iteration
/// loop. Centralized so every ``find_*`` method that takes a
/// ``path_regex`` parameter shares the same error reporting.
pub(crate) fn _compile_path_regex(re_str: Option<&str>) -> PyResult<Option<regex::Regex>> {
    match re_str {
        None => Ok(None),
        Some(s) => regex::Regex::new(s)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("invalid path regex {s:?}: {e}"))),
    }
}

/// Per-file predicate-fusion check. ``true`` when the file should be
/// processed (no regex, or regex matches its absolute path).
pub(crate) fn _path_re_matches(re: &Option<regex::Regex>, db: &dyn ProjectDb, file: File) -> bool {
    match re {
        None => true,
        Some(re) => re.is_match(&file_path_string(db, file)),
    }
}

/// Cheap text prefilter for identifier references before AST/semantic
/// validation. Mirrors `ty_ide::references::contains_identifier`
/// (vendor/ruff/crates/ty_ide/src/references.rs:198) so per-file
/// queries can skip the parse + walk when the file source doesn't
/// even mention the target identifier.
///
/// Matches an ASCII approximation of `\b<name>\b`: every occurrence
/// of `needle` in `source` whose surrounding bytes aren't identifier
/// continuations. Used by every decorator / construction / call /
/// method query that walks ``project_files``; saves the parse on the
/// (typically large) majority of files that don't reference the
/// query's target name at all.
pub(crate) fn _contains_identifier(source: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = source.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut start = 0;
    while let Some(rel) = source[start..].find(needle) {
        let pos = start + rel;
        let after = pos + needle_bytes.len();
        let boundary_before = pos == 0 || !_is_ident_continue(bytes[pos - 1]);
        let boundary_after = bytes
            .get(after)
            .is_none_or(|byte| !_is_ident_continue(*byte));
        if boundary_before && boundary_after {
            return true;
        }
        start = pos + 1;
    }
    false
}

pub(crate) fn _is_ident_continue(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// Multi-needle variant of [`_contains_identifier`]: returns ``true``
/// as soon as any one of ``needles`` is found. Used by queries that
/// take a list of names (decorator name set, ctor name set, …) so the
/// per-file prefilter passes when the file mentions any of them.
pub(crate) fn _contains_any_identifier(source: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| _contains_identifier(source, n))
}

/// Generic parallel per-file walk. ``per_file`` runs on a Salsa
/// snapshot of ``db`` (one ``Db::dyn_clone`` per worker, mirroring
/// the ty_ide find_references pattern at
/// ``vendor/ruff/crates/ty_ide/src/references.rs:107-130``) and
/// returns a ``Vec<T>`` of opaque per-file results.
///
/// Caller is responsible for releasing the GIL with
/// :meth:`pyo3::Python::allow_threads` — the closure passed in must
/// be ``Send + Sync`` and ``T`` must be ``Send``. Materializing
/// ``Py<SymbolNode>`` values (which are GIL-bound) belongs in the
/// caller AFTER ``allow_threads`` returns.
pub(crate) fn par_scan_files<T, F>(
    db: Box<dyn ProjectDb>,
    files: &[File],
    path_re: &Option<regex::Regex>,
    per_file: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(&dyn ProjectDb, File) -> Vec<T> + Send + Sync,
{
    let result = std::sync::Mutex::new(Vec::<T>::new());
    let per_file_ref = &per_file;
    let result_ref = &result;
    // `move` captures `db: Box<dyn ProjectDb>` by value — `dyn Db`
    // has a `Send` supertrait via `salsa::Database`, so the box is
    // Send, but `&dyn Db` is NOT Send (the trait isn't Sync), which
    // is why the box can't be borrowed across the rayon scope.
    rayon::scope(move |s| {
        for &file in files {
            if !_path_re_matches(path_re, &*db, file) {
                continue;
            }
            let db_t: Box<dyn ProjectDb> = ProjectDb::dyn_clone(&*db);
            s.spawn(move |_| {
                let local = per_file_ref(&*db_t, file);
                if !local.is_empty() {
                    result_ref.lock().unwrap().extend(local);
                }
            });
        }
    });
    result.into_inner().unwrap_or_default()
}

pub(crate) fn _to_iter(py: Python<'_>, items: Vec<Py<impl PyClass>>) -> PyResult<PyObject> {
    let list = pyo3::types::PyList::new_bound(py, items);
    let iter_obj = list.call_method0("__iter__")?;
    Ok(iter_obj.unbind())
}

/// Generate the `first` / `count` / `__iter__` boilerplate that every
/// chainable query exposes. Each variant is what the query's `.collect()`
/// supports — see the three call sites below.
///
/// `with_first $Q, $R`: query whose `collect()` returns `Vec<Py<$R>>`
/// and that wants a typed `.first() -> Option<Py<$R>>` shortcut.
///
/// `no_first $Q`: query that wants only `.count()` + `.__iter__()`
/// (typically returns plain `Py<SymbolNode>`).
///
/// `iter_only $Q`: query with a custom `.count()` (cheaper than
/// materializing `.collect()`); just gets `.__iter__()`.
macro_rules! impl_query_methods {
    (with_first $q:ty, $r:ty) => {
        #[pymethods]
        impl $q {
            fn first(&self, py: Python<'_>) -> PyResult<Option<Py<$r>>> {
                Ok(self.collect(py)?.into_iter().next())
            }
            fn count(&self, py: Python<'_>) -> PyResult<usize> {
                Ok(self.collect(py)?.len())
            }
            fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
                _to_iter(py, self.collect(py)?)
            }
        }
    };
    (no_first $q:ty) => {
        #[pymethods]
        impl $q {
            fn count(&self, py: Python<'_>) -> PyResult<usize> {
                Ok(self.collect(py)?.len())
            }
            fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
                _to_iter(py, self.collect(py)?)
            }
        }
    };
    (iter_only $q:ty) => {
        #[pymethods]
        impl $q {
            fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
                _to_iter(py, self.collect(py)?)
            }
        }
    };
}

/// Find decorated top-level functions / classes. Pick exactly one of:
/// * ``where_module(m).where_name(n)`` — ``@m.x`` / ``@x`` where ``x``
///   is imported from ``m``. ``m`` may be a single module string or a
///   list of modules (OR semantics — keep the row if its decorator
///   resolves through an import of any module in the list).
/// * ``where_callee(fqn)`` — fqn-form ``@<fqn>``.
/// * ``where_owner_attr(attrs)`` — ``@<owner>.<attr>(...)``;
///   ``decorator_owner`` carries the textual prefix.
/// * ``where_owner_attr_via(via, attrs)`` —
///   ``@<owner>.<via>.<attr>(...)`` two-level chain.
/// * ``in_decl(node).where_name(names)`` — ``@<node>.<name>``
///   same-file instance-method decorators.
#[pyclass(unsendable)]
pub(crate) struct DecoratorQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) modules: Option<Vec<String>>,
    pub(crate) callee_fqn: Option<String>,
    pub(crate) names: Option<Vec<String>>,
    pub(crate) owner_attrs: Option<Vec<String>>,
    pub(crate) via_attr: Option<String>,
    pub(crate) in_decl: Option<Py<SymbolNode>>,
    pub(crate) path_regex: Option<String>,
    pub(crate) kwarg_matchers: Vec<(String, KwargMatcher)>,
}

impl DecoratorQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            modules: None,
            callee_fqn: None,
            names: None,
            owner_attrs: None,
            via_attr: None,
            in_decl: None,
            path_regex: None,
            kwarg_matchers: Vec::new(),
        }
    }
}

#[pymethods]
impl DecoratorQuery {
    fn where_module<'py>(
        mut slf: PyRefMut<'py, Self>,
        module: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.modules = Some(_extract_str_or_list(py, module)?);
        Ok(slf)
    }
    fn where_callee<'py>(mut slf: PyRefMut<'py, Self>, fqn: String) -> PyRefMut<'py, Self> {
        slf.callee_fqn = Some(fqn);
        slf
    }
    fn where_name<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.names = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    fn where_owner_attr<'py>(
        mut slf: PyRefMut<'py, Self>,
        attrs: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.owner_attrs = Some(_extract_str_or_list(py, attrs)?);
        Ok(slf)
    }
    fn where_owner_attr_via<'py>(
        mut slf: PyRefMut<'py, Self>,
        via: String,
        attrs: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.via_attr = Some(via);
        slf.owner_attrs = Some(_extract_str_or_list(py, attrs)?);
        Ok(slf)
    }
    fn in_decl<'py>(mut slf: PyRefMut<'py, Self>, node: Py<SymbolNode>) -> PyRefMut<'py, Self> {
        slf.in_decl = Some(node);
        slf
    }
    fn where_path<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }

    /// Add a kwarg matcher. Multiple calls AND together.
    ///
    /// ``value`` must be a Python literal (``None`` / ``bool`` /
    /// ``int`` / ``float`` / ``str`` / ``list`` / ``tuple``). Passing
    /// any other type — including a :class:`SymbolNode` — raises
    /// ``ValueError``.
    fn where_kwarg<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        value: Py<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        let matcher = kwarg_matcher_from_py(py, &value)?;
        slf.kwarg_matchers.push((name, matcher));
        Ok(slf)
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<DecoratorRef>>> {
        let rows = self.decorator_rows(py)?;
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("DecoratorQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let mut refs: Vec<Py<DecoratorRef>> = Vec::with_capacity(rows.len());
        for row in rows {
            let args = args_to_py_vec(py, &row.call_args.args, nodes);
            let kwargs = kwargs_to_py_map(py, &row.call_args.kwargs, nodes);
            refs.push(Py::new(
                py,
                DecoratorRef {
                    decorated: nodes[row.decorated_idx].clone_ref(py),
                    decorator_name: row.decorator_name,
                    decorator_owner: row.decorator_owner,
                    decorator_via: row.decorator_via,
                    args,
                    kwargs,
                },
            )?);
        }
        Ok(refs)
    }
}

/// Idx-form intermediate produced by :meth:`DecoratorQuery::decorator_rows`.
/// Carries the same metadata as :class:`DecoratorRef` minus the
/// ``Py<SymbolNode>`` allocation — the node identity is just an index
/// into ``ctx.nodes()``. ``call_args`` is kept so kwarg filtering and
/// ``collect()``'s ``args`` / ``kwargs`` materialization can both share
/// the same upstream.
struct DecoratorRowIdx {
    decorated_idx: usize,
    decorator_name: Option<String>,
    decorator_owner: Option<String>,
    decorator_via: Option<String>,
    call_args: crate::helpers::CallArgs,
}

impl DecoratorQuery {
    /// Shared dispatch used by both terminals (:meth:`collect` and
    /// :meth:`row_indices`). Stays in idx-space throughout so
    /// ``row_indices`` can skip the ``Py<SymbolNode>`` and
    /// ``args_to_py_vec`` allocations entirely.
    fn decorator_rows(&self, py: Python<'_>) -> PyResult<Vec<DecoratorRowIdx>> {
        let ctx = self.ctx.borrow(py);
        let path_regex = self.path_regex.as_deref();
        let kwarg_matchers = &self.kwarg_matchers;
        let mut out: Vec<DecoratorRowIdx> = Vec::new();
        if let Some(owner_attrs) = &self.owner_attrs {
            let triples = if let Some(via) = &self.via_attr {
                ctx.find_handler_decorators_via(py, via, owner_attrs.clone(), path_regex)?
            } else {
                ctx.find_handler_decorators(py, owner_attrs.clone(), path_regex)?
            };
            for (owner_name, decorated_idx, call_args) in triples {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                out.push(DecoratorRowIdx {
                    decorated_idx,
                    decorator_name: None,
                    decorator_owner: Some(owner_name),
                    decorator_via: self.via_attr.clone(),
                    call_args,
                });
            }
        } else if let Some(in_decl_node) = &self.in_decl {
            let names = self.names.as_ref().ok_or_else(|| {
                PyValueError::new_err("DecoratorQuery.in_decl(...) requires .where_name(...)")
            })?;
            let in_decl_ref = in_decl_node.borrow(py);
            let decls = ctx.find_decorations_on(py, &in_decl_ref, names.clone(), path_regex)?;
            let owner_simple = in_decl_ref
                .fqname
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_string();
            drop(in_decl_ref);
            for (decorated_idx, call_args) in decls {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                out.push(DecoratorRowIdx {
                    decorated_idx,
                    decorator_name: None,
                    decorator_owner: Some(owner_simple.clone()),
                    decorator_via: None,
                    call_args,
                });
            }
        } else if let Some(fqn) = &self.callee_fqn {
            let decls = ctx.find_decorated(py, fqn, path_regex)?;
            for (decorated_idx, call_args) in decls {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                out.push(DecoratorRowIdx {
                    decorated_idx,
                    decorator_name: None,
                    decorator_owner: None,
                    decorator_via: None,
                    call_args,
                });
            }
        } else if let (Some(modules), Some(names)) = (&self.modules, &self.names) {
            let decls = ctx.find_decorated_decls(py, modules, names.clone(), path_regex)?;
            for (decorated_idx, call_args) in decls {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                out.push(DecoratorRowIdx {
                    decorated_idx,
                    decorator_name: None,
                    decorator_owner: None,
                    decorator_via: None,
                    call_args,
                });
            }
        } else {
            return Err(PyValueError::new_err(
                "DecoratorQuery requires one of: where_callee(...); \
                 where_module(...) + where_name(...); where_owner_attr(...); \
                 where_owner_attr_via(via, attrs); or in_decl(node) + where_name(...)",
            ));
        }
        Ok(out)
    }
}

impl_query_methods!(with_first DecoratorQuery, DecoratorRef);

#[pymethods]
impl DecoratorQuery {
    /// Index-form row terminal. Same dispatch as :meth:`collect`, but
    /// each row carries ``decorated_idx`` (a positional index into
    /// ``ctx.nodes()``) instead of a ``Py<SymbolNode>``, and the
    /// ``args`` / ``kwargs`` row payloads are skipped. Pairs with
    /// :meth:`ProjectContext.node_attrs` and :class:`AddEdgeByIdx` /
    /// :class:`AddNodeByIdx` to stay GIL-light end-to-end.
    fn row_indices(&self, py: Python<'_>) -> PyResult<Vec<Py<DecoratorIdxRef>>> {
        let rows = self.decorator_rows(py)?;
        rows.into_iter()
            .map(|row| {
                Py::new(
                    py,
                    DecoratorIdxRef {
                        decorated_idx: row.decorated_idx,
                        decorator_name: row.decorator_name,
                        decorator_owner: row.decorator_owner,
                        decorator_via: row.decorator_via,
                    },
                )
            })
            .collect()
    }
}

/// Find module-scope ``<var> = <Ctor>(...)`` sites. Pick exactly one
/// of ``where_module + where_name`` or
/// ``where_class(fqn, include_subclasses=...)``.
///
/// ``where_module`` accepts either a single module string or a list of
/// modules (OR semantics — match if the constructor is imported from
/// any of them).
#[pyclass(unsendable)]
pub(crate) struct ConstructionQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) modules: Option<Vec<String>>,
    pub(crate) names: Option<Vec<String>>,
    pub(crate) class_fqn: Option<String>,
    pub(crate) include_subclasses: bool,
    pub(crate) path_regex: Option<String>,
}

impl ConstructionQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            modules: None,
            names: None,
            class_fqn: None,
            include_subclasses: false,
            path_regex: None,
        }
    }
}

#[pymethods]
impl ConstructionQuery {
    fn where_module<'py>(
        mut slf: PyRefMut<'py, Self>,
        module: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.modules = Some(_extract_str_or_list(py, module)?);
        Ok(slf)
    }
    fn where_name<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.names = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    #[pyo3(signature = (fqn, *, include_subclasses = false))]
    fn where_class<'py>(
        mut slf: PyRefMut<'py, Self>,
        fqn: String,
        include_subclasses: bool,
    ) -> PyRefMut<'py, Self> {
        slf.class_fqn = Some(fqn);
        slf.include_subclasses = include_subclasses;
        slf
    }
    fn where_path<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<ConstructionRef>>> {
        let rows = self.construction_rows(py)?;
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("ConstructionQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let mut refs: Vec<Py<ConstructionRef>> = Vec::with_capacity(rows.len());
        for row in rows {
            let args = args_to_py_vec(py, &row.call_args.args, nodes);
            let kwargs = kwargs_to_py_map(py, &row.call_args.kwargs, nodes);
            refs.push(Py::new(
                py,
                ConstructionRef {
                    var: nodes[row.var_idx].clone_ref(py),
                    class_name: row.class_name,
                    args,
                    kwargs,
                },
            )?);
        }
        Ok(refs)
    }
}

/// Idx-form intermediate produced by :meth:`ConstructionQuery::construction_rows`.
struct ConstructionRowIdx {
    var_idx: usize,
    class_name: String,
    call_args: crate::helpers::CallArgs,
}

impl ConstructionQuery {
    /// Shared dispatch used by both terminals (:meth:`collect` and
    /// :meth:`row_indices`). Stays in idx-space throughout.
    fn construction_rows(&self, py: Python<'_>) -> PyResult<Vec<ConstructionRowIdx>> {
        let ctx = self.ctx.borrow(py);
        let path_regex = self.path_regex.as_deref();
        let mut out: Vec<ConstructionRowIdx> = Vec::new();
        if let Some(fqn) = &self.class_fqn {
            let pairs = ctx.find_constructions(py, fqn, self.include_subclasses, path_regex)?;
            let cls_name = fqn.rsplit('.').next().unwrap_or("").to_string();
            for (var_idx, call_args) in pairs {
                out.push(ConstructionRowIdx {
                    var_idx,
                    class_name: cls_name.clone(),
                    call_args,
                });
            }
        } else if let (Some(modules), Some(names)) = (&self.modules, &self.names) {
            let triples =
                ctx.find_instance_constructions(py, modules, names.clone(), path_regex)?;
            for (var_idx, name, call_args) in triples {
                out.push(ConstructionRowIdx {
                    var_idx,
                    class_name: name,
                    call_args,
                });
            }
        } else {
            return Err(PyValueError::new_err(
                "ConstructionQuery requires either where_class(...) \
                 or where_module(...) + where_name(...)",
            ));
        }
        Ok(out)
    }
}

impl_query_methods!(with_first ConstructionQuery, ConstructionRef);

#[pymethods]
impl ConstructionQuery {
    /// Index-form row terminal. Same dispatch as :meth:`collect`, but
    /// each row carries ``var_idx`` (a positional index into
    /// ``ctx.nodes()``) instead of a ``Py<SymbolNode>``, and the
    /// ``args`` / ``kwargs`` row payloads are skipped.
    fn row_indices(&self, py: Python<'_>) -> PyResult<Vec<Py<ConstructionIdxRef>>> {
        let rows = self.construction_rows(py)?;
        rows.into_iter()
            .map(|row| {
                Py::new(
                    py,
                    ConstructionIdxRef {
                        var_idx: row.var_idx,
                        class_name: row.class_name,
                    },
                )
            })
            .collect()
    }
}

/// Find call sites whose positional string-literal at the configured
/// index is captured. :meth:`string_arg_at` is required. Pick one of:
/// * ``where_module(m).where_name(n)`` — call to ``n`` imported from
///   ``m``. ``m`` may be a single string or a list of modules (OR
///   semantics).
/// * ``where_owner(o).where_attr(a)`` — ``<o>.<a>(...)`` literal
///   receiver match.
/// * ``where_attr(a)`` — ``<expr>.<a>(...)`` any receiver.
#[pyclass(unsendable)]
pub(crate) struct CallQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) modules: Option<Vec<String>>,
    pub(crate) name: Option<String>,
    pub(crate) owner: Option<String>,
    pub(crate) attr: Option<String>,
    pub(crate) arg_index: Option<usize>,
    pub(crate) required_positional: Option<usize>,
    pub(crate) path_regex: Option<String>,
    pub(crate) kwarg_matchers: Vec<(String, KwargMatcher)>,
}

impl CallQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            modules: None,
            name: None,
            owner: None,
            attr: None,
            arg_index: None,
            required_positional: None,
            path_regex: None,
            kwarg_matchers: Vec::new(),
        }
    }
}

#[pymethods]
impl CallQuery {
    fn where_module<'py>(
        mut slf: PyRefMut<'py, Self>,
        module: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.modules = Some(_extract_str_or_list(py, module)?);
        Ok(slf)
    }
    fn where_name<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.name = Some(name);
        slf
    }
    fn where_owner<'py>(mut slf: PyRefMut<'py, Self>, owner: String) -> PyRefMut<'py, Self> {
        slf.owner = Some(owner);
        slf
    }
    fn where_attr<'py>(mut slf: PyRefMut<'py, Self>, attr: String) -> PyRefMut<'py, Self> {
        slf.attr = Some(attr);
        slf
    }
    fn string_arg_at<'py>(mut slf: PyRefMut<'py, Self>, index: usize) -> PyRefMut<'py, Self> {
        slf.arg_index = Some(index);
        slf
    }
    #[pyo3(signature = (n=None))]
    fn where_required_positional<'py>(
        mut slf: PyRefMut<'py, Self>,
        n: Option<usize>,
    ) -> PyRefMut<'py, Self> {
        slf.required_positional = n;
        slf
    }
    fn where_path<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }

    /// Add a kwarg matcher. Multiple calls AND together.
    ///
    /// ``value`` must be a Python literal (``None`` / ``bool`` /
    /// ``int`` / ``float`` / ``str`` / ``list`` / ``tuple``). Passing
    /// any other type — including a :class:`SymbolNode` — raises
    /// ``ValueError``.
    fn where_kwarg<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        value: Py<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        let matcher = kwarg_matcher_from_py(py, &value)?;
        slf.kwarg_matchers.push((name, matcher));
        Ok(slf)
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<CallRef>>> {
        let rows = self.call_rows(py)?;
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("CallQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let mut refs: Vec<Py<CallRef>> = Vec::with_capacity(rows.len());
        for row in rows {
            let args = args_to_py_vec(py, &row.call_args.args, nodes);
            let kwargs = kwargs_to_py_map(py, &row.call_args.kwargs, nodes);
            refs.push(Py::new(
                py,
                CallRef {
                    owner: nodes[row.owner_idx].clone_ref(py),
                    string_arg: row.string_arg,
                    args,
                    kwargs,
                },
            )?);
        }
        Ok(refs)
    }
}

/// Idx-form intermediate produced by :meth:`CallQuery::call_rows`.
struct CallRowIdx {
    owner_idx: usize,
    string_arg: String,
    call_args: crate::helpers::CallArgs,
}

impl CallQuery {
    /// Shared dispatch used by both terminals (:meth:`collect` and
    /// :meth:`row_indices`). Stays in idx-space throughout.
    fn call_rows(&self, py: Python<'_>) -> PyResult<Vec<CallRowIdx>> {
        let ctx = self.ctx.borrow(py);
        let arg_index = self
            .arg_index
            .ok_or_else(|| PyValueError::new_err("CallQuery: .string_arg_at(index) is required"))?;
        let path_regex = self.path_regex.as_deref();
        let triples = if let (Some(modules), Some(name)) = (&self.modules, &self.name) {
            ctx.find_calls_to_imported(py, modules, name, arg_index, path_regex)?
        } else if let (Some(owner), Some(attr)) = (&self.owner, &self.attr) {
            ctx.find_calls_on_var(
                py,
                owner,
                attr,
                arg_index,
                self.required_positional,
                path_regex,
            )?
        } else if let Some(attr) = &self.attr {
            ctx.find_calls_on_attr(py, attr, arg_index, path_regex)?
        } else {
            return Err(PyValueError::new_err(
                "CallQuery requires one of: where_module(...) + where_name(...); \
                 where_owner(...) + where_attr(...); or where_attr(...)",
            ));
        };
        let kwarg_matchers = &self.kwarg_matchers;
        let mut out: Vec<CallRowIdx> = Vec::new();
        for (owner_idx, string_arg, call_args) in triples {
            if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                continue;
            }
            out.push(CallRowIdx {
                owner_idx,
                string_arg,
                call_args,
            });
        }
        Ok(out)
    }
}

impl_query_methods!(with_first CallQuery, CallRef);

#[pymethods]
impl CallQuery {
    /// Index-form row terminal. Same dispatch as :meth:`collect`, but
    /// each row carries ``owner_idx`` (a positional index into
    /// ``ctx.nodes()``) instead of a ``Py<SymbolNode>``, and the
    /// ``args`` / ``kwargs`` row payloads are skipped. ``string_arg``
    /// — the literal at the configured positional index — is preserved
    /// because most call-shape plugins key on it.
    fn row_indices(&self, py: Python<'_>) -> PyResult<Vec<Py<CallIdxRef>>> {
        let rows = self.call_rows(py)?;
        rows.into_iter()
            .map(|row| {
                Py::new(
                    py,
                    CallIdxRef {
                        owner_idx: row.owner_idx,
                        string_arg: row.string_arg,
                    },
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Subclass / Import / Class / Factory queries
// ---------------------------------------------------------------------------

/// Walk transitive subclasses of a base class. Pick exactly one of:
/// * ``of_fqn(fqn)`` — base by dotted name (external classes ok —
///   ``unittest.TestCase`` etc.).
/// * ``of_node(class_node)`` — base by project-local class node.
/// ``transitive(bool)`` controls whether the BFS walks past the
/// direct subclass frontier (default ``True``; for ``of_node`` the
/// BFS is always transitive).
#[pyclass(unsendable)]
pub(crate) struct SubclassQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) base_fqn: Option<String>,
    pub(crate) base_node: Option<Py<SymbolNode>>,
    pub(crate) transitive: bool,
}

impl SubclassQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            base_fqn: None,
            base_node: None,
            transitive: true,
        }
    }
}

#[pymethods]
impl SubclassQuery {
    fn of_fqn<'py>(mut slf: PyRefMut<'py, Self>, fqn: String) -> PyRefMut<'py, Self> {
        slf.base_fqn = Some(fqn);
        slf
    }
    fn of_node<'py>(mut slf: PyRefMut<'py, Self>, node: Py<SymbolNode>) -> PyRefMut<'py, Self> {
        slf.base_node = Some(node);
        slf
    }
    fn transitive<'py>(mut slf: PyRefMut<'py, Self>, value: bool) -> PyRefMut<'py, Self> {
        slf.transitive = value;
        slf
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let ctx = self.ctx.borrow(py);
        if let Some(fqn) = &self.base_fqn {
            ctx.find_subclasses(py, fqn, self.transitive)
        } else if let Some(node) = &self.base_node {
            ctx.find_subclasses_of(py, &node.borrow(py))
        } else {
            Err(PyValueError::new_err(
                "SubclassQuery requires .of_fqn(...) or .of_node(...)",
            ))
        }
    }

    /// Index-returning terminal. Same lookup as :meth:`collect`, but
    /// emits each subclass's positional index into ``ctx.nodes()``
    /// instead of allocating one ``Py<SymbolNode>`` per row.
    fn indices(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        let ctx = self.ctx.borrow(py);
        if let Some(fqn) = &self.base_fqn {
            ctx.find_subclasses_indices(py, fqn, self.transitive)
        } else if let Some(node) = &self.base_node {
            ctx.find_subclasses_of_indices(&node.borrow(py))
        } else {
            Err(PyValueError::new_err(
                "SubclassQuery requires .of_fqn(...) or .of_node(...)",
            ))
        }
    }
}

impl_query_methods!(no_first SubclassQuery);

/// Enumerate the ``kind="import"`` nodes that bind a name from a
/// given module. Requires ``of(module)``.
#[pyclass(unsendable)]
pub(crate) struct ImportQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) module: Option<String>,
}

impl ImportQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self { ctx, module: None }
    }

    fn module_or_err(&self) -> PyResult<&str> {
        self.module
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("ImportQuery requires .of(module)"))
    }
}

#[pymethods]
impl ImportQuery {
    fn of<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
    }
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        self.ctx
            .borrow(py)
            .find_imports_of(py, self.module_or_err()?)
    }
    /// Index-returning terminal. Reads positional indices straight
    /// out of the pre-built ``imports_by_module`` index — no Python
    /// allocation per row.
    fn indices(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        self.ctx
            .borrow(py)
            .find_imports_of_indices(self.module_or_err()?)
    }
    /// O(1) presence probe — does any project file import the
    /// configured module? Short-circuits on the first match without
    /// materialising a Python list. Preferred over ``.count() > 0`` /
    /// ``.collect()`` for plugin guards that just need a boolean.
    fn exists(&self, py: Python<'_>) -> PyResult<bool> {
        self.ctx.borrow(py).has_imports_of(self.module_or_err()?)
    }
    /// Count without allocating `Py<SymbolNode>` clones — read the
    /// length of the pre-built index entry directly.
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.ctx.borrow(py).imports_of_count(self.module_or_err()?))
    }
}

impl_query_methods!(iter_only ImportQuery);

/// Enumerate classes by structural property. Today the only filter
/// is ``defining_method(name)`` (matches classes whose body has a
/// ``FunctionDef`` with that name); easy to extend later.
#[pyclass(unsendable)]
pub(crate) struct ClassQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) defining_method: Option<String>,
}

impl ClassQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            defining_method: None,
        }
    }
}

#[pymethods]
impl ClassQuery {
    fn defining_method<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.defining_method = Some(name);
        slf
    }
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let ctx = self.ctx.borrow(py);
        let name = self
            .defining_method
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("ClassQuery requires .defining_method(name)"))?;
        ctx.find_classes_defining_method(py, name)
    }

    /// Index-returning terminal. Same per-file parallel walk as
    /// :meth:`collect`, but emits each class node's positional index
    /// into ``ctx.nodes()`` instead of allocating ``Py<SymbolNode>``
    /// clones.
    fn indices(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        let ctx = self.ctx.borrow(py);
        let name = self
            .defining_method
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("ClassQuery requires .defining_method(name)"))?;
        ctx.find_classes_defining_method_indices(py, name)
    }
}

impl_query_methods!(no_first ClassQuery);

/// One result row from :class:`FactoryQuery`. ``decl`` is the
/// owning top-level function or class; ``kinds`` is the sorted set
/// of constructor bare-names matched inside its body.
#[pyclass(frozen, get_all)]
pub(crate) struct FactoryRef {
    pub(crate) decl: Py<SymbolNode>,
    pub(crate) kinds: Vec<String>,
}

#[pymethods]
impl FactoryRef {
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.decl.borrow(py).path.clone()
    }
}

/// Walk function / class bodies for ``<Ctor>(...)`` calls where
/// ``Ctor`` is imported from ``of_module(...)`` and matches one of
/// ``where_name(...)``. Both filters are required.
///
/// ``of_module`` accepts either a single module string or a list of
/// modules (OR semantics).
#[pyclass(unsendable)]
pub(crate) struct FactoryQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) modules: Option<Vec<String>>,
    pub(crate) names: Option<Vec<String>>,
}

impl FactoryQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            modules: None,
            names: None,
        }
    }
}

#[pymethods]
impl FactoryQuery {
    fn of_module<'py>(
        mut slf: PyRefMut<'py, Self>,
        module: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.modules = Some(_extract_str_or_list(py, module)?);
        Ok(slf)
    }
    fn where_name<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.names = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<FactoryRef>>> {
        let pairs = self.factory_rows(py)?;
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("FactoryQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        pairs
            .into_iter()
            .map(|(decl_idx, kinds)| {
                Py::new(
                    py,
                    FactoryRef {
                        decl: nodes[decl_idx].clone_ref(py),
                        kinds,
                    },
                )
            })
            .collect()
    }
}

impl FactoryQuery {
    /// Shared dispatch used by both terminals (:meth:`collect` and
    /// :meth:`row_indices`). Stays in idx-space throughout.
    fn factory_rows(&self, py: Python<'_>) -> PyResult<Vec<(usize, Vec<String>)>> {
        let ctx = self.ctx.borrow(py);
        let modules = self
            .modules
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FactoryQuery requires .of_module(...)"))?;
        let names = self
            .names
            .clone()
            .ok_or_else(|| PyValueError::new_err("FactoryQuery requires .where_name(...)"))?;
        ctx.find_factory_decls(py, modules, names)
    }
}

impl_query_methods!(no_first FactoryQuery);

#[pymethods]
impl FactoryQuery {
    /// Index-form row terminal. Same dispatch as :meth:`collect`;
    /// each row carries ``decl_idx`` (a positional index into
    /// ``ctx.nodes()``) instead of a ``Py<SymbolNode>``. ``kinds`` is
    /// the same sorted set of matched constructor bare-names the
    /// node-returning terminal emits.
    fn row_indices(&self, py: Python<'_>) -> PyResult<Vec<Py<FactoryIdxRef>>> {
        let pairs = self.factory_rows(py)?;
        pairs
            .into_iter()
            .map(|(decl_idx, kinds)| Py::new(py, FactoryIdxRef { decl_idx, kinds }))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// EdgeQuery — filtered enumeration over the in-progress graph's edges
// ---------------------------------------------------------------------------

/// One graph edge with both endpoint nodes resolved. Avoids the
/// `nodes[src_idx]` / `nodes[dst_idx]` ping-pong that a Python-side
/// ``for src_idx, dst_idx, flags in ctx.edges()`` loop pays.
#[pyclass(frozen, get_all)]
pub(crate) struct EdgeRef {
    pub(crate) src: Py<SymbolNode>,
    pub(crate) dst: Py<SymbolNode>,
    pub(crate) flags: u32,
}

/// Filtered enumeration over the in-progress graph's edges. Predicates
/// AND together; any unset predicate doesn't filter.
///
/// * ``with_flags(mask)`` — keep edges where ``flags & mask != 0``.
/// * ``with_src_kind(kind)`` / ``with_dst_kind(kind)`` — keep edges
///   whose endpoint ``SymbolNode.kind`` matches the given string.
///
/// Avoids the per-edge Python ↔ rust FFI hop that a
/// ``for src_idx, dst_idx, flags in ctx.edges()`` plus
/// ``nodes = ctx.nodes(); nodes[src_idx]`` pattern pays — the entire
/// filter runs rust-side and only the surviving rows are materialized
/// into ``Py<SymbolNode>``.
#[pyclass(unsendable)]
pub(crate) struct EdgeQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) flag_mask: Option<u32>,
    pub(crate) src_kind: Option<String>,
    pub(crate) dst_kind: Option<String>,
}

impl EdgeQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            flag_mask: None,
            src_kind: None,
            dst_kind: None,
        }
    }
}

#[pymethods]
impl EdgeQuery {
    /// Keep edges where ``flags & mask != 0``. Pass an
    /// ``EdgeFlags`` constant (or OR of constants) to filter to a
    /// specific edge classification.
    fn with_flags<'py>(mut slf: PyRefMut<'py, Self>, mask: u32) -> PyRefMut<'py, Self> {
        slf.flag_mask = Some(mask);
        slf
    }
    /// Keep edges whose ``src`` node has the given ``kind`` (``"module"``,
    /// ``"function"``, ``"import"``, …). Matches by exact string compare
    /// against ``SymbolNode.kind``.
    fn with_src_kind<'py>(mut slf: PyRefMut<'py, Self>, kind: String) -> PyRefMut<'py, Self> {
        slf.src_kind = Some(kind);
        slf
    }
    /// Keep edges whose ``dst`` node has the given ``kind``.
    fn with_dst_kind<'py>(mut slf: PyRefMut<'py, Self>, kind: String) -> PyRefMut<'py, Self> {
        slf.dst_kind = Some(kind);
        slf
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<EdgeRef>>> {
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("EdgeQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let edges: &[(usize, usize, u32)] = &outputs.builder.edges;
        let flag_mask = self.flag_mask;
        let src_kind = self.src_kind.as_deref();
        let dst_kind = self.dst_kind.as_deref();
        let mut refs: Vec<Py<EdgeRef>> = Vec::new();
        for &(src_idx, dst_idx, flags) in edges {
            if let Some(mask) = flag_mask {
                if flags & mask == 0 {
                    continue;
                }
            }
            // Endpoint-kind predicates: borrow only the side(s) the
            // caller actually filtered on so the common
            // ``with_flags(...)`` shape pays nothing for kind checks.
            if let Some(needle) = src_kind {
                if nodes[src_idx].borrow(py).kind != needle {
                    continue;
                }
            }
            if let Some(needle) = dst_kind {
                if nodes[dst_idx].borrow(py).kind != needle {
                    continue;
                }
            }
            refs.push(Py::new(
                py,
                EdgeRef {
                    src: nodes[src_idx].clone_ref(py),
                    dst: nodes[dst_idx].clone_ref(py),
                    flags,
                },
            )?);
        }
        Ok(refs)
    }

    /// Index-returning terminal for edges. Same per-edge predicate
    /// pipeline as :meth:`collect`, but emits ``(src_idx, dst_idx,
    /// flags)`` triples instead of materialising one :class:`EdgeRef`
    /// (and two ``Py<SymbolNode>`` clones) per row.
    ///
    /// Faster than :meth:`collect` when you only need set-membership
    /// or counting over edge endpoints; pair with
    /// :meth:`ProjectContext.nodes_at` to revive the surviving
    /// endpoints on demand.
    fn index_triples(&self, py: Python<'_>) -> PyResult<Vec<(usize, usize, u32)>> {
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("EdgeQuery.index_triples")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let edges: &[(usize, usize, u32)] = &outputs.builder.edges;
        let flag_mask = self.flag_mask;
        let src_kind = self.src_kind.as_deref();
        let dst_kind = self.dst_kind.as_deref();
        let mut out: Vec<(usize, usize, u32)> = Vec::new();
        for &(src_idx, dst_idx, flags) in edges {
            if let Some(mask) = flag_mask {
                if flags & mask == 0 {
                    continue;
                }
            }
            if let Some(needle) = src_kind {
                if nodes[src_idx].borrow(py).kind != needle {
                    continue;
                }
            }
            if let Some(needle) = dst_kind {
                if nodes[dst_idx].borrow(py).kind != needle {
                    continue;
                }
            }
            out.push((src_idx, dst_idx, flags));
        }
        Ok(out)
    }
}

impl_query_methods!(with_first EdgeQuery, EdgeRef);

// ---------------------------------------------------------------------------
// DeclQuery — generic node-stream filter
// ---------------------------------------------------------------------------

/// Generic filter over every interned node in the in-progress graph.
///
/// Folds the per-node Python filter loops that show up in plugins
/// (filter on ``kind``, basename, simple-name, flag mask, path set,
/// path regex) down into one rust pass. All configured predicates are
/// AND-ed; an empty predicate set yields every node.
///
/// Pick the predicate(s) you need and call ``.collect()`` /
/// ``.first()`` / ``.count()`` / iterate. Predicates that take a
/// ``list[str]`` short-circuit on length-1 to avoid the hashset
/// overhead; pass a single ``str`` to the singular form when you
/// already know it's one.
#[pyclass(unsendable)]
pub(crate) struct DeclQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) kinds: Option<Vec<String>>,
    pub(crate) filenames: Option<Vec<String>>,
    pub(crate) simple_names: Option<Vec<String>>,
    pub(crate) paths: Option<Vec<String>>,
    pub(crate) path_regex: Option<String>,
    pub(crate) flags_mask: Option<u32>,
    pub(crate) flags_match_any: bool,
    pub(crate) fqname_prefix: Option<String>,
    /// Segment-bounded descendant filter, populated by
    /// :meth:`with_fqname_under`. When ``Some(parent)``, restrict to
    /// the node whose fqname equals ``parent`` plus every transitive
    /// descendant of ``parent`` in the fqname tree (modules and decls,
    /// segment-bounded). Backed by ``children_by_parent`` — see
    /// :meth:`with_fqname_under` for the contract and comparison
    /// against the raw-``starts_with`` :meth:`with_fqname_prefix`.
    pub(crate) fqname_under: Option<String>,
    /// `where_fqname` literal-string candidates. ``None`` if the
    /// predicate isn't configured; ``Some(vec)`` if at least one
    /// literal was passed. Combined with ``fqname_regexes`` via OR.
    pub(crate) fqname_literals: Option<Vec<String>>,
    /// `where_fqname` compiled-regex candidates. ``None`` if the
    /// predicate isn't configured; ``Some(vec)`` if at least one
    /// ``re.Pattern`` was passed. Combined with ``fqname_literals``
    /// via OR.
    pub(crate) fqname_regexes: Option<Vec<regex::Regex>>,
}

impl DeclQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            kinds: None,
            filenames: None,
            simple_names: None,
            paths: None,
            path_regex: None,
            flags_mask: None,
            flags_match_any: false,
            fqname_prefix: None,
            fqname_under: None,
            fqname_literals: None,
            fqname_regexes: None,
        }
    }
}

#[pymethods]
impl DeclQuery {
    /// Restrict to nodes with this ``kind`` string (``"function"`` /
    /// ``"class"`` / ``"variable"`` / ``"import"`` / ``"module"`` /
    /// ``"synthetic"`` / ``"type_alias"``).
    fn with_kind<'py>(mut slf: PyRefMut<'py, Self>, kind: String) -> PyRefMut<'py, Self> {
        slf.kinds = Some(vec![kind]);
        slf
    }
    /// Restrict to nodes whose ``kind`` is in ``kinds``.
    fn with_kinds<'py>(
        mut slf: PyRefMut<'py, Self>,
        kinds: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.kinds = Some(_extract_str_or_list(py, kinds)?);
        Ok(slf)
    }
    /// Restrict to nodes whose file basename equals ``name``.
    fn with_filename<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.filenames = Some(vec![name]);
        slf
    }
    /// Restrict to nodes whose file basename is in ``names``.
    fn with_filenames<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.filenames = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    /// Restrict to nodes whose ``fqname.rsplit('.', 1)[-1]`` equals
    /// ``name`` (i.e. the trailing segment).
    fn with_simple_name<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.simple_names = Some(vec![name]);
        slf
    }
    /// Restrict to nodes whose trailing fqname segment is in ``names``.
    fn with_simple_names<'py>(
        mut slf: PyRefMut<'py, Self>,
        names: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.simple_names = Some(_extract_str_or_list(py, names)?);
        Ok(slf)
    }
    /// Restrict to nodes whose absolute path is in ``paths``.
    fn with_paths<'py>(
        mut slf: PyRefMut<'py, Self>,
        paths: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.paths = Some(_extract_str_or_list(py, paths)?);
        Ok(slf)
    }
    /// Restrict to nodes whose absolute path matches ``regex``.
    fn with_path_regex<'py>(mut slf: PyRefMut<'py, Self>, regex: String) -> PyRefMut<'py, Self> {
        slf.path_regex = Some(regex);
        slf
    }
    /// Restrict to nodes whose ``flags & mask == mask`` (all bits in
    /// ``mask`` are set).
    fn with_flags<'py>(mut slf: PyRefMut<'py, Self>, mask: u32) -> PyRefMut<'py, Self> {
        slf.flags_mask = Some(mask);
        slf.flags_match_any = false;
        slf
    }
    /// Restrict to nodes whose ``flags & mask != 0`` (any bit in
    /// ``mask`` is set).
    fn with_any_flag<'py>(mut slf: PyRefMut<'py, Self>, mask: u32) -> PyRefMut<'py, Self> {
        slf.flags_mask = Some(mask);
        slf.flags_match_any = true;
        slf
    }
    /// Restrict to nodes whose ``fqname`` starts with ``prefix`` (raw
    /// string prefix — not segment-bounded). ``prefix="foo"`` matches
    /// both ``foo.bar`` and ``foobar``. Use :meth:`with_fqname_under`
    /// for the segment-bounded "descendants of this fqname" predicate
    /// that walks the fqname tree.
    fn with_fqname_prefix<'py>(
        mut slf: PyRefMut<'py, Self>,
        prefix: String,
    ) -> PyRefMut<'py, Self> {
        slf.fqname_prefix = Some(prefix);
        slf
    }

    /// Restrict to nodes whose ``fqname`` equals ``parent_fqn`` or is
    /// a transitive descendant of it in the fqname tree.
    ///
    /// Segment-bounded: ``parent_fqn="pkg.foo"`` matches ``pkg.foo``,
    /// ``pkg.foo.bar``, ``pkg.foo.bar.baz`` — but **not** ``pkg.foobar``.
    /// Backed by the project's ``children_by_parent`` index, so this
    /// is O(matches) instead of the O(all_nodes) scan
    /// :meth:`with_fqname_prefix` performs.
    fn with_fqname_under<'py>(
        mut slf: PyRefMut<'py, Self>,
        parent_fqn: String,
    ) -> PyRefMut<'py, Self> {
        slf.fqname_under = Some(parent_fqn);
        slf
    }

    /// Restrict to nodes whose ``fqname`` matches the predicate.
    /// Accepts ``str`` (literal equality), ``list[str]`` (literal
    /// equality against any element), ``re.Pattern`` (full-string
    /// regex search), ``list[re.Pattern]`` (any-of regex search),
    /// or a mixed sequence of ``str`` and ``re.Pattern`` (literal
    /// equality OR regex search). ``re.Pattern`` instances are
    /// recompiled rust-side using rust's ``regex`` crate, so any
    /// PCRE-only syntax is rejected with ``ValueError`` at this
    /// call.
    fn where_fqname<'py>(
        mut slf: PyRefMut<'py, Self>,
        value: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        let (literals, regexes) = _extract_str_or_regex_or_list(py, value)?;
        slf.fqname_literals = if literals.is_empty() {
            // An empty literal vec is the "matches-nothing" sentinel —
            // consistent with ``where_name([])`` semantics elsewhere in
            // the DSL. We still need to flip ``Some`` so the collect
            // predicate kicks in (otherwise a sole empty list would
            // match everything).
            Some(Vec::new())
        } else {
            Some(literals)
        };
        slf.fqname_regexes = if regexes.is_empty() {
            None
        } else {
            Some(regexes)
        };
        Ok(slf)
    }

    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let indices = self.indices(py)?;
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("DeclQuery.collect")?;
        Ok(indices
            .into_iter()
            .map(|i| outputs.builder.nodes[i].clone_ref(py))
            .collect())
    }

    /// Index-returning terminal. Same predicate semantics as
    /// :meth:`collect`, but emits each surviving node's positional
    /// index into ``ctx.nodes()`` (a plain ``list[int]``) instead of
    /// allocating one ``Py<SymbolNode>`` per row.
    ///
    /// Use when you only need set membership / counting on the
    /// surviving nodes (or want to feed an index-keyed
    /// :class:`AddEdgeByIdx`); call :meth:`ProjectContext.nodes_at`
    /// to materialize back to ``SymbolNode`` later.
    fn indices(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        let ctx = self.ctx.borrow(py);
        let outputs = ctx.materialized("DeclQuery.indices")?;
        let path_regex = _compile_path_regex(self.path_regex.as_deref())?;

        // Pre-compute hashsets for predicates that take lists. Singular
        // forms reuse the same vec (length 1) so the hashset cost is
        // amortized over the per-node loop.
        let kinds_set: Option<FxHashSet<&str>> = self
            .kinds
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let filenames_set: Option<FxHashSet<&str>> = self
            .filenames
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let simple_set: Option<FxHashSet<&str>> = self
            .simple_names
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let paths_set: Option<FxHashSet<&str>> = self
            .paths
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let fqname_literals_set: Option<FxHashSet<&str>> = self
            .fqname_literals
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());

        // For ``with_fqname_under``: BFS the children-by-parent tree
        // once to materialise the candidate index set, then restrict
        // the main loop to those indices. Empty set means "parent
        // unknown" -> the query returns nothing.
        let fqname_under_indices: Option<FxHashSet<usize>> =
            self.fqname_under.as_ref().map(|root| {
                let mut set: FxHashSet<usize> = FxHashSet::default();
                // The root may be a module or a top-level decl. Include
                // every direct hit before BFS-ing into modules below.
                if let Some(&module_idx) = outputs.module_by_fqname.get(root.as_str()) {
                    set.insert(module_idx);
                }
                if let Some(idxs) = outputs.decl_by_fqname.get(root.as_str()) {
                    for &i in idxs {
                        set.insert(i);
                    }
                }
                let mut queue: std::collections::VecDeque<String> =
                    std::collections::VecDeque::from([root.clone()]);
                while let Some(parent) = queue.pop_front() {
                    let Some(children) = outputs.children_by_parent.get(parent.as_str()) else {
                        continue;
                    };
                    for &child_idx in children {
                        if !set.insert(child_idx) {
                            continue;
                        }
                        let child = outputs.builder.nodes[child_idx].borrow(py);
                        if child.kind == "module" {
                            queue.push_back(child.fqname.clone());
                        }
                    }
                }
                set
            });

        let mut out = Vec::new();
        for (node_idx, node_py) in outputs.builder.nodes.iter().enumerate() {
            // fqname-under (cheapest gate first when set — restricts
            // to a tiny candidate set out of all nodes).
            if let Some(set) = &fqname_under_indices {
                if !set.contains(&node_idx) {
                    continue;
                }
            }
            let node = node_py.borrow(py);
            // kind
            if let Some(k) = &kinds_set {
                if !k.contains(node.kind) {
                    continue;
                }
            }
            // paths (exact set match against absolute path)
            if let Some(p) = &paths_set {
                if !p.contains(node.path.as_str()) {
                    continue;
                }
            }
            // filename basename
            if let Some(f) = &filenames_set {
                let path = node.path.as_str();
                let basename = path
                    .rsplit_once(std::path::MAIN_SEPARATOR)
                    .map(|(_, name)| name)
                    .unwrap_or(path);
                if !f.contains(basename) {
                    continue;
                }
            }
            // simple name (last fqname segment)
            if let Some(s) = &simple_set {
                let fq = node.fqname.as_str();
                let simple = fq.rsplit_once('.').map(|(_, n)| n).unwrap_or(fq);
                if !s.contains(simple) {
                    continue;
                }
            }
            // flags mask
            if let Some(mask) = self.flags_mask {
                if self.flags_match_any {
                    if node.flags & mask == 0 {
                        continue;
                    }
                } else if node.flags & mask != mask {
                    continue;
                }
            }
            // fqname prefix
            if let Some(prefix) = &self.fqname_prefix {
                if !node.fqname.starts_with(prefix.as_str()) {
                    continue;
                }
            }
            // where_fqname: any literal-equality OR any regex match.
            // ``literals=Some(vec![])`` is the "matches-nothing"
            // sentinel — preserved here so empty calls behave like
            // ``where_name([])``.
            if self.fqname_literals.is_some() || self.fqname_regexes.is_some() {
                let fq = node.fqname.as_str();
                let lit_match = fqname_literals_set.as_ref().is_some_and(|s| s.contains(fq));
                let re_match = self
                    .fqname_regexes
                    .as_ref()
                    .is_some_and(|v| v.iter().any(|r| r.is_match(fq)));
                if !lit_match && !re_match {
                    continue;
                }
            }
            // path regex
            if let Some(re) = &path_regex {
                if !re.is_match(node.path.as_str()) {
                    continue;
                }
            }
            out.push(node_idx);
        }
        Ok(out)
    }
}

impl_query_methods!(no_first DeclQuery);

#[cfg(test)]
mod tests {
    //! Pure-rust tests for the identifier-prefilter helpers used by
    //! every per-file query loop. The chainable query types themselves
    //! depend on `Python<'_>` / `Py<ProjectContext>` and are covered
    //! end-to-end by the python suite.
    use super::*;

    // -- _is_ident_continue -----------------------------------------------

    #[test]
    fn is_ident_continue_recognizes_word_bytes() {
        for byte in b'a'..=b'z' {
            assert!(_is_ident_continue(byte), "{byte} should be ident continue");
        }
        for byte in b'A'..=b'Z' {
            assert!(_is_ident_continue(byte));
        }
        for byte in b'0'..=b'9' {
            assert!(_is_ident_continue(byte));
        }
        assert!(_is_ident_continue(b'_'));
    }

    #[test]
    fn is_ident_continue_rejects_punctuation_and_whitespace() {
        for byte in [b'.', b',', b' ', b'\t', b'\n', b'(', b')', b'-', b'#', b':'] {
            assert!(!_is_ident_continue(byte));
        }
    }

    // -- _contains_identifier ---------------------------------------------

    #[test]
    fn contains_identifier_finds_isolated_match() {
        assert!(_contains_identifier("foo", "foo"));
        assert!(_contains_identifier("call(foo)", "foo"));
        assert!(_contains_identifier(" foo ", "foo"));
        assert!(_contains_identifier("foo.bar", "foo"));
        assert!(_contains_identifier("a.foo", "foo"));
    }

    #[test]
    fn contains_identifier_rejects_substring_inside_other_word() {
        assert!(!_contains_identifier("foobar", "foo"));
        assert!(!_contains_identifier("xfoo", "foo"));
        assert!(!_contains_identifier("foo_bar", "foo"));
        assert!(!_contains_identifier("foo1", "foo"));
        // Underscore prefix is also an identifier continuation.
        assert!(!_contains_identifier("_foo", "foo"));
        // Digit prefix
        assert!(!_contains_identifier("1foo", "foo"));
    }

    #[test]
    fn contains_identifier_handles_match_at_boundaries() {
        assert!(_contains_identifier("foo()", "foo"));
        assert!(_contains_identifier("(foo)", "foo"));
        assert!(_contains_identifier("foo+bar", "foo"));
    }

    #[test]
    fn contains_identifier_empty_needle_returns_false() {
        // Empty needle would otherwise match everywhere — explicitly handled.
        assert!(!_contains_identifier("anything", ""));
        assert!(!_contains_identifier("", ""));
    }

    #[test]
    fn contains_identifier_empty_source_returns_false() {
        assert!(!_contains_identifier("", "foo"));
    }

    #[test]
    fn contains_identifier_multiple_occurrences_one_valid() {
        // The first ("xfoo") is invalid; the second (" foo") matches.
        assert!(_contains_identifier("xfoo foo", "foo"));
        // Both fail.
        assert!(!_contains_identifier("xfooy zfoow", "foo"));
    }

    #[test]
    fn contains_identifier_handles_overlapping_advance() {
        // Even when the first start position fails the boundary check,
        // the scan must advance and try subsequent occurrences.
        assert!(_contains_identifier("aaa a", "a"));
    }

    #[test]
    fn contains_identifier_unicode_source_does_not_panic() {
        // The function uses byte indexing for the boundary check. ASCII
        // identifier continuations are all single-byte, and the
        // multi-byte UTF-8 leading/continuation bytes all have the high
        // bit set (>= 0x80), so they fail `_is_ident_continue` and the
        // boundary holds — which makes `foo` adjacent to emoji a hit,
        // even though Python would treat the unicode chars as
        // identifier-continuation. The prefilter is approximate by
        // design (`ty_ide::references::contains_identifier` has the
        // same shape) — we exercise the byte path here to lock in the
        // documented behavior.
        assert!(_contains_identifier("héllo foo", "foo"));
        assert!(_contains_identifier("foo🙂bar", "foo"));
        // No needle present.
        assert!(!_contains_identifier("héllo", "foo"));
        // ASCII boundary still wins over surrounding unicode.
        assert!(_contains_identifier(" foo ", "foo"));
    }

    #[test]
    fn contains_identifier_multi_char_needle() {
        assert!(_contains_identifier("def my_func(): pass", "my_func"));
        assert!(!_contains_identifier("def my_function(): pass", "my_func"));
        assert!(!_contains_identifier("my_funcx", "my_func"));
    }

    // -- _contains_any_identifier -----------------------------------------

    #[test]
    fn contains_any_identifier_returns_true_on_first_hit() {
        assert!(_contains_any_identifier("call(foo)", &["bar", "foo"]));
        assert!(_contains_any_identifier("foo()", &["foo"]));
    }

    #[test]
    fn contains_any_identifier_empty_list_returns_false() {
        assert!(!_contains_any_identifier("anything", &[]));
        assert!(!_contains_any_identifier("", &[]));
    }

    #[test]
    fn contains_any_identifier_returns_false_when_no_match() {
        assert!(!_contains_any_identifier("this has none", &["foo", "bar"]));
    }

    #[test]
    fn contains_any_identifier_respects_boundaries() {
        // Same substring-inside-word rule as _contains_identifier.
        assert!(!_contains_any_identifier("foobar", &["foo", "bar"]));
        // But the standalone "foo" still hits if present.
        assert!(_contains_any_identifier("foobar foo", &["foo"]));
    }
}
