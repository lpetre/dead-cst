//! The chainable query DSL exposed to Python via `ProjectContext.query()`.
//! Result types (`DecoratorRef`/`ConstructionRef`/`CallRef`/`FactoryRef`),
//! the `QueryBuilder` entry point, and the per-stream `Query` builders.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyClass;
use ruff_db::files::File;
use rustc_hash::FxHashMap;
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
}

pub(crate) fn _extract_str_or_list(py: Python<'_>, obj: PyObject) -> PyResult<Vec<String>> {
    let bound = obj.bind(py);
    if let Ok(s) = bound.extract::<String>() {
        Ok(vec![s])
    } else {
        bound.extract::<Vec<String>>()
    }
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
        let ctx = self.ctx.borrow(py);
        let path_regex = self.path_regex.as_deref();
        let mut refs: Vec<Py<DecoratorRef>> = Vec::new();
        // Cache a snapshot of the build's node pool for arg materialization.
        // Keep the borrow alive for the duration of the loop so we don't
        // reborrow on every iteration.
        let outputs = ctx.materialized("DecoratorQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let kwarg_matchers = &self.kwarg_matchers;
        if let Some(owner_attrs) = &self.owner_attrs {
            let triples = if let Some(via) = &self.via_attr {
                ctx.find_handler_decorators_via(py, via, owner_attrs.clone(), path_regex)?
            } else {
                ctx.find_handler_decorators(py, owner_attrs.clone(), path_regex)?
            };
            for (owner_name, decorated, call_args) in triples {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                let args = args_to_py_vec(py, &call_args.args, nodes);
                let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated,
                        decorator_name: None,
                        decorator_owner: Some(owner_name),
                        decorator_via: self.via_attr.clone(),
                        args,
                        kwargs,
                    },
                )?);
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
            for (d, call_args) in decls {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                let args = args_to_py_vec(py, &call_args.args, nodes);
                let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated: d,
                        decorator_name: None,
                        decorator_owner: Some(owner_simple.clone()),
                        decorator_via: None,
                        args,
                        kwargs,
                    },
                )?);
            }
        } else if let Some(fqn) = &self.callee_fqn {
            let decls = ctx.find_decorated(py, fqn, path_regex)?;
            for (d, call_args) in decls {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                let args = args_to_py_vec(py, &call_args.args, nodes);
                let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated: d,
                        decorator_name: None,
                        decorator_owner: None,
                        decorator_via: None,
                        args,
                        kwargs,
                    },
                )?);
            }
        } else if let (Some(modules), Some(names)) = (&self.modules, &self.names) {
            let decls = ctx.find_decorated_decls(py, modules, names.clone(), path_regex)?;
            for (d, call_args) in decls {
                if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                    continue;
                }
                let args = args_to_py_vec(py, &call_args.args, nodes);
                let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
                refs.push(Py::new(
                    py,
                    DecoratorRef {
                        decorated: d,
                        decorator_name: None,
                        decorator_owner: None,
                        decorator_via: None,
                        args,
                        kwargs,
                    },
                )?);
            }
        } else {
            return Err(PyValueError::new_err(
                "DecoratorQuery requires one of: where_callee(...); \
                 where_module(...) + where_name(...); where_owner_attr(...); \
                 where_owner_attr_via(via, attrs); or in_decl(node) + where_name(...)",
            ));
        }
        Ok(refs)
    }
}

impl_query_methods!(with_first DecoratorQuery, DecoratorRef);

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
        let ctx = self.ctx.borrow(py);
        let path_regex = self.path_regex.as_deref();
        let mut refs: Vec<Py<ConstructionRef>> = Vec::new();
        let outputs = ctx.materialized("ConstructionQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        if let Some(fqn) = &self.class_fqn {
            let pairs = ctx.find_constructions(py, fqn, self.include_subclasses, path_regex)?;
            let cls_name = fqn.rsplit('.').next().unwrap_or("").to_string();
            for (d, call_args) in pairs {
                let args = args_to_py_vec(py, &call_args.args, nodes);
                let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
                refs.push(Py::new(
                    py,
                    ConstructionRef {
                        var: d,
                        class_name: cls_name.clone(),
                        args,
                        kwargs,
                    },
                )?);
            }
        } else if let (Some(modules), Some(names)) = (&self.modules, &self.names) {
            let triples =
                ctx.find_instance_constructions(py, modules, names.clone(), path_regex)?;
            for (var, name, call_args) in triples {
                let args = args_to_py_vec(py, &call_args.args, nodes);
                let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
                refs.push(Py::new(
                    py,
                    ConstructionRef {
                        var,
                        class_name: name,
                        args,
                        kwargs,
                    },
                )?);
            }
        } else {
            return Err(PyValueError::new_err(
                "ConstructionQuery requires either where_class(...) \
                 or where_module(...) + where_name(...)",
            ));
        }
        Ok(refs)
    }
}

impl_query_methods!(with_first ConstructionQuery, ConstructionRef);

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
        let outputs = ctx.materialized("CallQuery.collect")?;
        let nodes: &[Py<SymbolNode>] = &outputs.builder.nodes;
        let kwarg_matchers = &self.kwarg_matchers;
        let mut refs: Vec<Py<CallRef>> = Vec::new();
        for (owner_node, s, call_args) in triples {
            if !call_args_match_kwargs(&call_args, kwarg_matchers) {
                continue;
            }
            let args = args_to_py_vec(py, &call_args.args, nodes);
            let kwargs = kwargs_to_py_map(py, &call_args.kwargs, nodes);
            refs.push(Py::new(
                py,
                CallRef {
                    owner: owner_node,
                    string_arg: s,
                    args,
                    kwargs,
                },
            )?);
        }
        Ok(refs)
    }
}

impl_query_methods!(with_first CallQuery, CallRef);

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
        let ctx = self.ctx.borrow(py);
        let modules = self
            .modules
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FactoryQuery requires .of_module(...)"))?;
        let names = self
            .names
            .clone()
            .ok_or_else(|| PyValueError::new_err("FactoryQuery requires .where_name(...)"))?;
        let pairs = ctx.find_factory_decls(py, modules, names)?;
        pairs
            .into_iter()
            .map(|(decl, kinds)| Py::new(py, FactoryRef { decl, kinds }))
            .collect()
    }
}

impl_query_methods!(no_first FactoryQuery);

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
}

impl_query_methods!(with_first EdgeQuery, EdgeRef);

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
