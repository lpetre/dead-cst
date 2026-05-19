//! The chainable query DSL exposed to Python via `ProjectContext.query()`.
//! Result types (`DecoratorRef`/`ConstructionRef`/`CallRef`/`FactoryRef`),
//! the `QueryBuilder` entry point, and the per-stream `Query` builders.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyClass;
use ruff_db::files::File;
use ty_project::Db as ProjectDb;

use crate::builder::not_materialized;
use crate::graph::{MainBlock, SymbolNode};
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
    pub(crate) kwargs: HashMap<String, Py<PyAny>>,
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
#[pyclass(frozen, get_all)]
pub(crate) struct ConstructionRef {
    pub(crate) var: Py<SymbolNode>,
    pub(crate) class_name: String,
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
    pub(crate) kwargs: HashMap<String, Py<PyAny>>,
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
/// :meth:`decorators`, :meth:`constructions`, or :meth:`calls`.
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

    // ----- Point lookups (no filter chain) ------------------------------

    /// Look up a module's synthetic node by dotted fqname. Mirrors
    /// :meth:`ProjectContext.find_module`.
    fn module(&self, py: Python<'_>, fqname: &str) -> PyResult<Option<Py<SymbolNode>>> {
        self.ctx.borrow(py).find_module(py, fqname)
    }
    /// All top-level declarations bound to the given dotted fqname.
    /// Mirrors :meth:`ProjectContext.find_declarations`.
    fn declarations(&self, py: Python<'_>, fqname: &str) -> PyResult<Vec<Py<SymbolNode>>> {
        self.ctx.borrow(py).find_declarations(py, fqname)
    }
    /// Every top-level declaration node of the named module.
    /// Mirrors :meth:`ProjectContext.find_module_top_level_decls`.
    fn module_top_level_decls(
        &self,
        py: Python<'_>,
        fqname: &str,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        self.ctx.borrow(py).find_module_top_level_decls(py, fqname)
    }
    /// The exported names listed in a module's ``__all__`` (when
    /// statically resolvable), or ``None`` when the module has no
    /// ``__all__``.
    /// Mirrors :meth:`ProjectContext.find_module_dunder_all_exports`.
    fn module_dunder_all_exports(
        &self,
        py: Python<'_>,
        fqname: &str,
    ) -> PyResult<Option<Vec<Py<SymbolNode>>>> {
        self.ctx
            .borrow(py)
            .find_module_dunder_all_exports(py, fqname)
    }
    /// Read the literal-list value of a top-level variable assignment
    /// (``X = ["a", "b"]``) and return the entries as strings.
    /// Returns ``None`` when the variable isn't a list / tuple of
    /// string literals.
    /// Mirrors :meth:`ProjectContext.find_literal_list_entries`.
    fn literal_list_entries(&self, py: Python<'_>, var_fqn: &str) -> PyResult<Option<Vec<String>>> {
        self.ctx.borrow(py).find_literal_list_entries(var_fqn)
    }
    /// All top-level ``__dunder__`` declarations across the project.
    /// Mirrors :meth:`ProjectContext.find_module_dunders`.
    fn module_dunders(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        self.ctx.borrow(py).find_module_dunders(py)
    }
    /// Every ``if __name__ == "__main__":`` block in the project,
    /// paired with the module and the decls inside.
    /// Mirrors :meth:`ProjectContext.find_main_blocks`.
    fn main_blocks(&self, py: Python<'_>) -> PyResult<Vec<MainBlock>> {
        self.ctx.borrow(py).find_main_blocks(py)
    }
    /// Match nodes against pre-classified path / fqname specs without
    /// crossing the FFI boundary per node.
    /// Mirrors :meth:`ProjectContext.find_nodes_matching_specs`.
    fn nodes_matching_specs(
        &self,
        py: Python<'_>,
        project_root: &str,
        regexes: Vec<String>,
        str_specs: Vec<String>,
        abs_paths: Vec<String>,
    ) -> PyResult<Vec<Py<SymbolNode>>> {
        self.ctx.borrow(py).find_nodes_matching_specs(
            py,
            project_root,
            regexes,
            str_specs,
            abs_paths,
        )
    }
    /// Comments matching ``pattern`` paired with the next
    /// declaration. Mirrors :meth:`ProjectContext.find_comment_patterns`.
    #[allow(clippy::type_complexity)]
    fn comment_patterns(
        &self,
        py: Python<'_>,
        pattern: &str,
    ) -> PyResult<Vec<(Py<SymbolNode>, String)>> {
        self.ctx.borrow(py).find_comment_patterns(py, pattern)
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

/// Find decorated top-level functions / classes. Pick exactly one of:
/// * ``where_module(m).where_name(n)`` — ``@m.x`` / ``@x`` where ``x``
///   is imported from ``m``.
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
    pub(crate) module: Option<String>,
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
            module: None,
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
    fn where_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
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
        let outputs_borrow = ctx.outputs.borrow();
        let outputs = outputs_borrow
            .as_ref()
            .ok_or_else(|| not_materialized("DecoratorQuery.collect"))?;
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
        } else if let (Some(module), Some(names)) = (&self.module, &self.names) {
            let decls = ctx.find_decorated_decls(py, module, names.clone(), path_regex)?;
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

    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<DecoratorRef>>> {
        Ok(self.collect(py)?.into_iter().next())
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

/// Find module-scope ``<var> = <Ctor>(...)`` sites. Pick exactly one
/// of ``where_module + where_name`` or
/// ``where_class(fqn, include_subclasses=...)``.
#[pyclass(unsendable)]
pub(crate) struct ConstructionQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) module: Option<String>,
    pub(crate) names: Option<Vec<String>>,
    pub(crate) class_fqn: Option<String>,
    pub(crate) include_subclasses: bool,
    pub(crate) path_regex: Option<String>,
}

impl ConstructionQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            module: None,
            names: None,
            class_fqn: None,
            include_subclasses: false,
            path_regex: None,
        }
    }
}

#[pymethods]
impl ConstructionQuery {
    fn where_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
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
        if let Some(fqn) = &self.class_fqn {
            let decls = ctx.find_constructions(py, fqn, self.include_subclasses, path_regex)?;
            let cls_name = fqn.rsplit('.').next().unwrap_or("").to_string();
            for d in decls {
                refs.push(Py::new(
                    py,
                    ConstructionRef {
                        var: d,
                        class_name: cls_name.clone(),
                    },
                )?);
            }
        } else if let (Some(module), Some(names)) = (&self.module, &self.names) {
            let pairs = ctx.find_instance_constructions(py, module, names.clone(), path_regex)?;
            for (var, name) in pairs {
                refs.push(Py::new(
                    py,
                    ConstructionRef {
                        var,
                        class_name: name,
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

    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<ConstructionRef>>> {
        Ok(self.collect(py)?.into_iter().next())
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

/// Find call sites whose positional string-literal at the configured
/// index is captured. :meth:`string_arg_at` is required. Pick one of:
/// * ``where_module(m).where_name(n)`` — call to ``n`` imported from
///   ``m``.
/// * ``where_owner(o).where_attr(a)`` — ``<o>.<a>(...)`` literal
///   receiver match.
/// * ``where_attr(a)`` — ``<expr>.<a>(...)`` any receiver.
#[pyclass(unsendable)]
pub(crate) struct CallQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) module: Option<String>,
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
            module: None,
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
    fn where_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
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
        let triples = if let (Some(module), Some(name)) = (&self.module, &self.name) {
            ctx.find_calls_to_imported(py, module, name, arg_index, path_regex)?
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
        let outputs_borrow = ctx.outputs.borrow();
        let outputs = outputs_borrow
            .as_ref()
            .ok_or_else(|| not_materialized("CallQuery.collect"))?;
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

    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<CallRef>>> {
        Ok(self.collect(py)?.into_iter().next())
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
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
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

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
}

#[pymethods]
impl ImportQuery {
    fn of<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
        slf
    }
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<SymbolNode>>> {
        let ctx = self.ctx.borrow(py);
        let module = self
            .module
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("ImportQuery requires .of(module)"))?;
        ctx.find_imports_of(py, module)
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

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
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}

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
#[pyclass(unsendable)]
pub(crate) struct FactoryQuery {
    pub(crate) ctx: Py<ProjectContext>,
    pub(crate) module: Option<String>,
    pub(crate) names: Option<Vec<String>>,
}

impl FactoryQuery {
    fn new(ctx: Py<ProjectContext>) -> Self {
        Self {
            ctx,
            module: None,
            names: None,
        }
    }
}

#[pymethods]
impl FactoryQuery {
    fn of_module<'py>(mut slf: PyRefMut<'py, Self>, module: String) -> PyRefMut<'py, Self> {
        slf.module = Some(module);
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
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<FactoryRef>>> {
        let ctx = self.ctx.borrow(py);
        let module = self
            .module
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("FactoryQuery requires .of_module(...)"))?;
        let names = self
            .names
            .clone()
            .ok_or_else(|| PyValueError::new_err("FactoryQuery requires .where_name(...)"))?;
        let pairs = ctx.find_factory_decls(py, module, names)?;
        pairs
            .into_iter()
            .map(|(decl, kinds)| Py::new(py, FactoryRef { decl, kinds }))
            .collect()
    }
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.collect(py)?.len())
    }
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        _to_iter(py, self.collect(py)?)
    }
}
