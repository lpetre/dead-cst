//! ty-backed native graph builder for dead-cst.
//!
//! Architecture is governed by the crate's `CLAUDE.md`: ty does every
//! piece of Python semantics, ruff is only used where ty hasn't surfaced
//! the structure we need, and there is **no per-file cache** (ty's
//! Salsa db is the cache).
//!
//! The pipeline is one method (`Project.build()`) returning one
//! project-wide `NativeGraph`:
//!
//! 1. **Phase 1 — decls**. For every project file, iterate every
//!    binding in the file's global scope via
//!    `UseDefMap::all_definitions_with_usage`, minting a node per
//!    binding (including each name brought in by `from foo import *`).
//!    Each node lands in a global `(File, target_range) → node_idx`
//!    index so cross-file edges can find it later.
//! 2. **Phase 2 — chain**. For every module node, emit the submodule
//!    edge to its parent. For every import-kind binding, resolve the
//!    upstream target via `ty_module_resolver::resolve_module` and
//!    emit `alias_node → upstream_node`; lazily mint a module-only
//!    node for any target outside the project (stdlib / site-packages).
//! 3. **Phase 3 — references**. For every Definition that owns an
//!    expression (function body, class body, assignment value,
//!    annotation), walk the contained `Name`s and resolve each to its
//!    reaching def via `visible_ancestor_scopes` +
//!    `end_of_scope_symbol_bindings` (Principle 2 — the local alias
//!    is the target, not the upstream definition). Module-level
//!    non-definition statements attribute to the module.

#![allow(clippy::useless_conversion)]

mod builder;
mod graph;
mod helpers;
mod ingest;
mod project;
mod query;

use pyo3::prelude::*;

use crate::builder::{AddEdge, AddEntrypoint, AddNode};
use crate::graph::{EdgeFlags, Import, NativeGraph, NodeFlags, SymbolNode, SyntheticTag};
use crate::project::{Project, ProjectContext};
use crate::query::{
    CallQuery, CallRef, ClassQuery, ConstructionQuery, ConstructionRef, DecoratorQuery,
    DecoratorRef, FactoryQuery, FactoryRef, ImportQuery, QueryBuilder, SubclassQuery,
};

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// Module-level alias for ``ctx.query()`` — exists for the ergonomic
/// ``from dead_cst import _native as native; native.query(ctx)...``
/// idiom that the plugins rely on.
#[pyfunction(name = "query")]
fn query_fn(slf: Py<ProjectContext>, _py: Python<'_>) -> QueryBuilder {
    QueryBuilder { ctx: slf }
}

#[pymodule]
#[pyo3(name = "_native")]
fn dead_cst_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Import>()?;
    m.add_class::<SyntheticTag>()?;
    m.add_class::<SymbolNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<Project>()?;
    m.add_class::<ProjectContext>()?;
    m.add_class::<AddEdge>()?;
    m.add_class::<AddEntrypoint>()?;
    m.add_class::<AddNode>()?;
    m.add_class::<NodeFlags>()?;
    m.add_class::<EdgeFlags>()?;
    m.add_class::<DecoratorRef>()?;
    m.add_class::<ConstructionRef>()?;
    m.add_class::<CallRef>()?;
    m.add_class::<QueryBuilder>()?;
    m.add_class::<DecoratorQuery>()?;
    m.add_class::<ConstructionQuery>()?;
    m.add_class::<CallQuery>()?;
    m.add_class::<SubclassQuery>()?;
    m.add_class::<ImportQuery>()?;
    m.add_class::<ClassQuery>()?;
    m.add_class::<FactoryQuery>()?;
    m.add_class::<FactoryRef>()?;
    m.add_function(wrap_pyfunction!(query_fn, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::helpers::{parse_noqa_tail, NoqaKind};

    #[test]
    fn parse_noqa_tail_multibyte_prefix_does_not_panic() {
        // Regression: byte-slicing `trimmed[..4]` panicked when the
        // comment started with a multi-byte UTF-8 char (e.g. the
        // 3-byte `─` box-drawing char in section banners).
        assert_eq!(parse_noqa_tail("── Top-level model (dataclass) ──"), None,);
        assert_eq!(parse_noqa_tail("─"), None);
        assert_eq!(parse_noqa_tail("héllo"), None);
        assert_eq!(parse_noqa_tail("🙂🙂"), None);
    }

    #[test]
    fn parse_noqa_tail_short_content_returns_none() {
        assert_eq!(parse_noqa_tail(""), None);
        assert_eq!(parse_noqa_tail("no"), None);
        assert_eq!(parse_noqa_tail("noq"), None);
    }

    #[test]
    fn parse_noqa_tail_recognizes_bare_directive() {
        assert_eq!(parse_noqa_tail(" noqa"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("NOQA"), Some(NoqaKind::Bare));
        assert_eq!(parse_noqa_tail("noqa: "), Some(NoqaKind::Bare));
    }

    #[test]
    fn parse_noqa_tail_recognizes_f401() {
        assert_eq!(parse_noqa_tail("noqa: F401"), Some(NoqaKind::F401Present));
        assert_eq!(
            parse_noqa_tail("noqa: E501, F401"),
            Some(NoqaKind::F401Present),
        );
        assert_eq!(parse_noqa_tail("noqa: E501"), Some(NoqaKind::OtherOnly));
    }
}
