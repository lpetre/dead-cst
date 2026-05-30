//! ty-backed native graph builder for dead-cst (the shared runtime).
//!
//! This crate holds the entire native implementation. It is built as both
//! an `rlib` (statically linked into the `_native` cdylib for the default
//! self-contained wheel) and a `dylib` (the shared runtime that external
//! native plugins link against; see `native_plugins`).
//!
//! Architecture is governed by `CLAUDE.md`: ty does every piece of Python
//! semantics, ruff is only used where ty hasn't surfaced the structure we
//! need, and there is **no per-file cache** (ty's Salsa db is the cache).
//!
//! The pipeline is one method (`Project.build()`) returning one
//! project-wide `NativeGraph`:
//!
//! 1. **Phase 1 — decls**. For every project file, iterate every
//!    binding in the file's global scope via
//!    `UseDefMap::all_definitions_with_usage`, minting a node per
//!    binding (including each name brought in by `from foo import *`).
//! 2. **Phase 2 — chain**. For every module node, emit the submodule
//!    edge to its parent. For every import-kind binding, resolve the
//!    upstream target and emit `alias_node → upstream_node`.
//! 3. **Phase 3 — references**. For every Definition that owns an
//!    expression, walk the contained `Name`s and resolve each to its
//!    reaching def.

#![allow(clippy::useless_conversion)]

mod builder;
mod file_payload;
mod file_ref_edges;
mod graph;
mod helpers;
mod ingest;
mod io;
pub mod native_plugins;
mod progress;
mod project;
mod query;

use pyo3::prelude::*;

use crate::builder::{
    AddEdge, AddEdgeByIdx, AddEntrypoint, AddEntrypointByIdx, AddNode, AddNodeByIdx, CollectedOps,
};
use crate::graph::{EdgeFlags, Import, NativeGraph, NodeFlags, SymbolNode};
use crate::helpers::{ArgLiteral, ArgNodeRef, ArgOpaque, NodeAttrs};
use crate::io::{read_graph, write_graph, GraphMetadata};
use crate::native_plugins::{
    _builtin_native_plugin, _main_block_run_count, _reset_main_block_run_count,
    _reset_server_config_run_count, _server_config_run_count, load_native_plugins, NativePlugin,
};
use crate::progress::ProgressHandle;
use crate::project::{ChangeEvent, Project, ProjectContext};
use crate::query::{
    CallIdxRef, CallQuery, ClassQuery, ConstructionIdxRef, ConstructionQuery, DeclQuery,
    DeclarationsQuery, DecoratorIdxRef, DecoratorQuery, EdgeQuery, FactoryIdxRef, FactoryQuery,
    ImportQuery, LiteralListQuery, MainBlockQuery, ModuleQuery, QueryBuilder, SubclassQuery,
    TraverseQuery,
};

/// Module-level alias for ``ctx.query()`` — exists for the ergonomic
/// ``from dead_cst import _native as native; native.query(ctx)...``
/// idiom that the plugins rely on.
#[pyfunction(name = "query")]
fn query_fn(slf: Py<ProjectContext>, _py: Python<'_>) -> QueryBuilder {
    QueryBuilder { ctx: slf }
}

/// Register every pyclass + function on the `_native` module object. The
/// `#[pymodule]` entry point lives in the thin `dead-cst-native` cdylib
/// shim, which simply forwards to this.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Import>()?;
    m.add_class::<SymbolNode>()?;
    m.add_class::<NativeGraph>()?;
    m.add_class::<Project>()?;
    m.add_class::<ProjectContext>()?;
    m.add_class::<ChangeEvent>()?;
    m.add_class::<AddEdge>()?;
    m.add_class::<AddEdgeByIdx>()?;
    m.add_class::<AddEntrypoint>()?;
    m.add_class::<AddEntrypointByIdx>()?;
    m.add_class::<AddNode>()?;
    m.add_class::<AddNodeByIdx>()?;
    m.add_class::<CollectedOps>()?;
    m.add_class::<NativePlugin>()?;
    m.add_class::<NodeFlags>()?;
    m.add_class::<EdgeFlags>()?;
    m.add_class::<DecoratorIdxRef>()?;
    m.add_class::<ConstructionIdxRef>()?;
    m.add_class::<CallIdxRef>()?;
    m.add_class::<ArgLiteral>()?;
    m.add_class::<ArgNodeRef>()?;
    m.add_class::<ArgOpaque>()?;
    m.add_class::<NodeAttrs>()?;
    m.add_class::<QueryBuilder>()?;
    m.add_class::<DecoratorQuery>()?;
    m.add_class::<ConstructionQuery>()?;
    m.add_class::<CallQuery>()?;
    m.add_class::<SubclassQuery>()?;
    m.add_class::<ImportQuery>()?;
    m.add_class::<ModuleQuery>()?;
    m.add_class::<ClassQuery>()?;
    m.add_class::<FactoryQuery>()?;
    m.add_class::<FactoryIdxRef>()?;
    m.add_class::<EdgeQuery>()?;
    m.add_class::<DeclQuery>()?;
    m.add_class::<DeclarationsQuery>()?;
    m.add_class::<MainBlockQuery>()?;
    m.add_class::<LiteralListQuery>()?;
    m.add_class::<TraverseQuery>()?;
    m.add_class::<GraphMetadata>()?;
    m.add_class::<ProgressHandle>()?;
    m.add_function(wrap_pyfunction!(query_fn, m)?)?;
    m.add_function(wrap_pyfunction!(write_graph, m)?)?;
    m.add_function(wrap_pyfunction!(read_graph, m)?)?;
    m.add_function(wrap_pyfunction!(_main_block_run_count, m)?)?;
    m.add_function(wrap_pyfunction!(_reset_main_block_run_count, m)?)?;
    m.add_function(wrap_pyfunction!(_server_config_run_count, m)?)?;
    m.add_function(wrap_pyfunction!(_reset_server_config_run_count, m)?)?;
    m.add_function(wrap_pyfunction!(load_native_plugins, m)?)?;
    m.add_function(wrap_pyfunction!(_builtin_native_plugin, m)?)?;
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
