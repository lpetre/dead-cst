//! Thin pyo3 cdylib shim for `dead_cst._native`.
//!
//! All of the implementation lives in the `dead-cst-runtime` crate. This
//! shim exists only to define the Python module entry point and forward to
//! `dead_cst_runtime::register`. Keeping the `#[pymodule]` here (in a cdylib)
//! while the types live in `dead-cst-runtime` (an rlib/dylib) is what lets an
//! external native plugin dynamically link the *same* runtime the extension
//! module uses — one shared salsa/ty instance.

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_native")]
fn dead_cst_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dead_cst_runtime::register(m)
}
