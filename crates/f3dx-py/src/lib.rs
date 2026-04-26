//! f3dx-py — PyO3 bridge.
//!
//! Single Python module `_f3dx` that re-exports the registered
//! pyclasses + pyfunctions from f3dx-rt, f3dx-http, f3dx-trace.
//! Each sibling crate owns its own `register()` fn so the surface
//! grows without touching this file as new layers ship.

use pyo3::prelude::*;

#[pymodule]
fn _f3dx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    f3dx_rt::register(m)?;
    f3dx_http::register(m)?;
    f3dx_trace::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
