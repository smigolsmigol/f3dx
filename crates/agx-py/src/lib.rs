//! agx-py — PyO3 bridge.
//!
//! Single Python module `_agx` that re-exports the registered
//! pyclasses + pyfunctions from agx-rt, agx-http, agx-trace.
//! Each sibling crate owns its own `register()` fn so the surface
//! grows without touching this file as new layers ship.

use pyo3::prelude::*;

#[pymodule]
fn _agx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    agx_rt::register(m)?;
    agx_http::register(m)?;
    agx_trace::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
