//! f3dx-py - PyO3 bridge.
//!
//! Single Python module `_f3dx` that re-exports the registered
//! pyclasses + pyfunctions from f3dx-rt, f3dx-http, f3dx-trace.
//! Each sibling crate owns its own `register()` fn so the surface
//! grows without touching this file as new layers ship.

use pyo3::prelude::*;
use std::sync::Once;

static PANIC_HOOK_INIT: Once = Once::new();

fn install_panic_hook() {
    PANIC_HOOK_INIT.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let loc = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "unknown location".to_string());
            let msg = if let Some(s) = info.payload().downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            eprintln!("[f3dx] rust panic at {loc}: {msg}");
            prev(info);
        }));
    });
}

#[pymodule]
fn _f3dx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    install_panic_hook();
    f3dx_rt::register(m)?;
    f3dx_http::register(m)?;
    f3dx_trace::register(m)?;
    f3dx_mcp::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
