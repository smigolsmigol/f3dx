//! agx-trace — OpenTelemetry span emission for agx.
//!
//! Phase F: wire OTel + OTLP/HTTP+protobuf exporter so every agx run
//! emits proper gen_ai.* semconv spans. Configurable via the Python
//! `agx.configure_otel(endpoint, headers, service_name, stdout)` function.
//!
//! Two backend modes:
//!   1. OTLP/HTTP -> any OTel-compatible collector incl. Pydantic Logfire
//!      (logfire-api.pydantic.dev/v1/traces with Authorization header)
//!   2. stdout -> debug printing for smoke tests
//!
//! agx-rt and agx-http call `agx_trace::tracer()` to get the configured
//! tracer. If no tracer is configured, returns None and the call sites
//! skip span emission (zero overhead).

use once_cell::sync::OnceCell;
use opentelemetry::global;
use opentelemetry::trace::{Tracer as _, TracerProvider as _};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::{Tracer, TracerProvider};
use opentelemetry_sdk::Resource;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

static TRACER: OnceCell<Tracer> = OnceCell::new();
static PROVIDER: OnceCell<Mutex<Option<TracerProvider>>> = OnceCell::new();

/// Configure OTel emission. Idempotent at the level of "first call wins";
/// subsequent calls are no-ops (TracerProvider/SDK setup is global state).
///
/// Args:
///   endpoint:     OTLP/HTTP endpoint URL (e.g. https://logfire-api.pydantic.dev/v1/traces)
///   headers:      dict of header strings (e.g. {"Authorization": "Bearer <token>"})
///   service_name: OpenTelemetry service.name resource attribute
///   stdout:       if True, also export to stdout for debugging
#[pyfunction]
#[pyo3(signature = (endpoint = None, headers = None, service_name = String::from("agx"), stdout = false))]
fn configure_otel(
    endpoint: Option<String>,
    headers: Option<HashMap<String, String>>,
    service_name: String,
    stdout: bool,
) -> PyResult<()> {
    if TRACER.get().is_some() {
        return Ok(());
    }

    let resource = Resource::new(vec![
        KeyValue::new("service.name", service_name.clone()),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        KeyValue::new("agx.runtime", "rust"),
    ]);

    let mut builder = TracerProvider::builder().with_config(
        opentelemetry_sdk::trace::Config::default().with_resource(resource),
    );

    if let Some(endpoint_url) = endpoint {
        let exporter = SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint_url)
            .with_timeout(Duration::from_secs(30))
            .with_headers(headers.unwrap_or_default())
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("OTLP exporter build: {e}")))?;
        builder = builder.with_batch_exporter(exporter, runtime::Tokio);
    }

    if stdout {
        let stdout_exporter = opentelemetry_stdout::SpanExporter::default();
        builder = builder.with_simple_exporter(stdout_exporter);
    }

    let provider = builder.build();
    let tracer = provider.tracer(service_name);
    let _ = TRACER.set(tracer);
    let _ = PROVIDER.set(Mutex::new(Some(provider.clone())));
    global::set_tracer_provider(provider);
    Ok(())
}

/// Force-flush + shutdown all configured exporters. Call on process exit.
#[pyfunction]
fn shutdown_otel() -> PyResult<()> {
    if let Some(slot) = PROVIDER.get() {
        if let Some(provider) = slot.lock().expect("provider mutex poisoned").take() {
            let _ = provider.force_flush();
            let _ = provider.shutdown();
        }
    }
    Ok(())
}

/// Borrow the global tracer if configured. None when not configured;
/// callers should treat None as "skip span emission".
pub fn tracer() -> Option<&'static Tracer> {
    TRACER.get()
}

/// Smoke-test entry: emit a single test span to verify the configured
/// exporter works end-to-end. Returns the trace_id as a hex string.
#[pyfunction]
fn emit_test_span<'py>(py: Python<'py>, name: String) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let Some(tracer) = tracer() else {
        out.set_item("ok", false)?;
        out.set_item("reason", "tracer not configured; call configure_otel first")?;
        return Ok(out);
    };
    use opentelemetry::trace::{Span, TraceContextExt};
    let mut span = tracer.start(name.clone());
    let trace_id = span.span_context().trace_id().to_string();
    let span_id = span.span_context().span_id().to_string();
    span.end();
    out.set_item("ok", true)?;
    out.set_item("trace_id", trace_id)?;
    out.set_item("span_id", span_id)?;
    out.set_item("name", name)?;
    Ok(out)
}

/// Register agx-trace's pyfunctions into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(configure_otel, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_otel, m)?)?;
    m.add_function(wrap_pyfunction!(emit_test_span, m)?)?;
    Ok(())
}
