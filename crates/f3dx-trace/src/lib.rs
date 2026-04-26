//! f3dx-trace — OpenTelemetry span emission for f3dx.
//!
//! Phase F: wire OTel + OTLP/HTTP+protobuf exporter so every f3dx run
//! emits proper gen_ai.* semconv spans. Configurable via the Python
//! `f3dx.configure_otel(endpoint, headers, service_name, stdout)` function.
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
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock as StdOnceLock};
use std::time::Duration;

/// Path of the JSONL trace sink, configured via `configure_traces(path)`.
/// Phase G V0: append-only JSON-Lines; one row per AgentRuntime.run.
/// Polars/DuckDB users scan via pl.scan_ndjson / duckdb.read_json.
/// V0.1 upgrades to Arrow/parquet.
static JSONL_SINK: StdOnceLock<Mutex<Option<PathBuf>>> = StdOnceLock::new();

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
#[pyo3(signature = (endpoint = None, headers = None, service_name = String::from("f3dx"), stdout = false))]
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
        KeyValue::new("f3dx.runtime", "rust"),
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

/// Configure a JSONL trace sink. Each AgentRuntime.run appends one row.
/// Path is opened append-only on each emit; safe under concurrent runs.
/// Set path=None to disable.
#[pyfunction]
#[pyo3(signature = (path = None))]
fn configure_traces(path: Option<String>) -> PyResult<()> {
    let slot = JSONL_SINK.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().expect("trace sink mutex poisoned");
    *guard = path.map(PathBuf::from);
    Ok(())
}

/// Append one row to the configured JSONL sink (no-op when not configured).
/// Called from f3dx-rt's AgentRuntime.run on completion.
pub fn emit_trace_row(row: &Value) {
    let Some(slot) = JSONL_SINK.get() else { return };
    let path = match slot.lock() {
        Ok(g) => match g.as_ref() {
            Some(p) => p.clone(),
            None => return,
        },
        Err(_) => return,
    };
    let line = match serde_json::to_string(row) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut f = match OpenOptions::new().append(true).create(true).open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = writeln!(f, "{}", line);
}

#[pyfunction]
fn trace_sink_path() -> Option<String> {
    let slot = JSONL_SINK.get()?;
    let guard = slot.lock().ok()?;
    guard.as_ref().map(|p| p.to_string_lossy().to_string())
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

/// Register f3dx-trace's pyfunctions into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(configure_otel, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_otel, m)?)?;
    m.add_function(wrap_pyfunction!(emit_test_span, m)?)?;
    m.add_function(wrap_pyfunction!(configure_traces, m)?)?;
    m.add_function(wrap_pyfunction!(trace_sink_path, m)?)?;
    Ok(())
}
