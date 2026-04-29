//! f3dx-http Anthropic Messages client + PyO3 binding.
//!
//! Drop-in shape for `from anthropic import Anthropic`. Same constructor
//! surface (api_key, base_url, timeout). Same `messages.create()` /
//! streaming-events surface. Auth via `x-api-key` + `anthropic-version`,
//! NOT bearer.
//!
//! Streaming format differs from OpenAI: multiple `event:` types
//! (message_start, content_block_start, ping, content_block_delta,
//! content_block_stop, message_delta, message_stop). We pass each
//! parsed event through to Python as a dict with the original `type`
//! field intact, so user code can dispatch on it cleanly.

use crate::otel;
use crate::stream::{PyAnthropicStream, spawn_anthropic_pump};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
const USER_AGENT_STR: &str = concat!("f3dx-http/", env!("CARGO_PKG_VERSION"));

#[pyclass(name = "AnthropicClient", module = "f3dx._f3dx")]
pub struct PyAnthropicClient {
    client: Arc<Client>,
    runtime: Arc<Runtime>,
    base_url: Arc<String>,
}

#[pymethods]
impl PyAnthropicClient {
    #[new]
    #[pyo3(signature = (
        api_key = None,
        base_url = None,
        anthropic_version = None,
        timeout = 60.0,
        http2 = true,
    ))]
    fn new(
        api_key: Option<String>,
        base_url: Option<String>,
        anthropic_version: Option<String>,
        timeout: f64,
        http2: bool,
    ) -> PyResult<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .unwrap_or_else(|| "api-key-not-set".into());
        let base_url = base_url
            .or_else(|| std::env::var("ANTHROPIC_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.into())
            .trim_end_matches('/')
            .to_string();
        let anthropic_version =
            anthropic_version.unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.into());

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&api_key)
                .map_err(|e| PyValueError::new_err(format!("invalid api_key: {e}")))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&anthropic_version)
                .map_err(|e| PyValueError::new_err(format!("invalid anthropic_version: {e}")))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(USER_AGENT, HeaderValue::from_static(USER_AGENT_STR));

        let mut builder = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs_f64(timeout))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(32)
            .tcp_nodelay(true);
        if http2 {
            builder = builder.http2_prior_knowledge();
        }
        let client = builder
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("reqwest build failed: {e}")))?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("f3dx-http-anthropic")
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime build: {e}")))?;

        Ok(Self {
            client: Arc::new(client),
            runtime: Arc::new(runtime),
            base_url: Arc::new(base_url),
        })
    }

    /// Sync messages create (non-streaming). Pass dict matching Anthropic
    /// /v1/messages request shape; returns dict matching response shape.
    fn messages_create<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut req = parse_request(py, request)?;
        if let Value::Object(ref mut map) = req {
            map.remove("stream"); // ensure not streaming
        }
        let url = format!("{}/v1/messages", self.base_url);

        let model = req
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let temperature = req
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|x| x as f32);
        let top_p = req.get("top_p").and_then(|v| v.as_f64()).map(|x| x as f32);
        let max_tokens = req
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32);
        let mut span = otel::start_http_span("messages", "anthropic", &model);
        if let Some(s) = span.as_mut() {
            otel::add_request_params(s, temperature, top_p, max_tokens, Some(false));
        }

        let client = Arc::clone(&self.client);
        let runtime = Arc::clone(&self.runtime);
        let parsed: Result<Value, reqwest::Error> = py.detach(|| {
            runtime.block_on(async move {
                client
                    .post(&url)
                    .json(&req)
                    .send()
                    .await?
                    .error_for_status()?
                    .json::<Value>()
                    .await
            })
        });

        let parsed = match parsed {
            Ok(p) => p,
            Err(e) => {
                if let Some(s) = span.take() {
                    otel::finish_err(s, format!("{e}"));
                }
                return Err(PyRuntimeError::new_err(format!("f3dx-http: {e}")));
            }
        };
        if let Some(mut s) = span.take() {
            otel::add_anthropic_response(&mut s, &parsed);
            otel::finish_ok(s);
        }
        value_to_pydict(py, &parsed)
    }

    /// Sync messages stream. Returns an iterator of event dicts.
    /// Each event has a 'type' field per Anthropic streaming protocol:
    ///   message_start, content_block_start, ping, content_block_delta,
    ///   content_block_stop, message_delta, message_stop
    fn messages_create_stream<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAnthropicStream>> {
        let mut req = parse_request(py, request)?;
        if let Value::Object(ref mut map) = req {
            map.insert("stream".into(), Value::Bool(true));
        }
        let url = format!("{}/v1/messages", self.base_url);

        let model = req
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let temperature = req
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|x| x as f32);
        let top_p = req.get("top_p").and_then(|v| v.as_f64()).map(|x| x as f32);
        let max_tokens = req
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32);
        let mut span = otel::start_http_span("messages", "anthropic", &model);
        if let Some(s) = span.as_mut() {
            otel::add_request_params(s, temperature, top_p, max_tokens, Some(true));
        }

        let stream = spawn_anthropic_pump(
            Arc::clone(&self.client),
            Arc::clone(&self.runtime),
            url,
            req,
            span,
        )?;
        Py::new(py, stream)
    }

    #[getter]
    fn base_url(&self) -> String {
        (*self.base_url).clone()
    }

    fn __repr__(&self) -> String {
        format!("AnthropicClient(base_url={:?})", &*self.base_url)
    }
}

// ---------- helpers ----------

fn parse_request(py: Python<'_>, request: Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(s) = request.cast::<PyString>() {
        return serde_json::from_str(&s.to_string_lossy())
            .map_err(|e| PyValueError::new_err(format!("request JSON parse: {e}")));
    }
    let json_module = py.import("json")?;
    let dumps = json_module.getattr("dumps")?;
    let body_str: String = dumps.call1((request,))?.extract()?;
    serde_json::from_str(&body_str)
        .map_err(|e| PyValueError::new_err(format!("request JSON parse: {e}")))
}

pub(crate) fn value_to_pydict<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyDict>> {
    let s = serde_json::to_string(v)
        .map_err(|e| PyRuntimeError::new_err(format!("response serialise: {e}")))?;
    let json_module = py.import("json")?;
    let loads = json_module.getattr("loads")?;
    let py_obj = loads.call1((s,))?;
    py_obj
        .cast_into::<PyDict>()
        .map_err(|e| PyRuntimeError::new_err(format!("response not dict: {e}")))
}
