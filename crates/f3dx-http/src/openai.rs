//! f3dx-http OpenAI async client + PyO3 binding.
//!
//! Async reqwest::Client + per-instance tokio Runtime. The Python surface
//! is sync (uses runtime.block_on internally) so callers don't have to
//! think about asyncio. Streaming returns a sync iterator backed by a
//! std::sync::mpsc channel; the Python iterator releases the GIL while
//! waiting on recv().

use crate::otel;
use crate::request::ChatCompletionRequest;
use crate::response::ChatCompletionResponse;
use crate::stream::{PyAssembledStream, PyChatCompletionStream};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const USER_AGENT_STR: &str = concat!("f3dx-http/", env!("CARGO_PKG_VERSION"));

#[pyclass(name = "OpenAIClient", module = "f3dx._f3dx")]
pub struct PyOpenAIClient {
    client: Arc<Client>,
    runtime: Arc<Runtime>,
    base_url: Arc<String>,
}

#[pymethods]
impl PyOpenAIClient {
    #[new]
    #[pyo3(signature = (
        api_key = None,
        base_url = None,
        timeout = 60.0,
        http2 = true,
    ))]
    fn new(
        api_key: Option<String>,
        base_url: Option<String>,
        timeout: f64,
        http2: bool,
    ) -> PyResult<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_else(|| "api-key-not-set".into());
        let base_url = base_url
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.into())
            .trim_end_matches('/')
            .to_string();

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| PyValueError::new_err(format!("invalid api_key for header: {e}")))?,
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
            .thread_name("f3dx-http")
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime build: {e}")))?;

        Ok(Self {
            client: Arc::new(client),
            runtime: Arc::new(runtime),
            base_url: Arc::new(base_url),
        })
    }

    /// Sync chat completion (non-streaming). Pass a dict matching the OpenAI
    /// request shape; returns a dict matching the OpenAI response shape.
    fn chat_completions_create<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let req = parse_request(py, request)?;
        let url = format!("{}/chat/completions", self.base_url);

        let mut span = otel::start_http_span("chat", "openai", &req.model);
        if let Some(s) = span.as_mut() {
            otel::add_request_params(s, req.temperature, req.top_p, req.max_tokens, Some(false));
        }

        let client = Arc::clone(&self.client);
        let runtime = Arc::clone(&self.runtime);
        let parsed: Result<ChatCompletionResponse, reqwest::Error> = py.allow_threads(|| {
            runtime.block_on(async move {
                client
                    .post(&url)
                    .json(&req)
                    .send()
                    .await?
                    .error_for_status()?
                    .json::<ChatCompletionResponse>()
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
        let out_str = serde_json::to_string(&parsed)
            .map_err(|e| PyRuntimeError::new_err(format!("response serialise: {e}")))?;
        if let Some(mut s) = span.take() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&out_str) {
                otel::add_openai_response(&mut s, &v);
            }
            otel::finish_ok(s);
        }
        let json_module = py.import("json")?;
        let loads = json_module.getattr("loads")?;
        let py_obj = loads.call1((out_str,))?;
        py_obj
            .downcast_into::<PyDict>()
            .map_err(|e| PyRuntimeError::new_err(format!("response not dict: {e}")))
    }

    /// Sync chat completion stream. Returns an iterator of chunk dicts.
    /// Each iteration releases the GIL while waiting for the next chunk
    /// from the SSE stream — concurrent Python threads can make progress.
    fn chat_completions_create_stream<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
    ) -> PyResult<Py<PyChatCompletionStream>> {
        let mut req = parse_request(py, request)?;
        req.stream = Some(true);
        // Auto-enable usage in the stream so the closing chunk carries
        // gen_ai.usage.* tokens. User can override by passing their own
        // stream_options. OpenAI-compatible endpoints that don't know the
        // flag will ignore it.
        req.extra
            .entry("stream_options".to_string())
            .or_insert_with(|| serde_json::json!({"include_usage": true}));

        let mut span = otel::start_http_span("chat", "openai", &req.model);
        if let Some(s) = span.as_mut() {
            otel::add_request_params(s, req.temperature, req.top_p, req.max_tokens, Some(true));
        }

        let url = format!("{}/chat/completions", self.base_url);
        PyChatCompletionStream::start(
            Arc::clone(&self.client),
            Arc::clone(&self.runtime),
            url,
            req,
            span,
        )
        .map(|s| Py::new(py, s))
        .and_then(|r| r)
    }

    /// Sync chat completion stream with Rust-side tool-call reassembly.
    /// Yields three event types instead of raw chunks:
    ///   {"type": "delta_content", "content": "..."}
    ///   {"type": "tool_call", "id": "...", "name": "...",
    ///    "arguments": {parsed dict}, "index": N}
    ///   {"type": "done", "finish_reason": "stop"}
    /// User code never has to accumulate arguments fragments or json.loads
    /// the assembled string — f3dx does it Rust-side.
    ///
    /// validate_json=True: accumulate delta.content fragments, attempt
    /// json.loads at terminal, emit one extra event before done:
    ///   {"type": "validated_output", "data": <parsed dict>}      on success
    ///   {"type": "validation_error", "raw": "...", "error": "..."}  on parse fail
    /// Use this with response_format={"type": "json_object"} to skip the
    /// "accumulate content + json.loads at end" boilerplate Python-side.
    /// When `output_schema` is provided (a JSON Schema dict), the terminal
    /// `validated_output` event is only emitted if the parsed JSON satisfies
    /// the schema; otherwise `validation_error` carries the schema violation.
    #[pyo3(signature = (request, validate_json = false, output_schema = None))]
    fn chat_completions_create_stream_assembled<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
        validate_json: bool,
        output_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAssembledStream>> {
        let mut req = parse_request(py, request)?;
        req.stream = Some(true);
        req.extra
            .entry("stream_options".to_string())
            .or_insert_with(|| serde_json::json!({"include_usage": true}));

        let schema_value = match output_schema {
            None => None,
            Some(obj) => {
                let json_module = py.import("json")?;
                let dumps = json_module.getattr("dumps")?;
                let s: String = dumps.call1((obj,))?.extract()?;
                Some(serde_json::from_str::<serde_json::Value>(&s).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "output_schema not JSON-serializable: {e}"
                    ))
                })?)
            }
        };

        let mut span = otel::start_http_span("chat", "openai", &req.model);
        if let Some(s) = span.as_mut() {
            use opentelemetry::trace::Span as _;
            otel::add_request_params(s, req.temperature, req.top_p, req.max_tokens, Some(true));
            s.set_attribute(opentelemetry::KeyValue::new("f3dx.assembled", true));
            if schema_value.is_some() {
                s.set_attribute(opentelemetry::KeyValue::new("f3dx.validate_schema", true));
            }
        }

        let url = format!("{}/chat/completions", self.base_url);
        PyAssembledStream::start(
            Arc::clone(&self.client),
            Arc::clone(&self.runtime),
            url,
            req,
            span,
            validate_json,
            schema_value,
        )
        .map(|s| Py::new(py, s))
        .and_then(|r| r)
    }

    #[getter]
    fn base_url(&self) -> String {
        (*self.base_url).clone()
    }

    fn __repr__(&self) -> String {
        format!("OpenAIClient(base_url={:?})", &*self.base_url)
    }
}

// ---------- helpers ----------

fn parse_request(py: Python<'_>, request: Bound<'_, PyAny>) -> PyResult<ChatCompletionRequest> {
    if let Ok(s) = request.downcast::<PyString>() {
        return serde_json::from_str(&s.to_string_lossy())
            .map_err(|e| PyValueError::new_err(format!("request JSON parse: {e}")));
    }
    let json_module = py.import("json")?;
    let dumps = json_module.getattr("dumps")?;
    let body_str: String = dumps.call1((request,))?.extract()?;
    serde_json::from_str(&body_str)
        .map_err(|e| PyValueError::new_err(format!("request JSON parse: {e}")))
}
