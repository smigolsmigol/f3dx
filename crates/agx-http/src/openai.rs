//! agx-http OpenAI sync client + PyO3 binding.
//!
//! Mirrors the `openai.OpenAI` constructor surface so swap-in is one import:
//!     from openai import OpenAI    -> from agx import OpenAI
//!
//! Phase A: sync `chat_completions_create(request_dict)` returns the full
//! response as a Python dict (passthrough of the model's JSON, no Pydantic
//! re-validation overhead — the JSON is already validated by serde).

use crate::request::ChatCompletionRequest;
use crate::response::ChatCompletionResponse;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const USER_AGENT_STR: &str = concat!("agx-http/", env!("CARGO_PKG_VERSION"));

/// agx OpenAI sync HTTP client. PyO3-exposed.
#[pyclass(name = "OpenAIClient", module = "agx._agx")]
pub struct PyOpenAIClient {
    client: Client,
    base_url: String,
    /// Stored only so we can rebuild headers if needed; never logged.
    _api_key: String,
}

#[pymethods]
impl PyOpenAIClient {
    /// Create a new client. Mirrors openai.OpenAI(api_key=..., base_url=..., timeout=...).
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

        Ok(Self {
            client,
            base_url,
            _api_key: api_key,
        })
    }

    /// Sync chat completion. Pass a dict matching the OpenAI request shape;
    /// returns a dict matching the OpenAI response shape.
    ///
    /// GIL is released during the HTTP call so concurrent Python threads
    /// can make progress (this composes with agx-rt's parallel tool dispatch).
    fn chat_completions_create<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Allow either a dict or a JSON string for the request body
        let req: ChatCompletionRequest = if let Ok(s) = request.downcast::<PyString>() {
            serde_json::from_str(&s.to_string_lossy())
                .map_err(|e| PyValueError::new_err(format!("request JSON parse: {e}")))?
        } else {
            // dict path: round-trip via Python's json.dumps for now (Phase B
            // will accept a typed Pydantic model directly to skip this hop)
            let json_module = py.import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let body_str: String = dumps.call1((request,))?.extract()?;
            serde_json::from_str(&body_str)
                .map_err(|e| PyValueError::new_err(format!("request JSON parse: {e}")))?
        };

        let url = format!("{}/chat/completions", self.base_url);

        // GIL released during HTTP. Threads waiting in agx-rt can run.
        let resp = py.allow_threads(|| {
            self.client
                .post(&url)
                .json(&req)
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.json::<ChatCompletionResponse>())
        });

        let parsed = resp.map_err(|e| PyRuntimeError::new_err(format!("agx-http: {e}")))?;

        // Marshal back to Python dict via JSON round-trip (one boundary cross,
        // amortised against the network call which dominates wall time)
        let out_str = serde_json::to_string(&parsed)
            .map_err(|e| PyRuntimeError::new_err(format!("response serialise: {e}")))?;
        let json_module = py.import("json")?;
        let loads = json_module.getattr("loads")?;
        let py_obj = loads.call1((out_str,))?;
        py_obj
            .downcast_into::<PyDict>()
            .map_err(|e| PyRuntimeError::new_err(format!("response not dict: {e}")))
    }

    /// Read-only; never returns the api_key.
    #[getter]
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn __repr__(&self) -> String {
        format!("OpenAIClient(base_url={:?})", self.base_url)
    }
}
