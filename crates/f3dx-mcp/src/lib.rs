//! f3dx-mcp — Model Context Protocol client for the f3dx agent loop.
//!
//! Phase Tier 6:
//!   V0   — stdio transport (dominant; covers Claude Desktop + npm-based servers)
//!   V0.1 — streamable-HTTP transport (remote MCP servers, Cloudflare Workers, hosted MCP)
//!   V0.2 — sampling-callback bridge: MCP servers can ask the agent's model
//!          for completions via a Python callback that runs on every
//!          create_message request.
//!
//! Future: full SSE + auth-header customization, structured-output streaming.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rmcp::model::{
    CallToolRequestParams, CreateMessageRequestParams, CreateMessageResult, ErrorData,
    SamplingMessage,
};
use rmcp::service::{RequestContext, RoleClient, RunningService};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{ClientHandler, ServiceExt};
use std::sync::Arc;
use tokio::process::Command;
use tokio::runtime::Runtime;

/// Client-side handler bridging MCP server-initiated requests to Python callbacks.
///
/// MCP servers can issue `sampling/createMessage` requests that ask the
/// connected client (us) to run a model and return a completion. When a
/// `sampling_callback` is registered, this struct forwards each request
/// to the Python callable; otherwise we surface the standard
/// "sampling unsupported" error that rmcp emits by default.
#[derive(Clone, Default)]
struct F3dxClientHandler {
    sampling_callback: Option<Arc<Py<PyAny>>>,
}

impl ClientHandler for F3dxClientHandler {
    async fn create_message(
        &self,
        params: CreateMessageRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> Result<CreateMessageResult, ErrorData> {
        let Some(cb) = self.sampling_callback.as_ref() else {
            return Err(ErrorData::method_not_found::<
                rmcp::model::CreateMessageRequestMethod,
            >());
        };

        // Serialize each SamplingMessage straight to JSON. The serde repr
        // for content varies across rmcp versions (text vs image vs other),
        // so we hand the full structure to Python and let the callback
        // reach in for whatever it needs.
        let messages_json = serde_json::to_string(&params.messages)
            .map_err(|e| ErrorData::internal_error(format!("messages serialize: {e}"), None))?;
        let system_prompt = params.system_prompt.clone().unwrap_or_default();

        let cb = Arc::clone(cb);
        let response_text = tokio::task::spawn_blocking(move || -> Result<String, String> {
            Python::with_gil(|py| {
                let bound = cb.bind(py);
                let out = bound
                    .call1((messages_json.as_str(), system_prompt.as_str()))
                    .map_err(|e| format!("sampling callback raised: {e}"))?;
                out.extract::<String>()
                    .map_err(|e| format!("sampling callback returned non-string: {e}"))
            })
        })
        .await
        .map_err(|e| ErrorData::internal_error(format!("sampling spawn_blocking: {e}"), None))?
        .map_err(|e| ErrorData::internal_error(e, None))?;

        Ok(CreateMessageResult::new(
            SamplingMessage::assistant_text(response_text),
            "f3dx-sampling-callback".to_string(),
        )
        .with_stop_reason(CreateMessageResult::STOP_REASON_END_TURN))
    }
}

#[pyclass(name = "MCPClient", module = "f3dx._f3dx", unsendable)]
pub struct PyMCPClient {
    runtime: Arc<Runtime>,
    service: Arc<RunningService<RoleClient, F3dxClientHandler>>,
}

#[pymethods]
impl PyMCPClient {
    /// Spawn an MCP server over stdio and connect.
    ///
    /// `command` is the binary to exec (e.g. "npx", "python", "/usr/bin/node").
    /// `args` is the argv tail. The server's stdin/stdout/stderr are wired
    /// for the JSON-RPC handshake; nothing leaks to the host process's
    /// stdio. `sampling_callback`, when provided, is invoked on every
    /// `sampling/createMessage` request the server issues; signature:
    /// `(messages_json: str, system_prompt: str) -> str` returning the
    /// assistant's text. Without the callback, sampling requests get the
    /// standard "method not supported" error.
    #[staticmethod]
    #[pyo3(signature = (command, args = None, sampling_callback = None))]
    fn stdio(
        command: String,
        args: Option<Vec<String>>,
        sampling_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let runtime = build_runtime()?;
        let mut cmd = Command::new(&command);
        if let Some(a) = args {
            cmd.args(&a);
        }

        let handler = F3dxClientHandler {
            sampling_callback: sampling_callback.map(Arc::new),
        };

        let service = runtime.block_on(async {
            let transport = TokioChildProcess::new(cmd)
                .map_err(|e| PyRuntimeError::new_err(format!("spawn {command:?}: {e}")))?;
            handler
                .serve(transport)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("mcp handshake failed: {e}")))
        })?;

        Ok(Self {
            runtime: Arc::new(runtime),
            service: Arc::new(service),
        })
    }

    /// Connect to a remote MCP server over streamable-HTTP.
    ///
    /// `url` is the server's MCP endpoint (e.g. https://my-mcp.example.com/mcp).
    /// `sampling_callback` works identically to the stdio version above.
    #[staticmethod]
    #[pyo3(signature = (url, sampling_callback = None))]
    fn streamable_http(url: String, sampling_callback: Option<Py<PyAny>>) -> PyResult<Self> {
        let runtime = build_runtime()?;
        let handler = F3dxClientHandler {
            sampling_callback: sampling_callback.map(Arc::new),
        };

        let service = runtime.block_on(async {
            let transport = StreamableHttpClientTransport::from_uri(url.as_str());
            handler
                .serve(transport)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("mcp http handshake failed: {e}")))
        })?;

        Ok(Self {
            runtime: Arc::new(runtime),
            service: Arc::new(service),
        })
    }

    /// Return the connected server's tool catalog as a list of dicts:
    /// [{"name": str, "description": str | None, "input_schema": dict}, ...]
    fn list_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let service = Arc::clone(&self.service);
        let runtime = Arc::clone(&self.runtime);
        let tools = py.allow_threads(|| {
            runtime.block_on(async move {
                service
                    .list_all_tools()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("list_tools: {e}")))
            })
        })?;

        let out = PyList::empty(py);
        for tool in tools {
            let d = PyDict::new(py);
            d.set_item("name", tool.name.as_ref())?;
            d.set_item("description", tool.description.as_ref().map(|s| s.as_ref()))?;
            let schema_json = serde_json::to_string(&tool.input_schema)
                .map_err(|e| PyRuntimeError::new_err(format!("input_schema serialize: {e}")))?;
            let json_module = py.import("json")?;
            let parsed = json_module.getattr("loads")?.call1((schema_json,))?;
            d.set_item("input_schema", parsed)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Call one tool by name with json-encoded args. Returns the text
    /// content of the first text-kind content block, or the JSON-encoded
    /// full response when no text block is present.
    #[pyo3(signature = (name, arguments = None))]
    fn call_tool(&self, py: Python<'_>, name: String, arguments: Option<String>) -> PyResult<String> {
        let arg_obj: Option<serde_json::Map<String, serde_json::Value>> = match arguments {
            None => None,
            Some(s) => {
                let parsed: serde_json::Value = serde_json::from_str(&s).map_err(|e| {
                    PyValueError::new_err(format!("arguments must be a JSON object: {e}"))
                })?;
                match parsed {
                    serde_json::Value::Object(m) => Some(m),
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "arguments must be a JSON object, got {other:?}"
                        )));
                    }
                }
            }
        };

        let mut params = CallToolRequestParams::new(name);
        params = params.with_arguments(arg_obj.unwrap_or_default());

        let service = Arc::clone(&self.service);
        let runtime = Arc::clone(&self.runtime);
        // Release the GIL during block_on so the sampling-callback path can
        // re-acquire it on the spawn_blocking thread without deadlocking.
        let result = py.allow_threads(|| {
            runtime.block_on(async move {
                service
                    .call_tool(params)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("call_tool: {e}")))
            })
        })?;

        for content in &result.content {
            if let Some(text) = content.as_text() {
                return Ok(text.text.clone());
            }
        }
        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("response serialize: {e}")))
    }
}

fn build_runtime() -> PyResult<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .thread_name("f3dx-mcp")
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime build: {e}")))
}

/// Register f3dx-mcp's pyclass into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMCPClient>()?;
    Ok(())
}
