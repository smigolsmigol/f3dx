//! f3dx-mcp — Model Context Protocol client for the f3dx agent loop.
//!
//! Phase Tier 6 V0: stdio transport only. Spawns an MCP server as a child
//! process (the dominant pattern; covers Claude Desktop's npx-based servers
//! plus every server-everything-shaped tool). list_tools surfaces the
//! server's tool catalog as plain dicts; call_tool runs one tool with
//! json-encoded args and returns the text content.
//!
//! V0.1 will add SSE + streamable-HTTP transports plus a sampling callback
//! bridge so MCP servers can ask the agent's model for completions.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use std::sync::Arc;
use tokio::process::Command;
use tokio::runtime::Runtime;

#[pyclass(name = "MCPClient", module = "f3dx._f3dx", unsendable)]
pub struct PyMCPClient {
    runtime: Arc<Runtime>,
    service: Arc<RunningService<rmcp::RoleClient, ()>>,
}

#[pymethods]
impl PyMCPClient {
    /// Spawn an MCP server over stdio and connect.
    ///
    /// `command` is the binary to exec (e.g. "npx", "python", "/usr/bin/node").
    /// `args` is the argv tail. The server's stdin/stdout/stderr are wired
    /// for the JSON-RPC handshake; nothing leaks to the host process's
    /// stdio. Returns a connected MCPClient.
    #[staticmethod]
    #[pyo3(signature = (command, args = None))]
    fn stdio(command: String, args: Option<Vec<String>>) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("f3dx-mcp")
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime build: {e}")))?;

        let mut cmd = Command::new(&command);
        if let Some(a) = args {
            cmd.args(&a);
        }

        let service = runtime
            .block_on(async {
                let transport = TokioChildProcess::new(cmd)
                    .map_err(|e| PyRuntimeError::new_err(format!("spawn {command:?}: {e}")))?;
                ()
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
    /// The transport handles the JSON-RPC handshake plus server-sent-event
    /// streaming for incremental tool results. Auth headers go in the URL
    /// query string today; explicit header support lands in V0.2 once we
    /// thread reqwest's HeaderMap through the transport builder.
    #[staticmethod]
    fn streamable_http(url: String) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("f3dx-mcp")
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime build: {e}")))?;

        let service = runtime.block_on(async {
            let transport = StreamableHttpClientTransport::from_uri(url.as_str());
            ()
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
        let tools = self
            .runtime
            .block_on(async move {
                service
                    .list_all_tools()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("list_tools: {e}")))
            })?;

        let out = PyList::empty(py);
        for tool in tools {
            let d = PyDict::new(py);
            d.set_item("name", tool.name.as_ref())?;
            d.set_item(
                "description",
                tool.description.as_ref().map(|s| s.as_ref()),
            )?;
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
    fn call_tool(&self, name: String, arguments: Option<String>) -> PyResult<String> {
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
        let result = self.runtime.block_on(async move {
            service
                .call_tool(params)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("call_tool: {e}")))
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

/// Register f3dx-mcp's pyclass into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMCPClient>()?;
    Ok(())
}
