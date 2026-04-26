//! agx-rt — agent runtime core.
//!
//! Whole-loop in Rust + concurrent tool dispatch via Python::allow_threads.
//! State lives in Rust for the duration of an agent run. Boundary crossings:
//!   1. Tool function calls (Python tool callables — fanned out across OS
//!      threads when the model returns multiple tool_calls per turn)
//!   2. Model HTTP request (mocked here; real HTTP comes from agx-http)
//!   3. Final result return
//!
//! Concurrent dispatch unlocks the real win: Python tools that do I/O
//! (HTTP, sleep, file) release the GIL during the wait. N tools each
//! taking T ms run in ~T ms wall-clock. Python pydantic-ai cannot hit
//! this because asyncio dispatches sequentially within one event loop.

use ahash::AHashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread;

// ---------- message types ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallReq>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallReq {
    pub id: String,
    pub name: String,
    /// JSON string of the args (mirrors OpenAI tool_calls[].function.arguments)
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockModelResponse {
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallReq>,
}

// ---------- agent runtime ----------

#[pyclass]
pub struct AgentRuntime {
    system_prompt: String,
    max_iterations: usize,
    max_tool_calls: usize,
    concurrent_tool_dispatch: bool,
}

#[pymethods]
impl AgentRuntime {
    #[new]
    #[pyo3(signature = (
        system_prompt = String::new(),
        max_iterations = 10,
        max_tool_calls = 20,
        concurrent_tool_dispatch = false,
    ))]
    pub fn new(
        system_prompt: String,
        max_iterations: usize,
        max_tool_calls: usize,
        concurrent_tool_dispatch: bool,
    ) -> Self {
        Self {
            system_prompt,
            max_iterations,
            max_tool_calls,
            concurrent_tool_dispatch,
        }
    }

    fn run<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        tools: Bound<'py, PyDict>,
        mock_responses: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut tool_table: AHashMap<String, Py<PyAny>> = AHashMap::with_capacity(tools.len());
        for (k, v) in tools.iter() {
            let name: String = k.extract()?;
            tool_table.insert(name, v.unbind());
        }
        let tool_table = Arc::new(tool_table);

        let mut messages: Vec<Message> = Vec::with_capacity(self.max_iterations * 4);
        if !self.system_prompt.is_empty() {
            messages.push(Message {
                role: "system".into(),
                content: self.system_prompt.clone(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            });
        }
        messages.push(Message {
            role: "user".into(),
            content: prompt,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });

        let mut tool_calls_executed: usize = 0;
        let mut final_answer = String::new();
        let mut iter_done: usize = 0;

        for iter_idx in 0..self.max_iterations {
            iter_done = iter_idx + 1;

            let mock_str: String = match mock_responses.get_item(iter_idx) {
                Ok(s) => s.extract()?,
                Err(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mock_responses ran out at iteration {iter_idx}"
                    )));
                }
            };
            let response: MockModelResponse =
                serde_json::from_str(&mock_str).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "mock response {iter_idx} not parseable: {e}"
                    ))
                })?;

            messages.push(Message {
                role: "assistant".into(),
                content: response.content.clone(),
                tool_calls: response.tool_calls.clone(),
                tool_call_id: None,
            });

            if response.tool_calls.is_empty() {
                final_answer = response.content;
                break;
            }

            let calls_to_run: Vec<ToolCallReq> = response
                .tool_calls
                .iter()
                .take(self.max_tool_calls.saturating_sub(tool_calls_executed))
                .cloned()
                .collect();

            let results: Vec<(String, String)> =
                if self.concurrent_tool_dispatch && calls_to_run.len() > 1 {
                    dispatch_parallel(py, &tool_table, &calls_to_run)
                } else {
                    dispatch_sequential(py, &tool_table, &calls_to_run)
                };

            tool_calls_executed += calls_to_run.len();

            for (id, result_str) in results {
                messages.push(Message {
                    role: "tool".into(),
                    content: result_str,
                    tool_calls: Vec::new(),
                    tool_call_id: Some(id),
                });
            }
        }

        let out = PyDict::new(py);
        out.set_item("answer", final_answer)?;
        out.set_item("iterations", iter_done)?;
        out.set_item("tool_calls", tool_calls_executed)?;

        let py_messages = PyList::empty(py);
        for m in messages.iter() {
            let d = PyDict::new(py);
            d.set_item("role", &m.role)?;
            d.set_item("content", &m.content)?;
            if !m.tool_calls.is_empty() {
                let tcs = PyList::empty(py);
                for tc in m.tool_calls.iter() {
                    let tcd = PyDict::new(py);
                    tcd.set_item("id", &tc.id)?;
                    tcd.set_item("name", &tc.name)?;
                    tcd.set_item("arguments", &tc.arguments)?;
                    tcs.append(tcd)?;
                }
                d.set_item("tool_calls", tcs)?;
            }
            if let Some(tcid) = &m.tool_call_id {
                d.set_item("tool_call_id", tcid)?;
            }
            py_messages.append(d)?;
        }
        out.set_item("messages", py_messages)?;

        Ok(out)
    }
}

// ---------- dispatch helpers ----------

fn dispatch_sequential(
    py: Python<'_>,
    tool_table: &AHashMap<String, Py<PyAny>>,
    calls: &[ToolCallReq],
) -> Vec<(String, String)> {
    calls
        .iter()
        .map(|tc| {
            let result = call_tool(py, tool_table, &tc.name, &tc.arguments);
            (tc.id.clone(), result)
        })
        .collect()
}

fn dispatch_parallel(
    py: Python<'_>,
    tool_table: &Arc<AHashMap<String, Py<PyAny>>>,
    calls: &[ToolCallReq],
) -> Vec<(String, String)> {
    py.allow_threads(|| {
        thread::scope(|s| {
            let handles: Vec<_> = calls
                .iter()
                .map(|tc| {
                    let table = Arc::clone(tool_table);
                    let name = tc.name.clone();
                    let args = tc.arguments.clone();
                    let id = tc.id.clone();
                    s.spawn(move || {
                        let result = Python::with_gil(|py| call_tool(py, &table, &name, &args));
                        (id, result)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("tool dispatch thread panicked"))
                .collect()
        })
    })
}

fn call_tool(
    py: Python<'_>,
    tool_table: &AHashMap<String, Py<PyAny>>,
    name: &str,
    arguments: &str,
) -> String {
    match tool_table.get(name) {
        Some(callable) => {
            let bound = callable.bind(py);
            match bound.call1((arguments,)) {
                Ok(ret) => ret
                    .extract()
                    .unwrap_or_else(|_| format!(r#"{{"error":"tool {name} returned non-string"}}"#)),
                Err(e) => format!(r#"{{"error":"tool {name} raised: {e}"}}"#),
            }
        }
        None => format!(r#"{{"error":"unknown tool {name}"}}"#),
    }
}

/// Register agx-rt's pyclasses into a parent Python module.
/// The agx-py crate calls this from its #[pymodule] root.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AgentRuntime>()?;
    Ok(())
}
