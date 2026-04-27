//! f3dx-rt — agent runtime core.
//!
//! Whole-loop in Rust + concurrent tool dispatch via Python::allow_threads.
//! State lives in Rust for the duration of an agent run. Boundary crossings:
//!   1. Tool function calls (Python tool callables — fanned out across OS
//!      threads when the model returns multiple tool_calls per turn)
//!   2. Model HTTP request (mocked here; real HTTP comes from f3dx-http)
//!   3. Final result return
//!
//! Concurrent dispatch unlocks the real win: Python tools that do I/O
//! (HTTP, sleep, file) release the GIL during the wait. N tools each
//! taking T ms run in ~T ms wall-clock. Python pydantic-ai cannot hit
//! this because asyncio dispatches sequentially within one event loop.

use ahash::AHashMap;
use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Status, Tracer as _};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

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
    /// Optional per-turn token usage. When present, AgentRuntime accumulates
    /// across turns and emits totals on the JSONL trace row. Mirrors what
    /// real OpenAI/Anthropic responses carry; mock harnesses set it explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<MockUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MockUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
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
        let t_start = Instant::now();
        let mut span = f3dx_trace::tracer().map(|t| {
            let mut s = t.start("f3dx.agent_runtime.run");
            s.set_attribute(KeyValue::new("gen_ai.system", "f3dx"));
            s.set_attribute(KeyValue::new(
                "f3dx.concurrent_tool_dispatch",
                self.concurrent_tool_dispatch,
            ));
            s.set_attribute(KeyValue::new(
                "f3dx.max_iterations",
                self.max_iterations as i64,
            ));
            s.set_attribute(KeyValue::new(
                "f3dx.max_tool_calls",
                self.max_tool_calls as i64,
            ));
            s.set_attribute(KeyValue::new(
                "gen_ai.prompt.length_chars",
                prompt.len() as i64,
            ));
            s
        });

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
        let captured_prompt = prompt.clone();
        messages.push(Message {
            role: "user".into(),
            content: prompt,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });

        let mut tool_calls_executed: usize = 0;
        let mut final_answer = String::new();
        let mut iter_done: usize = 0;
        let mut usage_input: u64 = 0;
        let mut usage_output: u64 = 0;

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
            let response: MockModelResponse = serde_json::from_str(&mock_str).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "mock response {iter_idx} not parseable: {e}"
                ))
            })?;

            if let Some(u) = &response.usage {
                usage_input = usage_input.saturating_add(u.input_tokens);
                usage_output = usage_output.saturating_add(u.output_tokens);
            }
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

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(s) = span.as_mut() {
            s.set_attribute(KeyValue::new("f3dx.iterations", iter_done as i64));
            s.set_attribute(KeyValue::new(
                "f3dx.tool_calls_executed",
                tool_calls_executed as i64,
            ));
            s.set_attribute(KeyValue::new(
                "f3dx.output.length_chars",
                final_answer.len() as i64,
            ));
            s.set_attribute(KeyValue::new("f3dx.duration_ms", elapsed_ms));
            s.set_status(Status::Ok);
            s.end();
        }

        // Phase G V0: emit one JSONL row per run when a sink is configured.
        // No-op when configure_traces hasn't been called. When the sink was
        // configured with capture_messages=True, the row carries prompt +
        // system_prompt + output so downstream replay tools (tracewright)
        // can rebuild the original request. Off by default (PII-safe).
        let mut row = serde_json::json!({
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            "duration_ms": elapsed_ms,
            "iterations": iter_done,
            "tool_calls_executed": tool_calls_executed,
            "concurrent_tool_dispatch": self.concurrent_tool_dispatch,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "system_prompt_chars": self.system_prompt.len(),
            "output_chars": final_answer.len(),
            "tool_calls": messages.iter()
                .flat_map(|m| m.tool_calls.iter().map(|tc| {
                    serde_json::json!({"name": tc.name, "id": tc.id})
                }))
                .collect::<Vec<_>>(),
            "messages_count": messages.len(),
        });
        if let Some(obj) = row.as_object_mut() {
            obj.insert(
                "input_tokens".into(),
                serde_json::Value::Number(serde_json::Number::from(usage_input)),
            );
            obj.insert(
                "output_tokens".into(),
                serde_json::Value::Number(serde_json::Number::from(usage_output)),
            );
        }
        if f3dx_trace::capture_messages_enabled() {
            if let Some(obj) = row.as_object_mut() {
                obj.insert("prompt".into(), serde_json::Value::String(captured_prompt));
                obj.insert(
                    "system_prompt".into(),
                    serde_json::Value::String(self.system_prompt.clone()),
                );
                obj.insert(
                    "output".into(),
                    serde_json::Value::String(final_answer.clone()),
                );
            }
        }
        f3dx_trace::emit_trace_row(&row);

        let out = PyDict::new(py);
        out.set_item("answer", final_answer)?;
        out.set_item("iterations", iter_done)?;
        out.set_item("tool_calls", tool_calls_executed)?;
        out.set_item("duration_ms", elapsed_ms)?;

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
                .zip(calls.iter())
                .map(|(h, tc)| match h.join() {
                    Ok(pair) => pair,
                    Err(payload) => {
                        let msg = panic_payload_msg(&payload);
                        (
                            tc.id.clone(),
                            format!(
                                r#"{{"error":"tool {} panicked: {}"}}"#,
                                tc.name.replace('"', "\\\""),
                                msg.replace('"', "\\\"")
                            ),
                        )
                    }
                })
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
                Ok(ret) => ret.extract().unwrap_or_else(|_| {
                    format!(r#"{{"error":"tool {name} returned non-string"}}"#)
                }),
                Err(e) => format!(r#"{{"error":"tool {name} raised: {e}"}}"#),
            }
        }
        None => format!(r#"{{"error":"unknown tool {name}"}}"#),
    }
}

fn panic_payload_msg(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Register f3dx-rt's pyclasses into a parent Python module.
/// The f3dx-py crate calls this from its #[pymodule] root.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AgentRuntime>()?;
    Ok(())
}
