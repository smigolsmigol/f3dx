//! f3dx-http SSE streaming.
//!
//! Three Python-facing streaming classes:
//!   PyChatCompletionStream         — OpenAI raw chunks (Phase B)
//!   PyAssembledStream              — OpenAI assembled events (Phase D):
//!                                    delta_content / tool_call / done
//!   PyAnthropicStream              — Anthropic raw events (Phase C)
//!
//! All three share the same architecture: a tokio task pumps SSE events
//! from the model endpoint into a std::sync::mpsc channel. The Python
//! iterator pulls chunks via .recv() with the GIL released, so concurrent
//! threads can run.

use crate::otel::{self, SpanT};
use crate::request::ChatCompletionRequest;
use crate::response::ChatCompletionChunk;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Client;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

enum StreamEvent {
    Chunk(String),
    Done,
    Err(String),
}

// ---------- OpenAI raw streaming ----------

#[pyclass(name = "ChatCompletionStream", module = "f3dx._f3dx", unsendable)]
pub struct PyChatCompletionStream {
    rx: Mutex<Receiver<StreamEvent>>,
    _runtime: Arc<Runtime>,
}

impl PyChatCompletionStream {
    pub fn start(
        client: Arc<Client>,
        runtime: Arc<Runtime>,
        url: String,
        req: ChatCompletionRequest,
        span: Option<SpanT>,
    ) -> PyResult<Self> {
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        let runtime_handle = Arc::clone(&runtime);
        runtime_handle.spawn(async move {
            let mut span = span;
            if let Err(e) = pump_openai(client, url, req, tx.clone(), &mut span).await {
                if let Some(s) = span.take() {
                    otel::finish_err(s, format!("{e}"));
                }
                let _ = tx.send(StreamEvent::Err(format!("f3dx-http stream: {e}")));
            } else if let Some(s) = span.take() {
                otel::finish_ok(s);
            }
        });
        Ok(Self {
            rx: Mutex::new(rx),
            _runtime: runtime,
        })
    }
}

async fn pump_openai(
    client: Arc<Client>,
    url: String,
    req: ChatCompletionRequest,
    tx: Sender<StreamEvent>,
    span: &mut Option<SpanT>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let resp = client
        .post(&url)
        .json(&req)
        .send()
        .await?
        .error_for_status()?;

    let mut sse = resp.bytes_stream().eventsource();
    while let Some(event) = sse.next().await {
        match event {
            Ok(ev) => {
                if ev.data == "[DONE]" {
                    let _ = tx.send(StreamEvent::Done);
                    return Ok(());
                }
                // Try to extract usage attrs (OpenAI emits a final chunk with usage
                // when stream_options.include_usage=true). Cheap parse.
                if let (Some(s), Ok(parsed)) =
                    (span.as_mut(), serde_json::from_str::<Value>(&ev.data))
                {
                    otel::add_openai_response(s, &parsed);
                }
                if tx.send(StreamEvent::Chunk(ev.data)).is_err() {
                    return Ok(());
                }
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Err(format!("sse parse: {e}")));
                return Ok(());
            }
        }
    }
    let _ = tx.send(StreamEvent::Done);
    Ok(())
}

#[pymethods]
impl PyChatCompletionStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        recv_chunk_to_dict(py, &self.rx)
    }
}

// ---------- OpenAI assembled streaming (Phase D) ----------
//
// Reassembles fragmented tool_call streams Rust-side. User code gets
// three clean event types:
//   {"type": "delta_content", "content": "tok"}
//   {"type": "tool_call", "id": "call_abc", "name": "search",
//    "arguments": {...parsed JSON dict...}, "index": 0}
//   {"type": "done", "finish_reason": "stop"}
//
// Compared to the raw OpenAI SDK streaming surface, user code does not
// have to: accumulate arguments fragments by index, reassemble the
// arguments string, json.loads at the end, dispatch on chunk shape.

#[pyclass(name = "AssembledStream", module = "f3dx._f3dx", unsendable)]
pub struct PyAssembledStream {
    rx: Mutex<Receiver<StreamEvent>>,
    _runtime: Arc<Runtime>,
}

impl PyAssembledStream {
    pub fn start(
        client: Arc<Client>,
        runtime: Arc<Runtime>,
        url: String,
        req: ChatCompletionRequest,
        span: Option<SpanT>,
        validate_json: bool,
        output_schema: Option<Value>,
    ) -> PyResult<Self> {
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        let runtime_handle = Arc::clone(&runtime);
        runtime_handle.spawn(async move {
            let mut span = span;
            if let Err(e) =
                pump_openai_assembled(
                    client, url, req, tx.clone(), &mut span, validate_json, output_schema,
                )
                .await
            {
                if let Some(s) = span.take() {
                    otel::finish_err(s, format!("{e}"));
                }
                let _ = tx.send(StreamEvent::Err(format!("f3dx-http stream: {e}")));
            } else if let Some(s) = span.take() {
                otel::finish_ok(s);
            }
        });
        Ok(Self {
            rx: Mutex::new(rx),
            _runtime: runtime,
        })
    }
}

#[derive(Default, Debug)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments_buf: String,
}

async fn pump_openai_assembled(
    client: Arc<Client>,
    url: String,
    req: ChatCompletionRequest,
    tx: Sender<StreamEvent>,
    span: &mut Option<SpanT>,
    validate_json: bool,
    output_schema: Option<Value>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let resp = client
        .post(&url)
        .json(&req)
        .send()
        .await?
        .error_for_status()?;

    // BTreeMap so we emit tool calls in deterministic index order at the end
    let mut partials: BTreeMap<u64, PartialToolCall> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;
    // Accumulator for validate_json mode (concatenates all delta.content)
    let mut content_buf = String::new();

    let mut sse = resp.bytes_stream().eventsource();
    while let Some(event) = sse.next().await {
        match event {
            Ok(ev) => {
                if ev.data == "[DONE]" {
                    break;
                }
                let chunk: ChatCompletionChunk = match serde_json::from_str(&ev.data) {
                    Ok(c) => c,
                    Err(_) => continue, // pass through bad-shape chunks silently
                };

                // Apply usage attrs whenever they appear (final chunk if
                // stream_options.include_usage=true)
                if let (Some(s), Ok(parsed)) =
                    (span.as_mut(), serde_json::from_str::<Value>(&ev.data))
                {
                    otel::add_openai_response(s, &parsed);
                }

                for choice in chunk.choices.iter() {
                    if let Some(reason) = choice.finish_reason.clone() {
                        finish_reason = Some(reason);
                    }

                    if let Some(content) = choice.delta.content.as_ref() {
                        if !content.is_empty() {
                            if validate_json {
                                content_buf.push_str(content);
                            }
                            let event_str = format!(
                                r#"{{"type":"delta_content","content":{}}}"#,
                                serde_json::to_string(content).unwrap_or_else(|_| "\"\"".into())
                            );
                            if tx.send(StreamEvent::Chunk(event_str)).is_err() {
                                return Ok(());
                            }
                        }
                    }

                    if let Some(tcs) = choice.delta.tool_calls.as_ref() {
                        accumulate_tool_calls(tcs, &mut partials);
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Err(format!("sse parse: {e}")));
                return Ok(());
            }
        }
    }

    // Emit assembled tool calls in index order
    for (index, partial) in partials.into_iter() {
        let parsed_args: Value = serde_json::from_str(&partial.arguments_buf)
            .unwrap_or_else(|_| Value::String(partial.arguments_buf.clone()));
        let event = serde_json::json!({
            "type": "tool_call",
            "id": partial.id,
            "name": partial.name,
            "arguments": parsed_args,
            "index": index,
        });
        let event_str = serde_json::to_string(&event).unwrap_or_default();
        if tx.send(StreamEvent::Chunk(event_str)).is_err() {
            return Ok(());
        }
    }

    // If validation requested, parse the accumulated content as JSON and
    // emit either validated_output or validation_error before done. When
    // output_schema is also present, run jsonschema against the parsed
    // value; only emit validated_output on schema-conformant input.
    if validate_json && !content_buf.is_empty() {
        match serde_json::from_str::<Value>(&content_buf) {
            Ok(parsed) => {
                let schema_ok = match output_schema.as_ref() {
                    None => Ok(()),
                    Some(schema) => match jsonschema::validator_for(schema) {
                        Ok(validator) => {
                            let errs: Vec<String> = validator
                                .iter_errors(&parsed)
                                .map(|e| format!("{} at /{}", e, e.instance_path))
                                .collect();
                            if errs.is_empty() {
                                Ok(())
                            } else {
                                Err(errs.join("; "))
                            }
                        }
                        Err(e) => Err(format!("invalid schema: {e}")),
                    },
                };
                let event = match schema_ok {
                    Ok(()) => serde_json::json!({"type": "validated_output", "data": parsed}),
                    Err(detail) => serde_json::json!({
                        "type": "validation_error",
                        "raw": content_buf,
                        "error": detail,
                        "kind": "schema",
                    }),
                };
                let _ = tx.send(StreamEvent::Chunk(event.to_string()));
            }
            Err(e) => {
                let event = serde_json::json!({
                    "type": "validation_error",
                    "raw": content_buf,
                    "error": e.to_string(),
                    "kind": "json_parse",
                });
                let _ = tx.send(StreamEvent::Chunk(event.to_string()));
            }
        }
    }

    let done_event = serde_json::json!({
        "type": "done",
        "finish_reason": finish_reason,
    });
    let _ = tx.send(StreamEvent::Chunk(done_event.to_string()));
    let _ = tx.send(StreamEvent::Done);
    Ok(())
}

fn accumulate_tool_calls(tcs: &Value, partials: &mut BTreeMap<u64, PartialToolCall>) {
    let Some(arr) = tcs.as_array() else { return };
    for tc in arr.iter() {
        let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let entry = partials.entry(index).or_default();
        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
            entry.id = id.to_string();
        }
        if let Some(func) = tc.get("function") {
            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                if !name.is_empty() {
                    entry.name = name.to_string();
                }
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                entry.arguments_buf.push_str(args);
            }
        }
    }
}

#[pymethods]
impl PyAssembledStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        recv_chunk_to_dict(py, &self.rx)
    }
}

// ---------- Anthropic streaming ----------

#[pyclass(name = "AnthropicStream", module = "f3dx._f3dx", unsendable)]
pub struct PyAnthropicStream {
    rx: Mutex<Receiver<StreamEvent>>,
    _runtime: Arc<Runtime>,
}

pub fn spawn_anthropic_pump(
    client: Arc<Client>,
    runtime: Arc<Runtime>,
    url: String,
    req: Value,
    span: Option<SpanT>,
) -> PyResult<PyAnthropicStream> {
    let (tx, rx) = mpsc::channel::<StreamEvent>();
    let runtime_handle = Arc::clone(&runtime);
    runtime_handle.spawn(async move {
        let mut span = span;
        if let Err(e) = pump_anthropic(client, url, req, tx.clone(), &mut span).await {
            if let Some(s) = span.take() {
                otel::finish_err(s, format!("{e}"));
            }
            let _ = tx.send(StreamEvent::Err(format!("f3dx-http stream: {e}")));
        } else if let Some(s) = span.take() {
            otel::finish_ok(s);
        }
    });
    Ok(PyAnthropicStream {
        rx: Mutex::new(rx),
        _runtime: runtime,
    })
}

async fn pump_anthropic(
    client: Arc<Client>,
    url: String,
    req: Value,
    tx: Sender<StreamEvent>,
    span: &mut Option<SpanT>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let resp = client
        .post(&url)
        .json(&req)
        .send()
        .await?
        .error_for_status()?;

    let mut sse = resp.bytes_stream().eventsource();
    while let Some(event) = sse.next().await {
        match event {
            Ok(ev) => {
                // Usage lands in two places: message_start.message.usage (input_tokens),
                // message_delta.usage (output_tokens). Cheap parse + apply.
                if matches!(ev.event.as_str(), "message_start" | "message_delta") {
                    if let (Some(s), Ok(parsed)) =
                        (span.as_mut(), serde_json::from_str::<Value>(&ev.data))
                    {
                        // message_start nests usage under message.usage
                        let normalised = if ev.event == "message_start" {
                            parsed
                                .get("message")
                                .cloned()
                                .unwrap_or(parsed.clone())
                        } else {
                            parsed.clone()
                        };
                        otel::add_anthropic_response(s, &normalised);
                    }
                }
                if ev.event == "message_stop" {
                    if tx.send(StreamEvent::Chunk(ev.data)).is_err() {
                        return Ok(());
                    }
                    let _ = tx.send(StreamEvent::Done);
                    return Ok(());
                }
                if ev.event == "ping" {
                    continue;
                }
                if tx.send(StreamEvent::Chunk(ev.data)).is_err() {
                    return Ok(());
                }
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Err(format!("sse parse: {e}")));
                return Ok(());
            }
        }
    }
    let _ = tx.send(StreamEvent::Done);
    Ok(())
}

#[pymethods]
impl PyAnthropicStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        recv_chunk_to_dict(py, &self.rx)
    }
}

// ---------- shared recv helper ----------

fn recv_chunk_to_dict<'py>(
    py: Python<'py>,
    rx: &Mutex<Receiver<StreamEvent>>,
) -> PyResult<Bound<'py, PyDict>> {
    let event = py.allow_threads(|| {
        let rx = rx.lock().expect("stream rx mutex poisoned");
        rx.recv_timeout(Duration::from_secs(120))
    });

    let event = match event {
        Ok(ev) => ev,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            return Err(PyRuntimeError::new_err("f3dx-http stream: timeout"));
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return Err(PyStopIteration::new_err(()));
        }
    };

    match event {
        StreamEvent::Chunk(data) => {
            let json_module = py.import("json")?;
            let loads = json_module.getattr("loads")?;
            let py_obj = loads.call1((data,))?;
            py_obj
                .downcast_into::<PyDict>()
                .map_err(|e| PyRuntimeError::new_err(format!("chunk not dict: {e}")))
        }
        StreamEvent::Done => Err(PyStopIteration::new_err(())),
        StreamEvent::Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}
