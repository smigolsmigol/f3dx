//! agx-http SSE streaming.
//!
//! Two streaming surfaces:
//!   PyChatCompletionStream  — OpenAI shape, terminator "[DONE]" sentinel
//!   PyAnthropicStream       — Anthropic Messages shape, multiple event:
//!                             types, terminator on event=message_stop
//!
//! Both share the same architecture: a tokio task pumps SSE events from
//! the model endpoint into a std::sync::mpsc channel. The Python iterator
//! pulls chunks via .recv() with the GIL released, so concurrent threads
//! can run.

use crate::request::ChatCompletionRequest;
use crate::response::ChatCompletionChunk;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Client;
use serde_json::Value;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

/// One streamed event sent from the tokio task to the Python iterator.
enum StreamEvent {
    /// Parsed chunk JSON (raw string; Python side parses to dict).
    Chunk(String),
    /// Terminal sentinel.
    Done,
    /// Stream error.
    Err(String),
}

// ---------- OpenAI streaming ----------

#[pyclass(name = "ChatCompletionStream", module = "agx._agx", unsendable)]
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
    ) -> PyResult<Self> {
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        let runtime_handle = Arc::clone(&runtime);
        runtime_handle.spawn(async move {
            if let Err(e) = pump_openai(client, url, req, tx.clone()).await {
                let _ = tx.send(StreamEvent::Err(format!("agx-http stream: {e}")));
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
                // Validate the chunk parses; pass through raw on schema mismatch
                // so vendor non-strict shapes still flow.
                let _: Result<ChatCompletionChunk, _> = serde_json::from_str(&ev.data);
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

// ---------- Anthropic streaming ----------

#[pyclass(name = "AnthropicStream", module = "agx._agx", unsendable)]
pub struct PyAnthropicStream {
    rx: Mutex<Receiver<StreamEvent>>,
    _runtime: Arc<Runtime>,
}

pub fn spawn_anthropic_pump(
    client: Arc<Client>,
    runtime: Arc<Runtime>,
    url: String,
    req: Value,
) -> PyResult<PyAnthropicStream> {
    let (tx, rx) = mpsc::channel::<StreamEvent>();
    let runtime_handle = Arc::clone(&runtime);
    runtime_handle.spawn(async move {
        if let Err(e) = pump_anthropic(client, url, req, tx.clone()).await {
            let _ = tx.send(StreamEvent::Err(format!("agx-http stream: {e}")));
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
                // Anthropic terminator: event=message_stop
                if ev.event == "message_stop" {
                    if tx.send(StreamEvent::Chunk(ev.data)).is_err() {
                        return Ok(());
                    }
                    let _ = tx.send(StreamEvent::Done);
                    return Ok(());
                }
                // Skip pings to reduce noise; real content always carries data
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
            return Err(PyRuntimeError::new_err("agx-http stream: timeout"));
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
