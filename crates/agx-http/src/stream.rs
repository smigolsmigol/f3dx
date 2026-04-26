//! agx-http SSE streaming.
//!
//! Architecture: tokio task pumps SSE events from the model endpoint
//! into a std::sync::mpsc channel. Python iterator on the other side
//! pulls chunks via .recv() with the GIL released, so concurrent
//! Python threads can run. Thread-safe by construction (mpsc::Sender
//! is Sync; we hold one Receiver behind a Mutex on the Python side).

use crate::request::ChatCompletionRequest;
use crate::response::ChatCompletionChunk;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Client;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

/// One streamed event sent from the tokio task to the Python iterator.
enum StreamEvent {
    /// Parsed chunk JSON (raw string; Python side parses to dict).
    Chunk(String),
    /// Terminal "[DONE]" sentinel from OpenAI.
    Done,
    /// Stream error.
    Err(String),
}

#[pyclass(name = "ChatCompletionStream", module = "agx._agx", unsendable)]
pub struct PyChatCompletionStream {
    rx: Mutex<Receiver<StreamEvent>>,
    /// Hold the runtime so the spawned task isn't aborted prematurely.
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
            if let Err(e) = pump(client, url, req, tx.clone()).await {
                let _ = tx.send(StreamEvent::Err(format!("agx-http stream: {e}")));
            }
        });

        Ok(Self {
            rx: Mutex::new(rx),
            _runtime: runtime,
        })
    }
}

async fn pump(
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
                // Validate the chunk parses as a ChatCompletionChunk so we
                // catch bad-shape responses before they hit the Python side.
                // Cheap: serde_json on a small JSON string per chunk.
                let parsed: Result<ChatCompletionChunk, _> = serde_json::from_str(&ev.data);
                if parsed.is_err() {
                    // Pass through raw — some vendors emit non-strict shapes
                    // we still want the user to see (vLLM prompt_logprobs etc).
                    let _ = tx.send(StreamEvent::Chunk(ev.data));
                    continue;
                }
                if tx.send(StreamEvent::Chunk(ev.data)).is_err() {
                    // Receiver dropped (Python iterator dropped); stop pumping.
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
        // Release GIL while waiting for the next chunk. Concurrent Python
        // threads (eg agx-rt parallel tool dispatch) can run during the wait.
        let event = py.allow_threads(|| {
            let rx = self.rx.lock().expect("stream rx mutex poisoned");
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
}
