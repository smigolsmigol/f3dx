//! agx-http — Rust LLM HTTP client.
//!
//! Phase A (shipped): sync chat completions via reqwest blocking.
//! Phase B (this turn): async reqwest + tokio + SSE streaming.
//!     `OpenAI.chat_completions_create_stream(req)` returns a sync
//!     Python iterator that yields chunk dicts. Backed by a tokio task
//!     that pumps SSE events through a std::sync::mpsc channel; the
//!     Python iterator releases the GIL while waiting on recv().
//! Phase C: vendor adapters (Anthropic, Gemini, vLLM extra_body, Mistral, xAI).

mod openai;
mod request;
mod response;
mod stream;

use pyo3::prelude::*;

pub use openai::PyOpenAIClient;
pub use stream::PyChatCompletionStream;

/// Register agx-http's pyclasses into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpenAIClient>()?;
    m.add_class::<PyChatCompletionStream>()?;
    Ok(())
}
