//! agx-http - Rust LLM HTTP client.
//!
//! Phase A (shipped): sync chat completions via reqwest.
//! Phase B (shipped): async reqwest + tokio + SSE streaming for OpenAI.
//!     5x speedup vs `openai` Python SDK on streaming bench.
//! Phase C (this turn): Anthropic Messages adapter (sync + streaming).
//!     Drop-in for `anthropic.Anthropic`. x-api-key auth + anthropic-version
//!     header. Native handling of Anthropic SSE event types
//!     (message_start, content_block_*, ping, message_delta, message_stop).
//! Phase D: tool-call streaming reassembly + structured-output streaming
//!     validation.
//! Phase E: per-vendor docs for OpenAI-compatible endpoints (vLLM,
//!     Mistral, xAI, Groq, Together, Fireworks all work today via
//!     agx.OpenAI with a different base_url).

mod anthropic;
mod openai;
mod otel;
mod request;
mod response;
mod stream;

use pyo3::prelude::*;

pub use anthropic::PyAnthropicClient;
pub use openai::PyOpenAIClient;
pub use stream::{PyAnthropicStream, PyAssembledStream, PyChatCompletionStream};

/// Register agx-http's pyclasses into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpenAIClient>()?;
    m.add_class::<PyChatCompletionStream>()?;
    m.add_class::<PyAssembledStream>()?;
    m.add_class::<PyAnthropicClient>()?;
    m.add_class::<PyAnthropicStream>()?;
    Ok(())
}
