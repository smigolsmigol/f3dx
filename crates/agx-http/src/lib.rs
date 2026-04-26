//! agx-http — Rust LLM HTTP client.
//!
//! Phase A: sync OpenAI chat-completions client.
//! Phase B (next turn): SSE streaming + streaming JSON validation.
//! Phase C: vendor adapters (Anthropic, Gemini, vLLM extra_body, Mistral, xAI).
//!
//! Drop-in shape for `from openai import OpenAI`. Same constructor surface
//! (api_key, base_url, http_client). Same `chat.completions.create()` shape.
//! Returns dicts mirroring the OpenAI response, no marshalling overhead.
//!
//! reqwest with rustls-tls + http2 + gzip + brotli. Connection pool stays
//! alive for the lifetime of the client. GIL released during HTTP via
//! py.allow_threads (Phase B will do the same for streaming).

mod openai;
mod request;
mod response;

use pyo3::prelude::*;

pub use openai::PyOpenAIClient;

/// Register agx-http's pyclasses into a parent Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpenAIClient>()?;
    Ok(())
}
