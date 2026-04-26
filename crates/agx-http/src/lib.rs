//! agx-http — Rust LLM HTTP client.
//!
//! Drop-in replacement for openai/anthropic/etc Python SDK clients.
//! reqwest connection pool + native SSE parser + streaming JSON
//! validation. GIL released during HTTP and SSE parse, re-acquired
//! only to deliver chunks to Python user code.
//!
//! Status: scaffolded, not yet implemented. Week 1-2 of agx_v1_plan.md
//! brings up the OpenAI chat-completions request + parser path with
//! reqwest. Vendor adapters (Anthropic, Gemini, vLLM extra_body,
//! Mistral, xAI) follow in week 3.

use pyo3::prelude::*;

/// Register agx-http's pyclasses into a parent Python module.
/// Currently a no-op until the OpenAI client lands in week 1-2.
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
