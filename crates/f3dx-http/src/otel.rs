//! gen_ai.* semantic-convention helpers for HTTP-level spans.
//!
//! Naming + attribute keys follow the OpenTelemetry GenAI semantic
//! conventions (https://opentelemetry.io/docs/specs/semconv/gen-ai/).
//! Span kind: client (we are the HTTP client calling out to the model).

use opentelemetry::trace::{Span, SpanKind, Status, Tracer as _};
use opentelemetry::{Array, KeyValue, StringValue, Value};

/// Concrete span type returned by f3dx_trace::tracer(). We re-borrow it
/// via the public Span trait so we do not have to depend on the SDK
/// crate from f3dx-http.
pub type SpanT = <opentelemetry_sdk::trace::Tracer as opentelemetry::trace::Tracer>::Span;

// We DO need the sdk type alias above; re-export through f3dx-trace would be
// cleaner long-term, but adding the SDK as a transitive type is the cheapest
// fix for V0.1. The dep is already present in the workspace via f3dx-trace.

/// Start a span for a chat-completion / messages request. Returns None
/// if the global tracer isn't configured (zero-overhead pathway).
pub fn start_http_span(op_name: &'static str, system: &'static str, model: &str) -> Option<SpanT> {
    let tracer = f3dx_trace::tracer()?;
    let mut span = tracer
        .span_builder(format!("{op_name} {model}"))
        .with_kind(SpanKind::Client)
        .start(tracer);
    span.set_attribute(KeyValue::new("gen_ai.system", system));
    span.set_attribute(KeyValue::new("gen_ai.operation.name", op_name));
    span.set_attribute(KeyValue::new("gen_ai.request.model", model.to_string()));
    Some(span)
}

pub fn add_request_params<S: Span>(
    span: &mut S,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
) {
    if let Some(t) = temperature {
        span.set_attribute(KeyValue::new("gen_ai.request.temperature", t as f64));
    }
    if let Some(p) = top_p {
        span.set_attribute(KeyValue::new("gen_ai.request.top_p", p as f64));
    }
    if let Some(m) = max_tokens {
        span.set_attribute(KeyValue::new("gen_ai.request.max_tokens", m as i64));
    }
    if let Some(s) = stream {
        span.set_attribute(KeyValue::new("gen_ai.request.stream", s));
    }
}

pub fn add_openai_response<S: Span>(span: &mut S, response: &serde_json::Value) {
    if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
        span.set_attribute(KeyValue::new("gen_ai.response.id", id.to_string()));
    }
    if let Some(model) = response.get("model").and_then(|v| v.as_str()) {
        span.set_attribute(KeyValue::new("gen_ai.response.model", model.to_string()));
    }
    if let Some(usage) = response.get("usage") {
        if let Some(n) = usage.get("prompt_tokens").and_then(|v| v.as_i64()) {
            span.set_attribute(KeyValue::new("gen_ai.usage.input_tokens", n));
        }
        if let Some(n) = usage.get("completion_tokens").and_then(|v| v.as_i64()) {
            span.set_attribute(KeyValue::new("gen_ai.usage.output_tokens", n));
        }
    }
    if let Some(choices) = response.get("choices").and_then(|v| v.as_array()) {
        let reasons: Vec<StringValue> = choices
            .iter()
            .filter_map(|c| c.get("finish_reason").and_then(|v| v.as_str()))
            .map(|s| StringValue::from(s.to_string()))
            .collect();
        if !reasons.is_empty() {
            span.set_attribute(KeyValue::new(
                "gen_ai.response.finish_reasons",
                Value::Array(Array::String(reasons)),
            ));
        }
    }
}

pub fn add_anthropic_response<S: Span>(span: &mut S, response: &serde_json::Value) {
    if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
        span.set_attribute(KeyValue::new("gen_ai.response.id", id.to_string()));
    }
    if let Some(model) = response.get("model").and_then(|v| v.as_str()) {
        span.set_attribute(KeyValue::new("gen_ai.response.model", model.to_string()));
    }
    if let Some(usage) = response.get("usage") {
        if let Some(n) = usage.get("input_tokens").and_then(|v| v.as_i64()) {
            span.set_attribute(KeyValue::new("gen_ai.usage.input_tokens", n));
        }
        if let Some(n) = usage.get("output_tokens").and_then(|v| v.as_i64()) {
            span.set_attribute(KeyValue::new("gen_ai.usage.output_tokens", n));
        }
    }
    if let Some(reason) = response.get("stop_reason").and_then(|v| v.as_str()) {
        span.set_attribute(KeyValue::new(
            "gen_ai.response.finish_reasons",
            Value::Array(Array::String(vec![StringValue::from(reason.to_string())])),
        ));
    }
}

pub fn finish_ok<S: Span>(mut span: S) {
    span.set_status(Status::Ok);
    span.end();
}

pub fn finish_err<S: Span>(mut span: S, msg: impl Into<String>) {
    span.set_status(Status::error(msg.into()));
    span.end();
}
