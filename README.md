# f3dx

F3D1 Rust-core runtime for [pydantic-ai](https://github.com/pydantic/pydantic-ai). Polars for agents.

```bash
pip install f3dx
```

```python
import f3dx

# 5x faster streaming, drop-in for openai SDK
client = f3dx.OpenAI(api_key="...", base_url="https://api.openai.com/v1")
for chunk in client.chat_completions_create_stream({"model": "gpt-4", "messages": [...]}):
    print(chunk["choices"][0]["delta"].get("content", ""), end="")

# Drop-in for anthropic SDK with native Messages event handling
client = f3dx.Anthropic(api_key="...")
for event in client.messages_create_stream({"model": "claude-3-5-sonnet", "max_tokens": 1024, "messages": [...]}):
    if event.get("type") == "content_block_delta":
        print(event["delta"].get("text", ""), end="")

# 5-10x faster agent runtime via concurrent tool dispatch
agent = f3dx.AgentRuntime(system_prompt="...", concurrent_tool_dispatch=True)
result = agent.run(user_prompt, tools={...}, mock_responses=[...])

# Tool-call streaming reassembly: skip the accumulate-fragments boilerplate
for ev in client.chat_completions_create_stream_assembled({...}):
    if ev["type"] == "tool_call":
        result = dispatch(ev["name"], ev["arguments"])  # arguments is parsed dict, ready

# Validated structured output: skip accumulate-then-json.loads at end
for ev in client.chat_completions_create_stream_assembled(req, validate_json=True):
    if ev["type"] == "validated_output":
        process(ev["data"])  # already parsed
    elif ev["type"] == "validation_error":
        log.warning("model emitted invalid JSON: %s", ev["error"])

# Logfire-compatible OTel spans by default — gen_ai.* semconv
f3dx.configure_otel(
    endpoint="https://logfire-api.pydantic.dev/v1/traces",
    headers={"Authorization": f"Bearer {LOGFIRE_TOKEN}"},
)
# Every Agent.run + every chat_completions / messages call now emits
# spans with gen_ai.system, gen_ai.request.model, gen_ai.usage.{input,output}_tokens, etc.
```

## Why

Compound AI systems (Zaharia BAIR 2024, Mei AIOS arXiv:2403.16971) are the dominant production pattern. The orchestration + HTTP layer is now the bottleneck, not the model. Every other AI infra layer is non-Python by 2026 (vLLM C++, TGI Rust, mistral.rs Rust, Outlines-core Rust, XGrammar C++). Orchestration is the last lane; f3dx ships it.

## Bench results (reproducible from `bench/`)

| What | vs | Speedup |
|---|---|---|
| `f3dx.AgentRuntime` concurrent dispatch | pure-python sequential agent loop | **5-10x** at 5-10 tools/turn |
| `f3dx.OpenAI` streaming | `openai` Python SDK | **5.10x** at 1000 chunks |
| `f3dx.Anthropic` streaming | `anthropic` Python SDK | **2.9-5.2x** at 50-1000 events |
| Tool-call assembled stream | raw fragment iteration | 17 chunks -> 2 events |
| `validate_json=True` | accumulate + json.loads + try/except | one extra event, zero user code |

All benches live under `bench/`, all use the stdlib mock servers in the same dir, all single-thread.

## Architecture

Cargo workspace, four crates, one PyPI package:

```
f3dx/
  crates/
    f3dx-py/      PyO3 bridge cdylib (the only crate with #[pymodule])
    f3dx-rt/      agent runtime + concurrent tool dispatch
    f3dx-http/    LLM HTTP client (reqwest + native SSE + streaming JSON validation)
    f3dx-trace/   OpenTelemetry span emission (Logfire-compatible, gen_ai.* semconv)
```

OpenAI-compatible endpoints (vLLM, Mistral, xAI, Groq, Together, Fireworks) all work via `f3dx.OpenAI` by setting `base_url`.

## Observability

Configure once with `f3dx.configure_otel(endpoint, headers, service_name, stdout)`. Every `AgentRuntime.run` emits a root span with `gen_ai.system="f3dx"` + `gen_ai.prompt.length_chars` + `f3dx.{concurrent_tool_dispatch,iterations,tool_calls_executed,duration_ms,output.length_chars}`.

Every `chat_completions_create*` / `messages_create*` emits a `SpanKind::Client` span:

```
gen_ai.system               openai | anthropic
gen_ai.operation.name       chat | messages
gen_ai.request.model        from request
gen_ai.request.{temperature, top_p, max_tokens, stream}
gen_ai.response.{id, model, finish_reasons}
gen_ai.usage.{input_tokens, output_tokens}
```

Streaming spans hold open until terminal chunk; usage attrs land when the closing chunk carries them (auto-injects `stream_options.include_usage=true` for OpenAI; reads `message_start.message.usage` + `message_delta.usage` for Anthropic).

Status: `Ok` on success, `Status::error("<msg>")` on HTTP failure.

## Layout

```
f3dx/
  bench/                     reproducible benches + stdlib mock servers
  crates/                    cargo workspace member crates
  python/f3dx/__init__.py    Python wrapper
  pyproject.toml             maturin build
  Cargo.toml                 cargo workspace root
  .github/workflows/ci.yml   ubuntu/macos/windows + python 3.12 + bench-as-test
```

## What's not here yet

- Gemini adapter (Phase C.2)
- Parent-child trace context propagation between AgentRuntime span and HTTP child spans (needs Python-side context bridge)
- jsonschema validation in `validate_json` mode (V0 only checks parseable JSON; Pydantic schema check coming)
- True fail-fast incremental JSON validation (V0 validates at terminal; V0.2 will use a streaming JSON parser state machine)
- Arrow trace store + parquet/DuckDB sinks (Phase G)
- Public PyPI release (after benchmarks against pydantic-ai's own runtime)

## License

MIT.
