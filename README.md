# f3dx

The Rust runtime your Python imports. Drop-in for `openai` and `anthropic` SDKs with native SSE streaming, an agent loop with concurrent tool dispatch, and Logfire-compatible OTel emission. PyO3 + abi3 wheels for ubuntu/macos/windows. Built for [pydantic-ai](https://github.com/pydantic/pydantic-ai).

The intellectual frame is Cruz's "AI Runtime Infrastructure" ([arXiv:2603.00495](https://arxiv.org/abs/2603.00495), Feb 2026): a distinct execution-time layer above the model and below the application that observes, reasons over, and intervenes in agent behavior at runtime. f3dx is that layer, in Rust, for Python apps.

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

## What this is not

f3dx is a Python-from-Rust runtime — a Rust core that ships as a Python wheel via PyO3. If you're building a pure Rust application and want an agent framework in your binary, look at [AutoAgents](https://github.com/saivishwak/autoagents) (Rust agent framework with role-based multi-agent), [rig](https://github.com/0xPlaygrounds/rig) (provider abstraction + RAG primitives in Rust), or [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (local inference engine). Different audience, different scope.

f3dx is not an inference engine. Use vLLM, TGI, mistral.rs, llama.cpp, or any OpenAI-compatible endpoint underneath; f3dx talks to them.

f3dx is not a multi-agent orchestration framework. It is the runtime layer below frameworks like pydantic-ai, LangChain, LlamaIndex, CrewAI, AutoGen.

## What's not here yet

- Gemini adapter (Phase C.2)
- Parent-child trace context propagation between AgentRuntime span and HTTP child spans (needs Python-side context bridge)
- jsonschema validation in `validate_json` mode (V0 only checks parseable JSON; Pydantic schema check coming)
- True fail-fast incremental JSON validation (V0 validates at terminal; V0.2 will use XGrammar as the streaming validator backend)
- Arrow trace store + parquet/DuckDB sinks (V0.1 of Phase G)
- Adapter packages: `f3dx[openai-compat]` (subclass shim for openai.OpenAI isinstance compatibility), `f3dx[pydantic-ai]` (Capability + WrapperModel sub-package), `langchain-f3dx` (separate PyPI package per LangChain partner-package convention)
- Public PyPI release (gated only on PyPI trusted-publisher config — see release workflow)

## License

MIT.
