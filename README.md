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

Cargo workspace, five crates, one PyPI package:

```
f3dx/
  crates/
    f3dx-py/      PyO3 bridge cdylib (the only crate with #[pymodule])
    f3dx-rt/      agent runtime + concurrent tool dispatch
    f3dx-http/    LLM HTTP client (reqwest + native SSE + streaming JSON validation)
    f3dx-trace/   OpenTelemetry span emission (Logfire-compatible, gen_ai.* semconv)
    f3dx-mcp/     Model Context Protocol client (rmcp + stdio transport)
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

JSONL trace sink for downstream replay-eval tools:

```python
f3dx.configure_traces("traces.jsonl", capture_messages=True)
# every AgentRuntime.run appends one row with prompt + system_prompt +
# output + input_tokens + output_tokens (capture_messages off by default;
# opt-in because PII-sensitive). Polars/DuckDB scan via pl.scan_ndjson /
# duckdb.read_json. Replay via tracewright.

# Or convert to columnar parquet for fast analytics:
# pip install f3dx[arrow]
from f3dx.analytics import jsonl_to_parquet, tail_jsonl_to_parquet
jsonl_to_parquet("traces.jsonl", "traces.parquet")             # batch convert
# Or live-tail a long-running production process:
tail_jsonl_to_parquet("traces.jsonl", "traces.parquet",
                      poll_seconds=10, batch_size=200,
                      until=lambda: time.time() > deadline)
# pl.scan_parquet("traces.parquet").filter(pl.col("output_tokens") > 100).collect()
```

## Layout

```
f3dx/
  bench/                            reproducible benches + verify scripts + stdlib mock servers
  crates/                           cargo workspace member crates
  python/f3dx/__init__.py           core Python wrapper (AgentRuntime, OpenAI, Anthropic, configure_otel)
  python/f3dx/compat/               opt-in subclass shims (f3dx[openai-compat])
  python/f3dx/pydantic_ai/          pydantic-ai integration (f3dx[pydantic-ai])
  python/f3dx/langchain/            langchain-openai integration (f3dx[langchain])
  pyproject.toml                    maturin build, optional extras
  Cargo.toml                        cargo workspace root + workspace lints
  rust-toolchain.toml               pinned to 1.90.0 for reproducible builds
  .github/workflows/ci.yml          ubuntu/macos/windows + clippy gate + built-wheel install
  .github/workflows/release.yml     glibc/musl x86_64+aarch64 wheels + macos x86_64+aarch64 + windows + sdist + OIDC PyPI publish
```

## What this is not

f3dx is a Python-from-Rust runtime — a Rust core that ships as a Python wheel via PyO3. If you're building a pure Rust application and want an agent framework in your binary, look at [AutoAgents](https://github.com/saivishwak/autoagents) (Rust agent framework with role-based multi-agent), [rig](https://github.com/0xPlaygrounds/rig) (provider abstraction + RAG primitives in Rust), or [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (local inference engine). Different audience, different scope.

f3dx is not an inference engine. Use vLLM, TGI, mistral.rs, llama.cpp, or any OpenAI-compatible endpoint underneath; f3dx talks to them.

f3dx is not a multi-agent orchestration framework. It is the runtime layer below frameworks like pydantic-ai, LangChain, LlamaIndex, CrewAI, AutoGen.

## Sibling project

[`tracewright`](https://github.com/smigolsmigol/tracewright) — replay-driven eval over f3dx and pydantic-ai JSONL traces. Take a recorded trace, swap the candidate model, get a per-case diff. Closes the loop from "we have observability" to "we have regression tests".

## Composition with ATLAS-RTC (Cruz)

```python
# pip install f3dx[atlas-rtc]
from atlas_rtc.adapters.mock_adapter import MockAdapter, MockScenario
from f3dx.atlas_rtc import controlled_completion

result = controlled_completion(
    prompt="Return JSON with name and age.",
    contract=["name", "age"],          # shorthand for JSONSchemaContract(required_keys=...)
    adapter=MockAdapter(scenario),     # or HFAdapter / VLLMAdapter for real models
)
# result.text='{"name":"alice","age":30}', result.valid=True, result.interventions=N
```

[ATLAS-RTC](https://github.com/cruz209/ATLAS-RTC) (Christopher Cruz, MIT) is a runtime control layer that enforces structured outputs at decode time — drift detection + logit masking + rollback during generation. f3dx's runtime sits at a different layer (transport + observability + agent loop). They compose: ATLAS-RTC owns the per-token control loop, f3dx owns the request transport and trace emission. Most useful with local vLLM / HuggingFace where decode-time control is reachable; cloud APIs (OpenAI, Anthropic) don't expose that surface.

## MCP client

```python
import f3dx, json

# spawn an MCP server over stdio (npm-based, Python-based, any binary)
client = f3dx.MCPClient.stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])

for tool in client.list_tools():
    print(tool["name"], tool["description"])

result = client.call_tool("get-sum", json.dumps({"a": 7, "b": 35}))
# 'The sum of 7 and 35 is 42.'
```

`f3dx-mcp` is a sibling cargo crate; the rmcp Rust SDK drives the JSON-RPC handshake + stdio transport. SSE + streamable-HTTP transports + sampling callback bridge land in V0.1.

## Adapter packages

```python
# pip install f3dx[openai-compat]
from f3dx.compat import OpenAI, AsyncOpenAI    # subclass openai.OpenAI / openai.AsyncOpenAI
import openai
client = OpenAI(api_key=...)
isinstance(client, openai.OpenAI)               # True — passes isinstance checks in
                                                # instructor, litellm, smolagents, langchain
out = client.chat.completions.create(...)       # routes through Rust, returns
                                                # openai.types.chat.ChatCompletion

# pip install f3dx[anthropic-compat]
from f3dx.compat import AsyncAnthropic         # subclass anthropic.AsyncAnthropic
client = AsyncAnthropic(api_key=...)           # also intercepts client.beta.messages.create
                                               # for pydantic-ai's BetaMessage validation path

# pip install f3dx[pydantic-ai]
from f3dx.pydantic_ai import openai_model, anthropic_model, F3dxCapability
from pydantic_ai import Agent
cap = F3dxCapability()
agent = Agent(openai_model('gpt-4', api_key=...), capabilities=[cap])
result = await agent.run('hi')                  # f3dx-routed HTTP, capability counts requests
# anthropic_model('claude-haiku-4', api_key=...) likewise

# pip install f3dx[langchain]
from f3dx.langchain import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', api_key=...)    # subclass of langchain_openai.ChatOpenAI
msg = llm.invoke('hi')                          # sync + ainvoke both routed via f3dx
```

## What's not here yet

- Gemini adapter (Phase C.2)
- MCP V0.1: SSE + streamable-HTTP transports + sampling callback bridge (V0 ships stdio only; covers Claude Desktop + every npm-based server + python-based servers via `python -m`)
- Parent-child trace context propagation between AgentRuntime span and HTTP child spans (needs Python-side context bridge)
- Phase E V0.2.1: incremental per-token schema validation in the streaming pump (V0.2 ships terminal-time `output_schema=` via `jsonschema-rs`; per-token needs a streaming JSON parser + schema state machine on top, planned next)
- Phase G V0.3: Rust-side parquet sink (V0.2 ships `AppendingParquetWriter` + `tail_jsonl_to_parquet` Python-side via pyarrow under `f3dx[arrow]`; a Rust-native sink would skip the JSONL middlefile but adds ~30MB to the wheel — deferred unless requested)
- `langchain-f3dx` standalone PyPI package per LangChain partner-package convention (today integrated via the `f3dx[langchain]` extra; standalone-package split happens before LangChain partner-registry submission)

## License

MIT.
