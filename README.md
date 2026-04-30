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

# Logfire-compatible OTel spans by default - gen_ai.* semconv
f3dx.configure_otel(
    endpoint="https://logfire-api.pydantic.dev/v1/traces",
    headers={"Authorization": f"Bearer {LOGFIRE_TOKEN}"},
)
# Every Agent.run + every chat_completions / messages call now emits
# spans with gen_ai.system, gen_ai.request.model, gen_ai.usage.{input,output}_tokens, etc.
```

## f3dx.cache + f3dx.router (bundled)

`f3dx.cache` is content-addressable LLM response cache + replay (sub-100us warm hit). `f3dx.router` is the in-process Rust router with sequential + hedged-parallel policies. Both shipped as separate PyPI packages until 2026-04-30; now bundled into the f3dx wheel.

```python
from f3dx.cache import Cache, cached_call

cache = Cache("eval_cache.redb")

# Wrap any closed-API call: first run records, subsequent replay deterministically
def fetch(req):
    return openai.OpenAI().chat.completions.create(**req).model_dump()

resp = cached_call(cache, request=req, fetch=fetch)
# Real measured: cold OpenAI call 4647ms, warm cache.peek 3.8us = 1,222,949x speedup

# Tool-result memoization: 50KB Read 223x speedup, gh CLI 111,415x speedup
from f3dx.cache import cache_tool_call, FileWitness, TTLWitness

result = cache_tool_call(
    cache, tool="Read", args={"path": "/abs/lib.py"},
    fetch=lambda a: open(a["path"]).read(),
    witness=FileWitness(["/abs/lib.py"]),  # mtime-based invalidation
)
```

```python
from f3dx.router import Router

router = Router(
    providers=[
        {"name": "openai", "kind": "openai", "base_url": "https://api.openai.com/v1", "api_key": "sk-..."},
        {"name": "groq",   "kind": "openai", "base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_..."},
    ],
    policy="hedged",  # "sequential" for failover; "hedged" for fastest-wins
    hedge_k=2,
)
response = router.chat_completions({"model": "gpt-4", "messages": [...]})
```

## f3dx.fast: client-side inference acceleration

Three pillars from the f3d1-fast thesis (six-pillar plan to cut Claude-Code-style 2-3 minute agentic responses to <1.5 min, software-only, against any closed API). All three real-API validated against OpenAI gpt-4o-mini with reproducible fixture-backed benches.

```python
from f3dx.fast import (
    CanonicalPrompt,        # Pillar 2: prefix-cache canonicalization
    cache_hit_ratio,
    budget_max_tokens,      # Pillar 6: token budget hinting from history
    estimate_from_history,
)
from f3dx.cache import cache_tool_call  # Pillar 3: tool-result memoization
```

| Pillar | What | Real-API measured |
|---|---|---|
| **2** prefix-cache canonicalization | Build prompts in static-first order, BLAKE3 prefix hash, model-aware cache thresholds | **91.1% cache hit rate, 45.6% input cost reduction, 3x TTFT** (gpt-4o-mini, 1280/1405 tokens cached) |
| **3** tool-result memoization | Memoize side-effect-free tool calls with FileWitness / TTLWitness invalidation | **223x** speedup on 50KB Read, **111,415x** on `gh run list` cached at 30s TTL |
| **6** token budget hinting | `max_tokens = ceil(p99(prior_counts) * safety_factor)` to kill the runaway-generation tail | **3,814 tokens of headroom saved per call** vs default 4096 on a runaway-prone prompt, **94% worst-case cost reduction**, zero truncations |

Reproduce any of these with `python examples/{prompt_canonical,tool_cache,budget}_real_api_bench.py`. All numbers are from `bench/fixtures/openai.redb` so re-runs replay deterministically without an API key. CI runs with `F3DX_BENCH_OFFLINE=1` so cache miss is a test failure, never a silent live hit. Convention doc: `docs/workflows/real_api_benches.md`.

Pillars not yet shipped: 4 (speculative tool execution, ICLR 2026 oral, 2-week build, 30-50% wall-clock cut), 5 (free-wins bundle, deprioritized for client-side scope), 1 (year-2 anchor candidate, hybrid local-target spec decoding via mistral.rs).

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

Cargo workspace, seven crates, one PyPI package:

```
f3dx/
  crates/
    f3dx-py/      PyO3 bridge cdylib (the only crate with #[pymodule])
    f3dx-rt/      agent runtime + concurrent tool dispatch
    f3dx-http/    LLM HTTP client (reqwest + native SSE + streaming JSON validation)
    f3dx-trace/   OpenTelemetry span emission (Logfire-compatible, gen_ai.* semconv)
    f3dx-mcp/     Model Context Protocol client (rmcp + stdio transport)
    f3dx-cache/   content-addressable LLM cache (redb + RFC 8785 JCS + BLAKE3)
    f3dx-replay/  trace replay primitives (diff modes, JSONL reader)
    f3dx-router/  in-process LLM provider router (sequential + hedged-parallel)
```

`f3dx-cache`, `f3dx-replay`, and `f3dx-router` were standalone PyPI packages until 2026-04-30; now consolidated into the f3dx wheel via `f3dx.cache` and `f3dx.router` Python sub-modules. Old PyPI packages remain as deprecation shims that re-export from the new home.

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

f3dx is a Python-from-Rust runtime - a Rust core that ships as a Python wheel via PyO3. If you're building a pure Rust application and want an agent framework in your binary, look at [AutoAgents](https://github.com/saivishwak/autoagents) (Rust agent framework with role-based multi-agent), [rig](https://github.com/0xPlaygrounds/rig) (provider abstraction + RAG primitives in Rust), or [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (local inference engine). Different audience, different scope.

f3dx is not an inference engine. Use vLLM, TGI, mistral.rs, llama.cpp, or any OpenAI-compatible endpoint underneath; f3dx talks to them.

f3dx is not a multi-agent orchestration framework. It is the runtime layer below frameworks like pydantic-ai, LangChain, LlamaIndex, CrewAI, AutoGen.

## Sibling projects

The f3d1 ecosystem alongside f3dx:

- [`tracewright`](https://github.com/smigolsmigol/tracewright) - `pip install tracewright`. Trace-replay adapter for [`pydantic-evals`](https://ai.pydantic.dev/evals/). Read an f3dx or pydantic-ai logfire JSONL trace, get a `pydantic_evals.Dataset` you can run any pydantic-evals evaluator against. Closes the loop from "we have observability" to "we have regression tests".
- [`pydantic-cal`](https://github.com/smigolsmigol/pydantic-cal) - `pip install pydantic-cal`. Calibration metrics for pydantic-evals: ECE, smECE, MCE, ACE, Brier, reliability diagrams, Murphy 1973 decomposition, temperature/Platt/isotonic scaling, Fisher-Rao geometry kernel. The calibration layer the eval world is missing.
- [`llmkit`](https://github.com/smigolsmigol/llmkit) - `pip install llmkit-sdk` or `npx @f3d1/llmkit-cli`. Hosted API gateway with budget enforcement, session tracking, cost dashboards, MCP server. The hosted complement to f3dx.router's in-process surface.

`f3dx-cache` and `f3dx-router` used to be standalone packages here. They consolidated into f3dx on 2026-04-30 (Phase B + C of the f3d1 repo consolidation). Old PyPI packages still resolve as deprecation shims; new code should `from f3dx.cache import Cache` and `from f3dx.router import Router`.

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

[ATLAS-RTC](https://github.com/cruz209/ATLAS-RTC) (Christopher Cruz, MIT) is a runtime control layer that enforces structured outputs at decode time - drift detection + logit masking + rollback during generation. f3dx's runtime sits at a different layer (transport + observability + agent loop). They compose: ATLAS-RTC owns the per-token control loop, f3dx owns the request transport and trace emission. Most useful with local vLLM / HuggingFace where decode-time control is reachable; cloud APIs (OpenAI, Anthropic) don't expose that surface.

```python
# pip install f3dx[vigil]
from f3dx.vigil import f3dx_jsonl_to_vigil_events

f3dx.configure_traces("traces.jsonl", capture_messages=True)
# ... agent runs ...
f3dx_jsonl_to_vigil_events("traces.jsonl", "events.jsonl", actor="robin_a")
# Robin B (cruz209/V.I.G.I.L) reads events.jsonl, builds Roses/Buds/Thorns
# diagnosis, proposes prompt + code adaptations.
```

[V.I.G.I.L / Robin B](https://github.com/cruz209/V.I.G.I.L) is the reflective-supervisor sibling: reads a JSONL event log, builds an "emotional bank" appraisal (Roses / Buds / Thorns), diagnoses reliability issues, proposes prompt + code patches. f3dx provides the runtime that produces the trace; this bridge converts the trace into VIGIL's expected event shape.

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

`f3dx-mcp` is a sibling cargo crate; the rmcp Rust SDK drives the JSON-RPC handshake. Stdio + streamable-HTTP transports + sampling-callback bridge ship today; SSE-only transport (rare in practice - streamable-HTTP subsumes it for MCP) skipped.

**Sampling callback** - the MCP server can ask the connected client for a model completion via `sampling/createMessage`. Pass a Python callback to `MCPClient.stdio` / `streamable_http` and it fires on every such request:

```python
def my_sampling(messages_json: str, system_prompt: str) -> str:
    # messages_json is the serialized rmcp message list; reach for whatever
    # field the request exposes. Run any model - f3dx.OpenAI, f3dx.Anthropic,
    # pydantic-ai Agent, ATLAS-RTC controlled_completion - and return text.
    return run_my_model(messages_json, system_prompt)

client = f3dx.MCPClient.stdio(
    "python", ["-m", "my_mcp_server"],
    sampling_callback=my_sampling,
)
```

Without a callback, sampling requests get the standard "method not supported" error.

**Server-side** - expose Python callables AS MCP tools that other MCP clients (Claude Desktop, IDE plugins, other f3dx-built clients) can call:

```python
import f3dx, json

def add(args_json: str) -> str:
    args = json.loads(args_json)
    return str(args["a"] + args["b"])

server = f3dx.MCPServer(name="my-server", version="0.0.1")
server.add_tool(
    "add",
    add,
    description="Add two numbers.",
    input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
)
server.serve_stdio()  # blocks until client closes
```

f3dx now ships the full bidirectional MCP surface: client (stdio + streamable-HTTP), server (stdio), and sampling-callback bridge so server-issued completions route back through user-controlled model code.

## Adapter packages

```python
# pip install f3dx[openai-compat]
from f3dx.compat import OpenAI, AsyncOpenAI    # subclass openai.OpenAI / openai.AsyncOpenAI
import openai
client = OpenAI(api_key=...)
isinstance(client, openai.OpenAI)               # True - passes isinstance checks in
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
- f3d1-fast Pillar 4: speculative tool execution (Sutradhara streaming JSON parser hooks SSE, fires sandboxed tools optimistically when tool_use chunks parse cleanly, rollback on retract). 2-week build, ICLR 2026 oral validates the approach, 30-50% wall-clock cut reported.
- f3d1-fast Pillar 1 (year-2 candidate): hybrid local-target speculative decoding via mistral.rs + Qwen2.5-Coder. 6-8 week build, 5-8x throughput on agentic loops where ~80% of turns can route to a local 7B target.
- f3dx.fast.CanonicalPrompt V0.1: Anthropic real-API validation with explicit `cache_control` markers + auto-tuning (measure hit rate, nudge marker placement). Depends on wata schema cache columns.
- Parent-child trace context propagation between AgentRuntime span and HTTP child spans (needs Python-side context bridge).
- Phase E V0.2.2: full streaming JSON parser (V0.2.1 ships fail-fast on invalid JSON prefix; bracket-balance + per-token schema FSM are the next steps).
- Phase G V0.3: Rust-side parquet sink (V0.2 ships `AppendingParquetWriter` + `tail_jsonl_to_parquet` Python-side via pyarrow under `f3dx[arrow]`; Rust-native sink would skip the JSONL middlefile but adds ~30MB to the wheel - deferred unless requested).
- `langchain-f3dx` standalone PyPI package per LangChain partner-package convention (today integrated via the `f3dx[langchain]` extra; standalone-package split happens before LangChain partner-registry submission).

## License

MIT.
