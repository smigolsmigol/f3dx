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

# 5-10x faster agent runtime via concurrent tool dispatch
agent = f3dx.AgentRuntime(system_prompt="...", concurrent_tool_dispatch=True)
result = agent.run(user_prompt, tools={...}, mock_responses=[...])

# Logfire-compatible OTel spans by default
f3dx.configure_otel(
    endpoint="https://logfire-api.pydantic.dev/v1/traces",
    headers={"Authorization": f"Bearer {LOGFIRE_TOKEN}"},
)
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

All benches live under `bench/`, all use the stdlib mock servers in the same dir, all single-thread.

## Architecture

Cargo workspace, four crates, one PyPI package:

```
f3dx/
  crates/
    f3dx-py/      PyO3 bridge cdylib (the only crate with #[pymodule])
    f3dx-rt/      agent runtime + concurrent tool dispatch
    f3dx-http/    LLM HTTP client (reqwest + native SSE + streaming JSON validation)
    f3dx-trace/   OpenTelemetry span emission (Logfire-compatible)
```

OpenAI-compatible endpoints (vLLM, Mistral, xAI, Groq, Together, Fireworks) all work via `f3dx.OpenAI` by setting `base_url`.

## Layout

```
f3dx/
  bench/                     reproducible benches + stdlib mock servers
  crates/                    cargo workspace member crates
  python/f3dx/__init__.py    Python wrapper
  pyproject.toml             maturin build
  Cargo.toml                 cargo workspace root
```

## What's not here yet

- HTTP-level OTel spans on OpenAIClient + AnthropicClient (Phase F V0.1)
- Streaming structured-output validation (Phase E)
- Gemini adapter (Phase C.2)
- Arrow trace store + parquet/DuckDB sinks (Phase G)
- Public PyPI release (after Phase F V0.1 + benchmarks against pydantic-ai's own runtime)

## License

MIT.
