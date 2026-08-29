# f3dx

[![CI](https://github.com/smigolsmigol/f3dx/actions/workflows/ci.yml/badge.svg)](https://github.com/smigolsmigol/f3dx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/f3dx)](https://pypi.org/project/f3dx/)
[![Python](https://img.shields.io/pypi/pyversions/f3dx)](https://pypi.org/project/f3dx/)
[![Rust](https://img.shields.io/badge/core-Rust-000000?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/github/license/smigolsmigol/f3dx)](LICENSE)

Rust execution primitives for Python AI systems.

f3dx moves selected runtime work below the Python application layer: model HTTP and SSE transport, concurrent tool dispatch, MCP, trace emission, response caching, and provider routing. It ships as a PyO3 wheel for Python 3.10 and newer.

It does not replace Pydantic AI, LangChain, or your application. Those layers still own agent behavior. f3dx is an opt-in runtime beneath them.

## Choose a surface

| Need | Import | Install |
| --- | --- | --- |
| Pydantic AI model transport | `f3dx.pydantic_ai` | `f3dx[pydantic-ai]` |
| OpenAI SDK-compatible client | `f3dx.compat.OpenAI` | `f3dx[openai-compat]` |
| Anthropic SDK-compatible client | `f3dx.compat.AsyncAnthropic` | `f3dx[anthropic-compat]` |
| Direct runtime and native clients | `f3dx.AgentRuntime`, `f3dx.OpenAI`, `f3dx.Anthropic` | `f3dx` |
| MCP client and server | `f3dx.MCPClient`, `f3dx.MCPServer` | `f3dx` |
| Content-addressed response cache | `f3dx.cache` | `f3dx[cache]` |
| In-process provider router | `f3dx.router` | `f3dx[router]` |

```bash
pip install f3dx
```

## Pydantic AI

The Pydantic AI adapter builds normal Pydantic AI models while routing requests through the f3dx transport:

```python
from pydantic_ai import Agent

from f3dx.pydantic_ai import F3dxCapability, openai_model

capability = F3dxCapability()
agent = Agent(
    openai_model(
        "gpt-4.1-mini",
        api_key="...",
        base_url="https://api.openai.com/v1",
    ),
    capabilities=[capability],
)

result = await agent.run("Explain why the request failed.")
print(result.output)
print(capability.model_requests)
```

The same package exposes `anthropic_model`.

## SDK-compatible transport

Use the compatibility layer when a library expects an actual upstream SDK type:

```python
from f3dx.compat import OpenAI

client = OpenAI(api_key="...")
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Give me one sentence."}],
)

print(response.choices[0].message.content)
```

`f3dx.compat.OpenAI` subclasses `openai.OpenAI`; responses use upstream OpenAI response types. Sync and streaming requests are exercised against local protocol fixtures in CI. Async OpenAI and Anthropic compatibility shims are available through the matching extras.

OpenAI-compatible providers can be selected with `base_url`.

## Agent runtime

`AgentRuntime` is a bounded execution loop with configurable iteration and tool-call limits. Independent tool calls from one model turn can run concurrently. The runtime consumes caller-supplied model-turn payloads and dispatches Python callables, which keeps orchestration deterministic and testable.

It is not an end-to-end agent client. Provider clients and framework adapters remain separate so model transport is never hidden inside the loop. The executable fixture in [`bench/bench_concurrent.py`](bench/bench_concurrent.py) shows the complete contract.

For runs that need crash recovery evidence, pass `session_journal_path="run.journal"`. The native journal records bounded metadata for run starts, model turns, tool-result ids, and completion. It never captures prompts or tool arguments, repairs only a partial final frame, and rejects complete frames whose hash chain or event JSON is invalid. The focused consumer check is [`bench/verify_session_journal.py`](bench/verify_session_journal.py).

On Windows, an application that already owns a child process can also pass its raw `process_handle`. f3dx places that process in a Job Object for the run lifetime; closing the lease reaps descendants. The caller retains ownership of the process handle, and the option is unsupported unless explicitly requested on Windows.

`f3dx.SessionJournal(path, session_id)` is the small inspection and append surface for applications that need to read the verified records directly. Its `validate_json()` report is intentionally diagnostic: duplicate effect ids and cross-session or dangling references are surfaced for the caller to gate. A journal path is single-writer; this slice does not pretend to coordinate multiple processes or make effects idempotent.

## MCP

```python
import json

import f3dx

client = f3dx.MCPClient.stdio(
    "npx",
    ["-y", "@modelcontextprotocol/server-everything"],
)

for tool in client.list_tools():
    print(tool["name"])

print(client.call_tool("get-sum", json.dumps({"a": 7, "b": 35})))
```

The current wheel includes an MCP client for stdio and streamable HTTP, a stdio MCP server, and a callback boundary for server-issued sampling requests.

## Trace and replay boundary

The native extension can append runtime records to JSONL. Prompt and output capture is opt-in because those fields can contain credentials, personal data, or customer content. The exact capture contract is exercised in [`bench/verify_capture_messages.py`](bench/verify_capture_messages.py).

[`tracewright`](https://github.com/smigolsmigol/tracewright) turns enriched f3dx rows into replay cases or a Pydantic Evals dataset. Trace configuration is currently a low-level native API, so this README does not present it as a stable top-level convenience function.

For local analytics, `f3dx[arrow]` adds JSONL-to-Parquet helpers under `f3dx.analytics`.

## Cache and router

The wheel also contains two independent runtime components:

- `f3dx.cache`: content-addressed response storage backed by redb, with canonical JSON request keys
- `f3dx.router`: sequential and hedged provider policies with retry and failure routing

They are opt-in modules, not global runtime state. An application can use either without adopting `AgentRuntime`.

## Evidence

The `bench/` directory contains executable protocol and runtime fixtures for:

- OpenAI and Anthropic SDK compatibility
- sync and streaming response types
- Pydantic AI integration
- sequential versus concurrent tool dispatch
- MCP client, server, and sampling callbacks
- OTel and JSONL trace emission
- cache, routing, and replay integration

These fixtures use local mock servers or deterministic model-turn payloads. They measure transport and orchestration overhead, not internet latency or model inference speed. Benchmark ratios are therefore development evidence, not a claim that an end-to-end agent will become a fixed multiple faster.

CI builds and installs the wheel on Linux, macOS, and Windows, then runs Rust formatting and lint checks plus the Python verification scripts.

## Architecture

```text
Python application or framework
        |
        v
PyO3 package: f3dx
        |
        +-- f3dx-rt       bounded agent loop and tool dispatch
        +-- f3dx-http     HTTP and SSE transport
        +-- f3dx-trace    OpenTelemetry and JSONL evidence
        +-- f3dx-mcp      MCP transports and callbacks
        +-- f3dx-cache    response cache
        +-- f3dx-router   provider selection
        +-- f3dx-session  crash-safe session journal and Windows process lease
```

The model endpoint remains external. f3dx is not an inference engine, a hosted gateway, or a multi-agent product.

## Current boundaries

- Native clients return f3dx types; compatibility clients return upstream SDK types.
- Trace capture is process-local and must be enabled explicitly.
- `AgentRuntime` coordinates supplied model turns; it does not select or host a model.
- Session journaling is opt-in and records metadata only; it is not a prompt archive or an idempotency service.
- Local mock benchmarks isolate runtime overhead and do not predict provider behavior.
- Optional integrations can change as upstream SDK contracts change; the current CI gate tests the resolved dependency set on every supported operating system.

## Development

```bash
python -m pip install maturin pytest openai anthropic pydantic-ai langchain-openai tracewright
maturin develop --release
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
python bench/verify_compat.py
python bench/verify_pydantic_ai.py
```

See `.github/workflows/ci.yml` for the complete cross-platform gate.

Related: [tracewright](https://github.com/smigolsmigol/tracewright) consumes replayable traces. [LLMKit](https://github.com/smigolsmigol/llmkit) is a separate hosted gateway and cost-control project.

MIT licensed.
