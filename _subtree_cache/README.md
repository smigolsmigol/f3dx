# f3dx-cache

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/smigolsmigol/f3dx-cache/badge)](https://scorecard.dev/viewer/?uri=github.com/smigolsmigol/f3dx-cache)

LLM testing burns money. A 200-test suite re-running the same prompts against gpt-4o costs $4-20/run; a 2000-test suite costs more than a junior dev hour. Salesforce published a $500k/year saving from a mock-LLM CI rig in 2024. Every team running evals pays this tax.

`f3dx-cache` is the Rust + redb cache that makes the tax go to zero. Identical (model, messages, tools, temp) requests fingerprint identically via RFC 8785 JCS + BLAKE3 and the cached response returns at <100µs. First test run records; every subsequent run replays. The model never gets called twice for the same input.

```bash
pip install f3dx-cache
```

```python
import pytest
from f3dx_cache import Cache

@pytest.mark.f3dx_cache
def test_extract_invoice_total(f3dx_cache_obj: Cache):
    req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "..."}]}
    cached = f3dx_cache_obj.get(req)
    if cached is None:
        resp = call_openai(req)            # only on cold-cache; happens once
        f3dx_cache_obj.put(req, resp.body, model=resp.model)
        cached = resp.body
    assert b'"total":42' in cached
```

`pytest -p f3dx_cache` autoloads the plugin. Cold run: real OpenAI calls + records. Warm run: zero token cost, ~5ms total runtime.

## Why redb + JCS + BLAKE3

`redb` is the only embedded ACID store in pure Rust with PyO3-friendly abi3 wheels (no C toolchain in the wheel). RocksDB and LMDB carry C dependencies that break the cross-platform wheel story.

RFC 8785 [JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785) sorts object keys + normalizes numeric forms so semantically identical requests collide. Without canonicalization, `{"model":"gpt-4","temp":0}` and `{"temp":0,"model":"gpt-4"}` would fingerprint differently and the cache would miss.

[BLAKE3](https://github.com/BLAKE3-team/BLAKE3) is faster than SHA-256 by 10-30x and the gold standard for content addressing in 2026.

## Architecture

```
f3dx-cache/
  crates/
    f3dx-cache/      core: redb tables, JCS canonicalize, BLAKE3 fingerprint
    f3dx-replay/     read JSONL/parquet trace bundles, diff outputs
    f3dx-cache-py/   PyO3 bridge cdylib (the only crate with #[pymodule])
  python/
    f3dx_cache/
      __init__.py        Cache class wrapping the native PyO3 surface
      pytest_plugin.py   pytest11 entry point: @pytest.mark.f3dx_cache
```

Three redb tables in one file:
- `requests`  fingerprint -> canonicalized request bytes
- `responses` fingerprint -> response bytes
- `meta`      fingerprint -> {created_at_ms, hit_count, model, system_fingerprint, response_duration_ms}

## Replay layer

`f3dx-replay` reads a JSONL trace bundle (the same shape `f3dx_trace::emit_trace_row` writes) and diffs outputs against a target config. Layered determinism modes:

| Mode | What it compares | Cost | Use case |
|------|------------------|------|----------|
| `bytes` | exact byte equality | <1µs | structured-output tests where canonical JSON is expected |
| `structured` | parsed JSON post-canonicalization | ~10µs | tool-call extraction, agent step output |
| `embedding` | embedding-cosine under threshold | ~10ms | natural-language responses (V0.1) |
| `judge` | LLM-as-judge call | ~500ms | semantic correctness checks (V0.1) |

```python
from f3dx_cache import diff, read_jsonl

rows = read_jsonl("traces.jsonl")
for row in rows:
    new_output = replay_against(row["model"], row["prompt"])
    ok, note = diff(row["output"], new_output, mode="structured")
    if not ok:
        print(f"regression on {row['trace_id']}: {note}")
```

## Layout

```
f3dx-cache/
  crates/                 cargo workspace (3 crates)
  python/f3dx_cache/      Python wrapper + pytest plugin
  pyproject.toml          maturin build, pytest11 entry-point
  Cargo.toml              cargo workspace root + workspace lints
  rust-toolchain.toml     pinned to 1.90.0
  tests/                  integration tests
  bench/                  reproducible benches
  examples/               drop-in patterns for f3dx, openai, anthropic SDKs
```

## What this is not

`f3dx-cache` is a TEST-MODE primitive. Production traffic does not point at it. The headline win is `$0 + 2sec` CI runs, not `$0 + 2sec` for end users.

`f3dx-cache` is not a semantic cache. [GPTCache](https://github.com/zilliztech/GPTCache) does embedding-similarity matching for production cache hits - wrong abstraction for testing, where false positives mask regressions.

`f3dx-cache` is not tied to f3dx. Any LLM SDK that emits JSON-shaped requests works (openai, anthropic, instructor, litellm, langchain, ai-sdk).

## Sibling projects

The f3d1 ecosystem:

- [`f3dx`](https://github.com/smigolsmigol/f3dx) - Rust runtime your Python imports. Drop-in for openai + anthropic SDKs with native SSE streaming, agent loop with concurrent tool dispatch, OTel emission. `pip install f3dx`.
- [`tracewright`](https://github.com/smigolsmigol/tracewright) - Trace-replay adapter for `pydantic-evals`. Read an f3dx or pydantic-ai logfire JSONL trace, get a `pydantic_evals.Dataset`. `pip install tracewright`.
- [`pydantic-cal`](https://github.com/smigolsmigol/pydantic-cal) - Calibration metrics for `pydantic-evals`: ECE, MCE, ACE, Brier, reliability diagrams, Fisher-Rao geometry kernel. `pip install pydantic-cal`.
- [`f3dx-router`](https://github.com/smigolsmigol/f3dx-router) - In-process Rust router for LLM providers. Hedged-parallel + 429/5xx hot-swap. `pip install f3dx-router`.
- [`f3dx-bench`](https://github.com/smigolsmigol/f3dx-bench) - Public real-prod-traffic LLM benchmark dashboard. CF Worker + R2 + duckdb-wasm. [Live](https://f3dx-bench.pages.dev).
- [`llmkit`](https://github.com/smigolsmigol/llmkit) - Hosted API gateway with budget enforcement, session tracking, cost dashboards, MCP server. [llmkit.sh](https://llmkit.sh).
- [`keyguard`](https://github.com/smigolsmigol/keyguard) - Security linter for open source projects. Finds and fixes what others only report.

## License

MIT.
