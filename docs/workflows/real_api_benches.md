# Real-API benches in the f3d1 ecosystem

This is the f3d1-wide convention for any code that calls a real LLM
provider (OpenAI / Anthropic / Gemini / others) for benchmarking,
demonstration, calibration, or integration testing.

## The rule

**If a number leaves the screen and is not reproducible by another
person, it didn't happen.**

Concretely: every bench, every example, every integration test that
hits a closed-API provider routes its calls through `f3dx.cache.cached_call`,
backed by a fixture file committed to the repo. First run records;
subsequent runs replay deterministically. CI never hits the live API.

This kills three problems at once:
1. **Reproducibility.** Anyone can clone the repo and re-run the bench
   without an API key, getting the same numbers we got.
2. **Throttling.** Soft rate limits below the documented TPM/RPM bands
   (Iriden, 2026-04-30) make raw bench loops flaky. Cache replay sees
   zero load.
3. **Cost.** Calibration runs that need 1000+ calls (pydantic-cal)
   become a one-time spend, not a per-rerun spend.

## The canonical pattern

```python
from pathlib import Path
from f3dx.cache import Cache, cached_call

fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
fixture_path.parent.mkdir(parents=True, exist_ok=True)
fixture = Cache(str(fixture_path))


def _fetch_openai(request: dict) -> dict:
    """Real API call. Returns a JSON-serializable dict."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(**request)
    return resp.model_dump()


request = {
    "model": "gpt-4o-mini",
    "messages": [...],
    "temperature": 0.0,
    "max_tokens": 200,
}

response = cached_call(fixture, request, _fetch_openai, model="gpt-4o-mini")
# First run: hits OpenAI, records to fixture.
# Subsequent runs: ~22us replay, no network, no API key needed.
```

The cache key is the JSON-canonicalized request dict (RFC 8785 JCS +
BLAKE3, identical to f3dx-cache's internal fingerprint). Two requests
with the same fields in different orders hash identically. Any change
to model, prompt, max_tokens, temperature, etc. is a new cache key.

## Three modes via env vars

```
$ python examples/foo_real_api_bench.py            # default: replay if cached, else hit API
$ F3DX_BENCH_REFRESH=1 python examples/foo_real_api_bench.py  # bust cache, re-record
$ F3DX_BENCH_OFFLINE=1 python examples/foo_real_api_bench.py  # cache miss = LookupError
```

- **Default** (no env var): cache-first, fall through to live API on miss.
  This is what devs run locally during development.
- **REFRESH=1**: bypass the cache, force a live API call, overwrite the
  cached entry with the new response. Use when intentionally re-recording
  the fixture (e.g. after the upstream model changes behavior, or after
  changing the request shape). Requires API key.
- **OFFLINE=1**: cache miss raises `LookupError`. Use in CI: workflows
  pass `F3DX_BENCH_OFFLINE=1` so test runs fail loudly if the fixture
  doesn't cover a request, instead of silently hitting the live API
  during automated runs.

## The fixture file

Commit `bench/fixtures/<provider>.redb` to the repo. redb is a single
file, binary, ~5KB per cached entry, git-friendly at this scale (tens
of MB total before it becomes a concern). Diffs are opaque but
attribution is preserved through the commit history.

Naming convention: one fixture per provider, not per bench. All OpenAI
calls across all benches in a repo share `bench/fixtures/openai.redb`,
so cache hits compose across benches and the fixture file stays small.

When refreshing: run with `F3DX_BENCH_REFRESH=1`, verify the new
numbers look right, then commit the updated `.redb` with a commit
message that names which benches got refreshed and why.

## CI integration

Every CI job that runs a bench should set `F3DX_BENCH_OFFLINE=1` in
the workflow yaml:

```yaml
- name: run benches
  env:
    F3DX_BENCH_OFFLINE: "1"
  run: |
    python examples/cache_real_api_bench.py
    python examples/router_real_api_bench.py
    python examples/budget_real_api_bench.py
```

Cache miss in CI = test failure, not silent live-API hit. This means
CI never has access to API keys (good security posture) and bench
correctness becomes a function of the committed fixture, not network
state.

## When to refresh

Refresh the fixture when:
- The upstream model changes behavior (vendor announcement, your
  observed-output diverges from what's cached)
- The bench's request shape changes (new field, different prompt,
  different temperature)
- A new bench adds requests that aren't in the fixture yet
- You're intentionally re-validating the published numbers in the
  bench README / blog / paper

Don't refresh on every PR. The fixture is a release artifact, not a
build artifact.

## Honest replay

Latency numbers from REPLAYED runs are the recorded latency from the
cold call, not a fresh measurement. The bench output should make this
explicit:

```
mode: replay
[1/3] Cold call (real API or fixture replay)...
  cold: 4647 ms (recorded)  (672 bytes)
```

If the bench measures cache-hit latency (e.g. `cache.peek()` warm-path),
that measurement IS fresh on every run because the cache hit is a real
in-process operation, not a network call. Distinguish carefully in the
output.

## Example benches in this repo

All three follow this pattern. Read them for the structure:

- `examples/cache_real_api_bench.py` -- f3dx.cache speedup vs cold call
- `examples/router_real_api_bench.py` -- f3dx.router end-to-end
- `examples/budget_real_api_bench.py` -- f3dx.fast.budget hinter

## Roll-forward across the f3d1 ecosystem

Other repos that hit closed APIs (or will, soon):

| Repo | Real-API surface | Fixture path |
|---|---|---|
| `pydantic-cal` | calibration datasets (1000+ calls) | `bench/fixtures/<provider>.redb` |
| `tracewright` | trace replay + scorer integration tests | same convention |
| `f3dx-bench` | dashboard demo screenshots | same convention |
| `f3d1-volta` | Phase 1+ executor backend tests | same convention |
| `f3d1-wata` | Phase 1+ schema validation runs | same convention |

All of these depend on `f3dx>=0.0.18` (which carries `f3dx.cache` since
the Phase B consolidation on 2026-04-30) for free.

## What this is NOT

- Not a replacement for production caching. f3dx.cache works for that
  too, but the bench fixture is a separate fixture file with a
  different lifecycle.
- Not a free pass to ship synthetic data as "real numbers". The
  fixture is a recording of REAL API behavior; refresh it when you
  need new data, don't fabricate.
- Not a way to avoid having an API key forever. Devs need a key when
  they refresh; readers don't need one to replay.
- Not the f3d1-fast Pillar 3 (tool-result memoization). That extends
  the SAME cache substrate to mid-agentic-loop tool calls. The bench
  fixture pattern is its precursor and shares no risk.
