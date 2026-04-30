"""Real-API benchmark for f3dx.cache against OpenAI GPT-4o-mini.

Cache-backed via the canonical f3dx convention (see
docs/workflows/real_api_benches.md). First run hits OpenAI; all subsequent
runs replay deterministically from `bench/fixtures/openai.redb`. CI runs
this with F3DX_BENCH_OFFLINE=1 to forbid live calls. Refresh the fixture
with F3DX_BENCH_REFRESH=1 (API key required).

What this bench measures: cold OpenAI call latency (recorded once), then
the speedup of `cache.get()` and `cache.peek()` against that recorded
baseline. The "cold" number on the FIRST run is the real network hit;
on subsequent runs it's the recorded duration metadata, so the speedup
ratio stays stable across re-runs.

Run:
    python examples/cache_real_api_bench.py                       # replay-default
    F3DX_BENCH_REFRESH=1 python examples/cache_real_api_bench.py  # re-record
    F3DX_BENCH_OFFLINE=1 python examples/cache_real_api_bench.py  # CI strict
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from f3dx.cache import Cache, cached_call


def _fetch_openai(request: dict) -> dict:
    """Real OpenAI call wrapped to JSON-roundtrip-clean."""
    from openai import OpenAI

    client = OpenAI()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**request)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    payload = resp.model_dump()
    payload["__bench_cold_ms"] = elapsed_ms
    return payload


def main() -> int:
    fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = Cache(str(fixture_path))

    if not os.environ.get("OPENAI_API_KEY") and not fixture.peek({"__probe": 1}) is not None:
        # Best-effort: only warn if there's truly nothing in the fixture and no key.
        # The cached_call below will raise LookupError in OFFLINE mode anyway.
        if os.environ.get("F3DX_BENCH_OFFLINE") != "1":
            print("OPENAI_API_KEY not set; will fail on cache miss", file=sys.stderr)

    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence.",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }

    print("== f3dx.cache real-API bench against gpt-4o-mini ==\n")
    fixture_existed = fixture.peek(request) is not None
    print(f"fixture: {fixture_path}")
    print(f"mode: {'replay' if fixture_existed else 'cold-record'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}"
          f"{' (offline)' if os.environ.get('F3DX_BENCH_OFFLINE') == '1' else ''}\n")

    print("[1/3] Cold call (real API or fixture replay)...")
    payload = cached_call(fixture, request, _fetch_openai, model="gpt-4o-mini")
    cold_ms = payload.get("__bench_cold_ms", 0)
    body = json.dumps(payload).encode()
    reply = payload["choices"][0]["message"]["content"][:60]
    print(f"  cold: {cold_ms} ms (recorded)  ({len(body)} bytes)")
    print(f"  reply preview: {reply!r}")

    # The bench-internal cache for the warm-path measurement is a fresh
    # tempfile so cache.get() / cache.peek() numbers aren't affected by
    # the fixture's file-system cache state. We're measuring the cache
    # implementation, not the fixture file.
    with tempfile.TemporaryDirectory() as tmp:
        warm_cache = Cache(str(Path(tmp) / "warm.redb"))
        warm_cache.put(request, body, model="gpt-4o-mini", response_duration_ms=cold_ms)

        print("\n[2/3] 100 warm cache hits (read with hit-count bump)...")
        warm_get = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            hit = warm_cache.get(request)
            warm_get.append(time.perf_counter_ns() - t0)
            assert hit == body, "warm hit must match cold response"
        warm_get_us = [n / 1000 for n in warm_get]
        print(f"  cache.get() median {statistics.median(warm_get_us):.1f} us  "
              f"p95 {sorted(warm_get_us)[94]:.1f} us  "
              f"min {min(warm_get_us):.1f} us")

        print("\n[3/3] 100 warm cache peeks (read-only, sub-100us target)...")
        warm_peek = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            hit = warm_cache.peek(request)
            warm_peek.append(time.perf_counter_ns() - t0)
            assert hit == body, "warm peek must match cold response"
        warm_peek_us = [n / 1000 for n in warm_peek]
        print(f"  cache.peek() median {statistics.median(warm_peek_us):.1f} us  "
              f"p95 {sorted(warm_peek_us)[94]:.1f} us  "
              f"min {min(warm_peek_us):.1f} us")

        cold_seconds = cold_ms / 1000.0
        warm_get_median_seconds = statistics.median(warm_get_us) / 1_000_000
        warm_peek_median_seconds = statistics.median(warm_peek_us) / 1_000_000
        print("\n== speedup vs recorded cold ==")
        if cold_seconds > 0:
            print(f"  cache.get() vs cold:  {cold_seconds / warm_get_median_seconds:,.0f}x")
            print(f"  cache.peek() vs cold: {cold_seconds / warm_peek_median_seconds:,.0f}x")
        print("\n== Phase B validation: f3dx.cache works against real API, fixture-backed ==")
        return 0


if __name__ == "__main__":
    sys.exit(main())
