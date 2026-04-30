"""Real-API benchmark for f3dx.cache against OpenAI GPT-4o-mini.

Not a mock. Hits the actual API once cold, caches the response,
hits the cache N times warm, prints real latency numbers.

Phase B validation (2026-04-30): proves the consolidated f3dx.cache
surface actually saves time on real LLM round-trips, not just on
synthetic in-memory smoke fixtures.

Run:
    python examples/cache_real_api_bench.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from f3dx.cache import Cache


def cold_openai_call(model: str, messages: list[dict]) -> tuple[bytes, float]:
    """One real API call. Returns (response bytes, wall-clock seconds)."""
    from openai import OpenAI

    client = OpenAI()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=200,
    )
    elapsed = time.perf_counter() - t0
    return resp.model_dump_json().encode(), elapsed


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; cannot run real-API bench", file=sys.stderr)
        return 1

    model = "gpt-4o-mini"
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France? Answer in one sentence.",
        }
    ]
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
    }

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "real_bench.redb"
        cache = Cache(str(cache_path))

        print(f"== f3dx.cache real-API bench against {model} ==\n")

        # Cold: real network call
        print("[1/3] Cold call (real API hit)...")
        body, cold_seconds = cold_openai_call(model, messages)
        cache.put(request_payload, body, model=model, response_duration_ms=int(cold_seconds * 1000))
        print(f"  cold: {cold_seconds*1000:.1f} ms  ({len(body)} bytes)")
        first_words = json.loads(body)["choices"][0]["message"]["content"][:60]
        print(f"  reply preview: {first_words!r}")

        # Warm batch: 100 cache hits
        print("\n[2/3] 100 warm cache hits (read with hit-count bump)...")
        warm_get = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            hit = cache.get(request_payload)
            warm_get.append(time.perf_counter_ns() - t0)
            assert hit == body, "warm hit must match cold response"
        warm_get_us = [n / 1000 for n in warm_get]
        print(f"  cache.get() median {statistics.median(warm_get_us):.1f} us  "
              f"p95 {sorted(warm_get_us)[94]:.1f} us  "
              f"min {min(warm_get_us):.1f} us")

        # Warm peek: 100 cache hits, no hit-count bump (sub-100us target)
        print("\n[3/3] 100 warm cache peeks (read-only, sub-100us target)...")
        warm_peek = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            hit = cache.peek(request_payload)
            warm_peek.append(time.perf_counter_ns() - t0)
            assert hit == body, "warm peek must match cold response"
        warm_peek_us = [n / 1000 for n in warm_peek]
        print(f"  cache.peek() median {statistics.median(warm_peek_us):.1f} us  "
              f"p95 {sorted(warm_peek_us)[94]:.1f} us  "
              f"min {min(warm_peek_us):.1f} us")

        # Speedup
        print("\n== speedup ==")
        warm_get_median_seconds = statistics.median(warm_get_us) / 1_000_000
        warm_peek_median_seconds = statistics.median(warm_peek_us) / 1_000_000
        print(f"  cache.get() vs cold:  {cold_seconds / warm_get_median_seconds:,.0f}x")
        print(f"  cache.peek() vs cold: {cold_seconds / warm_peek_median_seconds:,.0f}x")

        print("\n== Phase B validation: f3dx.cache via consolidated workspace works against real API ==")
        return 0


if __name__ == "__main__":
    sys.exit(main())
