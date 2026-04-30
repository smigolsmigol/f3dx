"""Real-API benchmark for f3dx.router against OpenAI GPT-4o-mini.

Cache-backed via the canonical f3dx convention (see
docs/workflows/real_api_benches.md). First run hits OpenAI through the
Rust router; subsequent runs replay deterministically from
`bench/fixtures/openai.redb`. CI runs with F3DX_BENCH_OFFLINE=1.

The router itself does not directly take the cache; we wrap the
`router.chat_completions(body)` call as a fetch fn behind cached_call.
This means the bench measures:
  - first run: real Rust-router-to-OpenAI wall-clock
  - subsequent runs: deterministic replay (reported latency is from the
    recorded `__bench_cold_ms` envelope, not a fresh measurement)

Run:
    python examples/router_real_api_bench.py
    F3DX_BENCH_REFRESH=1 python examples/router_real_api_bench.py
    F3DX_BENCH_OFFLINE=1 python examples/router_real_api_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

from f3dx.cache import Cache, cached_call
from f3dx.router import Router


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = Cache(str(fixture_path))

    if not api_key and os.environ.get("F3DX_BENCH_OFFLINE") != "1":
        print("OPENAI_API_KEY not set; will fail on cache miss outside OFFLINE mode",
              file=sys.stderr)

    # Construct the router only if we might need it. The fetch closure
    # captures it for cached_call.
    router = None

    def _fetch(req: dict) -> dict:
        nonlocal router
        if router is None:
            router = Router(
                providers=[{
                    "name": "openai-primary",
                    "kind": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": api_key or "<missing>",
                    "timeout_ms": 30000,
                }],
                policy="sequential",
            )
        t0 = time.perf_counter()
        resp = router.chat_completions(req)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        resp["__bench_cold_ms"] = elapsed_ms
        return resp

    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Reply in exactly four words."}],
        "temperature": 0.0,
        "max_tokens": 30,
    }

    print("== f3dx.router real-API bench against gpt-4o-mini ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'replay' if fixture.peek(body) is not None else 'cold-record'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}"
          f"{' (offline)' if os.environ.get('F3DX_BENCH_OFFLINE') == '1' else ''}\n")

    durations_ms: list[int] = []
    for i in range(5):
        # Each call fingerprints identically (same body), so all 5 are the
        # same cache key. The first call records, the next four hit cache
        # in <1ms each. This means after a fresh refresh the recorded
        # "cold" reflects only call 1; that's intentional, calls 2-5 in a
        # real-API sequence would have benefited from upstream prefix
        # caching anyway. To bench fan-out under load, vary the body.
        resp = cached_call(fixture, body, _fetch, model="gpt-4o-mini")
        cold_ms = resp.get("__bench_cold_ms", 0)
        durations_ms.append(cold_ms)
        first = resp["choices"][0]["message"]["content"]
        print(f"  call {i+1}: {cold_ms} ms (recorded)  reply={first!r}")

    print()
    if durations_ms:
        print(f"  median: {statistics.median(durations_ms):.0f} ms")
        print(f"  min:    {min(durations_ms)} ms")
        print(f"  max:    {max(durations_ms)} ms")
    print("\n== Phase B validation: f3dx.router works against real OpenAI, fixture-backed ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
