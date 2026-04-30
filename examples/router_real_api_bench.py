"""Real-API benchmark for f3dx.router against OpenAI GPT-4o-mini.

Not mocked. Hits the actual OpenAI endpoint via the consolidated
f3dx.router surface, prints real wall-clock latency. Sequential
policy with one provider acts as a passthrough; what we're really
testing is that the consolidated module path (f3dx.router) wires
the Rust core to a real production endpoint without breaking.

Phase B validation (2026-04-30): proves the f3dx-router subtree
merge preserves end-to-end behavior against a real provider.

Run:
    python examples/router_real_api_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

from f3dx.router import Router


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set; cannot run real-API bench", file=sys.stderr)
        return 1

    router = Router(
        providers=[
            {
                "name": "openai-primary",
                "kind": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": api_key,
                "timeout_ms": 30000,
            }
        ],
        policy="sequential",
    )

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Reply in exactly four words."}
        ],
        "temperature": 0.0,
        "max_tokens": 30,
    }

    print("== f3dx.router real-API bench against gpt-4o-mini ==\n")

    # 5 sequential calls
    durations_ms = []
    for i in range(5):
        t0 = time.perf_counter()
        resp = router.chat_completions(body)
        elapsed = (time.perf_counter() - t0) * 1000
        durations_ms.append(elapsed)
        first = resp["choices"][0]["message"]["content"]
        print(f"  call {i+1}: {elapsed:.0f} ms  reply={first!r}")

    print()
    print(f"  median: {statistics.median(durations_ms):.0f} ms")
    print(f"  min:    {min(durations_ms):.0f} ms")
    print(f"  max:    {max(durations_ms):.0f} ms")

    print("\n== Phase B validation: f3dx.router via consolidated workspace works end-to-end against real OpenAI ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
