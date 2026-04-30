"""Real-API benchmark for f3dx.fast.budget against OpenAI gpt-4o-mini.

Cache-backed via the canonical f3dx convention (see
docs/workflows/real_api_benches.md). First run hits OpenAI; subsequent
runs replay from `bench/fixtures/openai.redb`. CI runs with
F3DX_BENCH_OFFLINE=1.

Two scenarios:
  1. Tight prompt "what is 7*8" -> 8 tokens consistently. Hint floors at
     50 (safety_factor*p99 < floor). No regression.
  2. Runaway-prone prompt "explain why the sky is blue" -> 207-235 tokens.
     Hint computes max_tokens=282 vs default 4096 = ~93% headroom saved.

Run:
    python examples/budget_real_api_bench.py
    F3DX_BENCH_REFRESH=1 python examples/budget_real_api_bench.py
    F3DX_BENCH_OFFLINE=1 python examples/budget_real_api_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

from f3dx.cache import Cache, cached_call
from f3dx.fast import budget_max_tokens, estimate_from_history


def _fetch_openai(request: dict) -> dict:
    from openai import OpenAI

    client = OpenAI()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**request)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    payload = resp.model_dump()
    payload["__bench_cold_ms"] = elapsed_ms
    return payload


def _call(fixture: Cache, model: str, prompt: str, max_tokens: int) -> tuple[int, float, str]:
    """One real-or-replayed call. Returns (completion_tokens, ms, reply)."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    payload = cached_call(fixture, request, _fetch_openai, model=model)
    completion_tokens = payload.get("usage", {}).get("completion_tokens", 0)
    cold_ms = payload.get("__bench_cold_ms", 0)
    reply = payload["choices"][0]["message"]["content"]
    return completion_tokens, cold_ms, (reply or "").strip()


def main() -> int:
    fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = Cache(str(fixture_path))

    api_key = os.environ.get("OPENAI_API_KEY")
    offline = os.environ.get("F3DX_BENCH_OFFLINE") == "1"
    if not api_key and not offline:
        print("OPENAI_API_KEY not set; will fail on cache miss outside OFFLINE mode",
              file=sys.stderr)

    print("== f3dx.fast.budget real-API bench against gpt-4o-mini ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'offline-strict' if offline else 'replay-default'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}\n")

    # The body's `max_tokens` is part of the cache key, so the calibration
    # phase and the test phase below produce different cached entries even
    # though the user prompt is identical. That's the desired behavior.

    print("[1/3] Calibration: 10 calls with max_tokens=4096 default...")
    calib: list[int] = []
    for i in range(10):
        ct, ms, reply = _call(fixture, "gpt-4o-mini",
                              "Reply in one short sentence. What is 7 times 8?",
                              4096)
        calib.append(ct)
        print(f"  call {i+1}: {ct} completion_tokens, {ms}ms, reply={reply!r}")
    print(f"\n  observed: min={min(calib)}, max={max(calib)}, "
          f"median={statistics.median(calib):.0f}, mean={statistics.mean(calib):.1f}")

    rec = budget_max_tokens(calib)
    print(f"\n[2/3] Budget hint computed:")
    print(f"  p99_observed       = {rec.p99_observed}")
    print(f"  safety_factor      = {rec.safety_factor}")
    print(f"  recommended max_tokens = {rec.max_tokens}")
    print(f"  sample_size        = {rec.sample_size}, confidence = {rec.confidence}")
    print(f"  estimate_from_history() returns: {estimate_from_history(calib)}")

    print(f"\n[3/3] 5 calls @ max_tokens={rec.max_tokens} (hint) vs 5 @ max_tokens=4096:")
    hint_tokens: list[int] = []
    for i in range(5):
        ct, ms, reply = _call(fixture, "gpt-4o-mini",
                              "Reply in one short sentence. What is 7 times 8?",
                              rec.max_tokens)
        hint_tokens.append(ct)
        print(f"  hint   call {i+1}: {ct} tokens, {ms}ms, reply={reply!r}")

    default_tokens: list[int] = []
    for i in range(5):
        ct, ms, reply = _call(fixture, "gpt-4o-mini",
                              "Reply in one short sentence. What is 7 times 8?",
                              4096)
        default_tokens.append(ct)
        print(f"  default call {i+1}: {ct} tokens, {ms}ms, reply={reply!r}")

    print(f"\n  hint    : {sum(hint_tokens)} total tokens / 5 calls "
          f"(mean {statistics.mean(hint_tokens):.1f}/call)")
    print(f"  default : {sum(default_tokens)} total tokens / 5 calls "
          f"(mean {statistics.mean(default_tokens):.1f}/call)")
    truncations_hint = sum(1 for ct in hint_tokens if ct >= rec.max_tokens)
    print(f"  truncated-at-cap: hint={truncations_hint}/5")
    if truncations_hint == 0:
        print("  scenario 1 OK: tight prompt -> hint matches default behavior, no regressions")

    print("\n" + "=" * 60)
    print("[scenario 2] runaway-prone: 'Explain why the sky is blue.'")
    print("=" * 60 + "\n")
    runaway_calib: list[int] = []
    for i in range(10):
        ct, ms, _ = _call(fixture, "gpt-4o-mini", "Explain why the sky is blue.", 4096)
        runaway_calib.append(ct)
        print(f"  call {i+1}: {ct} tokens, {ms}ms")

    runaway_rec = budget_max_tokens(runaway_calib)
    headroom_saved = 4096 - runaway_rec.max_tokens
    pct = headroom_saved / 4096 * 100
    print(f"\nhint computed: max_tokens={runaway_rec.max_tokens} "
          f"(p99={runaway_rec.p99_observed}, conf={runaway_rec.confidence})")
    print(f"  vs default cap 4096 -> headroom saved per call: {headroom_saved} tokens")
    print(f"  worst-case cost reduction: {pct:.0f}%")
    print(f"\n== Phase 6 V0 validation: budget hint shrinks max_tokens by {headroom_saved} "
          f"tokens vs default 4096, fixture-backed ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
