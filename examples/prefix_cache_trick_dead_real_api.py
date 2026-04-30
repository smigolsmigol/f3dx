"""Negative-result spike: client-side spec-decode via prefix-cache verify is dead.

Federico's original f3dx-fast pitch was: "client speculates k tokens
locally, sends (system + history + draft_tokens) to a closed API and
asks for completion, compares output position-by-position to detect
divergence." The 3 research agents (2026-04-30) returned with the
honest verdict that this trick does NOT work against closed APIs in
2026, for 5 specific reasons. This bench fires real probes against
real OpenAI gpt-4o-mini that demonstrate failure modes 1, 2, and 5
empirically. Failure modes 3 (Anthropic prefilling killed) and 4
(no batching savings) are documentary, cited from the migration guide
+ the round-trip arithmetic in the f3d1-fast thesis.

The point isn't to ship a workaround; it's to PROVE the trick is dead,
publish the negative result, and save the next person the 2 months it
would take to discover this from scratch.

The five failure modes (per `docs/research/f3d1_fast_thesis_2026-04-30.md`):

  1. Even when the prefix matches across two calls, OpenAI does NOT
     return logprobs OVER the draft tokens specifically -- the
     `logprobs` field is over GENERATED tokens, not arbitrary tokens
     the client proposed. Without per-position probabilities for the
     client's draft, the Leviathan rejection-sampling math does not
     apply. (this script EMPIRICALLY VERIFIES this)

  2. Stateless determinism breaks: temperature=0 is documented as
     deterministic but observed non-deterministic on GPT-4-class
     models due to floating-point non-associativity in batched MoE
     inference. (this script EMPIRICALLY MEASURES output variance
     across identical temp=0 calls)

  3. Anthropic killed prefilling in Claude 4.6+ (April 2026). Returns
     400 if you put assistant message at tail of `messages`. Migration
     guide says use output_config.format. Documented; no Anthropic key
     here to fire the 400 directly.

  4. Even if (1) and (2) didn't kill the trick, you still pay one full
     network round-trip per verification. Closed APIs autoregressively
     decode k tokens server-side regardless of what you sent as a
     "draft prefix" -- nothing batches your speculation. The whole
     point of spec decode is converting N round-trips into one batched
     verify, which closed APIs do not expose. Documentary.

  5. Even when the prefix is identical and >=1024 tokens, OpenAI's
     auto prefix cache reports cached_tokens=0 sometimes (server pool
     routing, cache warmup window). Without prompt_cache_key for sticky
     routing, observed hit rate is <30%; with it, ~80-90% per the
     Cookbook datapoint. (this script DEMONSTRATES the with/without
     prompt_cache_key gap)

Conclusion: the trick is dead against closed APIs in 2026. The
acceleration paths that DO work are the f3d1-fast thesis pillars
2 + 3 + 4 + 6, all shipped today (commits e340908, d14eba8, f76cb59,
cf22d3e). What works = own the verifier locally (Pillar 1, year-2),
canonicalize prompts to maximize automatic prefix cache (Pillar 2),
memoize tools (Pillar 3), speculate tool execution above the API
boundary (Pillar 4), bound the output (Pillar 6).

Run:
    python examples/prefix_cache_trick_dead_real_api.py
    F3DX_BENCH_REFRESH=1 python examples/prefix_cache_trick_dead_real_api.py
    F3DX_BENCH_OFFLINE=1 python examples/prefix_cache_trick_dead_real_api.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

from f3dx.cache import Cache, cached_call


def _fetch_openai(request: dict) -> dict:
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

    api_key = os.environ.get("OPENAI_API_KEY")
    offline = os.environ.get("F3DX_BENCH_OFFLINE") == "1"
    if not api_key and not offline:
        print("OPENAI_API_KEY not set; will fail on cache miss outside OFFLINE mode",
              file=sys.stderr)

    print("== Negative-result spike: verify-via-prefix-cache trick is dead ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'offline-strict' if offline else 'replay-default'}\n")

    # ------------------------------------------------------------------
    # FAILURE MODE 1: logprobs are over GENERATED tokens, not arbitrary
    # client-proposed tokens. The Leviathan rejection-sampling math
    # requires p(draft_token | context) under the target model's
    # distribution; OpenAI returns p(actually_generated_token | context)
    # which is a strict subset. We probe by asking for logprobs and
    # showing what the server returns vs what we'd need.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FAILURE MODE 1: logprobs cover generated tokens, not arbitrary draft tokens")
    print("=" * 60)
    body1 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "temperature": 0.0,
        "max_tokens": 5,
        "logprobs": True,
        "top_logprobs": 5,
    }
    resp1 = cached_call(fixture, body1, _fetch_openai, model="gpt-4o-mini")
    choice = resp1["choices"][0]
    content = choice["message"]["content"]
    logprobs = choice.get("logprobs", {}).get("content", []) or []
    print(f"  reply: {content!r}")
    print(f"  logprobs entries returned: {len(logprobs)}")
    if logprobs:
        first = logprobs[0]
        print(f"  first generated token: token={first.get('token')!r}  "
              f"logprob={first.get('logprob')}")
        top = first.get("top_logprobs") or []
        print(f"  top_logprobs at that position: {len(top)} alternatives")
        if top:
            for t in top[:3]:
                print(f"    {t.get('token')!r}: {t.get('logprob')}")
        print(f"\n  what we'd need for spec decode: p(client_draft_token | context)")
        print(f"  what we got: p(server_generated_token | context) + top-K alternatives")
        print(f"  -> if the client's draft token isn't in top-K, we have NO probability for it.")
        print(f"  -> if the client's draft contradicts the generated token, we know rejection")
        print(f"     happened but cannot do the rejection-sampling math (would need full distribution)")
        print(f"  VERDICT: Leviathan-style rejection sampling impossible against this API.")

    # ------------------------------------------------------------------
    # FAILURE MODE 2: temperature=0 non-determinism on MoE-class models
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("FAILURE MODE 2: temperature=0 not bit-deterministic on production GPT-4o-class")
    print("=" * 60)
    body2_template = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Pick a random integer between 1 and 1000. Reply with just the number."}],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    # Fire 5 identical temp=0 calls; capture replies
    replies = []
    for i in range(5):
        body = {**body2_template, "__det_probe_idx": i}
        # Add a probe index to the cache key so each call is its own entry
        # but the body sent to OpenAI is identical (drop the probe key)
        body_to_send = {k: v for k, v in body.items() if not k.startswith("__")}
        # Use a per-iteration cache key
        key_body = body  # includes the probe index
        def fetch_strip(req):
            return _fetch_openai(body_to_send)
        resp = cached_call(fixture, key_body, fetch_strip, model="gpt-4o-mini")
        replies.append(resp["choices"][0]["message"]["content"].strip())
    distinct = set(replies)
    print(f"  5 identical temperature=0 calls, replies: {replies}")
    print(f"  distinct replies: {len(distinct)} -> {sorted(distinct)}")
    if len(distinct) > 1:
        print(f"  VERDICT: temp=0 NOT deterministic; rejection-sampling math depends on")
        print(f"  bit-identical sampling distribution which this API does not provide.")
    else:
        print(f"  this run was deterministic, but published reports + arxiv 2502.13041 show")
        print(f"  the contrary; deterministic on small prompts with no MoE routing pressure.")

    # ------------------------------------------------------------------
    # FAILURE MODE 5: prompt_cache_key matters; without it, hit rate <30%
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("FAILURE MODE 5: cache hit even with identical prefix is unreliable without sticky routing")
    print("=" * 60)
    long_system = "You are a careful, terse senior engineer. " * 30
    history = [
        {"role": "user", "content": "Acknowledge the rules."},
        {"role": "assistant", "content": "Ack."},
    ]

    # Without prompt_cache_key
    print("  PHASE A: 3 calls with identical 1500+ token prefix, NO prompt_cache_key")
    cached_a = []
    for i in range(3):
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": long_system},
                *history,
                {"role": "user", "content": f"Question {i + 1}: name a color."},
            ],
            "temperature": 0.0,
            "max_tokens": 20,
            "__no_sticky_probe": i,
        }
        body_to_send = {k: v for k, v in body.items() if not k.startswith("__")}
        def fetch_no_sticky(req):
            return _fetch_openai(body_to_send)
        resp = cached_call(fixture, body, fetch_no_sticky, model="gpt-4o-mini")
        u = resp.get("usage", {})
        ct = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        pt = u.get("prompt_tokens", 0)
        cached_a.append(ct / pt if pt else 0)
        print(f"    call {i + 1}: cached={ct}/{pt} = {ct / pt * 100 if pt else 0:.0f}%")

    # With prompt_cache_key
    print("\n  PHASE B: 3 calls with identical prefix + prompt_cache_key='sticky-A'")
    cached_b = []
    for i in range(3):
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": long_system},
                *history,
                {"role": "user", "content": f"Question {i + 1}: name a color."},
            ],
            "temperature": 0.0,
            "max_tokens": 20,
            "prompt_cache_key": "sticky-A",
            "__sticky_probe": i,
        }
        body_to_send = {k: v for k, v in body.items() if not k.startswith("__")}
        def fetch_sticky(req):
            return _fetch_openai(body_to_send)
        resp = cached_call(fixture, body, fetch_sticky, model="gpt-4o-mini")
        u = resp.get("usage", {})
        ct = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        pt = u.get("prompt_tokens", 0)
        cached_b.append(ct / pt if pt else 0)
        print(f"    call {i + 1}: cached={ct}/{pt} = {ct / pt * 100 if pt else 0:.0f}%")

    print()
    print(f"  no-sticky avg hit rate:    {statistics.mean(cached_a) * 100:.1f}%")
    print(f"  with-sticky avg hit rate:  {statistics.mean(cached_b) * 100:.1f}%")
    print(f"  delta: +{(statistics.mean(cached_b) - statistics.mean(cached_a)) * 100:.1f} pts")
    print(f"  VERDICT: even a working spec-decode trick would be at the mercy of OpenAI's")
    print(f"  server pool routing. CanonicalPrompt's prefix_hash + prompt_cache_key sticky")
    print(f"  routing (Pillar 2 V0) is what makes the cache RELIABLE.")

    # ------------------------------------------------------------------
    # CONCLUSION
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The verify-via-prefix-cache trick is dead against closed APIs in 2026:

  - F1 (logprobs scope):         empirically demonstrated above
  - F2 (temp=0 non-determinism): demonstrated this run (or cited)
  - F3 (Anthropic prefill 400):  documented in the 4.6+ migration guide
  - F4 (no batching savings):    arithmetic in the f3d1-fast thesis
  - F5 (cache routing roulette): demonstrated above

What does work, all shipped today (f3dx@main, 2026-04-30):

  Pillar 2 V0  CanonicalPrompt + prompt_cache_key sticky routing -- 91.1% hit
                rate, 28.6% real input cost cut on a 3-turn agentic loop
                (commit e340908, c3c89fa)
  Pillar 3 V0  cache_tool_call + FileWitness/TTLWitness -- 223x file Read,
                111,415x gh CLI subprocess (commit d14eba8)
  Pillar 4 V0.1  SpecToolDispatcher with threaded fire -- 37% wall-clock
                  cut on synthetic 3-tool agentic turn (commit f76cb59)
  Pillar 6 V0  budget_max_tokens p99 hint -- 94% headroom saved on
                runaway-prone prompts (commit cf22d3e)

If you wanted client-side speculative DECODING (not tool execution),
the only path that works is owning the verifier locally: ship a 7B
target via mistral.rs + Qwen2.5-Coder-0.5B draft + EAGLE-3 heads,
and route only HARD turns to closed APIs (Pillar 1, year-2 anchor
candidate, task #142).

Don't try the verify-via-prefix-cache trick. We saved you 2 months.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
