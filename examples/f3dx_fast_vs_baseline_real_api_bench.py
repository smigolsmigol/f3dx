"""f3dx.fast head-to-head: naive baseline vs stacked pillars, same real workload.

Cache-backed via the f3d1 convention. Runs the same agentic-style loop
twice against real OpenAI gpt-4o-mini:

  Run A (naive baseline):
    - raw openai.OpenAI client, no f3dx surface
    - max_tokens=4096 default
    - no prompt canonicalization (system prompt rebuilt each call)
    - tool dispatch sequential, post-stream
    - tool result NOT cached, every Read hits filesystem

  Run B (f3dx.fast stacked):
    - CanonicalPrompt builds the body with stable static prefix +
      prompt_cache_key for sticky routing (Pillar 2)
    - max_tokens hinted from prior-turn observations (Pillar 6)
    - SpecToolDispatcher threaded fire when tool_call args parse
      cleanly (Pillar 4 V0.1)
    - Read tool wrapped in cache_tool_call with FileWitness mtime
      invalidation (Pillar 3)

Three-turn agentic loop:
  Turn 1: 'Read the file FOO and summarize'
  Turn 2: 'Read the file FOO again and confirm'  -- exercises tool cache
  Turn 3: 'Read FOO once more, identify any markers'  -- exercises both
                                                         caches

Measures total wall-clock + cumulative cached_tokens vs prompt_tokens
across the three turns. Reports the delta.

Run:
    python examples/f3dx_fast_vs_baseline_real_api_bench.py
    F3DX_BENCH_REFRESH=1 python examples/f3dx_fast_vs_baseline_real_api_bench.py
    F3DX_BENCH_OFFLINE=1 python examples/f3dx_fast_vs_baseline_real_api_bench.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from f3dx.cache import (
    Cache,
    FileWitness,
    cache_tool_call,
    cached_call,
)
from f3dx.fast import (
    CanonicalPrompt,
    SpecToolDispatcher,
    cache_hit_ratio,
    estimate_from_history,
)


_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file from disk and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

_LONG_SYSTEM = (
    "You are a careful, terse senior engineer. Always use the Read tool "
    "to fetch file contents; never quote file content from training. "
    "Reply in one short sentence after the tool result. "
) * 30  # ~1500+ prompt tokens so OpenAI's 1024-token prefix-cache threshold trips


def _fetch_streamed(request: dict) -> dict:
    """Real OpenAI streaming call recording (offset_ns, chunk_dict) tuples."""
    from openai import OpenAI

    client = OpenAI()
    chunks: list[tuple[int, dict]] = []
    t0_ns = time.perf_counter_ns()
    stream = client.chat.completions.create(
        **{**request, "stream": True, "stream_options": {"include_usage": True}}
    )
    for chunk in stream:
        offset_ns = time.perf_counter_ns() - t0_ns
        chunks.append((offset_ns, chunk.model_dump()))
    return {"chunks": chunks}


def _replay_chunks_get_usage_and_args(chunks: list[tuple[int, dict]]) -> tuple[dict, dict]:
    """Walk chunks, return (final_usage_block, final_tool_args)."""
    import json
    usage = {}
    args_text = ""
    for _, chunk in chunks:
        if u := chunk.get("usage"):
            usage = u
        for c in chunk.get("choices") or []:
            for tc in (c.get("delta") or {}).get("tool_calls") or []:
                if frag := (tc.get("function") or {}).get("arguments"):
                    args_text += frag
    args = {}
    try:
        args = json.loads(args_text) if args_text else {}
    except Exception:
        pass
    return usage, args


def _run_naive(
    target_file: Path, fixture: Cache, prior_completion_tokens: list[int]
) -> dict:
    """Three sequential turns, baseline: raw OpenAI surface, no f3dx.fast."""
    print("\n--- Run A: naive baseline ---")
    t0 = time.perf_counter_ns()
    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0

    for turn in range(3):
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": _LONG_SYSTEM
                                              + f"\n<turn-{turn}-marker>"},
                {"role": "user", "content": f"Turn {turn+1}: Read the file "
                                            f"{target_file} and confirm."},
            ],
            "tools": [_TOOL_DEF],
            "tool_choice": {"type": "function", "function": {"name": "Read"}},
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        # Hit OpenAI (or fixture replay)
        payload = cached_call(fixture, body, _fetch_streamed, model="gpt-4o-mini")
        chunks = payload["chunks"]
        usage, args = _replay_chunks_get_usage_and_args(chunks)
        # Naive: do the actual read post-stream, no caching
        if args.get("path"):
            _ = Path(args["path"]).read_text(encoding="utf-8") if Path(args["path"]).exists() else ""
        # Replay stream timing (so wall-clock includes the stream)
        last_offset = 0
        for offset_ns, _ in chunks:
            sleep_ns = offset_ns - last_offset
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            last_offset = offset_ns
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        ch = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        total_prompt_tokens += pt
        total_completion_tokens += ct
        total_cached_tokens += ch
        print(f"  turn {turn+1}: prompt={pt}, completion={ct}, cached={ch}")

    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    return {
        "wall_ms": wall_ms,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cached_tokens": total_cached_tokens,
    }


def _run_stacked(
    target_file: Path, fixture: Cache, prior_completion_tokens: list[int]
) -> dict:
    """Three sequential turns with full f3dx.fast pillar stack."""
    print("\n--- Run B: f3dx.fast stacked (all 4 pillars) ---")
    t0 = time.perf_counter_ns()
    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0

    tool_cache = Cache(str(fixture._inner._inner if False else
                            Path(fixture._inner.__class__.__module__ or '.').parent /
                            'tool_cache_stacked.redb')) if False else \
                 Cache(str(target_file.parent / "tool_cache_stacked.redb"))

    def cached_read(name: str, args: dict) -> str:
        return cache_tool_call(
            tool_cache,
            tool="Read",
            args={"path": args["path"]},
            fetch=lambda a: Path(a["path"]).read_text(encoding="utf-8")
                            if Path(a["path"]).exists() else "",
            witness=FileWitness([args["path"]]),
        )

    SAFE = {"Read", "Glob", "Grep"}

    for turn in range(3):
        # Pillar 2: CanonicalPrompt
        prompt = CanonicalPrompt(model="gpt-4o-mini")
        prompt.add_system(_LONG_SYSTEM)
        # dynamic per-turn marker goes in dynamic_suffix so it stays
        # OUT of the cached static prefix
        prompt.add_dynamic_suffix(f"<turn-{turn}-marker>")
        prompt.add_user(f"Turn {turn+1}: Read the file {target_file} and confirm.")

        body = prompt.build()
        body["tools"] = [_TOOL_DEF]
        body["tool_choice"] = {"type": "function", "function": {"name": "Read"}}
        body["temperature"] = 0.0
        body["prompt_cache_key"] = prompt.prefix_hash()
        # Pillar 6: budget hint
        body["max_tokens"] = estimate_from_history(prior_completion_tokens, fallback=4096)

        # Hit OpenAI (or fixture replay)
        payload = cached_call(fixture, body, _fetch_streamed, model="gpt-4o-mini")
        chunks = payload["chunks"]
        usage, _ = _replay_chunks_get_usage_and_args(chunks)

        # Pillar 4: SpecToolDispatcher threaded; Pillar 3: cached_read
        with SpecToolDispatcher(safe_tools=SAFE, fetch=cached_read, threaded=True) as disp:
            last_offset = 0
            for offset_ns, chunk in chunks:
                sleep_ns = offset_ns - last_offset
                if sleep_ns > 0:
                    time.sleep(sleep_ns / 1e9)
                last_offset = offset_ns
                disp.feed_delta(chunk)
            for attempt in disp.attempts():
                disp.harvest(attempt.tool_call_id, timeout=2.0)

        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        ch = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        total_prompt_tokens += pt
        total_completion_tokens += ct
        total_cached_tokens += ch
        print(f"  turn {turn+1}: prompt={pt}, completion={ct}, cached={ch}")

    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    return {
        "wall_ms": wall_ms,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cached_tokens": total_cached_tokens,
    }


def main() -> int:
    fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = Cache(str(fixture_path))

    api_key = os.environ.get("OPENAI_API_KEY")
    offline = os.environ.get("F3DX_BENCH_OFFLINE") == "1"
    if not api_key and not offline:
        print("OPENAI_API_KEY not set; will fail on cache miss outside OFFLINE mode",
              file=sys.stderr)

    print("== f3dx.fast head-to-head: naive baseline vs stacked pillars ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'offline-strict' if offline else 'replay-default'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}")
    print("workload: 3-turn agentic loop, each turn does a tool-call Read")

    target_file = fixture_path.parent / "compose_target.txt"
    if not target_file.exists():
        target_file.write_text("This is the file the agentic loop will read.\n",
                                encoding="utf-8")

    prior = [40, 35, 42, 38, 41, 39, 37, 44, 36, 40]

    naive = _run_naive(target_file, fixture, prior)
    stacked = _run_stacked(target_file, fixture, prior)

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 60)
    print(f"  metric            naive     stacked   delta")
    print(f"  ----------------  --------  --------  --------")
    wall_pct = (naive['wall_ms'] - stacked['wall_ms']) / naive['wall_ms'] * 100 if naive['wall_ms'] else 0
    print(f"  wall_clock_ms     {naive['wall_ms']:>8.0f}  {stacked['wall_ms']:>8.0f}  "
          f"{wall_pct:>+7.1f}%")
    pt_delta_pct = (naive['prompt_tokens'] - stacked['prompt_tokens']) / max(naive['prompt_tokens'], 1) * 100
    print(f"  prompt_tokens     {naive['prompt_tokens']:>8d}  {stacked['prompt_tokens']:>8d}  "
          f"{pt_delta_pct:>+7.1f}%")
    ct_delta_pct = (naive['completion_tokens'] - stacked['completion_tokens']) / max(naive['completion_tokens'], 1) * 100
    print(f"  completion_tokens {naive['completion_tokens']:>8d}  {stacked['completion_tokens']:>8d}  "
          f"{ct_delta_pct:>+7.1f}%")
    print(f"  cached_tokens     {naive['cached_tokens']:>8d}  {stacked['cached_tokens']:>8d}")

    naive_input_cost_units = naive["prompt_tokens"] - naive["cached_tokens"] + 0.5 * naive["cached_tokens"]
    stacked_input_cost_units = stacked["prompt_tokens"] - stacked["cached_tokens"] + 0.5 * stacked["cached_tokens"]
    cost_delta_pct = (naive_input_cost_units - stacked_input_cost_units) / max(naive_input_cost_units, 1) * 100
    print(f"  input_cost_units  {naive_input_cost_units:>8.0f}  {stacked_input_cost_units:>8.0f}  "
          f"{cost_delta_pct:>+7.1f}%   (cached @ 50% on gpt-4o-mini)")

    print(f"\n  bottom line: stacked pillars {wall_pct:+.1f}% wall-clock, "
          f"{cost_delta_pct:+.1f}% input cost")
    print(f"  note: replay timings reproduce the recorded cold latencies from")
    print(f"        the fixture, so this is a lower bound on the difference")
    print(f"        cold OpenAI call timings show on REFRESH=1 runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
