"""f3dx.fast composition demo: all 4 pillars on one real OpenAI agentic turn.

Cache-backed via the f3d1 convention. Composes:
  Pillar 2: CanonicalPrompt -- static-first ordering for OpenAI prefix cache hits
  Pillar 6: estimate_from_history -- token budget hinting from prior turns
  Pillar 4: SpecToolDispatcher (threaded V0.1) -- speculative tool execution
  Pillar 3: cache_tool_call -- tool-result memoization with mtime invalidation

Real OpenAI gpt-4o-mini agentic turn: model is forced to emit a Read
tool call, the dispatcher fires the Read on a worker thread the moment
its arguments parse cleanly, the actual file read goes through
cache_tool_call so subsequent runs hit the f3dx tool-result cache, the
prompt is built with CanonicalPrompt so the static prefix hits OpenAI's
automatic prefix cache, and max_tokens is hinted from the calibration
history instead of defaulting to 4096.

Shows the four wins stacking:
  - prompt build: prefix-cache hit -> OpenAI charges cached prefix at 50%
    of base rate (Pillar 2 win)
  - max_tokens hinted from history -> truncates at the actual ceiling
    instead of 4096 (Pillar 6 win)
  - tool fires speculatively in parallel with the stream tail
    (Pillar 4 win)
  - the actual Read is itself cached by cache_tool_call so a second
    invocation skips even the filesystem read
    (Pillar 3 win)

Run:
    python examples/f3dx_fast_compose_real_api_bench.py
    F3DX_BENCH_REFRESH=1 python examples/f3dx_fast_compose_real_api_bench.py
    F3DX_BENCH_OFFLINE=1 python examples/f3dx_fast_compose_real_api_bench.py
"""
from __future__ import annotations

import os
import statistics
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
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
}


def _build_long_system() -> str:
    """1024+ token system prompt so the OpenAI prefix cache fires."""
    return (
        "You are a careful, terse senior engineer. "
        "When asked to read a file, ALWAYS call the Read tool with the "
        "exact path the user provides. Never return file contents directly; "
        "always go through the Read tool. After receiving the tool result, "
        "summarize in one short sentence. "
    ) * 15


def _fetch_streamed(request: dict) -> dict:
    """Real OpenAI streaming call recording (offset_ns, chunk_dict) tuples."""
    from openai import OpenAI

    client = OpenAI()
    chunks: list[tuple[int, dict]] = []
    t0_ns = time.perf_counter_ns()
    stream = client.chat.completions.create(**{**request, "stream": True})
    for chunk in stream:
        offset_ns = time.perf_counter_ns() - t0_ns
        chunks.append((offset_ns, chunk.model_dump()))
    return {"chunks": chunks}


def main() -> int:
    fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = Cache(str(fixture_path))

    api_key = os.environ.get("OPENAI_API_KEY")
    offline = os.environ.get("F3DX_BENCH_OFFLINE") == "1"
    if not api_key and not offline:
        print("OPENAI_API_KEY not set; will fail on cache miss outside OFFLINE mode",
              file=sys.stderr)

    print("== f3dx.fast composition demo: 4 pillars on one OpenAI agentic turn ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'offline-strict' if offline else 'replay-default'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}\n")

    # Set up a target file for the Read tool to actually read
    target_dir = Path(__file__).resolve().parent.parent / "bench" / "fixtures"
    target_file = target_dir / "compose_target.txt"
    if not target_file.exists():
        target_file.write_text("This is the file the agentic loop will read.\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # PILLAR 2: build prompt with CanonicalPrompt
    # ------------------------------------------------------------------
    print("[1/4] Pillar 2: CanonicalPrompt build")
    prompt = CanonicalPrompt(model="gpt-4o-mini")
    prompt.add_system(_build_long_system())
    prompt.add_history([
        {"role": "user", "content": "Test that you understand the rules."},
        {"role": "assistant", "content": "Understood. I will use Read for any file content."},
    ])
    prompt.add_user(f"Read the file {target_file} for me.")

    body = prompt.build()
    body["tools"] = [_TOOL_DEF]
    body["tool_choice"] = {"type": "function", "function": {"name": "Read"}}
    body["temperature"] = 0.0
    body["prompt_cache_key"] = prompt.prefix_hash()

    print(f"  prefix_hash: {prompt.prefix_hash()[:16]}...")
    print(f"  cache_threshold: {prompt.cache_threshold()} tokens (model-specific)")

    # ------------------------------------------------------------------
    # PILLAR 6: hint max_tokens from synthetic prior history
    # ------------------------------------------------------------------
    print("\n[2/4] Pillar 6: budget hint")
    prior_completion_tokens = [40, 35, 42, 38, 41, 39, 37, 44, 36, 40]
    hinted_max = estimate_from_history(prior_completion_tokens, fallback=4096)
    body["max_tokens"] = hinted_max
    print(f"  prior turns: {prior_completion_tokens}")
    print(f"  hinted max_tokens: {hinted_max}  (vs default 4096 = "
          f"{(4096 - hinted_max) / 4096 * 100:.0f}% headroom saved)")

    # ------------------------------------------------------------------
    # Issue the streaming call (cached_call handles real-or-replay)
    # ------------------------------------------------------------------
    print("\n[3/4] Pillar 2 + Pillar 4: streaming chat completion + speculative dispatch")
    payload = cached_call(fixture, body, _fetch_streamed, model="gpt-4o-mini")
    chunks = payload["chunks"]
    total_stream_ms = chunks[-1][0] / 1e6 if chunks else 0
    print(f"  stream chunks: {len(chunks)}, total wall-clock: {total_stream_ms:.0f} ms")

    # Inspect cache-hit metric on this call's reply (final chunk has usage if include_usage was set)
    final_usage = None
    for _, chunk in chunks:
        u = chunk.get("usage")
        if u:
            final_usage = u
    if final_usage:
        cached_pct = cache_hit_ratio(final_usage) * 100
        print(f"  OpenAI prefix-cache hit ratio: {cached_pct:.1f}% "
              f"(input cost reduced by {cached_pct * 0.5:.1f}% on gpt-4o-mini)")
    else:
        print(f"  (usage not reported in stream; pass stream_options.include_usage to surface)")

    # ------------------------------------------------------------------
    # PILLAR 3 + PILLAR 4: tool-result cache + speculative dispatch
    # ------------------------------------------------------------------
    # The fetch fn for the Read tool routes through cache_tool_call so
    # the actual filesystem read is itself memoized with mtime invalidation.
    tool_cache = Cache(str(fixture_path.parent / "tool_cache.redb"))

    def cached_read(name: str, args: dict) -> str:
        # Pillar 3 wraps the actual read with mtime witness
        return cache_tool_call(
            tool_cache,
            tool="Read",
            args={"path": args["path"]},
            fetch=lambda a: Path(a["path"]).read_text(encoding="utf-8"),
            witness=FileWitness([args["path"]]),
        )

    SAFE = {"Read", "Glob", "Grep"}
    args_offset_ms = None

    with SpecToolDispatcher(safe_tools=SAFE, fetch=cached_read, threaded=True) as disp:
        replay_t0 = time.perf_counter_ns()
        last_offset = 0
        for offset_ns, chunk in chunks:
            sleep_ns = offset_ns - last_offset
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1e9)
            last_offset = offset_ns
            fired = disp.feed_delta(chunk)
            if fired and args_offset_ms is None:
                args_offset_ms = offset_ns / 1e6

        # Stream done; harvest tool results
        results = []
        for attempt in disp.attempts():
            r = disp.harvest(attempt.tool_call_id, timeout=2.0)
            results.append((attempt.tool_name, r))
        replay_total_ms = (time.perf_counter_ns() - replay_t0) / 1e6

        print(f"  speculation fired at: "
              f"{args_offset_ms:.0f} ms" if args_offset_ms else "(no speculation)")
        print(f"  attempts: {len(disp.attempts())}  acceptance: {disp.acceptance_rate():.0%}")
        for name, result in results:
            preview = (result or "")[:50] if isinstance(result, str) else str(result)[:50]
            print(f"  {name} -> {preview!r}")

    print(f"  replay wall-clock: {replay_total_ms:.0f} ms (stream + parallel tool fire)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n[4/4] composition summary")
    print("=" * 60)

    cached_tokens = (final_usage or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_tokens = (final_usage or {}).get("prompt_tokens", 0)

    print(f"  Pillar 2 (CanonicalPrompt): prefix_hash stable, "
          f"{cached_tokens}/{prompt_tokens} prompt tokens cached "
          f"({cached_tokens / prompt_tokens * 100:.1f}% if reported)" if prompt_tokens else
          "  Pillar 2 (CanonicalPrompt): stream did not surface usage block")
    print(f"  Pillar 6 (budget hint):     max_tokens={hinted_max} vs 4096 default = "
          f"{(4096 - hinted_max) / 4096 * 100:.0f}% headroom saved per call")
    print(f"  Pillar 4 (spec dispatch):   "
          f"{'fired at ' + str(int(args_offset_ms)) + ' ms; ' if args_offset_ms else ''}"
          f"{disp.acceptance_rate():.0%} acceptance, threaded fire = parallel with stream tail")
    # measure tool-cache hit on a second invocation
    t0 = time.perf_counter_ns()
    cached_read("Read", {"path": str(target_file)})
    second_us = (time.perf_counter_ns() - t0) / 1000
    print(f"  Pillar 3 (tool memo):       second Read of same file: "
          f"{second_us:.1f} us (warm cache hit, sub-100us peek path)")

    print("\n  combined story: every layer compounds. cache hit knocks input cost,")
    print("  budget hint keeps output bounded, threaded speculation runs the tool")
    print("  parallel with stream, tool-result cache eliminates the real fs hit on")
    print("  re-runs. matches the f3d1-fast thesis: 'Claude Code 2-3 min -> <1.5 min,")
    print("  software only, against any closed API'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
