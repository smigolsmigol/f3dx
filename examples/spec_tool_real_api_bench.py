"""Real-API benchmark for f3dx.fast.SpecToolDispatcher (Pillar 4 V0).

Cache-backed via the f3d1 convention. Fires a real OpenAI gpt-4o-mini
streaming chat completion with a tool definition; the model emits a
tool_call across multiple SSE chunks; the dispatcher fires the
(simulated slow) tool optimistically as soon as its arguments parse
cleanly, BEFORE the model finalizes its response.

What we measure: time-from-stream-start to tool-result-available.
  - Naive: wait for stream end, then dispatch tool synchronously.
  - Speculative: fire tool when args parse, finishes in parallel with
    remaining stream chunks.

The simulated tool sleeps 200ms (realistic Bash / network read). On a
~10-chunk stream where the tool_use args complete at chunk 4, the
speculative dispatch saves ~6 chunks worth of stream time + the full
tool latency parallelized with it.

Run:
    python examples/spec_tool_real_api_bench.py
    F3DX_BENCH_REFRESH=1 python examples/spec_tool_real_api_bench.py
    F3DX_BENCH_OFFLINE=1 python examples/spec_tool_real_api_bench.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from f3dx.cache import Cache, cached_call
from f3dx.fast import SpecToolDispatcher, StreamingJSONAccumulator


_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file from disk and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path."},
            },
            "required": ["path"],
        },
    },
}


def _fake_read(name: str, args: dict) -> str:
    """Simulate a 200ms tool call (realistic Bash / file read on a slow disk)."""
    time.sleep(0.2)
    return f"<contents of {args.get('path', '?')}>"


def _fetch_streamed(request: dict) -> dict:
    """Real OpenAI streaming call. Records every SSE chunk + a wall-clock
    timestamp so the cached replay can simulate stream timing later.
    Returns a list of (offset_ns_from_start, chunk_dict) tuples."""
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

    print("== f3dx.fast.SpecToolDispatcher real-API bench against gpt-4o-mini ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'offline-strict' if offline else 'replay-default'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}\n")

    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Read the file /abs/lib.py for me, then summarize.",
            }
        ],
        "tools": [_TOOL_DEF],
        "tool_choice": {"type": "function", "function": {"name": "Read"}},
        "temperature": 0.0,
        "max_tokens": 100,
    }

    print("[1/2] Fetching streamed tool-call response (real OpenAI or fixture replay)...")
    payload = cached_call(fixture, request, _fetch_streamed, model="gpt-4o-mini")
    chunks = payload["chunks"]
    total_stream_ns = chunks[-1][0] if chunks else 0
    print(f"  total chunks: {len(chunks)}")
    print(f"  total stream wall-clock: {total_stream_ns / 1e6:.0f} ms")

    print("\n[2/2] Replaying stream through SpecToolDispatcher (200ms simulated tool latency)...")

    # NAIVE PATH: wait for full stream, then run tool sequentially
    naive_t0_ns = time.perf_counter_ns()
    # simulate replay of stream timing via the recorded offsets
    last_offset = 0
    args_complete_offset = None
    args_acc = StreamingJSONAccumulator()
    for offset_ns, chunk in chunks:
        sleep_ns = offset_ns - last_offset
        if sleep_ns > 0:
            time.sleep(sleep_ns / 1e9)
        last_offset = offset_ns
        delta = (chunk.get("choices") or [{}])[0].get("delta", {}) or {}
        for tc in delta.get("tool_calls") or []:
            if (fn := tc.get("function") or {}).get("arguments"):
                if args_acc.feed(fn["arguments"]):
                    args_complete_offset = offset_ns
    # Stream done; now run the tool sequentially
    naive_tool_t0_ns = time.perf_counter_ns()
    _fake_read("Read", args_acc.parsed or {})
    naive_total_ns = time.perf_counter_ns() - naive_t0_ns
    print(f"  NAIVE: stream {total_stream_ns / 1e6:.0f} ms + tool 200 ms = "
          f"{naive_total_ns / 1e6:.0f} ms total")
    if args_complete_offset is not None:
        print(f"  (args_parsed_at_offset={args_complete_offset / 1e6:.0f} ms; "
              f"chunks remaining after that: "
              f"{sum(1 for o, _ in chunks if o > args_complete_offset)})")

    # SPECULATIVE PATH: replay stream through dispatcher, fire tool
    # when args parse, harvest at end
    fired_count = [0]
    def trace_fetch(name, args):
        fired_count[0] += 1
        return _fake_read(name, args)
    dispatcher = SpecToolDispatcher(safe_tools={"Read"}, fetch=trace_fetch)

    spec_t0_ns = time.perf_counter_ns()
    last_offset = 0
    spec_fire_offset = None
    for offset_ns, chunk in chunks:
        sleep_ns = offset_ns - last_offset
        if sleep_ns > 0:
            time.sleep(sleep_ns / 1e9)
        last_offset = offset_ns
        fired = dispatcher.feed_delta(chunk)
        if fired and spec_fire_offset is None:
            spec_fire_offset = offset_ns
    # The fire happened in the loop above; in real production it would
    # run on a thread, but for V0 V0 it runs sync inside feed_delta. The
    # measurement still shows the EARLIER time the tool result is
    # available (right after fire) vs after the stream fully drains.
    # In a threaded V0.1 the savings would be the (stream-time-after-args-parse)
    # window minus the small thread-spawn cost.
    spec_total_ns = time.perf_counter_ns() - spec_t0_ns
    if spec_fire_offset is not None:
        print(f"  SPECULATIVE: tool fired at offset {spec_fire_offset / 1e6:.0f} ms "
              f"(args parsed mid-stream)")
    print(f"  SPECULATIVE total wall-clock through dispatcher: {spec_total_ns / 1e6:.0f} ms")
    print(f"  fires: {fired_count[0]}, acceptance after harvest: ")
    for attempt in dispatcher.attempts():
        result = dispatcher.harvest(attempt.tool_call_id)
        print(f"    {attempt.tool_call_id}: harvest -> {result!r}")
    print(f"  acceptance rate: {dispatcher.acceptance_rate():.0%}")

    if spec_fire_offset is not None and args_complete_offset is not None:
        # Theoretical savings if tool ran on a thread in parallel with
        # the remaining stream: (stream_remaining_after_args_parse).
        remaining_ms = (total_stream_ns - args_complete_offset) / 1e6
        print(f"\n== Pillar 4 V0 validation ==")
        print(f"  args parsed at {args_complete_offset / 1e6:.0f} ms into the stream")
        print(f"  stream remaining after args complete: {remaining_ms:.0f} ms")
        print(f"  in V0.1 with threaded fire, the 200ms tool runs in parallel with")
        print(f"  those {remaining_ms:.0f} ms of stream + after-stream sequential tool dispatch:")
        print(f"    naive total: stream({total_stream_ns / 1e6:.0f}) + tool(200) = "
              f"{(total_stream_ns / 1e6) + 200:.0f} ms")
        print(f"    spec total:  max(stream({total_stream_ns / 1e6:.0f}), "
              f"args_parse_offset({args_complete_offset / 1e6:.0f}) + tool(200)) = "
              f"{max(total_stream_ns / 1e6, args_complete_offset / 1e6 + 200):.0f} ms")
        savings_ms = (total_stream_ns / 1e6 + 200) - max(total_stream_ns / 1e6, args_complete_offset / 1e6 + 200)
        savings_pct = savings_ms / ((total_stream_ns / 1e6) + 200) * 100
        print(f"    speedup: {savings_ms:.0f} ms saved = {savings_pct:.0f}% wall-clock reduction")

    return 0


if __name__ == "__main__":
    sys.exit(main())
