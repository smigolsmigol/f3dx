"""Phase 6 verify: f3dx.bench.auto_attach() tails JSONL into beacons.

Stubs the worker thread (no real POST) so beacons stay in the queue
where we can inspect them. Writes synthetic trace rows directly to the
JSONL file - the AgentRuntime path is covered by verify_traces.py.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

import f3dx
from f3dx import bench


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}")

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="f3dx_auto_attach_")
    os.close(fd)
    os.remove(path)
    print(f"jsonl path: {path}")

    # Wire f3dx-rt's sink so auto_attach finds an existing path.
    f3dx._f3dx.configure_traces(path, True)

    # Stub: mark enabled WITHOUT starting the worker thread, so queued
    # beacons stick around for the assertion. Set a fake worker_thread
    # so opt_in's "if alive" check short-circuits.
    bench._state["enabled"] = True
    bench._state["install_id"] = "test-install"
    bench._state["install_hmac"] = "test-hmac"
    bench._state["worker_thread"] = threading.Thread(target=lambda: None)
    bench._state["worker_thread"].start()
    bench._state["worker_thread"].join()
    bench._state["queue"].clear()

    tailed = bench.auto_attach(poll_seconds=0.2)
    print(f"tailing: {tailed}")
    assert tailed == path, f"expected tail path {path}, got {tailed}"

    rows = [
        {
            "ts": time.time(),
            "duration_ms": 1234.5,
            "iterations": 2,
            "tool_calls_executed": 1,
            "messages_count": 4,
            "model": "gpt-4o-mini",
            "input_tokens": 120,
            "output_tokens": 45,
        },
        {
            "ts": time.time(),
            "duration_ms": 567.8,
            "iterations": 1,
            "tool_calls_executed": 0,
            "messages_count": 2,
            "model": "claude-3-5-sonnet-latest",
            "input_tokens": 80,
            "output_tokens": 30,
        },
        {
            # row missing 'model' - should be silently skipped by emit_from_trace_row
            "ts": time.time(),
            "duration_ms": 100.0,
            "iterations": 1,
        },
    ]

    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to JSONL")

    # Tailer polls every 0.2s; give it 1s to drain
    time.sleep(1.0)

    queued = list(bench._state["queue"])
    print(f"\nqueue size: {len(queued)}")
    for b in queued:
        print(
            f"  beacon: model={b['model']:<32} provider={b['provider']:<8} "
            f"latency={b['latency_ms_total']}ms in={b['input_tokens']} out={b['output_tokens']}"
        )

    assert len(queued) == 2, f"expected 2 beacons (1 row missing model skipped), got {len(queued)}"
    assert queued[0]["model"] == "gpt-4o-mini"
    assert queued[0]["provider"] == "openai", f"expected openai, got {queued[0]['provider']}"
    assert queued[0]["latency_ms_total"] == 1234
    assert queued[1]["model"] == "claude-3-5-sonnet-latest"
    assert queued[1]["provider"] == "anthropic", f"expected anthropic, got {queued[1]['provider']}"
    assert all(b["status_code"] == 200 for b in queued)
    assert all(b["install_id"] == "test-install" for b in queued)

    # Clean up
    bench._state["tail_stop"].set()
    bench._state["enabled"] = False
    f3dx._f3dx.configure_traces(None, False)
    os.remove(path)

    print("\nOK - Phase 6 auto_attach verified")
    print("   tailer picked up trace rows, mapped to beacons, queued for ingest")


if __name__ == "__main__":
    main()
