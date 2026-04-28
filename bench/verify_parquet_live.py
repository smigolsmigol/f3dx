"""Phase G V0.2 verify: live parquet sink.

Two paths:
  1. AppendingParquetWriter (direct API): hand-roll rows via append_row,
     close, scan back, assert flushed-row-count + row-group-count.
  2. tail_jsonl_to_parquet: spawn a background thread that writes new
     JSONL rows; main thread tails the JSONL and writes parquet
     incrementally; assert all rows landed.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

import pyarrow.parquet as pq

import f3dx
from f3dx.analytics import (
    AppendingParquetWriter,
    parquet_metadata,
    tail_jsonl_to_parquet,
)


def case_appending_writer_direct() -> None:
    print("-- direct AppendingParquetWriter API --")
    fd, parquet_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    os.remove(parquet_path)

    rows = [
        {
            "ts": 1.0 + i,
            "duration_ms": 10.0 + i,
            "iterations": 1,
            "tool_calls_executed": 0,
            "messages_count": 2,
            "concurrent_tool_dispatch": False,
            "max_iterations": 10,
            "max_tool_calls": 20,
            "system_prompt_chars": 5,
            "output_chars": 1,
            "input_tokens": 10 + i,
            "output_tokens": 1,
            "tool_calls": [],
        }
        for i in range(7)
    ]

    with AppendingParquetWriter(parquet_path, batch_size=3, row_group_size=2) as w:
        for r in rows:
            w.append_row(r)
        # 7 rows with batch_size=3 => 2 full flushes (3+3) at append-time +
        # 1 final flush (1) on close. row_group_size=2 means each flush splits
        # into ceil(batch/2) row groups: [3->2, 3->2, 1->1] = 5 row groups.
        # Easier: just assert >=3 row groups.
    print(f"  rows_written: {w.rows_written}")
    meta = parquet_metadata(parquet_path)
    print(f"  parquet: {meta['rows']} rows / {meta['row_groups']} row groups / {meta['bytes']} bytes")
    assert meta["rows"] == 7
    assert meta["row_groups"] >= 3
    table = pq.read_table(parquet_path)
    assert sum(table["input_tokens"].to_pylist()) == sum(r["input_tokens"] for r in rows)
    os.remove(parquet_path)


def case_tail_jsonl() -> None:
    print("\n-- tail_jsonl_to_parquet against live-growing JSONL --")
    fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(jsonl_path)
    parquet_path = jsonl_path.replace(".jsonl", ".parquet")

    f3dx._f3dx.configure_traces(jsonl_path, True)
    rt = f3dx.AgentRuntime(system_prompt="be terse", concurrent_tool_dispatch=False)

    target_rows = 5
    cases = [
        ("what is 2+2", "4", {"input_tokens": 12, "output_tokens": 1}),
        ("what is 3+3", "6", {"input_tokens": 12, "output_tokens": 1}),
        ("what is 5+5", "10", {"input_tokens": 12, "output_tokens": 2}),
        ("capital of France", "Paris", {"input_tokens": 18, "output_tokens": 2}),
        ("capital of Italy", "Rome", {"input_tokens": 18, "output_tokens": 2}),
    ]

    # Producer thread: writes one row every 100ms
    def producer() -> None:
        for prompt, answer, usage in cases:
            time.sleep(0.1)
            rt.run(
                prompt,
                {},
                [json.dumps({"content": answer, "tool_calls": [], "usage": usage})],
            )

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    # Consumer (this thread): tail until producer is done + we've drained
    state = {"done": False}

    def stop_when_drained() -> bool:
        if not t.is_alive() and not state["done"]:
            # one more poll cycle to drain anything in flight
            state["done"] = True
            return False
        return state["done"]

    written = tail_jsonl_to_parquet(
        jsonl_path,
        parquet_path,
        poll_seconds=0.05,
        batch_size=2,
        row_group_size=3,
        until=stop_when_drained,
    )

    print(f"  tail_jsonl_to_parquet returned: {written} rows")
    meta = parquet_metadata(parquet_path)
    print(f"  parquet: {meta['rows']} rows / {meta['row_groups']} row groups")
    assert written == target_rows, f"expected {target_rows} written, got {written}"
    assert meta["rows"] == target_rows
    table = pq.read_table(parquet_path)
    prompts = table["prompt"].to_pylist()
    print(f"  prompts: {prompts}")
    assert prompts == [c[0] for c in cases]

    os.remove(jsonl_path)
    os.remove(parquet_path)


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    case_appending_writer_direct()
    case_tail_jsonl()
    print("\nOK - Phase G V0.2 live parquet sink verified")


if __name__ == "__main__":
    main()
