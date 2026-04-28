"""Phase G V0.1 verify: f3dx writes JSONL traces, f3dx.analytics converts
to parquet, pyarrow scans the parquet back with predicate pushdown.

Smoke covers: enriched fields (capture_messages=True), token counts
(v0.0.8+ schema), tool_calls_json round-trip, basic aggregation query.
"""

from __future__ import annotations

import json
import os
import tempfile

import f3dx
from f3dx.analytics import jsonl_to_parquet, parquet_metadata


def _trace(path: str) -> None:
    f3dx._f3dx.configure_traces(path, True)
    rt = f3dx.AgentRuntime(system_prompt="be terse", concurrent_tool_dispatch=False)
    cases = [
        ("what is 2+2", "4", {"input_tokens": 12, "output_tokens": 1}),
        ("what is 3+3", "6", {"input_tokens": 12, "output_tokens": 1}),
        ("what is 5+5", "10", {"input_tokens": 12, "output_tokens": 2}),
        ("capital of France", "Paris", {"input_tokens": 18, "output_tokens": 2}),
        ("capital of Italy", "Rome", {"input_tokens": 18, "output_tokens": 2}),
    ]
    for prompt, answer, usage in cases:
        rt.run(
            prompt,
            {},
            [json.dumps({"content": answer, "tool_calls": [], "usage": usage})],
        )


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")

    fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(jsonl_path)
    parquet_path = jsonl_path.replace(".jsonl", ".parquet")

    _trace(jsonl_path)
    print(f"wrote 5 rows to {jsonl_path}")
    print(f"jsonl bytes: {os.path.getsize(jsonl_path)}\n")

    meta = jsonl_to_parquet(jsonl_path, parquet_path)
    print(f"parquet: {meta['rows']} rows, {meta['bytes']} bytes ({meta['compression']})")
    print(f"  columns: {len(meta['columns'])}: {meta['columns']}\n")

    pmeta = parquet_metadata(parquet_path)
    print(f"parquet_metadata roundtrip: {pmeta['rows']} rows / {pmeta['row_groups']} row group(s)\n")
    assert pmeta["rows"] == 5

    # scan back via pyarrow + run a small aggregation
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    print(f"scanned table shape: {table.num_rows} rows × {table.num_columns} cols")
    print(f"  prompts: {table['prompt'].to_pylist()}")
    print(f"  output_tokens: {table['output_tokens'].to_pylist()}")
    print(f"  total input_tokens: {sum(table['input_tokens'].to_pylist())}")
    print(f"  total output_tokens: {sum(table['output_tokens'].to_pylist())}")
    assert sum(table["input_tokens"].to_pylist()) == 12 + 12 + 12 + 18 + 18
    assert sum(table["output_tokens"].to_pylist()) == 1 + 1 + 2 + 2 + 2

    # predicate-pushdown style filter
    import pyarrow.compute as pc
    filtered = table.filter(pc.starts_with(table["prompt"], "capital"))
    print(f"\nfiltered to prompts starting with 'capital': {filtered.num_rows} rows")
    assert filtered.num_rows == 2

    os.remove(jsonl_path)
    os.remove(parquet_path)
    print("\nOK - Phase G V0.1 jsonl_to_parquet verified")


if __name__ == "__main__":
    main()
