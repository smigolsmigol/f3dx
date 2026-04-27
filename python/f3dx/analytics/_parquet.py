"""JSONL trace -> parquet via pyarrow.

We bind the schema to the f3dx-rt JSONL trace shape so the resulting
parquet file is queryable without inferring types row-by-row. Fields
that haven't been written by the running f3dx version (e.g. token
counts on traces older than v0.0.8, prompt/output without
capture_messages=True) become nulls in the column — pyarrow doesn't
mind, polars/duckdb scan them as null.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError(
        "f3dx.analytics requires pyarrow. Install with: pip install f3dx[arrow]"
    ) from e


_CANONICAL_SCHEMA = pa.schema(
    [
        # Always present in every f3dx-rt trace row
        pa.field("ts", pa.float64()),
        pa.field("duration_ms", pa.float64()),
        pa.field("iterations", pa.int64()),
        pa.field("tool_calls_executed", pa.int64()),
        pa.field("messages_count", pa.int64()),
        pa.field("concurrent_tool_dispatch", pa.bool_()),
        pa.field("max_iterations", pa.int64()),
        pa.field("max_tool_calls", pa.int64()),
        pa.field("system_prompt_chars", pa.int64()),
        pa.field("output_chars", pa.int64()),
        # Token counts (v0.0.8+; older traces null)
        pa.field("input_tokens", pa.int64()),
        pa.field("output_tokens", pa.int64()),
        # capture_messages=True (v0.0.6+; null otherwise)
        pa.field("prompt", pa.string()),
        pa.field("system_prompt", pa.string()),
        pa.field("output", pa.string()),
        pa.field("model", pa.string()),
        # tool_calls is a list of {name, id} structs; encoded as a json-string
        # column for parquet portability (most query engines read string cleanly,
        # struct-of-list arrow types vary in cross-engine support)
        pa.field("tool_calls_json", pa.string()),
    ]
)


def jsonl_to_parquet(
    jsonl_path: str | Path,
    parquet_path: str | Path,
    *,
    compression: str = "snappy",
    row_group_size: int = 5000,
) -> dict[str, Any]:
    """Convert a JSONL f3dx trace into a columnar parquet file.

    Args:
        jsonl_path: source JSONL trace (typically the path passed to
            `f3dx.configure_traces`).
        parquet_path: destination parquet file. Created or overwritten.
        compression: parquet codec — snappy (default), zstd, gzip, brotli, lz4, none.
        row_group_size: rows per parquet row group. Smaller = better
            predicate pushdown granularity, larger = better compression.

    Returns:
        dict with row count, column names, file size in bytes, and the
        compression codec used. Useful for assertions in pipelines and
        for log shipping.
    """
    jsonl_path = Path(jsonl_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[Any]] = {f.name: [] for f in _CANONICAL_SCHEMA}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in _CANONICAL_SCHEMA:
                name = field.name
                if name == "tool_calls_json":
                    raw = row.get("tool_calls")
                    columns[name].append(json.dumps(raw) if raw is not None else None)
                else:
                    columns[name].append(row.get(name))

    table = pa.Table.from_pydict(columns, schema=_CANONICAL_SCHEMA)
    pq.write_table(
        table,
        parquet_path,
        compression=compression,
        row_group_size=row_group_size,
    )

    return {
        "rows": table.num_rows,
        "columns": [f.name for f in _CANONICAL_SCHEMA],
        "bytes": parquet_path.stat().st_size,
        "compression": compression,
        "row_group_size": row_group_size,
    }


def parquet_metadata(parquet_path: str | Path) -> dict[str, Any]:
    """Return a small metadata dict for a parquet file: row count, columns,
    row-group count, file size. Lighter than reading the whole file."""
    parquet_path = Path(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    return {
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "columns": pf.schema.names,
        "bytes": parquet_path.stat().st_size,
    }
