"""f3dx.analytics - convert JSONL trace sinks into Arrow / parquet.

Phase G V0.1: pyarrow-side conversion. The f3dx-rt sink writes JSONL
append-only because rolling-parquet over a long-lived process needs
careful lifecycle handling. Once the trace run is complete, this module
turns the JSONL into a columnar parquet that polars/duckdb scan in
milliseconds with predicate pushdown + column pruning.

    pip install f3dx[arrow]

Typical use:

    f3dx.configure_traces("traces.jsonl", capture_messages=True)
    # ... agent runs ...
    from f3dx.analytics import jsonl_to_parquet
    meta = jsonl_to_parquet("traces.jsonl", "traces.parquet")
    # then: pl.scan_parquet("traces.parquet") or duckdb.read_parquet(...)

V0.2 will add a live parquet sink (rolling files, configurable row-group
size). V0.1 is the converter - covers every f3dx user today since the
JSONL sink is the only one shipping.
"""

from f3dx.analytics._parquet import (
    AppendingParquetWriter,
    jsonl_to_parquet,
    parquet_metadata,
    tail_jsonl_to_parquet,
)

__all__ = [
    "AppendingParquetWriter",
    "jsonl_to_parquet",
    "parquet_metadata",
    "tail_jsonl_to_parquet",
]
