"""f3dx JSONL trace -> VIGIL/Robin B events.jsonl bridge.

VIGIL's event shape is per-action (each tool call, each agent response,
each user feedback is one line). f3dx's trace shape is per-run (one
line per AgentRuntime.run with the full state). Each f3dx row expands
into multiple VIGIL events: one `agent.run` event for the run itself,
plus one `tool.call` event per tool dispatched during the run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts_to_iso(ts: float) -> str:
    """f3dx writes unix-seconds floats; VIGIL expects ISO-8601 with Z suffix."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def f3dx_row_to_vigil_events(
    row: dict[str, Any],
    *,
    actor: str = "agent",
    source: str = "f3dx",
) -> list[dict[str, Any]]:
    """Convert one f3dx JSONL trace row into a list of VIGIL events.

    Yields:
        - one `agent.run` event with status `ok` if the run produced
          output, `fail` otherwise. payload includes duration_ms,
          iterations, token counts, and any prompt/output that
          capture_messages=True wrote.
        - zero or more `tool.call` events, one per tool_calls entry
          on the run, with payload {name, id}.
    """
    ts = row.get("ts", 0.0)
    iso_ts = _ts_to_iso(ts) if isinstance(ts, (int, float)) and ts > 0 else _ts_to_iso(0)

    output = row.get("output")
    has_output = output is not None and output != ""

    run_payload: dict[str, Any] = {
        "duration_ms": row.get("duration_ms"),
        "iterations": row.get("iterations"),
        "tool_calls_executed": row.get("tool_calls_executed"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
    }
    # capture_messages-enriched fields when present
    if "prompt" in row:
        run_payload["prompt"] = row["prompt"]
    if "system_prompt" in row:
        run_payload["system_prompt"] = row["system_prompt"]
    if has_output:
        run_payload["output"] = output

    events: list[dict[str, Any]] = []
    events.append(
        {
            "ts": iso_ts,
            "actor": actor,
            "kind": "agent.run",
            "status": "ok" if has_output else "fail",
            "payload": run_payload,
            "note": f"f3dx run, {row.get('messages_count', 0)} messages",
            "source": source,
        }
    )

    for tc in row.get("tool_calls") or []:
        events.append(
            {
                "ts": iso_ts,
                "actor": actor,
                "kind": "tool.call",
                "status": "ok",
                "payload": {"name": tc.get("name"), "id": tc.get("id")},
                "note": f"dispatched during {row.get('iterations', 1)}-iter run",
                "source": source,
            }
        )

    return events


def f3dx_jsonl_to_vigil_events(
    f3dx_jsonl_path: str | Path,
    vigil_events_path: str | Path,
    *,
    actor: str = "agent",
    source: str = "f3dx",
) -> dict[str, Any]:
    """Convert an entire f3dx JSONL trace into a VIGIL-shape events.jsonl.

    Args:
        f3dx_jsonl_path: source f3dx trace (typical: the path passed to
            `f3dx.configure_traces`).
        vigil_events_path: destination events.jsonl. Created or overwritten.
        actor: name of the supervised agent (defaults to 'agent'; VIGIL
            sample data uses 'robin_a').
        source: VIGIL `source` field; defaults to 'f3dx'.

    Returns:
        dict with input row count, output event count, and the path written.
    """
    f3dx_jsonl_path = Path(f3dx_jsonl_path)
    vigil_events_path = Path(vigil_events_path)
    vigil_events_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in = 0
    events_out = 0
    with (
        open(f3dx_jsonl_path, encoding="utf-8") as inp,
        open(vigil_events_path, "w", encoding="utf-8") as out,
    ):
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_in += 1
            for event in f3dx_row_to_vigil_events(row, actor=actor, source=source):
                out.write(json.dumps(event) + "\n")
                events_out += 1

    return {
        "rows_in": rows_in,
        "events_out": events_out,
        "events_path": str(vigil_events_path),
    }
