"""f3dx[vigil] verify: bridge f3dx JSONL trace into VIGIL/Robin B's
events.jsonl shape so a Robin B reflection cycle can run over real
f3dx-driven agent runs.

Verifies (1) per-row conversion expands a single f3dx run into one
agent.run event plus per-tool-call events, (2) full-file conversion
preserves the row order and event-count math, (3) the resulting
events.jsonl matches VIGIL's documented shape (ts ISO-8601 with Z,
actor + kind + status + payload + note + source fields).
"""

from __future__ import annotations

import json
import os
import tempfile

import f3dx
from f3dx.vigil import f3dx_jsonl_to_vigil_events, f3dx_row_to_vigil_events


def case_per_row_expansion() -> None:
    print("-- per-row: one f3dx run with two tool calls -> 1 agent.run + 2 tool.call --")
    row = {
        "ts": 1777252734.5,
        "duration_ms": 12.3,
        "iterations": 2,
        "tool_calls_executed": 2,
        "messages_count": 5,
        "input_tokens": 50,
        "output_tokens": 12,
        "prompt": "what is 2+2",
        "system_prompt": "be terse",
        "output": "4",
        "tool_calls": [{"name": "calc", "id": "tc-1"}, {"name": "verify", "id": "tc-2"}],
    }
    events = f3dx_row_to_vigil_events(row, actor="robin_a")
    print(f"  emitted: {len(events)} events")
    for e in events:
        print(f"    kind={e['kind']!r} status={e['status']!r} actor={e['actor']!r}")
    assert len(events) == 3
    assert events[0]["kind"] == "agent.run" and events[0]["status"] == "ok"
    assert events[1]["kind"] == "tool.call" and events[1]["payload"]["name"] == "calc"
    assert events[2]["kind"] == "tool.call" and events[2]["payload"]["name"] == "verify"
    # All events share the same ISO timestamp + Z suffix
    assert events[0]["ts"].endswith("Z") and "T" in events[0]["ts"]


def case_failed_run_status() -> None:
    print("\n-- per-row: run with no output -> agent.run status=fail --")
    row = {
        "ts": 1777252734.0,
        "duration_ms": 1.5,
        "iterations": 1,
        "tool_calls_executed": 0,
        "messages_count": 2,
        "tool_calls": [],
        "output": "",
    }
    events = f3dx_row_to_vigil_events(row, actor="robin_a")
    assert len(events) == 1 and events[0]["status"] == "fail"
    print(f"  status={events[0]['status']!r}")


def case_full_file_conversion() -> None:
    print("\n-- full-file: f3dx writes 4 enriched rows -> Robin-B-shape events.jsonl --")
    fd, f3dx_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(f3dx_path)
    vigil_path = f3dx_path.replace(".jsonl", ".vigil.jsonl")

    f3dx._f3dx.configure_traces(f3dx_path, True)
    rt = f3dx.AgentRuntime(system_prompt="be terse", concurrent_tool_dispatch=False)
    cases = [
        ("what is 2+2", "4", {"input_tokens": 12, "output_tokens": 1}),
        ("what is 3+3", "6", {"input_tokens": 12, "output_tokens": 1}),
        ("capital of France", "Paris", {"input_tokens": 18, "output_tokens": 2}),
        ("capital of Italy", "Rome", {"input_tokens": 18, "output_tokens": 2}),
    ]
    for prompt, answer, usage in cases:
        rt.run(
            prompt,
            {},
            [json.dumps({"content": answer, "tool_calls": [], "usage": usage})],
        )

    meta = f3dx_jsonl_to_vigil_events(f3dx_path, vigil_path, actor="robin_a")
    print(f"  rows_in={meta['rows_in']} events_out={meta['events_out']}")
    # 4 rows, each with zero tool_calls -> 4 agent.run events
    assert meta["rows_in"] == 4
    assert meta["events_out"] == 4

    with open(vigil_path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    print(f"  first event: kind={events[0]['kind']!r} payload.prompt={events[0]['payload'].get('prompt')!r}")
    print(f"               source={events[0]['source']!r}")
    assert all(e["actor"] == "robin_a" for e in events)
    assert all(e["source"] == "f3dx" for e in events)
    assert all(e["kind"] == "agent.run" and e["status"] == "ok" for e in events)
    # Schema match against VIGIL sample (ts/actor/kind/status/payload/note/source)
    required_keys = {"ts", "actor", "kind", "status", "payload", "note", "source"}
    assert all(required_keys.issubset(e.keys()) for e in events)

    os.remove(f3dx_path)
    os.remove(vigil_path)


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    case_per_row_expansion()
    case_failed_run_status()
    case_full_file_conversion()
    print()
    print("OK - f3dx[vigil] bridge verified")
    print("composes: f3dx (transport + JSONL trace) -> VIGIL/Robin B (reflective supervisor)")


if __name__ == "__main__":
    main()
