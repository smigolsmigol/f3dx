"""Phase G V0 verify: JSONL trace sink writes one row per AgentRuntime.run.

Configures a temp JSONL path, runs the agent N times with varying tool
counts, then reads the file back and asserts row shape + count.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import f3dx


def io_search(args_json: str) -> str:
    args = json.loads(args_json)
    time.sleep(0.005)
    return json.dumps({"results": [], "total": 0, "query": args.get("query", "")})


def make_mock(turns: int, tools_per_turn: int) -> list[str]:
    out = []
    for t in range(turns - 1):
        out.append(json.dumps({
            "content": f"turn {t}",
            "tool_calls": [
                {"id": f"call_{t}_{i}", "name": "io_search",
                 "arguments": json.dumps({"query": f"q{t}{i}"})}
                for i in range(tools_per_turn)
            ],
        }))
    out.append(json.dumps({"content": "final", "tool_calls": []}))
    return out


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="f3dx_trace_")
    os.close(fd)
    os.remove(path)
    print(f"sink path: {path}")

    f3dx._f3dx.configure_traces(path)
    print(f"sink configured: {f3dx._f3dx.trace_sink_path()}\n")

    rt = f3dx.AgentRuntime(
        system_prompt="you are a smoke test",
        max_iterations=10,
        max_tool_calls=20,
        concurrent_tool_dispatch=True,
    )
    tools = {"io_search": io_search}

    cases = [(2, 1), (3, 2), (3, 3), (4, 2), (2, 5)]
    print(f"running {len(cases)} agent calls...")
    for turns, tpt in cases:
        rt.run(f"prompt {turns}x{tpt}", tools, make_mock(turns, tpt))
    print("done.\n")

    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"sink contents: {len(rows)} rows")
    assert len(rows) == len(cases), f"expected {len(cases)} rows, got {len(rows)}"

    for i, row in enumerate(rows):
        print(f"  row {i}: dur={row['duration_ms']:6.1f}ms iters={row['iterations']} "
              f"tool_calls={row['tool_calls_executed']:>2} concurrent={row['concurrent_tool_dispatch']} "
              f"messages={row['messages_count']}")
        # shape assertions
        for key in ["ts", "duration_ms", "iterations", "tool_calls_executed",
                    "concurrent_tool_dispatch", "max_iterations", "max_tool_calls",
                    "system_prompt_chars", "output_chars", "tool_calls",
                    "messages_count"]:
            assert key in row, f"row {i} missing {key}: {row}"
        assert isinstance(row["tool_calls"], list)

    print("\nOK - Phase G V0 verified")
    print("   pl.scan_ndjson(path) or duckdb.read_json(path) ready to use")
    os.remove(path)


if __name__ == "__main__":
    main()
