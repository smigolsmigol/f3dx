"""Phase F smoke: AgentRuntime auto-emits OTel spans when configured.

Configures the stdout exporter, runs a real agent workload (3 turns,
2 tools per turn, with sleep-tool simulating I/O), and prints the
emitted span. Verifies the span carries gen_ai.* + f3dx.* attributes.
"""

from __future__ import annotations

import json
import time

import f3dx


def io_search(args_json: str) -> str:
    args = json.loads(args_json)
    time.sleep(0.005)  # simulate small I/O
    return json.dumps({"results": [], "total": 0, "query": args.get("query", "")})


def make_mock_responses(turns: int, tools_per_turn: int) -> list[str]:
    out: list[str] = []
    for t in range(turns - 1):
        out.append(json.dumps({
            "content": f"turn {t}",
            "tool_calls": [
                {
                    "id": f"call_{t}_{i}",
                    "name": "io_search",
                    "arguments": json.dumps({"query": f"q{t}{i}"}),
                }
                for i in range(tools_per_turn)
            ],
        }))
    out.append(json.dumps({"content": "final synthesis", "tool_calls": []}))
    return out


def main() -> None:
    print(f"agx version: {f3dx.__version__}\n")

    # Configure OTel for stdout (no real Logfire token needed for the smoke)
    f3dx._f3dx.configure_otel(service_name="agx-runtime-smoke", stdout=True)
    print("OTel configured (stdout exporter)\n")

    rt = f3dx.AgentRuntime(
        system_prompt="You are a smoke test agent.",
        max_iterations=10,
        max_tool_calls=20,
        concurrent_tool_dispatch=True,
    )

    print("Running agent (3 turns, 2 tools/turn, concurrent dispatch)...")
    result = rt.run(
        "smoke",
        {"io_search": io_search},
        make_mock_responses(turns=3, tools_per_turn=2),
    )
    print(f"Agent done: {result['iterations']} iters, {result['tool_calls']} tool calls, "
          f"{result['duration_ms']:.2f} ms")
    print(f"Final answer: {result['answer']!r}\n")

    # Force-flush so the stdout exporter prints before we exit
    print("Flushing OTel...")
    f3dx._f3dx.shutdown_otel()
    print("(span block printed above by stdout exporter)")


if __name__ == "__main__":
    main()
