"""Day-1 micro-bench: agx Rust path vs pure-Python equivalent.

Measures the two operations the agent loop hits N times per run:
  1. build_next_request — splice tool results into prior history
  2. render_messages — flatten message list to model-input string

If Rust beats Python by ≥5x on either operation, the architecture is
sound and the 8-week build is greenlit. If not, the boundary cost
exceeds the work; pivot the architecture before committing.
"""

from __future__ import annotations

import time
from statistics import median

import f3dx


def py_build_next_request(prior: list[dict], results: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    for msg in prior:
        out.append(msg)
        for tc in msg.get("tool_calls", []) or []:
            if tc["id"] in results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": results[tc["id"]],
                        "tool_calls": [],
                    }
                )
    return out


def py_render_messages(messages: list[dict]) -> str:
    return "".join(f"[{m['role']}] {m['content']}\n" for m in messages)


def make_history(turns: int, tools_per_turn: int) -> tuple[list[dict], dict[str, str]]:
    prior: list[dict] = []
    results: dict[str, str] = {}
    for t in range(turns):
        tcs = [
            {"id": f"call_{t}_{i}", "name": f"tool_{i}", "args_json": '{"q":"x"}'}
            for i in range(tools_per_turn)
        ]
        prior.append(
            {
                "role": "assistant",
                "content": f"turn {t} thinking, calling {tools_per_turn} tools to find evidence",
                "tool_calls": tcs,
            }
        )
        for tc in tcs:
            results[tc["id"]] = (
                f'{{"results": [{{"id": "doc_{t}_{tc["id"]}", "score": 0.87, '
                f'"snippet": "evidence chunk for turn {t}"}}], "total": 1}}'
            )
    return prior, results


def bench(name: str, fn, *args, n_iters: int = 1000, n_runs: int = 5) -> float:
    """Return median ns/op across n_runs each of n_iters calls."""
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            fn(*args)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iters)
    median_ns = median(times)
    print(f"  {name:32s} {median_ns:10.1f} ns/op  ({1e9 / median_ns:>10,.0f} ops/sec)")
    return median_ns


def main() -> None:
    print(f"agx version: {f3dx.__version__}\n")

    for turns, tools in [(5, 1), (10, 3), (20, 5), (50, 5)]:
        prior, results = make_history(turns, tools)
        n_msgs_after = turns + (turns * tools)
        print(f"== {turns} turns x {tools} tools/turn = {n_msgs_after} messages after splice ==")

        py_t = bench("py_build_next_request", py_build_next_request, prior, results)
        rs_t = bench("f3dx.build_next_request", f3dx.build_next_request, prior, results)
        print(f"  -> rust speedup: {py_t / rs_t:.2f}x\n")

        msgs_after = py_build_next_request(prior, results)
        py_t = bench("py_render_messages", py_render_messages, msgs_after)
        rs_t = bench("f3dx.render_messages", f3dx.render_messages, msgs_after)
        print(f"  -> rust speedup: {py_t / rs_t:.2f}x\n")


if __name__ == "__main__":
    main()
