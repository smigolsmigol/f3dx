"""Day-3 bench: concurrent tool dispatch with simulated I/O tools.

Day-2 proved orchestration is ~5% of agent wall-time, not the value prop.
THIS bench measures the real value prop: when a model returns N tool_calls
in one turn, agx fans them out across OS threads while Python tools
release the GIL during I/O. Python pydantic-ai's asyncio loop dispatches
sequentially within a turn (one event loop, one tool at a time unless
explicitly gathered).

Tool: time.sleep(0.020) — simulates a 20ms network round-trip per tool.
Real tools (HTTP via httpx/requests/urllib) release the GIL the same way.

Expected: with 5 tools per turn, sequential = ~100ms/turn; concurrent
= ~20-30ms/turn. Target: ≥3x speedup. If hit, the agx wedge is real.
If not, GIL contention killed it and the architecture has a problem.
"""

from __future__ import annotations

import json
import time
from statistics import median
from typing import Callable

import agx


def make_mock_responses(turns: int, tools_per_turn: int) -> list[str]:
    responses: list[str] = []
    for t in range(turns - 1):
        tool_calls = [
            {
                "id": f"call_{t}_{i}",
                "name": "io_search",
                "arguments": json.dumps({"query": f"turn {t} hit {i}", "k": 3}),
            }
            for i in range(tools_per_turn)
        ]
        responses.append(
            json.dumps(
                {
                    "content": f"Iteration {t}: gathering evidence in parallel.",
                    "tool_calls": tool_calls,
                }
            )
        )
    responses.append(
        json.dumps(
            {
                "content": "Synthesized answer based on parallel evidence.",
                "tool_calls": [],
            }
        )
    )
    return responses


def io_search(args_json: str) -> str:
    """Stand-in for a real HTTP search call. time.sleep releases GIL,
    so multiple threads can actually wait in parallel."""
    args = json.loads(args_json)
    time.sleep(0.020)  # 20ms simulated network RTT
    return json.dumps(
        {
            "results": [
                {"id": f"doc_{i}", "snippet": f"hit for {args['query']}", "score": 0.85}
                for i in range(args.get("k", 3))
            ],
            "total": args.get("k", 3),
        }
    )


def py_run_concurrent(
    *,
    system_prompt: str,
    prompt: str,
    tools: dict[str, Callable[[str], str]],
    mock_responses: list[str],
    max_iterations: int = 20,
    max_tool_calls: int = 50,
) -> dict:
    """Pure-Python equivalent using ThreadPoolExecutor for fair comparison.

    Note: pydantic-ai itself does NOT do this by default. Production
    pydantic-ai dispatches sequentially within a single event loop.
    This is the strongest possible Python competitor — and we're
    still expecting agx to win because it skips the asyncio overhead.
    """
    from concurrent.futures import ThreadPoolExecutor

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    tool_calls_executed = 0
    final_answer = ""
    iter_done = 0

    pool = ThreadPoolExecutor(max_workers=16)
    try:
        for iter_idx in range(max_iterations):
            iter_done = iter_idx + 1
            response = json.loads(mock_responses[iter_idx])
            messages.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": response.get("tool_calls", []),
                }
            )
            tcs = response.get("tool_calls", [])
            if not tcs:
                final_answer = response["content"]
                break

            calls_to_run = tcs[: max(0, max_tool_calls - tool_calls_executed)]
            tool_calls_executed += len(calls_to_run)

            def _exec(tc):
                fn = tools.get(tc["name"])
                if fn is None:
                    return tc["id"], json.dumps({"error": f"unknown tool {tc['name']}"})
                try:
                    return tc["id"], fn(tc["arguments"])
                except Exception as e:
                    return tc["id"], json.dumps({"error": f"tool raised: {e}"})

            for tc_id, result_str in pool.map(_exec, calls_to_run):
                messages.append(
                    {"role": "tool", "content": result_str, "tool_call_id": tc_id}
                )
    finally:
        pool.shutdown(wait=True)

    return {
        "answer": final_answer,
        "messages": messages,
        "iterations": iter_done,
        "tool_calls": tool_calls_executed,
    }


def py_run_sequential(
    *,
    system_prompt: str,
    prompt: str,
    tools: dict[str, Callable[[str], str]],
    mock_responses: list[str],
    max_iterations: int = 20,
    max_tool_calls: int = 50,
) -> dict:
    """Pure-Python sequential — what pydantic-ai effectively does today
    when tool calls are dispatched one at a time within a single asyncio
    event loop iteration."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    tool_calls_executed = 0
    final_answer = ""
    iter_done = 0

    for iter_idx in range(max_iterations):
        iter_done = iter_idx + 1
        response = json.loads(mock_responses[iter_idx])
        messages.append(
            {
                "role": "assistant",
                "content": response["content"],
                "tool_calls": response.get("tool_calls", []),
            }
        )
        tcs = response.get("tool_calls", [])
        if not tcs:
            final_answer = response["content"]
            break

        calls_to_run = tcs[: max(0, max_tool_calls - tool_calls_executed)]
        for tc in calls_to_run:
            tool_calls_executed += 1
            fn = tools.get(tc["name"])
            if fn is None:
                result_str = json.dumps({"error": f"unknown tool {tc['name']}"})
            else:
                try:
                    result_str = fn(tc["arguments"])
                except Exception as e:
                    result_str = json.dumps({"error": f"tool raised: {e}"})
            messages.append(
                {"role": "tool", "content": result_str, "tool_call_id": tc["id"]}
            )

    return {
        "answer": final_answer,
        "messages": messages,
        "iterations": iter_done,
        "tool_calls": tool_calls_executed,
    }


def bench(name: str, fn, n_iters: int = 20, n_runs: int = 3) -> float:
    """Return median ms/op across n_runs each of n_iters calls."""
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iters / 1_000_000)  # ms
    median_ms = median(times)
    print(f"  {name:42s} {median_ms:>8.2f} ms/run  ({1000 / median_ms:>7.1f} runs/sec)")
    return median_ms


def main() -> None:
    print(f"agx version: {agx.__version__}\n")

    system_prompt = "You are a forensic analyst."
    user_prompt = "Investigate the call patterns."
    tools_py = {"io_search": io_search}

    rt_seq = agx.AgentRuntime(
        system_prompt=system_prompt,
        max_iterations=20,
        max_tool_calls=50,
        concurrent_tool_dispatch=False,
    )
    rt_par = agx.AgentRuntime(
        system_prompt=system_prompt,
        max_iterations=20,
        max_tool_calls=50,
        concurrent_tool_dispatch=True,
    )

    for turns, tools_per_turn in [(2, 1), (2, 3), (2, 5), (2, 10), (3, 5)]:
        mock = make_mock_responses(turns, tools_per_turn)
        ideal_sequential_ms = (turns - 1) * tools_per_turn * 20  # 20ms per tool
        ideal_concurrent_ms = (turns - 1) * 20  # all tools in parallel per turn

        print(
            f"== {turns} turns, {tools_per_turn} tools/turn  "
            f"(ideal sequential ~{ideal_sequential_ms}ms, ideal concurrent ~{ideal_concurrent_ms}ms) =="
        )

        py_seq_t = bench(
            "pure_python_sequential",
            lambda: py_run_sequential(
                system_prompt=system_prompt,
                prompt=user_prompt,
                tools=tools_py,
                mock_responses=mock,
            ),
        )
        py_par_t = bench(
            "pure_python_threadpool (best Py case)",
            lambda: py_run_concurrent(
                system_prompt=system_prompt,
                prompt=user_prompt,
                tools=tools_py,
                mock_responses=mock,
            ),
        )
        agx_seq_t = bench(
            "agx (sequential dispatch)",
            lambda: rt_seq.run(user_prompt, tools_py, mock),
        )
        agx_par_t = bench(
            "agx (CONCURRENT dispatch)",
            lambda: rt_par.run(user_prompt, tools_py, mock),
        )

        print(f"  agx-par vs python-seq:        {py_seq_t / agx_par_t:>5.2f}x")
        print(f"  agx-par vs python-threadpool: {py_par_t / agx_par_t:>5.2f}x")
        print(f"  agx-par vs agx-seq:           {agx_seq_t / agx_par_t:>5.2f}x")
        print()


if __name__ == "__main__":
    main()
