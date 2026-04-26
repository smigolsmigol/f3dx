"""Day-2 bench: whole-loop architecture vs pure-Python equivalent.

The day-1 micro-bench proved per-primitive ports lose to PyO3 boundary
cost. This bench measures the corrected architecture: keep state in
Rust for the whole agent run, cross the boundary only for tool calls.

Setup: a fake "search" tool. 5 iterations of a multi-turn agent loop,
each iteration the model "calls" search with different args, finally
the model returns an answer. Mock model responses are pre-baked
(no HTTP), so we measure orchestration overhead only.
"""

from __future__ import annotations

import json
import time
from statistics import median
from typing import Callable

import agx


# ---------- pre-baked mock responses (one per loop iteration) ----------

def make_mock_responses(turns: int, tools_per_turn: int) -> list[str]:
    """Build a list of OpenAI-shaped mock responses. Final one has no tool calls."""
    responses: list[str] = []
    for t in range(turns - 1):
        tool_calls = [
            {
                "id": f"call_{t}_{i}",
                "name": "fake_search",
                "arguments": json.dumps({"query": f"turn {t} hit {i}", "k": 3}),
            }
            for i in range(tools_per_turn)
        ]
        responses.append(
            json.dumps(
                {
                    "content": f"Iteration {t}: I need more evidence, calling search.",
                    "tool_calls": tool_calls,
                }
            )
        )
    # Final iteration: model has enough evidence, returns answer with no tool calls
    responses.append(
        json.dumps(
            {
                "content": "Final synthesis: based on the gathered evidence, the answer is X.",
                "tool_calls": [],
            }
        )
    )
    return responses


# ---------- fake tool ----------

def fake_search(args_json: str) -> str:
    """Stand-in for a real OpenSearch / vector-store search call.
    Trivial work (json parse + dict lookup) so we measure orchestration,
    not tool work.
    """
    args = json.loads(args_json)
    return json.dumps(
        {
            "results": [
                {"id": f"doc_{i}", "snippet": f"hit for {args['query']}", "score": 0.85}
                for i in range(args.get("k", 3))
            ],
            "total": args.get("k", 3),
        }
    )


# ---------- pure-python equivalent loop ----------

def py_run(
    *,
    system_prompt: str,
    prompt: str,
    tools: dict[str, Callable[[str], str]],
    mock_responses: list[str],
    max_iterations: int = 10,
    max_tool_calls: int = 20,
) -> dict:
    """Reference implementation matching agx.AgentRuntime.run semantics."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    tool_calls_executed = 0
    final_answer = ""
    iter_done = 0

    for iter_idx in range(max_iterations):
        iter_done = iter_idx + 1
        if iter_idx >= len(mock_responses):
            raise ValueError(f"mock_responses ran out at iteration {iter_idx}")
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

        for tc in tcs:
            if tool_calls_executed >= max_tool_calls:
                break
            tool_calls_executed += 1
            fn = tools.get(tc["name"])
            if fn is None:
                result_str = json.dumps({"error": f"unknown tool {tc['name']}"})
            else:
                try:
                    result_str = fn(tc["arguments"])
                except Exception as e:
                    result_str = json.dumps({"error": f"tool {tc['name']} raised: {e}"})
            messages.append(
                {
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tc["id"],
                }
            )

    return {
        "answer": final_answer,
        "messages": messages,
        "iterations": iter_done,
        "tool_calls": tool_calls_executed,
    }


# ---------- bench harness ----------

def bench(name: str, fn, n_iters: int = 200, n_runs: int = 5) -> float:
    """Return median ns/op across n_runs each of n_iters calls."""
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iters)
    median_ns = median(times)
    print(
        f"  {name:32s} {median_ns / 1000:>10.1f} us/run  "
        f"({1e9 / median_ns:>10,.0f} runs/sec)"
    )
    return median_ns


def main() -> None:
    print(f"agx version: {agx.__version__}\n")

    system_prompt = "You are a forensic analyst."
    user_prompt = "Investigate the suspect's call patterns."
    tools_py = {"fake_search": fake_search}
    rt = agx.AgentRuntime(system_prompt=system_prompt, max_iterations=20, max_tool_calls=50)

    for turns, tools_per_turn in [(3, 1), (5, 2), (10, 3), (20, 5)]:
        mock = make_mock_responses(turns, tools_per_turn)
        # pre-validate equivalence (sanity)
        py_out = py_run(
            system_prompt=system_prompt,
            prompt=user_prompt,
            tools=tools_py,
            mock_responses=mock,
            max_iterations=20,
            max_tool_calls=50,
        )
        rs_out = rt.run(user_prompt, tools_py, mock)
        assert py_out["answer"] == rs_out["answer"], (
            f"mismatch: py={py_out['answer']!r} rs={rs_out['answer']!r}"
        )
        assert py_out["iterations"] == rs_out["iterations"]
        assert py_out["tool_calls"] == rs_out["tool_calls"]

        n_msgs = len(py_out["messages"])
        print(
            f"== {turns} turns, {tools_per_turn} tools/turn -> "
            f"{rs_out['iterations']} iters, {rs_out['tool_calls']} tool calls, "
            f"{n_msgs} messages =="
        )

        py_t = bench(
            "pure_python_loop",
            lambda: py_run(
                system_prompt=system_prompt,
                prompt=user_prompt,
                tools=tools_py,
                mock_responses=mock,
                max_iterations=20,
                max_tool_calls=50,
            ),
        )
        rs_t = bench(
            "agx.AgentRuntime.run",
            lambda: rt.run(user_prompt, tools_py, mock),
        )
        speedup = py_t / rs_t
        marker = "WIN" if speedup >= 1.5 else ("wash" if speedup >= 0.9 else "LOSS")
        print(f"  -> rust speedup: {speedup:.2f}x  [{marker}]\n")


if __name__ == "__main__":
    main()
