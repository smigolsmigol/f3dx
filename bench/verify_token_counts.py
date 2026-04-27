"""Phase G V0.0.5 verify: f3dx-rt accumulates per-turn usage and writes
input_tokens + output_tokens to the JSONL row. Tracewright reads them and
runs cost-budget enforcement end-to-end."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import f3dx

# tracewright sibling repo
sys.path.insert(
    0,
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tracewright", "src")),
)


def main() -> None:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)

    f3dx._f3dx.configure_traces(path, True)
    rt = f3dx.AgentRuntime(system_prompt="be terse", concurrent_tool_dispatch=False)

    cases = [
        ("what is 2+2", "4", {"input_tokens": 12, "output_tokens": 1}),
        ("what is 3+3", "6", {"input_tokens": 12, "output_tokens": 1}),
        ("what is the capital of France", "Paris", {"input_tokens": 18, "output_tokens": 2}),
    ]
    for prompt, answer, usage in cases:
        rt.run(
            prompt,
            {},
            [json.dumps({"content": answer, "tool_calls": [], "usage": usage})],
        )

    print(f"f3dx wrote {len(cases)} rows to {path}\n")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for r in rows:
        print(f"  prompt={r['prompt']!r} input_tokens={r['input_tokens']} output_tokens={r['output_tokens']}")

    expected_input = sum(u["input_tokens"] for _, _, u in cases)
    expected_output = sum(u["output_tokens"] for _, _, u in cases)
    actual_input = sum(r["input_tokens"] for r in rows)
    actual_output = sum(r["output_tokens"] for r in rows)
    assert actual_input == expected_input, f"input mismatch: {actual_input} != {expected_input}"
    assert actual_output == expected_output, f"output mismatch: {actual_output} != {expected_output}"
    print(f"\ntotals: {actual_input} in / {actual_output} out")

    try:
        from tracewright import (  # type: ignore[import-not-found]
            ReplayEngine,
            Report,
            enforce_budgets,
            parse_budgets,
            parse_jsonl,
        )
        from tracewright._parse import filter_replayable  # type: ignore[import-not-found]
    except ImportError:
        os.remove(path)
        print("\n(tracewright not installed; cost-rollup half skipped)")
        print("OK — f3dx token-counts in trace row verified")
        return

    replay_rows = list(filter_replayable(parse_jsonl(path)))
    engine = ReplayEngine(candidate_fn=lambda c: c.baseline_output, candidate_model="echo")
    report = Report.from_results(engine.replay_many(replay_rows), candidate_model="echo")

    print("\ntracewright sees:")
    print(f"  baseline_tokens.input = {report.baseline_tokens.input_tokens}")
    print(f"  baseline_tokens.output = {report.baseline_tokens.output_tokens}")
    print(f"  baseline_tokens.total = {report.baseline_tokens.total_tokens}")
    assert report.baseline_tokens.input_tokens == expected_input
    assert report.baseline_tokens.output_tokens == expected_output

    print("\nbudget: tokens_total <= 50 (should PASS, total is 46)")
    failures = enforce_budgets(report, parse_budgets("tokens_total=<=50"))
    print(f"  failures: {len(failures)}")
    assert failures == []

    print("\nbudget: tokens_total <= 30 (should FAIL)")
    failures = enforce_budgets(report, parse_budgets("tokens_total=<=30"))
    print(f"  failures: {len(failures)}")
    assert len(failures) == 1
    print(f"  detail: {failures[0].detail}")

    os.remove(path)
    print("\nOK — f3dx token-counts -> tracewright cost-rollup verified end to end")


if __name__ == "__main__":
    main()
