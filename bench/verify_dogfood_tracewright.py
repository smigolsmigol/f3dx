"""End-to-end dogfood: f3dx writes enriched JSONL traces, tracewright
parses + replays them. The actual reason capture_messages exists."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import f3dx

# tracewright is in a sibling repo at local/tracewright; import via relative path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tracewright", "src")))


def main() -> None:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)

    f3dx._f3dx.configure_traces(path, True)
    rt = f3dx.AgentRuntime(system_prompt="answer in one number", concurrent_tool_dispatch=False)
    cases = [
        ("what is 2+2", "4"),
        ("what is 3+3", "6"),
        ("what is 5+5", "10"),
    ]
    for prompt, mock_answer in cases:
        rt.run(prompt, {}, [json.dumps({"content": mock_answer, "tool_calls": []})])

    print(f"f3dx wrote {len(cases)} enriched rows to {path}\n")

    from tracewright import ReplayCase, ReplayEngine, parse_jsonl
    from tracewright._parse import filter_replayable

    rows = list(parse_jsonl(path))
    print(f"tracewright parsed {len(rows)} rows from f3dx output")
    replayable = list(filter_replayable(rows))
    assert len(replayable) == len(cases), f"all rows should replay: {len(replayable)}/{len(cases)}"
    print(f"all {len(replayable)} rows are replayable (have prompt + output)")

    def echo(case: ReplayCase) -> str:
        return case.baseline_output

    def reverser(case: ReplayCase) -> str:
        return case.baseline_output[::-1]

    print("\n-- replay against echo (perfect candidate) --")
    engine = ReplayEngine(candidate_fn=echo, candidate_model="echo")
    results = list(engine.replay_many(replayable))
    assert all(r.all_passed for r in results)
    print(f"  {sum(r.all_passed for r in results)}/{len(results)} passed")

    print("\n-- replay against reverser (deliberate divergence) --")
    engine = ReplayEngine(candidate_fn=reverser, candidate_model="reverser")
    results = list(engine.replay_many(replayable))
    for r in results:
        marker = "PASS" if r.all_passed else "FAIL"
        print(f"  [{marker}] prompt={r.case.prompt!r} baseline={r.case.baseline_output!r} candidate={r.candidate_output!r}")

    os.remove(path)
    print("\nOK - f3dx -> tracewright dogfood loop verified end to end")


if __name__ == "__main__":
    main()
