"""v0.0.6 part 1 verify: configure_traces(path, capture_messages=True) emits
enriched rows with prompt + system_prompt + output. Default still PII-safe.
"""

from __future__ import annotations

import json
import os
import tempfile

import f3dx


def main() -> None:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)

    # default: capture off
    f3dx._f3dx.configure_traces(path)
    assert f3dx._f3dx.trace_capture_messages() is False
    rt = f3dx.AgentRuntime(system_prompt="be terse", concurrent_tool_dispatch=False)
    rt.run("hi", {}, [json.dumps({"content": "hello", "tool_calls": []})])
    with open(path) as f:
        bare = json.loads(f.readline())
    assert "prompt" not in bare and "output" not in bare
    print("default row (PII-safe):", sorted(bare.keys()))
    os.remove(path)

    # opt-in
    f3dx._f3dx.configure_traces(path, True)
    assert f3dx._f3dx.trace_capture_messages() is True
    rt.run("what is 2+2", {}, [json.dumps({"content": "4", "tool_calls": []})])
    with open(path) as f:
        enriched = json.loads(f.readline())
    print("enriched row keys:", sorted(enriched.keys()))
    assert enriched["prompt"] == "what is 2+2"
    assert enriched["system_prompt"] == "be terse"
    assert enriched["output"] == "4"
    print("prompt:", enriched["prompt"])
    print("system_prompt:", enriched["system_prompt"])
    print("output:", enriched["output"])
    os.remove(path)
    print("\nOK - capture_messages opt-in verified")


if __name__ == "__main__":
    main()
