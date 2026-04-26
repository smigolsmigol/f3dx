"""Smoke test: tool that raises a Python exception is captured as a structured
error string, never aborts host CPython. Tier 1 safety fix verification."""

import json

import f3dx


def boom(_arguments: str) -> str:
    raise RuntimeError("synthetic tool failure")


def fine(_arguments: str) -> str:
    return '{"ok": true}'


agent = f3dx.AgentRuntime(system_prompt="test", concurrent_tool_dispatch=True)
mock_responses = [
    json.dumps({
        "content": "fanning out",
        "tool_calls": [
            {"id": "1", "name": "boom", "arguments": "{}"},
            {"id": "2", "name": "fine", "arguments": "{}"},
        ],
    }),
    json.dumps({"content": "done", "tool_calls": []}),
]
result = agent.run("trigger", tools={"boom": boom, "fine": fine}, mock_responses=mock_responses)
print("agent returned without aborting host. result:", repr(result)[:300])
