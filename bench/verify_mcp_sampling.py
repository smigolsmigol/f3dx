"""MCP V0.2 verify: f3dx.MCPClient.stdio(sampling_callback=fn) bridges
server-initiated sampling/createMessage requests to a Python callback.

Tests against the in-repo bench/mock_mcp_sampling_server.py which uses
the official mcp package to spawn a stdio MCP server with one tool
(`ask_llm`) that issues a createMessage request to the connected client
and returns whatever the client's sampling callback produces.
"""

from __future__ import annotations

import json
import os
import sys

import f3dx

PYTHON = sys.executable
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mock_mcp_sampling_server.py")


def case_no_callback() -> None:
    print("-- no sampling_callback: ask_llm tool gets the standard not-supported error --")
    client = f3dx.MCPClient.stdio(PYTHON, [SERVER_SCRIPT])
    try:
        out = client.call_tool("ask_llm", json.dumps({"prompt": "say hi"}))
        print(f"  result: {out[:200]!r}")
        assert (
            "not" in out.lower()
            or "support" in out.lower()
            or "error" in out.lower()
        ), f"expected error mention, got {out!r}"
    except Exception as e:
        print(f"  raised: {e}")
        assert "method" in str(e).lower() or "not" in str(e).lower()


def case_with_callback() -> None:
    print("\n-- with sampling_callback: callback receives messages, returns assistant text --")

    # Capture invocations so the assertions can check what the server sent us.
    received: list[dict] = []

    def my_sampling(messages_json: str, system_prompt: str) -> str:
        msgs = json.loads(messages_json)
        received.append({"messages": msgs, "system_prompt": system_prompt})
        # Stub LLM: echo the last user content with a prefix, keeping it deterministic
        last_user = next(
            (m["content"] for m in reversed(msgs) if m.get("role") == "user"),
            "<no user message>",
        )
        # `last_user` may be a dict (text content struct) or string depending on rmcp version
        if isinstance(last_user, dict):
            last_user = last_user.get("text") or json.dumps(last_user)
        return f"f3dx-stub-llm: {last_user}"

    client = f3dx.MCPClient.stdio(
        PYTHON,
        [SERVER_SCRIPT],
        sampling_callback=my_sampling,
    )
    out = client.call_tool("ask_llm", json.dumps({"prompt": "what is 2+2?"}))
    print(f"  ask_llm result: {out[:200]!r}")
    print(f"  callback invocations: {len(received)}")
    assert len(received) >= 1, "expected sampling callback to fire at least once"
    if received:
        first = received[0]
        print(f"  first invocation messages_count: {len(first['messages'])}")
        print(f"  first invocation system_prompt[:60]: {first['system_prompt'][:60]!r}")
    # The server returns the assistant text our callback produced
    assert "f3dx-stub-llm" in out, f"expected our stub text in tool result, got {out!r}"


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    case_no_callback()
    case_with_callback()
    print()
    print("OK - MCP V0.2 sampling-callback bridge verified end to end")


if __name__ == "__main__":
    main()
