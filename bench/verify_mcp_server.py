"""f3dx.MCPServer verify: spawn the test server in a child process,
connect via f3dx.MCPClient.stdio, list tools, call both registered
tools, assert the dispatched Python callbacks fired with the right
arguments and returned the expected text.
"""

from __future__ import annotations

import json
import os
import sys

import f3dx

PYTHON = sys.executable
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "f3dx_test_server.py")


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    print(f"spawning {SERVER_SCRIPT}...")
    client = f3dx.MCPClient.stdio(PYTHON, [SERVER_SCRIPT])
    print("connected.\n")

    tools = client.list_tools()
    print(f"server exposes {len(tools)} tools:")
    for t in tools:
        print(f"  {t['name']:<8} {t.get('description', '')}")
    names = sorted(t["name"] for t in tools)
    assert names == ["add", "echo"], f"unexpected tool list: {names}"

    print("\ncall add(7, 35)...")
    out = client.call_tool("add", json.dumps({"a": 7, "b": 35}))
    print(f"  result: {out!r}")
    assert out == "42"

    print("\ncall echo({'message': 'hello f3dx'})...")
    out = client.call_tool("echo", json.dumps({"message": "hello f3dx"}))
    print(f"  result: {out!r}")
    assert out == "echo: hello f3dx"

    print("\ncall unknown tool 'subtract'...")
    try:
        out = client.call_tool("subtract", json.dumps({"a": 1, "b": 2}))
        print(f"  unexpected success: {out!r}")
        # Some rmcp versions surface the error inside the tool result text
        assert "unknown" in out.lower() or "not found" in out.lower()
    except Exception as e:
        print(f"  raised: {e}")
        assert "unknown" in str(e).lower() or "not found" in str(e).lower()

    print("\nOK — f3dx.MCPServer (full bidirectional MCP surface) verified end to end")


if __name__ == "__main__":
    main()
