"""Tier 6 V0 verify: f3dx.MCPClient.stdio() spawns an MCP server, lists
tools, calls one. Uses the official @modelcontextprotocol/server-everything
which exposes echo, add, longRunningOperation, and a few others."""

from __future__ import annotations

import json
import sys

import f3dx


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    print("spawning npx -y @modelcontextprotocol/server-everything...")
    client = f3dx.MCPClient.stdio(
        "npx.cmd" if sys.platform == "win32" else "npx",
        ["-y", "@modelcontextprotocol/server-everything"],
    )
    print("connected.\n")

    tools = client.list_tools()
    print(f"server exposes {len(tools)} tools:")
    for t in tools[:5]:
        desc = (t.get("description") or "").split("\n")[0][:60]
        print(f"  {t['name']:<30} {desc}")
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more")

    print("\ncalling echo with {'message': 'hi from f3dx'}...")
    out = client.call_tool("echo", json.dumps({"message": "hi from f3dx"}))
    print(f"echo result: {out!r}")
    assert "hi from f3dx" in out, f"expected echo to contain our message, got {out!r}"

    print("\ncalling get-sum with {'a': 7, 'b': 35}...")
    out = client.call_tool("get-sum", json.dumps({"a": 7, "b": 35}))
    print(f"get-sum result: {out!r}")
    assert "42" in out, f"expected sum 42, got {out!r}"

    print("\nOK — MCP stdio client verified end to end")


if __name__ == "__main__":
    main()
