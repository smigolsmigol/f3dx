"""MCP V0.1 verify: f3dx.MCPClient.streamable_http() connects to a remote
MCP server, lists tools, calls one. Smoke test against
`@modelcontextprotocol/server-everything streamableHttp` running on
localhost:3001 (the package's default port for that mode).

Skipped on CI by default - the streamable-http test path needs a long-lived
HTTP server which the per-job sandbox doesn't suit. Re-enable once we ship
an in-process Rust mock MCP HTTP server."""

from __future__ import annotations

import json
import socket
import sys

import f3dx

URL = "http://127.0.0.1:3001/mcp"


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    print(f"f3dx version: {f3dx.__version__}\n")

    # Pre-flight: only run when the server is actually reachable. TCP-level
    # probe so any HTTP response (including 400 / 404) counts as alive.
    if not _port_open("127.0.0.1", 3001):
        print("no MCP server reachable at 127.0.0.1:3001 - skip", file=sys.stderr)
        print("start with: npx -y @modelcontextprotocol/server-everything streamableHttp",
              file=sys.stderr)
        return 0

    print(f"connecting to {URL}...")
    client = f3dx.MCPClient.streamable_http(URL)
    print("connected.\n")

    tools = client.list_tools()
    print(f"server exposes {len(tools)} tools:")
    for t in tools[:5]:
        print(f"  {t['name']:<30} {(t.get('description') or '').split(chr(10))[0][:60]}")
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more")

    print("\ncalling echo with {'message': 'hi via http'}...")
    out = client.call_tool("echo", json.dumps({"message": "hi via http"}))
    print(f"echo result: {out!r}")
    assert "hi via http" in out

    print("\ncalling get-sum with {'a': 7, 'b': 35}...")
    out = client.call_tool("get-sum", json.dumps({"a": 7, "b": 35}))
    print(f"get-sum result: {out!r}")
    assert "42" in out

    print("\nOK - MCP streamable-http client verified end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
