"""Tiny f3dx.MCPServer example used by verify_mcp_server.py.

Run as a child stdio process. Registers two Python callables as MCP
tools and serves them on stdin/stdout."""

from __future__ import annotations

import json

import f3dx


def add_tool_cb(args_json: str) -> str:
    args = json.loads(args_json)
    return str(args["a"] + args["b"])


def echo_tool_cb(args_json: str) -> str:
    args = json.loads(args_json)
    return f"echo: {args.get('message', '')}"


def main() -> None:
    server = f3dx.MCPServer(name="f3dx-test-server", version="0.0.1")
    server.add_tool(
        "add",
        add_tool_cb,
        description="Add two numbers.",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    server.add_tool(
        "echo",
        echo_tool_cb,
        description="Echo back the message field.",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )
    server.serve_stdio()


if __name__ == "__main__":
    main()
