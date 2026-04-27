"""Tier 3 verify: f3dx.pydantic_ai integration end-to-end against an inline
mock OpenAI server. Builds an Agent with f3dx-routed model + F3dxCapability,
runs a single turn, asserts the capability counted the request and pydantic-ai
returned a normal AgentRunResult."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic_ai import Agent

from f3dx.pydantic_ai import F3dxCapability, openai_model

PORT = 8780

NON_STREAM_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi from pydantic-ai via f3dx"},
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        if n:
            self.rfile.read(n)
        data = json.dumps(NON_STREAM_BODY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


async def run() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        cap = F3dxCapability()
        agent = Agent(
            openai_model(
                "gpt-4",
                api_key="test",
                base_url=f"http://127.0.0.1:{PORT}/v1",
                f3dx_options={"http2": False, "timeout": 30.0},
            ),
            capabilities=[cap],
        )
        result = await agent.run("hi")
        print("agent output:", result.output)
        print("capability counted model_requests:", cap.model_requests)
        print("capability counted tool_executes:", cap.tool_executes)
        assert cap.model_requests == 1, f"expected 1 model request, got {cap.model_requests}"
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(run())
