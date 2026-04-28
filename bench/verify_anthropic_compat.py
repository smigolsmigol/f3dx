"""v0.0.6 part 2 verify: f3dx.compat.AsyncAnthropic isinstance + drop-in.
Plus f3dx.pydantic_ai.anthropic_model() builds a working AnthropicModel.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic

from f3dx.compat import AsyncAnthropic
from f3dx.pydantic_ai import anthropic_model

PORT = 8782

NON_STREAM_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4",
    "content": [{"type": "text", "text": "hello from rust"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 7},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if not self.path.endswith("/v1/messages"):
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
        client = AsyncAnthropic(
            api_key="test",
            base_url=f"http://127.0.0.1:{PORT}",
            f3dx_options={"http2": False, "timeout": 30.0},
        )
        assert isinstance(client, anthropic.AsyncAnthropic)
        print("isinstance(AsyncAnthropic, anthropic.AsyncAnthropic):", True)

        msg = await client.messages.create(
            model="claude-haiku-4",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )
        print("type:", type(msg).__module__ + "." + type(msg).__name__)
        print("content[0].text:", msg.content[0].text)
        print("usage:", msg.usage.input_tokens, "in /", msg.usage.output_tokens, "out")

        from pydantic_ai import Agent
        agent = Agent(
            anthropic_model(
                "claude-haiku-4",
                api_key="test",
                base_url=f"http://127.0.0.1:{PORT}",
                f3dx_options={"http2": False, "timeout": 30.0},
            )
        )
        result = await agent.run("hi")
        print("\npydantic_ai agent.run output:", result.output)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    asyncio.run(run())
