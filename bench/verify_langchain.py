"""Tier 4 verify: f3dx.langchain.ChatOpenAI invokes against an inline mock,
sync invoke and async ainvoke both route through the f3dx Rust core."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import langchain_openai

from f3dx.langchain import ChatOpenAI

PORT = 8781

NON_STREAM_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi from langchain via f3dx"},
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


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            api_key="test",
            base_url=f"http://127.0.0.1:{PORT}/v1",
            f3dx_options={"http2": False, "timeout": 30.0},
        )
        assert isinstance(llm, langchain_openai.ChatOpenAI)
        print("isinstance(llm, langchain_openai.ChatOpenAI):", True)

        msg = llm.invoke("hi")
        print("sync invoke type:", type(msg).__module__ + "." + type(msg).__name__)
        print("sync content:", msg.content)

        msg2 = asyncio.run(llm.ainvoke("hi"))
        print("async invoke type:", type(msg2).__module__ + "." + type(msg2).__name__)
        print("async content:", msg2.content)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
