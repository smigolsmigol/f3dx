"""Tier 1 verify: f3dx.compat.OpenAI passes isinstance(client, openai.OpenAI),
non-streaming returns a real openai.types.ChatCompletion, streaming yields
ChatCompletionChunk objects. All three checks share an inline mock server."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai

from f3dx.compat import OpenAI

PORT = 8779

NON_STREAM_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello from rust"},
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}


def stream_chunk(content: str | None, *, finish: bool = False) -> bytes:
    payload = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop" if finish else None,
                "delta": {"content": content} if content is not None else {},
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n else b""
        is_stream = b'"stream": true' in body or b'"stream":true' in body

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("hi ", "from ", "rust"):
                self.wfile.write(stream_chunk(piece))
            self.wfile.write(stream_chunk(None, finish=True))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
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
        client = OpenAI(
            api_key="test",
            base_url=f"http://127.0.0.1:{PORT}/v1",
            f3dx_options={"http2": False, "timeout": 30.0},
        )
        assert isinstance(client, openai.OpenAI)
        print("isinstance(client, openai.OpenAI):", True)

        out = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        )
        print("non-stream type:", type(out).__module__ + "." + type(out).__name__)
        print("content:", out.choices[0].message.content)
        print("prompt_tokens:", out.usage.prompt_tokens)

        chunks = list(
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )
        print("stream type:", type(chunks[0]).__module__ + "." + type(chunks[0]).__name__)
        joined = "".join(c.choices[0].delta.content or "" for c in chunks)
        print("stream content joined:", joined)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
