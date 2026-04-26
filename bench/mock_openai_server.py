"""Tiny stdlib-only OpenAI-compatible mock SSE server for the streaming bench.

Emits N SSE chunks of typical token-streaming shape, then [DONE].
Same response for every request (deterministic), so client-side parsing
is the only variable being measured.

Run alone:
    python mock_openai_server.py 8765 1000
        -> serves N=1000 chunks per request on port 8765
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_CHUNKS = 500


def make_chunk(index: int, total: int, chunk_id: str = "chatcmpl-bench") -> bytes:
    """Build a single OpenAI-shaped streaming chunk."""
    if index == 0:
        delta = {"role": "assistant", "content": ""}
    elif index == total - 1:
        delta = {}
    else:
        delta = {"content": f" tok{index:04d}"}

    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1745700000,
        "model": "agx-bench-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": "stop" if index == total - 1 else None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    n_chunks: int = DEFAULT_CHUNKS

    def log_message(self, format: str, *args: object) -> None:
        # silence default per-request stderr logs (would skew timing prints)
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        # Drain request body so connection can be reused
        n = int(self.headers.get("Content-Length", "0"))
        if n:
            self.rfile.read(n)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for i in range(self.n_chunks):
            self.wfile.write(make_chunk(i, self.n_chunks))
            if i % 100 == 0:
                self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int, n_chunks: int) -> ThreadingHTTPServer:
    Handler.n_chunks = n_chunks
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHUNKS
    server = serve(port, n_chunks)
    print(f"mock OpenAI on http://127.0.0.1:{port} emitting {n_chunks} chunks/request")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
