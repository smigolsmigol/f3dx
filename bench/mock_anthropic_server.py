"""Tiny stdlib-only Anthropic-compatible mock SSE server for the bench.

Emits N content_block_delta events (the actual tokens) plus the
required Anthropic event scaffolding: message_start, content_block_start,
N x content_block_delta, content_block_stop, message_delta, message_stop.

Total events per request = N + 5.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_TOKENS = 500


def sse_event(event_name: str, data: dict) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    n_tokens: int = DEFAULT_TOKENS

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/messages":
            self.send_error(404)
            return

        n = int(self.headers.get("Content-Length", "0"))
        if n:
            self.rfile.read(n)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        msg_id = "msg_bench"
        model = "claude-bench-model"
        N = self.n_tokens

        # message_start
        self.wfile.write(sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }))
        # content_block_start
        self.wfile.write(sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))
        # content_block_delta x N
        for i in range(N):
            self.wfile.write(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": f" tok{i:04d}"},
            }))
            if i % 100 == 0:
                self.wfile.flush()
        # content_block_stop
        self.wfile.write(sse_event("content_block_stop", {
            "type": "content_block_stop", "index": 0,
        }))
        # message_delta
        self.wfile.write(sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": N},
        }))
        # message_stop
        self.wfile.write(sse_event("message_stop", {"type": "message_stop"}))
        self.wfile.flush()


class _ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True


def serve(port: int, n_tokens: int) -> ThreadingHTTPServer:
    Handler.n_tokens = n_tokens
    server = _ReusableServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8769
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOKENS
    server = serve(port, n_tokens)
    print(f"mock Anthropic on http://127.0.0.1:{port} emitting {n_tokens} delta events/request")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
