"""Mock OpenAI server that streams JSON-shaped content for the
streaming-validation verify (Phase E). The model "speaks" JSON
fragmented across N delta chunks; the final chunk has finish_reason.

Toggle MODE:
  good = emit a complete valid JSON object across the deltas
  bad  = emit a truncated / malformed JSON to exercise validation_error
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD_JSON = '{"intent":"search","query":"forensic intel","filters":{"resident_id":"0000862794","k":5}}'
BAD_JSON = '{"intent":"search","query":"forensic intel",'  # truncated — invalid


def chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def fragment(s: str, n: int) -> list[str]:
    if n <= 1:
        return [s]
    step = max(1, len(s) // n)
    return [s[i : i + step] for i in range(0, len(s), step)]


class Handler(BaseHTTPRequestHandler):
    mode: str = "good"
    n_fragments: int = 8

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
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

        # role header
        self.wfile.write(chunk({
            "id": "chatcmpl-json", "object": "chat.completion.chunk", "created": 1,
            "model": "json-bench", "choices": [{
                "index": 0, "finish_reason": None,
                "delta": {"role": "assistant"},
            }],
        }))

        body = GOOD_JSON if self.mode == "good" else BAD_JSON
        for piece in fragment(body, self.n_fragments):
            self.wfile.write(chunk({
                "id": "chatcmpl-json", "object": "chat.completion.chunk", "created": 1,
                "model": "json-bench", "choices": [{
                    "index": 0, "finish_reason": None,
                    "delta": {"content": piece},
                }],
            }))

        self.wfile.write(chunk({
            "id": "chatcmpl-json", "object": "chat.completion.chunk", "created": 1,
            "model": "json-bench", "choices": [{
                "index": 0, "finish_reason": "stop",
                "delta": {},
            }],
        }))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int, mode: str = "good", n_fragments: int = 8) -> ThreadingHTTPServer:
    Handler.mode = mode
    Handler.n_fragments = n_fragments
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8775
    mode = sys.argv[2] if len(sys.argv) > 2 else "good"
    serve(port, mode)
    print(f"mock OpenAI JSON server on http://127.0.0.1:{port} mode={mode}")
    threading.Event().wait()
