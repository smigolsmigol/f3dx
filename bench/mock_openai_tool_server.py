"""Mock OpenAI server that streams a fragmented tool_call.

Emits an OpenAI-shape SSE stream where the model is "calling" a tool:
the tool_call's arguments string is fragmented across N chunks, exactly
how the real API streams it. Lets us verify agx Phase D reassembles
correctly.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The full args we want the assembled tool_call to contain.
ARGS_JSON = '{"query":"forensic intel","resident_id":"0000862794","k":5}'


def chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def fragment_args(s: str, n_pieces: int) -> list[str]:
    """Split the args string into n_pieces roughly equal fragments."""
    if n_pieces <= 1:
        return [s]
    step = max(1, len(s) // n_pieces)
    pieces = [s[i : i + step] for i in range(0, len(s), step)]
    return pieces


class Handler(BaseHTTPRequestHandler):
    n_arg_fragments: int = 8

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

        # role + tool_call header (id + name + empty args)
        self.wfile.write(chunk({
            "id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1,
            "model": "tool-bench", "choices": [{
                "index": 0, "finish_reason": None,
                "delta": {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search_evidence", "arguments": ""},
                }]},
            }],
        }))

        # arguments fragmented across N chunks
        for piece in fragment_args(ARGS_JSON, self.n_arg_fragments):
            self.wfile.write(chunk({
                "id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1,
                "model": "tool-bench", "choices": [{
                    "index": 0, "finish_reason": None,
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": piece},
                    }]},
                }],
            }))

        # finish_reason
        self.wfile.write(chunk({
            "id": "chatcmpl-tool", "object": "chat.completion.chunk", "created": 1,
            "model": "tool-bench", "choices": [{
                "index": 0, "finish_reason": "tool_calls",
                "delta": {},
            }],
        }))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int, n_fragments: int) -> ThreadingHTTPServer:
    Handler.n_arg_fragments = n_fragments
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    server = serve(port, n)
    print(f"mock OpenAI tool-call server on http://127.0.0.1:{port}, args fragmented into {n} pieces")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
