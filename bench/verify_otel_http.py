"""Phase F V0.1 verify: HTTP-level OTel spans on OpenAI + Anthropic clients.

Configures the stdout exporter, hits both mock servers with streaming
requests, prints the emitted spans. Verifies each span carries the
gen_ai.* + f3dx.* attributes per the OTel GenAI semconv.
"""

from __future__ import annotations

import f3dx
from mock_anthropic_server import serve as serve_anthropic
from mock_openai_server import serve as serve_openai

OPENAI_PORT = 8772
ANTHROPIC_PORT = 8773


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")

    f3dx._f3dx.configure_otel(service_name="f3dx-http-smoke", stdout=True)
    print("OTel configured (stdout exporter)\n")

    # ---- OpenAI streaming ----
    srv_o = serve_openai(OPENAI_PORT, 50)
    try:
        client = f3dx.OpenAI(
            api_key="test",
            base_url=f"http://127.0.0.1:{OPENAI_PORT}/v1",
            timeout=30.0,
            http2=False,
        )
        stream = client.chat_completions_create_stream({
            "model": "openai-bench",
            "messages": [{"role": "user", "content": "stream test"}],
            "temperature": 0.7,
            "max_tokens": 1024,
        })
        n = sum(1 for _ in stream)
        print(f"OpenAI stream: {n} chunks consumed")
    finally:
        srv_o.shutdown()

    # ---- Anthropic streaming ----
    srv_a = serve_anthropic(ANTHROPIC_PORT, 50)
    try:
        client = f3dx.Anthropic(
            api_key="test",
            base_url=f"http://127.0.0.1:{ANTHROPIC_PORT}",
            timeout=30.0,
            http2=False,
        )
        stream = client.messages_create_stream({
            "model": "claude-bench",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "stream test"}],
            "temperature": 0.5,
        })
        n = sum(1 for _ in stream)
        print(f"Anthropic stream: {n} events consumed")
    finally:
        srv_a.shutdown()

    print("\nFlushing OTel...")
    f3dx._f3dx.shutdown_otel()
    print("(span blocks printed above by stdout exporter)")


if __name__ == "__main__":
    main()
