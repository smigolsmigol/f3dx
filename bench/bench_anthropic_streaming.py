"""Phase C bench: agx Anthropic streaming vs anthropic Python SDK.

Mock Anthropic server emits N content_block_delta events plus the
required scaffolding (message_start, content_block_start/stop,
message_delta, message_stop). Both clients drain the full stream.
"""

from __future__ import annotations

import time
from statistics import median

from anthropic import Anthropic as AnthropicPy

import f3dx
from mock_anthropic_server import serve

PORT = 8770


def consume_anthropic_py(client: AnthropicPy) -> int:
    stream = client.messages.create(
        model="claude-bench-model",
        max_tokens=1024,
        messages=[{"role": "user", "content": "stream test"}],
        stream=True,
    )
    count = 0
    for _ in stream:
        count += 1
    return count


def consume_agx(client: f3dx.Anthropic) -> int:
    stream = client.messages_create_stream({
        "model": "claude-bench-model",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "stream test"}],
    })
    count = 0
    for _ in stream:
        count += 1
    return count


def bench(name: str, fn, n_iters: int, n_runs: int = 3) -> float:
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iters / 1_000_000)  # ms
    median_ms = median(times)
    print(f"  {name:42s} {median_ms:>9.2f} ms/req  ({1000 / median_ms:>7.1f} req/sec)")
    return median_ms


def main() -> None:
    print(f"agx version: {f3dx.__version__}\n")

    py_client = AnthropicPy(api_key="test", base_url=f"http://127.0.0.1:{PORT}", timeout=30.0)
    rs_client = f3dx.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{PORT}", timeout=30.0, http2=False)

    for n_tokens in [50, 200, 500, 1000]:
        srv = serve(PORT, n_tokens)
        try:
            consume_anthropic_py(py_client)
            consume_agx(rs_client)

            n_iters = max(5, min(30, 2000 // max(n_tokens, 50)))

            print(f"== {n_tokens} delta events/request ({n_iters} iters/run x 3 runs) ==")
            py_t = bench("anthropic SDK (python httpx)", lambda: consume_anthropic_py(py_client), n_iters)
            rs_t = bench("f3dx.Anthropic (rust reqwest+sse)", lambda: consume_agx(rs_client), n_iters)
            speedup = py_t / rs_t
            tag = "WIN" if speedup >= 1.5 else ("wash" if speedup >= 0.9 else "LOSS")
            print(f"  -> agx speedup vs anthropic SDK: {speedup:.2f}x  [{tag}]")
            print()
        finally:
            srv.shutdown()


if __name__ == "__main__":
    main()
