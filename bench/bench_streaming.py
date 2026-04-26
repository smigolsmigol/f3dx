"""Phase B bench: agx streaming vs openai SDK streaming.

Mock OpenAI-compatible server emits N SSE chunks per request.
Both clients hit the same server. We measure full wall-clock per
streamed completion (drain all chunks). Single-thread, repeat to
get a stable median.

Expected: agx faster — Rust SSE parser + reqwest connection pool
vs Python SSE parsing + httpx pool. Magnitude TBD by chunk count
(per-chunk parse cost dominates as N grows).
"""

from __future__ import annotations

import time
from statistics import median

from openai import OpenAI as OpenAIPy

import agx
from mock_openai_server import serve

PORT = 8767


def consume_openai_py(client: OpenAIPy, n_chunks: int) -> int:
    stream = client.chat.completions.create(
        model="agx-bench-model",
        messages=[{"role": "user", "content": "stream test"}],
        stream=True,
    )
    count = 0
    for _ in stream:
        count += 1
    return count


def consume_agx(client: agx.OpenAI, n_chunks: int) -> int:
    stream = client.chat_completions_create_stream(
        {
            "model": "agx-bench-model",
            "messages": [{"role": "user", "content": "stream test"}],
        }
    )
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
    print(f"agx version: {agx.__version__}\n")

    py_client = OpenAIPy(api_key="test", base_url=f"http://127.0.0.1:{PORT}/v1", timeout=30.0)
    rs_client = agx.OpenAI(api_key="test", base_url=f"http://127.0.0.1:{PORT}/v1", timeout=30.0, http2=False)

    for n_chunks in [50, 200, 500, 1000]:
        # Restart server so it serves the new chunk count
        srv = serve(PORT, n_chunks)
        try:
            # warm both clients (one request each, not measured)
            consume_openai_py(py_client, n_chunks)
            consume_agx(rs_client, n_chunks)

            # decide iteration count: more iters for small N, fewer for large
            n_iters = max(5, min(30, 2000 // max(n_chunks, 50)))

            print(f"== {n_chunks} chunks/request ({n_iters} iters/run x 3 runs) ==")
            py_t = bench("openai SDK (python httpx)", lambda: consume_openai_py(py_client, n_chunks), n_iters)
            rs_t = bench("agx.OpenAI (rust reqwest+sse)", lambda: consume_agx(rs_client, n_chunks), n_iters)
            speedup = py_t / rs_t
            tag = "WIN" if speedup >= 1.5 else ("wash" if speedup >= 0.9 else "LOSS")
            print(f"  -> agx speedup vs openai SDK: {speedup:.2f}x  [{tag}]")
            print()
        finally:
            srv.shutdown()


if __name__ == "__main__":
    main()
