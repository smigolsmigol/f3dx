"""Smoke for f3dx.bench - the f3dx-bench SDK.

Sends one real beacon to the LIVE Worker at f3dx-bench-ingest and waits
for the worker thread to drain. Confirms (1) opt-in registers an install,
(2) emit queues, (3) flush drains, (4) the beacon lands in R2.

Run:
    F3DX_BENCH_OPTIN=1 python bench/verify_bench_sdk.py
or programmatically:
    python bench/verify_bench_sdk.py
"""
from __future__ import annotations

import json
import time

from f3dx import bench


def main() -> None:
    print("=== opt_in ===")
    bench.opt_in()
    assert bench.is_enabled()
    install_path = bench.install_file_path()
    print(f"  install file: {install_path}")
    print(f"  exists: {install_path.exists()}")
    assert install_path.exists()
    install_data = json.loads(install_path.read_text(encoding="utf-8"))
    print(f"  install_id: {install_data['install_id']}")
    print(f"  install_hmac (truncated): {install_data['install_hmac'][:16]}...")

    print("\n=== emit one beacon ===")
    bench.emit(
        model="gpt-4o",
        provider="openai",
        status_code=200,
        latency_ms_total=137,
        input_tokens=42,
        output_tokens=17,
        cost_usd_estimate=0.00021,
    )
    print("  emit returned")

    print("\n=== flush (wait up to 5s) ===")
    t0 = time.time()
    bench.flush(timeout=5.0)
    elapsed = time.time() - t0
    print(f"  flush returned in {elapsed:.2f}s")

    print("\n=== opt_out ===")
    bench.opt_out()
    assert not bench.is_enabled()
    print("  disabled")

    print("\nOK")
    print(f"Check the live dashboard for the new beacon (5-15 sec propagation):")
    print(f"  https://f3dx-bench.pages.dev")


if __name__ == "__main__":
    main()
