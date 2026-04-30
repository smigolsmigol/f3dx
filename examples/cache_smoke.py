"""Smoke test for the consolidated f3dx.cache module (Phase B).

Mirrors the original f3dx-cache examples/smoke.py, but imports from
the new f3dx.cache namespace instead of the old f3dx_cache top-level package.
Verifies that the subtree-merge + binding refactor preserves the surface.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from f3dx.cache import Cache, diff, read_jsonl


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "smoke.redb"
        cache = Cache(str(cache_path))

        req_a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
        req_b = {"temperature": 0.0, "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}

        fp_a = cache.fingerprint(req_a)
        fp_b = cache.fingerprint(req_b)
        assert fp_a == fp_b, "JCS must canonicalize key order"
        assert len(fp_a) == 64, "BLAKE3 hex must be 64 chars"
        print(f"fingerprint OK ({fp_a[:16]}...)")

        body = b'{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}'

        miss = cache.get(req_a)
        assert miss is None, "first get must miss"
        print("cold-cache miss OK")

        t0 = time.perf_counter_ns()
        cache.put(req_a, body, model="gpt-4o", response_duration_ms=850)
        put_ns = time.perf_counter_ns() - t0

        t0 = time.perf_counter_ns()
        hit = cache.get(req_b)
        get_ns = time.perf_counter_ns() - t0

        assert hit == body, "warm-cache hit must return original bytes"
        print(f"warm-cache hit OK (put {put_ns/1000:.1f}us, get {get_ns/1000:.1f}us)")

        t0 = time.perf_counter_ns()
        peek = cache.peek(req_b)
        peek_ns = time.perf_counter_ns() - t0
        assert peek == body, "peek must return original bytes (read-only)"
        print(f"peek OK ({peek_ns/1000:.1f}us, no hit-count bump)")

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["hits"] >= 1
        print(f"stats OK ({stats})")

        ok, note = diff(
            '{"name":"alice","age":30}',
            '{"age":30,"name":"alice"}',
            mode="structured",
        )
        assert ok, f"structured diff must pass on key reorder: {note}"
        print("diff structured OK")

        ok, note = diff('{"a":1}', '{"a":2}', mode="structured")
        assert not ok, "structured diff must fail on value mismatch"
        print(f"diff structured fail-path OK ({note})")

        # read_jsonl smoke
        jsonl_path = Path(tmp) / "trace.jsonl"
        jsonl_path.write_text('{"a":1}\n{"b":[1,2,3]}\n', encoding="utf-8")
        rows = read_jsonl(str(jsonl_path))
        assert len(rows) == 2
        # read_jsonl is from f3dx-replay; adds nullable trace fields. Just
        # verify the user-supplied keys round-trip.
        assert rows[0].get("a") == 1
        assert rows[1].get("b") == [1, 2, 3]
        print(f"read_jsonl OK ({len(rows)} rows)")

        print("\nALL SMOKE TESTS PASSED -- f3dx.cache consolidated module surface intact")


if __name__ == "__main__":
    main()
