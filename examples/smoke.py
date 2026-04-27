"""End-to-end smoke test for f3dx-cache.

Exercises the Python surface: open, fingerprint, put, get, stats, diff,
read_jsonl. Run with `python examples/smoke.py` after `maturin develop`.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from f3dx_cache import Cache, diff, read_jsonl


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
        assert not ok, "structured diff must fail on value change"
        print(f"diff structured fail-path OK ({note})")

        jsonl_path = Path(tmp) / "trace.jsonl"
        rows = [
            {"trace_id": "t1", "model": "gpt-4o", "prompt": "hi", "output": '{"r":1}'},
            {"trace_id": "t2", "model": "gpt-4o", "prompt": "hello", "output": '{"r":2}'},
        ]
        jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        loaded = read_jsonl(str(jsonl_path))
        assert len(loaded) == 2
        assert loaded[0]["trace_id"] == "t1"
        print(f"read_jsonl OK ({len(loaded)} rows)")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
