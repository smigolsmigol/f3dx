"""f3dx-cache: content-addressable LLM response cache + replay.

The Cache class wraps a redb file with three tables (requests, responses,
meta). Identical (model, messages, tools, temp, ...) requests fingerprint
identically via RFC 8785 JCS + BLAKE3, so a cached response returns at
<100 microseconds with no model call.

Built for the test loop. Production paths do not point at this; CI suites,
replay tooling, and notebooks do.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from f3dx_cache._native import Cache as _NativeCache
from f3dx_cache._native import diff, read_jsonl

__all__ = ["Cache", "diff", "read_jsonl"]


class Cache:
    """Open-or-create a cache file at the given path.

    Wraps the native PyO3 class so callers can pass dicts directly without
    pre-serializing.
    """

    def __init__(self, path: str | bytes) -> None:
        self._inner = _NativeCache(str(path))

    def fingerprint(self, request: Mapping[str, Any]) -> str:
        return self._inner.fingerprint(json.dumps(request))

    def put(
        self,
        request: Mapping[str, Any],
        response: bytes | str,
        *,
        model: str | None = None,
        system_fingerprint: str | None = None,
        response_duration_ms: int | None = None,
    ) -> str:
        if isinstance(response, str):
            response = response.encode("utf-8")
        return self._inner.put(
            json.dumps(request),
            response,
            model=model,
            system_fingerprint=system_fingerprint,
            response_duration_ms=response_duration_ms,
        )

    def get(self, request: Mapping[str, Any]) -> bytes | None:
        return self._inner.get(json.dumps(request))

    def stats(self) -> dict[str, int]:
        return dict(self._inner.stats())
