"""f3dx.cache: content-addressable LLM response cache + replay.

The Cache class wraps a redb file with three tables (requests, responses,
meta). Identical (model, messages, tools, temp, ...) requests fingerprint
identically via RFC 8785 JCS + BLAKE3, so a cached response returns at
<100 microseconds with no model call.

Built for the test loop. Production paths do not point at this; CI suites,
replay tooling, notebooks, and bench fixtures do. The `cached_call` helper
is the canonical wrapper for any closed-API call across the f3d1 ecosystem;
see `docs/workflows/real_api_benches.md` for the full convention.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from f3dx._f3dx.cache import Cache as _NativeCache  # type: ignore[attr-defined]
from f3dx._f3dx.cache import diff, read_jsonl  # type: ignore[attr-defined]

__all__ = [
    "Cache",
    "cached_call",
    "diff",
    "read_jsonl",
    # Pillar 3: tool-result memoization (re-exported from .tools submodule)
    "FileWitness",
    "TTLWitness",
    "EnvWitness",
    "CompositeWitness",
    "cache_tool_call",
    "fingerprint_args",
]


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

    def peek(self, request: Mapping[str, Any]) -> bytes | None:
        """Read-only lookup: skips the hit-count bump for sub-100us warm hits."""
        return self._inner.peek(json.dumps(request))

    def stats(self) -> dict[str, int]:
        return dict(self._inner.stats())


def cached_call(
    cache: Cache,
    request: Mapping[str, Any],
    fetch: Callable[[Mapping[str, Any]], Any],
    *,
    model: str | None = None,
    encoder: Callable[[Any], bytes] = lambda obj: json.dumps(obj).encode(),
    decoder: Callable[[bytes], Any] = lambda b: json.loads(b.decode()),
) -> Any:
    """Cache-backed wrapper for any closed-API call.

    The canonical pattern for real-API benches, integration tests, and demos
    across the f3d1 ecosystem. Records the first call to `fetch(request)`
    in `cache`, replays from cache on subsequent identical requests.

    Two env-var modes:

      F3DX_BENCH_OFFLINE=1
        Cache miss raises `LookupError` instead of calling `fetch`. Use in
        CI: tests pass against a committed fixture cache, never touch the
        live API. Set in CI workflow yaml; never default to it locally.

      F3DX_BENCH_REFRESH=1
        Force-refresh: bypass cache, call `fetch`, overwrite the cached
        entry with the new response. Use when intentionally re-recording
        the fixture (e.g. after the upstream model changes behavior).

    `encoder` / `decoder` default to JSON; pass custom funcs if the
    response object isn't directly JSON-serializable. The `model` kwarg
    is stored as cache metadata for later analytics.

    Raises:
        LookupError: F3DX_BENCH_OFFLINE=1 and the request was not cached.
    """
    refresh = os.environ.get("F3DX_BENCH_REFRESH") == "1"
    offline = os.environ.get("F3DX_BENCH_OFFLINE") == "1"

    if not refresh:
        cached = cache.peek(request)
        if cached is not None:
            return decoder(cached)

    if offline:
        raise LookupError(
            "F3DX_BENCH_OFFLINE=1 and request not in cache. "
            "Refresh the fixture by re-running with F3DX_BENCH_REFRESH=1 "
            "(API key required) and commit the updated cache file."
        )

    response = fetch(request)
    cache.put(request, encoder(response), model=model)
    return response


# Re-export tools submodule symbols at package level so users can do
# `from f3dx.cache import cache_tool_call, FileWitness` without the
# extra import path. Done at the bottom to avoid circular import.
from f3dx.cache.tools import (  # noqa: E402
    CompositeWitness,
    EnvWitness,
    FileWitness,
    TTLWitness,
    cache_tool_call,
    fingerprint_args,
)
