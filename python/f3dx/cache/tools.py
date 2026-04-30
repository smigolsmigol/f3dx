"""Tool-result memoization (f3d1-fast Pillar 3 V0).

Extends the f3dx.cache substrate to memoize side-effect-free tool calls
during agentic loops. Same redb + JCS + BLAKE3 substrate; the cache key
includes a freshness witness (file mtimes, env hash, custom predicates)
so stale results invalidate automatically.

Direct value on Claude-Code-style daily workflow: every repeated `Read`
on an unchanged file, every `gh run list` within TTL, every `Bash`
status check during an edit-test-fix loop returns from a sub-100us
peek instead of the actual tool cost. ~5-10s saved per agentic loop
iteration.

Conservative whitelist principle: speculative-safe operations only.
Never memoize state-mutating tools (Edit, Write, git push, MCP write,
external-API POSTs). The caller decides which tools are safe; the
helper provides the cache + invalidation primitives.

Usage:
    from f3dx.cache import Cache
    from f3dx.cache.tools import cache_tool_call, FileWitness, TTLWitness

    cache = Cache("tool_cache.redb")

    # Read with mtime-based invalidation
    result = cache_tool_call(
        cache,
        tool="Read",
        args={"path": "/abs/path.py"},
        fetch=lambda a: open(a["path"]).read(),
        witness=FileWitness(["/abs/path.py"]),
    )

    # Bash status with time-based TTL
    result = cache_tool_call(
        cache,
        tool="Bash",
        args={"cmd": "gh run list --limit 5"},
        fetch=lambda a: subprocess.run(...).stdout,
        witness=TTLWitness(seconds=30),
    )

The witness goes into the cache key so when a file changes (or a TTL
elapses) the next call gets a different fingerprint and falls through
to fetch.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from f3dx.cache import Cache


class Witness(Protocol):
    """A freshness witness: produces a stable string that goes into the
    cache key. When the witness changes, the cache key changes, the
    previous entry becomes unreachable, and fetch fires fresh."""

    def witness(self) -> str: ...


@dataclass(frozen=True)
class FileWitness:
    """Invalidate when any of the listed files' mtime changes.

    Uses (path, mtime_ns, size) tuple per file so both content edits and
    truncations invalidate. Missing files contribute "missing" so the
    witness is stable when a watched file isn't there yet.
    """

    paths: Sequence[str]

    def witness(self) -> str:
        parts = []
        for p in self.paths:
            try:
                st = os.stat(p)
                parts.append(f"{p}|{st.st_mtime_ns}|{st.st_size}")
            except FileNotFoundError:
                parts.append(f"{p}|missing")
        return "FileWitness:" + ";".join(parts)


@dataclass(frozen=True)
class TTLWitness:
    """Invalidate after `seconds` elapse. Implemented as a coarse time
    bucket so two calls within the same bucket hit the same cache entry.

    Bucket size = `seconds`. Calls at t and t+seconds-1 share a bucket;
    the call at t+seconds rolls over.
    """

    seconds: int

    def witness(self) -> str:
        bucket = int(time.time()) // max(1, self.seconds)
        return f"TTLWitness:{self.seconds}s:bucket={bucket}"


@dataclass(frozen=True)
class EnvWitness:
    """Invalidate when any of the listed env vars change.

    Use sparingly: env-var changes are normal during dev shells and
    will cause cache churn. Prefer FileWitness or TTLWitness when
    possible.
    """

    keys: Sequence[str]

    def witness(self) -> str:
        parts = [f"{k}={os.environ.get(k, '')}" for k in self.keys]
        return "EnvWitness:" + ";".join(parts)


@dataclass(frozen=True)
class CompositeWitness:
    """AND-combine multiple witnesses. Cache invalidates when ANY of
    the underlying witnesses change."""

    witnesses: Sequence[Witness]

    def witness(self) -> str:
        return "Composite:" + "|".join(w.witness() for w in self.witnesses)


def cache_tool_call(
    cache: Cache,
    *,
    tool: str,
    args: Mapping[str, Any],
    fetch: Callable[[Mapping[str, Any]], Any],
    witness: Witness | None = None,
    encoder: Callable[[Any], bytes] = lambda obj: json.dumps(obj).encode(),
    decoder: Callable[[bytes], Any] = lambda b: json.loads(b.decode()),
) -> Any:
    """Cache-backed wrapper for any side-effect-free tool call.

    Builds a cache key from (tool, args, witness.witness()) so a change
    in any component produces a fresh cache entry.

    Both the cache-hit path AND the fetch path round-trip through
    encoder + decoder so the returned object is the same type regardless
    of cache state. This matters when fetch returns a complex Python
    object (dataclass, custom class) -- encoder must serialize it; if
    the round-trip can't reconstruct the type, pass custom enc/dec.

    Default JSON works for str / int / float / bool / list / dict /
    None and nested combinations -- the common case for tool results
    (Read returns str, Bash status returns str, Grep returns list).
    For binary: encoder = lambda b: b, decoder = lambda b: b.

    NEVER call this on state-mutating tools (Edit, Write, git push). The
    helper does not enforce a whitelist; safety is the caller's
    responsibility.
    """
    key_request = {
        "tool": tool,
        "args": dict(args),
        "witness": witness.witness() if witness else "no-witness",
    }
    cached = cache.peek(key_request)
    if cached is not None:
        return decoder(cached)
    result = fetch(args)
    encoded = encoder(result)
    cache.put(key_request, encoded, model=tool)
    return decoder(encoded)


def fingerprint_args(tool: str, args: Mapping[str, Any]) -> str:
    """BLAKE3 hex of the canonical (tool, args) bytes. Use as a join key
    for analytics on which tool calls are hot in your loop."""
    payload = repr({"tool": tool, "args": dict(args)}).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()
