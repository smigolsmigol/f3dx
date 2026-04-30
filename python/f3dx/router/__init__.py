"""f3dx.router: in-process Rust router for LLM providers.

Composes with hosted gateways like llmkit. The hosted gateway owns
billing, dashboards, multi-tenant config. f3dx.router owns the
in-process Rust hot path inside an agent loop where the network hop
to the gateway would burn budget.

Two routing policies:
  - sequential: try providers in order, hot-swap on 429/5xx, sub-1ms
  - hedged: fan out N parallel requests, first response wins

Built into the f3dx wheel as part of Phase B consolidation
(2026-04-30); previously shipped as the standalone f3dx-router
package on PyPI.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from f3dx._f3dx.router import Router as _NativeRouter  # type: ignore[attr-defined]

__all__ = ["Router"]


class Router:
    """Build a router from a list of provider dicts.

    Each provider dict needs:
        name        str             logical id used in hot-swap logs
        kind        "openai"|"anthropic"
        base_url    str             provider HTTP endpoint
        api_key     str             auth token
        timeout_ms  int = 30000     per-request timeout
        weight      int = 1         relative weight for weighted policies

    Policies:
        "sequential" (default) -- try in order, hot-swap on 429/5xx
        "hedged"               -- fan out hedge_k parallel; first wins

    The Python wrapper accepts a request body as a dict and serializes;
    the native Router takes a JSON string body to keep the FFI surface
    flat.
    """

    def __init__(
        self,
        providers: list[dict[str, Any]],
        *,
        policy: str = "sequential",
        hedge_k: int = 2,
    ) -> None:
        self._inner = _NativeRouter(providers, policy=policy, hedge_k=hedge_k)

    def chat_completions(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return self._inner.chat_completions(json.dumps(body))
