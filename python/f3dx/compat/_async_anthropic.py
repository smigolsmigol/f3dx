"""anthropic.AsyncAnthropic subclass routing messages.create through f3dx Rust.

Same async-bridge pattern as f3dx.compat.AsyncOpenAI: wrap the sync rust
client with asyncio.to_thread. Rust releases the GIL during the network
wait so other asyncio tasks make progress on the same loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

try:
    import anthropic as _anthropic
except ImportError as e:
    raise ImportError(
        "f3dx.compat.AsyncAnthropic requires the anthropic SDK. "
        "Install with: pip install f3dx[anthropic-compat]"
    ) from e

from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from f3dx import Anthropic as _F3dxAnthropic

_SENTINELS: tuple[type, ...] = ()
try:
    from anthropic._types import NotGiven as _NotGiven, Omit as _Omit
    _SENTINELS = (_Omit, _NotGiven)
except ImportError:
    pass


def _strip_omit(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in d.items()
        if v is not None and (not _SENTINELS or not isinstance(v, _SENTINELS))
    }


class _F3dxAsyncRawResponse:
    """Mimics anthropic.APIResponse[Message] enough for downstream callers."""

    def __init__(self, parsed: Any) -> None:
        self._parsed = parsed
        self.headers: dict[str, str] = {}
        self.http_response: Any = None

    def parse(self) -> Any:
        return self._parsed


class _F3dxAsyncRawResponseProxy:
    def __init__(self, create_fn: Any) -> None:
        self._create_fn = create_fn

    async def create(self, **kwargs: Any) -> _F3dxAsyncRawResponse:
        parsed = await self._create_fn(**kwargs)
        return _F3dxAsyncRawResponse(parsed)


class AsyncAnthropic(_anthropic.AsyncAnthropic):
    """Drop-in for anthropic.AsyncAnthropic with messages.create routed via f3dx.

    Non-streaming returns anthropic.types.Message. Streaming returns an async
    iterator of raw event dicts (anthropic SSE event shape). Other resources
    (beta, batches, files) fall through to the upstream SDK unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        f3dx_opts = kwargs.pop("f3dx_options", None) or {}
        super().__init__(*args, **kwargs)
        api_key = kwargs.get("api_key") or getattr(self, "api_key", None)
        base_url = kwargs.get("base_url")
        if base_url is None:
            base_url_attr = getattr(self, "base_url", None)
            base_url = str(base_url_attr) if base_url_attr is not None else None
        self._f3dx = _F3dxAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=f3dx_opts.get("timeout", 60.0),
            http2=f3dx_opts.get("http2", True),
        )
        self.messages.create = self._f3dx_create  # type: ignore[method-assign]
        self.messages.with_raw_response = _F3dxAsyncRawResponseProxy(self._f3dx_create)
        # pydantic-ai's AnthropicModel calls client.beta.messages.create — same
        # wire format, but the SDK validates the dict into BetaMessage (with
        # BetaTextBlock / BetaToolUseBlock content items) rather than the
        # plain Message. Route both surfaces through f3dx; pick the right
        # validator per surface so downstream isinstance checks pass.
        self.beta.messages.create = self._f3dx_create_beta  # type: ignore[method-assign]
        self.beta.messages.with_raw_response = _F3dxAsyncRawResponseProxy(self._f3dx_create_beta)

    async def _f3dx_create(self, **kwargs: Any) -> Message | AsyncIterator[dict[str, Any]]:
        if kwargs.pop("stream", False):
            return self._f3dx_create_stream(_strip_omit(kwargs))
        out = await asyncio.to_thread(self._f3dx.messages_create, _strip_omit(kwargs))
        return Message.model_validate(out)

    async def _f3dx_create_beta(
        self, **kwargs: Any
    ) -> BetaMessage | AsyncIterator[dict[str, Any]]:
        if kwargs.pop("stream", False):
            return self._f3dx_create_stream(_strip_omit(kwargs))
        out = await asyncio.to_thread(self._f3dx.messages_create, _strip_omit(kwargs))
        return BetaMessage.model_validate(out)

    async def _f3dx_create_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        sync_iter = await asyncio.to_thread(self._f3dx.messages_create_stream, request)
        sentinel = object()

        def _next() -> Any:
            try:
                return next(sync_iter)
            except StopIteration:
                return sentinel

        while True:
            event = await asyncio.to_thread(_next)
            if event is sentinel:
                return
            yield event
