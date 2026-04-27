"""openai.AsyncOpenAI subclass routing chat.completions.create through f3dx Rust core.

Async path: the f3dx OpenAI client is sync (uses tokio block_on under the hood),
so we bridge to asyncio via asyncio.to_thread. The Rust client releases the GIL
during the network wait, so other asyncio tasks make progress on the same event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

try:
    import openai as _openai
except ImportError as e:
    raise ImportError(
        "f3dx.compat.AsyncOpenAI requires the openai SDK. Install with: pip install f3dx[openai-compat]"
    ) from e

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from f3dx import OpenAI as _F3dxOpenAI

_SENTINELS: tuple[type, ...] = ()
try:
    from openai._types import NotGiven as _NotGiven, Omit as _Omit
    _SENTINELS = (_Omit, _NotGiven)
except ImportError:  # older openai versions
    pass


def _strip_omit(d: dict[str, Any]) -> dict[str, Any]:
    """Drop openai SDK Omit/NotGiven sentinels and None values so f3dx sees a clean request."""
    return {
        k: v
        for k, v in d.items()
        if v is not None and (not _SENTINELS or not isinstance(v, _SENTINELS))
    }


class _F3dxAsyncRawResponse:
    """Async sibling of _F3dxRawResponse for langchain's ainvoke path."""

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


class AsyncOpenAI(_openai.AsyncOpenAI):
    """Drop-in for openai.AsyncOpenAI with chat.completions.create routed via f3dx.

    Non-streaming returns openai.types.chat.ChatCompletion. Streaming returns an
    async iterator of ChatCompletionChunk. Other resources fall through to the
    upstream openai SDK unchanged. Pass f3dx_options={'http2': bool, 'timeout': float}
    to surface the rust-only knobs without leaking them into the openai ctor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        f3dx_opts = kwargs.pop("f3dx_options", None) or {}
        super().__init__(*args, **kwargs)
        api_key = kwargs.get("api_key") or getattr(self, "api_key", None)
        base_url = kwargs.get("base_url")
        if base_url is None:
            base_url_attr = getattr(self, "base_url", None)
            base_url = str(base_url_attr) if base_url_attr is not None else None
        self._f3dx = _F3dxOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=f3dx_opts.get("timeout", 60.0),
            http2=f3dx_opts.get("http2", True),
        )
        self.chat.completions.create = self._f3dx_create  # type: ignore[method-assign]
        self.chat.completions.with_raw_response = _F3dxAsyncRawResponseProxy(self._f3dx_create)

    async def _f3dx_create(self, **kwargs: Any) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if kwargs.pop("stream", False):
            return self._f3dx_create_stream(_strip_omit(kwargs))
        out = await asyncio.to_thread(self._f3dx.chat_completions_create, _strip_omit(kwargs))
        return ChatCompletion.model_validate(out)

    async def _f3dx_create_stream(
        self, request: dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        sync_iter = await asyncio.to_thread(self._f3dx.chat_completions_create_stream, request)
        sentinel = object()

        def _next() -> Any:
            try:
                return next(sync_iter)
            except StopIteration:
                return sentinel

        while True:
            chunk = await asyncio.to_thread(_next)
            if chunk is sentinel:
                return
            yield ChatCompletionChunk.model_validate(chunk)
