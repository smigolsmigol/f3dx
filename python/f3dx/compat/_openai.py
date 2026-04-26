"""openai.OpenAI subclass routing chat.completions.create through f3dx Rust core.

Why subclass: instructor / litellm / smolagents / langchain-openai / pydantic-ai's
OpenAIModel all do `isinstance(client, openai.OpenAI)` checks. A standalone
PyO3 class fails them. Subclassing lets the Rust path slot in transparently.
"""

from __future__ import annotations

from typing import Any, Iterator

try:
    import openai as _openai
except ImportError as e:
    raise ImportError(
        "f3dx.compat.OpenAI requires the openai SDK. Install with: pip install f3dx[openai-compat]"
    ) from e

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from f3dx import OpenAI as _F3dxOpenAI


class OpenAI(_openai.OpenAI):
    """Drop-in for openai.OpenAI with chat.completions.create routed via f3dx Rust.

    Non-streaming: returns openai.types.chat.ChatCompletion (pydantic-validated
    from the Rust dict). Streaming: returns an iterator yielding ChatCompletionChunk.
    Other resources (embeddings, images, files, batches, etc.) fall through to
    the upstream openai SDK unchanged.
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

    def _f3dx_create(self, **kwargs: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if kwargs.pop("stream", False):
            return self._f3dx_create_stream(kwargs)
        out = self._f3dx.chat_completions_create(kwargs)
        return ChatCompletion.model_validate(out)

    def _f3dx_create_stream(self, request: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        for chunk in self._f3dx.chat_completions_create_stream(request):
            yield ChatCompletionChunk.model_validate(chunk)
