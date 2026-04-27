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

_SENTINELS: tuple[type, ...] = ()
try:
    from openai._types import NotGiven as _NotGiven, Omit as _Omit
    _SENTINELS = (_Omit, _NotGiven)
except ImportError:
    pass


def _strip_omit(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in d.items()
        if v is not None and (not _SENTINELS or not isinstance(v, _SENTINELS))
    }


class _F3dxRawResponse:
    """Mimics openai.APIResponse[ChatCompletion] enough for langchain.

    langchain-openai 1.x calls `client.with_raw_response.create(**payload)` then
    `raw.parse()`. We pre-parse on the f3dx side and return the same object
    via parse(). headers/http_response stay empty since the rust client doesn't
    surface them today.
    """

    def __init__(self, parsed: Any) -> None:
        self._parsed = parsed
        self.headers: dict[str, str] = {}
        self.http_response: Any = None

    def parse(self) -> Any:
        return self._parsed


class _F3dxRawResponseProxy:
    def __init__(self, create_fn: Any) -> None:
        self._create_fn = create_fn

    def create(self, **kwargs: Any) -> _F3dxRawResponse:
        return _F3dxRawResponse(self._create_fn(**kwargs))


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
        self.chat.completions.with_raw_response = _F3dxRawResponseProxy(self._f3dx_create)

    def _f3dx_create(self, **kwargs: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if kwargs.pop("stream", False):
            return self._f3dx_create_stream(_strip_omit(kwargs))
        out = self._f3dx.chat_completions_create(_strip_omit(kwargs))
        return ChatCompletion.model_validate(out)

    def _f3dx_create_stream(self, request: dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        for chunk in self._f3dx.chat_completions_create_stream(request):
            yield ChatCompletionChunk.model_validate(chunk)
