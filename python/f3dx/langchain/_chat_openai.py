"""ChatOpenAI subclass that injects f3dx-routed openai clients.

Why subclass: langchain_openai.ChatOpenAI builds its own openai.OpenAI /
openai.AsyncOpenAI from api_key + http_client unless `root_client` and
`root_async_client` are passed explicitly. Subclassing lets us hand it
f3dx.compat clients on construction so every chain that includes this
ChatOpenAI runs over the rust transport without callers thinking about it.
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
except ImportError as e:
    raise ImportError(
        "f3dx.langchain requires langchain-openai. Install with: pip install f3dx[langchain]"
    ) from e

from f3dx.compat import AsyncOpenAI, OpenAI


class ChatOpenAI(_ChatOpenAI):
    """Drop-in for langchain_openai.ChatOpenAI with f3dx-routed transport.

    Pass `f3dx_options={'http2': bool, 'timeout': float}` to surface the
    rust-only knobs. All other kwargs forward to the upstream ChatOpenAI ctor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        f3dx_opts = kwargs.pop("f3dx_options", None)
        api_key = kwargs.get("api_key") or kwargs.get("openai_api_key")
        base_url = kwargs.get("base_url") or kwargs.get("openai_api_base")

        if "root_client" not in kwargs and "client" not in kwargs:
            sync = OpenAI(api_key=api_key, base_url=base_url, f3dx_options=f3dx_opts)
            kwargs["root_client"] = sync
            kwargs["client"] = sync.chat.completions
        if "root_async_client" not in kwargs and "async_client" not in kwargs:
            async_ = AsyncOpenAI(api_key=api_key, base_url=base_url, f3dx_options=f3dx_opts)
            kwargs["root_async_client"] = async_
            kwargs["async_client"] = async_.chat.completions

        super().__init__(*args, **kwargs)
