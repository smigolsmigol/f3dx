"""Model factories that build pydantic-ai Models with f3dx-routed transport.

The integration leans on pydantic-ai's own provider injection: rather than
fork OpenAIChatModel / AnthropicModel, we hand them a client whose
chat.completions.create / messages.create routes through the f3dx Rust core.
This keeps the f3dx integration on the supported extension surface - every
pydantic-ai feature (toolsets, capabilities, streaming, structured output)
keeps working unchanged.
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError as e:
    raise ImportError(
        "f3dx.pydantic_ai requires pydantic-ai. Install with: pip install f3dx[pydantic-ai]"
    ) from e

from f3dx.compat import AsyncAnthropic, AsyncOpenAI


def openai_model(
    model_name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    f3dx_options: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> OpenAIChatModel:
    """Build a pydantic-ai OpenAIChatModel whose HTTP path goes through f3dx.

    All `model_kwargs` (settings, profile, system_prompt_role, ...) forward to
    OpenAIChatModel unchanged. Use `f3dx_options={'http2': bool, 'timeout': float}`
    to surface the rust-only knobs.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, f3dx_options=f3dx_options)
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client), **model_kwargs)


def anthropic_model(
    model_name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    f3dx_options: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> AnthropicModel:
    """Build a pydantic-ai AnthropicModel whose HTTP path goes through f3dx.

    Mirror of openai_model: provider injection via the documented
    AnthropicProvider(anthropic_client=...) hook. f3dx_options forwards
    {'http2': bool, 'timeout': float} to the rust client.
    """
    client = AsyncAnthropic(api_key=api_key, base_url=base_url, f3dx_options=f3dx_options)
    return AnthropicModel(
        model_name,
        provider=AnthropicProvider(anthropic_client=client),
        **model_kwargs,
    )
