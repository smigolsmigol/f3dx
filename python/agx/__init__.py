"""agx — Rust-core runtime for pydantic-ai.

Surfaces shipped:
  agx.AgentRuntime    -- agent loop with concurrent tool dispatch (5-10x on multi-tool turns)
  agx.OpenAI          -- drop-in for openai.OpenAI; chat completions sync + streaming via Rust
  agx.Anthropic       -- drop-in for anthropic.Anthropic; messages sync + streaming via Rust
"""

from agx._agx import (  # type: ignore[attr-defined]
    AgentRuntime,
    AnthropicClient as Anthropic,
    OpenAIClient as OpenAI,
    __version__,
)

__all__ = ["AgentRuntime", "Anthropic", "OpenAI", "__version__"]
