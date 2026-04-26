"""f3dx — F3D1 Rust-core runtime for pydantic-ai.

Surfaces shipped:
  f3dx.AgentRuntime    -- agent loop with concurrent tool dispatch (5-10x on multi-tool turns)
  f3dx.OpenAI          -- drop-in for openai.OpenAI; chat completions sync + streaming via Rust
  f3dx.Anthropic       -- drop-in for anthropic.Anthropic; messages sync + streaming via Rust
"""

from f3dx._f3dx import (  # type: ignore[attr-defined]
    AgentRuntime,
    AnthropicClient as Anthropic,
    OpenAIClient as OpenAI,
    __version__,
)

__all__ = ["AgentRuntime", "Anthropic", "OpenAI", "__version__"]
