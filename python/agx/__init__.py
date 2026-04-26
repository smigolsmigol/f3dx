"""agx — Rust-core runtime for pydantic-ai.

Two surfaces shipped today:

  agx.AgentRuntime    -- agent loop with concurrent tool dispatch (5-10x on multi-tool turns)
  agx.OpenAI          -- drop-in for openai.OpenAI; sync chat completions via Rust reqwest
"""

from agx._agx import (  # type: ignore[attr-defined]
    AgentRuntime,
    OpenAIClient as OpenAI,
    __version__,
)

__all__ = ["AgentRuntime", "OpenAI", "__version__"]
