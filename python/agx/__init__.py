"""agx — Rust-core runtime for pydantic-ai.

Whole-loop architecture: state lives in Rust for the run duration,
boundary crossings only for tool dispatch + final result.
"""

from agx._agx import AgentRuntime, __version__

__all__ = ["AgentRuntime", "__version__"]
