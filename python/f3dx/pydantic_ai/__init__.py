"""f3dx.pydantic_ai — integration package for pydantic-ai.

Drop-in factories that build pydantic-ai Models routed through the f3dx Rust
core, plus an AbstractCapability that tags spans + counts dispatches.

    pip install f3dx[pydantic-ai]

Usage:
    from f3dx.pydantic_ai import openai_model, F3dxCapability
    from pydantic_ai import Agent

    agent = Agent(openai_model('gpt-4'), capabilities=[F3dxCapability()])
    result = await agent.run('hello')
"""

from f3dx.pydantic_ai._capability import F3dxCapability
from f3dx.pydantic_ai._models import anthropic_model, openai_model

__all__ = ["F3dxCapability", "anthropic_model", "openai_model"]
