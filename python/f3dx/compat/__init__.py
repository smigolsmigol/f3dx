"""f3dx.compat — opt-in subclass shims for upstream SDK isinstance compatibility.

Importing requires the matching extra:
    pip install f3dx[openai-compat]
"""

from f3dx.compat._async_openai import AsyncOpenAI
from f3dx.compat._openai import OpenAI

__all__ = ["AsyncOpenAI", "OpenAI"]
