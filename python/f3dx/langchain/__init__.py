"""f3dx.langchain — langchain-openai integration package.

ChatOpenAI subclass routes through the f3dx Rust core via the documented
root_client / root_async_client injection points. Drop-in for
langchain_openai.ChatOpenAI; every langchain feature (LCEL, structured
output, tool binding, streaming) keeps working unchanged.

    pip install f3dx[langchain]
"""

from f3dx.langchain._chat_openai import ChatOpenAI

__all__ = ["ChatOpenAI"]
