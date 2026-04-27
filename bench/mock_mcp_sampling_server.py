"""Tiny Python MCP server that issues sampling/createMessage on tool call.

Used by verify_mcp_sampling.py to exercise the sampling-callback bridge
without depending on a specific external server's behavior.

Run as a child process from the verify script via stdio transport. The
sole tool `ask_llm` issues a createMessage request to the connected
client (us) and returns whatever the client's sampling callback produces.
"""

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

mcp = FastMCP("f3dx-test-sampling-server")


@mcp.tool()
async def ask_llm(prompt: str, ctx: Context) -> str:
    """Ask the connected client's LLM via sampling/createMessage."""
    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            ),
        ],
        max_tokens=200,
        system_prompt="be terse",
    )
    if hasattr(result.content, "text"):
        return result.content.text
    return str(result.content)


if __name__ == "__main__":
    mcp.run()
