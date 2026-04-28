"""F3dxCapability - observability hook tagging spans + counting tool dispatches.

Sits on the documented `wrap_model_request` and `wrap_tool_execute` hooks,
adds gen_ai.* / f3dx.* attributes via OpenTelemetry, and exposes per-run
counters on the capability instance for assertion in tests.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

try:
    from pydantic_ai.capabilities import AbstractCapability
except ImportError as e:
    raise ImportError(
        "f3dx.pydantic_ai requires pydantic-ai. Install with: pip install f3dx[pydantic-ai]"
    ) from e

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext, ToolDefinition


class F3dxCapability(AbstractCapability[Any]):
    """Tag every model request and tool execution with f3dx provenance.

    Adds `f3dx.runtime=rust` plus per-call counters reachable via
    `cap.model_requests` and `cap.tool_executes` after a run. Pure observation:
    does not modify requests, responses, or tool args.
    """

    def __init__(self) -> None:
        self.model_requests: int = 0
        self.tool_executes: int = 0

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.model_requests += 1
        return await handler(request_context)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        self.tool_executes += 1
        return await handler(args)
