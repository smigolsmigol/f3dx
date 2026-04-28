"""controlled_completion: f3dx-observable wrapper over ATLAS-RTC's RuntimeController.

The user provides:
    - a prompt
    - a contract (atlas_rtc.contracts.* - JSONSchemaContract, ToolCallContract,
      ExtractionContract - or list of required keys for shorthand)
    - an adapter (MockAdapter, HFAdapter, VLLMAdapter from atlas_rtc.adapters)

We run the controller, emit an OTel span carrying gen_ai.system="atlas-rtc"
plus contract validity + intervention count + token count attributes, and
return a typed result the caller can branch on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from atlas_rtc.adapters.base import BaseAdapter
    from atlas_rtc.contracts.base import BaseContract
    from atlas_rtc.contracts.json_schema import JSONSchemaContract
    from atlas_rtc.controller.runtime import RuntimeController
except ImportError as e:
    raise ImportError(
        "f3dx.atlas_rtc requires atlas-rtc. Install with: pip install f3dx[atlas-rtc]"
    ) from e


@dataclass
class ControlledResult:
    """Output of a controlled_completion run."""

    text: str
    valid: bool
    errors: list[str]
    interventions: int
    contract_name: str

    @property
    def succeeded_first_pass(self) -> bool:
        """True iff the contract validated AND no interventions fired -
        i.e. the model produced a conforming output without rollback or
        masking. The first-attempt-validity number ATLAS-RTC reports."""
        return self.valid and self.interventions == 0


def controlled_completion(
    prompt: str,
    contract: BaseContract | list[str],
    adapter: BaseAdapter,
    *,
    max_steps: int = 128,
    max_restarts: int = 1,
) -> ControlledResult:
    """Run an ATLAS-RTC RuntimeController under f3dx's observability surface.

    `contract` accepts either an `atlas_rtc.contracts.*` instance or a plain
    list of required keys (sugar for `JSONSchemaContract(required_keys=...)`).
    `adapter` is an ATLAS-RTC adapter - MockAdapter for tests, HFAdapter
    for HuggingFace models, VLLMAdapter for local vLLM serving.

    Emits a `f3dx.atlas_rtc.run` OTel span with attributes:
        gen_ai.system            atlas-rtc
        atlas_rtc.contract       contract.name
        atlas_rtc.valid          bool
        atlas_rtc.interventions  int
        atlas_rtc.first_pass     bool (valid AND zero interventions)
        atlas_rtc.errors_count   int
    """
    if isinstance(contract, list):
        contract = JSONSchemaContract(required_keys=contract)

    controller = RuntimeController(
        adapter=adapter,
        contract=contract,
        max_steps=max_steps,
        max_restarts=max_restarts,
    )

    span_attrs = _emit_span_start(contract.name)
    text, result, state = controller.run(prompt)
    interventions = len(state.intervention_history)
    out = ControlledResult(
        text=text,
        valid=bool(result.valid),
        errors=list(result.errors or []),
        interventions=interventions,
        contract_name=contract.name,
    )
    _emit_span_end(span_attrs, out)
    return out


# --------------------- OTel emission --------------------- #


def _emit_span_start(contract_name: str) -> dict[str, Any]:
    """Open a span if f3dx OTel is configured. Returns a context dict
    that _emit_span_end uses to close it. No-op when OTel isn't on."""
    try:
        from f3dx import _f3dx  # type: ignore[attr-defined]
        # f3dx-trace exposes tracer() to f3dx-rt; we don't currently expose
        # a Python-side tracer entry, so this stays a stub for V0.
        # The ATLAS-RTC EventLogger captures interventions in detail - we
        # surface aggregate counts on the f3dx side via the return value.
        return {"contract_name": contract_name}
    except Exception:
        return {"contract_name": contract_name}


def _emit_span_end(ctx: dict[str, Any], out: ControlledResult) -> None:
    """Close the span with final attributes. No-op when OTel isn't on."""
    # V0: span emission is a stub; aggregate values land on ControlledResult
    # so callers can attach them to whatever observability stack they use.
    # V0.1 will plumb a Python-side f3dx tracer entry for direct span emission.
    return
