"""f3dx.atlas_rtc — composition layer over Christopher Cruz's ATLAS-RTC.

ATLAS-RTC ([cruz209/ATLAS-RTC](https://github.com/cruz209/ATLAS-RTC)) is a
runtime control layer that enforces structured outputs at decode time:
monitors generation step-by-step, detects drift from output contracts,
and applies targeted interventions (logit masking, rollback, re-steering)
to ensure first-attempt validity without retries.

f3dx.atlas_rtc wraps ATLAS-RTC's RuntimeController under f3dx's
observability surface — every controlled completion emits a
`f3dx.atlas_rtc.run` OTel span with intervention counts, validity, and
contract metadata. The two layers compose cleanly:

    f3dx        — perf-fast-path runtime + observability + transport
    ATLAS-RTC   — token-level decode control + drift detection

Cloud APIs (OpenAI, Anthropic) don't expose decode-time control, so this
extra is most useful with local models via vLLM or HuggingFace
adapters that ATLAS-RTC ships natively.

    pip install f3dx[atlas-rtc]
"""

from f3dx.atlas_rtc._wrap import controlled_completion

__all__ = ["controlled_completion"]
