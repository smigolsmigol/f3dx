"""Token budget hinting from trace history (f3d1-fast Pillar 6).

Set `max_tokens` dynamically from prior-turn observations:
    max_tokens = ceil(p99(prior_counts) * safety_factor)

Avoids the runaway-generation tail where the model rambles past the
useful answer. Precedent: SelfBudgeter (arXiv 2505.11274).

Usage:
    from f3dx.fast import budget_max_tokens, estimate_from_history

    # From a list of prior completion-token counts:
    rec = budget_max_tokens([42, 38, 51, 45, 39, 47, 44, 41, 49, 43])
    print(rec.max_tokens)   # 61
    print(rec.confidence)   # 'high' (n >= 10)

    # Convenience: returns int directly, with a sample-size gate:
    cap = estimate_from_history(prior_counts, fallback=4096)
    request["max_tokens"] = cap

The estimator is intentionally pure-Python and dependency-free so it
ships in the f3dx wheel with no extra cost. Real-API validation sits in
examples/budget_real_api_bench.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ConfidenceLevel = Literal["low", "medium", "high"]

_CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class BudgetEstimate:
    """The recommended cap plus context for the calling code's gating decision."""

    max_tokens: int
    p99_observed: int
    sample_size: int
    confidence: ConfidenceLevel
    safety_factor: float


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile on a sorted list. q in [0, 1]."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("empty sequence")
    idx = max(0, min(n - 1, math.ceil(q * n) - 1))
    return sorted_values[idx]


def budget_max_tokens(
    prior_counts: list[int],
    *,
    safety_factor: float = 1.2,
    floor: int = 50,
    ceiling: int = 8192,
) -> BudgetEstimate:
    """Compute recommended max_tokens from prior turn token counts.

    `prior_counts` is the list of `completion_tokens` (or equivalent)
    from previous turns of similar shape. Returns a `BudgetEstimate`
    with the recommended cap plus confidence based on sample size.

    `floor` and `ceiling` clamp the recommendation against degenerate
    cases (single short reply -> at least `floor`; single very long
    reply -> at most `ceiling`).

    Sample-size buckets:
        n >= 10 -> 'high'
        5 <= n <= 9 -> 'medium'
        n <= 4 -> 'low' (caller should treat as untrustworthy)
    """
    if floor < 1:
        raise ValueError("floor must be >= 1")
    if ceiling < floor:
        raise ValueError("ceiling must be >= floor")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be > 0")

    if not prior_counts:
        return BudgetEstimate(
            max_tokens=ceiling,
            p99_observed=0,
            sample_size=0,
            confidence="low",
            safety_factor=safety_factor,
        )

    sorted_counts = sorted(int(c) for c in prior_counts)
    p99 = _percentile(sorted_counts, 0.99)
    raw = math.ceil(p99 * safety_factor)
    capped = max(floor, min(ceiling, raw))

    n = len(sorted_counts)
    if n >= 10:
        confidence: ConfidenceLevel = "high"
    elif n >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    return BudgetEstimate(
        max_tokens=capped,
        p99_observed=p99,
        sample_size=n,
        confidence=confidence,
        safety_factor=safety_factor,
    )


def estimate_from_history(
    prior_counts: list[int],
    *,
    fallback: int = 4096,
    confidence_required: ConfidenceLevel = "medium",
    safety_factor: float = 1.2,
) -> int:
    """Convenience wrapper: returns the recommended max_tokens int directly,
    or `fallback` if the sample size doesn't reach `confidence_required`.

    This is the right entry point when the caller doesn't care about the
    underlying statistics, just wants a number to hand to the API.
    """
    rec = budget_max_tokens(prior_counts, safety_factor=safety_factor)
    if _CONFIDENCE_ORDER[rec.confidence] < _CONFIDENCE_ORDER[confidence_required]:
        return fallback
    return rec.max_tokens
