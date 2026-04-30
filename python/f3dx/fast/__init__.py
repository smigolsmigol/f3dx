"""f3dx.fast: client-side inference acceleration utilities.

Implements pillars from the f3d1-fast thesis (2026-04-30; canonical doc:
docs/research/f3d1_fast_thesis_2026-04-30.md). The thesis identifies six
software-only levers for cutting Claude-Code-style 2-3 minute agentic
responses to <1.5 min, working against any closed API.

Currently shipped:
- Pillar 6 -- `budget_max_tokens` and friends from `f3dx.fast.budget`.
  Set max_tokens dynamically from prior-turn observations to avoid the
  runaway-generation tail. Precedent: SelfBudgeter (arXiv 2505.11274).

Roadmap (tasks #135-#142):
- Pillar 4: speculative tool execution (Sutradhara streaming JSON parser)
- Pillar 2: prefix-cache canonicalization with measured auto-tuning
- Pillar 3: tool-result memoization (extends f3dx.cache)
- Pillar 5: free-wins bundle (fast tokenizers, h2 pool, compiled templates)
- Pillar 1 (year-2): hybrid local-target spec decoding via mistral.rs
"""
from __future__ import annotations

from f3dx.fast.budget import (
    BudgetEstimate,
    budget_max_tokens,
    estimate_from_history,
)
from f3dx.fast.prompt import CanonicalPrompt, cache_hit_ratio

__all__ = [
    "BudgetEstimate",
    "CanonicalPrompt",
    "budget_max_tokens",
    "cache_hit_ratio",
    "estimate_from_history",
]
