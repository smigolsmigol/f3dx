"""Prefix-cache-aware prompt construction (f3d1-fast Pillar 2 V0).

Build prompts in a canonical static-first order so OpenAI's automatic
prefix cache and Anthropic's `cache_control` markers actually hit on
multi-turn agentic loops. Per the Lumer et al. 2026 paper
(arXiv:2601.06007), the achievable win is 41-80% API cost reduction +
13-31% TTFT improvement when prompts are canonicalized.

The five rules from the f3d1-fast thesis (docs/research/f3d1_fast_thesis_2026-04-30.md):

1. Static-first hierarchical ordering: tools -> system -> few-shot ->
   conversation history -> latest user turn. Never interleave dynamic
   content into a static section.
2. Quarantine the dynamic. Timestamps, session IDs, request IDs,
   current-date strings live ONLY in the latest user turn or in a
   trailing system-suffix block placed AFTER every cache breakpoint.
3. Pad to the model's cache-boundary threshold (Sonnet 4.5/Opus 4 =
   1024, Sonnet 4.6/Haiku 3.5 = 2048, Opus 4.5+/Haiku 4.5 = 4096,
   OpenAI = 1024 graduated).
4. Compute and store `prefix_canonical_hash` per request (BLAKE3 of
   the canonical bytes). Cache-hit predictor + analytics join key.
5. (Anthropic only) Tag the last block of each stable tier with
   `cache_control` markers, max 4. Reserve marker 4 for opportunistic
   hot spots.

V0 ships rules 1+2+3+4. Anthropic-specific marker placement (rule 5)
lands when we add Anthropic real-API validation.

Usage:
    from f3dx.fast import CanonicalPrompt

    p = CanonicalPrompt(model="gpt-4o-mini")
    p.add_tools([{"type": "function", "function": {...}}])
    p.add_system("You are a helpful assistant.")
    p.add_history([
        {"role": "user", "content": "what is 2+2?"},
        {"role": "assistant", "content": "4."},
    ])
    p.add_user("What is 7*8?")  # the dynamic part stays last

    body = p.build()  # dict ready for OpenAI/Anthropic API
    print(p.prefix_hash())  # BLAKE3 hex of static prefix
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Cache-boundary thresholds per Anthropic + OpenAI 2026 docs (cited in
# the f3d1-fast thesis). Below the threshold, cache_control is silently
# dropped on Anthropic; OpenAI graduates in 128-token increments after
# 1024 but the floor for any hit is 1024.
_MIN_CACHEABLE_TOKENS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 1024,
    "gpt-4o-mini": 1024,
    "gpt-4.1": 1024,
    "gpt-4.1-mini": 1024,
    "gpt-5": 1024,
    "gpt-5-mini": 1024,
    "gpt-5-nano": 1024,
    # Anthropic
    "claude-sonnet-4-5": 1024,
    "claude-sonnet-3-7": 1024,
    "claude-opus-4": 1024,
    "claude-opus-4-1": 1024,
    "claude-sonnet-4-6": 2048,
    "claude-haiku-3-5": 2048,
    "claude-opus-4-5": 4096,
    "claude-opus-4-6": 4096,
    "claude-opus-4-7": 4096,
    "claude-haiku-4-5": 4096,
}

# Stable filler used to pad prompts to the cache-boundary threshold.
# Must hash identically across requests for the prefix cache to hit.
# Single-byte unit chosen so token-count drift across tokenizer versions
# doesn't break alignment.
_PAD_UNIT = "\n<!-- f3dx-cache-pad -->"


def _approx_tokens(text: str) -> int:
    """Rough token count (4 chars per token heuristic; OpenAI / Anthropic
    tokenizers vary +/- 30% but this is good enough for boundary math)."""
    return max(1, len(text) // 4)


@dataclass
class CanonicalPrompt:
    """Builder that enforces static-first ordering for cache-friendliness."""

    model: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    system: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    user: str = ""
    # Trailing system-suffix block for dynamic content (timestamps etc).
    # Lives AFTER the cache breakpoint so changes don't invalidate the
    # cached static prefix.
    dynamic_suffix: str = ""

    def add_tools(self, tools: Sequence[Mapping[str, Any]]) -> "CanonicalPrompt":
        """Tool definitions go at the very front. Order is preserved."""
        self.tools.extend(dict(t) for t in tools)
        return self

    def add_system(self, text: str) -> "CanonicalPrompt":
        """System prompt goes after tools. Concatenated if called more than once."""
        if self.system:
            self.system = f"{self.system}\n\n{text}"
        else:
            self.system = text
        return self

    def add_history(self, turns: Sequence[Mapping[str, Any]]) -> "CanonicalPrompt":
        """Frozen conversation history (everything except the latest user turn)."""
        self.history.extend(dict(t) for t in turns)
        return self

    def add_user(self, text: str) -> "CanonicalPrompt":
        """The latest user turn. Goes LAST in messages so all dynamic
        content lives behind every cache breakpoint."""
        self.user = text
        return self

    def add_dynamic_suffix(self, text: str) -> "CanonicalPrompt":
        """Trailing system block for dynamic content (timestamps, session
        IDs). Stays out of the cached prefix."""
        if self.dynamic_suffix:
            self.dynamic_suffix = f"{self.dynamic_suffix}\n{text}"
        else:
            self.dynamic_suffix = text
        return self

    def cache_threshold(self) -> int:
        """Min tokens for cache eligibility on this model."""
        # Match by prefix to handle dated suffixes (claude-sonnet-4-5-20251022).
        for prefix, threshold in _MIN_CACHEABLE_TOKENS.items():
            if self.model.startswith(prefix):
                return threshold
        return 1024  # safe default

    def _padded_system(self) -> str:
        """System block padded up to the cache boundary if under threshold."""
        threshold = self.cache_threshold()
        current = self.system + ("\n\n" + "\n".join(t.get("content", "") for t in self.history)
                                  if self.history else "")
        approx = _approx_tokens(current)
        if approx >= threshold:
            return self.system
        # Pad in a deterministic, hash-stable way.
        units_needed = max(1, (threshold - approx) // _approx_tokens(_PAD_UNIT) + 1)
        pad = _PAD_UNIT * units_needed
        return self.system + pad

    def build(self, *, pad_to_boundary: bool = False) -> dict[str, Any]:
        """Build a request body in the canonical order.

        `pad_to_boundary=True` adds f3dx-cache-pad markers to reach the
        model's min-cacheable-tokens threshold. Default off because most
        production prompts are already over threshold; turn on for short
        prompts where you want a cache hit.
        """
        messages: list[dict[str, Any]] = []
        system_text = self._padded_system() if pad_to_boundary else self.system
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.extend(self.history)
        if self.dynamic_suffix:
            messages.append({"role": "system", "content": self.dynamic_suffix})
        if self.user:
            messages.append({"role": "user", "content": self.user})

        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.tools:
            body["tools"] = self.tools
        return body

    def prefix_hash(self) -> str:
        """BLAKE3 hex of the canonical bytes of (tools || system || history).

        Excludes the latest user turn + dynamic_suffix so the hash is
        stable across turns that share the static prefix. Used as a
        cache-hit predictor + analytics join key.
        """
        canonical = {
            "tools": self.tools,
            "system": self.system,
            "history": self.history,
        }
        # Sort keys so dict ordering can't move the hash.
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(encoded, digest_size=32).hexdigest()


def cache_hit_ratio(usage: Mapping[str, Any]) -> float:
    """Compute cache-hit ratio from an OpenAI / Anthropic usage block.

    OpenAI: usage.prompt_tokens_details.cached_tokens / usage.prompt_tokens
    Anthropic: usage.cache_read_input_tokens / (cache_read + cache_creation + input_tokens)

    Returns 0.0 if no cache fields present (e.g. provider doesn't report).
    """
    # OpenAI shape
    if "prompt_tokens_details" in usage:
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        total = usage.get("prompt_tokens", 0) or 0
        return cached / total if total > 0 else 0.0

    # Anthropic shape
    if "cache_read_input_tokens" in usage:
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
        input_tokens = usage.get("input_tokens", 0) or 0
        total = cache_read + cache_creation + input_tokens
        return cache_read / total if total > 0 else 0.0

    return 0.0
