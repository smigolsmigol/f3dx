"""Speculative tool execution (f3d1-fast Pillar 4 V0).

Start running side-effect-free tools as soon as the streamed `tool_use`
chunks contain enough info, BEFORE the model finalizes the call. When
the model finalizes as expected, results are already there. When the
model retracts mid-stream, rollback the speculation.

Implements the primitive from Sutradhara (arXiv:2601.12967) -- streaming
JSON parser on the SSE event stream, gates dispatch on the parser's
"complete object" signal (not partial fields, that's the failure mode
the papers explicitly handle). Six 2025-2026 papers cluster on this
idea reporting 30-50% wall-clock reductions on agentic loops.

Failure-mode contract per Sherlock (arXiv:2511.00330) and Speculative
Actions (arXiv:2510.04371, ICLR 2026 oral):
  - Lossless ONLY for sandboxed/reversible operations: filesystem
    reads, MCP reads, idempotent Bash queries, HTTP GETs
  - NOT lossless for state-mutating tools without compensating action:
    file writes, git pushes, DB writes, external POSTs
  - Tool whitelist required; never speculate state-mutating tools

Usage:
    import json
    from f3dx.fast.spec_tool import StreamingJSONAccumulator, SpecToolDispatcher

    SAFE_TOOLS = {"Read", "Glob", "Grep", "Bash_status", "MCP_read"}

    def run_tool(name: str, args: dict) -> str:
        # caller-defined dispatcher; Read/Glob/Grep/etc.
        ...

    dispatcher = SpecToolDispatcher(safe_tools=SAFE_TOOLS, fetch=run_tool)

    for event in client.chat_completions_create_stream({...}):
        # Tool calls arrive as `delta.tool_calls[i].function.{name, arguments}` chunks.
        # Arguments stream as JSON fragments. Feed each chunk into the
        # accumulator; when a tool_call's arguments parse cleanly, the
        # dispatcher fires the tool optimistically and stores the result.
        dispatcher.feed_delta(event)

    # When the agent loop finalizes its tool calls, harvest:
    for tool_call in finalized_tool_calls:
        result = dispatcher.harvest(tool_call)  # sub-100us if speculation hit
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamingJSONAccumulator:
    """Per-tool-call buffer that ingests JSON-argument fragments and
    detects when the full object has parsed cleanly.

    OpenAI / Anthropic stream tool-call arguments as text fragments
    that concatenate into a JSON object string. We try to parse on
    each fragment; first successful parse = "complete object" signal.
    """

    fragments: list[str] = field(default_factory=list)
    parsed: dict[str, Any] | None = None
    complete: bool = False

    def feed(self, fragment: str) -> bool:
        """Append a fragment; return True if just became complete."""
        if self.complete:
            return False
        self.fragments.append(fragment)
        text = "".join(self.fragments)
        try:
            self.parsed = json.loads(text)
            self.complete = True
            return True
        except json.JSONDecodeError:
            return False

    def text(self) -> str:
        return "".join(self.fragments)


@dataclass
class SpecAttempt:
    """Record of one speculation attempt. Drops into wata's spec_attempts
    table (proposed schema in task #140)."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    speculation_started_at_ns: int
    speculation_finished_at_ns: int = 0
    result: Any = None
    error: str | None = None
    accepted: bool = False  # True if model finalized as expected
    rolled_back: bool = False  # True if model retracted; result discarded


class SpecToolDispatcher:
    """Watch a streaming chat-completion event flow; fire safe tools as
    soon as their arguments parse cleanly; track speculation outcomes.

    Per the Sherlock + Speculative Actions papers, this is lossless ONLY
    for sandboxed/reversible operations. The `safe_tools` whitelist is
    the only safety boundary; the dispatcher does not enforce semantics.

    Federico's recommended whitelist for Claude-Code-style loops:
        SAFE = {"Read", "Glob", "Grep", "Bash_status", "MCP_read"}
        UNSAFE = {"Edit", "Write", "Bash_mutate", "MCP_write", "git_push"}
    """

    def __init__(
        self,
        *,
        safe_tools: set[str],
        fetch: Callable[[str, dict[str, Any]], Any],
        threaded: bool = True,
        max_workers: int = 8,
    ) -> None:
        """`threaded=True` (V0.1 default) runs each speculative tool on a
        worker thread so it executes in parallel with the remaining
        stream chunks. `threaded=False` runs the tool synchronously
        inside feed_delta -- useful for unit tests or when the caller
        needs deterministic single-thread execution.

        `max_workers` caps the threadpool when threaded=True. 8 is
        comfortable for typical agent loops with up to a few concurrent
        tool calls per turn; bump for fan-out heavy workloads.
        """
        self.safe_tools = safe_tools
        self.fetch = fetch
        self.threaded = threaded
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="f3dx-spec")
            if threaded else None
        )
        # Keyed by `index` (stable across all deltas of a single
        # tool_call) -- OpenAI sends `id` only on the FIRST delta and
        # `index` on every delta. Anthropic similarly uses
        # content_block_start.index. The id (when present) gets
        # recorded for harvest lookups.
        self._buffers: dict[int, StreamingJSONAccumulator] = {}
        self._names: dict[int, str] = {}
        self._ids: dict[int, str] = {}
        # Attempts keyed by both index and id so harvest(call_xyz) and
        # harvest_by_index(0) both work.
        self._attempts_by_index: dict[int, SpecAttempt] = {}
        self._attempts_by_id: dict[str, SpecAttempt] = {}
        # Open Futures (only populated when threaded=True). Resolved
        # results land back into the SpecAttempt via the future callback.
        self._futures: dict[int, Future[Any]] = {}

    def __enter__(self) -> SpecToolDispatcher:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """Drain pending speculations and release the threadpool."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def feed_delta(self, event: dict[str, Any]) -> list[SpecAttempt]:
        """Ingest one streamed delta event. Returns the list of attempts
        that just transitioned to "fired speculatively" on this delta.

        Expects OpenAI streaming chat completion shape:
          event["choices"][0]["delta"]["tool_calls"] = [
              {"index": 0, "id": "call_xyz", "function": {"name": "...", "arguments": "..."}},
              ...
          ]

        Anthropic shape (input_json_delta blocks) needs a small adapter;
        not in V0.
        """
        fired: list[SpecAttempt] = []
        choices = event.get("choices") or []
        if not choices:
            return fired
        delta = choices[0].get("delta") or {}
        tool_calls = delta.get("tool_calls") or []

        for tc in tool_calls:
            idx = tc.get("index", 0)
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            name_chunk = fn.get("name")
            args_chunk = fn.get("arguments")

            if tc_id and idx not in self._ids:
                self._ids[idx] = tc_id
            if name_chunk and idx not in self._names:
                self._names[idx] = name_chunk
            if idx not in self._buffers:
                self._buffers[idx] = StreamingJSONAccumulator()

            if args_chunk:
                just_complete = self._buffers[idx].feed(args_chunk)
                if just_complete:
                    name = self._names.get(idx, "")
                    args = self._buffers[idx].parsed or {}
                    if name in self.safe_tools:
                        resolved_id = self._ids.get(idx) or f"idx-{idx}"
                        attempt = self._fire(idx, resolved_id, name, args)
                        fired.append(attempt)

        return fired

    def _fire(
        self, idx: int, tc_id: str, name: str, args: dict[str, Any]
    ) -> SpecAttempt:
        """Run the tool optimistically. In threaded mode submits to the
        worker pool and returns immediately; otherwise runs sync."""
        attempt = SpecAttempt(
            tool_call_id=tc_id,
            tool_name=name,
            args=args,
            speculation_started_at_ns=time.perf_counter_ns(),
        )
        self._attempts_by_index[idx] = attempt
        self._attempts_by_id[tc_id] = attempt

        if self.threaded and self._executor is not None:
            # Submit to threadpool; record completes via callback so
            # harvest() can either return cached result or block on
            # Future.result().
            future = self._executor.submit(self.fetch, name, args)
            self._futures[idx] = future

            def _on_done(fut: Future[Any]) -> None:
                try:
                    attempt.result = fut.result()
                except Exception as e:
                    attempt.error = f"{type(e).__name__}: {e}"
                attempt.speculation_finished_at_ns = time.perf_counter_ns()

            future.add_done_callback(_on_done)
        else:
            # Sync path: run in-line
            try:
                attempt.result = self.fetch(name, args)
            except Exception as e:
                attempt.error = f"{type(e).__name__}: {e}"
            attempt.speculation_finished_at_ns = time.perf_counter_ns()

        return attempt

    def harvest(self, tool_call_id: str, *, timeout: float | None = None) -> Any | None:
        """Return the speculated result for a finalized tool call, or
        None if no speculation hit. Blocks on the worker thread's Future
        if the speculation is still running (threaded mode); returns
        immediately in sync mode. `timeout` raises TimeoutError if the
        speculation hasn't finished by then."""
        attempt = self._attempts_by_id.get(tool_call_id)
        if attempt is None:
            return None
        # Find idx for the future lookup
        idx = next(
            (i for i, a in self._attempts_by_index.items() if a is attempt),
            None,
        )
        if idx is not None and idx in self._futures:
            self._futures[idx].result(timeout=timeout)  # block until done
        if attempt.error is not None:
            return None
        attempt.accepted = True
        return attempt.result

    def harvest_by_index(self, idx: int, *, timeout: float | None = None) -> Any | None:
        """Same as harvest() but lookup by tool_call.index instead of id."""
        attempt = self._attempts_by_index.get(idx)
        if attempt is None:
            return None
        if idx in self._futures:
            self._futures[idx].result(timeout=timeout)
        if attempt.error is not None:
            return None
        attempt.accepted = True
        return attempt.result

    def rollback(self, tool_call_id: str) -> None:
        """Mark the speculation as rolled-back when the model retracts."""
        if tool_call_id in self._attempts_by_id:
            self._attempts_by_id[tool_call_id].rolled_back = True

    def attempts(self) -> list[SpecAttempt]:
        """All speculation records so far. For wata spec_attempts table."""
        return list(self._attempts_by_index.values())

    def acceptance_rate(self) -> float:
        """Fraction of speculations that the model finalized as expected.
        Above 70% is "the speculation is paying off"; below 30% means
        we're wasting cycles - tighten the whitelist."""
        attempts = self.attempts()
        if not attempts:
            return 0.0
        accepted = sum(1 for a in attempts if a.accepted)
        return accepted / len(attempts)
