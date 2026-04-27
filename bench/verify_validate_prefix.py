"""Phase E V0.2.1 verify: validate_json=True now fails fast when the
model prefaces with prose before emitting JSON. The validation_error
event fires the moment the first non-whitespace character of accumulated
content is not `{` or `[` — before the rest of the stream arrives,
saving the user wasted tokens.

Three scenarios via the json mock server:
  good  — emits valid JSON cleanly. Existing terminal validated_output fires.
  prose — emits "Sure, here is the JSON: {..." prefix. NEW V0.2.1 path:
          validation_error kind=json_prefix fires early.
  bad   — emits truncated JSON ("{...,"). Terminal validation_error
          kind=json_parse fires (prefix check passes since first char IS `{`).
"""

from __future__ import annotations

import f3dx
from mock_openai_json_server import serve

PORT_GOOD = 8790
PORT_PROSE = 8791
PORT_BAD = 8792


def run_case(label: str, port: int, mode: str, expect_event_type: str, expect_kind: str | None) -> None:
    print(f"-- {label} --")
    srv = serve(port, mode=mode)
    try:
        client = f3dx.OpenAI(
            api_key="test",
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout=30.0,
            http2=False,
        )
        events = list(
            client.chat_completions_create_stream_assembled(
                {"model": "json-bench", "messages": [{"role": "user", "content": "go"}]},
                validate_json=True,
            )
        )
        terminal = next(
            (e for e in events if e["type"] in ("validated_output", "validation_error")),
            None,
        )
        assert terminal is not None, "expected a terminal validation event"
        print(f"  emitted: type={terminal['type']}", end="")
        if terminal["type"] == "validation_error":
            print(f" kind={terminal.get('kind')}")
            print(f"  detail: {terminal.get('error', '')[:120]}")
        else:
            print()
            print(f"  data: {terminal.get('data')}")
        assert terminal["type"] == expect_event_type, (
            f"expected {expect_event_type}, got {terminal['type']}"
        )
        if expect_kind is not None:
            assert terminal.get("kind") == expect_kind, (
                f"expected kind={expect_kind}, got {terminal.get('kind')}"
            )
        # Verify we don't double-emit validation events
        validation_count = sum(
            1 for e in events if e["type"] in ("validated_output", "validation_error")
        )
        assert validation_count == 1, f"expected 1 validation event, got {validation_count}"
    finally:
        srv.shutdown()
    print()


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    run_case("good payload -> validated_output (existing V0.2 path)",
             PORT_GOOD, "good", "validated_output", None)
    run_case("prose-prefix payload -> validation_error kind=json_prefix (NEW V0.2.1 fail-fast)",
             PORT_PROSE, "prose", "validation_error", "json_prefix")
    run_case("truncated-JSON payload -> validation_error kind=json_parse (terminal; prefix is {)",
             PORT_BAD, "bad", "validation_error", "json_parse")
    print("OK — Phase E V0.2.1 incremental fail-fast on invalid JSON prefix verified")


if __name__ == "__main__":
    main()
