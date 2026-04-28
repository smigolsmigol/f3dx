"""Phase E V0 verify: validate_json=True path on assembled stream.

Stand the mock JSON-mode server up twice:
  1. mode=good -> stream a valid JSON across N fragments
                  expect: validated_output event with parsed dict
  2. mode=bad  -> stream an invalid (truncated) JSON across N fragments
                  expect: validation_error event with raw + error msg

Both cases save the user the accumulate-deltas-then-json.loads dance.
"""

from __future__ import annotations

import f3dx
from mock_openai_json_server import GOOD_JSON, serve

import json

PORT = 8775
N_FRAGMENTS = 10


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")

    # ---- positive path ----
    print("== mode=good (valid JSON streamed) ==")
    srv = serve(PORT, mode="good", n_fragments=N_FRAGMENTS)
    try:
        client = f3dx.OpenAI(
            api_key="test",
            base_url=f"http://127.0.0.1:{PORT}/v1",
            timeout=30.0,
            http2=False,
        )
        events = list(client.chat_completions_create_stream_assembled(
            {
                "model": "json-bench",
                "messages": [{"role": "user", "content": "give me JSON"}],
                "response_format": {"type": "json_object"},
            },
            validate_json=True,
        ))
        types = [e["type"] for e in events]
        validated = [e for e in events if e["type"] == "validated_output"]
        assert len(validated) == 1, f"expected 1 validated_output, got {types}"
        assert validated[0]["data"] == json.loads(GOOD_JSON), (
            f"data mismatch: {validated[0]['data']}"
        )
        print(f"  events: {types}")
        print(f"  validated data: {validated[0]['data']}")
        print("  -> validated_output emitted with parsed dict\n")
    finally:
        srv.shutdown()
        srv.server_close()

    # ---- negative path ----
    print("== mode=bad (truncated JSON streamed) ==")
    srv = serve(PORT, mode="bad", n_fragments=N_FRAGMENTS)
    try:
        client = f3dx.OpenAI(
            api_key="test",
            base_url=f"http://127.0.0.1:{PORT}/v1",
            timeout=30.0,
            http2=False,
        )
        events = list(client.chat_completions_create_stream_assembled(
            {
                "model": "json-bench",
                "messages": [{"role": "user", "content": "give me JSON"}],
                "response_format": {"type": "json_object"},
            },
            validate_json=True,
        ))
        types = [e["type"] for e in events]
        errors = [e for e in events if e["type"] == "validation_error"]
        assert len(errors) == 1, f"expected 1 validation_error, got {types}"
        print(f"  events: {types}")
        print(f"  validation_error.raw: {errors[0]['raw'][:60]}...")
        print(f"  validation_error.error: {errors[0]['error']}")
        print("  -> validation_error emitted with raw payload + parser msg\n")
    finally:
        srv.shutdown()
        srv.server_close()

    print("OK — Phase E V0 verified (validate_json=True works on both paths)")


if __name__ == "__main__":
    main()
