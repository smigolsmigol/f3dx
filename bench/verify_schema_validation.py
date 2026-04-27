"""Phase E V0.2 verify: chat_completions_create_stream_assembled accepts an
output_schema kwarg. When the parsed JSON conforms, validated_output fires.
When it doesn't, validation_error carries the schema violation with kind=schema.
Reuses the JSON-streaming mock server."""

from __future__ import annotations

import f3dx
from mock_openai_json_server import serve

PORT_GOOD = 8783
PORT_BAD = 8784

# Schema the GOOD payload satisfies and the BAD payload violates. The mock
# server's GOOD body is:
#   {"intent":"search","query":"forensic intel","filters":{"resident_id":"0000862794","k":5}}
SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["search", "summarize"]},
        "query": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "resident_id": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["resident_id"],
        },
    },
    "required": ["intent", "query", "filters"],
}

# Schema the GOOD payload VIOLATES (intent enum doesn't include 'search')
TIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["summarize", "translate"]},
    },
    "required": ["intent"],
}


def run_case(label: str, port: int, mode: str, schema, expect_event_type: str) -> None:
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
                output_schema=schema,
            )
        )
        terminal = next((e for e in events if e["type"] in ("validated_output", "validation_error")), None)
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
    finally:
        srv.shutdown()
    print()


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    run_case("good payload + permissive schema -> validated_output",
             PORT_GOOD, "good", SCHEMA, "validated_output")
    run_case("good payload + strict-enum schema -> validation_error (kind=schema)",
             PORT_GOOD + 10, "good", TIGHT_SCHEMA, "validation_error")
    run_case("malformed payload + any schema -> validation_error (kind=json_parse)",
             PORT_BAD, "bad", SCHEMA, "validation_error")
    print("OK — Phase E V0.2 schema-aware validation verified")


if __name__ == "__main__":
    main()
