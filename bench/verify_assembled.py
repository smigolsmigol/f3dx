"""Phase D verify: agx assembles fragmented tool_calls Rust-side.

Stand the mock tool-call server up, hit it with both surfaces:
  1. raw chat_completions_create_stream         -> sees N fragment chunks
  2. chat_completions_create_stream_assembled   -> sees ONE tool_call
                                                   event with parsed dict args

The win is in lines-of-user-code: raw mode forces the user to
accumulate fragments + json.loads at the end. Assembled mode hands
them a parsed dict.
"""

from __future__ import annotations

import json

import f3dx
from mock_openai_tool_server import ARGS_JSON, serve

PORT = 8771
N_FRAGMENTS = 12  # fragment the args string into 12 pieces


def main() -> None:
    print(f"agx version: {f3dx.__version__}\n")
    srv = serve(PORT, N_FRAGMENTS)
    try:
        client = f3dx.OpenAI(
            api_key="test",
            base_url=f"http://127.0.0.1:{PORT}/v1",
            timeout=30.0,
            http2=False,
        )

        # ---- 1. raw stream: user must reassemble manually ----
        print("== raw stream (user sees fragments) ==")
        raw_chunks = list(client.chat_completions_create_stream({
            "model": "tool-bench",
            "messages": [{"role": "user", "content": "go"}],
        }))
        n_arg_fragments = sum(
            1 for c in raw_chunks
            for choice in c.get("choices", [])
            for tc in (choice.get("delta", {}).get("tool_calls") or [])
            if tc.get("function", {}).get("arguments")
        )
        print(f"  total chunks: {len(raw_chunks)}")
        print(f"  argument-fragment chunks: {n_arg_fragments}")
        # what user code would have to write to reassemble
        accum = ""
        for c in raw_chunks:
            for choice in c.get("choices", []):
                for tc in (choice.get("delta", {}).get("tool_calls") or []):
                    accum += tc.get("function", {}).get("arguments", "")
        try:
            parsed = json.loads(accum)
        except json.JSONDecodeError as e:
            parsed = f"<json error: {e}>"
        print(f"  user-side reassembly result: {parsed!r}")
        print()

        # ---- 2. assembled stream: agx hands user the parsed dict ----
        print("== assembled stream (agx reassembles Rust-side) ==")
        events = list(client.chat_completions_create_stream_assembled({
            "model": "tool-bench",
            "messages": [{"role": "user", "content": "go"}],
        }))
        for e in events:
            print(f"  event: {e}")
        print()

        # ---- assertions ----
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) == 1, f"expected 1 tool_call event, got {len(tool_events)}"
        tc = tool_events[0]
        assert tc["id"] == "call_abc"
        assert tc["name"] == "search_evidence"
        assert tc["index"] == 0
        # Critical: arguments is a parsed dict, NOT a string
        assert isinstance(tc["arguments"], dict), f"arguments not dict: {type(tc['arguments'])}"
        assert tc["arguments"] == json.loads(ARGS_JSON), f"args mismatch: {tc['arguments']}"

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["finish_reason"] == "tool_calls"

        print("OK — Phase D verified")
        print(f"   raw stream: {len(raw_chunks)} chunks, user must accumulate {n_arg_fragments} fragments + json.loads")
        print(f"   assembled:  {len(events)} events, agx hands user parsed args dict ready to dispatch")
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
