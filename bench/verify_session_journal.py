"""Verify the opt-in AgentRuntime session journal contract."""

import json
import tempfile
from pathlib import Path

import f3dx


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.journal"
        runtime = f3dx.AgentRuntime(
            system_prompt="be terse",
            session_journal_path=str(path),
        )
        mock = [
            json.dumps(
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "call-1", "name": "echo", "arguments": "{}"}
                    ],
                }
            ),
            json.dumps({"content": "done", "tool_calls": []}),
        ]
        result = runtime.run("hello", {"echo": lambda _args: "ok"}, mock)
        assert result["answer"] == "done"

        records = json.loads(f3dx.SessionJournal(str(path), "inspect").records_json())
        kinds = [record["kind"] for record in records]
        assert kinds == ["run_started", "model_turn", "tool_result", "model_turn", "run_completed"]
        assert records[2]["effect_id"] == "call-1"
        assert "prompt" not in records[0]
        assert [record["sequence"] for record in records] == list(range(len(records)))
    print("session journal: PASS")


if __name__ == "__main__":
    main()
