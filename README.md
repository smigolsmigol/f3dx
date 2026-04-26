# agx

Rust-core runtime primitives for [pydantic-ai](https://github.com/pydantic/pydantic-ai).

Polars for agents.

## Status

Day-1 spike. Two primitives shipped to validate the PyO3 boundary:

- `build_next_request(prior_messages, tool_results)` — splice tool results into prior history
- `render_messages(messages)` — flatten message list to model-input string

If the day-1 micro-bench beats pure-Python by 5x or more, the full Agent runtime port goes ahead.

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on linux/mac
pip install maturin
maturin develop --release
python bench/bench_day1.py
```

## Why

Production AI is compound systems (Zaharia, BAIR 2024). The orchestrator dominates wall-clock for multi-tool runs on cheap models. Nobody has rewritten the Python orchestration layer in Rust. agx does, drop-in via pydantic-ai's runtime extension point.

Side effect: every run emits an Arrow-shaped trace row. Agent traces become first-class Polars frames.

## License

MIT.
