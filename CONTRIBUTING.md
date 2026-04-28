# Contributing to f3dx

## Dev setup

```bash
git clone https://github.com/smigolsmigol/f3dx
cd f3dx
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate
pip install maturin pytest
maturin develop --release
```

`maturin develop --release` builds all four crates (`f3dx-py`, `f3dx-rt`, `f3dx-http`, `f3dx-trace`) and installs the `f3dx` Python package as editable. First build takes ~1 minute, incremental ~5 seconds.

## Running benches

Every bench under `bench/` is reproducible and uses stdlib mock servers - no live API keys required.

```bash
python bench/bench_concurrent.py            # AgentRuntime concurrent dispatch (5-10x)
python bench/bench_streaming.py             # f3dx.OpenAI vs openai SDK (5x)
python bench/bench_anthropic_streaming.py   # f3dx.Anthropic vs anthropic SDK (3-5x)
python bench/verify_assembled.py            # tool-call reassembly
python bench/verify_otel.py                 # AgentRuntime span emission
python bench/verify_otel_http.py            # HTTP-level spans on OpenAI + Anthropic
```

## Layout

```
crates/
  f3dx-py     PyO3 bridge cdylib (the only crate with #[pymodule])
  f3dx-rt     Agent runtime + concurrent tool dispatch
  f3dx-http   LLM HTTP client (reqwest + native SSE + streaming JSON validation)
  f3dx-trace  OpenTelemetry span emission (Logfire-compatible, gen_ai.* semconv)
python/f3dx/  Python wrapper (one import surface)
bench/        Reproducible benches + stdlib mock servers
```

## Pull requests

- Branch off `main`, push to your fork, open a PR
- Keep the diff focused: one architectural change per PR
- Bench numbers in the PR body if perf-sensitive
- Commits: lowercase, imperative, no Co-Authored-By
- The CI in `.github/workflows/ci.yml` runs cargo + maturin build + Python smoke; PR must be green

## Reporting issues

Bugs: open a GitHub issue with a minimal reproducer (Python script + Rust panic if any).

Security: do not file public issues. Email `smigolsmigol@protonmail.com` with the details. We follow coordinated disclosure.

## License

MIT, by contributing you agree your work is licensed the same way.
