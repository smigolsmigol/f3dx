# Security Policy

## Reporting a vulnerability

Email `fed1@f3d1.dev` with:

- A minimal reproducer
- The version of `f3dx` affected
- The platform you reproduced on (OS, Python version)

Please do **not** file a public GitHub issue or PR for the report itself.

We aim to acknowledge reports within 48 hours, fix critical issues within 7 days, and disclose coordinated with the reporter.

## Scope

- The `f3dx` Python package and the four Rust crates (`f3dx-py`, `f3dx-rt`, `f3dx-http`, `f3dx-trace`)
- Build system (`Cargo.toml`, `pyproject.toml`, CI workflow)

Out of scope: third-party model APIs (OpenAI, Anthropic, etc.) we proxy to.

## Known classes of relevant issue

- HTTP-side: TLS misconfig, header leakage, request smuggling
- SSE-side: parser ambiguities, chunk-boundary attacks
- PyO3 boundary: GIL handling bugs, memory safety
- OTel-side: trace attribute injection, header secret leakage
