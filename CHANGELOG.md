# Changelog

All notable changes to f3dx are documented here. Format follows
[keep-a-changelog](https://keepachangelog.com/en/1.1.0/) 1.1.0;
project tracks [SemVer](https://semver.org/).

## [Unreleased]

## [0.0.19] - 2026-04-30

Single-day arc: pyo3 0.24 -> 0.28.3, repo consolidation phase A->C
(cache + replay + router merged into the workspace as Cargo members
+ Python sub-modules), and four `f3dx.fast` inference-acceleration
pillars shipped V0.

### Added

- `f3dx.fast.CanonicalPrompt` (Pillar 2): prefix-cache-aware prompt
  builder. Static prefix held bytes-identical so OpenAI prefix-cache
  fires from turn two onward. Measured against real gpt-4o-mini:
  91.1% cache hit, 45.6% input cost cut, TTFT 1445ms cold to
  461-586ms warm. `cache_hit_ratio()` helper handles OpenAI +
  Anthropic usage shapes.
- `f3dx.cache.cache_tool_call` (Pillar 3): tool-result memoization
  with `FileWitness`, `TTLWitness`, `EnvWitness`, `CompositeWitness`.
  Real bench: 50KB Read 9.4ms cold to 42us warm (223x), `gh run list`
  702ms cold to 6.3us warm (111,415x via subprocess elimination).
- `f3dx.fast.SpecToolDispatcher` (Pillar 4): speculative tool
  execution. `StreamingJSONAccumulator` parses tool args as the SSE
  stream arrives; safe tools fire before assistant text completes
  via a `ThreadPoolExecutor`. Synthetic 3-tool turn: 1601ms sync to
  1002ms threaded (37% wall-clock cut).
- `f3dx.fast.budget_max_tokens` (Pillar 6): `max_tokens` hint from
  trace history. Real bench on a runaway-prone prompt: 282 hint vs
  4096 default = 3,814 tokens of headroom saved per call (93% worst-
  case cost reduction), zero truncations on calibration set.
- `f3dx.cache.cached_call`: f3d1-wide convention for real-API benches
  to record once + replay forever via `bench/fixtures/<provider>.redb`.
  CI sets `F3DX_BENCH_OFFLINE=1` so cache miss = test failure.
- `f3dx.cache` and `f3dx.router` are now bundled in the f3dx wheel.
  Standalone `f3dx-cache` and `f3dx-router` PyPI packages stay as
  deprecation shims that re-export from `f3dx.cache` / `f3dx.router`.

### Changed

- pyo3 bumped 0.24 -> 0.28.3 across the workspace. Migrations:
  `with_gil` -> `attach`, `allow_threads` -> `detach`, `downcast_into`
  -> `cast_into`, `downcast` -> `cast`. `.cargo/config.toml` adds
  `RUST_MIN_STACK=33554432` to dodge windows-sys 0.61.2 codegen
  stack overflow.
- README rewritten for the consolidated stack (5 crates -> 7 with
  cache + replay + router merged in) and the f3d1-fast pillars.
  Voice-cleaned to match the Colvin/alexmojaki maintainer register.

### Fixed

- `cargo fmt --check` regression on `router.rs` `chat_completions`
  arm wrap.
- semgrep `dangerous-subprocess-use` flagged `shell=True` in
  `tool_cache_real_bench.py`. Switched to `shlex.split` + list args
  with explicit `noqa`.

### Security

- `.gitignore` adds `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`
  defense-in-depth (no current leak; gitleaks + GitHub native
  secret-scanning both green).

## [0.0.18] - 2026-04-28

### Added

- `f3dx.bench.auto_attach()` and `F3DX_BENCH_AUTO_ATTACH=1`: tail
  the JSONL trace sink and forward every `AgentRuntime.run` as a
  bench beacon. Provider inferred from model prefix. Closes the
  bench flywheel.

### Changed

- Org-level `.github` substrate adopted: SECURITY.md, CODE_OF_CONDUCT,
  CONTRIBUTING, ISSUE_TEMPLATE, PR template, FUNDING, CITATION all
  inherited from `smigolsmigol/.github`. Scorecard, security, and
  release-please workflows shrunk to 5-line shims calling reusable
  workflows in the org repo.

### Fixed

- Scorecard shim grants write perms at workflow level so consumer
  jobs elevate to security-events + id-token correctly.

## [0.0.17] - earlier

- Trusted Publisher + sigstore attestation pipeline established.
- Initial `f3dx.bench` opt-in telemetry SDK.
- Banned-chars sweep across the repo (em-dashes, smart quotes,
  arrows, ellipses replaced with ASCII).

## Earlier versions

Full git history at https://github.com/smigolsmigol/f3dx/commits/main.
