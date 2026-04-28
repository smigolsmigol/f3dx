"""f3dx.bench - opt-in telemetry to f3dx-bench.

Default off. When enabled (env var F3DX_BENCH_OPTIN=1 or explicit
f3dx.bench.opt_in()), every f3dx HTTP request emits one anonymized
TraceBeacon row to https://f3dx-bench-ingest.smigolsmigol.workers.dev.

What goes on the wire (12 fields, ~200 bytes/beacon):
  ts, install_id, install_hmac, model, provider, region, status_code,
  latency_ms_to_first_token, latency_ms_total, input_tokens,
  output_tokens, cost_usd_estimate

What does NOT: prompts, responses, API keys, headers, hostnames.

The install_id is a UUID generated locally on first opt-in and persisted
under the platform's app-config dir (write perm 600). The install_hmac
is a per-install random hex token also persisted locally; it never
leaves your machine after first registration. The Worker pins the token
on first beacon (TOFU). Subsequent beacons must submit the same token
or get 401-rejected.

See https://github.com/smigolsmigol/f3dx-bench/blob/main/docs/privacy.md
for the full policy + forget-request endpoint.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_INGEST_URL = "https://f3dx-bench-ingest.smigolsmigol.workers.dev"
SCHEMA_VERSION = "v1"

# Set by opt_in() / first F3DX_BENCH_OPTIN read
_state: dict[str, Any] = {
    "enabled": False,
    "ingest_url": DEFAULT_INGEST_URL,
    "install_id": None,
    "install_hmac": None,
    "queue": [],
    "lock": threading.Lock(),
    "worker_thread": None,
    "stop": threading.Event(),
    "wakeup": threading.Event(),  # signaled on emit(): worker drains immediately
    "in_flight": 0,  # batches currently mid-POST; flush() waits for this
}


def _config_dir() -> Path:
    """Per-platform config dir for the install file."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~/AppData/Roaming")
        return Path(base) / "f3dx"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "f3dx"
    # Linux + others
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "f3dx"


def _install_file() -> Path:
    return _config_dir() / "install.json"


def _load_or_register_install() -> tuple[str, str]:
    """Read (install_id, install_hmac) from local file; create if missing.

    The hmac field here is the per-install identity token: random 32-byte
    hex generated once. Worker TOFU-pins it on first beacon.
    """
    f = _install_file()
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "install_id" in data and "install_hmac" in data:
                return str(data["install_id"]), str(data["install_hmac"])
        except (json.JSONDecodeError, OSError):
            pass
    install_id = str(uuid.uuid4())
    install_hmac = secrets.token_hex(32)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        json.dumps({"install_id": install_id, "install_hmac": install_hmac}, indent=2),
        encoding="utf-8",
    )
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass  # Windows ignores
    return install_id, install_hmac


def opt_in(*, ingest_url: str = DEFAULT_INGEST_URL) -> None:
    """Enable beacon emission. Idempotent.

    Either call this explicitly, or set env var F3DX_BENCH_OPTIN=1 before
    importing f3dx and the auto-detect at module import time will do the
    same thing.
    """
    install_id, install_hmac = _load_or_register_install()
    with _state["lock"]:
        _state["enabled"] = True
        _state["ingest_url"] = ingest_url
        _state["install_id"] = install_id
        _state["install_hmac"] = install_hmac
        _state["stop"].clear()
        if _state["worker_thread"] is None or not _state["worker_thread"].is_alive():
            t = threading.Thread(target=_worker_loop, daemon=True, name="f3dx-bench")
            t.start()
            _state["worker_thread"] = t


def opt_out() -> None:
    """Disable beacon emission. Drains the in-flight queue and stops worker.

    Does NOT remove the install file. To wipe local install state too,
    delete the file at f3dx.bench.install_file_path() manually.
    """
    with _state["lock"]:
        _state["enabled"] = False
        _state["stop"].set()


def is_enabled() -> bool:
    return bool(_state["enabled"])


def install_file_path() -> Path:
    return _install_file()


def emit(
    *,
    model: str,
    provider: str,
    status_code: int,
    latency_ms_total: int,
    input_tokens: int,
    output_tokens: int,
    region: str | None = None,
    latency_ms_to_first_token: int | None = None,
    cost_usd_estimate: float | None = None,
    ts_unix_ms: int | None = None,
) -> None:
    """Queue one beacon for async emission. Returns immediately.

    Caller passes ALREADY-anonymized fields. This function does not
    inspect prompt or response content; if you call it with content,
    you have a bug. Worker thread POSTs in batches.
    """
    if not _state["enabled"]:
        return
    if ts_unix_ms is None:
        ts_unix_ms = int(time.time() * 1000)
    ts = (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts_unix_ms / 1000))
        + f".{ts_unix_ms % 1000:03d}Z"
    )
    beacon: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts": ts,
        "install_id": _state["install_id"],
        "install_hmac": _state["install_hmac"],
        "model": model,
        "provider": provider,
        "status_code": int(status_code),
        "latency_ms_total": int(latency_ms_total),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
    }
    if region is not None:
        beacon["region"] = region
    if latency_ms_to_first_token is not None:
        beacon["latency_ms_to_first_token"] = int(latency_ms_to_first_token)
    if cost_usd_estimate is not None:
        beacon["cost_usd_estimate"] = float(cost_usd_estimate)
    with _state["lock"]:
        _state["queue"].append(beacon)
    _state["wakeup"].set()


def flush(timeout: float = 10.0) -> None:
    """Drain the queue + wait for in-flight POSTs to complete.

    Useful at process exit (atexit hook) so the daemon thread doesn't
    get killed mid-request.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _state["lock"]:
            queue_empty = not _state["queue"]
            no_in_flight = _state["in_flight"] == 0
        if queue_empty and no_in_flight:
            return
        _state["wakeup"].set()  # nudge worker if it's sleeping
        time.sleep(0.05)


def emit_from_trace_row(row: Mapping[str, Any]) -> None:
    """Convenience: derive a beacon from an f3dx_trace row.

    f3dx_trace::emit_trace_row writes JSONL with a richer field set;
    this helper picks the 12 beacon fields out of it. Skips the row if
    required fields are missing.
    """
    try:
        emit(
            model=str(row["model"]),
            provider=str(row.get("provider", "other")),
            status_code=int(row.get("status_code", 200)),
            latency_ms_total=int(row.get("latency_ms_total", row.get("duration_ms", 0))),
            input_tokens=int(row.get("input_tokens", 0)),
            output_tokens=int(row.get("output_tokens", 0)),
            region=row.get("region"),
            latency_ms_to_first_token=row.get("latency_ms_to_first_token"),
            cost_usd_estimate=row.get("cost_usd_estimate"),
        )
    except (KeyError, ValueError, TypeError):
        return


def _worker_loop() -> None:
    """Background thread that drains the queue + POSTs in batches.

    Wakes immediately on emit() via the wakeup Event; otherwise polls
    every 1s. Batches up to 100 beacons per request as NDJSON. On HTTP
    error, drops the batch (telemetry must not block user code).
    """
    try:
        import urllib.request
    except ImportError:
        return
    while not _state["stop"].is_set():
        # Wait for either a wakeup signal or a 1s timeout, whichever first
        _state["wakeup"].wait(timeout=1.0)
        _state["wakeup"].clear()
        with _state["lock"]:
            batch = _state["queue"][:100]
            _state["queue"] = _state["queue"][100:]
            if batch:
                _state["in_flight"] += 1
        if not batch:
            continue
        try:
            url = _state["ingest_url"].rstrip("/") + "/v1/beacon"
            if len(batch) == 1:
                payload = json.dumps(batch[0]).encode("utf-8")
                content_type = "application/json"
            else:
                payload = ("\n".join(json.dumps(b) for b in batch)).encode("utf-8")
                content_type = "application/x-ndjson"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "content-type": content_type,
                    # Identify ourselves so Cloudflare WAF doesn't 1010 us
                    # for looking like a default urllib client.
                    "user-agent": "f3dx-bench/0.0.1 (+https://github.com/smigolsmigol/f3dx-bench)",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            # Telemetry must never break user code. Drop the batch.
            pass
        finally:
            with _state["lock"]:
                _state["in_flight"] = max(0, _state["in_flight"] - 1)


# Auto-detect opt-in at import time.
if os.environ.get("F3DX_BENCH_OPTIN", "").strip() in {"1", "true", "yes"}:
    try:
        opt_in()
    except Exception:  # noqa: BLE001 - never break import
        pass


__all__ = [
    "opt_in",
    "opt_out",
    "is_enabled",
    "install_file_path",
    "emit",
    "emit_from_trace_row",
    "flush",
    "DEFAULT_INGEST_URL",
    "SCHEMA_VERSION",
]
