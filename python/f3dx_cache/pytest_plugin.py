"""pytest-f3dx-cache: deterministic LLM tests via @pytest.mark.f3dx_cache.

Marked tests get a Cache instance on a temp path. The first run records;
subsequent runs replay from cache (zero token cost, <100us per call).

Usage:

    import pytest
    from f3dx_cache import Cache

    @pytest.mark.f3dx_cache
    def test_my_agent(f3dx_cache_obj: Cache):
        request = {"model": "gpt-4o", "messages": [...]}
        cached = f3dx_cache_obj.get(request)
        if cached is None:
            response = call_real_llm(request)
            f3dx_cache_obj.put(request, response.body, model=response.model)
            cached = response.body
        assert b"expected substring" in cached

The simple form is enough for V0. V0.1 ships a context manager that
intercepts f3dx.OpenAI / f3dx.Anthropic transparently, so the test body
doesn't see the cache plumbing at all.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from f3dx_cache import Cache


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "f3dx_cache: enable f3dx-cache for this test (records on miss, replays on hit)",
    )


@pytest.fixture
def f3dx_cache_obj(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Cache]:
    marker = request.node.get_closest_marker("f3dx_cache")
    if marker is None:
        pytest.skip("f3dx_cache fixture requires @pytest.mark.f3dx_cache")
    cache_dir = os.environ.get("F3DX_CACHE_DIR", str(tmp_path))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = Cache(str(Path(cache_dir) / "cache.redb"))
    yield cache
