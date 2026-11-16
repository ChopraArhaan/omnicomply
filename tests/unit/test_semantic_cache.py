"""Unit tests for SemanticCache."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.semantic_cache import CacheStats, SemanticCache


@pytest.fixture
def mock_cache():
    cache = SemanticCache(redis_url="redis://localhost:6379/0", similarity_threshold=0.90)
    mock_redis = AsyncMock()
    mock_redis.hgetall.return_value = {
        b"total_lookups": b"10",
        b"total_hits": b"8",
        b"total_latency_ms": b"15.5",
    }

    async def mock_scan_iter(match=None):
        for key in [b"entry1", b"entry2"]:
            yield key

    mock_redis.scan_iter = mock_scan_iter
    cache._redis = mock_redis
    return cache


@pytest.mark.asyncio
async def test_semantic_cache_initial_stats(mock_cache: SemanticCache):
    stats = await mock_cache.get_stats()
    assert isinstance(stats, CacheStats)
    assert stats.total_entries >= 0
    assert 0.0 <= stats.hit_rate <= 1.0


@pytest.mark.asyncio
async def test_semantic_cache_get_miss(mock_cache: SemanticCache):
    mock_ft = AsyncMock()
    mock_ft.search.return_value = MagicMock(total=0, docs=[])
    mock_cache._redis.ft.return_value = mock_ft

    dummy_embedding = [0.1] * 1024
    result = await mock_cache.get("test query", query_embedding=dummy_embedding)
    assert result is None
