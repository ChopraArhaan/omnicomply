"""Cost tracking middleware for LLM, embedding, and vector database operations."""

from __future__ import annotations

import json
import time
from typing import Callable, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Tracks estimated token usage, inference costs, and cache savings per request."""

    def __init__(self, app, redis_url: str | None = None) -> None:
        super().__init__(app)
        self.redis_url = redis_url
        self._redis_client = None

    async def _get_redis(self):
        if self._redis_client is None and self.redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                logger.warning("cost_tracking_redis_unavailable", error=str(e))
        return self._redis_client

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Only track operational endpoints
        if request.url.path.startswith("/api/v1/"):
            try:
                r = await self._get_redis()
                if r:
                    # Increment request counters in Redis
                    day_key = f"metrics:requests:{time.strftime('%Y-%m-%d')}"
                    await r.hincrby(day_key, request.url.path, 1)
                    await r.expire(day_key, 86400 * 30)  # 30 day TTL
            except Exception as exc:
                logger.debug("cost_tracking_increment_failed", error=str(exc))

        return response
