"""Request logging middleware with structured logging and timing metrics."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests and outgoing responses with latency and status code."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        # Attach request_id to context
        structlog.contextvars.bind_contextvars(request_id=request_id)

        logger.info(
            "http_request_start",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = str(duration_ms)

            logger.info(
                "http_request_complete",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "http_request_error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise exc
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
