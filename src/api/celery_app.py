"""Celery task worker configuration for asynchronous ingestion and processing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
from celery import Celery

from src.config.settings import Settings

logger = structlog.get_logger(__name__)
settings = Settings()

celery_app = Celery(
    "omnicomply_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 min max for large PDFs
)


@celery_app.task(bind=True, name="tasks.ingest_document")
def ingest_document_task(
    self, file_path_str: str, document_id: str | None = None
) -> dict[str, Any]:
    """Execute full ingestion pipeline in background Celery worker."""
    from src.ingestion.pipeline import IngestionPipeline

    logger.info("celery_task_ingest_started", file=file_path_str, task_id=self.request.id)
    pipeline = IngestionPipeline()
    path = Path(file_path_str)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        processed = loop.run_until_complete(pipeline.ingest(path))
        return {
            "status": "COMPLETED",
            "document_id": processed.document_id,
            "chunks_count": len(processed.chunks),
            "entities_count": len(processed.entities),
            "processed_at": processed.processed_at,
        }
    except Exception as exc:
        logger.error("celery_task_failed", error=str(exc))
        return {
            "status": "FAILED",
            "error": str(exc),
        }
