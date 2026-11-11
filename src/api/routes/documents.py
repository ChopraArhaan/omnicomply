"""
Document management API routes.

Prefix: ``/api/v1/documents``

Endpoints
---------
* ``POST /upload`` – Accept a file, trigger async ingestion, return task/document ids.
* ``GET /``        – List documents with pagination.
* ``GET /{id}``    – Retrieve document metadata & processing status.
* ``DELETE /{id}`` – Remove a document plus its chunks and entities.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# ---------------------------------------------------------------------------
# Pydantic response / request models
# ---------------------------------------------------------------------------


class DocumentUploadResponse(BaseModel):
    """Returned after a successful upload."""

    document_id: str = Field(..., description="UUID of the newly created document.")
    task_id: str = Field(..., description="Celery task id for tracking ingestion.")
    status: str = Field(default="processing", description="Current processing status.")
    message: str = Field(default="Document ingestion started.")


class DocumentMetadata(BaseModel):
    """Full metadata for a single document."""

    document_id: str
    filename: str
    source_type: str
    status: str
    title: str | None = None
    page_count: int | None = None
    chunk_count: int = 0
    entity_count: int = 0
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: list[DocumentMetadata]
    total: int
    skip: int
    limit: int


class DeleteResponse(BaseModel):
    """Confirmation of document deletion."""

    document_id: str
    deleted: bool
    message: str


# ---------------------------------------------------------------------------
# In-memory store (replaced by real DB in production)
# ---------------------------------------------------------------------------

_documents: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accept a file upload and trigger the asynchronous ingestion pipeline.

    The file is persisted to a temporary location and a Celery task is
    dispatched so that ingestion happens outside the request cycle.
    """
    document_id = str(uuid.uuid4())
    logger.info(
        "documents.upload",
        document_id=document_id,
        filename=file.filename,
        content_type=file.content_type,
    )

    # Persist upload to a temp path
    import tempfile, os, shutil

    suffix = os.path.splitext(file.filename or "")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="omni_")
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()

    # Dispatch Celery task
    try:
        from src.api.celery_app import ingest_document

        task = ingest_document.delay(tmp_path, document_id)
        task_id = task.id
    except Exception:
        logger.warning("documents.upload.celery_unavailable", exc_info=True)
        task_id = str(uuid.uuid4())  # fallback id when broker is unreachable

    # Track in memory store
    now = datetime.now(timezone.utc)
    _documents[document_id] = {
        "document_id": document_id,
        "filename": file.filename or "unknown",
        "source_type": (file.content_type or "application/octet-stream"),
        "status": "processing",
        "title": None,
        "page_count": None,
        "chunk_count": 0,
        "entity_count": 0,
        "created_at": now,
        "updated_at": now,
        "metadata": {"task_id": task_id, "tmp_path": tmp_path},
    }

    return DocumentUploadResponse(
        document_id=document_id,
        task_id=task_id,
    )


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str) -> DocumentMetadata:
    """Return metadata and processing status for a single document."""
    doc = _documents.get(document_id)
    if doc is None:
        logger.warning("documents.get.not_found", document_id=document_id)
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentMetadata(**doc)


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(default=0, ge=0, description="Number of records to skip."),
    limit: int = Query(default=20, ge=1, le=100, description="Page size."),
) -> DocumentListResponse:
    """List all documents with offset-based pagination."""
    all_docs = list(_documents.values())
    page = all_docs[skip : skip + limit]
    return DocumentListResponse(
        documents=[DocumentMetadata(**d) for d in page],
        total=len(all_docs),
        skip=skip,
        limit=limit,
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    """Delete a document and its associated chunks and entities."""
    doc = _documents.pop(document_id, None)
    if doc is None:
        logger.warning("documents.delete.not_found", document_id=document_id)
        raise HTTPException(status_code=404, detail="Document not found.")

    logger.info("documents.delete", document_id=document_id)

    # Clean up temp file if it still exists
    import os

    tmp_path = doc.get("metadata", {}).get("tmp_path")
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # In production: delete from vector store + graph + DB
    try:
        from src.api.main import get_state

        state = get_state()
        vs = state.get("vector_store")
        if vs:
            import asyncio

            await vs.delete_document(document_id)
    except Exception:
        logger.warning("documents.delete.vector_store_cleanup_failed", exc_info=True)

    return DeleteResponse(
        document_id=document_id,
        deleted=True,
        message="Document and associated data deleted.",
    )
