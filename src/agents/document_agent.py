"""Document Agent — Handles multimodal document ingestion, parsing, and vectorization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.ingestion.pipeline import IngestionPipeline, ProcessedDocument
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.knowledge_graph import Entity as GraphEntity
from src.retrieval.knowledge_graph import KnowledgeGraphStore
from src.retrieval.vector_store import DocumentChunk as VectorChunk
from src.retrieval.vector_store import PgVectorStore

if TYPE_CHECKING:
    from src.agents.state import AgentState

logger = structlog.get_logger(__name__)


class DocumentAgent:
    """Orchestrates parsing, chunking, NER extraction, and vectorization of documents."""

    def __init__(
        self,
        pipeline: IngestionPipeline | None = None,
        embeddings: EmbeddingService | None = None,
        vector_store: PgVectorStore | None = None,
        kg_store: KnowledgeGraphStore | None = None,
    ) -> None:
        self.pipeline = pipeline or IngestionPipeline()
        self.embeddings = embeddings or EmbeddingService()
        self.vector_store = vector_store or PgVectorStore()
        self.kg_store = kg_store

    async def process(self, state: AgentState) -> AgentState:
        """Process document from file path in state."""
        file_path_str = state.get("file_path")
        if not file_path_str:
            state["error"] = "No file_path provided to DocumentAgent"
            return state

        path = Path(file_path_str)
        if not path.exists():
            state["error"] = f"File not found: {file_path_str}"
            return state

        logger.info("document_agent_processing_file", path=str(path))
        try:
            # 1. Run ingestion pipeline
            processed_doc: ProcessedDocument = await self.pipeline.ingest(path)

            # 2. Extract chunks and embeddings
            chunk_texts = [c.content for c in processed_doc.chunks]
            if chunk_texts:
                embeddings = self.embeddings.embed_texts(chunk_texts)

                # Convert to vector store format
                vector_chunks = [
                    VectorChunk(
                        chunk_id=c.chunk_id,
                        document_id=processed_doc.document_id,
                        content=c.content,
                        metadata=c.metadata,
                        token_count=c.token_count,
                    )
                    for c in processed_doc.chunks
                ]

                # Store in PostgreSQL + pgvector
                await self.vector_store.insert_chunks(vector_chunks, embeddings)

            # 3. Store entities in Knowledge Graph (if available)
            if self.kg_store and processed_doc.entities:
                graph_entities = [
                    GraphEntity(
                        text=e.text,
                        label=e.label,
                        confidence=e.confidence,
                        metadata={"document_id": processed_doc.document_id},
                    )
                    for e in processed_doc.entities
                ]
                await self.kg_store.add_entities(processed_doc.document_id, graph_entities)

            # 4. Update state
            state["parsed_document"] = {
                "document_id": processed_doc.document_id,
                "source_type": processed_doc.source_type,
                "metadata": processed_doc.metadata,
                "processed_at": processed_doc.processed_at,
            }
            state["chunks"] = [c.model_dump() for c in processed_doc.chunks]
            state["entities"] = [e.model_dump() for e in processed_doc.entities]
            state["next_step"] = "compliance_agent"

            logger.info(
                "document_agent_success",
                doc_id=processed_doc.document_id,
                chunks=len(processed_doc.chunks),
                entities=len(processed_doc.entities),
            )
        except Exception as exc:
            logger.error("document_agent_failed", error=str(exc))
            state["error"] = f"Document processing failed: {exc}"

        return state
