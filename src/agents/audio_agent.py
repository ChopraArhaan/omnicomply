"""Audio Agent — Transcribes and processes regulatory hearings, earnings calls, and panel recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from src.agents.state import AgentState
from src.ingestion.audio_processor import AudioProcessor, Transcript
from src.ingestion.chunker import AdaptiveChunker
from src.ingestion.ner import LegalNERExtractor
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import DocumentChunk as VectorChunk, PgVectorStore

logger = structlog.get_logger(__name__)


class AudioAgent:
    """Orchestrates Whisper transcription, segmentation, entity recognition, and indexing."""

    def __init__(
        self,
        audio_processor: AudioProcessor | None = None,
        chunker: AdaptiveChunker | None = None,
        ner_extractor: LegalNERExtractor | None = None,
        embeddings: EmbeddingService | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        self.audio_processor = audio_processor or AudioProcessor()
        self.chunker = chunker or AdaptiveChunker()
        self.ner_extractor = ner_extractor or LegalNERExtractor()
        self.embeddings = embeddings or EmbeddingService()
        self.vector_store = vector_store or PgVectorStore()

    async def process(self, state: AgentState) -> AgentState:
        """Process audio file path in state."""
        file_path_str = state.get("file_path")
        if not file_path_str:
            state["error"] = "No file_path provided to AudioAgent"
            return state

        path = Path(file_path_str)
        if not path.exists():
            state["error"] = f"Audio file not found: {file_path_str}"
            return state

        logger.info("audio_agent_processing_file", path=str(path))
        try:
            # 1. Transcribe audio with Whisper
            transcript: Transcript = await self.audio_processor.transcribe(path)

            # 2. Extract entities from full transcript text
            entities = self.ner_extractor.extract_entities(transcript.text)

            # 3. Chunk transcript with timestamps preserved in metadata
            from src.ingestion.pdf_parser import PageContent, ParsedDocument
            fake_doc = ParsedDocument(
                content=transcript.text,
                pages=[PageContent(page_num=1, text=transcript.text)],
                metadata={
                    "source": str(path.name),
                    "duration_seconds": transcript.duration_seconds,
                    "language": transcript.language,
                },
            )
            chunks = self.chunker.chunk(fake_doc)

            # 4. Embed and store
            chunk_texts = [c.content for c in chunks]
            if chunk_texts:
                embeddings = self.embeddings.embed_texts(chunk_texts)
                vector_chunks = [
                    VectorChunk(
                        chunk_id=c.chunk_id,
                        document_id=f"audio_{path.stem}",
                        content=c.content,
                        metadata={**c.metadata, "language": transcript.language},
                        token_count=c.token_count,
                    )
                    for c in chunks
                ]
                await self.vector_store.insert_chunks(vector_chunks, embeddings)

            # 5. Update state
            state["parsed_document"] = {
                "document_id": f"audio_{path.stem}",
                "source_type": "audio",
                "metadata": {
                    "duration_seconds": transcript.duration_seconds,
                    "language": transcript.language,
                    "segments_count": len(transcript.segments),
                },
            }
            state["chunks"] = [c.model_dump() for c in chunks]
            state["entities"] = [e.model_dump() for e in entities]
            state["next_step"] = "compliance_agent"

            logger.info(
                "audio_agent_success",
                duration=transcript.duration_seconds,
                chunks=len(chunks),
                entities=len(entities),
            )
        except Exception as exc:
            logger.error("audio_agent_failed", error=str(exc))
            state["error"] = f"Audio processing failed: {exc}"

        return state
