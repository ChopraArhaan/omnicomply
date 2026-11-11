-- ==========================================================================
-- OmniComply — Database Initialisation Script
-- ==========================================================================
-- Run against a fresh PostgreSQL database to set up the core schema.
--
--   psql -U omnicomply -d omnicomply -f scripts/init_db.sql
-- ==========================================================================

-- Enable pgvector for embedding storage & similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable uuid generation (available by default in PG 13+, explicit for safety)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Documents ───────────────────────────────────────────────────────────────
-- Top-level record for every ingested document.

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(50)  NOT NULL,          -- pdf, image, audio, text
    title       VARCHAR(512) NOT NULL DEFAULT '',
    metadata    JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE documents IS 'Registry of all ingested documents.';

-- ── Chunks ──────────────────────────────────────────────────────────────────
-- Token-bounded slices of documents, each with a 1024-dim embedding.

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT         NOT NULL,
    embedding   vector(1024),                   -- populated after embedding pass
    metadata    JSONB        NOT NULL DEFAULT '{}'::jsonb,
    token_count INT          NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE chunks IS 'Document chunks with vector embeddings for retrieval.';

-- IVFFlat index for approximate nearest-neighbour cosine search.
-- Tune `lists` upward as data grows (rule of thumb: sqrt(row_count)).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- B-tree index for foreign-key look-ups
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- ── Entities ────────────────────────────────────────────────────────────────
-- Named entities extracted via NER, linked to the chunk they appear in.

CREATE TABLE IF NOT EXISTS entities (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id     UUID         NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_text  VARCHAR(512) NOT NULL,
    entity_label VARCHAR(50)  NOT NULL,          -- REGULATION_ID, ORGANIZATION, …
    confidence   FLOAT        NOT NULL DEFAULT 0.0,
    metadata     JSONB        NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE entities IS 'Named entities extracted from document chunks.';

CREATE INDEX IF NOT EXISTS idx_entities_chunk_id
    ON entities (chunk_id);

CREATE INDEX IF NOT EXISTS idx_entities_label
    ON entities (entity_label);
