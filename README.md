# OmniComply

**A modular, AI-assisted workspace for exploring regulatory compliance.**

OmniComply is an early-stage Python project that brings document ingestion, regulatory Q&A, risk analysis, and evaluation into one codebase. It is designed to process regulatory material, retrieve relevant passages, and return answers with citations that can be checked against source documents.

> **Project status: Alpha.** Some integrations and user-interface elements are still evolving; review the current implementation and test it with your own data before relying on it.

## What it includes

- **Multi-format ingestion:** processing paths for PDF, image, audio, and text files, with parsing or OCR/transcription, chunking, and legal-entity extraction.
- **Hybrid retrieval building blocks:** dense search with PostgreSQL/pgvector, sparse BM25 search, Neo4j graph lookup, reciprocal-rank fusion, reranking, and a Redis-backed semantic cache.
- **Compliance Q&A:** a LangGraph-based agent workflow and API endpoints for questions, retrieval strategies, and citation-bearing responses.
- **Regulatory analysis:** utilities and endpoints for risk assessment, semantic comparison of text versions, and forecasting from historical risk scores.
- **Evaluation harness:** a golden Q&A dataset, quality metrics, JSON reports, and CI/nightly evaluation workflows.
- **Developer interfaces:** a FastAPI service, an early Streamlit dashboard, Docker Compose infrastructure, and unit/integration tests.

## How it fits together

The codebase separates ingestion, retrieval, agents, analysis, API, and evaluation so these components can be developed and tested independently. The intended flow is:

1. Ingest a regulatory document or transcript.
2. Parse or transcribe it, split it into chunks, and extract entities.
3. Index content for vector, keyword, and graph-based retrieval.
4. Retrieve relevant context for a question and synthesize an answer with source citations.
5. Evaluate answer quality against the golden dataset and inspect the generated report.

The retrieval and storage components require their supporting services and initialized data. Not every path is wired end-to-end yet.

## Technology

| Area | Technologies |
| --- | --- |
| API and validation | FastAPI, Pydantic |
| Agent orchestration | LangGraph |
| Dense retrieval | PostgreSQL, pgvector, sentence-transformers |
| Sparse and graph retrieval | BM25, Neo4j |
| Cache and background work | Redis, Celery |
| Analysis and dashboard | Python, Streamlit, Plotly |
| Evaluation and tests | Custom RAG evaluators, pytest, Ruff |

## Requirements

- Python 3.11 or newer
- Docker Compose for the PostgreSQL, Redis, and Neo4j services
- An LLM provider and credentials if you want to use a hosted model
- Optional system dependencies for document processing, such as `ffmpeg` for audio

## Getting started

### 1. Get the code and create an environment

```shell
git clone https://github.com/ChopraArhaan/omnicomply.git
cd omnicomply
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```shell
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies and configure the environment

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Create a local environment file and edit it for your setup:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```shell
# macOS / Linux
cp .env.example .env
```

Set the model/provider credentials and service connection values you need in `.env`. See [`src/config/settings.py`](src/config/settings.py) for the application settings and [`.env.example`](.env.example) for the available configuration template. Keep real credentials out of source control.

### 3. Start the supporting services

```shell
docker compose up -d postgres redis neo4j
```

To run the API and worker in containers as well, start the full Compose stack instead:

```shell
docker compose up --build
```

### 4. Run the API

For a locally installed development environment:

```shell
python -m uvicorn src.api.main:app --reload
```

The API is served at `http://localhost:8000`. Interactive endpoint documentation is available at `http://localhost:8000/docs`; liveness and readiness probes are at `/health` and `/ready`.

### 5. Open the dashboard (optional)

In another terminal with the environment activated:

```shell
python -m streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. It is an early demonstration UI; some displayed flows and sample values are illustrative rather than connected to persisted production data.

## API overview

The current FastAPI application exposes these routes:

| Route | Purpose |
| --- | --- |
| `POST /api/v1/documents/upload` | Submit a document for ingestion |
| `GET /api/v1/documents/` | List uploaded documents |
| `GET /api/v1/documents/{document_id}` | Get document status and metadata |
| `DELETE /api/v1/documents/{document_id}` | Delete a document |
| `POST /api/v1/query/ask` | Ask a compliance question |
| `POST /api/v1/query/ask/stream` | Stream a query response |
| `GET /api/v1/query/history` | Query-history endpoint |
| `POST /api/v1/analysis/risk-assessment` | Score regulatory text for risk signals |
| `POST /api/v1/analysis/semantic-diff` | Compare two text versions |
| `POST /api/v1/analysis/forecast` | Forecast from historical risk scores |
| `GET /api/v1/analysis/risk-dashboard` | Retrieve a high-level risk summary |

For request schemas and response details, use the interactive API documentation at `/docs`.

## Evaluation and tests

Run the evaluation against the included golden dataset:

```shell
python -m eval.pipelines.ci_eval_gate
```

Run the strict CI gate or the nightly regression pipeline:

```shell
python -m eval.pipelines.ci_eval_gate --strict
python -m eval.pipelines.nightly_regression
```

Reports are written under `eval/reports/outputs/` (this generated directory is ignored by Git). The GitHub Actions nightly workflow also uploads its report as an artifact.

Run the fast unit suite:

```shell
python -m pytest tests/unit/ -v --tb=short -m "not integration and not eval and not slow"
```

Integration tests may require the supporting services. The repository also provides corresponding `make` targets; see the [Makefile](Makefile).

## Repository layout

```text
eval/       Golden dataset, metrics, evaluators, pipelines, and reports
scripts/    Database initialization and sample-data utilities
src/
  agents/   LangGraph agents and workflow state
  analysis/ Classification, risk scoring, semantic diff, and forecasting
  api/      FastAPI application, middleware, and routes
  config/   Settings and prompt templates
  ingestion/Parsing, OCR, transcription, chunking, and entity extraction
  retrieval/Embeddings, vector store, graph store, caching, and reranking
tests/      Unit and integration tests
```

## Important limitations

OmniComply is a software development project, not a legal service or a source of authoritative legal advice. Model-generated answers and heuristic scores can be incomplete or incorrect. Always verify conclusions against current primary sources and qualified counsel.

The repository is marked Alpha. Some components use in-memory state, illustrative dashboard content, or incomplete service wiring. Validate the specific ingestion, persistence, retrieval, and model-provider paths you plan to use before deploying or making compliance decisions.

## Contributing

Contributions are welcome. Open an issue to discuss a bug or proposed change, and include relevant tests for code changes. Run the unit suite and formatting/lint checks before submitting a pull request:

```shell
python -m pytest tests/unit/ -v --tb=short -m "not integration and not eval and not slow"
ruff check src/ tests/ eval/
ruff format --check src/ tests/ eval/
```

## License

OmniComply is distributed under the [MIT License](LICENSE).
