# ==============================================================================
# OmniComply Dockerfile — Multi-stage build for FastAPI application
# ==============================================================================
FROM python:3.11-slim AS base

# System dependencies for audio processing and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage: Dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage: Production
# ---------------------------------------------------------------------------
FROM deps AS production

COPY src/ ./src/
COPY eval/ ./eval/
COPY pyproject.toml .

# Create non-root user
RUN groupadd -r omnicomply && useradd -r -g omnicomply omnicomply
USER omnicomply

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

