# ==============================================================================
# OmniComply — Makefile
# ==============================================================================

.PHONY: dev test lint eval eval-ci docker-up docker-down migrate seed clean install format help

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------
install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

dev: ## Run development server with hot reload
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

celery-dev: ## Run Celery worker for development
	celery -A src.api.celery_app worker --loglevel=info --concurrency=2

streamlit: ## Run Streamlit dashboard
	streamlit run src/dashboard/app.py --server.port 8501

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint: ## Run linter (ruff)
	ruff check src/ tests/ eval/
	ruff format --check src/ tests/ eval/

format: ## Auto-format code
	ruff check --fix src/ tests/ eval/
	ruff format src/ tests/ eval/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run unit tests
	pytest tests/unit/ -v --tb=short -m "not integration and not eval and not slow"

test-integration: ## Run integration tests (requires services)
	pytest tests/integration/ -v --tb=short -m "integration"

test-all: ## Run all tests with coverage
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term-missing

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
eval: ## Run full evaluation suite
	python -m eval.pipelines.ci_eval_gate

eval-ci: ## Run evaluation with CI gate (fails if below thresholds)
	python -m eval.pipelines.ci_eval_gate --strict

eval-nightly: ## Run nightly regression evaluation
	python -m eval.pipelines.nightly_regression

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-build: ## Rebuild Docker images
	docker compose build --no-cache

docker-logs: ## Tail service logs
	docker compose logs -f

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create a new migration
	@read -p "Migration message: " msg; alembic revision --autogenerate -m "$$msg"

seed: ## Seed database with sample data
	python scripts/seed_data.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ .coverage

