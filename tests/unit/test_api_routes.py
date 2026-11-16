"""Unit tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_risk_assessment():
    payload = {
        "text": "SEC enforcement action under Rule 10b-5 imposing $5,000,000 fine for insider trading.",
        "domain": "financial",
    }
    response = client.post("/api/v1/analysis/risk-assessment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "risk_level" in data
    assert "recommendations" in data


def test_analysis_semantic_diff():
    from src.analysis.semantic_diff import SemanticDiff

    mock_diff = SemanticDiff(
        added_sections=[],
        removed_sections=[],
        modified_sections=[],
        overall_similarity=0.85,
    )
    with patch("src.api.routes.analysis.get_diff_engine") as mock_get:
        mock_get.return_value.compute_diff.return_value = mock_diff
        payload = {
            "base_text": "Brokers must retain records for 3 years.",
            "updated_text": "Brokers must retain records for 5 years and report annually.",
        }
        response = client.post("/api/v1/analysis/semantic-diff", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "overall_similarity" in data


def test_query_ask_mocked():
    with patch("src.api.routes.query.run_graph", new_callable=AsyncMock) as mock_graph:
        mock_graph.return_value = {
            "response": "Under SEC Rule 10b5-1, a 90-day cooling off period is required.",
            "citations": [
                {
                    "source": "SEC Filing",
                    "page": 1,
                    "chunk_id": "c1",
                    "score": 0.95,
                    "origin": "hybrid",
                }
            ],
            "error": None,
        }
        payload = {"question": "What is the cooling off period under Rule 10b5-1?"}
        response = client.post("/api/v1/query/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "cooling off period" in data["answer"]
        assert len(data["citations"]) == 1
