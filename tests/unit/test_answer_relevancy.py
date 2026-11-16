"""Unit tests for AnswerRelevancyMetric and calibrated QA scoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from eval.metrics.answer_relevancy import (
    AnswerRelevancyMetric,
    _get_sentence_transformer,
    calibrate_relevancy_score,
)


def test_known_good_qa_pair_scores_above_threshold() -> None:
    """Test (i): A known-good regulatory QA pair scores >= threshold (0.85)."""
    metric = AnswerRelevancyMetric(threshold=0.85)

    question = (
        "What is the mandatory disclosure timeline under SEC Form 8-K Item 1.05 "
        "for a material cybersecurity incident?"
    )
    answer = (
        "Under SEC Form 8-K Item 1.05, registrants must disclose a material cybersecurity incident "
        "within four business days after determining that the incident is material."
    )

    result = metric.score(question=question, answer=answer)

    assert result.passed is True
    assert result.score >= 0.85
    assert result.threshold == 0.85
    assert "answer_relevancy" == result.name


def test_irrelevant_pair_scores_well_below_threshold() -> None:
    """Test (ii): An irrelevant/off-topic pair scores well below the 0.85 threshold."""
    metric = AnswerRelevancyMetric(threshold=0.85)

    question = (
        "What are the administrative fines for infringements of basic processing principles "
        "under GDPR Article 83(5)?"
    )
    answer = (
        "We recommend preheating your air fryer to 375 degrees Fahrenheit and "
        "cooking the potatoes for fifteen minutes."
    )

    result = metric.score(question=question, answer=answer)

    assert result.passed is False
    assert result.score < 0.60
    assert result.score < metric.threshold


def test_model_loader_called_only_once_across_many_invocations() -> None:
    """Test (iii): The model loader _get_sentence_transformer is called only once across multiple invocations."""
    _get_sentence_transformer.cache_clear()

    with patch("sentence_transformers.SentenceTransformer") as mock_st_cls:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance

        # Call loader directly multiple times
        model_1 = _get_sentence_transformer("test-model")
        model_2 = _get_sentence_transformer("test-model")
        model_3 = _get_sentence_transformer("test-model")

        assert model_1 is mock_instance
        assert model_2 is mock_instance
        assert model_3 is mock_instance
        # The constructor must only have been invoked exactly once
        assert mock_st_cls.call_count == 1

        # Check LRU cache metrics confirm multiple cache hits
        info = _get_sentence_transformer.cache_info()
        assert info.hits == 2
        assert info.misses == 1

    # Restore real cache
    _get_sentence_transformer.cache_clear()


def test_empty_and_evasive_answers() -> None:
    """Edge cases: empty answers and evasive responses must fail with proper penalties."""
    metric = AnswerRelevancyMetric(threshold=0.85)

    # Empty answer
    res_empty = metric.score("What is GDPR?", "   ")
    assert res_empty.score == 0.0
    assert res_empty.passed is False

    # Short evasive answer
    res_evasive = metric.score("What is GDPR?", "I don't know the answer to this question.")
    assert res_evasive.score == 0.20
    assert res_evasive.passed is False


def test_calibration_function_piecewise_properties() -> None:
    """Verify piecewise mathematical properties of calibrate_relevancy_score."""
    # Negative cosine similarity clamped to 0
    assert calibrate_relevancy_score(-0.1) == 0.0
    assert calibrate_relevancy_score(0.0) == 0.0

    # Empirical noise floor (< 0.20) maps below 0.45
    assert calibrate_relevancy_score(0.10) == 0.225
    assert calibrate_relevancy_score(0.19) < 0.45

    # Transition zone [0.20, 0.35]
    assert calibrate_relevancy_score(0.20) == 0.45
    assert calibrate_relevancy_score(0.35) == 0.85

    # Valid QA zone [0.35, 1.0] >= 0.85
    assert calibrate_relevancy_score(0.50) > 0.85
    assert calibrate_relevancy_score(0.75) == 1.0
    assert calibrate_relevancy_score(0.95) == 1.0
