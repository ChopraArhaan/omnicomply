"""
Regulatory Document Classifier.

Provides zero-shot classification of regulatory documents into regulatory
domains, urgency levels, and document types using a pre-trained NLI model
(facebook/bart-large-mnli) via the Hugging Face transformers pipeline.
"""

from __future__ import annotations

import enum
from typing import Any

import structlog
from pydantic import BaseModel, Field
from transformers import pipeline  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "facebook/bart-large-mnli"

REGULATORY_DOMAINS: list[str] = [
    "financial",
    "healthcare",
    "environmental",
    "data_privacy",
    "labor",
    "trade",
    "technology",
]

DOCUMENT_TYPES: list[str] = [
    "rule",
    "guidance",
    "enforcement_action",
    "amendment",
    "notice",
]


# ---------------------------------------------------------------------------
# Enums & Pydantic models
# ---------------------------------------------------------------------------


class UrgencyLevel(enum.StrEnum):
    """Urgency classification for a regulatory document."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ClassificationResult(BaseModel):
    """Result of classifying a document into a regulatory domain."""

    domain: str = Field(..., description="Top predicted regulatory domain.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the top prediction.",
    )
    all_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of each candidate domain to its score.",
    )


# ---------------------------------------------------------------------------
# Urgency keyword heuristics
# ---------------------------------------------------------------------------

_URGENCY_KEYWORDS: dict[UrgencyLevel, list[str]] = {
    UrgencyLevel.CRITICAL: [
        "immediate effect",
        "emergency",
        "cease and desist",
        "imminent threat",
        "effective immediately",
        "criminal penalty",
        "injunction",
    ],
    UrgencyLevel.HIGH: [
        "within 30 days",
        "compliance deadline",
        "enforcement action",
        "significant penalty",
        "mandatory recall",
        "suspension",
        "revocation",
    ],
    UrgencyLevel.MEDIUM: [
        "within 90 days",
        "proposed rule",
        "comment period",
        "amendment",
        "review required",
        "guidance update",
    ],
    UrgencyLevel.LOW: [
        "informational",
        "advisory",
        "request for comment",
        "notice of inquiry",
        "annual report",
        "voluntary",
    ],
}

_URGENCY_LABELS: list[str] = [
    "This document requires critical immediate action",
    "This document has high urgency and tight deadlines",
    "This document has medium urgency with upcoming deadlines",
    "This document is low urgency informational material",
]

_URGENCY_LABEL_MAP: dict[str, UrgencyLevel] = {
    _URGENCY_LABELS[0]: UrgencyLevel.CRITICAL,
    _URGENCY_LABELS[1]: UrgencyLevel.HIGH,
    _URGENCY_LABELS[2]: UrgencyLevel.MEDIUM,
    _URGENCY_LABELS[3]: UrgencyLevel.LOW,
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class RegulatoryClassifier:
    """Zero-shot regulatory document classifier.

    Uses the Hugging Face ``zero-shot-classification`` pipeline backed by
    ``facebook/bart-large-mnli`` to classify documents into regulatory
    domains, urgency levels, and document types.

    Args:
        model_name: Hugging Face model identifier. Defaults to
            ``facebook/bart-large-mnli``.
        device: Device ordinal (``-1`` for CPU, ``0`` for first GPU).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int = -1,
        classifier_pipeline: Any = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._classifier_instance = classifier_pipeline

    @property
    def _classifier(self) -> Any:
        if self._classifier_instance is None:
            logger.info(
                "classifier.loading_model",
                model=self._model_name,
                device=self._device,
            )
            self._classifier_instance = pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=self._device,
            )
            logger.info("classifier.model_loaded", model=self._model_name)
        return self._classifier_instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_shot(
        self,
        text: str,
        candidate_labels: list[str],
        *,
        multi_label: bool = False,
        hypothesis_template: str = "This text is about {}.",
    ) -> dict[str, Any]:
        """Run the zero-shot pipeline and return raw output dict."""
        # Truncate to ~512 tokens worth of chars to stay within model limits.
        truncated = text[:2048]
        result: dict[str, Any] = self._classifier(
            truncated,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template,
        )
        return result

    @staticmethod
    def _keyword_urgency_boost(text: str) -> dict[UrgencyLevel, float]:
        """Compute a keyword-based urgency boost for each level."""
        text_lower = text.lower()
        boosts: dict[UrgencyLevel, float] = {}
        for level, keywords in _URGENCY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            boosts[level] = count * 0.05  # each keyword match adds 0.05
        return boosts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_domain(self, text: str) -> ClassificationResult:
        """Classify a regulatory document into one of the predefined domains.

        Domains: financial, healthcare, environmental, data_privacy, labor,
        trade, technology.

        Args:
            text: Full or partial text of the regulatory document.

        Returns:
            A ``ClassificationResult`` with the top domain, confidence, and
            the full score distribution.
        """
        logger.info("classifier.classify_domain.start", text_len=len(text))

        result = self._zero_shot(
            text,
            candidate_labels=REGULATORY_DOMAINS,
            hypothesis_template="This regulatory document pertains to {} regulation.",
        )

        labels: list[str] = result["labels"]
        scores: list[float] = result["scores"]
        all_scores = dict(zip(labels, scores, strict=False))

        classification = ClassificationResult(
            domain=labels[0],
            confidence=scores[0],
            all_scores=all_scores,
        )

        logger.info(
            "classifier.classify_domain.done",
            domain=classification.domain,
            confidence=round(classification.confidence, 4),
        )
        return classification

    def classify_urgency(self, text: str) -> UrgencyLevel:
        """Classify the urgency level of a regulatory document.

        Uses a combination of zero-shot NLI classification and keyword
        heuristics to produce a robust urgency estimate.

        Args:
            text: Full or partial text of the regulatory document.

        Returns:
            An ``UrgencyLevel`` enum value (CRITICAL, HIGH, MEDIUM, LOW).
        """
        logger.info("classifier.classify_urgency.start", text_len=len(text))

        # Zero-shot classification.
        result = self._zero_shot(
            text,
            candidate_labels=_URGENCY_LABELS,
            hypothesis_template="{}",
        )

        labels: list[str] = result["labels"]
        scores: list[float] = result["scores"]

        # Combine with keyword heuristics.
        keyword_boosts = self._keyword_urgency_boost(text)

        combined_scores: dict[UrgencyLevel, float] = {}
        for label, score in zip(labels, scores, strict=False):
            level = _URGENCY_LABEL_MAP[label]
            boost = keyword_boosts.get(level, 0.0)
            combined_scores[level] = score + boost

        # Pick the highest combined score.
        best_level = max(combined_scores, key=lambda lvl: combined_scores[lvl])

        logger.info(
            "classifier.classify_urgency.done",
            urgency=best_level.value,
            scores={k.value: round(v, 4) for k, v in combined_scores.items()},
        )
        return best_level

    def classify_document_type(self, text: str) -> str:
        """Classify the type of regulatory document.

        Types: rule, guidance, enforcement_action, amendment, notice.

        Args:
            text: Full or partial text of the regulatory document.

        Returns:
            The predicted document type string.
        """
        logger.info("classifier.classify_document_type.start", text_len=len(text))

        result = self._zero_shot(
            text,
            candidate_labels=DOCUMENT_TYPES,
            hypothesis_template="This document is a regulatory {}.",
        )

        doc_type: str = result["labels"][0]

        logger.info(
            "classifier.classify_document_type.done",
            document_type=doc_type,
            confidence=round(result["scores"][0], 4),
        )
        return doc_type
