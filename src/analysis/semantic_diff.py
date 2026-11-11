"""
Semantic Diff Engine for Regulatory Documents.

Computes paragraph-level semantic diffs between document versions and detects
conflicts between regulations using sentence-transformer embeddings.
"""

from __future__ import annotations

import enum
import re
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.60  # paragraphs with similarity above this are "aligned"
CONFLICT_SIMILARITY_THRESHOLD = 0.70  # minimum similarity to consider a conflict pair


# ---------------------------------------------------------------------------
# Enums & Pydantic models
# ---------------------------------------------------------------------------


class ChangeType(str, enum.Enum):
    """Type of change detected between two paragraph versions."""

    MINOR_EDIT = "minor_edit"
    MAJOR_REVISION = "major_revision"
    SEMANTIC_SHIFT = "semantic_shift"
    REFORMATTED = "reformatted"


class ModifiedSection(BaseModel):
    """A section that was modified between document versions."""

    old_text: str = Field(..., description="Original paragraph text.")
    new_text: str = Field(..., description="Updated paragraph text.")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the two versions.",
    )
    change_type: ChangeType = Field(
        ..., description="Category of the change."
    )


class SemanticDiff(BaseModel):
    """Full semantic diff between two document versions."""

    added_sections: list[str] = Field(
        default_factory=list,
        description="Paragraphs present only in the new document.",
    )
    removed_sections: list[str] = Field(
        default_factory=list,
        description="Paragraphs present only in the old document.",
    )
    modified_sections: list[ModifiedSection] = Field(
        default_factory=list,
        description="Paragraphs that exist in both but have been changed.",
    )
    overall_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average cosine similarity across all aligned paragraph pairs.",
    )


class ConflictType(str, enum.Enum):
    """Type of conflict between two regulations."""

    CONTRADICTORY = "contradictory"
    OVERLAPPING = "overlapping"
    AMBIGUOUS = "ambiguous"


class ConflictSeverity(str, enum.Enum):
    """Severity of a detected conflict."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Conflict(BaseModel):
    """A detected conflict between two regulatory documents."""

    section_a: str = Field(
        ..., description="Text from the first regulation."
    )
    section_b: str = Field(
        ..., description="Text from the second regulation."
    )
    conflict_type: ConflictType = Field(
        ..., description="Category of the conflict."
    )
    severity: ConflictSeverity = Field(
        ..., description="Severity of the conflict."
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of the conflict."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs."""
    paragraphs = _PARAGRAPH_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _cosine_similarity_matrix(
    embeddings_a: np.ndarray, embeddings_b: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of embeddings."""
    # Normalize rows.
    norm_a = embeddings_a / (
        np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-10
    )
    norm_b = embeddings_b / (
        np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-10
    )
    return norm_a @ norm_b.T


def _classify_change(similarity: float) -> ChangeType:
    """Determine the type of change based on cosine similarity."""
    if similarity >= 0.95:
        return ChangeType.REFORMATTED
    if similarity >= 0.80:
        return ChangeType.MINOR_EDIT
    if similarity >= 0.60:
        return ChangeType.MAJOR_REVISION
    return ChangeType.SEMANTIC_SHIFT


# ---------------------------------------------------------------------------
# Conflict detection keywords
# ---------------------------------------------------------------------------

_CONTRADICTORY_SIGNALS = [
    ("shall", "shall not"),
    ("must", "must not"),
    ("required", "prohibited"),
    ("allowed", "forbidden"),
    ("permit", "deny"),
    ("mandatory", "voluntary"),
]


def _detect_contradiction_signals(text_a: str, text_b: str) -> bool:
    """Heuristic: check if two texts contain contradictory obligation language."""
    a_lower = text_a.lower()
    b_lower = text_b.lower()
    for positive, negative in _CONTRADICTORY_SIGNALS:
        if (positive in a_lower and negative in b_lower) or (
            negative in a_lower and positive in b_lower
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SemanticDiffEngine:
    """Compute semantic diffs and detect conflicts between regulatory documents.

    Uses sentence-transformer embeddings at the paragraph level for alignment
    and conflict detection.

    Args:
        model_name: Sentence-transformer model identifier.
        device: Device string (e.g. ``"cpu"``, ``"cuda:0"``).
        similarity_threshold: Minimum similarity to consider two paragraphs
            as aligned (i.e. the same section, possibly modified).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        logger.info(
            "semantic_diff.loading_model",
            model=model_name,
            device=device,
        )
        self._model = SentenceTransformer(model_name, device=device)
        self._threshold = similarity_threshold
        logger.info("semantic_diff.model_loaded", model=model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings: Any = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_diff(self, old_doc: str, new_doc: str) -> SemanticDiff:
        """Compute a paragraph-level semantic diff between two documents.

        Each paragraph in the old document is aligned to its most similar
        paragraph in the new document (and vice-versa). Unmatched paragraphs
        are classified as added or removed; matched paragraphs with imperfect
        similarity are classified as modified.

        Args:
            old_doc: Full text of the previous document version.
            new_doc: Full text of the updated document version.

        Returns:
            A ``SemanticDiff`` describing what changed.
        """
        logger.info("semantic_diff.compute_diff.start")

        old_paras = _split_paragraphs(old_doc)
        new_paras = _split_paragraphs(new_doc)

        if not old_paras and not new_paras:
            return SemanticDiff(overall_similarity=1.0)
        if not old_paras:
            return SemanticDiff(
                added_sections=new_paras, overall_similarity=0.0
            )
        if not new_paras:
            return SemanticDiff(
                removed_sections=old_paras, overall_similarity=0.0
            )

        old_emb = self._encode(old_paras)
        new_emb = self._encode(new_paras)
        sim_matrix = _cosine_similarity_matrix(old_emb, new_emb)

        # Greedy alignment: for each old paragraph, find the best match in new.
        old_matched: set[int] = set()
        new_matched: set[int] = set()
        modified: list[ModifiedSection] = []
        similarities: list[float] = []

        # Sort potential alignments by similarity (descending) for greedy match.
        pairs: list[tuple[float, int, int]] = []
        for i in range(len(old_paras)):
            for j in range(len(new_paras)):
                pairs.append((float(sim_matrix[i, j]), i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for sim, i, j in pairs:
            if i in old_matched or j in new_matched:
                continue
            if sim < self._threshold:
                break  # remaining pairs are below threshold
            old_matched.add(i)
            new_matched.add(j)
            similarities.append(sim)

            if sim < 1.0:
                modified.append(
                    ModifiedSection(
                        old_text=old_paras[i],
                        new_text=new_paras[j],
                        similarity_score=round(sim, 4),
                        change_type=_classify_change(sim),
                    )
                )

        removed = [
            old_paras[i] for i in range(len(old_paras)) if i not in old_matched
        ]
        added = [
            new_paras[j]
            for j in range(len(new_paras))
            if j not in new_matched
        ]

        overall = float(np.mean(similarities)) if similarities else 0.0

        result = SemanticDiff(
            added_sections=added,
            removed_sections=removed,
            modified_sections=modified,
            overall_similarity=round(overall, 4),
        )

        logger.info(
            "semantic_diff.compute_diff.done",
            added=len(added),
            removed=len(removed),
            modified=len(modified),
            overall_similarity=result.overall_similarity,
        )
        return result

    def detect_conflicts(
        self,
        regulation_a: str,
        regulation_b: str,
    ) -> list[Conflict]:
        """Detect potential conflicts between two regulatory documents.

        Identifies paragraph pairs across the two documents that are
        topically similar but contain contradictory obligation language,
        overlapping scope, or ambiguous provisions.

        Args:
            regulation_a: Full text of the first regulation.
            regulation_b: Full text of the second regulation.

        Returns:
            A list of ``Conflict`` instances describing detected conflicts.
        """
        logger.info("semantic_diff.detect_conflicts.start")

        paras_a = _split_paragraphs(regulation_a)
        paras_b = _split_paragraphs(regulation_b)

        if not paras_a or not paras_b:
            logger.info("semantic_diff.detect_conflicts.done", conflicts=0)
            return []

        emb_a = self._encode(paras_a)
        emb_b = self._encode(paras_b)
        sim_matrix = _cosine_similarity_matrix(emb_a, emb_b)

        conflicts: list[Conflict] = []

        for i in range(len(paras_a)):
            for j in range(len(paras_b)):
                sim = float(sim_matrix[i, j])
                if sim < CONFLICT_SIMILARITY_THRESHOLD:
                    continue

                text_a = paras_a[i]
                text_b = paras_b[j]

                # Determine conflict type and severity.
                is_contradiction = _detect_contradiction_signals(text_a, text_b)

                if is_contradiction:
                    conflict_type = ConflictType.CONTRADICTORY
                    severity = ConflictSeverity.HIGH
                    explanation = (
                        f"Paragraphs are semantically similar (cosine={sim:.3f}) "
                        f"but contain contradictory obligation language."
                    )
                elif sim >= 0.90:
                    # Very high similarity across different regulations →
                    # overlapping scope.
                    conflict_type = ConflictType.OVERLAPPING
                    severity = ConflictSeverity.MEDIUM
                    explanation = (
                        f"Paragraphs are nearly identical (cosine={sim:.3f}), "
                        f"indicating overlapping regulatory scope that may "
                        f"cause duplicative compliance obligations."
                    )
                else:
                    # Similar topic, different wording → ambiguous.
                    conflict_type = ConflictType.AMBIGUOUS
                    severity = ConflictSeverity.LOW
                    explanation = (
                        f"Paragraphs address similar topics (cosine={sim:.3f}) "
                        f"with differing language that could create ambiguity "
                        f"in compliance interpretation."
                    )

                conflicts.append(
                    Conflict(
                        section_a=text_a,
                        section_b=text_b,
                        conflict_type=conflict_type,
                        severity=severity,
                        explanation=explanation,
                    )
                )

        logger.info(
            "semantic_diff.detect_conflicts.done",
            conflicts=len(conflicts),
        )
        return conflicts
