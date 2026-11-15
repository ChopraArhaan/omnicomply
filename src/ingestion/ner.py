"""
Legal Named Entity Recognition (NER) extractor.

Uses a ``transformers`` token-classification pipeline for general NER and
augments it with rule-based post-processing and regex patterns specifically
tuned for regulatory / legal text.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Label taxonomy ───────────────────────────────────────────────────────────

EntityLabel = Literal[
    "REGULATION_ID",
    "ORGANIZATION",
    "DATE",
    "PENALTY_AMOUNT",
    "JURISDICTION",
    "PERSON",
]


# ── Pydantic models ─────────────────────────────────────────────────────────


class Entity(BaseModel):
    """A single named entity extracted from text."""

    text: str = Field(..., description="Surface form of the entity.")
    label: EntityLabel = Field(..., description="Semantic category.")
    start: int = Field(..., ge=0, description="Character offset start (inclusive).")
    end: int = Field(..., ge=0, description="Character offset end (exclusive).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score.")


# ── NER Extractor ────────────────────────────────────────────────────────────


class LegalNERExtractor:
    """Extract legal and regulatory entities from text.

    Strategy:
    1. Run a transformer-based token-classification model to get generic NER
       labels (PER, ORG, LOC, DATE, MISC).
    2. Map generic labels to the legal taxonomy.
    3. Apply regex-based rules for regulation IDs, penalty amounts, and
       jurisdiction patterns that models frequently miss.
    4. Merge sub-word tokens and deduplicate.
    """

    # Default model — can be overridden for domain-fine-tuned checkpoints
    DEFAULT_MODEL = "dslim/bert-base-NER"

    # ── regex patterns for legal entities ────────────────────────────────
    _REGULATION_ID_RE = re.compile(
        r"\b("
        r"\d{1,2}\s+(?:CFR|U\.?S\.?C\.?|C\.F\.R\.)\s+(?:§\s*)?\d+[\w.\-]*"
        r"|(?:SEC\s+Rule|Rule)\s+\d+[a-z]*(?:-\d+)?"
        r"|(?:Regulation\s+[A-Z]{1,3})"
        r"|(?:Pub\.?\s*L\.?\s*\d+-\d+)"
        r"|(?:(?:S\.|H\.R\.)\s*\d+)"
        r")\b",
        re.IGNORECASE,
    )
    _PENALTY_AMOUNT_RE = re.compile(
        r"\$\s*[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|thousand|per\s+(?:day|violation))?",
        re.IGNORECASE,
    )
    _JURISDICTION_RE = re.compile(
        r"\b("
        r"(?:United States|U\.S\.)"
        r"|(?:European Union|EU)"
        r"|(?:State of [A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        r"|(?:(?:New York|California|Texas|Florida|Illinois|Delaware"
        r"|Massachusetts|Washington|Oregon|Georgia|Virginia"
        r"|New Jersey|Pennsylvania|Ohio|Michigan|Colorado)\b)"
        r"|(?:District of Columbia|D\.C\.)"
        r")\b"
    )
    _DATE_RE = re.compile(
        r"\b("
        r"(?:January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        r"|\d{1,2}/\d{1,2}/\d{4}"
        r"|\d{4}-\d{2}-\d{2}"
        r")\b"
    )

    # Map generic NER labels → legal taxonomy
    _LABEL_MAP: ClassVar[dict[str, EntityLabel]] = {
        "PER": "PERSON",
        "B-PER": "PERSON",
        "I-PER": "PERSON",
        "ORG": "ORGANIZATION",
        "B-ORG": "ORGANIZATION",
        "I-ORG": "ORGANIZATION",
        "LOC": "JURISDICTION",
        "B-LOC": "JURISDICTION",
        "I-LOC": "JURISDICTION",
        "DATE": "DATE",
        "B-DATE": "DATE",
        "I-DATE": "DATE",
        "MISC": "REGULATION_ID",
        "B-MISC": "REGULATION_ID",
        "I-MISC": "REGULATION_ID",
    }

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = self._load_pipeline()

    # ── public API ───────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities from text.

        Parameters
        ----------
        text:
            Input text to analyse.

        Returns
        -------
        list[Entity]
            Deduplicated, sorted list of entities.
        """
        if not text.strip():
            return []

        entities: list[Entity] = []

        # 1. Transformer-based NER
        if self._pipeline is not None:
            try:
                raw_entities = self._pipeline(text)
                merged = self._merge_subword_entities(raw_entities, text)
                entities.extend(merged)
            except Exception:
                logger.warning("Transformer NER failed; using regex fallback only")

        # 2. Regex-based extraction (always runs to catch what models miss)
        regex_entities = self._regex_extract(text)

        # 3. Post-process and merge
        entities = self._post_process_legal_entities(entities, regex_entities)

        logger.info("NER extraction complete", entity_count=len(entities))
        return entities

    # ── private helpers ──────────────────────────────────────────────────

    def _load_pipeline(self):
        """Load the HuggingFace token-classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline(
                "token-classification",
                model=self._model_name,
                aggregation_strategy="none",
            )
            logger.info("NER pipeline loaded", model=self._model_name)
            return pipe
        except Exception:
            logger.warning(
                "Failed to load NER pipeline; using regex-only extraction",
                model=self._model_name,
            )
            return None

    def _merge_subword_entities(self, raw_entities: list[dict], text: str) -> list[Entity]:
        """Merge BPE/WordPiece sub-word tokens into whole entities.

        The ``transformers`` pipeline with ``aggregation_strategy="none"``
        returns one prediction per sub-word token.  This method stitches
        consecutive tokens with the same (or B-/I-) label into a single entity.
        """
        if not raw_entities:
            return []

        merged: list[Entity] = []
        current_label: str | None = None
        current_start: int = 0
        current_end: int = 0
        current_scores: list[float] = []

        for token in raw_entities:
            label_raw = token.get("entity", token.get("entity_group", "O"))
            score = float(token.get("score", 0.0))
            start = int(token.get("start", 0))
            end = int(token.get("end", 0))

            # Strip B-/I- prefix for comparison
            base_label = label_raw.split("-", 1)[-1] if "-" in label_raw else label_raw

            is_continuation = (
                label_raw.startswith("I-")
                and current_label is not None
                and base_label == current_label
            )
            is_same_label = (
                current_label == base_label
                and start <= current_end + 1  # allow 1-char gap for spaces
            )

            if is_continuation or is_same_label:
                current_end = end
                current_scores.append(score)
            else:
                # Emit previous entity
                if current_label and current_label != "O":
                    mapped = self._LABEL_MAP.get(current_label)
                    if mapped is None:
                        mapped = self._LABEL_MAP.get(f"B-{current_label}")
                    if mapped:
                        merged.append(
                            Entity(
                                text=text[current_start:current_end].strip(),
                                label=mapped,
                                start=current_start,
                                end=current_end,
                                confidence=sum(current_scores) / len(current_scores),
                            )
                        )

                current_label = base_label
                current_start = start
                current_end = end
                current_scores = [score]

        # Emit trailing entity
        if current_label and current_label != "O":
            mapped = self._LABEL_MAP.get(current_label)
            if mapped is None:
                mapped = self._LABEL_MAP.get(f"B-{current_label}")
            if mapped:
                merged.append(
                    Entity(
                        text=text[current_start:current_end].strip(),
                        label=mapped,
                        start=current_start,
                        end=current_end,
                        confidence=sum(current_scores) / len(current_scores),
                    )
                )

        return merged

    def _regex_extract(self, text: str) -> list[Entity]:
        """Extract entities using hand-crafted regex patterns."""
        entities: list[Entity] = []

        for match in self._REGULATION_ID_RE.finditer(text):
            entities.append(
                Entity(
                    text=match.group().strip(),
                    label="REGULATION_ID",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                )
            )

        for match in self._PENALTY_AMOUNT_RE.finditer(text):
            entities.append(
                Entity(
                    text=match.group().strip(),
                    label="PENALTY_AMOUNT",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90,
                )
            )

        for match in self._JURISDICTION_RE.finditer(text):
            entities.append(
                Entity(
                    text=match.group().strip(),
                    label="JURISDICTION",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                )
            )

        for match in self._DATE_RE.finditer(text):
            entities.append(
                Entity(
                    text=match.group().strip(),
                    label="DATE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90,
                )
            )

        return entities

    def _post_process_legal_entities(
        self,
        model_entities: list[Entity],
        regex_entities: list[Entity],
    ) -> list[Entity]:
        """Merge model and regex entities, deduplicate, and sort.

        When a regex entity and a model entity overlap, keep the one with
        higher confidence.  Prefer the regex entity for REGULATION_ID since
        the pattern is more precise than a generic MISC label.
        """
        all_entities = list(model_entities)

        for regex_ent in regex_entities:
            overlaps = False
            for i, existing in enumerate(all_entities):
                # Check for overlapping spans
                if regex_ent.start < existing.end and regex_ent.end > existing.start:
                    overlaps = True
                    # Prefer regex for regulation IDs; otherwise keep higher confidence
                    if (
                        regex_ent.label == "REGULATION_ID"
                        or regex_ent.confidence > existing.confidence
                    ):
                        all_entities[i] = regex_ent
                    break
            if not overlaps:
                all_entities.append(regex_ent)

        # Filter out very low-confidence entities
        all_entities = [e for e in all_entities if e.confidence >= 0.3]

        # Sort by start position
        all_entities.sort(key=lambda e: e.start)

        return all_entities
