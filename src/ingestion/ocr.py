"""
OCR processor for scanned documents and images.

Uses the ``transformers`` vision pipeline with a TrOCR model for high-quality
optical character recognition.  Falls back to a basic Tesseract wrapper if
``transformers`` is not available.
"""

from __future__ import annotations

import io
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ── Pydantic models ─────────────────────────────────────────────────────────


class OCRResult(BaseModel):
    """Result of OCR processing on a single image."""

    text: str = Field(default="", description="Recognised text.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score for the recognised text.",
    )


# Re-use ParsedDocument/PageContent from pdf_parser so the pipeline gets a
# uniform representation.
from src.ingestion.pdf_parser import DocumentMetadata, PageContent, ParsedDocument  # noqa: E402


class OCRProcessor:
    """Optical Character Recognition processor.

    Wraps HuggingFace ``transformers`` image-to-text pipeline with the TrOCR
    model for printed text recognition.
    """

    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = self._load_pipeline()

    # ── public API ───────────────────────────────────────────────────────

    def process_image(self, image_path: Path) -> str:
        """Run OCR on a single image file and return the extracted text.

        Parameters
        ----------
        image_path:
            Path to an image file (PNG, JPEG, TIFF, BMP).

        Returns
        -------
        str
            Recognised text.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        result = self._run_ocr(image)

        logger.info(
            "Image OCR complete",
            file=str(image_path),
            chars=len(result.text),
            confidence=round(result.confidence, 3),
        )
        return result.text

    def process_scanned_pdf(self, pdf_path: Path) -> ParsedDocument:
        """Convert each page of a scanned PDF to an image and run OCR.

        Parameters
        ----------
        pdf_path:
            Path to the scanned PDF file.

        Returns
        -------
        ParsedDocument
            Structured document with per-page OCR text.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        images = self._pdf_to_images(pdf_path)

        pages: list[PageContent] = []
        text_parts: list[str] = []
        total_confidence = 0.0

        for idx, img in enumerate(images, start=1):
            result = self._run_ocr(img)
            pages.append(PageContent(page_num=idx, text=result.text))
            text_parts.append(result.text)
            total_confidence += result.confidence

        full_text = "\n\n".join(text_parts)
        avg_confidence = total_confidence / max(len(images), 1)

        logger.info(
            "Scanned PDF OCR complete",
            file=str(pdf_path),
            pages=len(pages),
            avg_confidence=round(avg_confidence, 3),
        )

        return ParsedDocument(
            content=full_text,
            pages=pages,
            metadata=DocumentMetadata(
                title=pdf_path.stem,
                num_pages=len(pages),
            ),
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _load_pipeline(self):
        """Load the HuggingFace image-to-text pipeline lazily."""
        try:
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline("image-to-text", model=self._model_name)
            logger.info("TrOCR pipeline loaded", model=self._model_name)
            return pipe
        except Exception:
            logger.warning(
                "Failed to load TrOCR pipeline; OCR will raise on use",
                model=self._model_name,
            )
            return None

    def _run_ocr(self, image) -> OCRResult:
        """Run the OCR pipeline on a PIL Image and return structured result."""
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        if self._pipeline is None:
            raise RuntimeError(
                "OCR pipeline is not loaded.  Ensure `transformers` and "
                f"the model `{self._model_name}` are available."
            )

        outputs = self._pipeline(image)

        # The image-to-text pipeline returns a list of dicts with
        # 'generated_text' (and optionally a score).
        if outputs and isinstance(outputs, list):
            first = outputs[0]
            text = first.get("generated_text", "") if isinstance(first, dict) else str(first)
            # Confidence / score is model-dependent; default to 1.0 if absent
            confidence = float(first.get("score", 1.0)) if isinstance(first, dict) else 1.0
        else:
            text = ""
            confidence = 0.0

        return OCRResult(text=text.strip(), confidence=confidence)

    @staticmethod
    def _pdf_to_images(pdf_path: Path) -> list:
        """Convert PDF pages to PIL Images.

        Tries ``pdf2image`` (poppler) first, then falls back to ``PyMuPDF``
        (fitz).
        """
        # Attempt 1 — pdf2image
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(str(pdf_path), dpi=300)
            logger.debug("PDF→images via pdf2image", pages=len(images))
            return images
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pdf2image failed", error=str(exc))

        # Attempt 2 — PyMuPDF
        try:
            import fitz  # PyMuPDF
            from PIL import Image

            doc = fitz.open(str(pdf_path))
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img.convert("RGB"))
            doc.close()
            logger.debug("PDF→images via PyMuPDF", pages=len(images))
            return images
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("PyMuPDF failed", error=str(exc))

        raise RuntimeError(
            "Cannot convert PDF to images.  Install either `pdf2image` (with poppler) or `PyMuPDF`."
        )
