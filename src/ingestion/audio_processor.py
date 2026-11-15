"""
Audio processing module for transcription and classification.

Uses OpenAI Whisper for speech-to-text transcription and a HuggingFace
``audio-classification`` pipeline for content classification (e.g.,
distinguishing earnings calls from legal depositions).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Supported audio extensions ───────────────────────────────────────────────

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".wma", ".aac"}


# ── Pydantic models ─────────────────────────────────────────────────────────


class Segment(BaseModel):
    """A time-aligned segment of a transcription."""

    start: float = Field(..., ge=0.0, description="Segment start time in seconds.")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds.")
    text: str = Field(..., description="Transcribed text for this segment.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Average token probability.")


class Transcript(BaseModel):
    """Full transcription result."""

    text: str = Field(default="", description="Complete transcribed text.")
    segments: list[Segment] = Field(default_factory=list, description="Time-aligned segments.")
    language: str = Field(default="en", description="Detected language code.")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Total audio duration.")


class AudioClassification(BaseModel):
    """Classification result for an audio file."""

    label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence.")


# ── Audio Processor ─────────────────────────────────────────────────────────


class AudioProcessor:
    """Transcribe and classify audio files.

    Transcription uses OpenAI Whisper (local model).  Classification uses a
    HuggingFace ``audio-classification`` pipeline.
    """

    DEFAULT_WHISPER_MODEL = "base"
    DEFAULT_CLASSIFIER_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(
        self,
        whisper_model: str | None = None,
        classifier_model: str | None = None,
    ) -> None:
        self._whisper_model_name = whisper_model or self.DEFAULT_WHISPER_MODEL
        self._classifier_model_name = classifier_model or self.DEFAULT_CLASSIFIER_MODEL
        self._whisper = self._load_whisper()
        self._classifier = self._load_classifier()

    # ── public API ───────────────────────────────────────────────────────

    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe an audio file to text using Whisper.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        Returns
        -------
        Transcript
            Structured transcription with segments.
        """
        audio_path = Path(audio_path)
        self._validate_audio(audio_path)

        # Convert to WAV for consistent Whisper input
        wav_path = self._convert_to_wav(audio_path)

        try:
            result = self._whisper.transcribe(
                str(wav_path),
                fp16=False,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Whisper transcription failed", error=str(exc))
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        segments: list[Segment] = []
        for seg in result.get("segments", []):
            # Whisper provides avg_logprob; convert to a 0-1 confidence
            avg_logprob = seg.get("avg_logprob", 0.0)
            # Clamp e^logprob to [0, 1]
            confidence = min(max(2.0**avg_logprob, 0.0), 1.0)

            segments.append(
                Segment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg.get("text", "").strip(),
                    confidence=round(confidence, 4),
                )
            )

        # Calculate duration from last segment or fallback
        duration = segments[-1].end if segments else 0.0

        transcript = Transcript(
            text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language", "en"),
            duration_seconds=round(duration, 2),
        )

        logger.info(
            "Audio transcribed",
            file=str(audio_path),
            duration=transcript.duration_seconds,
            segments=len(segments),
            language=transcript.language,
        )
        return transcript

    def classify_audio(self, audio_path: Path) -> AudioClassification:
        """Classify the content/type of an audio file.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        Returns
        -------
        AudioClassification
            Top predicted label with confidence.
        """
        audio_path = Path(audio_path)
        self._validate_audio(audio_path)

        if self._classifier is None:
            raise RuntimeError(
                "Audio classification pipeline not loaded.  "
                "Ensure `transformers` and `librosa` are installed."
            )

        wav_path = self._convert_to_wav(audio_path)

        try:
            results = self._classifier(str(wav_path))
        except Exception as exc:
            logger.error("Audio classification failed", error=str(exc))
            raise RuntimeError(f"Classification failed: {exc}") from exc

        if results and isinstance(results, list):
            top = results[0]
            classification = AudioClassification(
                label=top.get("label", "unknown"),
                confidence=round(float(top.get("score", 0.0)), 4),
            )
        else:
            classification = AudioClassification(label="unknown", confidence=0.0)

        logger.info(
            "Audio classified",
            file=str(audio_path),
            label=classification.label,
            confidence=classification.confidence,
        )
        return classification

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _validate_audio(audio_path: Path) -> None:
        """Check that the file exists and has a known audio extension."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in _AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}.  "
                f"Supported: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
            )

    @staticmethod
    def _convert_to_wav(audio_path: Path) -> Path:
        """Convert an audio file to 16 kHz mono WAV for model consumption.

        Uses ``pydub`` (with ffmpeg) if available, otherwise attempts a direct
        ffmpeg subprocess call.

        Returns
        -------
        Path
            Path to the WAV file (may be a tempfile).
        """
        if audio_path.suffix.lower() == ".wav":
            return audio_path

        wav_path = Path(tempfile.mktemp(suffix=".wav"))

        # Try pydub first
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(str(wav_path), format="wav")
            logger.debug("Converted to WAV via pydub", source=str(audio_path))
            return wav_path
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("pydub conversion failed", error=str(exc))

        # Fallback to raw ffmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav_path),
                ],
                capture_output=True,
                check=True,
            )
            logger.debug("Converted to WAV via ffmpeg", source=str(audio_path))
            return wav_path
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Cannot convert audio to WAV.  Install pydub (pip install pydub) or ffmpeg."
            ) from exc

    def _load_whisper(self):
        """Load the Whisper model."""
        try:
            import whisper

            model = whisper.load_model(self._whisper_model_name)
            logger.info("Whisper model loaded", model=self._whisper_model_name)
            return model
        except ImportError:
            logger.warning("openai-whisper not installed; transcription unavailable")
            return None
        except Exception as exc:
            logger.warning(
                "Failed to load Whisper model",
                model=self._whisper_model_name,
                error=str(exc),
            )
            return None

    def _load_classifier(self):
        """Load the HuggingFace audio-classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline(
                "audio-classification",
                model=self._classifier_model_name,
            )
            logger.info("Audio classifier loaded", model=self._classifier_model_name)
            return pipe
        except Exception:
            logger.warning(
                "Failed to load audio classifier; classification unavailable",
                model=self._classifier_model_name,
            )
            return None
