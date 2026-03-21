import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import partial

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordInfo] = field(default_factory=list)
    language: str = "he"
    duration: float = 0.0
    inference_time_ms: int = 0


class WhisperService:
    def __init__(self):
        self.model: WhisperModel | None = None
        self.model_path: str = ""
        self.device: str = "cpu"
        self.compute_type: str = "int8"

    async def load_model(
        self,
        model_path: str,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        logger.info(f"Loading Whisper model from {model_path} on {device} ({compute_type})")
        self.model = await asyncio.to_thread(
            WhisperModel, model_path, device=device, compute_type=compute_type
        )
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        logger.info("Whisper model loaded successfully")

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")

        start_time = time.time()

        def _transcribe():
            segments, info = self.model.transcribe(
                audio_path,
                language="he",
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            )
            all_segments = list(segments)
            return all_segments, info

        all_segments, info = await asyncio.to_thread(_transcribe)
        full_text = " ".join(seg.text.strip() for seg in all_segments)

        words = []
        for seg in all_segments:
            if seg.words:
                for w in seg.words:
                    words.append(
                        WordInfo(
                            word=w.word.strip(),
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

        inference_time_ms = int((time.time() - start_time) * 1000)

        return TranscriptionResult(
            text=full_text,
            words=words,
            language=info.language,
            duration=info.duration,
            inference_time_ms=inference_time_ms,
        )

    async def swap_model(self, new_model_path: str) -> None:
        logger.info(f"Hot-swapping model to {new_model_path}")
        await self.load_model(new_model_path, self.device, self.compute_type)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Singleton instance
whisper_service = WhisperService()
