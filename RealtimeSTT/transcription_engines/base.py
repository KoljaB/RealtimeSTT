from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class TranscriptionInfo:
    language: Optional[str] = None
    language_probability: float = 0.0


@dataclass
class TranscriptionResult:
    text: str
    info: TranscriptionInfo = field(default_factory=TranscriptionInfo)


@dataclass
class TranscriptionEngineConfig:
    model: str
    download_root: Optional[str] = None
    compute_type: str = "default"
    gpu_device_index: Union[int, List[int]] = 0
    device: str = "cpu"
    beam_size: int = 5
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    suppress_tokens: Optional[List[int]] = None
    batch_size: int = 0
    vad_filter: bool = True
    normalize_audio: bool = False
    engine_options: Optional[Dict[str, Any]] = None


class TranscriptionEngineError(RuntimeError):
    pass


class UnsupportedTranscriptionEngineError(TranscriptionEngineError):
    pass


class StreamingTranscriptionSession(ABC):
    @abstractmethod
    def reset(self):
        """Reset the session for a new utterance."""
        raise NotImplementedError

    @abstractmethod
    def accept_audio(self, audio, sample_rate=None):
        """Accept one chunk of audio for the current utterance."""
        raise NotImplementedError

    def decode(self):
        """Run any pending decode work for already accepted audio."""

    @abstractmethod
    def get_result(self) -> TranscriptionResult:
        """Return the current partial or final transcription result."""
        raise NotImplementedError

    def finish(self) -> TranscriptionResult:
        """Finalize the utterance and return the final transcription result."""
        self.decode()
        return self.get_result()

    def close(self):
        """Release any resources owned by the session."""


class BaseTranscriptionEngine(ABC):
    engine_name = "base"
    supports_streaming = False

    def __init__(self, config: TranscriptionEngineConfig):
        self.config = config

    def warmup(self, audio):
        self.transcribe(audio, language="en", use_prompt=False)

    @abstractmethod
    def transcribe(self, audio, language=None, use_prompt=True) -> TranscriptionResult:
        raise NotImplementedError

    def create_streaming_session(self, language=None, use_prompt=True) -> StreamingTranscriptionSession:
        raise TranscriptionEngineError(
            "%s does not support chunk streaming." % self.engine_name
        )

    def _normalize_audio(self, audio):
        if audio is None or audio.size == 0:
            raise TranscriptionEngineError("Received None audio for transcription")

        if not self.config.normalize_audio:
            return audio

        peak = abs(audio).max()
        if peak > 0:
            audio = (audio / peak) * 0.95
        return audio

    def _get_prompt(self, use_prompt):
        if use_prompt and self.config.initial_prompt:
            return self.config.initial_prompt
        return None
