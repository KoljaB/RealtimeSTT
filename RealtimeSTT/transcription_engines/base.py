"""
Defines common transcription engine result types and interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class TranscriptionInfo:
    """
    Carries language metadata for a transcription result.
    """

    language: Optional[str] = None
    language_probability: float = 0.0


@dataclass
class TranscriptionResult:
    """
    Carries recognized text and optional language metadata.
    """

    text: str
    info: TranscriptionInfo = field(default_factory=TranscriptionInfo)


@dataclass
class TranscriptionEngineConfig:
    """
    Collects shared configuration for transcription engine adapters.
    """

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
    """
    Reports transcription engine setup or runtime failures.
    """

    pass


class UnsupportedTranscriptionEngineError(TranscriptionEngineError):
    """
    Reports an unknown transcription engine name.
    """

    pass


class StreamingTranscriptionSession(ABC):
    """
    Defines the streaming transcription session interface.
    """

    @abstractmethod
    def reset(self):
        """
        Resets the session for a new utterance.
        """
        raise NotImplementedError

    @abstractmethod
    def accept_audio(self, audio, sample_rate=None):
        """
        Accepts one audio chunk for the current utterance.
        """
        raise NotImplementedError

    def decode(self):
        """
        Runs any pending decode work for accepted audio.
        """

    @abstractmethod
    def get_result(self) -> TranscriptionResult:
        """
        Returns the current partial or final transcription result.
        """
        raise NotImplementedError

    def finish(self) -> TranscriptionResult:
        """
        Finalizes the utterance and returns the final result.
        """
        self.decode()
        return self.get_result()

    def close(self):
        """
        Releases resources owned by the session.
        """


class BaseTranscriptionEngine(ABC):
    """
    Defines the synchronous transcription engine interface.
    """

    engine_name = "base"
    supports_streaming = False

    def __init__(self, config: TranscriptionEngineConfig):
        """
        Stores shared engine configuration.
        """
        self.config = config

    def warmup(self, audio):
        """
        Warms the engine with a short English transcription call.
        """
        self.transcribe(audio, language="en", use_prompt=False)

    @abstractmethod
    def transcribe(self, audio, language=None, use_prompt=True) -> TranscriptionResult:
        """
        Transcribes audio and returns a normalized result.
        """
        raise NotImplementedError

    def create_streaming_session(self, language=None, use_prompt=True) -> StreamingTranscriptionSession:
        """
        Creates a streaming session when the engine supports one.
        """
        raise TranscriptionEngineError(
            "%s does not support chunk streaming." % self.engine_name
        )

    def _normalize_audio(self, audio):
        """
        Normalizes audio when the engine configuration requests it.
        """
        if audio is None or audio.size == 0:
            raise TranscriptionEngineError("Received None audio for transcription")

        if not self.config.normalize_audio:
            return audio

        peak = abs(audio).max()
        if peak > 0:
            audio = (audio / peak) * 0.95
        return audio

    def _get_prompt(self, use_prompt):
        """
        Returns the configured prompt when prompting is enabled.
        """
        if use_prompt and self.config.initial_prompt:
            return self.config.initial_prompt
        return None
