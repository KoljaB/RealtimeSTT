"""
Exports transcription engine interfaces and factory helpers.
"""

from .base import (
    BaseEngine,
    BaseTranscriptionEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
    UnsupportedTranscriptionEngineError,
)
from .factory import create_transcription_engine, get_supported_transcription_engines
from .sherpa_onnx_engine import PARAKEET_MODEL_MANIFEST
from .nemotron_engine import (
    DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL,
    NEMOTRON_DOWNLOAD_URL,
    NEMOTRON_MODEL_MANIFEST,
    NemotronEngine,
    NemotronStreamingSession,
    SherpaOnnxNemotronBackend,
    SherpaOnnxNemotronEngine,
    SherpaOnnxNemotronStreamingSession,
)

__all__ = [
    "BaseEngine",
    "BaseTranscriptionEngine",
    "StreamingTranscriptionSession",
    "TranscriptionEngineConfig",
    "TranscriptionEngineError",
    "TranscriptionInfo",
    "TranscriptionResult",
    "UnsupportedTranscriptionEngineError",
    "create_transcription_engine",
    "get_supported_transcription_engines",
    "DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL",
    "NEMOTRON_DOWNLOAD_URL",
    "NEMOTRON_MODEL_MANIFEST",
    "PARAKEET_MODEL_MANIFEST",
    "NemotronEngine",
    "NemotronStreamingSession",
    "SherpaOnnxNemotronBackend",
    "SherpaOnnxNemotronEngine",
    "SherpaOnnxNemotronStreamingSession",
]
