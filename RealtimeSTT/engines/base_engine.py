"""
Public base interfaces for custom RealtimeSTT transcription engines.
"""

from ..transcription_engines.base import (
    BaseEngine,
    BaseTranscriptionEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
    UnsupportedTranscriptionEngineError,
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
]
