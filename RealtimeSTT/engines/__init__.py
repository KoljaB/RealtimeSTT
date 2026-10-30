"""
Public custom-engine authoring interfaces.

The concrete built-in ASR adapters live in RealtimeSTT.transcription_engines.
This package exposes the stable lightweight contract for third-party engines.
"""

from .base_engine import (
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
