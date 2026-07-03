"""
Exposes the public RealtimeSTT package objects through lazy imports.
"""

__all__ = [
    "AudioToTextRecorder",
    "AudioToTextRecorderClient",
    "AudioInput",
    "BaseEngine",
    "BaseTranscriptionEngine",
    "RealtimeSpeechBoundaryDetector",
    "SpeechBoundaryEvent",
    "SpeechBoundaryResult",
    "StreamingTranscriptionSession",
    "TranscriptionEngineConfig",
    "TranscriptionEngineError",
    "TranscriptionInfo",
    "TranscriptionResult",
    "UnsupportedTranscriptionEngineError",
]


def __getattr__(name):
    """
    Loads exported package attributes lazily.
    """

    if name == "AudioToTextRecorder":
        from .audio_recorder import AudioToTextRecorder

        return AudioToTextRecorder
    if name == "AudioToTextRecorderClient":
        from .audio_recorder_client import AudioToTextRecorderClient

        return AudioToTextRecorderClient
    if name == "AudioInput":
        from .audio_input import AudioInput

        return AudioInput
    if name == "BaseEngine":
        from .engines import BaseEngine

        return BaseEngine
    if name == "BaseTranscriptionEngine":
        from .engines import BaseTranscriptionEngine

        return BaseTranscriptionEngine
    if name == "StreamingTranscriptionSession":
        from .engines import StreamingTranscriptionSession

        return StreamingTranscriptionSession
    if name == "TranscriptionEngineConfig":
        from .engines import TranscriptionEngineConfig

        return TranscriptionEngineConfig
    if name == "TranscriptionEngineError":
        from .engines import TranscriptionEngineError

        return TranscriptionEngineError
    if name == "TranscriptionInfo":
        from .engines import TranscriptionInfo

        return TranscriptionInfo
    if name == "TranscriptionResult":
        from .engines import TranscriptionResult

        return TranscriptionResult
    if name == "UnsupportedTranscriptionEngineError":
        from .engines import UnsupportedTranscriptionEngineError

        return UnsupportedTranscriptionEngineError
    if name == "RealtimeSpeechBoundaryDetector":
        from .core.realtime_boundary_detector import RealtimeSpeechBoundaryDetector

        return RealtimeSpeechBoundaryDetector
    if name == "SpeechBoundaryEvent":
        from .core.realtime_boundary_detector import SpeechBoundaryEvent

        return SpeechBoundaryEvent
    if name == "SpeechBoundaryResult":
        from .core.realtime_boundary_detector import SpeechBoundaryResult

        return SpeechBoundaryResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
