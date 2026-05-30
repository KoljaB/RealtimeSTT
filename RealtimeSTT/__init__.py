__all__ = [
    "AudioToTextRecorder",
    "AudioToTextRecorderClient",
    "AudioInput",
    "RealtimeSpeechBoundaryDetector",
    "SpeechBoundaryEvent",
    "SpeechBoundaryResult",
]


def __getattr__(name):
    if name == "AudioToTextRecorder":
        from .audio_recorder import AudioToTextRecorder

        return AudioToTextRecorder
    if name == "AudioToTextRecorderClient":
        from .audio_recorder_client import AudioToTextRecorderClient

        return AudioToTextRecorderClient
    if name == "AudioInput":
        from .audio_input import AudioInput

        return AudioInput
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
