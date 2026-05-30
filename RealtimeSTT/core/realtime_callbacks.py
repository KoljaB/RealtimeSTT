"""Internal realtime callback publication helpers."""

from .state import run_callback


def publish_realtime_transcription_stabilized(recorder, text):
    """
    Publishes stabilized realtime text while recording is active.
    """
    if recorder.on_realtime_transcription_stabilized:
        if recorder.is_recording:
            run_callback(recorder, recorder.on_realtime_transcription_stabilized, text)


def publish_realtime_transcription_update(recorder, text):
    """
    Publishes realtime preview text while recording is active.
    """
    if recorder.on_realtime_transcription_update:
        if recorder.is_recording:
            run_callback(recorder, recorder.on_realtime_transcription_update, text)
