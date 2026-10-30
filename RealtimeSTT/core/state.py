"""
Internal recorder state, callback, and spinner helpers.
"""

import logging
import threading

import halo


logger = logging.getLogger("realtimestt")


def _spinner_text(recorder, attribute, default):
    """
    Returns configured spinner text with legacy recorder-like fallback.
    """
    return getattr(recorder, attribute, default)


def _wakeword_spinner_text(recorder):
    """
    Returns configured wake-word spinner text with wake-word interpolation.
    """
    text = _spinner_text(recorder, "spinner_wakeword_text", "say {wake_words}")
    if isinstance(text, str):
        return text.format(wake_words=getattr(recorder, "wake_words", ""))
    return text


def run_callback(recorder, cb, *args, **kwargs):
    """
    Runs a callback according to the recorder threading setting.
    """
    if recorder.start_callback_in_new_thread:
        threading.Thread(target=cb, args=args, kwargs=kwargs, daemon=True).start()
    else:
        cb(*args, **kwargs)


def set_recorder_state(recorder, new_state):
    """
    Updates recorder state and fires matching transition callbacks.

    Args:
    - recorder: Recorder-like object whose state should be updated.
    - new_state: New recorder state.
    """
    if new_state == recorder.state:
        return

    old_state = recorder.state
    recorder.state = new_state

    logger.info(f"State changed from '{old_state}' to '{new_state}'")

    if old_state == "listening":
        if recorder.on_vad_detect_stop:
            run_callback(recorder, recorder.on_vad_detect_stop)
    elif old_state == "wakeword":
        if recorder.on_wakeword_detection_end:
            run_callback(recorder, recorder.on_wakeword_detection_end)

    if new_state == "listening":
        if recorder.on_vad_detect_start:
            run_callback(recorder, recorder.on_vad_detect_start)
        set_spinner(
            recorder,
            _spinner_text(recorder, "spinner_listening_text", "speak now"),
        )
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 250
    elif new_state == "wakeword":
        if recorder.on_wakeword_detection_start:
            run_callback(recorder, recorder.on_wakeword_detection_start)
        set_spinner(recorder, _wakeword_spinner_text(recorder))
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 500
    elif new_state == "transcribing":
        set_spinner(
            recorder,
            _spinner_text(
                recorder,
                "spinner_transcribing_text",
                "transcribing",
            ),
        )
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 50
    elif new_state == "recording":
        set_spinner(
            recorder,
            _spinner_text(recorder, "spinner_recording_text", "recording"),
        )
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 100
    elif new_state == "inactive":
        if recorder.spinner and recorder.halo:
            recorder.halo.stop()
            recorder.halo = None


def set_spinner(recorder, text):
    """
    Updates the active spinner or creates one when needed.

    Args:
    - recorder: Recorder-like object whose spinner should be updated.
    - text: Text displayed alongside the spinner.
    """
    if recorder.spinner:
        if recorder.halo is None:
            recorder.halo = halo.Halo(text=text)
            recorder.halo.start()
        else:
            recorder.halo.text = text
