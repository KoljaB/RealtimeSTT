"""Internal recorder state, callback, and spinner helpers."""

import logging
import threading

import halo


logger = logging.getLogger("realtimestt")


def run_callback(recorder, cb, *args, **kwargs):
    if recorder.start_callback_in_new_thread:
        # Run the callback in a new thread to avoid blocking the main thread
        threading.Thread(target=cb, args=args, kwargs=kwargs, daemon=True).start()
    else:
        # Run the callback in the main thread to avoid threading issues
        cb(*args, **kwargs)


def set_recorder_state(recorder, new_state):
    """
    Update the current state of the recorder and execute
    corresponding state-change callbacks.

    Args:
        recorder: Recorder-like object whose state should be updated.
        new_state (str): The new state to set.

    """
    # Check if the state has actually changed
    if new_state == recorder.state:
        return

    # Store the current state for later comparison
    old_state = recorder.state

    # Update to the new state
    recorder.state = new_state

    # Log the state change
    logger.info(f"State changed from '{old_state}' to '{new_state}'")

    # Execute callbacks based on transitioning FROM a particular state
    if old_state == "listening":
        if recorder.on_vad_detect_stop:
            recorder._run_callback(recorder.on_vad_detect_stop)
    elif old_state == "wakeword":
        if recorder.on_wakeword_detection_end:
            recorder._run_callback(recorder.on_wakeword_detection_end)

    # Execute callbacks based on transitioning TO a particular state
    if new_state == "listening":
        if recorder.on_vad_detect_start:
            recorder._run_callback(recorder.on_vad_detect_start)
        recorder._set_spinner("speak now")
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 250
    elif new_state == "wakeword":
        if recorder.on_wakeword_detection_start:
            recorder._run_callback(recorder.on_wakeword_detection_start)
        recorder._set_spinner(f"say {recorder.wake_words}")
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 500
    elif new_state == "transcribing":
        recorder._set_spinner("transcribing")
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 50
    elif new_state == "recording":
        recorder._set_spinner("recording")
        if recorder.spinner and recorder.halo:
            recorder.halo._interval = 100
    elif new_state == "inactive":
        if recorder.spinner and recorder.halo:
            recorder.halo.stop()
            recorder.halo = None


def set_spinner(recorder, text):
    """
    Update the spinner's text or create a new
    spinner with the provided text.

    Args:
        recorder: Recorder-like object whose spinner should be updated.
        text (str): The text to be displayed alongside the spinner.
    """
    if recorder.spinner:
        # If the Halo spinner doesn't exist, create and start it
        if recorder.halo is None:
            recorder.halo = halo.Halo(text=text)
            recorder.halo.start()
        # If the Halo spinner already exists, just update the text
        else:
            recorder.halo.text = text
