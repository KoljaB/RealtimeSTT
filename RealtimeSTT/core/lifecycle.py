"""Internal recorder lifecycle API helpers."""

import copy
import logging
import time

from .realtime_text_stabilizer import RealtimeTextStabilizer
from .recording_buffers import (
    get_next_recorded_audio,
    queue_recorded_audio,
    set_audio_from_frames,
)
from .state import run_callback, set_recorder_state
from .voice_activity import reset_silero_vad_state


logger = logging.getLogger("realtimestt")


def start_recording(recorder, frames=None):
    # Ensure there's a minimum interval
    # between stopping and starting recording
    if (time.time() - recorder.recording_stop_time
            < recorder.min_gap_between_recordings):
        logger.info("Attempted to start recording "
                    "too soon after stopping."
                    )
        recorder._pending_preroll_selection = None
        recorder.last_preroll_selection = None
        return recorder

    logger.info("recording started")
    set_recorder_state(recorder, "recording")
    recorder.text_storage = []
    recorder.realtime_stabilized_text = ""
    recorder.realtime_stabilized_safetext = ""
    recorder.realtime_observation_sequence = 0
    recorder.realtime_recording_id = (
        getattr(recorder, "realtime_recording_id", 0) + 1
    )
    recorder.recording_start_monotonic = time.monotonic()
    recorder.last_preroll_selection = getattr(
        recorder,
        "_pending_preroll_selection",
        None,
    )
    recorder._pending_preroll_selection = None
    recorder.wakeword_detected = False
    recorder.wake_word_detect_time = 0
    recorder.frames = []
    if frames:
        recorder.frames = frames

    recorder.recording_start_time = time.time()
    recorder.speech_end_silence_candidate_start = 0
    realtime_text_stabilizer = getattr(
        recorder,
        "realtime_text_stabilizer",
        None,
    )
    if realtime_text_stabilizer is None:
        realtime_text_stabilizer = RealtimeTextStabilizer()
        recorder.realtime_text_stabilizer = realtime_text_stabilizer
    realtime_text_stabilizer.reset(
        recorder.realtime_recording_id,
        started_at_monotonic=recorder.recording_start_monotonic,
        started_at_wall_time=recorder.recording_start_time,
    )
    reset_silero_vad_state(recorder)
    recorder.is_recording = True
    recorder.is_webrtc_speech_active = False
    recorder.stop_recording_event.clear()
    recorder.start_recording_event.set()

    if recorder.on_recording_start:
        run_callback(recorder, recorder.on_recording_start)

    return recorder


def stop_recording(
        recorder,
        backdate_stop_seconds=0.0,
        backdate_resume_seconds=0.0,
):
    # Ensure there's a minimum interval
    # between starting and stopping recording
    if (time.time() - recorder.recording_start_time
            < recorder.min_length_of_recording):
        logger.info("Attempted to stop recording "
                    "too soon after starting."
                    )
        return recorder

    logger.info("recording stopped")
    stopped_frames = copy.deepcopy(recorder.frames)
    recorder.last_frames = copy.deepcopy(stopped_frames)
    recorder.backdate_stop_seconds = backdate_stop_seconds
    recorder.backdate_resume_seconds = backdate_resume_seconds
    queue_recorded_audio(
        recorder,
        stopped_frames,
        backdate_stop_seconds,
        backdate_resume_seconds,
    )
    recorder.frames = []
    recorder.is_recording = False
    recorder.recording_stop_time = time.time()
    realtime_text_stabilizer = getattr(
        recorder,
        "realtime_text_stabilizer",
        None,
    )
    if realtime_text_stabilizer is not None:
        realtime_text_stabilizer.finalize()
    reset_silero_vad_state(recorder)
    recorder.is_webrtc_speech_active = False
    recorder.silero_check_time = 0
    recorder.start_recording_event.clear()
    recorder.stop_recording_event.set()

    recorder.last_recording_start_time = recorder.recording_start_time
    recorder.last_recording_stop_time = recorder.recording_stop_time

    if recorder.on_recording_stop:
        run_callback(recorder, recorder.on_recording_stop)

    return recorder


def listen_for_voice_activity(recorder):
    recorder.listen_start = time.time()
    set_recorder_state(recorder, "listening")
    reset_silero_vad_state(recorder)
    recorder.start_recording_on_voice_activity = True


def wait_for_recorded_audio(recorder):
    armed_for_voice_activity = False

    try:
        logger.info("Setting listen time")
        if recorder.listen_start == 0:
            recorder.listen_start = time.time()

        queued_recording = get_next_recorded_audio(recorder)

        # If not yet started recording, wait for voice activity to initiate.
        if queued_recording is None and not recorder.is_recording and not recorder.frames:
            set_recorder_state(recorder, "listening")
            reset_silero_vad_state(recorder)
            recorder.start_recording_on_voice_activity = True
            armed_for_voice_activity = True

            # Wait until recording starts
            logger.debug('Waiting for recording start')
            while not recorder.interrupt_stop_event.is_set():
                if recorder.start_recording_event.wait(timeout=0.02):
                    break

        # If recording is ongoing, wait for voice inactivity
        # to finish recording.
        if queued_recording is None and recorder.is_recording:
            recorder.stop_recording_on_voice_deactivity = True

            # Wait until recording stops
            logger.debug('Waiting for recording stop')
            while not recorder.interrupt_stop_event.is_set():
                if (recorder.stop_recording_event.wait(timeout=0.02)):
                    break

        if queued_recording is None:
            queued_recording = get_next_recorded_audio(recorder)

        if queued_recording is not None:
            frames = queued_recording["frames"]
            backdate_stop_seconds = queued_recording["backdate_stop_seconds"]
            backdate_resume_seconds = queued_recording["backdate_resume_seconds"]
        else:
            frames = recorder.frames
            if len(frames) == 0:
                frames = recorder.last_frames
            backdate_stop_seconds = recorder.backdate_stop_seconds
            backdate_resume_seconds = recorder.backdate_resume_seconds

        frames_to_read = set_audio_from_frames(
            recorder,
            frames,
            backdate_stop_seconds,
            backdate_resume_seconds,
        )

        if not recorder.is_recording:
            recorder.frames.clear()
            recorder.last_frames.clear()
            recorder.frames.extend(frames_to_read)

        # Reset backdating parameters
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0

        recorder.listen_start = 0

        if not recorder.is_recording:
            set_recorder_state(recorder, "inactive")

        if (
                armed_for_voice_activity
                and not recorder.use_wake_words
                and not recorder.interrupt_stop_event.is_set()
                and not recorder.is_shut_down):
            recorder.continuous_listening = True
            reset_silero_vad_state(recorder)
            recorder.start_recording_on_voice_activity = True
            recorder.stop_recording_on_voice_deactivity = True

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt in wait_audio, shutting down")
        recorder.shutdown()
        raise  # Re-raise the exception after cleanup


def wakeup_recorder(recorder):
    recorder.listen_start = time.time()


def abort_recording(recorder):
    state = recorder.state
    recorder.start_recording_on_voice_activity = False
    recorder.stop_recording_on_voice_deactivity = False
    recorder.interrupt_stop_event.set()
    if recorder.state != "inactive": # if inactive, was_interrupted will never be set
        recorder.was_interrupted.wait()
        set_recorder_state(recorder, "transcribing")
    recorder.was_interrupted.clear()
    if recorder.is_recording: # if recording, make sure to stop the recorder
        recorder.stop()
