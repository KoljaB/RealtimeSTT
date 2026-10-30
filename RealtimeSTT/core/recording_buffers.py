"""
Internal recording buffer and queued-audio helpers.
"""

import copy
import logging
import queue
import threading

import numpy as np

from .tail_transcription import (
    FINAL_TRANSCRIPTION_TAIL_SECONDS,
    append_pcm16_tail,
    pcm16_bytes_to_float_audio,
    snapshot_pcm16_tail,
)
from .voice_activity import clear_pre_recording_buffer


logger = logging.getLogger("realtimestt")

INT16_MAX_ABS_VALUE = 32768.0


def get_frames_lock(recorder):
    """
    Returns the shared lock guarding active recording frame buffers.
    """
    lock = getattr(recorder, "frames_lock", None)
    if lock is None:
        lock = threading.RLock()
        recorder.frames_lock = lock
    return lock


def snapshot_frames(recorder, attr_name="frames"):
    """
    Returns a stable tuple snapshot of one recorder frame list.
    """
    with get_frames_lock(recorder):
        return tuple(getattr(recorder, attr_name, None) or ())


def set_active_speech_tail_from_frames(recorder, frames):
    """
    Rebuilds the rolling tail from a frame sequence.
    """
    recorder.active_speech_tail_buffer = bytearray()
    for frame in frames or ():
        append_pcm16_tail(recorder, frame)


def snapshot_active_speech_tail_audio(recorder):
    """
    Returns the rolling PCM16 tail as normalized float audio.
    """
    return pcm16_bytes_to_float_audio(snapshot_pcm16_tail(recorder))


def tail_audio_from_frames(recorder, frames):
    """
    Returns a bounded float tail when a rolling buffer is unavailable.
    """
    recorder.active_speech_tail_buffer = bytearray()
    tail_seconds = getattr(
        recorder,
        "preview_transcription_tail_seconds",
        FINAL_TRANSCRIPTION_TAIL_SECONDS,
    )
    for frame in frames or ():
        append_pcm16_tail(recorder, frame, tail_seconds)
    return snapshot_active_speech_tail_audio(recorder)


def set_audio_from_frames(
        recorder,
        frames,
        backdate_stop_seconds=0.0,
        backdate_resume_seconds=0.0,
):
    """
    Stores recorded frames as float audio and keeps optional resume audio.
    """
    frames = frames or []

    samples_to_keep = int(recorder.sample_rate * backdate_resume_seconds)

    full_audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
    full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

    if samples_to_keep > 0:
        samples_to_keep = min(samples_to_keep, len(full_audio))
        frames_to_read_audio = full_audio[-samples_to_keep:]

        frames_to_read_int16 = (frames_to_read_audio * INT16_MAX_ABS_VALUE).astype(np.int16)
        frame_bytes = frames_to_read_int16.tobytes()

        FRAME_SIZE = 2048  # Historical recorder frame byte size.
        frames_to_read = []
        for i in range(0, len(frame_bytes), FRAME_SIZE):
            frame = frame_bytes[i:i + FRAME_SIZE]
            if frame:  # Only add non-empty frames
                frames_to_read.append(frame)
    else:
        frames_to_read = []

    samples_to_remove = int(recorder.sample_rate * backdate_stop_seconds)

    if samples_to_remove > 0:
        if samples_to_remove < len(full_audio):
            recorder.audio = full_audio[:-samples_to_remove]
            logger.debug(f"Removed {samples_to_remove} samples "
                f"({samples_to_remove/recorder.sample_rate:.3f}s) from end of audio")
        else:
            recorder.audio = np.array([], dtype=np.float32)
            logger.debug("Cleared audio (samples_to_remove >= audio length)")
    else:
        recorder.audio = full_audio
        logger.debug(f"No samples removed, final audio length: {len(recorder.audio)}")

    return frames_to_read


def queue_recorded_audio(
        recorder,
        frames,
        backdate_stop_seconds=0.0,
        backdate_resume_seconds=0.0,
        force_lowercase_start=None,
        tail_audio=None,
        live_text="",
):
    """
    Queues a completed recording for final transcription.
    """
    if not frames:
        return

    if tail_audio is None:
        tail_audio = tail_audio_from_frames(recorder, frames)
    elif isinstance(tail_audio, (bytes, bytearray, memoryview)):
        tail_audio = pcm16_bytes_to_float_audio(tail_audio)
    else:
        tail_audio = np.array(tail_audio, copy=True)

    recorder.recorded_audio_queue.put({
        "frames": copy.deepcopy(frames),
        "tail_audio": copy.deepcopy(tail_audio),
        "live_text": live_text or "",
        "backdate_stop_seconds": backdate_stop_seconds,
        "backdate_resume_seconds": backdate_resume_seconds,
        "force_lowercase_start": (
            getattr(recorder, "_force_current_recording_lowercase_start", False)
            if force_lowercase_start is None
            else force_lowercase_start
        ),
    })


def get_next_recorded_audio(recorder):
    """
    Returns the next queued recording, if one is available.
    """
    try:
        return recorder.recorded_audio_queue.get_nowait()
    except queue.Empty:
        return None


def has_pending_recordings(recorder):
    """
    Reports whether final-transcription audio is queued.
    """
    return not recorder.recorded_audio_queue.empty()


def flush_buffered_audio(recorder, min_abs_level=50):
    """
    Queues buffered audio when it contains enough non-silent signal.
    """
    if recorder.is_recording:
        recorder.stop()
        return True

    frames = list(recorder.audio_buffer)
    if not frames:
        return False

    audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
    if audio_array.size == 0:
        return False

    if np.max(np.abs(audio_array.astype(np.int32))) < min_abs_level:
        return False

    queue_recorded_audio(recorder, frames)
    clear_pre_recording_buffer(recorder)
    return True


def clear_audio_queue(recorder):
    """
    Empties queued audio fragments after recorder wakeup or reset.
    """
    clear_pre_recording_buffer(recorder)
    try:
        while True:
            recorder.audio_queue.get_nowait()
    except:
        # PyTorch's mp.Queue doesn't have a specific Empty exception
        # when the queue is empty.
        pass
