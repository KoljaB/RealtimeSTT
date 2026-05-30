"""Internal recording buffer and queued-audio helpers."""

import copy
import logging
import queue

import numpy as np

from .voice_activity import clear_pre_recording_buffer


logger = logging.getLogger("realtimestt")

INT16_MAX_ABS_VALUE = 32768.0


def set_audio_from_frames(
        recorder,
        frames,
        backdate_stop_seconds=0.0,
        backdate_resume_seconds=0.0,
):
    frames = frames or []

    # Calculate samples needed for backdating resume
    samples_to_keep = int(recorder.sample_rate * backdate_resume_seconds)

    # First convert all current frames to audio array
    full_audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
    full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

    # Calculate how many samples we need to keep for backdating resume
    if samples_to_keep > 0:
        samples_to_keep = min(samples_to_keep, len(full_audio))
        # Keep the last N samples for backdating resume
        frames_to_read_audio = full_audio[-samples_to_keep:]

        # Convert the audio back to int16 bytes for frames
        frames_to_read_int16 = (frames_to_read_audio * INT16_MAX_ABS_VALUE).astype(np.int16)
        frame_bytes = frames_to_read_int16.tobytes()

        # Split into appropriate frame sizes (assuming standard frame size)
        FRAME_SIZE = 2048  # Typical frame size
        frames_to_read = []
        for i in range(0, len(frame_bytes), FRAME_SIZE):
            frame = frame_bytes[i:i + FRAME_SIZE]
            if frame:  # Only add non-empty frames
                frames_to_read.append(frame)
    else:
        frames_to_read = []

    # Process backdate stop seconds
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
):
    if not frames:
        return

    recorder.recorded_audio_queue.put({
        "frames": copy.deepcopy(frames),
        "backdate_stop_seconds": backdate_stop_seconds,
        "backdate_resume_seconds": backdate_resume_seconds,
    })


def get_next_recorded_audio(recorder):
    try:
        return recorder.recorded_audio_queue.get_nowait()
    except queue.Empty:
        return None


def has_pending_recordings(recorder):
    return not recorder.recorded_audio_queue.empty()


def flush_buffered_audio(recorder, min_abs_level=50):
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
    Safely empties the audio queue to ensure no remaining audio
    fragments get processed e.g. after waking up the recorder.
    """
    clear_pre_recording_buffer(recorder)
    try:
        while True:
            recorder.audio_queue.get_nowait()
    except:
        # PyTorch's mp.Queue doesn't have a specific Empty exception
        # so we catch any exception that might occur when the queue is empty
        pass
