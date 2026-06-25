"""
Internal manual audio input helpers.
"""

from pathlib import Path
import os
import time

from scipy.signal import resample
import numpy as np
import soundfile as sf


INT16_MAX_ABS_VALUE = 32767.0
TARGET_SAMPLE_RATE = 16000


def _numpy_audio_to_pcm16_bytes(chunk, original_sample_rate):
    """
    Converts mono/stereo NumPy audio to 16 kHz mono int16 PCM bytes.
    """
    original_is_float = np.issubdtype(chunk.dtype, np.floating)

    if chunk.ndim == 2:
        chunk = np.mean(chunk, axis=1)
    elif chunk.ndim > 2:
        raise ValueError("Audio chunk must be mono or stereo")

    if original_sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(chunk) * TARGET_SAMPLE_RATE / original_sample_rate)
        chunk = resample(chunk, num_samples)

    if original_is_float:
        chunk = np.clip(chunk, -1.0, 1.0) * INT16_MAX_ABS_VALUE
    else:
        chunk = np.clip(chunk, -32768, 32767)

    return chunk.astype(np.int16).tobytes()


def _normalize_audio_peak(audio, target_peak=0.95):
    """
    Peak-normalizes floating audio without changing silence.
    """
    target_peak = float(target_peak)
    if target_peak <= 0 or target_peak > 1:
        raise ValueError("target_peak must be greater than 0 and at most 1")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio
    return np.clip(audio * (target_peak / peak), -1.0, 1.0)


def _validate_audio_file(filename):
    """
    Returns a readable audio file path or raises a specific validation error.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Audio file is not readable: {path}")
    return path


def _read_audio_file(filename):
    """
    Decodes an audio file to floating samples and returns samples plus sample rate.
    """
    path = _validate_audio_file(filename)
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"Audio file cannot be decoded as PCM audio: {path}") from exc

    if audio is None or np.asarray(audio).size == 0:
        raise ValueError(f"Audio file contains no audio samples: {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _iter_pcm_chunks(pcm_bytes, chunk_size_bytes):
    """
    Yields processable PCM chunks, padding the final short chunk with silence.
    """
    if not pcm_bytes:
        return

    for start in range(0, len(pcm_bytes), chunk_size_bytes):
        chunk = pcm_bytes[start:start + chunk_size_bytes]
        actual_samples = len(chunk) // 2
        if len(chunk) < chunk_size_bytes:
            chunk = chunk + (b"\x00" * (chunk_size_bytes - len(chunk)))
        yield chunk, actual_samples


def _coerce_audio_filenames(filenames):
    """
    Normalizes a single path or iterable of paths into a tuple.
    """
    if isinstance(filenames, (str, os.PathLike)):
        return (filenames,)
    try:
        return tuple(filenames)
    except TypeError as exc:
        raise TypeError("filenames must be a path or an iterable of paths") from exc


def feed_audio_file(
        recorder,
        filenames,
        normalize=True,
        target_peak=0.95,
):
    """
    Decodes audio file(s) and feeds them through feed_audio in realtime.
    """
    chunk_size_bytes = 2 * recorder.buffer_size

    for filename in _coerce_audio_filenames(filenames):
        audio, sample_rate = _read_audio_file(filename)
        if normalize:
            audio = _normalize_audio_peak(audio, target_peak=target_peak)

        pcm_bytes = _numpy_audio_to_pcm16_bytes(audio, sample_rate)
        playback_started_at = time.monotonic()
        scheduled_elapsed = 0.0
        for chunk, actual_samples in _iter_pcm_chunks(pcm_bytes, chunk_size_bytes):
            feed_audio(recorder, chunk, TARGET_SAMPLE_RATE)
            if actual_samples > 0:
                scheduled_elapsed += actual_samples / float(TARGET_SAMPLE_RATE)
                sleep_seconds = playback_started_at + scheduled_elapsed - time.monotonic()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)


def feed_audio(recorder, chunk, original_sample_rate=16000):
    """
    Buffers manually supplied audio and queues complete Silero-sized chunks.
    """
    if isinstance(chunk, (str, os.PathLike)):
        return feed_audio_file(recorder, chunk)

    if not hasattr(recorder, 'buffer'):
        recorder.buffer = bytearray()

    if isinstance(chunk, np.ndarray):
        chunk = _numpy_audio_to_pcm16_bytes(chunk, original_sample_rate)

    recorder.buffer += chunk
    buf_size = 2 * recorder.buffer_size  # silero complains if too short

    while len(recorder.buffer) >= buf_size:
        to_process = recorder.buffer[:buf_size]
        recorder.buffer = recorder.buffer[buf_size:]

        recorder.audio_queue.put(to_process)
