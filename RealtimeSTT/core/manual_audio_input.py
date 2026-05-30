"""Internal manual audio input helpers."""

from scipy.signal import resample
import numpy as np


def feed_audio(recorder, chunk, original_sample_rate=16000):
    """
    Buffers manually supplied audio and queues complete Silero-sized chunks.
    """
    if not hasattr(recorder, 'buffer'):
        recorder.buffer = bytearray()

    if isinstance(chunk, np.ndarray):
        if chunk.ndim == 2:
            chunk = np.mean(chunk, axis=1)

        if original_sample_rate != 16000:
            num_samples = int(len(chunk) * 16000 / original_sample_rate)
            chunk = resample(chunk, num_samples)

        chunk = chunk.astype(np.int16)
        chunk = chunk.tobytes()

    recorder.buffer += chunk
    buf_size = 2 * recorder.buffer_size  # silero complains if too short

    while len(recorder.buffer) >= buf_size:
        to_process = recorder.buffer[:buf_size]
        recorder.buffer = recorder.buffer[buf_size:]

        recorder.audio_queue.put(to_process)
