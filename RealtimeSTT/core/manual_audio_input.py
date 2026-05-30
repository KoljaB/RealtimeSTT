"""Internal manual audio input helpers."""

from scipy.signal import resample
import numpy as np


def feed_audio(recorder, chunk, original_sample_rate=16000):
    # Check if the buffer attribute exists, if not, initialize it
    if not hasattr(recorder, 'buffer'):
        recorder.buffer = bytearray()

    # Check if input is a NumPy array
    if isinstance(chunk, np.ndarray):
        # Handle stereo to mono conversion if necessary
        if chunk.ndim == 2:
            chunk = np.mean(chunk, axis=1)

        # Resample to 16000 Hz if necessary
        if original_sample_rate != 16000:
            num_samples = int(len(chunk) * 16000 / original_sample_rate)
            chunk = resample(chunk, num_samples)

        # Ensure data type is int16
        chunk = chunk.astype(np.int16)

        # Convert the NumPy array to bytes
        chunk = chunk.tobytes()

    # Append the chunk to the buffer
    recorder.buffer += chunk
    buf_size = 2 * recorder.buffer_size  # silero complains if too short

    # Check if the buffer has reached or exceeded the buffer_size
    while len(recorder.buffer) >= buf_size:
        # Extract recorder.buffer_size amount of data from the buffer
        to_process = recorder.buffer[:buf_size]
        recorder.buffer = recorder.buffer[buf_size:]

        # Feed the extracted data to the audio_queue
        recorder.audio_queue.put(to_process)
