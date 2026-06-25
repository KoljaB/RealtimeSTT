from pathlib import Path
import tempfile
import unittest
import wave


try:
    import numpy as np

    from RealtimeSTT.core.manual_audio_input import feed_audio, feed_audio_file
except ModuleNotFoundError as exc:
    np = None
    feed_audio = None
    feed_audio_file = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FakeAudioQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class RecorderStub:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.audio_queue = FakeAudioQueue()


class ManualAudioInputTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"Manual audio input dependency missing: {IMPORT_ERROR}")

    def write_wav(self, path, samples, sample_rate=16000):
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.astype(np.int16).tobytes())

    def test_feed_audio_keeps_raw_pcm_bytes_unchanged(self):
        recorder = RecorderStub(buffer_size=5)
        raw_pcm = np.array([-1000, -1, 0, 1, 1000], dtype=np.int16).tobytes()

        feed_audio(recorder, raw_pcm)

        self.assertEqual(recorder.audio_queue.items, [raw_pcm])
        self.assertEqual(len(recorder.buffer), 0)

    def test_feed_audio_keeps_int16_numpy_samples_unchanged(self):
        recorder = RecorderStub(buffer_size=5)
        samples = np.array([-30000, -123, 0, 123, 30000], dtype=np.int16)

        feed_audio(recorder, samples)

        self.assertEqual(
            np.frombuffer(recorder.audio_queue.items[0], dtype=np.int16).tolist(),
            samples.tolist(),
        )
        self.assertEqual(len(recorder.buffer), 0)

    def test_feed_audio_keeps_partial_chunk_buffered(self):
        recorder = RecorderStub(buffer_size=5)
        first = np.array([1, 2], dtype=np.int16)
        second = np.array([3, 4, 5], dtype=np.int16)

        feed_audio(recorder, first)

        self.assertEqual(recorder.audio_queue.items, [])
        self.assertEqual(
            np.frombuffer(recorder.buffer, dtype=np.int16).tolist(),
            [1, 2],
        )

        feed_audio(recorder, second)

        self.assertEqual(
            np.frombuffer(recorder.audio_queue.items[0], dtype=np.int16).tolist(),
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(len(recorder.buffer), 0)

    def test_feed_audio_scales_normalized_float_numpy_samples_to_pcm16(self):
        recorder = RecorderStub(buffer_size=5)
        samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        feed_audio(recorder, samples)

        self.assertEqual(
            np.frombuffer(recorder.audio_queue.items[0], dtype=np.int16).tolist(),
            [-32767, -16383, 0, 16383, 32767],
        )

    def test_feed_audio_averages_stereo_float_before_pcm16_scaling(self):
        recorder = RecorderStub(buffer_size=5)
        samples = np.array(
            [
                [-1.0, 1.0],
                [-0.5, 0.0],
                [0.0, 0.0],
                [0.0, 0.5],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        feed_audio(recorder, samples)

        self.assertEqual(
            np.frombuffer(recorder.audio_queue.items[0], dtype=np.int16).tolist(),
            [0, -8191, 0, 8191, 32767],
        )

    def test_feed_audio_clips_normalized_float_numpy_samples(self):
        recorder = RecorderStub(buffer_size=3)
        samples = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

        feed_audio(recorder, samples)

        self.assertEqual(
            np.frombuffer(recorder.audio_queue.items[0], dtype=np.int16).tolist(),
            [-32767, 0, 32767],
        )

    def test_feed_audio_file_feeds_minimal_chunks_and_pads_tail(self):
        recorder = RecorderStub(buffer_size=4)
        samples = np.arange(10, dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "tiny.wav"
            self.write_wav(wav_path, samples)

            feed_audio_file(
                recorder,
                wav_path,
                normalize=False,
            )

        self.assertEqual(len(recorder.audio_queue.items), 3)
        self.assertTrue(
            all(
                len(chunk) == 2 * recorder.buffer_size
                for chunk in recorder.audio_queue.items
            )
        )
        self.assertEqual(len(recorder.buffer), 0)
        self.assertEqual(
            np.frombuffer(recorder.audio_queue.items[-1], dtype=np.int16)[-2:].tolist(),
            [0, 0],
        )

    def test_feed_audio_file_validates_missing_file(self):
        recorder = RecorderStub(buffer_size=4)

        with self.assertRaises(FileNotFoundError):
            feed_audio_file(
                recorder,
                "does-not-exist.wav",
            )

    def test_feed_audio_file_can_normalize_before_feeding(self):
        recorder = RecorderStub(buffer_size=4)
        samples = np.array([1000, -1000, 0, 0], dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "quiet.wav"
            self.write_wav(wav_path, samples)

            feed_audio_file(
                recorder,
                wav_path,
                normalize=True,
            )

        fed = np.frombuffer(recorder.audio_queue.items[0], dtype=np.int16)
        self.assertGreater(int(np.max(np.abs(fed))), 1000)

    def test_feed_audio_file_rejects_invalid_normalization_peak(self):
        recorder = RecorderStub(buffer_size=4)
        samples = np.array([1000, -1000, 0, 0], dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "quiet.wav"
            self.write_wav(wav_path, samples)

            with self.assertRaises(ValueError):
                feed_audio_file(
                    recorder,
                    wav_path,
                    normalize=True,
                    target_peak=0,
                )

    def test_feed_audio_accepts_audio_file_path(self):
        recorder = RecorderStub(buffer_size=4)
        samples = np.arange(4, dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "direct.wav"
            self.write_wav(wav_path, samples)

            feed_audio(recorder, wav_path)

        self.assertEqual(len(recorder.audio_queue.items), 1)
        self.assertEqual(len(recorder.buffer), 0)


if __name__ == "__main__":
    unittest.main()
