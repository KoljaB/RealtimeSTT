import json
import os
import re
import time
import unittest
import wave
from pathlib import Path


try:
    import numpy as np
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
except ModuleNotFoundError as exc:
    np = None
    AudioToTextRecorder = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


AUDIO_DIR = Path(__file__).with_name("audio")
MANIFEST_PATH = AUDIO_DIR / "manifest.json"


def print_test_message(message):
    print(f"\n[RealtimeSTT test] {message}")


class FakeAudioQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_wav_samples(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1:
        raise ValueError(f"{path.name} must be mono for these tests")
    if sample_width != 2:
        raise ValueError(f"{path.name} must be 16-bit PCM for these tests")

    return np.frombuffer(frames, dtype=np.int16), sample_rate


def normalize_transcript(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class AudioFixtureTests(unittest.TestCase):
    def test_manifest_points_to_valid_wav_and_transcript_files(self):
        manifest = load_manifest()

        self.assertEqual(manifest["dataset"], "LJ Speech 1.1")
        self.assertEqual(manifest["license"], "Public Domain")
        self.assertGreaterEqual(len(manifest["samples"]), 5)

        for sample in manifest["samples"]:
            with self.subTest(sample=sample["id"]):
                wav_path = AUDIO_DIR / sample["file"]
                transcript_path = AUDIO_DIR / sample["transcript_file"]
                print_test_message(
                    f"fixture {sample['id']} expected transcript: "
                    f"{sample['normalized_transcript']}"
                )

                self.assertTrue(wav_path.exists(), wav_path)
                self.assertTrue(transcript_path.exists(), transcript_path)
                self.assertEqual(
                    transcript_path.read_text(encoding="utf-8").strip(),
                    sample["normalized_transcript"],
                )

                with wave.open(str(wav_path), "rb") as wav:
                    self.assertEqual(wav.getnchannels(), 1)
                    self.assertEqual(wav.getsampwidth(), 2)
                    self.assertEqual(wav.getframerate(), manifest["sample_rate_hz"])
                    self.assertGreater(wav.getnframes(), 0)


class FeedAudioTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"RealtimeSTT import failed: {IMPORT_ERROR}")

    def make_recorder_stub(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.buffer_size = 512
        recorder.audio_queue = FakeAudioQueue()
        return recorder

    def test_feed_audio_chunks_raw_pcm_bytes(self):
        recorder = self.make_recorder_stub()
        chunk_size_bytes = 2 * recorder.buffer_size
        raw_pcm = (np.arange(1536, dtype=np.int16)).tobytes()

        recorder.feed_audio(raw_pcm[:600])
        self.assertEqual(recorder.audio_queue.items, [])

        recorder.feed_audio(raw_pcm[600:])

        self.assertEqual(len(recorder.audio_queue.items), 3)
        self.assertTrue(
            all(len(chunk) == chunk_size_bytes for chunk in recorder.audio_queue.items)
        )
        self.assertEqual(len(recorder.buffer), 0)

    def test_feed_audio_resamples_ljspeech_numpy_audio(self):
        manifest = load_manifest()
        sample = manifest["samples"][0]
        samples, sample_rate = read_wav_samples(AUDIO_DIR / sample["file"])
        recorder = self.make_recorder_stub()

        input_step = 257
        expected_resampled_bytes = 0

        for start in range(0, len(samples), input_step):
            audio_chunk = samples[start:start + input_step]
            expected_resampled_bytes += int(len(audio_chunk) * 16000 / sample_rate) * 2
            recorder.feed_audio(audio_chunk, original_sample_rate=sample_rate)

        chunk_size_bytes = 2 * recorder.buffer_size
        self.assertEqual(
            len(recorder.audio_queue.items),
            expected_resampled_bytes // chunk_size_bytes,
        )
        self.assertEqual(len(recorder.buffer), expected_resampled_bytes % chunk_size_bytes)
        self.assertTrue(
            all(len(chunk) == chunk_size_bytes for chunk in recorder.audio_queue.items)
        )


class GoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"RealtimeSTT import failed: {IMPORT_ERROR}")
        if os.environ.get("REALTIMESTT_RUN_GOLDEN_TRANSCRIPTION") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_GOLDEN_TRANSCRIPTION=1 to run the "
                "slow real-model transcription test"
            )

    def test_transcribes_ljspeech_sample_and_prints_result(self):
        manifest = load_manifest()
        sample = manifest["samples"][0]
        samples, sample_rate = read_wav_samples(AUDIO_DIR / sample["file"])
        expected = sample["normalized_transcript"]

        recorder = AudioToTextRecorder(
            use_microphone=False,
            spinner=False,
            model=os.environ.get("REALTIMESTT_TEST_MODEL", "tiny"),
            language="en",
            device=os.environ.get("REALTIMESTT_TEST_DEVICE", "cpu"),
            compute_type=os.environ.get("REALTIMESTT_TEST_COMPUTE_TYPE", "int8"),
            min_length_of_recording=0,
            min_gap_between_recordings=0,
            no_log_file=True,
        )

        try:
            recorder.start()
            for start in range(0, len(samples), 1024):
                recorder.feed_audio(samples[start:start + 1024], original_sample_rate=sample_rate)

            deadline = time.time() + 5
            while len(recorder.frames) == 0 and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(0.2)

            recorder.stop()
            actual = recorder.text()

            print_test_message(f"golden {sample['id']} expected: {expected}")
            print_test_message(f"golden {sample['id']} actual:   {actual}")

            self.assertTrue(actual.strip())
            self.assertIn(
                " ".join(normalize_transcript(expected).split()[:2]),
                normalize_transcript(actual),
            )
        finally:
            recorder.shutdown()


if __name__ == "__main__":
    unittest.main()
