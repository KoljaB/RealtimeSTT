import collections
import queue
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
    from RealtimeSTT.core.recording_buffers import get_next_recorded_audio
    from RealtimeSTT.core.voice_activity import (
        check_voice_activity,
        is_silero_speech,
        is_voice_active,
    )
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    get_next_recorded_audio = None
    check_voice_activity = None
    is_silero_speech = None
    is_voice_active = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


AUDIO_DIR = Path(__file__).with_name("audio")
REFERENCE_AUDIO = AUDIO_DIR / "asr-reference.wav"


def read_wav_samples(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1:
        raise ValueError(f"{path.name} must be mono for this test")
    if sample_width != 2:
        raise ValueError(f"{path.name} must be 16-bit PCM for this test")

    return np.frombuffer(frames, dtype=np.int16), sample_rate


def chunk_samples(samples, chunk_samples_count):
    usable = len(samples) - (len(samples) % chunk_samples_count)
    for start in range(0, usable, chunk_samples_count):
        yield samples[start:start + chunk_samples_count].tobytes()


class FakeSileroVad:
    def __init__(self):
        self.reset_count = 0
        self.call_count = 0

    def reset_states(self):
        self.reset_count += 1

    def __call__(self, *_args, **_kwargs):
        self.call_count += 1

        class Result:
            def item(self):
                return 1.0

        return Result()


class SlowFinalTranscriptionAudioGapReproTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"RealtimeSTT import failed: {IMPORT_ERROR}")

    def test_unarmed_recorder_keeps_only_last_pre_recording_window(self):
        samples, sample_rate = read_wav_samples(REFERENCE_AUDIO)
        buffer_size = 512
        pre_recording_buffer_duration = 1.0

        audio_buffer = collections.deque(
            maxlen=int((sample_rate // buffer_size) * pre_recording_buffer_duration)
        )

        simulated_block_seconds = 2.5
        simulated_samples = samples[:int(sample_rate * simulated_block_seconds)]
        chunks = list(chunk_samples(simulated_samples, buffer_size))

        for chunk in chunks:
            audio_buffer.append(chunk)

        retained_audio = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
        retained_start_sample = len(simulated_samples[:len(chunks) * buffer_size]) - len(retained_audio)

        self.assertGreater(retained_start_sample, 0)
        self.assertLessEqual(len(audio_buffer), audio_buffer.maxlen)
        self.assertAlmostEqual(
            len(retained_audio) / sample_rate,
            pre_recording_buffer_duration,
            delta=0.05,
        )

    def test_stopped_recording_is_queued_beyond_pre_recording_window(self):
        """Completed recordings are retained while final transcription blocks.

        This guards the slow-CPU path where the application is blocked in
        final transcription while the recorder worker completes another
        utterance in the background.
        """

        samples, sample_rate = read_wav_samples(REFERENCE_AUDIO)
        buffer_size = 512
        simulated_block_seconds = 2.5
        simulated_samples = samples[:int(sample_rate * simulated_block_seconds)]
        chunks = list(chunk_samples(simulated_samples, buffer_size))

        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.frames = chunks.copy()
        recorder.last_frames = []
        recorder.recorded_audio_queue = queue.Queue()
        recorder.recording_start_time = time.time() - 10
        recorder.min_length_of_recording = 0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.is_recording = True
        recorder.is_silero_speech_active = True
        recorder.is_webrtc_speech_active = True
        recorder.silero_check_time = 0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.on_recording_stop = None

        recorder.stop()

        queued_recording = get_next_recorded_audio(recorder)
        retained_audio = np.frombuffer(
            b"".join(queued_recording["frames"]),
            dtype=np.int16,
        )

        self.assertFalse(recorder.frames)
        np.testing.assert_array_equal(
            retained_audio,
            simulated_samples[:len(retained_audio)],
        )

    def test_wait_audio_consumes_queued_recording(self):
        samples, sample_rate = read_wav_samples(REFERENCE_AUDIO)
        buffer_size = 512
        simulated_samples = samples[:sample_rate]
        chunks = list(chunk_samples(simulated_samples, buffer_size))

        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.sample_rate = sample_rate
        recorder.frames = chunks.copy()
        recorder.last_frames = []
        recorder.audio = None
        recorder.recorded_audio_queue = queue.Queue()
        recorder.recording_start_time = time.time() - 10
        recorder.min_length_of_recording = 0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.is_recording = True
        recorder.is_silero_speech_active = True
        recorder.is_webrtc_speech_active = True
        recorder.silero_check_time = 0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.interrupt_stop_event = threading.Event()
        recorder.listen_start = 0
        recorder.use_wake_words = False
        recorder.is_shut_down = False
        recorder.start_recording_on_voice_activity = False
        recorder.stop_recording_on_voice_deactivity = False
        recorder.continuous_listening = False
        recorder.on_recording_stop = None

        recorder.stop()
        with mock.patch("RealtimeSTT.core.lifecycle.set_recorder_state"):
            recorder.wait_audio()

        expected = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0

        self.assertFalse(recorder.has_pending_recordings())
        np.testing.assert_allclose(recorder.audio, expected)

    def test_voice_activity_allows_delayed_silero_confirmation(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.is_webrtc_speech_active = False
        recorder.is_silero_speech_active = True
        recorder.last_webrtc_speech_time = time.time() - 0.2

        self.assertTrue(is_voice_active(recorder))

        recorder.last_webrtc_speech_time = time.time() - 2.0

        self.assertFalse(is_voice_active(recorder))

    def test_start_resets_silero_vad_state(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        silero = FakeSileroVad()
        recorder.silero_vad_model = silero
        recorder.silero_vad_lock = threading.Lock()
        recorder._silero_vad_generation = 0
        recorder.is_silero_speech_active = True
        recorder.recording_stop_time = 0
        recorder.min_gap_between_recordings = 0
        recorder.text_storage = []
        recorder.realtime_stabilized_text = ""
        recorder.realtime_stabilized_safetext = ""
        recorder.realtime_text_stabilizer = None
        recorder.realtime_recording_id = 0
        recorder.wakeword_detected = False
        recorder.wake_word_detect_time = 0
        recorder.frames = []
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.on_recording_start = None

        with mock.patch("RealtimeSTT.core.lifecycle.set_recorder_state"):
            recorder.start()

        self.assertEqual(silero.reset_count, 1)
        self.assertFalse(recorder.is_silero_speech_active)
        self.assertEqual(recorder._silero_vad_generation, 1)

    def test_stop_resets_silero_vad_state(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        silero = FakeSileroVad()
        recorder.silero_vad_model = silero
        recorder.silero_vad_lock = threading.Lock()
        recorder._silero_vad_generation = 0
        recorder.is_silero_speech_active = True
        recorder.frames = []
        recorder.last_frames = []
        recorder.recorded_audio_queue = queue.Queue()
        recorder.recording_start_time = time.time() - 10
        recorder.min_length_of_recording = 0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.on_recording_stop = None
        recorder.realtime_text_stabilizer = None

        recorder.stop()

        self.assertEqual(silero.reset_count, 1)
        self.assertFalse(recorder.is_silero_speech_active)
        self.assertEqual(recorder._silero_vad_generation, 1)

    def test_new_webrtc_speech_island_resets_silero_before_async_check(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        silero = FakeSileroVad()
        recorder.silero_vad_model = silero
        recorder.silero_vad_lock = threading.Lock()
        recorder._silero_vad_generation = 7
        recorder.is_webrtc_speech_active = False
        recorder.is_silero_speech_active = True
        recorder.silero_working = False
        def mark_webrtc_speech(recorder_arg, data):
            setattr(recorder_arg, "is_webrtc_speech_active", True)

        with mock.patch(
            "RealtimeSTT.core.voice_activity.is_webrtc_speech",
            side_effect=mark_webrtc_speech,
        ):
            with mock.patch("RealtimeSTT.core.voice_activity.threading.Thread") as thread_cls:
                check_voice_activity(recorder, b"audio")

        self.assertEqual(silero.reset_count, 1)
        self.assertFalse(recorder.is_silero_speech_active)
        self.assertEqual(recorder._silero_vad_generation, 8)
        self.assertTrue(recorder.silero_working)
        thread_cls.assert_called_once()
        self.assertEqual(thread_cls.call_args.kwargs["args"], (recorder, b"audio", 8))

    def test_stale_silero_generation_does_not_update_detection_state(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        silero = FakeSileroVad()
        recorder.silero_vad_model = silero
        recorder.sample_rate = 16000
        recorder._silero_vad_generation = 2
        recorder.silero_working = False
        recorder.is_silero_speech_active = False

        result = is_silero_speech(
            recorder,
            np.zeros(512, dtype=np.int16).tobytes(),
            generation=1,
        )

        self.assertFalse(result)
        self.assertFalse(recorder.is_silero_speech_active)
        self.assertFalse(recorder.silero_working)
        self.assertEqual(silero.call_count, 0)

    def test_flush_buffered_audio_queues_non_silent_tail(self):
        samples, _ = read_wav_samples(REFERENCE_AUDIO)
        chunks = list(chunk_samples(samples[:4096], 512))

        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.is_recording = False
        recorder.audio_buffer = collections.deque(chunks)
        recorder.recorded_audio_queue = queue.Queue()

        self.assertTrue(recorder.flush_buffered_audio())
        self.assertTrue(recorder.audio_buffer == collections.deque())

        queued_recording = get_next_recorded_audio(recorder)
        retained_audio = np.frombuffer(
            b"".join(queued_recording["frames"]),
            dtype=np.int16,
        )

        np.testing.assert_array_equal(
            retained_audio,
            samples[:len(retained_audio)],
        )

    def test_flush_buffered_audio_ignores_silence(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.is_recording = False
        recorder.audio_buffer = collections.deque([
            np.zeros(512, dtype=np.int16).tobytes()
        ])
        recorder.recorded_audio_queue = queue.Queue()

        self.assertFalse(recorder.flush_buffered_audio())
        self.assertFalse(recorder.has_pending_recordings())


if __name__ == "__main__":
    unittest.main()
