import queue
import threading
import time
import unittest
from unittest import mock

import numpy as np

from RealtimeSTT.core.tail_transcription import (
    FINAL_TRANSCRIPTION_TAIL_SECONDS,
    MIN_LIVE_WORDS_FOR_FUZZY_REPAIR,
    append_pcm16_tail,
    extract_audio_tail,
    find_tail_anchor,
    merge_live_and_tail_transcription,
    snapshot_pcm16_tail,
)
from RealtimeSTT.core.recording import _submit_preview_transcription_at_silence
from RealtimeSTT.core.transcription_api import perform_final_transcription
from RealtimeSTT.transcription_engines import (
    TranscriptionInfo,
    TranscriptionResult,
)


class RecorderStub:
    def __init__(self):
        self.transcription_lock = threading.RLock()
        self.audio = np.ones(8, dtype=np.float32)
        self.transcribe_count = 0
        self.language = "en"
        self.interrupt_stop_event = threading.Event()
        self.was_interrupted = threading.Event()
        self.is_recording = False
        self.state = "recording"
        self.on_vad_detect_stop = None
        self.on_wakeword_detection_end = None
        self.spinner = False
        self.halo = None
        self.allowed_to_early_transcribe = False
        self.detected_language = None
        self.detected_language_probability = 0.0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.last_transcription_metadata = None
        self.ensure_sentence_starting_uppercase = False
        self.ensure_sentence_ends_with_period = False
        self._current_transcription_force_lowercase_start = False
        self.print_transcription_time = False
        self.main_model_type = "fake"


class TailAwareExecutor:
    def __init__(self, full_text, tail_text):
        self.full_text = full_text
        self.tail_text = tail_text
        self.calls = []
        self.work_samples = 0

    def transcribe(self, audio, language=None, use_prompt=True, **_options):
        sample_count = int(audio.size)
        self.calls.append(sample_count)
        self.work_samples += sample_count
        text = self.tail_text if sample_count == 3 * 16000 else self.full_text
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )


class LatencyAwareTailExecutor(TailAwareExecutor):
    """
    Deterministic executor used to compare the pre- and post-tail paths.

    The sleep models inference cost from the amount of audio submitted. The
    assertion uses the submitted audio duration as the stable measurement;
    wall-clock time is recorded only as a diagnostic.
    """

    def __init__(self, full_text, tail_text, seconds_per_audio_second=0.01):
        super().__init__(full_text, tail_text)
        self.seconds_per_audio_second = seconds_per_audio_second
        self.inference_seconds = 0.0

    def transcribe(self, audio, language=None, use_prompt=True, **options):
        audio_seconds = float(audio.size) / 16000.0
        self.inference_seconds += audio_seconds * self.seconds_per_audio_second
        time.sleep(audio_seconds * self.seconds_per_audio_second)
        return super().transcribe(audio, language, use_prompt, **options)


def make_executor_recorder(executor, audio_seconds=20):
    recorder = RecorderStub()
    recorder._uses_external_transcription_executor = True
    recorder.transcription_executor = executor
    recorder._external_transcription_results = queue.Queue()
    recorder._external_transcription_threads = []
    recorder.enable_realtime_transcription = True
    recorder.realtime_transcription_text = "The quick brown fox jumps"
    recorder.sample_rate = 16000
    recorder.audio = np.arange(
        audio_seconds * recorder.sample_rate,
        dtype=np.float32,
    )
    return recorder


class TailTranscriptionHelperTests(unittest.TestCase):
    def test_exact_anchor_discards_tail_prefix_and_appends_only_suffix(self):
        result = merge_live_and_tail_transcription(
            "The quick brown fox jumps",
            "hallucinated prefix brown fox jumps over the fence",
        )

        self.assertTrue(result.matched)
        self.assertFalse(result.used_fuzzy_match)
        self.assertEqual(
            result.text,
            "The quick brown fox jumps over the fence",
        )

    def test_four_word_anchor_is_preferred_when_available(self):
        result = merge_live_and_tail_transcription(
            "The quick brown fox jumps over",
            "noise brown fox jumps over the fence",
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.anchor_length, 4)
        self.assertEqual(result.text, "The quick brown fox jumps over the fence")

    def test_fuzzy_anchor_handles_one_word_difference(self):
        result = merge_live_and_tail_transcription(
            "Please open the settings panel",
            "noise the setting panel now",
        )

        self.assertTrue(result.matched)
        self.assertTrue(result.used_fuzzy_match)
        self.assertEqual(result.text, "Please open the settings panel now")

    def test_fuzzy_anchor_replaces_incomplete_live_word(self):
        result = merge_live_and_tail_transcription(
            "First summarize what happened then give me three concre",
            "noise give me three concrete next steps",
        )

        self.assertTrue(result.matched)
        self.assertEqual(
            result.text,
            "First summarize what happened then give me three concrete next steps",
        )

    def test_last_live_word_is_replaced_when_it_is_complete_but_wrong(self):
        result = merge_live_and_tail_transcription(
            "First summarize what happened then give me three bananas",
            "noise give me three concrete next steps",
        )

        self.assertTrue(result.matched)
        self.assertEqual(
            result.text,
            "First summarize what happened then give me three concrete next steps",
        )

    def test_fuzzy_anchor_replaces_short_incomplete_live_word(self):
        result = merge_live_and_tail_transcription(
            "Change the plan to focus on la",
            "noise to focus on latency first",
        )

        self.assertTrue(result.matched)
        self.assertEqual(
            result.text,
            "Change the plan to focus on latency first",
        )

    def test_exact_anchor_repairs_partial_word_immediately_before_overlap(self):
        result = merge_live_and_tail_transcription(
            "And I feel now maybe I have a kind of nice working spee to text system",
            "speech to text system.",
        )

        self.assertTrue(result.matched)
        self.assertEqual(
            result.text,
            "And I feel now maybe I have a kind of nice working speech to text system.",
        )

    def test_stable_anchor_repairs_partial_boundary_and_drops_unstable_suffix(self):
        result = merge_live_and_tail_transcription(
            "And I feel now maybe I have a kind of nice working spee to text system spee",
            "speech to text system.",
        )

        self.assertTrue(result.matched)
        self.assertEqual(
            result.text,
            "And I feel now maybe I have a kind of nice working speech to text system.",
        )

    def test_fuzzy_anchor_repairs_partial_word_inside_overlap(self):
        result = merge_live_and_tail_transcription(
            "Please save this answer as a short no for tomorrow",
            "This answer as a short note for tomorrow morning.",
        )

        self.assertTrue(result.matched)
        self.assertTrue(result.used_fuzzy_match)
        self.assertEqual(
            result.text,
            "Please save this answer as a short note for tomorrow morning.",
        )

    def test_fuzzy_anchor_rejects_unrelated_word_substitution(self):
        result = merge_live_and_tail_transcription(
            "Please open the settings panel",
            "noise the banana panel now",
        )

        self.assertFalse(result.matched)

    def test_fuzzy_anchor_allows_one_inserted_final_word(self):
        result = merge_live_and_tail_transcription(
            "Please pause for a moment and continue recording",
            "noise pause for a moment and then continue recording",
        )

        self.assertTrue(result.matched)
        self.assertTrue(result.used_fuzzy_match)
        self.assertEqual(result.distance, 1)
        self.assertEqual(
            result.text,
            "Please pause for a moment and then continue recording",
        )

    def test_no_anchor_match_reports_safe_fallback(self):
        result = merge_live_and_tail_transcription(
            "The original live transcript",
            "unrelated final tail words",
        )

        self.assertFalse(result.matched)
        self.assertEqual(result.text, "The original live transcript")

    def test_short_live_transcript_does_not_use_weak_anchor(self):
        result = merge_live_and_tail_transcription(
            "hello world",
            "hello world again",
        )

        self.assertFalse(result.matched)

    def test_tail_anchor_defaults_stay_conservative_and_realtime_can_relax(self):
        self.assertIsNone(find_tail_anchor("hello world", "hello world again"))

        exact = find_tail_anchor(
            "hello world",
            "hello world again",
            min_anchor_words=2,
        )
        self.assertIsNotNone(exact)
        self.assertEqual(exact[2], 2)

        fuzzy = find_tail_anchor(
            "we cat",
            "we bat today",
            min_anchor_words=2,
            min_close_word_length=3,
        )
        self.assertIsNotNone(fuzzy)
        self.assertEqual(fuzzy[2], 2)
        self.assertTrue(fuzzy[3])

    def test_fuzzy_repair_default_allows_three_live_words(self):
        self.assertEqual(MIN_LIVE_WORDS_FOR_FUZZY_REPAIR, 3)

        result = merge_live_and_tail_transcription(
            "focus on la",
            "boundary focus on latency first",
        )

        self.assertTrue(result.matched)
        self.assertEqual(result.text, "focus on latency first")

    def test_fuzzy_repair_minimum_is_configurable(self):
        result = merge_live_and_tail_transcription(
            "focus on la",
            "boundary focus on latency first",
            min_live_words_for_fuzzy_repair=4,
        )

        self.assertFalse(result.matched)

    def test_pcm16_tail_keeps_exactly_configured_recent_samples(self):
        recorder = type("Recorder", (), {"sample_rate": 10})()
        recorder.active_speech_tail_buffer = bytearray()

        append_pcm16_tail(recorder, np.arange(40, dtype=np.int16).tobytes(), seconds=3)
        append_pcm16_tail(recorder, np.arange(40, 80, dtype=np.int16).tobytes(), seconds=3)

        retained = np.frombuffer(snapshot_pcm16_tail(recorder), dtype=np.int16)
        self.assertEqual(FINAL_TRANSCRIPTION_TAIL_SECONDS, 3.0)
        self.assertEqual(retained.tolist(), list(range(50, 80)))

    def test_audio_tail_is_limited_to_three_seconds(self):
        audio = np.arange(12 * 16000, dtype=np.float32)

        tail = extract_audio_tail(audio, sample_rate=16000)

        self.assertEqual(tail.size, 3 * 16000)
        np.testing.assert_array_equal(tail, audio[-3 * 16000:])


class TailTranscriptionApiTests(unittest.TestCase):
    def test_final_transcription_always_uses_full_utterance(self):
        """
        Final ASR remains authoritative and must receive the full utterance.

        Preview owns the bounded tail path; Final must not silently switch to
        a tail-only request or invoke a second full request after alignment.
        """

        full_text = "The quick brown fox jumps over the fence"
        tail_text = "boundary garbage brown fox jumps over the fence"

        before_executor = LatencyAwareTailExecutor(full_text, tail_text)
        before_recorder = RecorderStub()
        before_recorder._uses_external_transcription_executor = True
        before_recorder.transcription_executor = before_executor
        before_recorder._external_transcription_results = queue.Queue()
        before_recorder._external_transcription_threads = []
        before_recorder.sample_rate = 16000
        before_recorder.audio = np.ones(20 * 16000, dtype=np.float32)

        before_started = time.perf_counter()
        before_text = perform_final_transcription(before_recorder)
        before_wall_seconds = time.perf_counter() - before_started

        after_executor = LatencyAwareTailExecutor(full_text, tail_text)
        after_recorder = make_executor_recorder(after_executor, audio_seconds=20)

        after_started = time.perf_counter()
        after_text = perform_final_transcription(after_recorder)
        after_wall_seconds = time.perf_counter() - after_started

        self.assertEqual(before_text, after_text)
        self.assertEqual(before_text, full_text)
        self.assertEqual(before_executor.calls, [20 * 16000])
        self.assertEqual(after_executor.calls, [20 * 16000])
        self.assertEqual(before_executor.work_samples, 20 * 16000)
        self.assertEqual(after_executor.work_samples, 20 * 16000)
        self.assertAlmostEqual(before_executor.inference_seconds, 0.2)
        self.assertAlmostEqual(after_executor.inference_seconds, 0.2)

        # Wall-clock values are diagnostic only; the submitted-duration
        # assertions above are the deterministic regression gate.
        self.assertGreaterEqual(after_wall_seconds, 0.0)
        self.assertGreaterEqual(before_wall_seconds, 0.0)

    def test_confirmed_silence_submits_preview_with_only_the_rolling_tail(self):
        recorder = type("Recorder", (), {})()
        recorder.enable_preview_transcription = True
        recorder._preview_transcription_submitted = False
        recorder.realtime_transcription_text = "The quick brown fox jumps"
        recorder.realtime_stabilized_text = ""
        recorder.realtime_stabilized_safetext = ""
        recorder.realtime_recording_id = 1
        recorder.sample_rate = 16000
        recorder.language = "en"
        recorder.active_speech_tail_buffer = bytearray()
        append_pcm16_tail(
            recorder,
            np.arange(12 * 16000, dtype=np.int16).tobytes(),
        )

        with mock.patch(
            "RealtimeSTT.core.recording.submit_preview_transcription_request"
        ) as submit:
            submit.return_value = True
            self.assertTrue(_submit_preview_transcription_at_silence(recorder))

        submit.assert_called_once()
        submitted_audio = submit.call_args.args[1]
        self.assertEqual(submitted_audio.size, 3 * 16000)
        self.assertEqual(submit.call_args.args[2], recorder.realtime_transcription_text)

    def test_final_transcription_preserves_exact_expected_transcript(self):
        full_text = "The quick brown fox jumps over the fence"
        executor = TailAwareExecutor(
            full_text=full_text,
            tail_text="garbage brown fox jumps over the fence",
        )
        recorder = make_executor_recorder(executor)

        text = perform_final_transcription(recorder)

        self.assertEqual(text, full_text)
        self.assertEqual(executor.calls, [20 * 16000])

    def test_preview_does_not_reduce_authoritative_final_model_work(self):
        full_text = "The quick brown fox jumps over the fence"

        full_executor = TailAwareExecutor(full_text, full_text)
        full_recorder = RecorderStub()
        full_recorder._uses_external_transcription_executor = True
        full_recorder.transcription_executor = full_executor
        full_recorder._external_transcription_results = queue.Queue()
        full_recorder._external_transcription_threads = []
        full_recorder.sample_rate = 16000
        full_recorder.audio = np.ones(20 * 16000, dtype=np.float32)
        perform_final_transcription(full_recorder)

        tail_executor = TailAwareExecutor(
            full_text=full_text,
            tail_text="garbage brown fox jumps over the fence",
        )
        tail_recorder = make_executor_recorder(tail_executor)
        perform_final_transcription(tail_recorder)

        self.assertEqual(full_executor.work_samples, 20 * 16000)
        self.assertEqual(tail_executor.work_samples, 20 * 16000)

    def test_final_transcription_does_not_use_preview_alignment_or_fallback(self):
        full_text = "The original live transcript is complete"
        executor = TailAwareExecutor(
            full_text=full_text,
            tail_text="unrelated final tail words",
        )
        recorder = make_executor_recorder(executor)
        recorder.realtime_transcription_text = "The original live transcript"

        text = perform_final_transcription(recorder)

        self.assertEqual(text, full_text)
        self.assertEqual(executor.calls, [20 * 16000])

    def test_without_live_asr_existing_full_final_text_and_input_are_unchanged(self):
        full_text = "The complete final transcript"
        executor = TailAwareExecutor(full_text, "should not be used")
        recorder = RecorderStub()
        recorder._uses_external_transcription_executor = True
        recorder.transcription_executor = executor
        recorder._external_transcription_results = queue.Queue()
        recorder._external_transcription_threads = []
        recorder.sample_rate = 16000
        recorder.audio = np.ones(20 * 16000, dtype=np.float32)

        text = perform_final_transcription(recorder)

        self.assertEqual(text, full_text)
        self.assertEqual(executor.calls, [20 * 16000])


if __name__ == "__main__":
    unittest.main()
