import threading
import unittest

from RealtimeSTT.core.realtime_text_stabilizer import (
    RealtimeTextFinalObservation,
    RealtimeTextObservation,
    RealtimeTextStabilizationConfig,
    RealtimeTextStabilizer,
)

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    AUDIO_RECORDER_IMPORT_ERROR = exc
else:
    AUDIO_RECORDER_IMPORT_ERROR = None


class RealtimeTextStabilizerTests(unittest.TestCase):
    def setUp(self):
        self.config = RealtimeTextStabilizationConfig()
        self.stabilizer = RealtimeTextStabilizer(self.config)
        self.stabilizer.reset("rec-1", segment_id="seg-1")

    def observation(
        self,
        text,
        sequence,
        completed_at=None,
        recording_id="rec-1",
        publish_allowed=True,
        audio_start_sample=0,
        audio_end_sample_exclusive=None,
        trigger_reason="timer",
    ):
        if completed_at is None:
            completed_at = sequence * 0.11
        if audio_end_sample_exclusive is None:
            audio_end_sample_exclusive = sequence * 1600
        return RealtimeTextObservation(
            recording_id=recording_id,
            segment_id="seg-1",
            sequence=sequence,
            raw_text=text,
            audio_start_sample=audio_start_sample,
            audio_end_sample_exclusive=audio_end_sample_exclusive,
            sample_rate=16000,
            created_at_monotonic=max(0.0, completed_at - 0.02),
            completed_at_monotonic=completed_at,
            trigger_reason=trigger_reason,
            publish_allowed=publish_allowed,
        )

    def observe(self, text, sequence, **kwargs):
        return self.stabilizer.observe(
            self.observation(text, sequence, **kwargs)
        )

    def test_first_second_third_observation_threshold_behavior(self):
        first = self.observe("hello", 1, completed_at=0.00)
        second = self.observe("hello", 2, completed_at=0.40)
        third = self.observe("hello", 3, completed_at=0.85)

        self.assertEqual(first.stable_delta, "")
        self.assertEqual(first.stable_text, "")
        self.assertEqual(first.unstable_text, "hello")
        self.assertEqual(first.display_text, "hello")

        self.assertEqual(second.stable_delta, "")
        self.assertEqual(second.stable_text, "")

        self.assertEqual(third.stable_delta, "hello")
        self.assertEqual(third.stable_text, "hello")
        self.assertEqual(third.unstable_text, "")
        self.assertTrue(third.has_new_stable_text)
        self.assertEqual(third.commit_reason, "evidence-threshold")
        self.assertEqual(third.evidence.confirmation_count, 3)
        self.assertEqual(third.evidence.contributing_sequence_ids, (1, 2, 3))

    def test_fast_repeats_do_not_stabilize_before_elapsed_evidence_span(self):
        self.observe("hello", 1, completed_at=0.00)
        self.observe("hello", 2, completed_at=0.05)
        early = self.observe("hello", 3, completed_at=0.10)
        later = self.observe("hello", 4, completed_at=0.85)

        self.assertEqual(early.stable_text, "")
        self.assertEqual(early.unstable_text, "hello")
        self.assertEqual(later.stable_delta, "hello")

    def test_repeats_without_audio_progress_do_not_add_stability_evidence(self):
        first = self.observe(
            "set the timer for six seconds",
            1,
            completed_at=0.00,
            audio_end_sample_exclusive=16000,
        )
        second = self.observe(
            "set the timer for six seconds",
            2,
            completed_at=0.40,
            audio_end_sample_exclusive=16000,
        )
        third = self.observe(
            "set the timer for six seconds",
            3,
            completed_at=0.85,
            audio_end_sample_exclusive=16000,
        )
        progressed = self.observe(
            "set the timer for six seconds",
            4,
            completed_at=1.30,
            audio_end_sample_exclusive=32000,
        )

        self.assertEqual(first.stable_text, "")
        self.assertEqual(second.stable_text, "")
        self.assertEqual(third.stable_text, "")
        self.assertEqual(progressed.stable_delta, "set")
        self.assertEqual(progressed.evidence.contributing_sequence_ids, (1, 4))

    def test_wrong_initial_prefix_is_not_committed_before_correction_arrives(self):
        events = [
            self.observe("M now talking", 1, completed_at=0.00),
            self.observe("M now talking", 2, completed_at=0.15),
            self.observe("M now talking into", 3, completed_at=0.30),
            self.observe("I am now talking into", 4, completed_at=0.45),
            self.observe("I am now talking into", 5, completed_at=0.85),
            self.observe("I am now talking into", 6, completed_at=1.25),
            self.observe("I am now talking into", 7, completed_at=1.65),
        ]

        self.assertEqual(events[2].stable_text, "")
        self.assertTrue(events[3].accepted)
        self.assertFalse(events[3].is_outlier)
        self.assertTrue(all(not event.stable_text.startswith("M") for event in events))
        self.assertEqual(events[-2].stable_text, "I am")
        self.assertEqual(events[-1].stable_text, "I am now talking into")

    def test_stable_delta_is_incremental_and_stable_text_is_monotonic(self):
        events = [
            self.observe("inter", 1, completed_at=0.00),
            self.observe("inter", 2, completed_at=0.40),
            self.observe("inter", 3, completed_at=0.85),
            self.observe("international", 4, completed_at=1.20),
            self.observe("international", 5, completed_at=1.60),
            self.observe("international", 6, completed_at=2.05),
            self.observe("wrong branch", 7, completed_at=2.20),
        ]

        self.assertEqual(events[2].stable_delta, "inter")
        self.assertEqual(events[5].stable_delta, "national")
        self.assertEqual(events[5].stable_text, "international")
        self.assertEqual(events[6].stable_text, "international")
        self.assertTrue(events[6].is_outlier)

        stable_lengths = [len(event.stable_text) for event in events]
        self.assertEqual(stable_lengths, sorted(stable_lengths))

    def test_unstable_suffix_and_display_merge_after_partial_commit(self):
        self.observe("hello world", 1, completed_at=0.00)
        self.observe("hello world", 2, completed_at=0.40)
        event = self.observe("hello world", 3, completed_at=0.85)

        self.assertEqual(event.stable_text, "hello")
        self.assertEqual(event.stable_delta, "hello")
        self.assertEqual(event.unstable_text, " world")
        self.assertEqual(event.display_text, "hello world")

    def test_display_merge_does_not_duplicate_casing_or_punctuation_overlap(self):
        self.observe("hello", 1, completed_at=0.00)
        self.observe("hello", 2, completed_at=0.40)
        self.observe("hello", 3, completed_at=0.85)

        event = self.observe("Hello, world", 4, completed_at=1.00)

        self.assertEqual(event.stable_text, "hello")
        self.assertEqual(event.unstable_text, ", world")
        self.assertEqual(event.display_text, "hello, world")
        self.assertEqual(event.display_text.casefold().count("hello"), 1)

    def test_partial_word_can_commit_without_trailing_space_and_continue_later(self):
        self.observe("inter ", 1, completed_at=0.00)
        self.observe("inter ", 2, completed_at=0.40)
        fragment = self.observe("inter ", 3, completed_at=0.85)

        self.assertEqual(fragment.stable_delta, "inter")
        self.assertEqual(fragment.stable_text, "inter")
        self.assertFalse(fragment.stable_text.endswith(" "))
        self.assertEqual(fragment.unstable_text, " ")

        self.observe("international", 4, completed_at=1.20)
        self.observe("international", 5, completed_at=1.60)
        continuation = self.observe("international", 6, completed_at=2.05)

        self.assertEqual(continuation.stable_delta, "national")
        self.assertEqual(continuation.stable_text, "international")

    def test_spaces_are_delayed_until_right_context_is_stable(self):
        self.observe("book now", 1, completed_at=0.00)
        self.observe("book now", 2, completed_at=0.40)
        third = self.observe("book now", 3, completed_at=0.85)
        fourth = self.observe("book now", 4, completed_at=0.95)

        self.assertEqual(third.stable_text, "book")
        self.assertEqual(third.unstable_text, " now")
        self.assertEqual(fourth.stable_delta, " now")
        self.assertEqual(fourth.stable_text, "book now")

    def test_short_right_context_does_not_force_ambiguous_word_boundary(self):
        for sequence, completed_at in enumerate((0.00, 0.30, 0.60, 0.90), start=1):
            event = self.observe("book a ", sequence, completed_at=completed_at)

        self.assertEqual(event.stable_text, "book")
        self.assertFalse(event.stable_text.endswith(" "))
        self.assertNotEqual(event.stable_text, "book a ")

    def test_punctuation_churn_counts_as_compatible_but_is_not_inserted_late(self):
        self.observe("hello world", 1, completed_at=0.00)
        self.observe("hello, world", 2, completed_at=0.30)
        self.observe("Hello world", 3, completed_at=0.60)
        event = self.observe("hello, world", 4, completed_at=0.90)

        self.assertEqual(event.stable_text.casefold(), "hello world")
        self.assertNotIn(",", event.stable_text)

        punctuation_edge = self.observe("hello, world!", 5, completed_at=1.10)
        self.assertEqual(punctuation_edge.stable_delta, "")
        self.assertEqual(punctuation_edge.stable_text, event.stable_text)
        self.assertEqual(punctuation_edge.unstable_text, "!")

    def test_one_hallucination_is_ignored_and_prior_history_recovers(self):
        self.observe("turnon", 1, completed_at=0.00)
        self.observe("turnon", 2, completed_at=0.40)
        outlier = self.observe(
            "completely unrelated banana",
            3,
            completed_at=0.80,
        )
        recovered = self.observe("turnon", 4, completed_at=0.95)

        self.assertTrue(outlier.is_outlier)
        self.assertFalse(outlier.accepted)
        self.assertEqual(outlier.stable_text, "")
        self.assertEqual(recovered.stable_delta, "turnon")

    def test_large_growth_from_short_prefix_is_not_treated_as_outlier(self):
        first = self.observe("a", 1, completed_at=0.00)
        grown = self.observe("alpha beta", 2, completed_at=0.40)
        repeated = self.observe("alpha beta", 3, completed_at=0.85)
        stable_word = self.observe("alpha beta", 4, completed_at=1.25)
        stable_phrase = self.observe("alpha beta", 5, completed_at=1.65)

        self.assertTrue(first.accepted)
        self.assertTrue(grown.accepted)
        self.assertFalse(grown.is_outlier)
        self.assertEqual(repeated.stable_text, "a")
        self.assertEqual(stable_word.stable_text, "alpha")
        self.assertEqual(stable_phrase.stable_text, "alpha beta")

    def test_multiple_unrelated_observations_after_stable_text_do_not_revise_it(self):
        self.observe("hello", 1, completed_at=0.00)
        self.observe("hello", 2, completed_at=0.40)
        stable = self.observe("hello", 3, completed_at=0.85)
        outlier = self.observe("goodbye now", 4, completed_at=1.00)
        repeated_outlier = self.observe("goodbye now please", 5, completed_at=1.20)

        self.assertEqual(stable.stable_text, "hello")
        self.assertTrue(outlier.is_outlier)
        self.assertTrue(outlier.stable_prefix_conflict)
        self.assertEqual(outlier.stable_text, "hello")
        self.assertEqual(repeated_outlier.stable_text, "hello")
        self.assertEqual(repeated_outlier.display_text, stable.display_text)

    def test_internal_consensus_recovers_after_bad_public_prefix_without_new_public_delta(self):
        stabilizer = RealtimeTextStabilizer(
            RealtimeTextStabilizationConfig(
                min_char_evidence_span_seconds=0.20,
                space_min_confirmations=3,
                space_min_evidence_span_seconds=0.20,
                initial_prefix_min_evidence_span_seconds=0.20,
                max_char_evidence_window_seconds=3.00,
            )
        )
        stabilizer.reset("rec-1", segment_id="seg-1")

        def observe(text, sequence, completed_at):
            return stabilizer.observe(
                self.observation(text, sequence, completed_at=completed_at)
            )

        bad = observe("I would think that the card", 1, 0.00)
        bad = observe("I would think that the card", 2, 0.20)
        bad = observe("I would think that the card approached us as", 3, 0.40)
        self.assertTrue(bad.stable_text.startswith("I would think that the card"))
        public_text = bad.stable_text

        first_correction = observe(
            "I would think that the current approach has a kind of lot of potential",
            4,
            0.70,
        )
        second_correction = observe(
            "I would think that the current approach has a kind of lot of potential",
            5,
            1.00,
        )
        recovered = observe(
            "I would think that the current approach has a kind of lot of potential",
            6,
            1.30,
        )

        self.assertTrue(first_correction.accepted)
        self.assertFalse(first_correction.is_outlier)
        self.assertTrue(first_correction.stable_prefix_conflict)
        self.assertTrue(first_correction.internal_revision)
        self.assertEqual(first_correction.stable_text, public_text)
        self.assertEqual(first_correction.stable_delta, "")
        self.assertTrue(first_correction.display_text.startswith(public_text))
        self.assertIn("current approach", first_correction.unstable_text)
        self.assertNotEqual(
            first_correction.display_text,
            first_correction.consensus_display_text,
        )

        self.assertEqual(second_correction.stable_text, public_text)
        self.assertEqual(recovered.stable_text, public_text)
        self.assertEqual(recovered.stable_delta, "")
        self.assertTrue(recovered.display_text.startswith(public_text))
        self.assertIn("current approach", recovered.unstable_text)
        self.assertIn("current approach", recovered.consensus_text)
        self.assertEqual(
            recovered.consensus_text,
            "I would think that the current approach has a kind of lot of potential",
        )
        self.assertFalse(recovered.public_consensus_aligned)

    def test_stale_duplicate_out_of_order_and_wrong_recording_observations_are_ignored(self):
        first = self.observe("alpha", 1, completed_at=0.00)
        duplicate = self.observe("alpha", 1, completed_at=0.10)
        out_of_order = self.observe("alpha", 0, completed_at=0.11)
        wrong_recording = self.observe(
            "alpha",
            2,
            completed_at=0.12,
            recording_id="rec-2",
        )
        second = self.observe("alpha", 2, completed_at=0.21)

        self.assertTrue(first.accepted)
        self.assertEqual(duplicate.ignored_reason, "stale-sequence")
        self.assertEqual(out_of_order.ignored_reason, "stale-sequence")
        self.assertEqual(wrong_recording.ignored_reason, "wrong-recording")
        self.assertTrue(second.accepted)

    def test_empty_text_is_ignored_without_clearing_state(self):
        self.observe("hello", 1, completed_at=0.00)
        empty = self.observe("   ", 2, completed_at=0.40)
        recovered = self.observe("hello", 3, completed_at=0.85)

        self.assertEqual(empty.ignored_reason, "empty-text")
        self.assertEqual(empty.stable_text, "")
        self.assertTrue(recovered.accepted)

    def test_reset_starts_a_clean_recording_history(self):
        self.observe("hello", 1, completed_at=0.00)
        self.observe("hello", 2, completed_at=0.40)
        self.observe("hello", 3, completed_at=0.85)

        self.stabilizer.reset("rec-2", segment_id="seg-2")
        snapshot = self.stabilizer.snapshot()
        old_recording = self.observe(
            "hello",
            4,
            completed_at=1.00,
            recording_id="rec-1",
        )
        new_recording = self.observe(
            "new",
            1,
            completed_at=0.00,
            recording_id="rec-2",
        )

        self.assertEqual(snapshot.stable_text, "")
        self.assertEqual(snapshot.recording_id, "rec-2")
        self.assertEqual(old_recording.ignored_reason, "wrong-recording")
        self.assertEqual(new_recording.unstable_text, "new")

    def test_publish_suppressed_observations_contribute_evidence_without_losing_delta(self):
        self.observe("hello", 1, completed_at=0.00, publish_allowed=False)
        self.observe("hello", 2, completed_at=0.40, publish_allowed=False)
        suppressed = self.observe("hello", 3, completed_at=0.85, publish_allowed=False)
        published = self.observe("hello", 4, completed_at=1.00, publish_allowed=True)

        self.assertFalse(suppressed.should_publish)
        self.assertEqual(suppressed.stable_delta, "hello")
        self.assertTrue(published.should_publish)
        self.assertEqual(published.stable_delta, "hello")
        self.assertEqual(published.stable_text, "hello")

    def test_timing_and_audio_evidence_diagnostics_are_reported(self):
        self.observe(
            "hello",
            1,
            completed_at=0.00,
            audio_start_sample=0,
            audio_end_sample_exclusive=100,
        )
        self.observe(
            "hello",
            2,
            completed_at=0.40,
            audio_start_sample=0,
            audio_end_sample_exclusive=200,
        )
        event = self.observe(
            "hello",
            3,
            completed_at=0.85,
            audio_start_sample=0,
            audio_end_sample_exclusive=300,
        )

        self.assertEqual(event.evidence.first_sequence, 1)
        self.assertEqual(event.evidence.latest_sequence, 3)
        self.assertEqual(event.evidence.first_completed_at_monotonic, 0.00)
        self.assertEqual(event.evidence.latest_completed_at_monotonic, 0.85)
        self.assertEqual(event.evidence.audio_start_sample_min, 0)
        self.assertEqual(event.evidence.audio_end_sample_max, 300)
        self.assertIsNone(event.stable_audio_end_sample_exclusive)

    def test_repeated_phrases_do_not_duplicate_unstable_suffix(self):
        phrase = "to be or not to be"
        for sequence, completed_at in enumerate((0.00, 0.30, 0.60, 0.90), start=1):
            event = self.observe(phrase, sequence, completed_at=completed_at)

        self.assertEqual(event.stable_text, phrase)
        self.assertEqual(event.display_text, phrase)

        continued = self.observe("To be or not to be again", 5, completed_at=1.10)
        self.assertEqual(continued.display_text.casefold().count(phrase), 1)
        self.assertEqual(continued.unstable_text, " again")

    def test_final_text_agreement_and_mismatch_do_not_revise_stable_text(self):
        self.observe("hello", 1, completed_at=0.00)
        self.observe("hello", 2, completed_at=0.40)
        self.observe("hello", 3, completed_at=0.85)

        agreed = self.stabilizer.finalize(
            RealtimeTextFinalObservation(
                recording_id="rec-1",
                segment_id="seg-1",
                final_text="Hello world",
                completed_at_monotonic=0.50,
            )
        )
        after_final = self.observe("hello again", 4, completed_at=0.60)

        self.assertTrue(agreed.agrees_with_stable_prefix)
        self.assertEqual(agreed.final_suffix_after_stable, " world")
        self.assertFalse(agreed.stable_text_was_revised)
        self.assertEqual(after_final.ignored_reason, "finalized")

        self.stabilizer.reset("rec-2")
        for sequence, completed_at in enumerate((0.00, 0.40, 0.85), start=1):
            self.stabilizer.observe(
                self.observation(
                    "hello",
                    sequence,
                    completed_at=completed_at,
                    recording_id="rec-2",
                )
            )
        mismatch = self.stabilizer.finalize(
            RealtimeTextFinalObservation(
                recording_id="rec-2",
                final_text="yellow world",
                completed_at_monotonic=0.50,
            )
        )

        self.assertFalse(mismatch.agrees_with_stable_prefix)
        self.assertEqual(mismatch.mismatch_reason, "stable-prefix-mismatch")
        self.assertEqual(mismatch.stable_text, "hello")

    def test_recent_history_is_bounded_when_configured(self):
        stabilizer = RealtimeTextStabilizer(
            RealtimeTextStabilizationConfig(
                min_char_confirmations=1,
                min_char_evidence_span_seconds=0.0,
                max_recent_observations=3,
            )
        )
        stabilizer.reset("rec-1")

        for sequence, text in enumerate(("a", "ab", "abc", "abcd", "abcde"), start=1):
            stabilizer.observe(self.observation(text, sequence))

        snapshot = stabilizer.snapshot()
        self.assertEqual(snapshot.recent_observation_count, 3)
        self.assertEqual(snapshot.dropped_observation_count, 2)


class AudioRecorderRealtimeStabilizerIntegrationTests(unittest.TestCase):
    def setUp(self):
        if AUDIO_RECORDER_IMPORT_ERROR is not None:
            self.skipTest(f"AudioToTextRecorder import failed: {AUDIO_RECORDER_IMPORT_ERROR}")

    def test_start_creates_new_realtime_recording_generation_and_resets_stabilizer(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.recording_stop_time = 0
        recorder.min_gap_between_recordings = 0
        recorder.state = "inactive"
        recorder.spinner = False
        recorder.halo = None
        recorder.on_recording_start = None
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.realtime_text_stabilizer = RealtimeTextStabilizer()
        recorder.realtime_recording_id = 0
        recorder.realtime_observation_sequence = 7

        recorder.start(frames=[b"\x00\x00" * 1024])

        self.assertEqual(recorder.realtime_recording_id, 1)
        self.assertEqual(recorder.realtime_observation_sequence, 0)
        self.assertGreater(recorder.recording_start_monotonic, 0.0)
        self.assertEqual(
            recorder.realtime_text_stabilizer.snapshot().recording_id,
            1,
        )
        self.assertEqual(
            recorder.realtime_text_stabilizer.snapshot().stable_text,
            "",
        )


if __name__ == "__main__":
    unittest.main()
