import unittest

from RealtimeSTT.core.realtime_merge import (
    StickyRealtimeTranscriptionMerger,
)


class StickyRealtimeTranscriptionMergerTests(unittest.TestCase):
    def setUp(self):
        self.merger = StickyRealtimeTranscriptionMerger()
        self.merger.reset("recording-1")

    def slow(self, text, sequence, **kwargs):
        return self.merger.observe_slow(
            text,
            recording_id="recording-1",
            sequence=sequence,
            **kwargs,
        )

    def fast(self, text, sequence, **kwargs):
        return self.merger.observe_ultrafast(
            text,
            recording_id="recording-1",
            sequence=sequence,
            **kwargs,
        )

    def test_fast_before_slow_is_cached_but_not_published(self):
        result = self.fast("we should ship the feature today", 1)

        self.assertEqual(result.status, "waiting_for_slow")
        self.assertEqual(result.text, "")
        self.assertFalse(result.should_publish)

        result = self.slow("we should ship the feature", 1)
        self.assertEqual(result.text, "we should ship the feature today")
        self.assertEqual(result.status, "exact")
        self.assertTrue(result.should_publish)

    def test_exact_append_and_bounded_suffix(self):
        self.slow("we should ship the feature", 1)
        result = self.fast(
            "we should ship the feature today now later extra fifth sixth",
            1,
        )

        self.assertEqual(result.status, "exact")
        self.assertEqual(
            result.ultrafast_suffix,
            "today now later extra fifth",
        )
        self.assertEqual(
            result.text,
            "we should ship the feature today now later extra fifth",
        )

    def test_exact_result_can_only_extend_existing_suffix(self):
        self.slow("we should ship the feature", 1)
        first = self.fast("we should ship the feature today", 1)
        second = self.fast("we should ship the feature today now", 2)

        self.assertEqual(first.text, "we should ship the feature today")
        self.assertEqual(second.text, "we should ship the feature today now")
        self.assertTrue(second.should_publish)

    def test_failed_alignment_retains_fast_tail_then_new_slow_drops_it(self):
        self.slow("we should ship the feature", 1)
        accepted = self.fast("we should ship the feature today", 1)
        held = self.fast("completely unrelated words from another result", 2)

        self.assertEqual(accepted.text, "we should ship the feature today")
        self.assertEqual(held.status, "held_no_anchor")
        self.assertEqual(held.text, accepted.text)
        self.assertFalse(held.should_publish)

        advanced = self.slow("we should ship the feature is ready", 2)
        self.assertEqual(advanced.status, "slow_advanced_no_anchor")
        self.assertEqual(advanced.text, "we should ship the feature is ready")
        self.assertNotIn("today", advanced.text)
        self.assertTrue(advanced.should_publish)

    def test_fast_cannot_retract_or_rewrite_published_suffix(self):
        self.slow("we should ship the feature", 1)
        accepted = self.fast("we should ship the feature today", 1)
        shorter = self.fast("we should ship the feature", 2)
        conflicting = self.fast("we should ship the feature tomorrow", 3)

        self.assertEqual(shorter.status, "held_fast_conflict")
        self.assertEqual(shorter.text, accepted.text)
        self.assertEqual(conflicting.status, "held_fast_conflict")
        self.assertEqual(conflicting.text, accepted.text)
        self.assertFalse(conflicting.should_publish)

    def test_fuzzy_four_word_anchor_requires_two_compatible_observations(self):
        self.merger = StickyRealtimeTranscriptionMerger(fuzzy_confirmation_count=2)
        self.merger.reset("recording-1")
        slow = self.slow("we alpha bravo charlie delta", 1)
        first = self.fast("we alpha bravo charli delta today", 1)
        second = self.fast("we alpha bravo charli delta today", 2)

        self.assertEqual(slow.text, "we alpha bravo charlie delta")
        self.assertEqual(first.status, "held_fuzzy_pending")
        self.assertEqual(first.text, slow.text)
        self.assertFalse(first.should_publish)
        self.assertEqual(second.status, "fuzzy")
        self.assertEqual(second.text, "we alpha bravo charlie delta today")
        self.assertEqual(second.slow_text, "we alpha bravo charlie delta")
        self.assertEqual(second.ultrafast_suffix, "today")
        self.assertTrue(second.used_fuzzy_match)
        self.assertTrue(second.should_publish)

    def test_two_word_exact_anchor_is_accepted(self):
        self.slow("hello world", 1)
        result = self.fast("hello world today", 1)

        self.assertEqual(result.status, "exact")
        self.assertEqual(result.anchor_length, 2)
        self.assertEqual(result.text, "hello world today")

    def test_two_word_close_anchor_is_accepted_immediately_by_default(self):
        self.slow("we cat", 1)
        result = self.fast("we bat today", 1)

        self.assertEqual(result.status, "held_no_anchor")
        self.assertFalse(result.matched)
        self.assertEqual(result.ultrafast_suffix, "")
        self.assertEqual(result.text, "we cat")

    def test_soft_frontier_handles_deleted_slow_word_and_partial_frontier(self):
        self.slow("Know you are already de", 1)
        result = self.fast("Kno you are dealing with", 1)

        self.assertEqual(result.status, "soft_frontier")
        self.assertTrue(result.matched)
        self.assertTrue(result.used_fuzzy_match)
        self.assertGreaterEqual(result.anchor_length, 2)
        self.assertEqual(result.ultrafast_suffix, "with")
        self.assertEqual(result.text, "Know you are already de with")

    def test_soft_frontier_tracks_cumulative_hypotheses_to_visual_frontier(self):
        self.slow(
            "Know you are already dealing with a lot of visual",
            1,
        )
        result = self.fast(
            "Kno you are dealing with a lo visu information",
            1,
        )

        self.assertEqual(result.status, "soft_frontier")
        self.assertEqual(result.ultrafast_suffix, "information")
        self.assertEqual(
            result.text,
            "Know you are already dealing with a lot of visual information",
        )
        self.assertGreaterEqual(result.anchor_length, 2)

    def test_soft_frontier_rejects_unrelated_text(self):
        self.slow("Know you are already de", 1)
        result = self.fast("completely unrelated words from another result", 1)

        self.assertEqual(result.status, "held_no_anchor")
        self.assertFalse(result.matched)
        self.assertEqual(result.ultrafast_suffix, "")
        self.assertEqual(result.text, "Know you are already de")

    def test_soft_frontier_rejects_nonmonotonic_coincidences(self):
        self.slow("alpha bravo charlie delta", 1)
        result = self.fast("delta alpha bravo speculative", 1)

        self.assertEqual(result.status, "held_no_anchor")
        self.assertFalse(result.matched)
        self.assertEqual(result.ultrafast_suffix, "")

    def test_soft_frontier_rejects_large_gap_between_two_matches(self):
        self.slow("alpha bravo charlie delta", 1)
        result = self.fast(
            "alpha one two three four five delta speculative",
            1,
        )

        self.assertEqual(result.status, "held_no_anchor")
        self.assertFalse(result.matched)
        self.assertEqual(result.ultrafast_suffix, "")


    def test_soft_frontier_does_not_duplicate_deleted_slow_words(self):
        self.slow("the cat sat on the mat", 1)
        result = self.fast("the cat sat mat now", 1)

        self.assertEqual(result.status, "soft_frontier")
        self.assertEqual(result.ultrafast_suffix, "now")
        self.assertEqual(result.text, "the cat sat on the mat now")
        self.assertNotIn("sat mat", result.ultrafast_suffix)

    def test_soft_frontier_rejects_short_replacement_without_frontier_match(self):
        self.slow("I am talking into", 1)
        result = self.fast("I am talking to the microphone", 1)

        self.assertEqual(result.status, "held_no_anchor")
        self.assertFalse(result.matched)
        self.assertEqual(result.ultrafast_suffix, "")

    def test_soft_frontier_rejects_repeated_weak_function_word_ambiguity(self):
        self.slow("to be to", 1)
        result = self.fast("to noise to speculative", 1)

        self.assertEqual(result.status, "held_no_anchor")
        self.assertFalse(result.matched)
        self.assertEqual(result.ultrafast_suffix, "")

    def test_soft_frontier_ignores_long_historical_prefix(self):
        history = " ".join(f"history{index}" for index in range(20))
        self.slow(history + " eight nine Know you are already de", 1)
        result = self.fast("Kno you are dealing with", 1)

        self.assertEqual(result.status, "soft_frontier")
        self.assertEqual(result.ultrafast_suffix, "with")

    def test_one_word_utterance_does_not_use_weak_anchor(self):
        self.slow("hello", 1)
        result = self.fast("we hello today", 1)

        self.assertEqual(result.status, "held_no_anchor")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.ultrafast_suffix, "")
    def test_exact_one_word_bootstrap_then_normal_two_word_anchor(self):
        self.fast("I can test", 1)
        bootstrap = self.slow("I", 1)

        self.assertEqual(bootstrap.status, "bootstrap_exact")
        self.assertTrue(bootstrap.matched)
        self.assertFalse(bootstrap.used_fuzzy_match)
        self.assertEqual(bootstrap.anchor_length, 1)
        self.assertEqual(bootstrap.distance, 0)
        self.assertEqual(bootstrap.text, "I can test")
        self.assertEqual(bootstrap.ultrafast_suffix, "can test")

        advanced = self.slow("I can", 2)
        self.assertEqual(advanced.status, "exact")
        self.assertEqual(advanced.anchor_length, 2)
        self.assertEqual(advanced.text, "I can test")
        self.assertEqual(advanced.ultrafast_suffix, "test")

    def test_one_word_bootstrap_rejects_nonprefix_fast_text(self):
        self.fast("we I can", 1)
        result = self.slow("I", 1)

        self.assertEqual(result.status, "slow_only")
        self.assertFalse(result.matched)
        self.assertEqual(result.anchor_length, 0)
        self.assertEqual(result.text, "I")
        self.assertEqual(result.ultrafast_suffix, "")

    def test_recording_reset_clears_all_text_and_sequences(self):
        self.slow("we should ship the feature", 1)
        self.fast("we should ship the feature today", 1)

        reset = self.merger.reset("recording-2")
        self.assertEqual(reset.text, "")
        self.assertEqual(reset.status, "waiting_for_slow")
        self.assertEqual(reset.slow_generation, 0)

        result = self.merger.observe_slow(
            "new recording starts here",
            recording_id="recording-2",
            sequence=1,
        )
        self.assertEqual(result.text, "new recording starts here")
        self.assertEqual(result.slow_sequence, 1)

    def test_stale_sequences_and_recording_ids_do_not_change_state(self):
        accepted = self.slow("we should ship the feature", 2)
        stale_slow = self.slow("a different stale hypothesis", 1)
        stale_recording = self.merger.observe_slow(
            "wrong recording text",
            recording_id="recording-other",
            sequence=3,
        )

        self.assertEqual(stale_slow.status, "stale_ignored")
        self.assertEqual(stale_slow.text, accepted.text)
        self.assertEqual(stale_recording.status, "stale_ignored")
        self.assertEqual(stale_recording.text, accepted.text)
        self.assertEqual(self.merger.snapshot().text, accepted.text)

        self.fast("we should ship the feature today", 2)
        stale_fast = self.fast("we should ship the feature tomorrow", 1)
        self.assertEqual(stale_fast.status, "stale_ignored")
        self.assertEqual(stale_fast.text, "we should ship the feature today")

    def test_punctuation_only_slow_update_keeps_generation_and_tail(self):
        self.slow("Alpha bravo charlie delta", 1)
        self.fast("Alpha bravo charlie delta later", 1)
        result = self.slow("alpha bravo charlie delta.", 2)

        self.assertEqual(result.slow_generation, 1)
        self.assertEqual(result.slow_text, "alpha bravo charlie delta.")
        self.assertEqual(result.text, "alpha bravo charlie delta. later")
        self.assertEqual(result.status, "slow_only")
        self.assertTrue(result.should_publish)


if __name__ == "__main__":
    unittest.main()
