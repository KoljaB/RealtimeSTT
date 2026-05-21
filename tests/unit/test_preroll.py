import unittest

from RealtimeSTT.preroll import (
    REASON_BELOW_MINIMUM,
    REASON_EMPTY_BUFFER,
    REASON_FALLBACK_FULL_PREROLL,
    REASON_STABLE_SILENCE_FOUND,
    REASON_UNCERTAIN,
    PrerollFrameMetadata,
    select_preroll_frames,
)


SAMPLE_RATE = 16000


def frame(is_speech, rms, frame_ms=20):
    return PrerollFrameMetadata(
        sample_count=int(SAMPLE_RATE * frame_ms / 1000),
        is_speech=is_speech,
        rms=rms,
    )


def frames(count, is_speech, rms, frame_ms=20):
    return [frame(is_speech, rms, frame_ms=frame_ms) for _ in range(count)]


def sample_count(metadata):
    return sum(item.sample_count for item in metadata)


class PrerollSelectionTests(unittest.TestCase):
    def test_clear_long_silence_before_speech_trims_to_conservative_minimum(self):
        metadata = frames(30, False, 5.0) + frames(8, True, 300.0)

        selection = select_preroll_frames(metadata, SAMPLE_RATE)

        self.assertEqual(REASON_STABLE_SILENCE_FOUND, selection.reason)
        self.assertGreater(selection.start_index, 0)
        self.assertGreaterEqual(selection.included_seconds, 0.60)
        self.assertLess(selection.included_sample_count, sample_count(metadata))

    def test_no_clear_silence_keeps_full_preroll(self):
        metadata = frames(5, False, 5.0) + frames(30, True, 300.0)

        selection = select_preroll_frames(metadata, SAMPLE_RATE)

        self.assertEqual(REASON_UNCERTAIN, selection.reason)
        self.assertEqual(0, selection.start_index)
        self.assertEqual(sample_count(metadata), selection.included_sample_count)

    def test_short_vad_false_gap_inside_speech_is_not_boundary(self):
        metadata = frames(20, True, 280.0) + frames(2, False, 5.0) + frames(20, True, 300.0)

        selection = select_preroll_frames(
            metadata,
            SAMPLE_RATE,
            max_gap_ms=80.0,
        )

        self.assertEqual(REASON_FALLBACK_FULL_PREROLL, selection.reason)
        self.assertEqual("onset_at_buffer_start", selection.diagnostics["fallbackDetail"])

    def test_energy_refinement_only_helps_clear_silence(self):
        clear_low_energy = frames(12, None, 5.0) + frames(10, True, 300.0)
        uncertain_energy = frames(12, None, 120.0) + frames(10, True, 180.0)

        selected = select_preroll_frames(
            clear_low_energy,
            SAMPLE_RATE,
            energy_silence_rms=50.0,
            min_included_ms=300.0,
        )
        fallback = select_preroll_frames(
            uncertain_energy,
            SAMPLE_RATE,
            energy_silence_rms=50.0,
            min_included_ms=300.0,
        )

        self.assertEqual(REASON_STABLE_SILENCE_FOUND, selected.reason)
        self.assertLess(selected.included_sample_count, sample_count(clear_low_energy))
        self.assertEqual(REASON_UNCERTAIN, fallback.reason)
        self.assertEqual(sample_count(uncertain_energy), fallback.included_sample_count)

    def test_high_energy_pre_speech_tail_is_kept_after_stable_silence(self):
        metadata = frames(100, False, 5.0) + frames(10, None, 160.0) + frames(8, True, 300.0)

        selection = select_preroll_frames(
            metadata,
            SAMPLE_RATE,
            min_silence_ms=80.0,
            guard_ms=80.0,
            min_included_ms=300.0,
        )

        self.assertEqual(REASON_STABLE_SILENCE_FOUND, selection.reason)
        self.assertGreaterEqual(selection.diagnostics["preSpeechTailSeconds"], 0.19)
        self.assertLess(selection.included_sample_count, sample_count(metadata))
        self.assertLessEqual(selection.start_index, 96)

    def test_empty_and_short_buffers_are_safe(self):
        empty = select_preroll_frames([], SAMPLE_RATE)
        short = select_preroll_frames(
            frames(8, False, 5.0) + frames(2, True, 300.0),
            SAMPLE_RATE,
        )

        self.assertEqual(REASON_EMPTY_BUFFER, empty.reason)
        self.assertEqual(0, empty.included_sample_count)
        self.assertEqual(REASON_BELOW_MINIMUM, short.reason)
        self.assertEqual(10, short.selected_frame_count)

    def test_included_seconds_matches_selected_samples_exactly(self):
        metadata = frames(60, False, 5.0, frame_ms=10) + frames(
            20,
            True,
            300.0,
            frame_ms=10,
        )

        selection = select_preroll_frames(metadata, SAMPLE_RATE)

        self.assertEqual(REASON_STABLE_SILENCE_FOUND, selection.reason)
        self.assertEqual(
            selection.included_sample_count / float(SAMPLE_RATE),
            selection.included_seconds,
        )
        self.assertEqual(
            sample_count(metadata[selection.start_index:]),
            selection.included_sample_count,
        )

    def test_parameters_can_be_swept_without_server_state(self):
        metadata = frames(35, False, 5.0) + frames(10, True, 300.0)
        cases = [
            {"min_silence_ms": 150.0, "guard_ms": 120.0, "max_gap_ms": 40.0, "min_included_ms": 550.0},
            {"min_silence_ms": 200.0, "guard_ms": 160.0, "max_gap_ms": 80.0, "min_included_ms": 600.0},
            {"min_silence_ms": 250.0, "guard_ms": 180.0, "max_gap_ms": 100.0, "min_included_ms": 650.0},
        ]

        for case in cases:
            with self.subTest(case=case):
                selection = select_preroll_frames(
                    metadata,
                    SAMPLE_RATE,
                    **case,
                )

                self.assertEqual(REASON_STABLE_SILENCE_FOUND, selection.reason)
                self.assertGreaterEqual(
                    selection.included_seconds,
                    case["min_included_ms"] / 1000.0,
                )

    def test_default_regression_does_not_fall_to_tiny_preroll(self):
        metadata = frames(30, False, 5.0) + frames(8, True, 300.0)

        selection = select_preroll_frames(metadata, SAMPLE_RATE)

        self.assertGreaterEqual(selection.included_seconds, 0.60)
        self.assertGreater(selection.included_seconds, 0.08)


if __name__ == "__main__":
    unittest.main()
