import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from RealtimeSTT.realtime_text_stabilizer import RealtimeTextStabilizationConfig

from tools.evaluate_realtime_text_stabilizer import (
    SAMPLE_RATE,
    _revision_spans_from_events,
    aggregate_results,
    build_scenario_by_name,
    build_scenarios,
    compact_result,
    common_prefix_length,
    edit_distance,
    evaluate_scenario,
    false_commit_summary,
    main,
    make_observation,
    normalize_word,
    normalized_words,
    obs,
    percentile,
    prefix_partial_wer,
    scenario,
    summary_stats,
    timed_reference,
    word_error_rate,
    word_error_rate_words,
)


LEGACY_CORRELATED_EVIDENCE_CONFIG = RealtimeTextStabilizationConfig(
    min_char_confirmations=3,
    min_char_evidence_span_seconds=0.60,
    max_char_evidence_window_seconds=1.50,
    space_min_confirmations=4,
    space_min_evidence_span_seconds=0.75,
    punctuation_min_confirmations=4,
    initial_prefix_min_confirmations=3,
    initial_prefix_min_evidence_span_seconds=0.80,
    require_audio_progress_for_evidence=False,
)


class RealtimeTextStabilizerEvaluationNormalizationTests(unittest.TestCase):
    def test_normalize_word_removes_case_punctuation_and_apostrophes(self):
        self.assertEqual(normalize_word("Isn't?!"), "isnt")
        self.assertEqual(normalize_word("REAL-time"), "realtime")
        self.assertEqual(normalize_word("12.50dB"), "1250db")

    def test_normalized_words_matches_stabilizer_eval_contract(self):
        self.assertEqual(
            normalized_words("Hello, REAL-time isn't 5.0?"),
            ["hello", "real", "time", "isnt", "5", "0"],
        )

    def test_normalized_words_drops_empty_tokens(self):
        self.assertEqual(normalized_words("... --- \t\n"), [])


class RealtimeTextStabilizerEvaluationMetricTests(unittest.TestCase):
    def test_edit_distance_and_word_error_rate(self):
        expected = ["set", "the", "timer", "for", "fifteen", "seconds"]
        actual = ["set", "the", "timer", "for", "fifty", "seconds"]

        self.assertEqual(edit_distance(expected, actual), 1)
        self.assertAlmostEqual(word_error_rate_words(expected, actual), 1 / 6)

    def test_edit_distance_handles_insertions_deletions_and_empty_inputs(self):
        self.assertEqual(edit_distance([], []), 0)
        self.assertEqual(edit_distance(["one"], []), 1)
        self.assertEqual(edit_distance([], ["one", "two"]), 2)
        self.assertEqual(edit_distance(["a", "b"], ["a", "x", "b"]), 1)
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "c"]), 1)

    def test_word_error_rate_accepts_raw_text(self):
        self.assertEqual(word_error_rate("Hello, world!", "hello world"), 0.0)
        self.assertAlmostEqual(word_error_rate("hello world", "hello brave world"), 0.5)

    def test_common_prefix_length_handles_full_partial_and_empty_matches(self):
        self.assertEqual(common_prefix_length(["a", "b"], ["a", "b"]), 2)
        self.assertEqual(common_prefix_length(["a", "b"], ["a", "x"]), 1)
        self.assertEqual(common_prefix_length(["a"], []), 0)
        self.assertEqual(common_prefix_length([], ["a"]), 0)

    def test_false_commit_summary_counts_words_after_prefix_mismatch(self):
        reference = ["please", "open", "the", "settings", "panel"]
        committed = ["please", "open", "settings", "panel"]

        summary = false_commit_summary(reference, committed)

        self.assertEqual(summary["matchingPrefixWords"], 2)
        self.assertEqual(summary["falseCommitWords"], 2)
        self.assertEqual(summary["committedWords"], 4)
        self.assertEqual(summary["falseCommitRate"], 0.5)

    def test_false_commit_summary_handles_empty_and_correct_commits(self):
        empty = false_commit_summary(["hello"], [])
        self.assertEqual(empty["committedWords"], 0)
        self.assertEqual(empty["falseCommitWords"], 0)
        self.assertEqual(empty["falseCommitRate"], 0.0)

        correct = false_commit_summary(["hello", "world"], ["hello"])
        self.assertEqual(correct["matchingPrefixWords"], 1)
        self.assertEqual(correct["falseCommitWords"], 0)
        self.assertEqual(correct["committedReferenceCoverage"], 0.5)

    def test_prefix_partial_wer_ignores_missing_future_reference_words(self):
        reference = ["hello", "world", "this", "is", "clean"]
        partial = ["hello", "word"]

        self.assertEqual(common_prefix_length(reference, partial), 1)
        self.assertAlmostEqual(prefix_partial_wer(reference, partial), 0.5)

    def test_prefix_partial_wer_handles_empty_display(self):
        self.assertEqual(prefix_partial_wer([], []), 0.0)
        self.assertEqual(prefix_partial_wer(["hello"], []), 1.0)

    def test_percentile_interpolates_and_handles_empty_values(self):
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([3.0], 0.95), 3.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.25), 1.75)

    def test_summary_stats_rounds_expected_fields(self):
        empty = summary_stats([])
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["mean"])

        stats = summary_stats([1.0, 2.0, 4.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean"], 2.3333)
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["max"], 4.0)

    def test_revision_spans_from_events_counts_replaced_suffix_words(self):
        events = [
            {"displayWords": ["the", "card", "approached"]},
            {"displayWords": ["the", "current", "approach"]},
            {"displayWords": ["the", "current", "approach", "has"]},
            {"displayWords": ["the", "current"]},
        ]

        self.assertEqual(_revision_spans_from_events(events), [2, 2])


class RealtimeTextStabilizerEvaluationFixtureTests(unittest.TestCase):
    def test_timed_reference_matches_reference_word_count(self):
        text = "one two three"
        self.assertEqual(timed_reference(text), (0.45, 0.79, 1.13))

    def test_obs_defaults_audio_end_to_completed_at(self):
        replay_input = obs("hello", 1.25)

        self.assertEqual(replay_input.completed_at_s, 1.25)
        self.assertEqual(replay_input.audio_end_s, 1.25)
        self.assertTrue(replay_input.publish_allowed)

    def test_scenario_builds_tuple_observations_and_timing(self):
        replay = scenario(
            "tiny",
            "unit",
            "hello world",
            [obs("hello", 0.5), obs("hello world", 1.0)],
            "unit fixture",
            first_word_end_s=0.2,
            seconds_per_word=0.5,
        )

        self.assertEqual(replay.observations[0].text, "hello")
        self.assertEqual(replay.word_end_times_s, (0.2, 0.7))

    def test_make_observation_preserves_replay_timing_and_sample_count(self):
        replay_input = obs("hello", 1.25, 1.10, publish_allowed=False)
        observation = make_observation("case", replay_input, 7)

        self.assertEqual(observation.recording_id, "case")
        self.assertEqual(observation.segment_id, "case")
        self.assertEqual(observation.sequence, 7)
        self.assertEqual(observation.completed_at_monotonic, 1.25)
        self.assertEqual(observation.audio_end_sample_exclusive, int(1.10 * SAMPLE_RATE))
        self.assertFalse(observation.publish_allowed)

    def test_builtin_scenarios_have_unique_names_and_sane_timelines(self):
        scenarios = build_scenarios()
        names = [item.name for item in scenarios]
        categories = {item.category for item in scenarios}

        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(scenarios), 16)
        self.assertTrue(
            {
                "clean",
                "noise",
                "long_noise",
                "missing_word",
                "middle_substitution",
                "compound_substitution",
                "word_order",
                "hallucination",
                "numeric_substitution",
                "word_boundary",
                "correlated_repeats",
                "long_clean",
            }.issubset(categories)
        )
        for item in scenarios:
            self.assertTrue(item.reference_text.strip())
            self.assertEqual(len(item.word_end_times_s), len(normalized_words(item.reference_text)))
            self.assertGreaterEqual(len(item.observations), 3)
            completed_times = [observation.completed_at_s for observation in item.observations]
            self.assertEqual(completed_times, sorted(completed_times))

    def test_build_scenario_by_name_returns_expected_case(self):
        replay = build_scenario_by_name("long_noisy_card_current_branch")

        self.assertEqual(replay.category, "long_noise")
        self.assertIn("current approach", replay.reference_text)


class RealtimeTextStabilizerReplayEvaluationTests(unittest.TestCase):
    def test_clean_replay_has_no_false_commits_or_revisions(self):
        result = evaluate_scenario(build_scenario_by_name("clean_short_repeated_partials"))

        self.assertEqual(result["falseCommit"]["falseCommitWords"], 0)
        self.assertEqual(result["revisions"]["count"], 0)
        self.assertTrue(result["finalization"]["agreesWithStablePrefix"])
        self.assertGreater(result["commitLatencySeconds"]["committedCorrectWords"], 0)

    def test_noisy_replay_produces_false_commit_measurement(self):
        scenario = build_scenario_by_name("wrong_number_repeated_without_new_audio_context")
        result = evaluate_scenario(scenario, LEGACY_CORRELATED_EVIDENCE_CONFIG)

        self.assertGreater(result["falseCommit"]["falseCommitWords"], 0)
        self.assertFalse(result["finalization"]["agreesWithStablePrefix"])

    def test_wrong_prefix_replay_records_internal_revision_and_conflicts(self):
        result = evaluate_scenario(
            build_scenario_by_name("correlated_wrong_prefix_before_correction"),
            LEGACY_CORRELATED_EVIDENCE_CONFIG,
        )

        self.assertGreater(result["internalRevisionEvents"], 0)
        self.assertGreater(result["stablePrefixConflictEvents"], 0)
        self.assertEqual(result["finalStableText"], "M")
        self.assertIn("I am now", result["finalDisplayText"])

    def test_long_noisy_replay_exposes_large_revision_span(self):
        result = evaluate_scenario(
            build_scenario_by_name("long_noisy_card_current_branch"),
            LEGACY_CORRELATED_EVIDENCE_CONFIG,
        )

        self.assertGreater(result["revisions"]["count"], 0)
        self.assertGreaterEqual(result["revisions"]["spanWords"]["max"], 8)
        self.assertIn("card", result["finalStableText"])
        self.assertIn("current approach", result["finalDisplayText"])

    def test_all_builtin_scenarios_evaluate_to_required_report_shape(self):
        required_top_level = {
            "name",
            "category",
            "referenceText",
            "observationCount",
            "falseCommit",
            "commitLatencySeconds",
            "revisions",
            "partialWer",
            "unstablePartialWordRatio",
            "uiUpdateFrequency",
            "events",
        }

        for replay in build_scenarios():
            with self.subTest(replay=replay.name):
                result = evaluate_scenario(replay)
                self.assertTrue(required_top_level.issubset(result))
                self.assertEqual(result["observationCount"], len(result["events"]))
                self.assertLessEqual(0.0, result["falseCommit"]["falseCommitRate"])
                self.assertGreaterEqual(result["uiUpdateFrequency"]["durationSeconds"], 0.001)

    def test_false_commit_matrix_covers_multiple_failure_families(self):
        failure_cases = {
            "correlated_wrong_prefix_before_correction": "leading_noise",
            "middle_word_substitution_tap_tab": "middle_substitution",
            "missing_function_word_on_shift": "missing_function_word",
            "acronym_compound_website_websocket": "developer_compound",
            "word_order_color_swap": "word_order",
            "prefix_dropout_missing_you": "prefix_dropout",
            "numeric_slider_seventy_seven": "numeric_confusion",
            "wrong_number_repeated_without_new_audio_context": "same_audio_repeats",
            "tail_compound_after_noon_afternoon": "tail_compound",
        }

        for name, family in failure_cases.items():
            with self.subTest(family=family):
                result = evaluate_scenario(
                    build_scenario_by_name(name),
                    LEGACY_CORRELATED_EVIDENCE_CONFIG,
                )
                self.assertGreater(result["falseCommit"]["falseCommitWords"], 0)
                self.assertGreater(result["falseCommit"]["falseCommitRate"], 0.0)

    def test_false_commits_can_happen_without_visible_revision_spans(self):
        for name in (
            "word_order_color_swap",
            "prefix_dropout_missing_you",
            "numeric_slider_seventy_seven",
            "wrong_number_repeated_without_new_audio_context",
        ):
            with self.subTest(name=name):
                result = evaluate_scenario(
                    build_scenario_by_name(name),
                    LEGACY_CORRELATED_EVIDENCE_CONFIG,
                )
                self.assertGreater(result["falseCommit"]["falseCommitWords"], 0)
                self.assertEqual(result["revisions"]["count"], 0)
                self.assertGreater(result["stablePrefixConflictEvents"], 0)

    def test_clean_long_progressive_stream_stays_false_commit_free(self):
        result = evaluate_scenario(build_scenario_by_name("long_clean_progressive_undercommit"))

        self.assertEqual(result["falseCommit"]["falseCommitWords"], 0)
        self.assertGreater(result["commitLatencySeconds"]["correctCommitCoverage"], 0.1)
        self.assertGreater(result["partialWer"]["displayPrefixReference"]["mean"], -0.0001)

    def test_partial_word_stable_text_can_be_flagged_before_char_prefix_mismatch(self):
        result = evaluate_scenario(
            build_scenario_by_name("hallucinated_word_queen_clean"),
            LEGACY_CORRELATED_EVIDENCE_CONFIG,
        )

        self.assertGreater(result["falseCommit"]["falseCommitWords"], 0)
        self.assertTrue(result["finalization"]["agreesWithStablePrefix"])
        self.assertIn("Start", result["finalStableText"])
        self.assertIn("clean", result["finalStableText"])
        self.assertNotEqual(result["finalStableText"], result["referenceText"])

    def test_default_policy_reduces_false_commits_across_wide_corpus(self):
        legacy_results = [
            evaluate_scenario(scenario, LEGACY_CORRELATED_EVIDENCE_CONFIG)
            for scenario in build_scenarios()
        ]
        default_results = [evaluate_scenario(scenario) for scenario in build_scenarios()]

        legacy = aggregate_results(legacy_results)
        current = aggregate_results(default_results)

        self.assertGreater(legacy["falseCommitRate"], 0.4)
        self.assertLess(current["falseCommitRate"], 0.05)
        self.assertLess(current["falseCommitWords"], legacy["falseCommitWords"])
        self.assertGreater(current["committedWords"], 0)


class RealtimeTextStabilizerAggregateReportTests(unittest.TestCase):
    def test_aggregate_results_sums_counts_and_rates(self):
        results = [
            evaluate_scenario(build_scenario_by_name("clean_short_repeated_partials")),
            evaluate_scenario(build_scenario_by_name("wrong_number_repeated_without_new_audio_context")),
        ]

        aggregate = aggregate_results(results)

        self.assertEqual(aggregate["scenarioCount"], 2)
        self.assertEqual(
            aggregate["committedWords"],
            sum(result["falseCommit"]["committedWords"] for result in results),
        )
        self.assertEqual(
            aggregate["falseCommitWords"],
            sum(result["falseCommit"]["falseCommitWords"] for result in results),
        )
        self.assertGreaterEqual(aggregate["commitLatencySeconds"]["count"], 1)
        self.assertIn("displayPrefixReference", aggregate["partialWer"])

    def test_aggregate_results_handles_empty_result_list(self):
        aggregate = aggregate_results([])

        self.assertEqual(aggregate["scenarioCount"], 0)
        self.assertEqual(aggregate["committedWords"], 0)
        self.assertEqual(aggregate["falseCommitRate"], 0.0)
        self.assertEqual(aggregate["commitLatencySeconds"]["count"], 0)

    def test_compact_result_keeps_dashboard_fields(self):
        result = evaluate_scenario(build_scenario_by_name("clean_short_repeated_partials"))
        compact = compact_result(result)

        self.assertEqual(compact["name"], result["name"])
        self.assertIn("falseCommitRate", compact)
        self.assertIn("commitLatencyMedianS", compact)
        self.assertIn("displayChangesPerSecond", compact)
        self.assertFalse(compact["finalMismatch"])


class RealtimeTextStabilizerEvaluationCliTests(unittest.TestCase):
    def test_main_lists_cases_without_writing_report(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--list-cases"])

        self.assertEqual(exit_code, 0)
        names = json.loads(stdout.getvalue())
        self.assertIn("clean_short_repeated_partials", names)
        self.assertIn("wrong_number_repeated_without_new_audio_context", names)

    def test_main_writes_single_case_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--case",
                        "clean_short_repeated_partials",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["aggregate"]["scenarioCount"], 1)
            self.assertEqual(report["scenarios"][0]["name"], "clean_short_repeated_partials")
            compact_stdout = json.loads(stdout.getvalue())
            self.assertEqual(compact_stdout["output"], str(output_path))

    def test_main_rejects_unknown_case(self):
        with self.assertRaises(SystemExit):
            main(["--case", "does_not_exist"])


if __name__ == "__main__":
    unittest.main()
