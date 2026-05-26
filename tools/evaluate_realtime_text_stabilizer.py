"""Replay-based quality evaluation for RealtimeTextStabilizer.

The scenarios in this tool are deterministic ASR partial streams. They are
designed to measure the current stabilizer before changing its behavior:
false commits, commit latency, visible revisions, partial WER, unstable partial
words, and UI update cadence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RealtimeSTT.realtime_text_stabilizer import (
    RealtimeTextFinalObservation,
    RealtimeTextObservation,
    RealtimeTextStabilizationConfig,
    RealtimeTextStabilizer,
)


DEFAULT_OUTPUT = ROOT / "test_outputs" / "realtime_text_stabilizer_baseline.json"
SAMPLE_RATE = 16000
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True)
class ReplayInput:
    text: str
    completed_at_s: float
    audio_end_s: float
    publish_allowed: bool = True
    trigger_reason: str = "replay"


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    category: str
    reference_text: str
    observations: Tuple[ReplayInput, ...]
    word_end_times_s: Tuple[float, ...]
    notes: str


def normalize_word(value: str) -> str:
    value = str(value or "").lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "", value)


def normalized_words(text: str) -> List[str]:
    words = []
    for match in TOKEN_RE.finditer(str(text or "")):
        normalized = normalize_word(match.group(0))
        if normalized:
            words.append(normalized)
    return words


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(expected_text: str, actual_text: str) -> float:
    expected = normalized_words(expected_text)
    actual = normalized_words(actual_text)
    return edit_distance(expected, actual) / max(1, len(expected))


def word_error_rate_words(expected: Sequence[str], actual: Sequence[str]) -> float:
    return edit_distance(expected, actual) / max(1, len(expected))


def common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def timed_reference(
    text: str,
    *,
    first_word_end_s: float = 0.45,
    seconds_per_word: float = 0.34,
) -> Tuple[float, ...]:
    words = normalized_words(text)
    return tuple(
        round(first_word_end_s + index * seconds_per_word, 3)
        for index in range(len(words))
    )


def obs(
    text: str,
    completed_at_s: float,
    audio_end_s: Optional[float] = None,
    *,
    publish_allowed: bool = True,
) -> ReplayInput:
    return ReplayInput(
        text=text,
        completed_at_s=completed_at_s,
        audio_end_s=completed_at_s if audio_end_s is None else audio_end_s,
        publish_allowed=publish_allowed,
    )


def scenario(
    name: str,
    category: str,
    reference_text: str,
    observations: Iterable[ReplayInput],
    notes: str,
    *,
    first_word_end_s: float = 0.45,
    seconds_per_word: float = 0.34,
) -> ReplayScenario:
    return ReplayScenario(
        name=name,
        category=category,
        reference_text=reference_text,
        observations=tuple(observations),
        word_end_times_s=timed_reference(
            reference_text,
            first_word_end_s=first_word_end_s,
            seconds_per_word=seconds_per_word,
        ),
        notes=notes,
    )


def build_scenarios() -> Tuple[ReplayScenario, ...]:
    clean_reference = "Hello world, this is a clean realtime transcription test."
    long_reference = (
        "I would think that the current approach has a lot of potential, "
        "but the transcript should stay flexible until the longer sentence "
        "has enough right context."
    )
    insertion_reference = (
        "Please open the settings panel, pause for a moment, and then continue "
        "recording without splitting the turn."
    )
    long_tail_reference = (
        "When the input gets longer, the realtime text should avoid freezing "
        "early guesses, because a small noisy phrase near the beginning can "
        "shift every word that follows."
    )
    tap_tab_reference = "Please close the tab and save the draft."
    missing_word_reference = "Turn on the living room lights."
    websocket_reference = "Use WebSocket batch size four and keep the stream open."
    color_order_reference = "Move the blue file into the red folder."
    prefix_dropout_reference = "Can you turn the volume down a little."
    hallucination_reference = "Start the clean recording now."
    numeric_slider_reference = "Move slider seven to minus twelve point five decibels."
    long_clean_reference = (
        "The measurement harness should keep long correct hypotheses visible "
        "while still waiting before it calls them stable."
    )
    tail_compound_reference = "Schedule the meeting for Thursday afternoon."

    return (
        scenario(
            "clean_short_repeated_partials",
            "clean",
            clean_reference,
            (
                obs("Hello", 0.65, 0.45),
                obs("Hello world", 1.05, 0.85),
                obs("Hello world, this is", 1.75, 1.55),
                obs("Hello world, this is a clean", 2.30, 2.10),
                obs("Hello world, this is a clean realtime", 3.00, 2.80),
                obs("Hello world, this is a clean realtime transcription", 3.55, 3.35),
                obs("Hello world, this is a clean realtime transcription test.", 4.10, 3.90),
                obs("Hello world, this is a clean realtime transcription test.", 4.55, 3.90),
                obs("Hello world, this is a clean realtime transcription test.", 5.00, 3.90),
            ),
            "Benign growth case used to show the baseline can work on clean repeated partials.",
        ),
        scenario(
            "correlated_wrong_prefix_before_correction",
            "noise",
            "I am now talking into the microphone.",
            (
                obs("M now talking into", 1.20, 1.05),
                obs("M now talking into", 1.60, 1.05),
                obs("M now talking into", 2.05, 1.05),
                obs("I am now talking into", 2.25, 1.80),
                obs("I am now talking into the microphone", 2.85, 2.65),
                obs("I am now talking into the microphone", 3.30, 2.65),
                obs("I am now talking into the microphone", 3.75, 2.65),
            ),
            "A noisy leading phoneme repeats long enough to become public before the corrected branch arrives.",
        ),
        scenario(
            "long_noisy_card_current_branch",
            "long_noise",
            long_reference,
            (
                obs("I would think that the card", 2.05, 1.85),
                obs("I would think that the card", 2.45, 1.85),
                obs("I would think that the card approached us as", 2.90, 2.65),
                obs("I would think that the card approached us as", 3.30, 2.65),
                obs("I would think that the card approached us as a kind of lot of potential", 4.35, 4.10),
                obs("I would think that the current approach has a kind of lot of potential", 4.75, 4.10),
                obs("I would think that the current approach has a lot of potential", 5.15, 4.10),
                obs(
                    "I would think that the current approach has a lot of potential, "
                    "but the transcript should stay flexible",
                    6.65,
                    6.40,
                ),
                obs(
                    "I would think that the current approach has a lot of potential, "
                    "but the transcript should stay flexible until the longer sentence "
                    "has enough right context.",
                    8.95,
                    8.70,
                ),
                obs(
                    "I would think that the current approach has a lot of potential, "
                    "but the transcript should stay flexible until the longer sentence "
                    "has enough right context.",
                    9.45,
                    8.70,
                ),
                obs(
                    "I would think that the current approach has a lot of potential, "
                    "but the transcript should stay flexible until the longer sentence "
                    "has enough right context.",
                    9.95,
                    8.70,
                ),
            ),
            "The known 'card/current approach' shape: a plausible wrong branch freezes and later blocks public recovery.",
            seconds_per_word=0.32,
        ),
        scenario(
            "missing_word_insertion_shifts_long_prefix",
            "long_insertion",
            insertion_reference,
            (
                obs("Please open settings panel", 1.65, 1.45),
                obs("Please open settings panel", 2.05, 1.45),
                obs("Please open settings panel pause for a moment", 3.50, 3.25),
                obs("Please open settings panel pause for a moment", 3.90, 3.25),
                obs("Please open settings panel pause for a moment", 4.30, 3.25),
                obs("Please open the settings panel, pause for a moment", 4.80, 3.25),
                obs("Please open the settings panel, pause for a moment, and then continue", 5.15, 4.90),
                obs(
                    "Please open the settings panel, pause for a moment, and then continue "
                    "recording without splitting the turn.",
                    6.45,
                    6.20,
                ),
                obs(
                    "Please open the settings panel, pause for a moment, and then continue "
                    "recording without splitting the turn.",
                    6.95,
                    6.20,
                ),
                obs(
                    "Please open the settings panel, pause for a moment, and then continue "
                    "recording without splitting the turn.",
                    7.45,
                    6.20,
                ),
            ),
            "A missing early function word shifts all later word positions after the public prefix has grown.",
            seconds_per_word=0.36,
        ),
        scenario(
            "middle_word_substitution_tap_tab",
            "middle_substitution",
            tap_tab_reference,
            (
                obs("Please close the tap and save", 1.95, 1.75),
                obs("Please close the tap and save", 2.35, 1.75),
                obs("Please close the tap and save the draft", 2.80, 2.55),
                obs("Please close the tap and save the draft", 3.25, 2.55),
                obs("Please close the tab and save the draft", 3.70, 2.55),
                obs("Please close the tab and save the draft", 4.15, 2.55),
                obs("Please close the tab and save the draft", 4.60, 2.55),
            ),
            "A middle-word substitution repeats until the wrong word enters stable_text.",
        ),
        scenario(
            "missing_function_word_on_shift",
            "missing_word",
            missing_word_reference,
            (
                obs("Turn the living room lights", 1.55, 1.35),
                obs("Turn the living room lights", 1.95, 1.35),
                obs("Turn the living room lights", 2.40, 1.35),
                obs("Turn the living room lights", 2.85, 1.35),
                obs("Turn on the living room lights", 3.30, 2.25),
                obs("Turn on the living room lights", 3.75, 2.25),
                obs("Turn on the living room lights", 4.20, 2.25),
            ),
            "A missing function word near the front shifts the public stable prefix.",
        ),
        scenario(
            "acronym_compound_website_websocket",
            "compound_substitution",
            websocket_reference,
            (
                obs("Use website batch size four", 1.80, 1.60),
                obs("Use website batch size four", 2.20, 1.60),
                obs("Use website batch size four and keep the stream open", 2.85, 2.65),
                obs("Use website batch size four and keep the stream open", 3.30, 2.65),
                obs("Use WebSocket batch size four and keep the stream open", 3.75, 2.65),
                obs("Use WebSocket batch size four and keep the stream open", 4.20, 2.65),
                obs("Use WebSocket batch size four and keep the stream open", 4.65, 2.65),
            ),
            "A developer-term compound rewrite freezes a common-word substitute.",
        ),
        scenario(
            "word_order_color_swap",
            "word_order",
            color_order_reference,
            (
                obs("Move the red file into the blue folder", 2.15, 1.95),
                obs("Move the red file into the blue folder", 2.55, 1.95),
                obs("Move the red file into the blue folder", 3.00, 1.95),
                obs("Move the red file into the blue folder", 3.45, 1.95),
                obs("Move the blue file into the red folder", 3.90, 2.60),
                obs("Move the blue file into the red folder", 4.35, 2.60),
                obs("Move the blue file into the red folder", 4.80, 2.60),
            ),
            "A word-order/color swap keeps a valid prefix but commits the wrong object.",
        ),
        scenario(
            "prefix_dropout_missing_you",
            "missing_word",
            prefix_dropout_reference,
            (
                obs("Can turn the volume down a little", 1.90, 1.70),
                obs("Can turn the volume down a little", 2.30, 1.70),
                obs("Can turn the volume down a little", 2.75, 1.70),
                obs("Can turn the volume down a little", 3.20, 1.70),
                obs("Can you turn the volume down a little", 3.65, 2.40),
                obs("Can you turn the volume down a little", 4.10, 2.40),
                obs("Can you turn the volume down a little", 4.55, 2.40),
            ),
            "A dropped short word after the first token makes the stable prefix diverge.",
        ),
        scenario(
            "hallucinated_word_queen_clean",
            "hallucination",
            hallucination_reference,
            (
                obs("Start the queen recording now", 1.65, 1.45),
                obs("Start the queen recording now", 2.05, 1.45),
                obs("Start the queen recording now", 2.50, 1.45),
                obs("Start the clean recording now", 2.95, 1.95),
                obs("Start the clean recording now", 3.40, 1.95),
                obs("Start the clean recording now", 3.85, 1.95),
            ),
            "A plausible hallucinated content word becomes stable before the clean branch arrives.",
        ),
        scenario(
            "numeric_slider_seventy_seven",
            "numeric_substitution",
            numeric_slider_reference,
            (
                obs("Move slider seventy to minus twelve point five decibels", 2.95, 2.75),
                obs("Move slider seventy to minus twelve point five decibels", 3.35, 2.75),
                obs("Move slider seventy to minus twelve point five decibels", 3.80, 2.75),
                obs("Move slider seventy to minus twelve point five decibels", 4.25, 2.75),
                obs("Move slider seven to minus twelve point five decibels", 4.70, 3.35),
                obs("Move slider seven to minus twelve point five decibels", 5.15, 3.35),
                obs("Move slider seven to minus twelve point five decibels", 5.60, 3.35),
            ),
            "A numeric command confusion repeats long enough to freeze the wrong number.",
            seconds_per_word=0.30,
        ),
        scenario(
            "compound_word_boundary_churn",
            "word_boundary",
            "The Amazon is the largest rainforest on earth.",
            (
                obs("The Amazon is the largest rain forest", 2.55, 2.35),
                obs("The Amazon is the largest rain forest", 2.95, 2.35),
                obs("The Amazon is the largest rain forest on", 3.30, 2.75),
                obs("The Amazon is the largest rainforest on earth", 3.60, 3.20),
                obs("The Amazon is the largest rainforest on earth", 4.05, 3.20),
                obs("The Amazon is the largest rainforest on earth", 4.50, 3.20),
            ),
            "Word-boundary churn that character voting can expose as committed words before the compound form wins.",
        ),
        scenario(
            "wrong_number_repeated_without_new_audio_context",
            "correlated_repeats",
            "Set the timer for fifteen seconds.",
            (
                obs("Set the timer for six seconds", 1.30, 1.20),
                obs("Set the timer for six seconds", 1.70, 1.20),
                obs("Set the timer for six seconds", 2.15, 1.20),
                obs("Set the timer for six seconds", 2.60, 1.20),
                obs("Set the timer for six seconds", 3.05, 1.20),
                obs("Set the timer for fifteen seconds", 3.50, 2.60),
                obs("Set the timer for fifteen seconds", 3.95, 2.60),
                obs("Set the timer for fifteen seconds", 4.40, 2.60),
            ),
            "Repeated identical partials gain evidence from wall-clock time even though no additional audio context was processed.",
        ),
        scenario(
            "long_noisy_leading_phrase",
            "long_noise",
            long_tail_reference,
            (
                obs("When the input gets logger the real time text should", 3.05, 2.80),
                obs("When the input gets logger the real time text should", 3.45, 2.80),
                obs("When the input gets logger the real time text should avoid freezing", 4.25, 4.00),
                obs("When the input gets logger the real time text should avoid freezing", 4.65, 4.00),
                obs("When the input gets longer, the realtime text should avoid freezing", 5.10, 4.00),
                obs(
                    "When the input gets longer, the realtime text should avoid freezing early guesses, "
                    "because a small noisy phrase near the beginning",
                    7.35,
                    7.10,
                ),
                obs(
                    "When the input gets longer, the realtime text should avoid freezing early guesses, "
                    "because a small noisy phrase near the beginning can shift every word that follows.",
                    9.45,
                    9.20,
                ),
                obs(
                    "When the input gets longer, the realtime text should avoid freezing early guesses, "
                    "because a small noisy phrase near the beginning can shift every word that follows.",
                    9.95,
                    9.20,
                ),
                obs(
                    "When the input gets longer, the realtime text should avoid freezing early guesses, "
                    "because a small noisy phrase near the beginning can shift every word that follows.",
                    10.45,
                    9.20,
                ),
            ),
            "A longer dictation where early similar-sounding words become committed before the corrected longer form arrives.",
            seconds_per_word=0.31,
        ),
        scenario(
            "long_clean_progressive_undercommit",
            "long_clean",
            long_clean_reference,
            (
                obs("The measurement harness", 1.10, 0.90),
                obs("The measurement harness should keep", 1.75, 1.55),
                obs("The measurement harness should keep long correct", 2.45, 2.25),
                obs("The measurement harness should keep long correct hypotheses visible", 3.25, 3.05),
                obs(
                    "The measurement harness should keep long correct hypotheses visible "
                    "while still waiting",
                    4.15,
                    3.95,
                ),
                obs(
                    "The measurement harness should keep long correct hypotheses visible "
                    "while still waiting before it calls them stable.",
                    5.10,
                    4.90,
                ),
            ),
            "A correct but mostly non-repeated long stream measures latency and under-commit without a false prefix.",
            seconds_per_word=0.35,
        ),
        scenario(
            "tail_compound_after_noon_afternoon",
            "word_boundary",
            tail_compound_reference,
            (
                obs("Schedule the meeting for Thursday after noon", 2.10, 1.90),
                obs("Schedule the meeting for Thursday after noon", 2.50, 1.90),
                obs("Schedule the meeting for Thursday after noon", 2.95, 1.90),
                obs("Schedule the meeting for Thursday after noon", 3.40, 1.90),
                obs("Schedule the meeting for Thursday afternoon", 3.85, 2.20),
                obs("Schedule the meeting for Thursday afternoon", 4.30, 2.20),
                obs("Schedule the meeting for Thursday afternoon", 4.75, 2.20),
            ),
            "A tail word-boundary correction shows that even late compounds can become irreversible.",
        ),
    )


def make_observation(
    scenario_name: str,
    replay_input: ReplayInput,
    sequence: int,
) -> RealtimeTextObservation:
    audio_end_sample = int(round(replay_input.audio_end_s * SAMPLE_RATE))
    return RealtimeTextObservation(
        recording_id=scenario_name,
        segment_id=scenario_name,
        sequence=sequence,
        raw_text=replay_input.text,
        audio_start_sample=0,
        audio_end_sample_exclusive=audio_end_sample,
        sample_rate=SAMPLE_RATE,
        created_at_monotonic=max(0.0, replay_input.completed_at_s - 0.02),
        completed_at_monotonic=replay_input.completed_at_s,
        audio_start_time_seconds=0.0,
        audio_end_time_seconds=replay_input.audio_end_s,
        trigger_reason=replay_input.trigger_reason,
        publish_allowed=replay_input.publish_allowed,
    )


def false_commit_summary(
    reference_words: Sequence[str],
    committed_words: Sequence[str],
) -> Dict[str, Any]:
    matching_prefix = common_prefix_length(reference_words, committed_words)
    false_words = max(0, len(committed_words) - matching_prefix)
    return {
        "committedWords": len(committed_words),
        "matchingPrefixWords": matching_prefix,
        "falseCommitWords": false_words,
        "falseCommitRate": round(false_words / max(1, len(committed_words)), 4),
        "committedReferenceCoverage": round(
            matching_prefix / max(1, len(reference_words)),
            4,
        ),
    }


def prefix_partial_wer(reference_words: Sequence[str], display_words: Sequence[str]) -> float:
    if not display_words:
        return 0.0 if not reference_words else 1.0
    expected_prefix = reference_words[: len(display_words)]
    return word_error_rate_words(expected_prefix, display_words)


def evaluate_scenario(
    scenario: ReplayScenario,
    config: Optional[RealtimeTextStabilizationConfig] = None,
) -> Dict[str, Any]:
    stabilizer = RealtimeTextStabilizer(config)
    stabilizer.reset(scenario.name, segment_id=scenario.name)

    reference_words = normalized_words(scenario.reference_text)
    events = []
    display_word_history: List[List[str]] = []
    stable_word_history: List[List[str]] = []
    raw_word_history: List[List[str]] = []

    for sequence, replay_input in enumerate(scenario.observations, start=1):
        event = stabilizer.observe(
            make_observation(scenario.name, replay_input, sequence)
        )
        display_words = normalized_words(event.display_text)
        stable_words = normalized_words(event.stable_text)
        raw_words = normalized_words(event.raw_observation_text)
        display_word_history.append(display_words)
        stable_word_history.append(stable_words)
        raw_word_history.append(raw_words)
        events.append(
            {
                "sequence": event.sequence,
                "completedAtS": replay_input.completed_at_s,
                "audioEndS": replay_input.audio_end_s,
                "accepted": event.accepted,
                "ignoredReason": event.ignored_reason,
                "isOutlier": event.is_outlier,
                "internalRevision": event.internal_revision,
                "stablePrefixConflict": event.stable_prefix_conflict,
                "commitReason": event.commit_reason,
                "stableText": event.stable_text,
                "stableDelta": event.stable_delta,
                "unstableText": event.unstable_text,
                "displayText": event.display_text,
                "rawText": event.raw_observation_text,
                "displayWords": display_words,
                "stableWords": stable_words,
            }
        )

    final_event = stabilizer.finalize(
        RealtimeTextFinalObservation(
            recording_id=scenario.name,
            segment_id=scenario.name,
            final_text=scenario.reference_text,
            sequence=len(scenario.observations) + 1,
            completed_at_monotonic=scenario.observations[-1].completed_at_s
            if scenario.observations
            else None,
        )
    )

    last_stable_words = stable_word_history[-1] if stable_word_history else []
    false_commits = false_commit_summary(reference_words, last_stable_words)

    committed_at: Dict[int, float] = {}
    for event_data, stable_words in zip(events, stable_word_history):
        matching_prefix = common_prefix_length(reference_words, stable_words)
        for index in range(matching_prefix):
            committed_at.setdefault(index, event_data["completedAtS"])

    commit_latencies = []
    negative_commit_latencies = []
    for index, committed_s in committed_at.items():
        if index >= len(scenario.word_end_times_s):
            continue
        latency = committed_s - scenario.word_end_times_s[index]
        commit_latencies.append(latency)
        if latency < 0:
            negative_commit_latencies.append(latency)

    revision_count = 0
    revision_spans = []
    unstable_changed_words = 0
    previous_display_words_total = 0
    display_changes = 0
    raw_display_changes = 0

    for previous_words, current_words in zip(
        display_word_history,
        display_word_history[1:],
    ):
        if current_words != previous_words:
            display_changes += 1
        shared = common_prefix_length(previous_words, current_words)
        changed_span = max(0, len(previous_words) - shared)
        if changed_span:
            revision_count += 1
            revision_spans.append(changed_span)
        unstable_changed_words += changed_span
        previous_display_words_total += len(previous_words)

    for previous_words, current_words in zip(raw_word_history, raw_word_history[1:]):
        if current_words != previous_words:
            raw_display_changes += 1

    partial_wer_values = [
        word_error_rate_words(reference_words, display_words)
        for display_words in display_word_history
    ]
    prefix_partial_wer_values = [
        prefix_partial_wer(reference_words, display_words)
        for display_words in display_word_history
    ]
    raw_partial_wer_values = [
        word_error_rate_words(reference_words, raw_words)
        for raw_words in raw_word_history
    ]

    duration_s = 0.0
    if scenario.observations:
        duration_s = max(
            0.001,
            scenario.observations[-1].completed_at_s
            - scenario.observations[0].completed_at_s,
        )

    stable_update_count = sum(1 for event in events if event["stableDelta"])
    accepted_count = sum(1 for event in events if event["accepted"])
    ignored_count = len(events) - accepted_count
    outlier_count = sum(1 for event in events if event["isOutlier"])
    conflict_count = sum(1 for event in events if event["stablePrefixConflict"])
    internal_revision_count = sum(1 for event in events if event["internalRevision"])

    result = {
        "name": scenario.name,
        "category": scenario.category,
        "notes": scenario.notes,
        "referenceText": scenario.reference_text,
        "referenceWords": reference_words,
        "observationCount": len(scenario.observations),
        "acceptedObservations": accepted_count,
        "ignoredObservations": ignored_count,
        "outlierObservations": outlier_count,
        "stablePrefixConflictEvents": conflict_count,
        "internalRevisionEvents": internal_revision_count,
        "finalStableText": events[-1]["stableText"] if events else "",
        "finalDisplayText": events[-1]["displayText"] if events else "",
        "finalization": {
            "agreesWithStablePrefix": final_event.agrees_with_stable_prefix,
            "mismatchReason": final_event.mismatch_reason,
        },
        "falseCommit": false_commits,
        "commitLatencySeconds": {
            **summary_stats(commit_latencies),
            "negativeCount": len(negative_commit_latencies),
            "committedCorrectWords": len(committed_at),
            "referenceWords": len(reference_words),
            "correctCommitCoverage": round(
                len(committed_at) / max(1, len(reference_words)),
                4,
            ),
        },
        "revisions": {
            "count": revision_count,
            "spanWords": summary_stats(revision_spans),
            "internalRevisionEvents": internal_revision_count,
        },
        "partialWer": {
            "displayFullReference": summary_stats(partial_wer_values),
            "displayPrefixReference": summary_stats(prefix_partial_wer_values),
            "rawFullReference": summary_stats(raw_partial_wer_values),
            "finalDisplayWer": round(
                word_error_rate_words(reference_words, display_word_history[-1]),
                4,
            )
            if display_word_history
            else None,
            "finalStableWer": round(
                word_error_rate_words(reference_words, last_stable_words),
                4,
            ),
        },
        "unstablePartialWordRatio": round(
            unstable_changed_words / max(1, previous_display_words_total),
            4,
        ),
        "uiUpdateFrequency": {
            "durationSeconds": round(duration_s, 4),
            "displayChanges": display_changes,
            "displayChangesPerSecond": round(display_changes / duration_s, 4),
            "stableUpdates": stable_update_count,
            "stableUpdatesPerSecond": round(stable_update_count / duration_s, 4),
            "rawHypothesisChanges": raw_display_changes,
            "rawHypothesisChangesPerSecond": round(
                raw_display_changes / duration_s,
                4,
            ),
        },
        "events": events,
    }
    return result


def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    committed_words = sum(
        result["falseCommit"]["committedWords"] for result in results
    )
    false_words = sum(
        result["falseCommit"]["falseCommitWords"] for result in results
    )
    reference_words = sum(len(result["referenceWords"]) for result in results)
    matching_prefix_words = sum(
        result["falseCommit"]["matchingPrefixWords"] for result in results
    )
    display_changes = sum(
        result["uiUpdateFrequency"]["displayChanges"] for result in results
    )
    duration_s = sum(
        result["uiUpdateFrequency"]["durationSeconds"] for result in results
    )
    revision_count = sum(result["revisions"]["count"] for result in results)
    stable_prefix_conflicts = sum(
        result["stablePrefixConflictEvents"] for result in results
    )
    internal_revisions = sum(result["internalRevisionEvents"] for result in results)
    outliers = sum(result["outlierObservations"] for result in results)
    observations = sum(result["observationCount"] for result in results)

    all_commit_latencies = []
    all_partial_wer = []
    all_prefix_partial_wer = []
    all_revision_spans = []
    unstable_weighted_numerator = 0.0
    unstable_weighted_denominator = 0

    for result in results:
        latency_stats = result["commitLatencySeconds"]
        if latency_stats["count"]:
            # Recompute aggregate percentiles from event-level details.
            reference = result["referenceWords"]
            committed_times = {}
            word_end_times = build_scenario_by_name(result["name"]).word_end_times_s
            for event in result["events"]:
                stable_words = event["stableWords"]
                matching = common_prefix_length(reference, stable_words)
                for index in range(matching):
                    committed_times.setdefault(index, event["completedAtS"])
            for index, committed_s in committed_times.items():
                if index < len(word_end_times):
                    all_commit_latencies.append(committed_s - word_end_times[index])

        for event in result["events"]:
            display_words = event["displayWords"]
            all_partial_wer.append(
                word_error_rate_words(result["referenceWords"], display_words)
            )
            all_prefix_partial_wer.append(
                prefix_partial_wer(result["referenceWords"], display_words)
            )

        all_revision_spans.extend(
            _revision_spans_from_events(result["events"])
        )
        unstable_weighted_numerator += result["unstablePartialWordRatio"] * max(
            1,
            sum(len(event["displayWords"]) for event in result["events"][:-1]),
        )
        unstable_weighted_denominator += max(
            1,
            sum(len(event["displayWords"]) for event in result["events"][:-1]),
        )

    return {
        "scenarioCount": len(results),
        "observationCount": observations,
        "referenceWords": reference_words,
        "committedWords": committed_words,
        "matchingCommittedPrefixWords": matching_prefix_words,
        "falseCommitWords": false_words,
        "falseCommitRate": round(false_words / max(1, committed_words), 4),
        "correctCommitCoverage": round(
            matching_prefix_words / max(1, reference_words),
            4,
        ),
        "commitLatencySeconds": summary_stats(all_commit_latencies),
        "revisionCount": revision_count,
        "revisionSpanWords": summary_stats(all_revision_spans),
        "partialWer": {
            "displayFullReference": summary_stats(all_partial_wer),
            "displayPrefixReference": summary_stats(all_prefix_partial_wer),
        },
        "unstablePartialWordRatio": round(
            unstable_weighted_numerator / max(1, unstable_weighted_denominator),
            4,
        ),
        "uiUpdateFrequency": {
            "durationSeconds": round(duration_s, 4),
            "displayChanges": display_changes,
            "displayChangesPerSecond": round(display_changes / max(0.001, duration_s), 4),
        },
        "stablePrefixConflictEvents": stable_prefix_conflicts,
        "internalRevisionEvents": internal_revisions,
        "outlierObservations": outliers,
    }


def _revision_spans_from_events(events: Sequence[Dict[str, Any]]) -> List[int]:
    spans = []
    for previous, current in zip(events, events[1:]):
        previous_words = previous["displayWords"]
        current_words = current["displayWords"]
        shared = common_prefix_length(previous_words, current_words)
        changed_span = max(0, len(previous_words) - shared)
        if changed_span:
            spans.append(changed_span)
    return spans


def build_scenario_by_name(name: str) -> ReplayScenario:
    scenarios = {scenario.name: scenario for scenario in build_scenarios()}
    return scenarios[name]


def compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": result["name"],
        "category": result["category"],
        "falseCommitRate": result["falseCommit"]["falseCommitRate"],
        "falseCommitWords": result["falseCommit"]["falseCommitWords"],
        "committedWords": result["falseCommit"]["committedWords"],
        "correctCommitCoverage": result["commitLatencySeconds"]["correctCommitCoverage"],
        "commitLatencyMedianS": result["commitLatencySeconds"]["median"],
        "commitLatencyP95S": result["commitLatencySeconds"]["p95"],
        "revisionCount": result["revisions"]["count"],
        "maxRevisionSpanWords": result["revisions"]["spanWords"]["max"],
        "prefixPartialWerMean": result["partialWer"]["displayPrefixReference"]["mean"],
        "unstablePartialWordRatio": result["unstablePartialWordRatio"],
        "displayChangesPerSecond": result["uiUpdateFrequency"]["displayChangesPerSecond"],
        "stablePrefixConflictEvents": result["stablePrefixConflictEvents"],
        "internalRevisionEvents": result["internalRevisionEvents"],
        "finalMismatch": not result["finalization"]["agreesWithStablePrefix"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RealtimeTextStabilizer on deterministic replay streams."
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run one scenario by name. Can be repeated. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available scenario names and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the full JSON report.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the full JSON report instead of a compact summary.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scenarios = build_scenarios()
    if args.list_cases:
        print(json.dumps([scenario.name for scenario in scenarios], indent=2))
        return 0

    if args.case:
        selected = set(args.case)
        missing = sorted(selected - {scenario.name for scenario in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}")
        scenarios = tuple(scenario for scenario in scenarios if scenario.name in selected)

    results = [evaluate_scenario(scenario) for scenario in scenarios]
    report = {
        "kind": "realtime_text_stabilizer_replay_evaluation",
        "stabilizer": "RealtimeSTT.realtime_text_stabilizer.RealtimeTextStabilizer",
        "config": RealtimeTextStabilizationConfig().__dict__,
        "metrics": {
            "falseCommitRate": (
                "Committed words after the first committed/reference prefix mismatch "
                "divided by total committed words."
            ),
            "commitLatencySeconds": (
                "For correctly committed reference-prefix words, observation completion "
                "time minus reference word end time."
            ),
            "revisionCountSpan": (
                "Visible display updates that replace or remove previously displayed "
                "words, with span measured in words from the first changed position."
            ),
            "partialWer": (
                "WER of each display text against the full reference and against the "
                "same-length reference prefix."
            ),
            "unstablePartialWordRatio": (
                "Displayed words that are replaced or removed by the next display update "
                "divided by previously displayed words."
            ),
            "uiUpdateFrequency": "Display changes per replay second.",
        },
        "aggregate": aggregate_results(results),
        "scenarios": results,
        "compactScenarios": [compact_result(result) for result in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "aggregate": report["aggregate"],
                    "compactScenarios": report["compactScenarios"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
