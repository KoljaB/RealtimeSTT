"""Stabilizes realtime transcription text from structured observations.

Callers provide deterministic timestamps; the stabilizer returns events with
stable deltas, unstable preview text, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
import unicodedata


@dataclass(frozen=True)
class RealtimeTextStabilizationConfig:
    """
    Configures evidence thresholds for realtime text stabilization.
    """

    min_char_confirmations: int = 2
    min_char_evidence_span_seconds: float = 0.60
    max_char_evidence_window_seconds: float = 8.00
    space_min_confirmations: int = 4
    space_min_evidence_span_seconds: float = 0.75
    space_requires_stable_right_context: bool = True
    space_right_context_min_chars: int = 2
    punctuation_min_confirmations: int = 4
    punctuation_requires_stable_right_context: bool = True
    initial_prefix_min_confirmations: int = 2
    initial_prefix_min_evidence_span_seconds: float = 0.80
    outlier_similarity_threshold: float = 0.35
    max_single_outlier_gap: int = 1
    max_recent_observations: int = 128
    language_mode: str = "space-tokenized"
    require_audio_progress_for_evidence: bool = True


@dataclass(frozen=True)
class RealtimeTextObservation:
    """
    Captures one realtime transcription observation.
    """

    recording_id: Any
    sequence: int
    raw_text: str
    audio_start_sample: int
    audio_end_sample_exclusive: int
    sample_rate: int
    created_at_monotonic: float
    completed_at_monotonic: float
    segment_id: Optional[Any] = None
    audio_start_time_seconds: Optional[float] = None
    audio_end_time_seconds: Optional[float] = None
    recording_started_at_monotonic: Optional[float] = None
    recording_started_at_wall_time: Optional[float] = None
    received_at_wall_time: Optional[float] = None
    trigger_reason: Optional[str] = None
    display_text: Optional[str] = None
    language: Optional[str] = None
    language_probability: float = 0.0
    engine_name: Optional[str] = None
    model_name: Optional[str] = None
    queue_delay_seconds: Optional[float] = None
    inference_duration_seconds: Optional[float] = None
    total_latency_seconds: Optional[float] = None
    frame_count: Optional[int] = None
    sample_count: Optional[int] = None
    publish_allowed: bool = True
    awaiting_speech_end: bool = False
    boundary_event: Optional[Any] = None
    source: str = "realtime"


@dataclass(frozen=True)
class RealtimeTextFinalObservation:
    """
    Captures a final transcription observation.
    """

    recording_id: Any
    final_text: str
    segment_id: Optional[Any] = None
    sequence: Optional[int] = None
    completed_at_monotonic: Optional[float] = None
    received_at_wall_time: Optional[float] = None


@dataclass(frozen=True)
class RealtimeTextEvidenceDiagnostics:
    """
    Describes evidence used to stabilize text.
    """

    confirmation_count: int = 0
    first_sequence: Optional[int] = None
    latest_sequence: Optional[int] = None
    first_completed_at_monotonic: Optional[float] = None
    latest_completed_at_monotonic: Optional[float] = None
    audio_start_sample_min: Optional[int] = None
    audio_end_sample_max: Optional[int] = None
    contributing_sequence_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class RealtimeTextObservationTiming:
    """
    Carries timing metadata for a realtime observation.
    """

    created_at_monotonic: Optional[float] = None
    completed_at_monotonic: Optional[float] = None
    received_at_wall_time: Optional[float] = None
    queue_delay_seconds: Optional[float] = None
    inference_duration_seconds: Optional[float] = None
    total_latency_seconds: Optional[float] = None


@dataclass(frozen=True)
class RealtimeTextStabilizationEvent:
    """
    Describes the stabilizer result for one realtime observation.
    """

    recording_id: Any
    segment_id: Optional[Any]
    sequence: int
    accepted: bool
    ignored_reason: Optional[str]
    publish_allowed: bool
    should_publish: bool
    raw_observation_text: str
    stable_text: str
    stable_delta: str
    unstable_text: str
    display_text: str
    stable_normalized_offset: int
    stable_raw_end_offset: Optional[int]
    stable_audio_end_sample_exclusive: Optional[int]
    has_new_stable_text: bool
    is_outlier: bool
    stable_prefix_conflict: bool
    commit_reason: str
    evidence: RealtimeTextEvidenceDiagnostics
    timing: RealtimeTextObservationTiming
    trigger_reason: Optional[str] = None
    consensus_text: str = ""
    consensus_delta: str = ""
    consensus_unstable_text: str = ""
    consensus_display_text: str = ""
    consensus_normalized_offset: int = 0
    public_consensus_aligned: bool = True
    internal_revision: bool = False


@dataclass(frozen=True)
class RealtimeTextFinalizationEvent:
    """
    Describes how final text reconciles with stabilized text.
    """

    recording_id: Any
    segment_id: Optional[Any]
    stable_text: str
    final_text: str
    final_suffix_after_stable: str
    agrees_with_stable_prefix: bool
    mismatch_reason: Optional[str]
    stable_text_was_revised: bool = False
    commit_final_remainder: bool = False


@dataclass(frozen=True)
class RealtimeTextStabilizationSnapshot:
    """
    Captures the current realtime stabilization state.
    """

    recording_id: Any
    segment_id: Optional[Any]
    stable_text: str
    unstable_text: str
    display_text: str
    last_sequence: int
    stable_normalized_offset: int
    finalized: bool
    recent_observation_count: int
    ignored_observation_count: int
    dropped_observation_count: int
    consensus_text: str = ""
    consensus_normalized_offset: int = 0
    public_consensus_aligned: bool = True


@dataclass(frozen=True)
class _Projection:
    raw: str
    comparison_text: str
    raw_starts: Tuple[int, ...]
    raw_ends: Tuple[int, ...]


@dataclass(frozen=True)
class _EvidencePoint:
    sequence: int
    completed_at_monotonic: float
    audio_start_sample: int
    audio_end_sample_exclusive: int


@dataclass(frozen=True)
class _ObservationRecord:
    observation: RealtimeTextObservation
    projection: _Projection


class RealtimeTextStabilizer:
    """
    Stabilizes realtime transcription observations into publishable text.
    """

    def __init__(
        self,
        config: Optional[RealtimeTextStabilizationConfig] = None,
    ):
        """
        Initializes stabilizer state with optional threshold configuration.
        """
        self.config = config or RealtimeTextStabilizationConfig()
        self._recording_id = None
        self._segment_id = None
        self._started_at_monotonic = None
        self._started_at_wall_time = None
        self._last_sequence = 0
        self._stable_text = ""
        self._stable_comparison_text = ""
        self._consensus_text = ""
        self._consensus_comparison_text = ""
        self._unstable_text = ""
        self._display_text = ""
        self._evidence: Dict[Tuple[int, str], List[_EvidencePoint]] = {}
        self._recent_accepted: List[_ObservationRecord] = []
        self._recent_ignored: List[_ObservationRecord] = []
        self._outlier_streak: List[_ObservationRecord] = []
        self._dropped_observation_count = 0
        self._unpublished_stable_delta = ""
        self._finalized = False

    def reset(
        self,
        recording_id: Any,
        segment_id: Optional[Any] = None,
        started_at_monotonic: Optional[float] = None,
        started_at_wall_time: Optional[float] = None,
    ) -> None:
        """
        Resets stabilization state for a new recording.
        """
        self._recording_id = recording_id
        self._segment_id = segment_id
        self._started_at_monotonic = started_at_monotonic
        self._started_at_wall_time = started_at_wall_time
        self._last_sequence = 0
        self._stable_text = ""
        self._stable_comparison_text = ""
        self._consensus_text = ""
        self._consensus_comparison_text = ""
        self._unstable_text = ""
        self._display_text = ""
        self._evidence = {}
        self._recent_accepted = []
        self._recent_ignored = []
        self._outlier_streak = []
        self._dropped_observation_count = 0
        self._unpublished_stable_delta = ""
        self._finalized = False

    def observe(
        self,
        observation: RealtimeTextObservation,
    ) -> RealtimeTextStabilizationEvent:
        """
        Processes one realtime observation and returns a stabilization event.
        """
        if self._recording_id is None:
            self.reset(
                observation.recording_id,
                segment_id=observation.segment_id,
                started_at_monotonic=observation.recording_started_at_monotonic,
                started_at_wall_time=observation.recording_started_at_wall_time,
            )

        if observation.recording_id != self._recording_id:
            return self._ignored_event(observation, "wrong-recording")

        if self._finalized:
            return self._ignored_event(observation, "finalized")

        if observation.sequence <= self._last_sequence:
            return self._ignored_event(observation, "stale-sequence")

        self._last_sequence = observation.sequence

        projection = _project_text(observation.raw_text)
        if not projection.raw.strip():
            return self._ignored_event(observation, "empty-text")

        record = _ObservationRecord(observation=observation, projection=projection)
        outlier_decision = self._outlier_decision(record)

        if outlier_decision == "outlier":
            self._outlier_streak.append(record)
            self._remember_ignored(record)
            return self._ignored_event(
                observation,
                "outlier",
                projection=projection,
                is_outlier=True,
                stable_prefix_conflict=self._has_stable_prefix_conflict(projection),
            )

        branch_records: List[_ObservationRecord]
        if outlier_decision == "adopt-branch":
            branch_records = self._outlier_streak + [record]
            self._clear_unstable_branch()
        else:
            branch_records = [record]

        for branch_record in branch_records:
            self._remember_accepted(branch_record)
            self._add_evidence(branch_record)

        self._outlier_streak = []
        event = self._accepted_event(record)
        return event

    def finalize(
        self,
        final_observation: Optional[RealtimeTextFinalObservation] = None,
    ) -> RealtimeTextFinalizationEvent:
        """
        Finalizes stabilization against the completed transcription.
        """
        final_text = ""
        final_suffix = ""
        agrees = True
        mismatch_reason = None
        recording_id = self._recording_id
        segment_id = self._segment_id

        if final_observation is not None:
            recording_id = final_observation.recording_id
            segment_id = final_observation.segment_id or segment_id
            final_text = final_observation.final_text or ""

            if final_observation.recording_id != self._recording_id:
                agrees = False
                mismatch_reason = "wrong-recording"
            else:
                projection = _project_text(final_text)
                if projection.comparison_text.startswith(self._stable_comparison_text):
                    final_suffix = self._raw_suffix_after_stable(projection)
                else:
                    agrees = False
                    mismatch_reason = "stable-prefix-mismatch"

        self._finalized = True
        return RealtimeTextFinalizationEvent(
            recording_id=recording_id,
            segment_id=segment_id,
            stable_text=self._stable_text,
            final_text=final_text,
            final_suffix_after_stable=final_suffix,
            agrees_with_stable_prefix=agrees,
            mismatch_reason=mismatch_reason,
            stable_text_was_revised=False,
            commit_final_remainder=False,
        )

    def snapshot(self) -> RealtimeTextStabilizationSnapshot:
        """
        Returns the current stabilization snapshot.
        """
        return RealtimeTextStabilizationSnapshot(
            recording_id=self._recording_id,
            segment_id=self._segment_id,
            stable_text=self._stable_text,
            unstable_text=self._unstable_text,
            display_text=self._display_text,
            last_sequence=self._last_sequence,
            stable_normalized_offset=len(self._stable_comparison_text),
            finalized=self._finalized,
            recent_observation_count=len(self._recent_accepted),
            ignored_observation_count=len(self._recent_ignored),
            dropped_observation_count=self._dropped_observation_count,
            consensus_text=self._consensus_text,
            consensus_normalized_offset=len(self._consensus_comparison_text),
            public_consensus_aligned=self._public_consensus_aligned(),
        )

    def _accepted_event(
        self,
        record: _ObservationRecord,
    ) -> RealtimeTextStabilizationEvent:
        projection = record.projection
        observation = record.observation
        old_public_frontier = len(self._stable_comparison_text)
        old_consensus_comparison_text = self._consensus_comparison_text
        new_consensus_frontier = self._confirmed_frontier(projection, observation)
        new_consensus_comparison_text = (
            projection.comparison_text[:new_consensus_frontier]
        )
        new_consensus_text = self._build_delta(
            projection,
            0,
            new_consensus_frontier,
        )
        internal_revision = bool(
            old_consensus_comparison_text
            and new_consensus_comparison_text != old_consensus_comparison_text
            and not new_consensus_comparison_text.startswith(
                old_consensus_comparison_text
            )
        )
        consensus_delta = ""
        if new_consensus_comparison_text != old_consensus_comparison_text:
            if new_consensus_comparison_text.startswith(old_consensus_comparison_text):
                consensus_delta = self._build_delta(
                    projection,
                    len(old_consensus_comparison_text),
                    new_consensus_frontier,
                )
            self._consensus_comparison_text = new_consensus_comparison_text
            self._consensus_text = new_consensus_text

        public_consensus_aligned = self._public_consensus_aligned()
        public_delta = ""
        commit_reason = "none"
        evidence = RealtimeTextEvidenceDiagnostics()

        if (
            public_consensus_aligned
            and len(self._consensus_comparison_text) > old_public_frontier
        ):
            public_delta = self._build_delta(
                projection,
                old_public_frontier,
                len(self._consensus_comparison_text),
            )
            self._stable_text += public_delta
            self._stable_comparison_text = self._consensus_comparison_text
            evidence = self._evidence_for_offset(
                len(self._stable_comparison_text) - 1,
                self._stable_comparison_text[-1],
                observation.completed_at_monotonic,
            )
            commit_reason = (
                "space-confirmed"
                if public_delta and public_delta.strip() == ""
                else "evidence-threshold"
            )

        if not self._public_consensus_aligned():
            self._unpublished_stable_delta = ""
            event_delta = ""
        elif observation.publish_allowed:
            event_delta = self._unpublished_stable_delta + public_delta
            self._unpublished_stable_delta = ""
        else:
            event_delta = public_delta
            self._unpublished_stable_delta += public_delta

        consensus_unstable_text = self._raw_suffix_after_comparison(
            projection,
            self._consensus_comparison_text,
        )
        consensus_display_text = self._consensus_text + consensus_unstable_text

        self._unstable_text = self._public_unstable_preview(projection)
        self._display_text = self._stable_text + self._unstable_text

        return RealtimeTextStabilizationEvent(
            recording_id=observation.recording_id,
            segment_id=observation.segment_id or self._segment_id,
            sequence=observation.sequence,
            accepted=True,
            ignored_reason=None,
            publish_allowed=observation.publish_allowed,
            should_publish=observation.publish_allowed,
            raw_observation_text=observation.raw_text,
            stable_text=self._stable_text,
            stable_delta=event_delta,
            unstable_text=self._unstable_text,
            display_text=self._display_text,
            stable_normalized_offset=len(self._stable_comparison_text),
            stable_raw_end_offset=self._stable_raw_end_offset(projection),
            stable_audio_end_sample_exclusive=None,
            has_new_stable_text=bool(event_delta),
            is_outlier=False,
            stable_prefix_conflict=self._has_stable_prefix_conflict(projection),
            commit_reason=commit_reason if event_delta or public_delta else "none",
            evidence=evidence,
            timing=_timing_from_observation(observation),
            trigger_reason=observation.trigger_reason,
            consensus_text=self._consensus_text,
            consensus_delta=consensus_delta,
            consensus_unstable_text=consensus_unstable_text,
            consensus_display_text=consensus_display_text,
            consensus_normalized_offset=len(self._consensus_comparison_text),
            public_consensus_aligned=self._public_consensus_aligned(),
            internal_revision=internal_revision,
        )

    def _ignored_event(
        self,
        observation: RealtimeTextObservation,
        reason: str,
        projection: Optional[_Projection] = None,
        is_outlier: bool = False,
        stable_prefix_conflict: bool = False,
    ) -> RealtimeTextStabilizationEvent:
        stable_raw_end_offset = (
            self._stable_raw_end_offset(projection) if projection is not None else None
        )
        return RealtimeTextStabilizationEvent(
            recording_id=observation.recording_id,
            segment_id=observation.segment_id or self._segment_id,
            sequence=observation.sequence,
            accepted=False,
            ignored_reason=reason,
            publish_allowed=observation.publish_allowed,
            should_publish=False,
            raw_observation_text=observation.raw_text,
            stable_text=self._stable_text,
            stable_delta="",
            unstable_text=self._unstable_text,
            display_text=self._display_text,
            stable_normalized_offset=len(self._stable_comparison_text),
            stable_raw_end_offset=stable_raw_end_offset,
            stable_audio_end_sample_exclusive=None,
            has_new_stable_text=False,
            is_outlier=is_outlier,
            stable_prefix_conflict=stable_prefix_conflict,
            commit_reason="none",
            evidence=RealtimeTextEvidenceDiagnostics(),
            timing=_timing_from_observation(observation),
            trigger_reason=observation.trigger_reason,
            consensus_text=self._consensus_text,
            consensus_delta="",
            consensus_unstable_text=self._unstable_text,
            consensus_display_text=self._display_text,
            consensus_normalized_offset=len(self._consensus_comparison_text),
            public_consensus_aligned=self._public_consensus_aligned(),
            internal_revision=False,
        )

    def _confirmed_frontier(
        self,
        projection: _Projection,
        observation: RealtimeTextObservation,
    ) -> int:
        comparison_text = projection.comparison_text
        frontier = 0
        while frontier < len(comparison_text):
            char = comparison_text[frontier]
            if (
                char == " "
                and self.config.language_mode == "space-tokenized"
            ):
                if not self._space_is_stable(
                    frontier,
                    comparison_text,
                    observation.completed_at_monotonic,
                ):
                    break
            elif not self._char_is_stable(
                frontier,
                char,
                observation.completed_at_monotonic,
            ):
                break
            frontier += 1

        return frontier

    def _char_is_stable(
        self,
        offset: int,
        char: str,
        now: float,
    ) -> bool:
        evidence = self._evidence_for_offset(offset, char, now)
        if not self._stable_comparison_text:
            return _evidence_passes(
                evidence,
                self.config.initial_prefix_min_confirmations,
                self.config.initial_prefix_min_evidence_span_seconds,
            )

        return _evidence_passes(
            evidence,
            self.config.min_char_confirmations,
            self.config.min_char_evidence_span_seconds,
        )

    def _space_is_stable(
        self,
        offset: int,
        comparison_text: str,
        now: float,
    ) -> bool:
        evidence = self._evidence_for_offset(offset, " ", now)
        if not _evidence_passes(
            evidence,
            self.config.space_min_confirmations,
            self.config.space_min_evidence_span_seconds,
        ):
            return False

        if not self.config.space_requires_stable_right_context:
            return True

        stable_right_context = 0
        for index in range(offset + 1, len(comparison_text)):
            char = comparison_text[index]
            if char == " ":
                continue
            if self._char_is_stable(index, char, now):
                stable_right_context += 1
            if stable_right_context >= self.config.space_right_context_min_chars:
                return True

        return False

    def _evidence_for_offset(
        self,
        offset: int,
        char: str,
        now: float,
    ) -> RealtimeTextEvidenceDiagnostics:
        points = self._evidence.get((offset, char), [])
        window = self.config.max_char_evidence_window_seconds
        if window > 0:
            points = [
                point for point in points
                if now - point.completed_at_monotonic <= window
            ]

        if not points:
            return RealtimeTextEvidenceDiagnostics()

        first = min(points, key=lambda point: point.completed_at_monotonic)
        latest = max(points, key=lambda point: point.completed_at_monotonic)
        return RealtimeTextEvidenceDiagnostics(
            confirmation_count=len(points),
            first_sequence=first.sequence,
            latest_sequence=latest.sequence,
            first_completed_at_monotonic=first.completed_at_monotonic,
            latest_completed_at_monotonic=latest.completed_at_monotonic,
            audio_start_sample_min=min(point.audio_start_sample for point in points),
            audio_end_sample_max=max(
                point.audio_end_sample_exclusive for point in points
            ),
            contributing_sequence_ids=tuple(point.sequence for point in points),
        )

    def _add_evidence(self, record: _ObservationRecord) -> None:
        observation = record.observation
        for offset, char in enumerate(record.projection.comparison_text):
            key = (offset, char)
            points = self._evidence.setdefault(key, [])
            if points and points[-1].sequence == observation.sequence:
                continue
            if (
                self.config.require_audio_progress_for_evidence
                and points
                and observation.audio_end_sample_exclusive
                <= max(point.audio_end_sample_exclusive for point in points)
            ):
                continue
            points.append(
                _EvidencePoint(
                    sequence=observation.sequence,
                    completed_at_monotonic=observation.completed_at_monotonic,
                    audio_start_sample=observation.audio_start_sample,
                    audio_end_sample_exclusive=observation.audio_end_sample_exclusive,
                )
            )

    def _build_delta(
        self,
        projection: _Projection,
        old_frontier: int,
        new_frontier: int,
    ) -> str:
        parts: List[str] = []
        for offset in range(old_frontier, new_frontier):
            char = projection.comparison_text[offset]
            if char == " ":
                if parts and parts[-1] == " ":
                    continue
                parts.append(" ")
                continue

            start = projection.raw_starts[offset]
            end = projection.raw_ends[offset]
            parts.append(projection.raw[start:end])

        return "".join(parts).rstrip()

    def _raw_suffix_after_stable(self, projection: _Projection) -> str:
        return self._raw_suffix_after_comparison(
            projection,
            self._stable_comparison_text,
        )

    def _public_unstable_preview(self, projection: _Projection) -> str:
        if not self._stable_comparison_text:
            return projection.raw
        if projection.comparison_text.startswith(self._stable_comparison_text):
            return self._raw_suffix_after_comparison(
                projection,
                self._stable_comparison_text,
            )

        suffix = self._raw_suffix_after_shared_word_prefix(projection)
        if not suffix:
            return ""
        return " ... " + suffix

    def _raw_suffix_after_shared_word_prefix(self, projection: _Projection) -> str:
        shared_offset = _shared_word_prefix_length(
            self._stable_comparison_text,
            projection.comparison_text,
        )
        if shared_offset <= 0:
            return projection.raw.strip()
        if shared_offset > len(projection.raw_ends):
            return ""
        raw_offset = projection.raw_ends[shared_offset - 1]
        return projection.raw[raw_offset:].lstrip()

    def _raw_suffix_after_comparison(
        self,
        projection: _Projection,
        comparison_prefix: str,
    ) -> str:
        if not projection.comparison_text.startswith(comparison_prefix):
            return self._unstable_text
        if not comparison_prefix:
            return projection.raw
        if len(comparison_prefix) > len(projection.raw_ends):
            return self._unstable_text
        raw_offset = projection.raw_ends[len(comparison_prefix) - 1]
        return projection.raw[raw_offset:]

    def _stable_raw_end_offset(
        self,
        projection: Optional[_Projection],
    ) -> Optional[int]:
        if projection is None:
            return None
        if not projection.comparison_text.startswith(self._stable_comparison_text):
            return None
        if not self._stable_comparison_text:
            return 0
        if len(self._stable_comparison_text) > len(projection.raw_ends):
            return None
        return projection.raw_ends[len(self._stable_comparison_text) - 1]

    def _outlier_decision(self, record: _ObservationRecord) -> str:
        comparison_text = record.projection.comparison_text
        if not comparison_text:
            return "accept"

        if self._consensus_comparison_text:
            if comparison_text.startswith(self._consensus_comparison_text):
                return "accept"

        if not self._recent_accepted:
            return "accept"

        for item in self._recent_accepted[-5:]:
            recent_text = item.projection.comparison_text
            if (
                comparison_text.startswith(recent_text)
                or recent_text.startswith(comparison_text)
            ):
                return "accept"

        recent_similarity = max(
            _similarity(comparison_text, item.projection.comparison_text)
            for item in self._recent_accepted[-5:]
        )
        if recent_similarity >= self.config.outlier_similarity_threshold:
            return "accept"

        if not self._consensus_comparison_text and self._outlier_streak:
            streak_similarity = max(
                _similarity(comparison_text, item.projection.comparison_text)
                for item in self._outlier_streak[-3:]
            )
            if (
                streak_similarity >= self.config.outlier_similarity_threshold
                and len(self._outlier_streak) >= self.config.max_single_outlier_gap
            ):
                return "adopt-branch"

        return "outlier"

    def _has_stable_prefix_conflict(self, projection: _Projection) -> bool:
        return bool(
            self._stable_comparison_text
            and not projection.comparison_text.startswith(self._stable_comparison_text)
        )

    def _remember_accepted(self, record: _ObservationRecord) -> None:
        self._recent_accepted.append(record)
        max_recent = max(1, int(self.config.max_recent_observations))
        while len(self._recent_accepted) > max_recent:
            self._recent_accepted.pop(0)
            self._dropped_observation_count += 1

    def _remember_ignored(self, record: _ObservationRecord) -> None:
        self._recent_ignored.append(record)
        max_recent = max(1, int(self.config.max_recent_observations))
        while len(self._recent_ignored) > max_recent:
            self._recent_ignored.pop(0)

    def _clear_unstable_branch(self) -> None:
        self._evidence = {}
        self._recent_accepted = []
        self._consensus_text = ""
        self._consensus_comparison_text = ""
        self._unstable_text = ""
        self._display_text = self._stable_text

    def _public_consensus_aligned(self) -> bool:
        return self._consensus_comparison_text.startswith(
            self._stable_comparison_text
        )


def _project_text(text: str) -> _Projection:
    raw = text or ""
    comparison_chars: List[str] = []
    raw_starts: List[int] = []
    raw_ends: List[int] = []
    previous_was_space = False

    for raw_index, raw_char in enumerate(raw):
        normalized = unicodedata.normalize("NFKC", raw_char)
        for char in normalized:
            if char.isspace():
                if comparison_chars and not previous_was_space:
                    comparison_chars.append(" ")
                    raw_starts.append(raw_index)
                    raw_ends.append(raw_index + 1)
                    previous_was_space = True
                continue

            if _is_punctuation(char):
                continue

            for folded in char.casefold():
                if folded.isspace():
                    if comparison_chars and not previous_was_space:
                        comparison_chars.append(" ")
                        raw_starts.append(raw_index)
                        raw_ends.append(raw_index + 1)
                        previous_was_space = True
                    continue
                if _is_punctuation(folded):
                    continue
                comparison_chars.append(folded)
                raw_starts.append(raw_index)
                raw_ends.append(raw_index + 1)
                previous_was_space = False

    if comparison_chars and comparison_chars[-1] == " ":
        comparison_chars.pop()
        raw_starts.pop()
        raw_ends.pop()

    return _Projection(
        raw=raw,
        comparison_text="".join(comparison_chars),
        raw_starts=tuple(raw_starts),
        raw_ends=tuple(raw_ends),
    )


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _shared_word_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    common = 0
    while common < limit and left[common] == right[common]:
        common += 1

    if common == limit:
        return common

    last_space = left.rfind(" ", 0, common + 1)
    if last_space < 0:
        return 0
    return last_space + 1


def _evidence_passes(
    evidence: RealtimeTextEvidenceDiagnostics,
    min_confirmations: int,
    min_span_seconds: float,
) -> bool:
    if evidence.confirmation_count < min_confirmations:
        return False
    if evidence.first_completed_at_monotonic is None:
        return False
    if evidence.latest_completed_at_monotonic is None:
        return False
    return (
        evidence.latest_completed_at_monotonic
        - evidence.first_completed_at_monotonic
    ) >= min_span_seconds


def _timing_from_observation(
    observation: RealtimeTextObservation,
) -> RealtimeTextObservationTiming:
    return RealtimeTextObservationTiming(
        created_at_monotonic=observation.created_at_monotonic,
        completed_at_monotonic=observation.completed_at_monotonic,
        received_at_wall_time=observation.received_at_wall_time,
        queue_delay_seconds=observation.queue_delay_seconds,
        inference_duration_seconds=observation.inference_duration_seconds,
        total_latency_seconds=observation.total_latency_seconds,
    )
