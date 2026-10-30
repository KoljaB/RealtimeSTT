"""Sticky merge state for two concurrent realtime ASR streams.

The 1120 ms stream is the authoritative (slow) stream.  An optional
ultrafast stream can only contribute a bounded suffix after an anchor in the
slow text.  Once that suffix has been shown, it is sticky until the slow
stream advances to a new normalized hypothesis.

This module intentionally contains no recorder or worker integration.  It is
the small, deterministic state machine used by the integration layer.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .tail_transcription import (
    _is_close_word_match,
    _is_partial_word_match,
    _levenshtein_distance,
    _tokenize_words,
    find_tail_anchor,
)


_REALTIME_MIN_ANCHOR_WORDS = 2
_REALTIME_MIN_CLOSE_WORD_LENGTH = 3
_REALTIME_SOFT_MIN_SUPPORT = 2
_REALTIME_SOFT_MAX_GAPS = 4
_REALTIME_SOFT_SLOW_TAIL_WORDS = 8

Word = Tuple[str, str]


@dataclass(frozen=True)
class RealtimeTranscriptionMergeResult:
    """Immutable observation returned by the sticky realtime merger."""

    text: str = ""
    slow_text: str = ""
    ultrafast_text: str = ""
    ultrafast_suffix: str = ""
    status: str = "waiting_for_slow"
    recording_id: Any = 0
    slow_generation: int = 0
    slow_sequence: int = 0
    ultrafast_sequence: int = 0
    matched: bool = False
    held: bool = False
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    distance: int = 0
    should_publish: bool = False
    slow_audio_end_sample_exclusive: Optional[int] = None
    ultrafast_audio_end_sample_exclusive: Optional[int] = None


@dataclass(frozen=True)
class _Alignment:
    """Internal alignment plus the raw words available after its splice."""

    matched: bool
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    distance: int = 0
    suffix_words: Tuple[Word, ...] = ()
    bootstrap: bool = False
    soft_frontier: bool = False


@dataclass(frozen=True)
class _SoftFrontierState:
    """Best monotonic overlap ending at one slow/fast word pair."""

    slow_index: int
    fast_index: int
    score: int
    support_count: int
    gap_count: int
    distance: int
    pairs: Tuple[Tuple[int, int], ...] = ()


class StickyRealtimeTranscriptionMerger:
    """Merge an authoritative realtime stream with a sticky fast suffix.

    ``observe_slow`` and ``observe_ultrafast`` are deliberately synchronous.
    Callers may invoke them from separate workers as long as they serialize
    access to one merger instance (the recorder integration does that at its
    event-dispatch boundary).

    The merger never rewrites a suffix already accepted during a slow
    generation.  A new normalized slow hypothesis is the only operation that
    clears that suffix.
    """

    def __init__(
        self,
        max_ultrafast_tail_words: int = 5,
        fuzzy_confirmation_count: int = 1,
    ) -> None:
        try:
            max_ultrafast_tail_words = int(max_ultrafast_tail_words)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_ultrafast_tail_words must be positive") from exc
        try:
            fuzzy_confirmation_count = int(fuzzy_confirmation_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("fuzzy_confirmation_count must be positive") from exc
        if max_ultrafast_tail_words < 1:
            raise ValueError("max_ultrafast_tail_words must be positive")
        if fuzzy_confirmation_count < 1:
            raise ValueError("fuzzy_confirmation_count must be positive")

        self.max_ultrafast_tail_words = max_ultrafast_tail_words
        self.fuzzy_confirmation_count = fuzzy_confirmation_count
        self._recording_id: Any = None
        self._slow_text = ""
        self._slow_normalized = ""
        self._slow_sequence: Optional[int] = None
        self._slow_audio_end_sample_exclusive: Optional[int] = None
        self._ultrafast_text = ""
        self._ultrafast_sequence: Optional[int] = None
        self._ultrafast_audio_end_sample_exclusive: Optional[int] = None
        self._slow_generation = 0
        self._accepted_suffix_words: List[Word] = []
        self._pending_fuzzy: Optional[_Alignment] = None
        self._pending_fuzzy_count = 0
        self._status = "waiting_for_slow"
        self._last_matched = False
        self._last_used_fuzzy_match = False
        self._last_anchor_length = 0
        self._last_distance = 0

    def reset(self, recording_id: Any = 0) -> RealtimeTranscriptionMergeResult:
        """Start a fresh merge generation for ``recording_id``."""

        self._recording_id = 0 if recording_id is None else recording_id
        self._slow_text = ""
        self._slow_normalized = ""
        self._slow_sequence = None
        self._slow_audio_end_sample_exclusive = None
        self._ultrafast_text = ""
        self._ultrafast_sequence = None
        self._ultrafast_audio_end_sample_exclusive = None
        self._slow_generation = 0
        self._accepted_suffix_words = []
        self._pending_fuzzy = None
        self._pending_fuzzy_count = 0
        self._status = "waiting_for_slow"
        self._last_matched = False
        self._last_used_fuzzy_match = False
        self._last_anchor_length = 0
        self._last_distance = 0
        return self.snapshot()

    def observe_slow(
        self,
        text: str,
        recording_id: Any = None,
        sequence: Optional[int] = None,
        audio_end_sample_exclusive: Optional[int] = None,
    ) -> RealtimeTranscriptionMergeResult:
        """Observe a new 1120 ms hypothesis.

        A normalized-content change starts a new slow generation.  The latest
        cached ultrafast text is then aligned against the new slow text so a
        concurrently completed fast result is not lost.
        """

        if not self._accept_recording_id(recording_id):
            return self._stale_result()

        next_sequence = self._next_sequence(sequence, self._slow_sequence)
        if self._is_stale(next_sequence, self._slow_sequence):
            return self._stale_result()

        raw_text = _raw_text(text)
        normalized_text = _normalized_text(raw_text)
        old_merged_text = self._merged_text()
        had_slow_text = bool(self._slow_normalized)
        content_changed = normalized_text != self._slow_normalized

        self._slow_text = raw_text
        self._slow_normalized = normalized_text
        self._slow_sequence = next_sequence
        self._slow_audio_end_sample_exclusive = audio_end_sample_exclusive

        if content_changed:
            self._slow_generation += 1
            self._accepted_suffix_words = []
            self._clear_pending_fuzzy()

            alignment = self._align_cached_ultrafast()
            if alignment.matched and alignment.suffix_words:
                status, matched, fuzzy, anchor_length, distance = (
                    self._apply_alignment(alignment)
                )
                if status == "held_fuzzy_pending":
                    # A slow advance has already changed the authoritative
                    # text, so the pending fast candidate is not holding the
                    # previous generation's tail.
                    status = "held_fuzzy_pending"
            elif alignment.matched:
                status = "bootstrap_exact" if alignment.bootstrap else "slow_only"
                matched = True
                fuzzy = alignment.used_fuzzy_match
                anchor_length = alignment.anchor_length
                distance = alignment.distance
            else:
                status = "slow_only" if not had_slow_text else "slow_advanced_no_anchor"
                matched = False
                fuzzy = False
                anchor_length = 0
                distance = 0
        else:
            # Case, punctuation, and whitespace changes are safe formatting
            # updates.  They do not invalidate an already accepted fast tail.
            status = "slow_only"
            matched = bool(self._accepted_suffix_words)
            fuzzy = False
            anchor_length = 0
            distance = 0

        merged_text = self._merged_text()
        return self._remembered_result(
            status=status,
            matched=matched,
            held=status.startswith("held_"),
            used_fuzzy_match=fuzzy,
            anchor_length=anchor_length,
            distance=distance,
            should_publish=merged_text != old_merged_text,
        )

    def observe_ultrafast(
        self,
        text: str,
        recording_id: Any = None,
        sequence: Optional[int] = None,
        audio_end_sample_exclusive: Optional[int] = None,
    ) -> RealtimeTranscriptionMergeResult:
        """Observe a new ultrafast hypothesis and merge only safe additions."""

        if not self._accept_recording_id(recording_id):
            return self._stale_result()

        next_sequence = self._next_sequence(sequence, self._ultrafast_sequence)
        if self._is_stale(next_sequence, self._ultrafast_sequence):
            return self._stale_result()

        old_merged_text = self._merged_text()
        self._ultrafast_text = _raw_text(text)
        self._ultrafast_sequence = next_sequence
        self._ultrafast_audio_end_sample_exclusive = audio_end_sample_exclusive

        if not self._slow_normalized:
            self._clear_pending_fuzzy()
            return self._remembered_result(
                status="waiting_for_slow",
                matched=False,
                held=False,
                used_fuzzy_match=False,
                anchor_length=0,
                distance=0,
                should_publish=False,
            )

        alignment = self._align_cached_ultrafast()
        if not alignment.matched:
            self._clear_pending_fuzzy()
            status = "held_no_anchor"
            matched = False
            fuzzy = False
            anchor_length = 0
            distance = 0
        elif not alignment.suffix_words:
            # An aligned fast result with no new word is not permission to
            # retract a suffix that was already displayed.
            self._clear_pending_fuzzy()
            if self._accepted_suffix_words:
                status = "held_fast_conflict"
                matched = False
            else:
                status = "bootstrap_exact" if alignment.bootstrap else "slow_only"
                matched = True
            fuzzy = alignment.used_fuzzy_match
            anchor_length = alignment.anchor_length
            distance = alignment.distance
        else:
            status, matched, fuzzy, anchor_length, distance = (
                self._apply_alignment(alignment)
            )

        merged_text = self._merged_text()
        return self._remembered_result(
            status=status,
            matched=matched,
            held=status.startswith("held_"),
            used_fuzzy_match=fuzzy,
            anchor_length=anchor_length,
            distance=distance,
            should_publish=merged_text != old_merged_text,
        )

    def snapshot(self) -> RealtimeTranscriptionMergeResult:
        """Return the current merged state without publishing a new event."""

        return self._result(
            status=self._status,
            matched=self._last_matched,
            held=self._status.startswith("held_"),
            used_fuzzy_match=self._last_used_fuzzy_match,
            anchor_length=self._last_anchor_length,
            distance=self._last_distance,
            should_publish=False,
        )

    def _accept_recording_id(self, recording_id: Any) -> bool:
        if self._recording_id is None:
            self._recording_id = 0 if recording_id is None else recording_id
            return True
        if recording_id is None:
            return True
        return recording_id == self._recording_id

    @staticmethod
    def _next_sequence(
        sequence: Optional[int], previous: Optional[int]
    ) -> int:
        if sequence is None:
            return 1 if previous is None else previous + 1
        try:
            return int(sequence)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequence must be an integer") from exc

    @staticmethod
    def _is_stale(sequence: int, previous: Optional[int]) -> bool:
        return previous is not None and sequence <= previous

    def _align_cached_ultrafast(self) -> _Alignment:
        if not self._slow_normalized or not self._ultrafast_text:
            return _Alignment(matched=False)

        slow_words = _tokenize_words(self._slow_text)
        fast_words = _tokenize_words(self._ultrafast_text)
        if (
            self._slow_generation == 1
            and len(slow_words) == 1
            and fast_words
            and fast_words[0][1] == slow_words[0][1]
        ):
            return _Alignment(
                matched=True,
                anchor_length=1,
                distance=0,
                suffix_words=tuple(
                    fast_words[1 : 1 + self.max_ultrafast_tail_words]
                ),
                bootstrap=True,
            )

        alignment = find_tail_anchor(
            self._slow_text,
            self._ultrafast_text,
            allow_fuzzy=True,
            min_anchor_words=_REALTIME_MIN_ANCHOR_WORDS,
            min_close_word_length=_REALTIME_MIN_CLOSE_WORD_LENGTH,
        )
        # The shared fuzzy matcher permits a one-token-short window as a
        # Preview repair.  That is too permissive for a realtime frontier:
        # it can align an arbitrary preceding word and duplicate text.  Let
        # the cumulative soft matcher decide these cases instead.
        if alignment is not None:
            start, end, anchor_length, used_fuzzy_match, _ = alignment
            if used_fuzzy_match and end - start < anchor_length:
                alignment = None

        # A two-word same-length fuzzy window is not enough realtime evidence:
        # a common word can make an unrelated window look aligned (for
        # example ``we cat``/``we bat`` or ``the mat``/``the cat``).  The
        # cumulative soft matcher below can still accept it when a longer
        # monotonic context provides at least two strong supports.
        if alignment is not None:
            start, end, anchor_length, used_fuzzy_match, _ = alignment
            if used_fuzzy_match and end - start == anchor_length == 2:
                alignment = None

        if alignment is None:
            soft = _find_soft_frontier_alignment(slow_words, fast_words)
            if soft is None:
                return _Alignment(matched=False)
            fast_end, support_count, distance = soft
            return _Alignment(
                matched=True,
                used_fuzzy_match=True,
                anchor_length=support_count,
                distance=distance,
                suffix_words=tuple(
                    fast_words[fast_end : fast_end + self.max_ultrafast_tail_words]
                ),
                soft_frontier=True,
            )

        start, end, anchor_length, used_fuzzy_match, distance = alignment
        suffix_words = tuple(
            fast_words[end : end + self.max_ultrafast_tail_words]
        )
        return _Alignment(
            matched=True,
            used_fuzzy_match=used_fuzzy_match,
            anchor_length=anchor_length,
            distance=distance,
            suffix_words=suffix_words,
            bootstrap=False,
        )

    def _apply_alignment(
        self, alignment: _Alignment
    ) -> Tuple[str, bool, bool, int, int]:
        """Apply one non-empty alignment without ever retracting a suffix."""

        candidate_words = alignment.suffix_words
        candidate_normalized = tuple(word[1] for word in candidate_words)
        held_normalized = tuple(word[1] for word in self._accepted_suffix_words)

        if held_normalized and not _is_prefix(held_normalized, candidate_normalized):
            self._clear_pending_fuzzy()
            return (
                "held_fast_conflict",
                False,
                alignment.used_fuzzy_match,
                alignment.anchor_length,
                alignment.distance,
            )

        if alignment.used_fuzzy_match:
            if self._fuzzy_candidate_conflicts_with_pending(alignment):
                self._pending_fuzzy = alignment
                self._pending_fuzzy_count = 1
            elif self._pending_fuzzy is None:
                self._pending_fuzzy = alignment
                self._pending_fuzzy_count = 1
            else:
                self._pending_fuzzy_count += 1

            if self._pending_fuzzy_count < self.fuzzy_confirmation_count:
                return (
                    "held_fuzzy_pending",
                    False,
                    True,
                    alignment.anchor_length,
                    alignment.distance,
                )

        self._clear_pending_fuzzy()
        if held_normalized:
            # Preserve the raw spelling/case/punctuation of already published
            # words and append only genuinely new candidate words.
            new_words = candidate_words[len(self._accepted_suffix_words) :]
            if new_words:
                self._accepted_suffix_words.extend(new_words)
        else:
            self._accepted_suffix_words = list(candidate_words)

        return (
            "bootstrap_exact"
            if alignment.bootstrap
            else (
                "soft_frontier"
                if alignment.soft_frontier
                else ("fuzzy" if alignment.used_fuzzy_match else "exact")
            ),
            True,
            alignment.used_fuzzy_match,
            alignment.anchor_length,
            alignment.distance,
        )

    def _fuzzy_candidate_conflicts_with_pending(
        self, alignment: _Alignment
    ) -> bool:
        if self._pending_fuzzy is None:
            return False
        previous_words = self._pending_fuzzy.suffix_words
        current_words = alignment.suffix_words
        previous_normalized = tuple(word[1] for word in previous_words)
        current_normalized = tuple(word[1] for word in current_words)
        # A candidate may stay the same or append words, but may not regress.
        return not _is_prefix(previous_normalized, current_normalized)

    def _clear_pending_fuzzy(self) -> None:
        self._pending_fuzzy = None
        self._pending_fuzzy_count = 0

    def _merged_text(self) -> str:
        if not self._slow_text:
            return ""
        suffix = " ".join(raw_word for raw_word, _ in self._accepted_suffix_words)
        if not suffix:
            return self._slow_text
        return self._slow_text + " " + suffix

    def _stale_result(self) -> RealtimeTranscriptionMergeResult:
        return self._result(
            status="stale_ignored",
            matched=False,
            held=False,
            used_fuzzy_match=False,
            anchor_length=0,
            distance=0,
            should_publish=False,
        )

    def _remembered_result(
        self,
        *,
        status: str,
        matched: bool,
        held: bool,
        used_fuzzy_match: bool,
        anchor_length: int,
        distance: int,
        should_publish: bool,
    ) -> RealtimeTranscriptionMergeResult:
        self._status = status
        self._last_matched = matched
        self._last_used_fuzzy_match = used_fuzzy_match
        self._last_anchor_length = anchor_length
        self._last_distance = distance
        return self._result(
            status=status,
            matched=matched,
            held=held,
            used_fuzzy_match=used_fuzzy_match,
            anchor_length=anchor_length,
            distance=distance,
            should_publish=should_publish,
        )

    def _result(
        self,
        *,
        status: str,
        matched: bool,
        held: bool,
        used_fuzzy_match: bool,
        anchor_length: int,
        distance: int,
        should_publish: bool,
    ) -> RealtimeTranscriptionMergeResult:
        return RealtimeTranscriptionMergeResult(
            text=self._merged_text(),
            slow_text=self._slow_text,
            ultrafast_text=self._ultrafast_text,
            ultrafast_suffix=" ".join(
                raw_word for raw_word, _ in self._accepted_suffix_words
            ),
            status=status,
            recording_id=0 if self._recording_id is None else self._recording_id,
            slow_generation=self._slow_generation,
            slow_sequence=0 if self._slow_sequence is None else self._slow_sequence,
            ultrafast_sequence=(
                0
                if self._ultrafast_sequence is None
                else self._ultrafast_sequence
            ),
            matched=matched,
            held=held,
            used_fuzzy_match=used_fuzzy_match,
            anchor_length=anchor_length,
            distance=distance,
            should_publish=should_publish,
            slow_audio_end_sample_exclusive=self._slow_audio_end_sample_exclusive,
            ultrafast_audio_end_sample_exclusive=(
                self._ultrafast_audio_end_sample_exclusive
            ),
        )


def _raw_text(text: Any) -> str:
    return str(text or "").strip()


def _normalized_text(text: str) -> str:
    return " ".join(normalized for _, normalized in _tokenize_words(text))


def _is_prefix(prefix: Tuple[str, ...], value: Tuple[str, ...]) -> bool:
    return len(value) >= len(prefix) and value[: len(prefix)] == prefix


def _soft_word_relation(
    left: str, right: str
) -> Optional[Tuple[int, int, bool]]:
    """Return ``(score, distance, meaningful)`` for a soft word match.

    Realtime hypotheses often expose a partial token (``de``/``dealing`` or
    ``visual``/``visu``) and occasionally differ by one ASR spelling edit.
    The relation deliberately stays narrow: a prefix or one-character edit is
    evidence, while arbitrary substitutions are not.
    """

    if left == right:
        return 4, 0, min(len(left), len(right)) >= 2

    if min(len(left), len(right)) >= 2 and (
        left.startswith(right) or right.startswith(left)
    ):
        return 3, 1, True

    if _is_close_word_match(
        left,
        right,
        min_close_word_length=_REALTIME_MIN_CLOSE_WORD_LENGTH,
    ):
        return 2, _levenshtein_distance(left, right), True

    # Keep this explicit even though the prefix branch above handles the
    # current cases; it documents the accepted partial-word direction and
    # guards against future changes to the shared helper.
    if _is_partial_word_match(left, right) or _is_partial_word_match(
        right, left
    ):
        return 3, 1, True
    return None


def _soft_state_is_better(
    candidate: _SoftFrontierState,
    current: Optional[_SoftFrontierState],
) -> bool:
    if current is None:
        return True
    candidate_key = (
        candidate.score,
        candidate.support_count,
        -candidate.distance,
        len(candidate.pairs),
        candidate.fast_index,
    )
    current_key = (
        current.score,
        current.support_count,
        -current.distance,
        len(current.pairs),
        current.fast_index,
    )
    return candidate_key > current_key


def _find_soft_frontier_alignment(
    slow_words: Sequence[Word], fast_words: Sequence[Word]
) -> Optional[Tuple[int, int, int]]:
    """Find a low-risk monotonic frontier at the end of slow text.

    This is intentionally a fallback after the normal tail-anchor matcher.
    It aligns cumulative hypotheses with a bounded number of skipped tokens,
    requires the final slow token to participate, and only returns a suffix
    after at least two meaningful monotonic word matches.  The fast word that
    matches the final slow frontier is consumed as overlap; only words after
    it become speculative output.
    """

    if not slow_words or not fast_words:
        return None

    # Only the recent slow frontier is relevant for deciding where speculative
    # fast text starts.  Older words may differ substantially after a long
    # turn and should not consume the bounded edit budget.
    slow_words = slow_words[-_REALTIME_SOFT_SLOW_TAIL_WORDS:]
    slow_count = len(slow_words)
    fast_count = len(fast_words)
    max_gaps = min(
        _REALTIME_SOFT_MAX_GAPS,
        slow_count + fast_count,
    )
    # dp[i][j][g] is the best alignment of the first i slow and j fast words
    # using exactly g unmatched words before the next match.
    dp: List[List[List[Optional[_SoftFrontierState]]]] = [
        [
            [None for _ in range(max_gaps + 1)]
            for _ in range(fast_count + 1)
        ]
        for _ in range(slow_count + 1)
    ]
    # Fast hypotheses can begin with a few words that the slow stream has not
    # emitted yet.  Treat that leading fast prefix as outside the overlap;
    # gaps between supporting matches remain bounded below.
    for fast_prefix in range(fast_count + 1):
        dp[0][fast_prefix][0] = _SoftFrontierState(
            -1,
            fast_prefix - 1,
            0,
            0,
            0,
            0,
        )

    def consider(
        i: int,
        j: int,
        gaps: int,
        state: _SoftFrontierState,
    ) -> None:
        if gaps > max_gaps:
            return
        current = dp[i][j][gaps]
        if _soft_state_is_better(state, current):
            dp[i][j][gaps] = state

    for i in range(slow_count + 1):
        for j in range(fast_count + 1):
            for gaps in range(max_gaps + 1):
                state = dp[i][j][gaps]
                if state is None:
                    continue

                if i < slow_count:
                    consider(
                        i + 1,
                        j,
                        gaps + 1,
                        _SoftFrontierState(
                            state.slow_index,
                            state.fast_index,
                            state.score,
                            state.support_count,
                            gaps + 1,
                            state.distance + 1,
                            state.pairs,
                        ),
                    )
                if j < fast_count:
                    consider(
                        i,
                        j + 1,
                        gaps + 1,
                        _SoftFrontierState(
                            state.slow_index,
                            state.fast_index,
                            state.score,
                            state.support_count,
                            gaps + 1,
                            state.distance + 1,
                            state.pairs,
                        ),
                    )
                if i >= slow_count or j >= fast_count:
                    continue

                relation = _soft_word_relation(
                    slow_words[i][1], fast_words[j][1]
                )
                if relation is None:
                    continue
                score, distance, meaningful = relation
                consider(
                    i + 1,
                    j + 1,
                    gaps,
                    _SoftFrontierState(
                        i,
                        j,
                        state.score + score,
                        state.support_count + int(meaningful),
                        gaps,
                        state.distance + distance,
                        state.pairs + ((i, j),),
                    ),
                )

    candidates: List[_SoftFrontierState] = []
    final_slow_word = slow_words[-1][1]
    for fast_index, fast_word in enumerate(fast_words):
        relation = _soft_word_relation(final_slow_word, fast_word[1])
        if relation is None:
            continue
        score, distance, meaningful = relation
        for gaps in range(max_gaps + 1):
            prefix = dp[slow_count - 1][fast_index][gaps]
            if prefix is None:
                continue
            candidate = _SoftFrontierState(
                slow_count - 1,
                fast_index,
                prefix.score + score,
                prefix.support_count + int(meaningful),
                gaps,
                prefix.distance + distance,
                prefix.pairs + ((slow_count - 1, fast_index),),
            )
            if candidate.support_count < _REALTIME_SOFT_MIN_SUPPORT:
                continue
            strong_support = sum(
                min(
                    len(slow_words[slow_index][1]),
                    len(fast_words[fast_index][1]),
                )
                >= _REALTIME_MIN_CLOSE_WORD_LENGTH
                for slow_index, fast_index in candidate.pairs
            )
            if strong_support < _REALTIME_SOFT_MIN_SUPPORT:
                continue
            if candidate.score < 2 * 3:
                continue
            candidates.append(candidate)

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.support_count,
            -candidate.distance,
            -candidate.gap_count,
            candidate.fast_index,
        ),
    )
    return best.fast_index + 1, best.support_count, best.distance
