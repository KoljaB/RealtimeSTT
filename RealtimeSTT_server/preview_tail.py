"""Bounded-tail Preview decoding anchored on already-recognized turn text.

A full-turn Preview re-decodes the complete retained turn at every pause, so
its cost grows linearly with the turn.  Tail mode decodes only the audio after
an anchor (the previous Preview of the same turn, or the accurate Live
hypothesis) plus an overlap, then splices the anchor text with the tail text at
a word anchor found in the overlap.  Anything that cannot be proven safe is
reported as unmatched so the caller can fall back to the full-turn decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from RealtimeSTT.core.tail_transcription import (
    MAX_TAIL_ANCHOR_WORDS,
    MIN_TAIL_ANCHOR_WORDS,
    _tokenize_words,
    find_tail_anchor,
)

PREVIEW_INPUT_MODES = ("full", "tail")
PREVIEW_TAIL_OVERLAP_SECONDS = 1.5
# A Live hypothesis lags the audio it has been fed by the streaming model's
# look-ahead, so its words end earlier than its reported coverage end.
PREVIEW_TAIL_LIVE_EXTRA_OVERLAP_SECONDS = 0.5
# Below this many skipped seconds a tail decode saves too little to justify
# the alignment risk; decode the full turn instead.
PREVIEW_TAIL_MIN_SKIP_SECONDS = 0.5
# Hesitant speech can leave only one or two words in a fixed overlap. The
# overlap is therefore extended back to an earlier Preview's end so that it
# holds at least this many already-recognized words.
PREVIEW_TAIL_MIN_OVERLAP_WORDS = 4
PREVIEW_TAIL_HISTORY_MARGIN_SECONDS = 0.3
PREVIEW_TAIL_HISTORY_LIMIT = 32


@dataclass(frozen=True)
class PreviewTailAnchor:
    """Recognized text that covers the turn audio ``[0, end_sample)``."""

    text: str
    end_sample: int
    source: str  # "preview" or "live"


@dataclass(frozen=True)
class PreviewTailPlan:
    """The tail decode window chosen for one Preview request."""

    anchor: PreviewTailAnchor
    start_sample: int


@dataclass(frozen=True)
class PreviewTailComposition:
    """Result of splicing anchor text with one decoded tail."""

    text: str
    matched: bool
    reason: str
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    match_start: Optional[int] = None
    distance: int = 0


def _word_count(text: str) -> int:
    return len(_tokenize_words(text))


def plan_preview_tail(
    preview_anchor: Optional[PreviewTailAnchor],
    live_anchor: Optional[PreviewTailAnchor],
    input_end_sample: int,
    *,
    sample_rate: int,
    overlap_seconds: float = PREVIEW_TAIL_OVERLAP_SECONDS,
    live_extra_overlap_seconds: float = PREVIEW_TAIL_LIVE_EXTRA_OVERLAP_SECONDS,
    min_skip_seconds: float = PREVIEW_TAIL_MIN_SKIP_SECONDS,
    history: Sequence[PreviewTailAnchor] = (),
    min_overlap_words: int = PREVIEW_TAIL_MIN_OVERLAP_WORDS,
    history_margin_seconds: float = PREVIEW_TAIL_HISTORY_MARGIN_SECONDS,
) -> Tuple[Optional[PreviewTailPlan], str]:
    """Choose an anchor and tail start, or explain why the full turn is used.

    A previous Preview is preferred: it comes from the same model, so its
    words align exactly and keep the model's casing and punctuation.  Live
    text is used only when no Preview of this turn exists yet.  ``history``
    holds earlier Preview anchors of the same turn, oldest first.
    """

    usable = []
    for anchor in (preview_anchor, live_anchor):
        if anchor is None:
            continue
        if not 0 < anchor.end_sample <= input_end_sample:
            continue
        required_words = MIN_TAIL_ANCHOR_WORDS + (1 if anchor.source == "live" else 0)
        if _word_count(anchor.text) < required_words:
            continue
        usable.append(anchor)
    if not usable:
        return None, "no_anchor"
    anchor = usable[0]
    overlap = overlap_seconds + (
        live_extra_overlap_seconds if anchor.source == "live" else 0.0
    )
    start_sample = anchor.end_sample - int(round(overlap * sample_rate))
    if anchor.source == "preview":
        # Tail-mode text is frozen before the splice, so an earlier anchor's
        # word count marks a position inside the current anchor text.
        anchor_words = _word_count(anchor.text)
        needed_words = min(min_overlap_words, anchor_words)
        # The turn start is an implicit zero-word anchor: without an earlier
        # Preview far enough back, the overlap reaches the start and the
        # full turn is decoded.
        origin = PreviewTailAnchor(text="", end_sample=0, source="preview")
        for earlier in [origin, *history][::-1]:
            if earlier.end_sample >= anchor.end_sample:
                continue
            if _word_count(earlier.text) <= anchor_words - needed_words:
                start_sample = min(
                    start_sample,
                    earlier.end_sample
                    - int(round(history_margin_seconds * sample_rate)),
                )
                break
    if start_sample < int(round(min_skip_seconds * sample_rate)):
        return None, "turn_too_short"
    return PreviewTailPlan(anchor=anchor, start_sample=start_sample), "tail"


def _exact_anchor_occurrences(
    tail_words: Sequence[Tuple[str, str]],
    anchor_words: Sequence[str],
) -> list:
    length = len(anchor_words)
    normalized = [word for _, word in tail_words]
    return [
        start
        for start in range(len(normalized) - length + 1)
        if normalized[start:start + length] == list(anchor_words)
    ]


def _with_anchor_casing(anchor_raw: str, tail_raw: str) -> str:
    """Keep the anchor's word-initial case on a word re-decoded in the tail.

    A tail decode that starts at an anchor word treats it as a sentence
    start ("That", "Pauses").  Punctuation still comes from the tail, which
    saw the words that follow.
    """

    anchor_letter = next((char for char in anchor_raw if char.isalpha()), None)
    for index, char in enumerate(tail_raw):
        if char.isalpha():
            if anchor_letter is None or anchor_letter.isupper() == char.isupper():
                return tail_raw
            replacement = char.upper() if anchor_letter.isupper() else char.lower()
            return tail_raw[:index] + replacement + tail_raw[index + 1:]
    return tail_raw


def _splice(
    anchor_words: Sequence[Tuple[str, str]],
    tail_words: Sequence[Tuple[str, str]],
    length: int,
    start: int,
    window_length: int,
) -> str:
    region = [raw for raw, _ in tail_words[start:start + window_length]]
    if window_length == length:
        region = [
            _with_anchor_casing(anchor_raw, tail_raw)
            for (anchor_raw, _), tail_raw in zip(anchor_words[-length:], region)
        ]
    elif region:
        region[0] = _with_anchor_casing(anchor_words[-length][0], region[0])
    return " ".join(
        [raw for raw, _ in anchor_words[:-length]]
        + region
        + [raw for raw, _ in tail_words[start + window_length:]]
    )


def compose_tail_preview(
    anchor_text: str,
    tail_text: str,
    *,
    drop_last_anchor_word: bool = False,
) -> PreviewTailComposition:
    """Splice ``anchor_text`` and ``tail_text`` at a safe overlap anchor.

    The anchor's last 4 (else 3) words must occur exactly once in the tail;
    a repeated phrase is ambiguous and falls back.  Only when no exact
    occurrence exists is one constrained fuzzy edit accepted.  The anchor
    words take their punctuation from the tail, which decoded them with right
    context: an earlier pause may have closed the sentence ("see if.") that
    the speaker then continued.  Their initial case stays the anchor's.
    """

    anchor_words = _tokenize_words(anchor_text)
    tail_words = _tokenize_words(tail_text)
    if not tail_words:
        return PreviewTailComposition(text="", matched=False, reason="empty_tail")
    if drop_last_anchor_word:
        # The newest Live word is the least stable one; the tail replaces it.
        anchor_words = anchor_words[:-1]
    if len(anchor_words) < MIN_TAIL_ANCHOR_WORDS:
        return PreviewTailComposition(text="", matched=False, reason="short_anchor")

    anchor_normalized = [word for _, word in anchor_words]
    for length in range(MAX_TAIL_ANCHOR_WORDS, MIN_TAIL_ANCHOR_WORDS - 1, -1):
        if len(anchor_normalized) < length:
            continue
        occurrences = _exact_anchor_occurrences(
            tail_words,
            anchor_normalized[-length:],
        )
        if len(occurrences) > 1:
            return PreviewTailComposition(
                text="",
                matched=False,
                reason="ambiguous_anchor",
            )
        if occurrences:
            start = occurrences[0]
            return PreviewTailComposition(
                text=_splice(anchor_words, tail_words, length, start, length),
                matched=True,
                reason="exact",
                anchor_length=length,
                match_start=start,
            )

    alignment = find_tail_anchor(
        " ".join(raw for raw, _ in anchor_words),
        tail_text,
        allow_fuzzy=True,
    )
    if alignment is None:
        return PreviewTailComposition(text="", matched=False, reason="no_alignment")
    start, end, length, used_fuzzy_match, distance = alignment
    return PreviewTailComposition(
        text=_splice(anchor_words, tail_words, length, start, end - start),
        matched=True,
        reason="fuzzy" if used_fuzzy_match else "exact",
        used_fuzzy_match=used_fuzzy_match,
        anchor_length=length,
        match_start=start,
        distance=distance,
    )


__all__ = [
    "PREVIEW_INPUT_MODES",
    "PREVIEW_TAIL_LIVE_EXTRA_OVERLAP_SECONDS",
    "PREVIEW_TAIL_MIN_SKIP_SECONDS",
    "PREVIEW_TAIL_OVERLAP_SECONDS",
    "PreviewTailAnchor",
    "PreviewTailComposition",
    "PreviewTailPlan",
    "compose_tail_preview",
    "plan_preview_tail",
]
