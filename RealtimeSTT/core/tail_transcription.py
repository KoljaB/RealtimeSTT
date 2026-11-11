"""
Helpers for low-latency finalization from the most recent speech tail.
"""

from dataclasses import dataclass
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

import numpy as np


PREVIEW_TRANSCRIPTION_TAIL_SECONDS = 3.0
# Historical name kept for existing callers; tailing now belongs to Preview.
FINAL_TRANSCRIPTION_TAIL_SECONDS = PREVIEW_TRANSCRIPTION_TAIL_SECONDS
MIN_TAIL_ANCHOR_WORDS = 3
MAX_TAIL_ANCHOR_WORDS = 4
MIN_LIVE_WORDS_FOR_FUZZY_REPAIR = 3
MAX_FUZZY_ANCHOR_DISTANCE = 1
MIN_PARTIAL_WORD_PREFIX_LENGTH = 2
MIN_CLOSE_WORD_LENGTH = 4
INT16_MAX_ABS_VALUE = 32768.0

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_BRAHMIC_RE = re.compile(r"[\u0900-\u0dff]+")


@dataclass(frozen=True)
class TailMergeResult:
    """
    Describes the result of aligning a Final ASR tail to Live ASR text.
    """

    text: str
    matched: bool
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    match_start: Optional[int] = None
    match_end: Optional[int] = None
    distance: int = 0


def _tokenize_words(text: str) -> List[Tuple[str, str]]:
    """
    Returns raw whitespace-delimited words and normalized comparison forms.
    Preserves combining marks for Brahmic scripts to handle vowel signs correctly.
    """

    tokens = []
    for raw_word in str(text or "").split():
        folded = raw_word.casefold()
        # Keep mark-only tokens from becoming alignment evidence.
        if not _WORD_RE.search(folded):
            continue
        # Marks in these Brahmic blocks distinguish words; other scripts
        # retain their existing comparison rules.
        normalized = "".join(
            char for char in folded
            if char.isalnum()
            or (
                "\u0900" <= char <= "\u0dff"
                and unicodedata.category(char).startswith("M")
            )
        )
        # Canonically equivalent vowel signs must still match. Restrict NFC
        # to Brahmic runs so normalization elsewhere stays unchanged.
        normalized = _BRAHMIC_RE.sub(
            lambda match: unicodedata.normalize("NFC", match.group()), normalized
        )
        tokens.append((raw_word, normalized))
    return tokens


def _levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """
    Calculates edit distance between two short sequences.
    """

    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def _is_partial_word_match(live_word: str, final_word: str) -> bool:
    """
    Returns whether Final ASR completes a truncated Live ASR word.
    """

    return (
        live_word != final_word
        and len(live_word) >= MIN_PARTIAL_WORD_PREFIX_LENGTH
        and len(final_word) > len(live_word)
        and final_word.startswith(live_word)
    )


def _is_close_word_match(
    left: str,
    right: str,
    min_close_word_length: int = MIN_CLOSE_WORD_LENGTH,
) -> bool:
    """
    Allows one-character ASR spelling differences for sufficiently long words.

    Short arbitrary substitutions such as ``on`` -> ``in`` are intentionally
    not accepted as alignment evidence.  Callers that accept a softer realtime
    anchor may lower the minimum word length explicitly.
    """

    if min(len(left), len(right)) < min_close_word_length:
        return False
    return _levenshtein_distance(left, right) <= MAX_FUZZY_ANCHOR_DISTANCE


def _is_brahmic_mark_difference(left: str, right: str) -> bool:
    """Identify a word correction that changes only Brahmic combining marks."""

    if left == right or not (_BRAHMIC_RE.search(left) or _BRAHMIC_RE.search(right)):
        return False

    def without_brahmic_marks(word):
        return _BRAHMIC_RE.sub(
            lambda match: "".join(
                char for char in unicodedata.normalize("NFD", match.group())
                if not unicodedata.category(char).startswith("M")
            ),
            word,
        )

    left_base = without_brahmic_marks(left)
    return bool(left_base) and left_base == without_brahmic_marks(right)


def _fuzzy_window_distance(
    anchor_words: Sequence[str],
    window: Sequence[str],
    min_close_word_length: int = MIN_CLOSE_WORD_LENGTH,
) -> Optional[Tuple[int, int]]:
    """
    Classifies one constrained fuzzy anchor window.

    The only accepted one-edit cases are one inserted/deleted token, a
    one-character spelling difference in a long word, a Brahmic mark-only
    correction backed by two exact words, or a Final word that completes the
    last Live word. Arbitrary word substitutions are rejected.
    Returns ``(distance, exact_word_count)`` when safe.
    """

    anchor_length = len(anchor_words)
    window_length = len(window)

    if window_length == anchor_length:
        differences = [
            index
            for index, (anchor_word, final_word) in enumerate(zip(anchor_words, window))
            if anchor_word != final_word
        ]
        if len(differences) != 1:
            return None

        difference_index = differences[0]
        anchor_word = anchor_words[difference_index]
        final_word = window[difference_index]
        is_last_word_completion = (
            difference_index == anchor_length - 1
            and _is_partial_word_match(anchor_word, final_word)
        )
        is_mark_correction = (
            anchor_length >= MIN_TAIL_ANCHOR_WORDS
            and _is_brahmic_mark_difference(anchor_word, final_word)
        )
        if not (
            is_last_word_completion
            or is_mark_correction
            or _is_close_word_match(
                anchor_word,
                final_word,
                min_close_word_length=min_close_word_length,
            )
        ):
            return None
        return 1, anchor_length - 1

    if abs(window_length - anchor_length) != 1:
        return None

    if window_length == anchor_length - 1:
        for omitted_index in range(anchor_length):
            candidate = (
                list(anchor_words[:omitted_index])
                + list(anchor_words[omitted_index + 1 :])
            )
            if candidate == list(window):
                return 1, window_length
        return None

    for omitted_index in range(window_length):
        candidate = (
            list(window[:omitted_index]) + list(window[omitted_index + 1 :])
        )
        if candidate == list(anchor_words):
            return 1, anchor_length
    return None


def _find_exact_anchor(
    final_words: Sequence[Tuple[str, str]],
    anchor_words: Sequence[str],
) -> Optional[Tuple[int, int, int]]:
    """
    Finds the rightmost exact anchor occurrence.
    """

    anchor_length = len(anchor_words)
    if not anchor_length or len(final_words) < anchor_length:
        return None

    final_normalized = [normalized for _, normalized in final_words]
    for start in range(len(final_normalized) - anchor_length, -1, -1):
        if final_normalized[start:start + anchor_length] == list(anchor_words):
            return start, start + anchor_length, 0
    return None


def _find_fuzzy_anchor(
    final_words: Sequence[Tuple[str, str]],
    anchor_words: Sequence[str],
    min_close_word_length: int = MIN_CLOSE_WORD_LENGTH,
) -> Optional[Tuple[int, int, int]]:
    """
    Finds a safe close anchor occurrence using at most one constrained edit.
    """

    anchor_length = len(anchor_words)
    if not anchor_length:
        return None

    final_normalized = [normalized for _, normalized in final_words]
    best = None
    min_window_length = max(1, anchor_length - 1)
    max_window_length = anchor_length + 1

    for window_length in range(min_window_length, max_window_length + 1):
        if len(final_normalized) < window_length:
            continue
        for start in range(len(final_normalized) - window_length + 1):
            window = final_normalized[start:start + window_length]
            classified = _fuzzy_window_distance(
                anchor_words,
                window,
                min_close_word_length=min_close_word_length,
            )
            if classified is None:
                continue
            distance, exact_word_count = classified

            # Prefer the smallest edit distance, then a same-length window,
            # then the most exact word pairs and the rightmost occurrence.
            candidate_key = (
                distance,
                abs(window_length - anchor_length),
                -exact_word_count,
                -start,
            )
            if best is None or candidate_key < best[0]:
                best = (candidate_key, start, start + window_length, distance)

    if best is None:
        return None
    _, start, end, distance = best
    return start, end, distance


def _repair_partial_word_before_exact_suffix(
    live_words: Sequence[Tuple[str, str]],
    final_words: Sequence[Tuple[str, str]],
    allow_mark_correction: bool = False,
) -> Optional[TailMergeResult]:
    """Repair one truncated Live word backed by a strong exact suffix.

    The ordinary matcher prefers a three-word exact suffix. That is normally
    conservative, but it can leave the immediately preceding Live token
    truncated (``spee to text system``). Two exact following words plus a
    one-way prefix completion provide the same three-token evidence without
    accepting an arbitrary substitution. When enabled, a Brahmic mark-only
    correction can use the same exact suffix.
    """

    minimum_exact_suffix = max(2, MIN_TAIL_ANCHOR_WORDS - 1)
    maximum_exact_suffix = min(MAX_TAIL_ANCHOR_WORDS, len(live_words) - 1)
    for exact_length in range(
        maximum_exact_suffix,
        minimum_exact_suffix - 1,
        -1,
    ):
        live_anchor = [normalized for _, normalized in live_words[-exact_length:]]
        exact = _find_exact_anchor(final_words, live_anchor)
        if exact is None:
            continue
        start, end, _distance = exact
        live_partial_index = len(live_words) - exact_length - 1
        if start <= 0 or live_partial_index < 0:
            continue
        live_partial = live_words[live_partial_index][1]
        final_completion = final_words[start - 1][1]
        if not _is_partial_word_match(live_partial, final_completion) and not (
            allow_mark_correction
            and _is_brahmic_mark_difference(live_partial, final_completion)
        ):
            continue
        prefix = [raw_word for raw_word, _ in live_words[:live_partial_index]]
        suffix = [raw_word for raw_word, _ in final_words[start - 1 :]]
        return TailMergeResult(
            text=" ".join(prefix + suffix),
            matched=True,
            used_fuzzy_match=True,
            anchor_length=exact_length + 1,
            match_start=start - 1,
            match_end=end,
            distance=1,
        )
    return None


def find_tail_anchor(
    live_text: str,
    final_tail_text: str,
    allow_fuzzy: bool = True,
    min_anchor_words: int = MIN_TAIL_ANCHOR_WORDS,
    max_anchor_words: int = MAX_TAIL_ANCHOR_WORDS,
    min_close_word_length: int = MIN_CLOSE_WORD_LENGTH,
) -> Optional[Tuple[int, int, int, bool, int]]:
    """
    Finds a configurable Live ASR suffix inside the Final ASR tail.

    The defaults preserve Preview's conservative 3-/4-word exact and fuzzy
    alignment. Realtime callers can explicitly use a shorter, softer anchor
    without changing Preview behavior.

    Returns:
        ``(start, end, anchor_length, used_fuzzy_match, distance)`` in the
        Final ASR word array, or ``None`` when no safe alignment exists.
    """

    try:
        min_anchor_words = int(min_anchor_words)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_anchor_words must be a positive integer") from exc
    try:
        max_anchor_words = int(max_anchor_words)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_anchor_words must be a positive integer") from exc
    try:
        min_close_word_length = int(min_close_word_length)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "min_close_word_length must be a positive integer"
        ) from exc
    if min_anchor_words < 1:
        raise ValueError("min_anchor_words must be a positive integer")
    if max_anchor_words < 1:
        raise ValueError("max_anchor_words must be a positive integer")
    if max_anchor_words < min_anchor_words:
        raise ValueError("max_anchor_words must be >= min_anchor_words")
    if min_close_word_length < 1:
        raise ValueError("min_close_word_length must be a positive integer")

    live_words = _tokenize_words(live_text)
    final_words = _tokenize_words(final_tail_text)

    if not live_words or not final_words:
        return None

    live_normalized = [normalized for _, normalized in live_words]

    # Exact matches are always preferred over fuzzy matches. The longest
    # configured anchor wins, with shorter anchors as the fallback.
    for anchor_length in range(max_anchor_words, min_anchor_words - 1, -1):
        if len(live_normalized) < anchor_length:
            continue
        anchor = live_normalized[-anchor_length:]
        exact = _find_exact_anchor(final_words, anchor)
        if exact is not None:
            start, end, distance = exact
            return start, end, anchor_length, False, distance

    if not allow_fuzzy:
        return None

    for anchor_length in range(max_anchor_words, min_anchor_words - 1, -1):
        if len(live_normalized) < anchor_length:
            continue
        anchor = live_normalized[-anchor_length:]
        fuzzy = _find_fuzzy_anchor(
            final_words,
            anchor,
            min_close_word_length=min_close_word_length,
        )
        if fuzzy is not None:
            start, end, distance = fuzzy
            return start, end, anchor_length, True, distance

    return None


def merge_live_and_tail_transcription(
    live_text: str,
    final_tail_text: str,
    min_live_words_for_fuzzy_repair: int = MIN_LIVE_WORDS_FOR_FUZZY_REPAIR,
) -> TailMergeResult:
    """
    Replaces the lagging Live ASR suffix with the Final ASR tail suffix.

    Final ASR words before the matched anchor are intentionally discarded;
    they are outside the trusted overlap and may contain boundary garbage.
    The caller can use ``matched`` to choose a full-utterance fallback.
    """

    live_text = str(live_text or "").strip()
    final_tail_text = str(final_tail_text or "").strip()
    try:
        min_live_words_for_fuzzy_repair = int(
            min_live_words_for_fuzzy_repair
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "min_live_words_for_fuzzy_repair must be a positive integer"
        ) from exc
    if min_live_words_for_fuzzy_repair < 1:
        raise ValueError(
            "min_live_words_for_fuzzy_repair must be a positive integer"
        )

    live_words = _tokenize_words(live_text)
    final_words = _tokenize_words(final_tail_text)

    partial_boundary_repair = _repair_partial_word_before_exact_suffix(
        live_words,
        final_words,
        allow_mark_correction=len(live_words) >= min_live_words_for_fuzzy_repair,
    )
    if partial_boundary_repair is not None:
        return partial_boundary_repair

    # Preserve the existing exact-anchor behavior when the current Live
    # suffix is already present verbatim in Final ASR. The last Live word is
    # only treated as untrusted when exact alignment cannot prove it safe.
    exact_alignment = find_tail_anchor(
        live_text,
        final_tail_text,
        allow_fuzzy=False,
    )
    if exact_alignment is not None:
        start, end, anchor_length, used_fuzzy_match, distance = exact_alignment
        suffix = [raw_word for raw_word, _ in final_words[end:]]
        merged_words = [raw_word for raw_word, _ in live_words] + suffix
        return TailMergeResult(
            text=" ".join(merged_words),
            matched=True,
            used_fuzzy_match=used_fuzzy_match,
            anchor_length=anchor_length,
            match_start=start,
            match_end=end,
            distance=distance,
        )

    # Fuzzy alignment is intentionally disabled for very short Live
    # hypotheses. Exact anchors remain safe above; this gate only controls
    # the weaker repair path. Values below the three-word anchor length are
    # accepted as configuration but cannot make the matcher use fewer than
    # three anchor words.
    if len(live_words) < min_live_words_for_fuzzy_repair:
        return TailMergeResult(text=live_text, matched=False)

    alignment = find_tail_anchor(live_text, final_tail_text)
    has_mark_correction = False
    if alignment is not None:
        start, end, anchor_length, used_fuzzy_match, _distance = alignment
        has_mark_correction = (
            used_fuzzy_match
            and end - start == anchor_length
            and any(
                _is_brahmic_mark_difference(live_word[1], final_word[1])
                for live_word, final_word in zip(
                    live_words[-anchor_length:], final_words[start:end]
                )
            )
        )

    # Prefer a supported mark correction to a weaker stable-prefix match
    # that could mistake an earlier Live word for an omission.
    # The newest Live token is inherently the least stable one. When there
    # are enough preceding words for a real anchor, align without that token
    # first and let Final ASR provide its replacement plus the suffix.
    if len(live_words) >= MIN_TAIL_ANCHOR_WORDS + 1 and not has_mark_correction:
        stable_live_words = live_words[:-1]
        stable_partial_boundary_repair = _repair_partial_word_before_exact_suffix(
            stable_live_words,
            final_words,
            allow_mark_correction=True,
        )
        if stable_partial_boundary_repair is not None:
            return stable_partial_boundary_repair
        stable_live_text = " ".join(raw_word for raw_word, _ in stable_live_words)
        stable_alignment = find_tail_anchor(stable_live_text, final_tail_text)
        if stable_alignment is not None:
            start, end, anchor_length, used_fuzzy_match, distance = stable_alignment
            if end < len(final_words) and len(stable_live_words) >= anchor_length:
                live_prefix = stable_live_words[:-anchor_length]
                final_suffix = [raw_word for raw_word, _ in final_words[start:]]
                return TailMergeResult(
                    text=" ".join(
                        [raw_word for raw_word, _ in live_prefix] + final_suffix
                    ),
                    matched=True,
                    used_fuzzy_match=used_fuzzy_match,
                    anchor_length=anchor_length,
                    match_start=start,
                    match_end=end,
                    distance=distance,
                )

    if alignment is None:
        return TailMergeResult(text=live_text, matched=False)

    start, end, anchor_length, used_fuzzy_match, distance = alignment

    # For the fallback alignment, replace a final Live token only when the
    # matched Final token is demonstrably its completion or a close spelling.
    if used_fuzzy_match and end - start == anchor_length and live_words:
        final_match_words = final_words[start:end]
        live_word = live_words[-1][1]
        final_word = final_match_words[-1][1]
        if (
            live_word != final_word
            and (
                _is_partial_word_match(live_word, final_word)
                or _is_close_word_match(live_word, final_word)
            )
        ):
            live_words = live_words[:-1] + [final_match_words[-1]]

        # A mark-only correction can occur anywhere in the aligned window.
        live_start = len(live_words) - anchor_length
        for offset, final_match in enumerate(final_match_words):
            live_index = live_start + offset
            if _is_brahmic_mark_difference(
                live_words[live_index][1], final_match[1]
            ):
                live_words[live_index] = final_match

    suffix = [raw_word for raw_word, _ in final_words[end:]]
    merged_words = [raw_word for raw_word, _ in live_words] + suffix

    return TailMergeResult(
        text=" ".join(merged_words),
        matched=True,
        used_fuzzy_match=used_fuzzy_match,
        anchor_length=anchor_length,
        match_start=start,
        match_end=end,
        distance=distance,
    )


def append_pcm16_tail(recorder, data, seconds=FINAL_TRANSCRIPTION_TAIL_SECONDS):
    """
    Appends PCM16 data while retaining only the configured recent tail.
    """

    if data is None:
        return

    if isinstance(data, np.ndarray):
        payload = data.astype(np.int16, copy=False).tobytes()
    else:
        payload = bytes(data)
    if not payload:
        return

    if seconds == FINAL_TRANSCRIPTION_TAIL_SECONDS:
        seconds = getattr(
            recorder,
            "preview_transcription_tail_seconds",
            seconds,
        )
    sample_rate = int(getattr(recorder, "sample_rate", 16000) or 16000)
    max_samples = max(1, int(round(sample_rate * float(seconds))))
    max_bytes = max_samples * 2

    buffer = getattr(recorder, "active_speech_tail_buffer", None)
    if buffer is None:
        buffer = bytearray()
        recorder.active_speech_tail_buffer = buffer
    elif not isinstance(buffer, bytearray):
        buffer = bytearray(buffer)
        recorder.active_speech_tail_buffer = buffer

    buffer.extend(payload)
    if len(buffer) > max_bytes:
        del buffer[:-max_bytes]


def snapshot_pcm16_tail(recorder) -> bytes:
    """
    Copies the currently retained PCM16 tail.
    """

    buffer = getattr(recorder, "active_speech_tail_buffer", None)
    if not buffer:
        return b""
    return bytes(buffer)


def clear_pcm16_tail(recorder):
    """
    Clears the active-speech rolling tail.
    """

    recorder.active_speech_tail_buffer = bytearray()


def pcm16_bytes_to_float_audio(data):
    """
    Converts raw mono PCM16 bytes to the recorder's float audio format.
    """

    if data is None:
        return np.array([], dtype=np.float32)
    raw = bytes(data)
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return np.array([], dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE


def extract_audio_tail(
    audio,
    sample_rate,
    seconds=FINAL_TRANSCRIPTION_TAIL_SECONDS,
):
    """
    Returns a copy containing at most the configured recent audio seconds.
    """

    if audio is None:
        return np.array([], dtype=np.float32)
    if isinstance(audio, (bytes, bytearray, memoryview)):
        audio = pcm16_bytes_to_float_audio(audio)

    audio_array = np.asarray(audio)
    if audio_array.ndim != 1:
        audio_array = audio_array.reshape(-1)
    if audio_array.size == 0:
        return np.array([], dtype=np.float32)

    max_samples = max(1, int(round(float(sample_rate) * float(seconds))))
    return np.array(audio_array[-max_samples:], copy=True)
