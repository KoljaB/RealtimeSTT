"""Conservative pre-recording buffer selection helpers."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


REASON_BELOW_MINIMUM = "below_minimum"
REASON_EMPTY_BUFFER = "empty_buffer"
REASON_FALLBACK_FULL_PREROLL = "fallback_full_preroll"
REASON_STABLE_SILENCE_FOUND = "stable_silence_found"
REASON_UNCERTAIN = "uncertain"

DEFAULT_PREROLL_MIN_SILENCE_MS = 200.0
DEFAULT_PREROLL_GUARD_MS = 160.0
DEFAULT_PREROLL_MIN_INCLUDED_MS = 600.0
DEFAULT_PREROLL_MAX_GAP_MS = 80.0
DEFAULT_PREROLL_NOISE_FLOOR_MULTIPLIER = 2.5
DEFAULT_PREROLL_ENERGY_MARGIN_RMS = 25.0


@dataclass(frozen=True)
class PrerollFrameMetadata:
    """Metadata captured for one frame retained in the pre-recording buffer."""

    sample_count: int
    is_speech: Optional[bool]
    rms: Optional[float] = None
    start_sample: Optional[int] = None
    webrtc_is_speech: Optional[bool] = None
    silero_is_speech: Optional[bool] = None


@dataclass(frozen=True)
class PrerollSelection:
    """Selected pre-recording buffer tail and diagnostics.

    ``start_index`` identifies the first retained frame in the input metadata.
    ``included_sample_count`` and ``included_seconds`` describe the selected
    tail exactly, so callers can publish timeline metadata without recounting.
    """

    start_index: int
    selected_frame_count: int
    included_sample_count: int
    included_seconds: float
    reason: str
    diagnostics: Dict[str, Any]


def select_preroll_frames(
    frame_metadata: Sequence[PrerollFrameMetadata],
    sample_rate: int,
    min_silence_ms: float=DEFAULT_PREROLL_MIN_SILENCE_MS,
    guard_ms: float=DEFAULT_PREROLL_GUARD_MS,
    max_gap_ms: float=DEFAULT_PREROLL_MAX_GAP_MS,
    min_included_ms: float=DEFAULT_PREROLL_MIN_INCLUDED_MS,
    energy_silence_rms: Optional[float]=None,
    noise_floor_multiplier: float=DEFAULT_PREROLL_NOISE_FLOOR_MULTIPLIER,
    energy_margin_rms: float=DEFAULT_PREROLL_ENERGY_MARGIN_RMS,
) -> PrerollSelection:
    """Select a conservative tail from pre-recording frame metadata.

    The selector uses VAD metadata captured while audio flowed forward through
    the recorder. It never runs a second VAD pass. Energy is only a supporting
    signal for frames already marked non-speech or unknown; it cannot turn a
    VAD speech frame into silence or create a speech onset by itself.

    Args:
        frame_metadata: Prebuffer frame metadata in chronological order.
        sample_rate: Audio sample rate in samples per second.
        min_silence_ms: Required contiguous silence before speech onset.
        guard_ms: Audio to keep before speech onset, in milliseconds.
        max_gap_ms: Short VAD false gaps to merge into one speech run.
        min_included_ms: Minimum selected pre-roll tail, in milliseconds.
        energy_silence_rms: Optional absolute RMS ceiling for silence.
        noise_floor_multiplier: Multiplier applied to the local noise floor.
        energy_margin_rms: RMS margin added to the adaptive noise threshold.

    Returns:
        A ``PrerollSelection`` describing selected frame indices, exact sample
        count, seconds, reason, and diagnostics.

    Raises:
        ValueError: If ``sample_rate`` is not positive.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    frames = list(frame_metadata or ())
    total_sample_count = _sum_samples(frames)
    if not frames or total_sample_count <= 0:
        return _empty_selection(sample_rate)

    max_gap_samples = _milliseconds_to_samples(max_gap_ms, sample_rate)
    min_silence_samples = _milliseconds_to_samples(min_silence_ms, sample_rate)
    guard_samples = _milliseconds_to_samples(guard_ms, sample_rate)
    min_included_samples = _milliseconds_to_samples(min_included_ms, sample_rate)

    base_diagnostics = {
        "totalSampleCount": total_sample_count,
        "frameCount": len(frames),
        "speechSampleCount": _speech_sample_count(frames, "is_speech"),
        "webrtcSpeechSampleCount": _speech_sample_count(frames, "webrtc_is_speech"),
        "sileroSpeechSampleCount": _speech_sample_count(frames, "silero_is_speech"),
        "minSilenceSamples": min_silence_samples,
        "guardSamples": guard_samples,
        "minIncludedSamples": min_included_samples,
        "maxGapSamples": max_gap_samples,
    }
    if total_sample_count <= min_included_samples:
        return _full_selection(
            frames,
            sample_rate,
            REASON_BELOW_MINIMUM,
            dict(base_diagnostics, fallbackDetail="buffer_not_above_minimum"),
        )

    onset_index = _find_merged_speech_onset_index(
        frames,
        max_gap_samples=max_gap_samples,
    )
    base_diagnostics["speechOnsetIndex"] = onset_index
    if onset_index is None:
        return _full_selection(
            frames,
            sample_rate,
            REASON_UNCERTAIN,
            dict(base_diagnostics, fallbackDetail="no_speech_onset"),
        )
    if onset_index <= 0:
        return _full_selection(
            frames,
            sample_rate,
            REASON_FALLBACK_FULL_PREROLL,
            dict(base_diagnostics, fallbackDetail="onset_at_buffer_start"),
        )

    energy_threshold_rms, noise_floor_rms = _energy_threshold_rms(
        frames[:onset_index],
        energy_silence_rms=energy_silence_rms,
        noise_floor_multiplier=noise_floor_multiplier,
        energy_margin_rms=energy_margin_rms,
    )
    (
        silence_start_index,
        stable_silence_samples,
        effective_onset_index,
        pre_speech_tail_samples,
    ) = _stable_silence_before_onset(
        frames,
        onset_index=onset_index,
        energy_threshold_rms=energy_threshold_rms,
    )
    onset_sample = _sample_offset_for_index(frames, onset_index)
    effective_onset_sample = _sample_offset_for_index(frames, effective_onset_index)
    diagnostics = dict(
        base_diagnostics,
        stableSilenceStartIndex=silence_start_index,
        stableSilenceSamples=stable_silence_samples,
        stableSilenceSeconds=stable_silence_samples / float(sample_rate),
        effectiveSpeechOnsetIndex=effective_onset_index,
        effectiveSpeechOnsetSample=effective_onset_sample,
        preSpeechTailSamples=pre_speech_tail_samples,
        preSpeechTailSeconds=pre_speech_tail_samples / float(sample_rate),
        energyThresholdRms=energy_threshold_rms,
        noiseFloorRms=noise_floor_rms,
        speechOnsetSample=onset_sample,
    )
    if stable_silence_samples < min_silence_samples:
        return _full_selection(
            frames,
            sample_rate,
            REASON_UNCERTAIN,
            dict(diagnostics, fallbackDetail="stable_silence_too_short"),
        )

    latest_by_guard = effective_onset_sample - guard_samples
    latest_by_minimum = total_sample_count - min_included_samples
    selection_start_sample = min(latest_by_guard, latest_by_minimum)
    if selection_start_sample <= 0:
        return _full_selection(
            frames,
            sample_rate,
            REASON_BELOW_MINIMUM,
            dict(diagnostics, fallbackDetail="guard_or_minimum_consumes_buffer"),
        )

    start_index = _index_for_sample_offset(frames, selection_start_sample)
    included_sample_count = _sum_samples(frames[start_index:])
    return PrerollSelection(
        start_index=start_index,
        selected_frame_count=len(frames) - start_index,
        included_sample_count=included_sample_count,
        included_seconds=included_sample_count / float(sample_rate),
        reason=REASON_STABLE_SILENCE_FOUND,
        diagnostics=dict(
            diagnostics,
            selectionStartSample=selection_start_sample,
        ),
    )


def _empty_selection(sample_rate):
    return PrerollSelection(
        start_index=0,
        selected_frame_count=0,
        included_sample_count=0,
        included_seconds=0.0 / float(sample_rate),
        reason=REASON_EMPTY_BUFFER,
        diagnostics={"totalSampleCount": 0, "frameCount": 0},
    )


def _full_selection(frames, sample_rate, reason, diagnostics):
    total_sample_count = _sum_samples(frames)
    return PrerollSelection(
        start_index=0,
        selected_frame_count=len(frames),
        included_sample_count=total_sample_count,
        included_seconds=total_sample_count / float(sample_rate),
        reason=reason,
        diagnostics=diagnostics,
    )


def _find_merged_speech_onset_index(frames, max_gap_samples):
    current_start_index = None
    current_gap_samples = 0
    latest_run_start_index = None

    for index, frame in enumerate(frames):
        if _is_speech_frame(frame):
            if current_start_index is None or current_gap_samples > max_gap_samples:
                current_start_index = index
            current_gap_samples = 0
            latest_run_start_index = current_start_index
        elif current_start_index is not None:
            current_gap_samples += _frame_sample_count(frame)

    return latest_run_start_index


def _stable_silence_before_onset(frames, onset_index, energy_threshold_rms):
    pre_speech_tail_samples = 0
    effective_onset_index = onset_index
    index = onset_index - 1

    # WebRTC often misses quiet consonant lead-ins that still carry useful
    # speech energy. Treat those frames as a pre-speech tail to keep, then
    # search for stable silence before that tail.
    while index >= 0 and not _is_stable_silence_frame(frames[index], energy_threshold_rms):
        pre_speech_tail_samples += _frame_sample_count(frames[index])
        effective_onset_index = index
        index -= 1

    stable_silence_samples = 0
    silence_start_index = index + 1

    while index >= 0:
        frame = frames[index]
        if not _is_stable_silence_frame(frame, energy_threshold_rms):
            break
        stable_silence_samples += _frame_sample_count(frame)
        silence_start_index = index
        index -= 1

    return (
        silence_start_index,
        stable_silence_samples,
        effective_onset_index,
        pre_speech_tail_samples,
    )


def _is_stable_silence_frame(frame, energy_threshold_rms):
    is_speech = _optional_bool(frame.is_speech)
    if is_speech is True:
        return False

    rms = frame.rms
    has_rms = rms is not None
    if has_rms and energy_threshold_rms is not None:
        try:
            is_low_energy = float(rms) <= energy_threshold_rms
        except (TypeError, ValueError):
            is_low_energy = False
    else:
        is_low_energy = is_speech is False

    if is_speech is False:
        return is_low_energy
    return has_rms and is_low_energy


def _is_speech_frame(frame):
    return _optional_bool(frame.is_speech) is True


def _optional_bool(value):
    if value is None:
        return None
    return bool(value)


def _energy_threshold_rms(
    frames,
    energy_silence_rms,
    noise_floor_multiplier,
    energy_margin_rms,
) -> Tuple[Optional[float], Optional[float]]:
    rms_values = []
    for frame in frames:
        if frame.rms is None:
            continue
        try:
            rms = float(frame.rms)
        except (TypeError, ValueError):
            continue
        if rms >= 0:
            rms_values.append(rms)

    absolute_threshold = (
        None
        if energy_silence_rms is None
        else max(0.0, float(energy_silence_rms))
    )
    if not rms_values:
        return absolute_threshold, None

    sorted_values = sorted(rms_values)
    floor_count = max(1, int(math.ceil(len(sorted_values) * 0.2)))
    noise_floor = sum(sorted_values[:floor_count]) / float(floor_count)
    adaptive_threshold = (
        noise_floor * max(0.0, float(noise_floor_multiplier))
        + max(0.0, float(energy_margin_rms))
    )
    if absolute_threshold is None:
        return adaptive_threshold, noise_floor
    return min(absolute_threshold, adaptive_threshold), noise_floor


def _sample_offset_for_index(frames, index):
    return _sum_samples(frames[:index])


def _index_for_sample_offset(frames, sample_offset):
    running_sample_count = 0
    for index, frame in enumerate(frames):
        running_sample_count += _frame_sample_count(frame)
        if running_sample_count > sample_offset:
            return index
    return len(frames)


def _sum_samples(frames):
    return sum(_frame_sample_count(frame) for frame in frames)


def _speech_sample_count(frames, attr_name):
    sample_count = 0
    for frame in frames:
        if _optional_bool(getattr(frame, attr_name, None)) is True:
            sample_count += _frame_sample_count(frame)
    return sample_count


def _frame_sample_count(frame):
    return max(0, int(frame.sample_count))


def _milliseconds_to_samples(milliseconds, sample_rate):
    return max(0, int(math.ceil(float(milliseconds) * float(sample_rate) / 1000.0)))
