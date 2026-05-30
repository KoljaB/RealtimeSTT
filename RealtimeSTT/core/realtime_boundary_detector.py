"""
Lightweight realtime speech-boundary detection.

This module intentionally does not know anything about Whisper, recorder
threads, or GPU work. It watches short PCM chunks and emits low-cost acoustic
boundary events that can be used to schedule heavier realtime transcription.

The detector is heuristic. It detects likely voiced energy valleys, not
linguistic syllables with certainty.
"""

import math
import time

from typing import Dict, List, Optional

import numpy as np


INT16_MAX_ABS_VALUE = 32768.0


class SpeechBoundaryEvent:
    """A committed acoustic boundary candidate."""

    def __init__(
            self,
            boundary_sample: int,
            boundary_time_seconds: float,
            score: float,
            reason: str,
            energy_db: float,
            noise_floor_db: float,
            drop_db: float,
            valley_depth_db: float,
            latency_ms: float,
            created_at: Optional[float] = None):
        self.boundary_sample = boundary_sample
        self.boundary_time_seconds = boundary_time_seconds
        self.score = score
        self.reason = reason
        self.energy_db = energy_db
        self.noise_floor_db = noise_floor_db
        self.drop_db = drop_db
        self.valley_depth_db = valley_depth_db
        self.latency_ms = latency_ms
        self.created_at = time.time() if created_at is None else created_at

    def as_dict(self) -> Dict[str, float]:
        return {
            "boundary_sample": self.boundary_sample,
            "boundary_time_seconds": self.boundary_time_seconds,
            "score": self.score,
            "reason": self.reason,
            "energy_db": self.energy_db,
            "noise_floor_db": self.noise_floor_db,
            "drop_db": self.drop_db,
            "valley_depth_db": self.valley_depth_db,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return (
            "SpeechBoundaryEvent("
            "time={:.3f}s, score={:.2f}, reason={!r}, "
            "drop={:.1f}dB, valley={:.1f}dB)"
        ).format(
            self.boundary_time_seconds,
            self.score,
            self.reason,
            self.drop_db,
            self.valley_depth_db,
        )


class SpeechBoundaryResult:
    """Result returned for one processed chunk."""

    def __init__(
            self,
            events: List[SpeechBoundaryEvent],
            current_energy_db: float,
            current_rms: float,
            noise_floor_db: float,
            is_speech: bool,
            is_vowel_like: bool,
            voicing_score: float,
            processed_frames: int):
        self.events = events
        self.current_energy_db = current_energy_db
        self.current_rms = current_rms
        self.noise_floor_db = noise_floor_db
        self.is_speech = is_speech
        self.is_vowel_like = is_vowel_like
        self.voicing_score = voicing_score
        self.processed_frames = processed_frames

    @property
    def boundary_detected(self) -> bool:
        return bool(self.events)

    @property
    def latest_event(self) -> Optional[SpeechBoundaryEvent]:
        if not self.events:
            return None
        return self.events[-1]


class RealtimeSpeechBoundaryDetector:
    """
    Streaming detector for cheap speech boundary candidates.

    The detector builds a 10 ms log-energy envelope and looks for confirmed
    local energy valleys after voiced audio. A small lookahead is used so a
    candidate is only emitted once the next few frames confirm the dip.
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            frame_ms: float = 10.0,
            lookahead_ms: float = 30.0,
            history_ms: float = 900.0,
            sensitivity: float = 0.6,
            min_rms: float = 0.004,
            speech_margin_db: Optional[float] = None,
            drop_db: Optional[float] = None,
            valley_depth_db: Optional[float] = None,
            recovery_db: Optional[float] = None,
            min_boundary_interval_ms: float = 160.0,
            min_voiced_ms: float = 70.0,
            min_vowel_ms: float = 40.0,
            vowel_margin_db: Optional[float] = None,
            min_voicing_score: Optional[float] = None,
            max_vowel_zero_crossing_rate: Optional[float] = None):
        self.sample_rate = int(sample_rate)
        self.frame_ms = float(frame_ms)
        self.lookahead_ms = float(lookahead_ms)
        self.history_ms = float(history_ms)
        self.sensitivity = self._clamp(float(sensitivity), 0.0, 1.0)
        self.min_rms = float(min_rms)
        self.min_boundary_interval_ms = float(min_boundary_interval_ms)
        self.min_voiced_ms = float(min_voiced_ms)
        self.min_vowel_ms = float(min_vowel_ms)

        self.frame_samples = max(1, int(round(self.sample_rate * self.frame_ms / 1000.0)))
        self.lookahead_frames = max(1, int(round(self.lookahead_ms / self.frame_ms)))
        self.history_frames = max(
            self.lookahead_frames + 8,
            int(round(self.history_ms / self.frame_ms)),
        )
        self.min_voiced_frames = max(1, int(round(self.min_voiced_ms / self.frame_ms)))
        self.min_vowel_frames = max(1, int(round(self.min_vowel_ms / self.frame_ms)))
        self.before_window_frames = max(self.min_voiced_frames + 2, int(round(180.0 / self.frame_ms)))

        s = self.sensitivity
        self.speech_margin_db = (
            float(speech_margin_db) if speech_margin_db is not None else 10.5 - 4.5 * s
        )
        self.drop_db = float(drop_db) if drop_db is not None else 5.8 - 3.2 * s
        self.valley_depth_db = (
            float(valley_depth_db) if valley_depth_db is not None else 3.2 - 1.8 * s
        )
        self.recovery_db = float(recovery_db) if recovery_db is not None else 2.2 - 1.1 * s
        self.vowel_margin_db = (
            float(vowel_margin_db) if vowel_margin_db is not None else 14.0 - 5.0 * s
        )
        self.min_voicing_score = (
            float(min_voicing_score) if min_voicing_score is not None else 0.62 - 0.12 * s
        )
        self.max_vowel_zero_crossing_rate = (
            float(max_vowel_zero_crossing_rate)
            if max_vowel_zero_crossing_rate is not None
            else 0.20 + 0.05 * s
        )

        self.reset()

    def reset(self) -> None:
        self._pending_samples = np.empty(0, dtype=np.float32)
        self._frames: List[Dict[str, float]] = []
        self._processed_samples = 0
        self._frame_index = 0
        self._last_boundary_sample = -10**12
        self._last_evaluated_frame_index = -1
        self._noise_floor_db = -70.0
        self._current_energy_db = -120.0
        self._current_rms = 0.0
        self._current_is_speech = False
        self._current_is_vowel_like = False
        self._current_voicing_score = 0.0

    def process_bytes(self, pcm_chunk: bytes) -> SpeechBoundaryResult:
        """Process little-endian int16 PCM bytes."""
        if not pcm_chunk:
            return self._empty_result()

        usable_length = len(pcm_chunk) - (len(pcm_chunk) % 2)
        if usable_length <= 0:
            return self._empty_result()

        samples = np.frombuffer(pcm_chunk[:usable_length], dtype=np.int16)
        return self.process_samples(samples)

    def process_samples(self, samples) -> SpeechBoundaryResult:
        """Process int16 or float audio samples."""
        samples = self._samples_to_float32(samples)
        if samples.size == 0:
            return self._empty_result()

        if self._pending_samples.size:
            samples = np.concatenate((self._pending_samples, samples))

        processed_frames = 0
        events = []
        offset = 0

        while offset + self.frame_samples <= samples.size:
            frame = samples[offset:offset + self.frame_samples]
            frame_info = self._analyze_frame(frame)
            self._frames.append(frame_info)
            processed_frames += 1

            event = self._maybe_detect_boundary()
            if event is not None:
                events.append(event)

            self._trim_history()
            offset += self.frame_samples

        self._pending_samples = samples[offset:].copy()

        return SpeechBoundaryResult(
            events=events,
            current_energy_db=self._current_energy_db,
            current_rms=self._current_rms,
            noise_floor_db=self._noise_floor_db,
            is_speech=self._current_is_speech,
            is_vowel_like=self._current_is_vowel_like,
            voicing_score=self._current_voicing_score,
            processed_frames=processed_frames,
        )

    def _empty_result(self) -> SpeechBoundaryResult:
        return SpeechBoundaryResult(
            events=[],
            current_energy_db=self._current_energy_db,
            current_rms=self._current_rms,
            noise_floor_db=self._noise_floor_db,
            is_speech=self._current_is_speech,
            is_vowel_like=self._current_is_vowel_like,
            voicing_score=self._current_voicing_score,
            processed_frames=0,
        )

    def _samples_to_float32(self, samples) -> np.ndarray:
        if samples is None:
            return np.empty(0, dtype=np.float32)

        if isinstance(samples, bytes):
            usable_length = len(samples) - (len(samples) % 2)
            if usable_length <= 0:
                return np.empty(0, dtype=np.float32)
            samples = np.frombuffer(samples[:usable_length], dtype=np.int16)
        else:
            samples = np.asarray(samples)

        if samples.size == 0:
            return np.empty(0, dtype=np.float32)

        if np.issubdtype(samples.dtype, np.integer):
            return samples.astype(np.float32) / INT16_MAX_ABS_VALUE

        samples = samples.astype(np.float32, copy=False)
        return np.clip(samples, -1.0, 1.0)

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-12)
        energy_db = 20.0 * math.log10(rms + 1e-12)
        zero_crossing_rate = self._zero_crossing_rate(frame)
        voicing_score = self._voicing_score(frame, rms)

        self._update_noise_floor(energy_db)

        speech_threshold_db = self._noise_floor_db + self.speech_margin_db
        is_speech = rms >= self.min_rms and energy_db >= speech_threshold_db
        vowel_threshold_db = self._noise_floor_db + self.vowel_margin_db
        is_vowel_like = (
            is_speech
            and energy_db >= vowel_threshold_db
            and voicing_score >= self.min_voicing_score
            and zero_crossing_rate <= self.max_vowel_zero_crossing_rate
        )

        start_sample = self._processed_samples
        end_sample = start_sample + self.frame_samples

        self._processed_samples = end_sample
        self._frame_index += 1
        self._current_energy_db = energy_db
        self._current_rms = rms
        self._current_is_speech = is_speech
        self._current_is_vowel_like = is_vowel_like
        self._current_voicing_score = voicing_score

        return {
            "index": self._frame_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "center_sample": start_sample + self.frame_samples // 2,
            "rms": rms,
            "energy_db": energy_db,
            "noise_floor_db": self._noise_floor_db,
            "zero_crossing_rate": zero_crossing_rate,
            "voicing_score": voicing_score,
            "is_speech": is_speech,
            "is_vowel_like": is_vowel_like,
        }

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        if frame.size < 2:
            return 0.0
        signs = np.signbit(frame)
        return float(np.count_nonzero(signs[1:] != signs[:-1])) / float(frame.size - 1)

    def _voicing_score(self, frame: np.ndarray, rms: float) -> float:
        if frame.size < 3 or rms < self.min_rms:
            return 0.0

        centered = frame - float(np.mean(frame))
        energy = float(np.dot(centered, centered))
        if energy <= 1e-9:
            return 0.0

        min_lag = max(1, int(self.sample_rate / 350.0))
        max_lag = min(frame.size - 2, int(self.sample_rate / 75.0))
        if min_lag >= max_lag:
            return 0.0

        best = 0.0
        for lag in range(min_lag, max_lag + 1):
            left = centered[:-lag]
            right = centered[lag:]
            denom = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
            if denom <= 1e-9:
                continue
            corr = float(np.dot(left, right)) / denom
            if corr > best:
                best = corr

        return self._clamp(best, 0.0, 1.0)

    def _update_noise_floor(self, energy_db: float) -> None:
        if self._frame_index < 10:
            self._noise_floor_db = min(self._noise_floor_db, energy_db)
            return

        if energy_db < self._noise_floor_db + 3.0:
            self._noise_floor_db = 0.92 * self._noise_floor_db + 0.08 * energy_db
        else:
            capped_energy = min(energy_db, self._noise_floor_db + 18.0)
            self._noise_floor_db = 0.997 * self._noise_floor_db + 0.003 * capped_energy

        self._noise_floor_db = self._clamp(self._noise_floor_db, -100.0, -20.0)

    def _maybe_detect_boundary(self) -> Optional[SpeechBoundaryEvent]:
        candidate_pos = len(self._frames) - self.lookahead_frames - 1
        if candidate_pos < 1:
            return None

        candidate = self._frames[candidate_pos]
        candidate_index = int(candidate["index"])
        if candidate_index <= self._last_evaluated_frame_index:
            return None

        self._last_evaluated_frame_index = candidate_index

        min_interval_samples = int(
            round(self.sample_rate * self.min_boundary_interval_ms / 1000.0)
        )
        if candidate["center_sample"] - self._last_boundary_sample < min_interval_samples:
            return None

        before_start = max(0, candidate_pos - self.before_window_frames)
        before = self._frames[before_start:candidate_pos]
        after = self._frames[candidate_pos + 1:candidate_pos + 1 + self.lookahead_frames]

        if len(before) < self.min_voiced_frames or len(after) < self.lookahead_frames:
            return None

        recent_voiced = before[-self.min_voiced_frames:]
        voiced_before = sum(1 for frame in recent_voiced if frame["is_speech"])
        min_voiced_required = max(1, int(math.ceil(self.min_voiced_frames * 0.65)))
        if voiced_before < min_voiced_required:
            return None

        recent_vowel_window = before[-max(self.min_vowel_frames + 3, 8):]
        vowel_frames = [frame for frame in recent_vowel_window if frame["is_vowel_like"]]
        if len(vowel_frames) < self.min_vowel_frames:
            return None

        before_energies = np.array([frame["energy_db"] for frame in before], dtype=np.float32)
        after_energies = np.array([frame["energy_db"] for frame in after], dtype=np.float32)
        candidate_energy = float(candidate["energy_db"])

        peak_before = float(np.max(before_energies))
        avg_before = float(np.mean(before_energies[-min(len(before_energies), 5):]))
        peak_after = float(np.max(after_energies))
        avg_after = float(np.mean(after_energies))

        drop_db = peak_before - candidate_energy
        valley_depth_db = min(avg_before, peak_after) - candidate_energy
        recovery_db = peak_after - candidate_energy
        after_speech = any(frame["is_speech"] for frame in after)
        after_vowel = any(frame["is_vowel_like"] for frame in after)
        after_quiet = avg_after < candidate["noise_floor_db"] + self.speech_margin_db + 2.0
        candidate_vowel = candidate["is_vowel_like"]

        neighbor_frames = before[-2:] + after[:2]
        local_minimum = all(
            candidate_energy <= frame["energy_db"] + 0.8
            for frame in neighbor_frames
        )

        vowel_peak_energy = max(frame["energy_db"] for frame in vowel_frames)
        vowel_drop_db = vowel_peak_energy - candidate_energy
        vowel_ended = (
            not candidate_vowel
            and not after_vowel
            and vowel_drop_db >= max(1.5, self.drop_db * 0.55)
        )
        vowel_valley = (
            local_minimum
            and vowel_drop_db >= self.drop_db
            and valley_depth_db >= self.valley_depth_db
        )

        if not vowel_ended and not vowel_valley:
            return None

        score = 0.0
        score += self._score_component(vowel_drop_db, self.drop_db)
        score += self._score_component(valley_depth_db, self.valley_depth_db)
        score += 0.8 * self._score_component(
            max(frame["voicing_score"] for frame in vowel_frames),
            self.min_voicing_score,
        )

        if after_speech and not after_vowel:
            score += 0.7 * self._score_component(recovery_db, self.recovery_db)
        elif after_quiet:
            score += 0.45

        if candidate["rms"] < self.min_rms * 1.25:
            score += 0.2

        if score < 2.0:
            return None

        reason = "vowel-ended" if after_speech else "vowel-to-pause"
        boundary_sample = int(candidate["center_sample"])
        self._last_boundary_sample = boundary_sample

        latency_samples = self._processed_samples - boundary_sample
        latency_ms = 1000.0 * latency_samples / float(self.sample_rate)

        return SpeechBoundaryEvent(
            boundary_sample=boundary_sample,
            boundary_time_seconds=boundary_sample / float(self.sample_rate),
            score=score,
            reason=reason,
            energy_db=candidate_energy,
            noise_floor_db=float(candidate["noise_floor_db"]),
            drop_db=drop_db,
            valley_depth_db=valley_depth_db,
            latency_ms=latency_ms,
        )

    def _trim_history(self) -> None:
        max_frames = self.history_frames + self.lookahead_frames + 4
        if len(self._frames) > max_frames:
            del self._frames[:len(self._frames) - max_frames]

    def _score_component(self, value: float, threshold: float) -> float:
        if threshold <= 0:
            return 1.0
        return self._clamp(value / threshold, 0.0, 1.25)

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
