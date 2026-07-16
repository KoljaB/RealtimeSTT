import argparse
import asyncio
import collections
import datetime
import json
import logging
import math
import threading
import time
import uuid
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - numpy is a core dependency
    raise RuntimeError(
        "The FastAPI server requires numpy. Install project dependencies first."
    ) from exc

try:
    from .protocol import (
        AudioPacketError,
        decode_audio_packet,
        normalize_engine_name,
        parse_json_object,
        require_positive_int,
    )
except ImportError:
    from protocol import (
        AudioPacketError,
        decode_audio_packet,
        normalize_engine_name,
        parse_json_object,
        require_positive_int,
    )


LOGGER = logging.getLogger("realtimestt.fastapi")
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"
SERVER_SAMPLE_RATE = 16000
INT16_MAX_ABS_VALUE = 32768.0

BASE_TUNING_DEFAULTS = {
    "beam_size": 5,
    "beam_size_realtime": 3,
    "batch_size": 16,
    "realtime_batch_size": 16,
    "realtime_processing_pause": 0.02,
    "min_length_of_recording": 0.2,
    "post_speech_silence_duration": 0.55,
    "early_transcription_on_silence": 0.2,
}

TUNING_PROFILES = {
    "custom": {
        "description": "Use explicit CLI/default values.",
        "settings": {},
    },
    "parakeet-low-latency": {
        "description": "Parakeet profile tuned for frequent interim updates.",
        "settings": {
            "batch_size": 1,
            "realtime_batch_size": 1,
            "realtime_processing_pause": 0.04,
            "min_length_of_recording": 0.18,
            "post_speech_silence_duration": 0.45,
            "early_transcription_on_silence": 0.15,
        },
    },
    "parakeet-balanced": {
        "description": "Parakeet profile balancing latency and final stability.",
        "settings": {
            "batch_size": 8,
            "realtime_batch_size": 4,
            "realtime_processing_pause": 0.06,
            "min_length_of_recording": 0.2,
            "post_speech_silence_duration": 0.55,
            "early_transcription_on_silence": 0.2,
        },
    },
    "parakeet-accurate-final": {
        "description": "Parakeet profile favoring calmer segmentation and final quality.",
        "settings": {
            "batch_size": 16,
            "realtime_batch_size": 8,
            "realtime_processing_pause": 0.1,
            "min_length_of_recording": 0.3,
            "post_speech_silence_duration": 0.7,
            "early_transcription_on_silence": 0.35,
        },
    },
}

ACTIVE_RUNTIME_SETTINGS = {
    "log_level",
    "max_active_speakers",
    "max_audio_packet_bytes",
    "max_final_queue_depth_per_session",
    "max_global_inference_queue_depth",
    "max_realtime_queue_age_ms",
    "max_sessions",
    "realtime_degradation_threshold_ms",
}

NEW_SESSION_RUNTIME_SETTINGS = {
    "audio_queue_size",
    "early_transcription_on_silence",
    "initial_prompt",
    "initial_prompt_realtime",
    "max_audio_queue_seconds_per_session",
    "min_gap_between_recordings",
    "min_length_of_recording",
    "openwakeword_inference_framework",
    "openwakeword_model_paths",
    "post_speech_silence_duration",
    "pre_recording_buffer_duration",
    "realtime_batch_size",
    "realtime_boundary_detector_sensitivity",
    "realtime_boundary_followup_delays",
    "realtime_callback",
    "realtime_max_audio_seconds",
    "realtime_min_audio_seconds",
    "realtime_processing_pause",
    "realtime_transcription_use_syllable_boundaries",
    "silero_sensitivity",
    "vad_energy_threshold",
    "vad_filter",
    "wake_word_activation_delay",
    "wake_word_buffer_duration",
    "wake_word_followup_window",
    "wake_word_timeout",
    "wake_words",
    "wake_words_sensitivity",
    "wakeword_backend",
    "webrtc_sensitivity",
}

STARTUP_ONLY_SETTINGS = {
    "batch_size",
    "beam_size",
    "beam_size_realtime",
    "compute_type",
    "device",
    "download_root",
    "gpu_device_index",
    "host",
    "language",
    "model",
    "model_warmup",
    "normalize_audio",
    "port",
    "realtime_model",
    "realtime_transcription_engine",
    "realtime_transcription_engine_options",
    "transcription_engine",
    "transcription_engine_options",
    "tuning_description",
    "tuning_profile",
    "use_main_model_for_realtime",
}

INT_SETTINGS = {
    "audio_queue_size",
    "batch_size",
    "beam_size",
    "beam_size_realtime",
    "gpu_device_index",
    "max_active_speakers",
    "max_audio_packet_bytes",
    "max_final_queue_depth_per_session",
    "max_global_inference_queue_depth",
    "max_realtime_queue_age_ms",
    "max_sessions",
    "port",
    "realtime_batch_size",
    "realtime_degradation_threshold_ms",
    "webrtc_sensitivity",
}

FLOAT_SETTINGS = {
    "early_transcription_on_silence",
    "max_audio_queue_seconds_per_session",
    "min_gap_between_recordings",
    "min_length_of_recording",
    "post_speech_silence_duration",
    "pre_recording_buffer_duration",
    "realtime_boundary_detector_sensitivity",
    "realtime_max_audio_seconds",
    "realtime_min_audio_seconds",
    "realtime_processing_pause",
    "silero_sensitivity",
    "vad_energy_threshold",
    "wake_word_activation_delay",
    "wake_word_buffer_duration",
    "wake_word_followup_window",
    "wake_word_timeout",
    "wake_words_sensitivity",
}

BOOL_SETTINGS = {
    "model_warmup",
    "normalize_audio",
    "realtime_transcription_use_syllable_boundaries",
    "use_main_model_for_realtime",
    "vad_filter",
}

OPTIONAL_STRING_SETTINGS = {
    "download_root",
    "initial_prompt",
    "initial_prompt_realtime",
    "openwakeword_model_paths",
    "realtime_transcription_engine",
}

DICT_SETTINGS = {
    "realtime_transcription_engine_options",
    "transcription_engine_options",
}

TUPLE_FLOAT_SETTINGS = {"realtime_boundary_followup_delays"}


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8010
    tuning_profile: str = "custom"
    tuning_description: str = TUNING_PROFILES["custom"]["description"]
    model: str = "small.en"
    realtime_model: str = "tiny.en"
    language: str = "en"
    transcription_engine: str = "faster_whisper"
    realtime_transcription_engine: Optional[str] = None
    transcription_engine_options: Optional[Dict[str, Any]] = None
    realtime_transcription_engine_options: Optional[Dict[str, Any]] = None
    download_root: Optional[str] = None
    compute_type: str = "default"
    device: str = "cuda"
    gpu_device_index: int = 0
    beam_size: int = 5
    beam_size_realtime: int = 3
    batch_size: int = 16
    realtime_batch_size: int = 16
    vad_filter: bool = True
    normalize_audio: bool = False
    realtime_callback: str = "update"
    min_length_of_recording: float = 0.2
    min_gap_between_recordings: float = 0.0
    post_speech_silence_duration: float = 0.55
    silero_sensitivity: float = 0.05
    webrtc_sensitivity: int = 3
    realtime_processing_pause: float = 0.02
    realtime_transcription_use_syllable_boundaries: bool = False
    realtime_boundary_detector_sensitivity: float = 0.6
    realtime_boundary_followup_delays: Tuple[float, ...] = (0.05, 0.2)
    early_transcription_on_silence: float = 0.2
    initial_prompt: Optional[str] = None
    initial_prompt_realtime: Optional[str] = None
    wakeword_backend: str = ""
    openwakeword_model_paths: Optional[str] = None
    openwakeword_inference_framework: str = "onnx"
    wake_words: str = ""
    wake_words_sensitivity: float = 0.5
    wake_word_activation_delay: float = 0.0
    wake_word_timeout: float = 5.0
    wake_word_buffer_duration: float = 0.1
    wake_word_followup_window: float = 0.0
    use_main_model_for_realtime: bool = False
    audio_queue_size: int = 128
    max_audio_packet_bytes: int = 512 * 1024
    log_level: str = "INFO"
    max_sessions: int = 4
    max_active_speakers: int = 4
    max_audio_queue_seconds_per_session: float = 30.0
    pre_recording_buffer_duration: float = 0.75
    max_realtime_queue_age_ms: int = 1500
    max_final_queue_depth_per_session: int = 8
    max_global_inference_queue_depth: int = 64
    realtime_degradation_threshold_ms: int = 1500
    realtime_min_audio_seconds: float = 0.25
    realtime_max_audio_seconds: float = 20.0
    vad_energy_threshold: float = 250.0
    model_warmup: bool = True

    def public_dict(self):
        data = asdict(self)
        data.pop("transcription_engine_options", None)
        data.pop("realtime_transcription_engine_options", None)
        data["wake_word_enabled"] = self.wake_word_enabled()
        return data

    def wake_word_enabled(self):
        return bool(str(self.wakeword_backend or "").strip() and str(self.wake_words or "").strip())


def runtime_settings_contract():
    return {
        "activeSessionSafe": sorted(ACTIVE_RUNTIME_SETTINGS),
        "newSessionOnly": sorted(NEW_SESSION_RUNTIME_SETTINGS),
        "startupOnly": sorted(STARTUP_ONLY_SETTINGS),
    }


def coerce_setting_value(name, value):
    if name in BOOL_SETTINGS:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value
    if name in INT_SETTINGS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value
    if name in FLOAT_SETTINGS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)
    if name in TUPLE_FLOAT_SETTINGS:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be a list of numbers")
        return tuple(float(item) for item in value)
    if name in DICT_SETTINGS:
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object or null")
        return value
    if name in OPTIONAL_STRING_SETTINGS:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


class SegmentState:
    def __init__(self):
        self._lock = threading.Lock()
        self._segment_id = 1
        self._has_realtime = False

    def realtime(self):
        with self._lock:
            self._has_realtime = True
            return self._segment_id

    def final(self):
        with self._lock:
            segment_id = self._segment_id
            self._segment_id += 1
            self._has_realtime = False
            return segment_id

    def current(self):
        with self._lock:
            return self._segment_id

    def reset(self):
        with self._lock:
            self._segment_id += 1
            self._has_realtime = False
            return self._segment_id


def timestamp_iso(timestamp):
    return (
        datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def segment_text_fields(segment):
    fields = {}
    for key in (
        "durationSeconds",
        "endReason",
        "preRecordingBuffer",
        "recordingEndedAt",
        "recordingEndedAtIso",
        "recordingStartedAt",
        "recordingStartedAtIso",
        "wakeWord",
    ):
        if key in segment:
            fields[key] = segment[key]
    return fields


class SegmentTimelineTracker:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._segments = {}
        self._current_segment_id = None
        self._wakeword_wait_started_at = None
        self._pending_wakeword_detected_at = None
        self._last_wakeword_timeout_at = None

    def reset(self):
        with self._lock:
            self._segments.clear()
            self._current_segment_id = None
            self._wakeword_wait_started_at = None
            self._pending_wakeword_detected_at = None
            self._last_wakeword_timeout_at = None

    def mark_wakeword_wait_started(self, timestamp=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            self._wakeword_wait_started_at = timestamp
            self._pending_wakeword_detected_at = None
            return {
                "wakeWord": self._wakeword_payload(
                    wait_started_at=timestamp,
                    state="waiting_for_wake_word",
                )
            }

    def mark_wakeword_wait_ended(self, timestamp=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            wait_started_at = self._wakeword_wait_started_at
            self._wakeword_wait_started_at = None
            return {
                "wakeWord": self._wakeword_payload(
                    wait_started_at=wait_started_at,
                    wait_ended_at=timestamp,
                    state="wake_word_wait_ended",
                )
            }

    def mark_wakeword_detected(self, timestamp=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            self._pending_wakeword_detected_at = timestamp
            return {
                "wakeWord": self._wakeword_payload(
                    wait_started_at=self._wakeword_wait_started_at,
                    detected_at=timestamp,
                    state="wake_word_detected_waiting_for_voice",
                )
            }

    def mark_wakeword_timeout(self, timestamp=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            self._last_wakeword_timeout_at = timestamp
            self._pending_wakeword_detected_at = None
            return {
                "wakeWord": self._wakeword_payload(
                    wait_started_at=self._wakeword_wait_started_at,
                    timeout_at=timestamp,
                    state="wake_word_timeout",
                )
            }

    def mark_recording_started(self, segment_id, actual_preroll_seconds=None, timestamp=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        configured_preroll = max(0.0, float(self.settings.pre_recording_buffer_duration))
        included_preroll = (
            max(0.0, float(actual_preroll_seconds))
            if actual_preroll_seconds is not None
            else configured_preroll
        )
        prebuffer = {
            "configuredSeconds": configured_preroll,
            "includedSeconds": included_preroll,
            "startTimestamp": timestamp - included_preroll,
            "startTimestampIso": timestamp_iso(timestamp - included_preroll),
            "endTimestamp": timestamp,
            "endTimestampIso": timestamp_iso(timestamp),
            "exact": actual_preroll_seconds is not None,
        }
        with self._lock:
            segment = self._segment_locked(segment_id)
            segment.update({
                "segmentId": segment_id,
                "recordingStartedAt": timestamp,
                "recordingStartedAtIso": timestamp_iso(timestamp),
                "recordingEndedAt": None,
                "recordingEndedAtIso": None,
                "durationSeconds": None,
                "endReason": None,
                "preRecordingBuffer": prebuffer,
            })
            if self._pending_wakeword_detected_at is not None:
                segment["wakeWord"] = self._wakeword_payload(
                    wait_started_at=self._wakeword_wait_started_at,
                    detected_at=self._pending_wakeword_detected_at,
                    state="recording",
                )
            self._current_segment_id = segment_id
            return self._copy_segment_locked(segment_id)

    def mark_recording_ended(
        self,
        reason,
        segment_id=None,
        actual_duration_seconds=None,
        timestamp=None,
    ):
        timestamp = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            if segment_id is None:
                segment_id = self._current_segment_id
            if segment_id is None:
                return None
            segment = self._segment_locked(segment_id)
            started_at = segment.get("recordingStartedAt")
            duration = actual_duration_seconds
            if duration is None and started_at is not None:
                duration = max(0.0, timestamp - float(started_at))
            segment.update({
                "recordingEndedAt": timestamp,
                "recordingEndedAtIso": timestamp_iso(timestamp),
                "durationSeconds": duration,
                "endReason": reason,
            })
            self._current_segment_id = None
            self._pending_wakeword_detected_at = None
            return self._copy_segment_locked(segment_id)

    def snapshot(self, segment_id=None):
        with self._lock:
            if segment_id is None:
                segment_id = self._current_segment_id
            if segment_id is None:
                return None
            return self._copy_segment_locked(segment_id)

    def _segment_locked(self, segment_id):
        return self._segments.setdefault(segment_id, {"segmentId": segment_id})

    def _copy_segment_locked(self, segment_id):
        segment = self._segments.get(segment_id)
        if segment is None:
            return None
        payload = {}
        for key, value in segment.items():
            if value is None:
                continue
            if isinstance(value, dict):
                payload[key] = dict(value)
            else:
                payload[key] = value
        return payload

    def _wakeword_payload(
        self,
        *,
        wait_started_at=None,
        wait_ended_at=None,
        detected_at=None,
        timeout_at=None,
        state=None,
    ):
        payload = {
            "enabled": self.settings.wake_word_enabled(),
            "backend": self.settings.wakeword_backend,
            "wakeWords": self.settings.wake_words,
            "state": state,
        }
        timestamps = {
            "waitStartedAt": wait_started_at,
            "waitEndedAt": wait_ended_at,
            "detectedAt": detected_at,
            "timeoutAt": timeout_at,
        }
        for key, value in timestamps.items():
            if value is None:
                continue
            payload[key] = value
            payload[f"{key}Iso"] = timestamp_iso(value)
        return payload


class RunningStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0
        self._total = 0.0
        self._max = 0.0
        self._recent = collections.deque(maxlen=256)

    def record(self, value):
        value = float(value)
        with self._lock:
            self._count += 1
            self._total += value
            self._max = max(self._max, value)
            self._recent.append(value)

    def snapshot_ms(self):
        with self._lock:
            recent = sorted(self._recent)
            count = self._count
            total = self._total
            max_value = self._max

        def percentile(fraction):
            if not recent:
                return 0.0
            index = min(len(recent) - 1, int(round((len(recent) - 1) * fraction)))
            return recent[index] * 1000.0

        return {
            "count": count,
            "avgMs": (total / count * 1000.0) if count else 0.0,
            "maxMs": max_value * 1000.0,
            "p50Ms": percentile(0.50),
            "p95Ms": percentile(0.95),
        }


@dataclass(frozen=True)
class InferenceJob:
    request_id: str
    session_id: str
    kind: str
    audio: Any
    language: Optional[str]
    use_prompt: bool
    segment_id: int
    sequence: int
    generation: int
    created_at: float
    deadline_at: Optional[float] = None
    sample_rate: int = SERVER_SAMPLE_RATE


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    session_id: str
    kind: str
    segment_id: int
    sequence: int
    generation: int
    text: str
    error: Optional[str]
    created_at: float
    started_at: float
    completed_at: float
    queue_delay: float
    inference_duration: float
    total_latency: float


@dataclass(frozen=True)
class QueueSubmitResult:
    accepted: bool
    reason: str = ""
    coalesced: bool = False


class ConnectionManager:
    def __init__(self):
        self._connections = {}
        self._lock = asyncio.Lock()
        self._loop = None

    def bind_loop(self, loop):
        self._loop = loop

    async def connect(self, session_id, websocket):
        await websocket.accept()
        async with self._lock:
            self._connections[session_id] = websocket

    async def disconnect(self, session_id):
        async with self._lock:
            self._connections.pop(session_id, None)

    async def send(self, session_id, message):
        payload = json.dumps(message, separators=(",", ":"))
        async with self._lock:
            websocket = self._connections.get(session_id)

        if websocket is None:
            return False

        try:
            await websocket.send_text(payload)
            return True
        except Exception:
            async with self._lock:
                if self._connections.get(session_id) is websocket:
                    self._connections.pop(session_id, None)
            return False

    async def send_all(self, message):
        payload = json.dumps(message, separators=(",", ":"))
        async with self._lock:
            connections = list(self._connections.items())

        stale = []
        for session_id, websocket in connections:
            try:
                await websocket.send_text(payload)
            except Exception:
                stale.append((session_id, websocket))

        if stale:
            async with self._lock:
                for session_id, websocket in stale:
                    if self._connections.get(session_id) is websocket:
                        self._connections.pop(session_id, None)

    def publish_session(self, session_id, message):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.send(session_id, message), self._loop)

    def publish_all(self, message):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.send_all(message), self._loop)


class FairInferenceQueue:
    def __init__(self, name, settings: ServerSettings, drop_callback=None):
        self.name = name
        self.settings = settings
        self.drop_callback = drop_callback
        self._condition = threading.Condition()
        self._sessions = {}
        self._ordered_sessions = collections.deque()
        self._queued_session_ids = set()
        self._total_queued = 0
        self._closed = False
        self._coalesced_realtime = 0
        self._stale_realtime_dropped = 0
        self._rejected_jobs = 0

    def submit(self, job: InferenceJob):
        dropped_jobs = []
        with self._condition:
            if self._closed:
                return QueueSubmitResult(False, "scheduler is stopped")

            state = self._sessions.setdefault(
                job.session_id,
                {"final": collections.deque(), "realtime": None},
            )

            if job.kind == "final":
                if len(state["final"]) >= self.settings.max_final_queue_depth_per_session:
                    self._rejected_jobs += 1
                    return QueueSubmitResult(False, "session final queue is full")
                if self._total_queued >= self.settings.max_global_inference_queue_depth:
                    self._rejected_jobs += 1
                    return QueueSubmitResult(False, "global inference queue is full")
                state["final"].append(job)
                self._total_queued += 1
                self._ensure_session_locked(job.session_id)
                self._condition.notify()
                return QueueSubmitResult(True)

            if job.kind != "realtime":
                self._rejected_jobs += 1
                return QueueSubmitResult(False, f"unknown inference kind: {job.kind}")

            old_job = state["realtime"]
            if old_job is not None:
                state["realtime"] = job
                dropped_jobs.append((old_job, "coalesced"))
                self._coalesced_realtime += 1
                self._ensure_session_locked(job.session_id)
                self._condition.notify()
                result = QueueSubmitResult(True, coalesced=True)
            elif self._total_queued >= self.settings.max_global_inference_queue_depth:
                self._rejected_jobs += 1
                result = QueueSubmitResult(False, "global inference queue is full")
            else:
                state["realtime"] = job
                self._total_queued += 1
                self._ensure_session_locked(job.session_id)
                self._condition.notify()
                result = QueueSubmitResult(True)

        self._notify_drops(dropped_jobs)
        return result

    def get(self):
        while True:
            stale_jobs = []
            job = None
            with self._condition:
                while not self._closed:
                    job = None
                    now = time.monotonic()
                    while self._ordered_sessions:
                        session_id = self._ordered_sessions.popleft()
                        self._queued_session_ids.discard(session_id)
                        state = self._sessions.get(session_id)
                        if state is None:
                            continue

                        job = None
                        if state["final"]:
                            job = state["final"].popleft()
                        elif state["realtime"] is not None:
                            realtime_job = state["realtime"]
                            state["realtime"] = None
                            if (
                                realtime_job.deadline_at is not None
                                and realtime_job.deadline_at < now
                            ):
                                self._total_queued -= 1
                                self._stale_realtime_dropped += 1
                                stale_jobs.append((realtime_job, "stale"))
                                self._cleanup_session_locked(session_id)
                                continue
                            job = realtime_job

                        if job is None:
                            self._cleanup_session_locked(session_id)
                            continue

                        self._total_queued -= 1
                        if self._session_has_work_locked(session_id):
                            self._ensure_session_locked(session_id)
                        else:
                            self._cleanup_session_locked(session_id)
                        break

                    if job is not None:
                        break
                    if stale_jobs:
                        break
                    self._condition.wait(timeout=0.2)

                if self._closed:
                    return None

            self._notify_drops(stale_jobs)
            if job is not None:
                return job

    def cancel_session(self, session_id):
        dropped = []
        with self._condition:
            state = self._sessions.pop(session_id, None)
            self._queued_session_ids.discard(session_id)
            if state is None:
                return
            while state["final"]:
                dropped.append((state["final"].popleft(), "cancelled"))
                self._total_queued -= 1
            if state["realtime"] is not None:
                dropped.append((state["realtime"], "cancelled"))
                self._total_queued -= 1
                state["realtime"] = None
            self._condition.notify_all()
        self._notify_drops(dropped)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            per_session = {
                session_id: {
                    "final": len(state["final"]),
                    "realtime": 1 if state["realtime"] is not None else 0,
                }
                for session_id, state in self._sessions.items()
            }
            return {
                "name": self.name,
                "queued": self._total_queued,
                "sessions": len(self._sessions),
                "perSession": per_session,
                "coalescedRealtime": self._coalesced_realtime,
                "staleRealtimeDropped": self._stale_realtime_dropped,
                "rejectedJobs": self._rejected_jobs,
            }

    def _ensure_session_locked(self, session_id):
        if session_id not in self._queued_session_ids:
            self._queued_session_ids.add(session_id)
            self._ordered_sessions.append(session_id)

    def _session_has_work_locked(self, session_id):
        state = self._sessions.get(session_id)
        if state is None:
            return False
        return bool(state["final"]) or state["realtime"] is not None

    def _cleanup_session_locked(self, session_id):
        if not self._session_has_work_locked(session_id):
            self._sessions.pop(session_id, None)

    def _notify_drops(self, dropped_jobs):
        if not self.drop_callback:
            return
        for job, reason in dropped_jobs:
            try:
                self.drop_callback(job, reason, self.name)
            except Exception:
                LOGGER.exception("Drop callback failed")


class SharedEngineWorker:
    def __init__(
        self,
        name,
        settings: ServerSettings,
        queue: FairInferenceQueue,
        engine_factory: Callable[[], Any],
        result_callback: Callable[[InferenceResult], None],
        error_callback: Optional[Callable[[str, Exception], None]] = None,
    ):
        self.name = name
        self.settings = settings
        self.queue = queue
        self.engine_factory = engine_factory
        self.result_callback = result_callback
        self.error_callback = error_callback
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.thread = None
        self.engine = None
        self.load_error = None
        self.busy_seconds = 0.0
        self.started_at = time.monotonic()
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.queue_delay = RunningStats()
        self.inference_duration = RunningStats()
        self.total_latency = RunningStats()

    def start(self):
        self.thread = threading.Thread(
            target=self._worker,
            name=f"RealtimeSTT-{self.name}-Inference",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.queue.close()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def snapshot(self):
        elapsed = max(0.001, time.monotonic() - self.started_at)
        return {
            "name": self.name,
            "ready": self.ready.is_set(),
            "healthy": self.load_error is None,
            "completedJobs": self.completed_jobs,
            "failedJobs": self.failed_jobs,
            "busyRatio": min(1.0, self.busy_seconds / elapsed),
            "queueDelay": self.queue_delay.snapshot_ms(),
            "inferenceDuration": self.inference_duration.snapshot_ms(),
            "totalLatency": self.total_latency.snapshot_ms(),
        }

    def _worker(self):
        try:
            self.engine = self.engine_factory()
            self._warmup()
        except Exception as exc:
            self.load_error = exc
            LOGGER.exception("Could not initialize %s inference engine", self.name)
            if self.error_callback:
                self.error_callback(self.name, exc)
        finally:
            self.ready.set()

        while not self.stop_event.is_set():
            job = self.queue.get()
            if job is None:
                break

            started_at = time.monotonic()
            text = ""
            error = None

            try:
                if self.engine is None:
                    raise RuntimeError(f"{self.name} inference engine is unavailable")
                result = self.engine.transcribe(
                    job.audio,
                    language=job.language if job.language else None,
                    use_prompt=job.use_prompt,
                )
                text = (getattr(result, "text", "") or "").strip()
                self.completed_jobs += 1
            except Exception as exc:
                self.failed_jobs += 1
                error = str(exc)
                LOGGER.exception("Inference job failed: %s", job.request_id)

            completed_at = time.monotonic()
            queue_delay = max(0.0, started_at - job.created_at)
            inference_duration = max(0.0, completed_at - started_at)
            total_latency = max(0.0, completed_at - job.created_at)
            self.busy_seconds += inference_duration
            self.queue_delay.record(queue_delay)
            self.inference_duration.record(inference_duration)
            self.total_latency.record(total_latency)
            self.result_callback(
                InferenceResult(
                    request_id=job.request_id,
                    session_id=job.session_id,
                    kind=job.kind,
                    segment_id=job.segment_id,
                    sequence=job.sequence,
                    generation=job.generation,
                    text=text,
                    error=error,
                    created_at=job.created_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    queue_delay=queue_delay,
                    inference_duration=inference_duration,
                    total_latency=total_latency,
                )
            )

    def _warmup(self):
        if not self.settings.model_warmup or self.engine is None:
            return
        warmup_path = Path(__file__).resolve().parents[1] / "RealtimeSTT" / "warmup_audio.wav"
        try:
            audio = read_wav_float32(warmup_path).samples
            self.engine.warmup(audio)
        except Exception:
            LOGGER.debug("Warmup skipped for %s", self.name, exc_info=True)


class InferenceScheduler:
    def __init__(
        self,
        settings: ServerSettings,
        result_callback: Callable[[InferenceResult], None],
        drop_callback: Optional[Callable[[InferenceJob, str, str], None]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None,
    ):
        self.settings = settings
        self.result_callback = result_callback
        self.drop_callback = drop_callback
        self.error_callback = error_callback
        self.main_queue = FairInferenceQueue("main", settings, drop_callback)
        self.realtime_queue = (
            self.main_queue
            if settings.use_main_model_for_realtime
            else FairInferenceQueue("realtime", settings, drop_callback)
        )
        self.main_worker = SharedEngineWorker(
            "main",
            settings,
            self.main_queue,
            self._create_main_engine,
            result_callback,
            error_callback,
        )
        self.realtime_worker = None
        if not settings.use_main_model_for_realtime:
            self.realtime_worker = SharedEngineWorker(
                "realtime",
                settings,
                self.realtime_queue,
                self._create_realtime_engine,
                result_callback,
                error_callback,
            )

    def start(self):
        self.main_worker.start()
        if self.realtime_worker is not None:
            self.realtime_worker.start()

    def stop(self):
        self.main_worker.stop()
        if self.realtime_worker is not None:
            self.realtime_worker.stop()

    def wait_ready(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        workers = [self.main_worker]
        if self.realtime_worker is not None:
            workers.append(self.realtime_worker)

        for worker in workers:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not worker.ready.wait(timeout=remaining):
                return False
        return True

    def healthy(self):
        if self.main_worker.load_error is not None:
            return False
        if self.realtime_worker is not None and self.realtime_worker.load_error is not None:
            return False
        return True

    def submit(self, job: InferenceJob):
        if job.kind == "realtime" and not self.settings.use_main_model_for_realtime:
            return self.realtime_queue.submit(job)
        return self.main_queue.submit(job)

    def cancel_session(self, session_id):
        self.main_queue.cancel_session(session_id)
        if self.realtime_queue is not self.main_queue:
            self.realtime_queue.cancel_session(session_id)

    def snapshot(self):
        data = {
            "mode": (
                "low-memory-one-model"
                if self.settings.use_main_model_for_realtime
                else "balanced-main-plus-realtime"
            ),
            "queues": {"main": self.main_queue.snapshot()},
            "workers": {"main": self.main_worker.snapshot()},
        }
        if self.realtime_queue is not self.main_queue:
            data["queues"]["realtime"] = self.realtime_queue.snapshot()
        if self.realtime_worker is not None:
            data["workers"]["realtime"] = self.realtime_worker.snapshot()
        return data

    def _create_main_engine(self):
        from RealtimeSTT.transcription_engines import (
            TranscriptionEngineConfig,
            create_transcription_engine,
        )

        return create_transcription_engine(
            self.settings.transcription_engine,
            TranscriptionEngineConfig(
                model=self.settings.model,
                download_root=self.settings.download_root,
                compute_type=self.settings.compute_type,
                gpu_device_index=self.settings.gpu_device_index,
                device=effective_device(self.settings.device),
                beam_size=self.settings.beam_size,
                initial_prompt=self.settings.initial_prompt,
                batch_size=self.settings.batch_size,
                vad_filter=self.settings.vad_filter,
                normalize_audio=self.settings.normalize_audio,
                engine_options=self.settings.transcription_engine_options,
            ),
        )

    def _create_realtime_engine(self):
        from RealtimeSTT.transcription_engines import (
            TranscriptionEngineConfig,
            create_transcription_engine,
        )

        return create_transcription_engine(
            self.settings.realtime_transcription_engine
            or self.settings.transcription_engine,
            TranscriptionEngineConfig(
                model=self.settings.realtime_model or self.settings.model,
                download_root=self.settings.download_root,
                compute_type=self.settings.compute_type,
                gpu_device_index=self.settings.gpu_device_index,
                device=effective_device(self.settings.device),
                beam_size=self.settings.beam_size_realtime,
                initial_prompt=self.settings.initial_prompt_realtime,
                batch_size=self.settings.realtime_batch_size,
                vad_filter=self.settings.vad_filter,
                normalize_audio=self.settings.normalize_audio,
                engine_options=(
                    self.settings.realtime_transcription_engine_options
                    if self.settings.realtime_transcription_engine_options is not None
                    else self.settings.transcription_engine_options
                ),
            ),
        )


class SchedulerTranscriptionExecutor:
    def __init__(self, service, session_id, kind):
        self.service = service
        self.session_id = session_id
        self.kind = kind

    def transcribe(self, audio, language=None, use_prompt=True):
        return self.service.transcribe_for_recorder(
            self.session_id,
            self.kind,
            audio,
            language,
            use_prompt,
        )


class VoiceActivityDetector:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.vad = None
        try:
            import webrtcvad

            self.vad = webrtcvad.Vad()
            self.vad.set_mode(int(settings.webrtc_sensitivity))
        except Exception:
            self.vad = None

    def is_speech(self, samples):
        if samples is None or samples.size == 0:
            return False

        if self.vad is not None and samples.size >= 160:
            try:
                frame_samples = 320
                usable = samples.size - (samples.size % frame_samples)
                speech_frames = 0
                checked_frames = 0
                for start in range(0, usable, frame_samples):
                    frame = samples[start:start + frame_samples]
                    checked_frames += 1
                    if self.vad.is_speech(frame.astype(np.int16).tobytes(), SERVER_SAMPLE_RATE):
                        speech_frames += 1
                if checked_frames:
                    return speech_frames / checked_frames >= 0.25
            except Exception:
                LOGGER.debug("webrtcvad failed; falling back to energy VAD", exc_info=True)

        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return rms >= float(self.settings.vad_energy_threshold)


class RealtimeSession:
    def __init__(self, service, session_id):
        self.service = service
        self.settings = replace(service.settings)
        self.session_id = session_id
        self.segment_state = SegmentState()
        self.timeline = SegmentTimelineTracker(self.settings)
        self.vad = VoiceActivityDetector(self.settings)
        self.lock = threading.RLock()
        self.streaming = False
        self.recording = False
        self.status = "idle"
        self.generation = 0
        self.active_segment_id = None
        self.latest_realtime_sequence = 0
        self.last_realtime_submit_at = 0.0
        self.last_speech_at = 0.0
        self.recording_started_at = 0.0
        self.recording_frames: List[Any] = []
        self.recording_sample_count = 0
        self.prebuffer = collections.deque()
        self.prebuffer_sample_count = 0
        self.dropped_audio_chunks = 0
        self.rejected_audio_chunks = 0
        self.coalesced_realtime = 0
        self.stale_realtime_discarded = 0
        self.cancelled_jobs = 0
        self.realtime_submitted = 0
        self.final_submitted = 0
        self.realtime_completed = 0
        self.final_completed = 0
        self.realtime_rejected = 0
        self.final_rejected = 0
        self.final_queue_full = 0
        self.queue_delay = {"realtime": RunningStats(), "final": RunningStats()}
        self.inference_duration = {"realtime": RunningStats(), "final": RunningStats()}
        self.total_latency = {"realtime": RunningStats(), "final": RunningStats()}

    def start_streaming(self):
        with self.lock:
            self.streaming = True
            self.status = (
                "wakeword_wait"
                if self.settings.wake_word_enabled()
                and self.settings.wake_word_activation_delay <= 0
                else "listening"
            )
        self.publish_status(self.status)

    def stop_streaming(self):
        jobs = []
        with self.lock:
            self.streaming = False
            final_job = self._finish_recording_locked("stop")
            if final_job is not None:
                jobs.append(final_job)
            self.status = "idle"
        for job in jobs:
            self.service.submit_inference_job(job)
        self.service.deactivate_speaker(self.session_id)
        self.publish_status("idle")

    def close(self):
        with self.lock:
            self.generation += 1
            self.streaming = False
            self.recording = False
            self.recording_frames = []
            self.recording_sample_count = 0
            self.prebuffer.clear()
            self.prebuffer_sample_count = 0
            self.timeline.reset()
        self.service.scheduler.cancel_session(self.session_id)
        self.service.deactivate_speaker(self.session_id)

    def clear(self):
        with self.lock:
            self.generation += 1
            self.recording = False
            self.recording_frames = []
            self.recording_sample_count = 0
            self.prebuffer.clear()
            self.prebuffer_sample_count = 0
            self.active_segment_id = None
            self.latest_realtime_sequence = 0
            next_segment = self.segment_state.reset()
            self.timeline.reset()
            self.status = self._waiting_state_locked()
        self.service.scheduler.cancel_session(self.session_id)
        self.service.deactivate_speaker(self.session_id)
        self.service.manager.publish_session(
            self.session_id,
            {
                "type": "clear",
                "sessionId": self.session_id,
                "nextSegmentId": next_segment,
            },
        )
        self.publish_status(self.status)

    def ingest_audio_packet(self, packet):
        samples = self.service.packet_to_server_samples(packet)
        now = time.monotonic()
        jobs = []
        warnings = []

        with self.lock:
            self.streaming = True
            speech = self.vad.is_speech(samples)

            if not self.recording:
                if not speech:
                    self._append_prebuffer_locked(samples)
                    return True, None
                if not self.service.try_activate_speaker(self.session_id):
                    self.rejected_audio_chunks += 1
                    return False, "Server active speaker limit reached; audio chunk was ignored."
                self._start_recording_locked(now)
                self.recording_frames.append(samples.copy())
                self.recording_sample_count += int(samples.size)
                self.last_speech_at = now
                self.status = "recording"
                warnings.append(None)
            else:
                self.recording_frames.append(samples.copy())
                self.recording_sample_count += int(samples.size)
                if speech:
                    self.last_speech_at = now

            if self.recording:
                realtime_job = self._maybe_create_realtime_job_locked(now)
                if realtime_job is not None:
                    jobs.append(realtime_job)

                recording_seconds = self.recording_sample_count / float(SERVER_SAMPLE_RATE)
                silence_seconds = now - self.last_speech_at if self.last_speech_at else 0.0
                if (
                    recording_seconds >= self.settings.min_length_of_recording
                    and silence_seconds >= self.settings.post_speech_silence_duration
                ):
                    final_job = self._finish_recording_locked("silence")
                    if final_job is not None:
                        jobs.append(final_job)
                elif recording_seconds >= self.settings.max_audio_queue_seconds_per_session:
                    final_job = self._finish_recording_locked("max_duration")
                    if final_job is not None:
                        jobs.append(final_job)
                    warnings.append("Maximum per-session audio buffer reached; finalized the current segment.")

        for job in jobs:
            self.service.submit_inference_job(job)
        for warning in warnings:
            if warning:
                self.service.manager.publish_session(
                    self.session_id,
                    {"type": "warning", "sessionId": self.session_id, "message": warning},
                )
        self.publish_status(self.status)
        return True, None

    def handle_inference_result(self, result: InferenceResult):
        with self.lock:
            if result.generation != self.generation:
                if result.kind == "realtime":
                    self.stale_realtime_discarded += 1
                return

            if result.kind == "realtime":
                if (
                    not self.recording
                    or result.segment_id != self.active_segment_id
                    or result.sequence < self.latest_realtime_sequence
                ):
                    self.stale_realtime_discarded += 1
                    return
                self.realtime_completed += 1
            else:
                self.final_completed += 1

            self.queue_delay[result.kind].record(result.queue_delay)
            self.inference_duration[result.kind].record(result.inference_duration)
            self.total_latency[result.kind].record(result.total_latency)

        if result.error:
            self.service.manager.publish_session(
                self.session_id,
                {
                    "type": "error",
                    "sessionId": self.session_id,
                    "message": result.error,
                    "where": result.kind,
                    "requestId": result.request_id,
                },
            )
            return

        if not result.text:
            return

        event_timestamp = time.time()
        event = {
            "type": result.kind,
            "sessionId": self.session_id,
            "segmentId": result.segment_id,
            "text": result.text,
            "timestamp": event_timestamp,
            "timestampIso": timestamp_iso(event_timestamp),
            "requestId": result.request_id,
            "queueDelayMs": result.queue_delay * 1000.0,
            "inferenceMs": result.inference_duration * 1000.0,
            "latencyMs": result.total_latency * 1000.0,
        }
        segment = self.timeline.snapshot(result.segment_id)
        if segment is not None:
            event["segment"] = segment
            event.update(segment_text_fields(segment))
        self.service.manager.publish_session(self.session_id, event)
        if result.kind == "final":
            self.publish_status("listening" if self.streaming else "idle")

    def on_job_dropped(self, job: InferenceJob, reason: str):
        with self.lock:
            if reason == "coalesced" and job.kind == "realtime":
                self.coalesced_realtime += 1
            elif reason == "stale" and job.kind == "realtime":
                self.stale_realtime_discarded += 1
            elif reason == "cancelled":
                self.cancelled_jobs += 1

    def on_submit_result(self, job: InferenceJob, result: QueueSubmitResult):
        with self.lock:
            if result.accepted:
                if job.kind == "realtime":
                    self.realtime_submitted += 1
                    if result.coalesced:
                        self.coalesced_realtime += 1
                else:
                    self.final_submitted += 1
                return

            if job.kind == "realtime":
                self.realtime_rejected += 1
            else:
                self.final_rejected += 1
                if "final queue" in result.reason:
                    self.final_queue_full += 1

        message = (
            "Realtime transcription is overloaded; interim update was dropped."
            if job.kind == "realtime"
            else f"Final transcription was rejected: {result.reason}"
        )
        self.service.manager.publish_session(
            self.session_id,
            {
                "type": "warning" if job.kind == "realtime" else "error",
                "sessionId": self.session_id,
                "message": message,
                "where": "scheduler",
            },
        )

    def publish_status(self, state=None):
        with self.lock:
            state = state or self.status
            queue_depth = self.recording_sample_count / float(SERVER_SAMPLE_RATE)
            message = {
                "type": "status",
                "sessionId": self.session_id,
                "state": state,
                "activeClientId": self.session_id if self.streaming else None,
                "queueDepth": round(queue_depth, 3),
                "droppedChunks": self.dropped_audio_chunks,
                "coalescedRealtime": self.coalesced_realtime,
                "staleRealtimeDiscarded": self.stale_realtime_discarded,
                "activeSessions": self.service.session_count(),
                "activeSpeakers": self.service.active_speaker_count(),
                "wakeWordEnabled": self.settings.wake_word_enabled(),
            }
        self.service.manager.publish_session(self.session_id, message)

    def snapshot(self):
        with self.lock:
            return {
                "sessionId": self.session_id,
                "streaming": self.streaming,
                "recording": self.recording,
                "state": self.status,
                "currentSegmentId": self.segment_state.current(),
                "recordingSeconds": self.recording_sample_count / float(SERVER_SAMPLE_RATE),
                "droppedAudioChunks": self.dropped_audio_chunks,
                "rejectedAudioChunks": self.rejected_audio_chunks,
                "coalescedRealtime": self.coalesced_realtime,
                "staleRealtimeDiscarded": self.stale_realtime_discarded,
                "cancelledJobs": self.cancelled_jobs,
                "realtimeSubmitted": self.realtime_submitted,
                "finalSubmitted": self.final_submitted,
                "realtimeCompleted": self.realtime_completed,
                "finalCompleted": self.final_completed,
                "realtimeRejected": self.realtime_rejected,
                "finalRejected": self.final_rejected,
                "finalQueueFull": self.final_queue_full,
                "queueDelay": {
                    "realtime": self.queue_delay["realtime"].snapshot_ms(),
                    "final": self.queue_delay["final"].snapshot_ms(),
                },
                "inferenceDuration": {
                    "realtime": self.inference_duration["realtime"].snapshot_ms(),
                    "final": self.inference_duration["final"].snapshot_ms(),
                },
                "totalLatency": {
                    "realtime": self.total_latency["realtime"].snapshot_ms(),
                    "final": self.total_latency["final"].snapshot_ms(),
                },
            }

    def _start_recording_locked(self, now):
        self.recording = True
        self.active_segment_id = self.segment_state.realtime()
        self.recording_started_at = now
        self.last_speech_at = now
        self.last_realtime_submit_at = 0.0
        self.recording_frames = [frame.copy() for frame in self.prebuffer]
        self.recording_sample_count = sum(int(frame.size) for frame in self.recording_frames)
        self.timeline.mark_recording_started(
            self.active_segment_id,
            actual_preroll_seconds=self.prebuffer_sample_count / float(SERVER_SAMPLE_RATE),
            timestamp=time.time(),
        )
        self.prebuffer.clear()
        self.prebuffer_sample_count = 0

    def _finish_recording_locked(self, reason):
        if not self.recording and not self.recording_frames:
            return None
        audio = self._recording_audio_float32_locked()
        recording_seconds = audio.size / float(SERVER_SAMPLE_RATE) if audio is not None else 0.0
        segment_id = self.active_segment_id
        self.timeline.mark_recording_ended(
            reason,
            segment_id=segment_id,
            actual_duration_seconds=recording_seconds,
            timestamp=time.time(),
        )
        self.recording = False
        self.recording_frames = []
        self.recording_sample_count = 0
        self.active_segment_id = None
        self.last_realtime_submit_at = 0.0
        self.status = self._waiting_state_locked()
        self.service.deactivate_speaker(self.session_id)
        if audio is None or recording_seconds < self.settings.min_length_of_recording:
            return None
        segment_id = self.segment_state.final()
        return InferenceJob(
            request_id=uuid.uuid4().hex,
            session_id=self.session_id,
            kind="final",
            audio=audio,
            language=self.settings.language,
            use_prompt=True,
            segment_id=segment_id,
            sequence=0,
            generation=self.generation,
            created_at=time.monotonic(),
        )

    def _maybe_create_realtime_job_locked(self, now):
        pause = max(0.0, float(self.settings.realtime_processing_pause))
        if pause > 0 and now - self.last_realtime_submit_at < pause:
            return None
        if self.recording_sample_count < int(self.settings.realtime_min_audio_seconds * SERVER_SAMPLE_RATE):
            return None
        audio = self._recording_audio_float32_locked(max_seconds=self.settings.realtime_max_audio_seconds)
        if audio is None or audio.size == 0:
            return None
        self.latest_realtime_sequence += 1
        self.last_realtime_submit_at = now
        return InferenceJob(
            request_id=uuid.uuid4().hex,
            session_id=self.session_id,
            kind="realtime",
            audio=audio,
            language=self.settings.language,
            use_prompt=True,
            segment_id=self.active_segment_id or self.segment_state.realtime(),
            sequence=self.latest_realtime_sequence,
            generation=self.generation,
            created_at=time.monotonic(),
            deadline_at=time.monotonic() + (self.settings.max_realtime_queue_age_ms / 1000.0),
        )

    def _append_prebuffer_locked(self, samples):
        if samples is None or samples.size == 0:
            return
        self.prebuffer.append(samples.copy())
        self.prebuffer_sample_count += int(samples.size)
        max_samples = int(self.settings.pre_recording_buffer_duration * SERVER_SAMPLE_RATE)
        while max_samples >= 0 and self.prebuffer_sample_count > max_samples and self.prebuffer:
            dropped = self.prebuffer.popleft()
            self.prebuffer_sample_count -= int(dropped.size)

    def _recording_audio_float32_locked(self, max_seconds=None):
        if not self.recording_frames:
            return None
        frames = self.recording_frames
        if max_seconds is not None and max_seconds > 0:
            max_samples = int(max_seconds * SERVER_SAMPLE_RATE)
            total = 0
            selected = []
            for frame in reversed(frames):
                selected.append(frame)
                total += int(frame.size)
                if total >= max_samples:
                    break
            frames = list(reversed(selected))
        audio_int16 = np.concatenate(frames).astype(np.int16)
        if max_seconds is not None and max_seconds > 0:
            max_samples = int(max_seconds * SERVER_SAMPLE_RATE)
            if audio_int16.size > max_samples:
                audio_int16 = audio_int16[-max_samples:]
        return audio_int16.astype(np.float32) / INT16_MAX_ABS_VALUE

    def _waiting_state_locked(self, streaming=None):
        if streaming is None:
            streaming = self.streaming
        if not streaming:
            return "idle"
        if self.settings.wake_word_enabled():
            return "wakeword_wait"
        return "listening"


class RecorderBackedRealtimeSession:
    def __init__(self, service, session_id):
        self.service = service
        self.settings = replace(service.settings)
        self.session_id = session_id
        self.segment_state = SegmentState()
        self.timeline = SegmentTimelineTracker(self.settings)
        self.lock = threading.RLock()
        self.streaming = False
        self.status = "idle"
        self.generation = 0
        self.reject_current_recording = False
        self.dropped_audio_chunks = 0
        self.rejected_audio_chunks = 0
        self.coalesced_realtime = 0
        self.stale_realtime_discarded = 0
        self.cancelled_jobs = 0
        self.realtime_submitted = 0
        self.final_submitted = 0
        self.realtime_completed = 0
        self.final_completed = 0
        self.realtime_rejected = 0
        self.final_rejected = 0
        self.forced_finalizations = 0
        self.dropped_recorded_segments = 0
        self.recording_sample_count = 0
        self._recorded_chunk_callback_seen = False
        self._force_finalize_in_progress = False
        self._wakeword_voice_window = False
        self._wakeword_followup_generation = 0
        self._recorder_wake_word_timeout_before_followup = None
        self._recorder_start_recording_before_followup = None
        self._recorder_stop_recording_before_followup = None
        self.queue_delay = {"realtime": RunningStats(), "final": RunningStats()}
        self.inference_duration = {"realtime": RunningStats(), "final": RunningStats()}
        self.total_latency = {"realtime": RunningStats(), "final": RunningStats()}
        self.recorder = self._create_recorder()
        self.text_thread = threading.Thread(
            target=self._text_worker,
            name=f"RealtimeSTTSessionText-{session_id}",
            daemon=True,
        )
        self.text_thread.start()

    def _create_recorder(self):
        recorder_factory = self.service.recorder_factory
        use_structured_stabilization = recorder_factory is None
        if recorder_factory is None:
            from RealtimeSTT import AudioToTextRecorder

            recorder_factory = AudioToTextRecorder

        callback_key = (
            "on_realtime_transcription_stabilized"
            if self.settings.realtime_callback == "stabilized"
            else "on_realtime_transcription_update"
        )
        realtime_engine = self.settings.realtime_transcription_engine
        config = {
            "spinner": False,
            "use_microphone": False,
            "model": self.settings.model,
            "realtime_model_type": self.settings.realtime_model,
            "language": self.settings.language,
            "transcription_engine": self.settings.transcription_engine,
            "realtime_transcription_engine": realtime_engine,
            "transcription_engine_options": self.settings.transcription_engine_options,
            "realtime_transcription_engine_options": (
                self.settings.realtime_transcription_engine_options
            ),
            "download_root": self.settings.download_root,
            "compute_type": self.settings.compute_type,
            "device": self.settings.device,
            "gpu_device_index": self.settings.gpu_device_index,
            "beam_size": self.settings.beam_size,
            "beam_size_realtime": self.settings.beam_size_realtime,
            "batch_size": self.settings.batch_size,
            "realtime_batch_size": self.settings.realtime_batch_size,
            "faster_whisper_vad_filter": self.settings.vad_filter,
            "normalize_audio": self.settings.normalize_audio,
            "enable_realtime_transcription": True,
            "use_main_model_for_realtime": self.settings.use_main_model_for_realtime,
            "realtime_processing_pause": self.settings.realtime_processing_pause,
            "realtime_transcription_use_syllable_boundaries": (
                self.settings.realtime_transcription_use_syllable_boundaries
            ),
            "realtime_boundary_detector_sensitivity": (
                self.settings.realtime_boundary_detector_sensitivity
            ),
            "realtime_boundary_followup_delays": (
                self.settings.realtime_boundary_followup_delays
            ),
            "silero_sensitivity": self.settings.silero_sensitivity,
            "webrtc_sensitivity": self.settings.webrtc_sensitivity,
            "warmup_vad": self.settings.model_warmup,
            "post_speech_silence_duration": self.settings.post_speech_silence_duration,
            "min_length_of_recording": self.settings.min_length_of_recording,
            "min_gap_between_recordings": self.settings.min_gap_between_recordings,
            "early_transcription_on_silence": self.settings.early_transcription_on_silence,
            "initial_prompt": self.settings.initial_prompt,
            "initial_prompt_realtime": self.settings.initial_prompt_realtime,
            "wakeword_backend": self.settings.wakeword_backend,
            "openwakeword_model_paths": self.settings.openwakeword_model_paths,
            "openwakeword_inference_framework": self.settings.openwakeword_inference_framework,
            "wake_words": self.settings.wake_words,
            "wake_words_sensitivity": self.settings.wake_words_sensitivity,
            "wake_word_activation_delay": self.settings.wake_word_activation_delay,
            "wake_word_timeout": self.settings.wake_word_timeout,
            "wake_word_buffer_duration": self.settings.wake_word_buffer_duration,
            "pre_recording_buffer_duration": self.settings.pre_recording_buffer_duration,
            "allowed_latency_limit": self.settings.audio_queue_size,
            "handle_buffer_overflow": True,
            "on_recording_start": self._on_recording_start,
            "on_recording_stop": self._on_recording_stop,
            "on_transcription_start": self._on_transcription_start,
            "on_wakeword_detected": self._on_wakeword_detected,
            "on_wakeword_timeout": self._on_wakeword_timeout,
            "on_wakeword_detection_start": self._on_wakeword_detection_start,
            "on_wakeword_detection_end": self._on_wakeword_detection_end,
            "on_vad_start": self._on_vad_start,
            "on_vad_stop": self._on_vad_stop,
            "on_vad_detect_start": self._on_vad_detect_start,
            "on_vad_detect_stop": self._on_vad_detect_stop,
            "on_recorded_chunk": self._on_recorded_chunk,
            "no_log_file": True,
            "transcription_executor": SchedulerTranscriptionExecutor(
                self.service,
                self.session_id,
                "final",
            ),
            "realtime_transcription_executor": SchedulerTranscriptionExecutor(
                self.service,
                self.session_id,
                "realtime",
            ),
        }
        if use_structured_stabilization:
            config["on_realtime_text_stabilization_update"] = (
                self._on_realtime_stabilization_event
            )
        else:
            config[callback_key] = self._on_realtime_text
        return recorder_factory(**config)

    def start_streaming(self):
        with self.lock:
            self.streaming = True
            self.status = (
                "wakeword_wait"
                if self.settings.wake_word_enabled()
                and self.settings.wake_word_activation_delay <= 0
                else "listening"
            )
        self.publish_status(self.status)

    def stop_streaming(self):
        with self.lock:
            self.streaming = False
            self.status = "idle"
        try:
            self.recorder.flush_buffered_audio()
            self._trim_recorded_audio_queue()
        except Exception:
            LOGGER.debug("Could not flush buffered audio for %s", self.session_id, exc_info=True)
        finally:
            self.service.deactivate_speaker(self.session_id)
        self.publish_status("idle")

    def close(self):
        with self.lock:
            self.generation += 1
            self.streaming = False
            self.status = "closed"
            self.timeline.reset()
            self._wakeword_voice_window = False
            self._wakeword_followup_generation += 1
            self._clear_recorder_followup_gate_locked()
        self.service.scheduler.cancel_session(self.session_id)
        self.service.cancel_pending_recorder_transcriptions(self.session_id)
        self.service.deactivate_speaker(self.session_id)
        try:
            self.recorder.shutdown()
        except Exception:
            LOGGER.debug("Recorder shutdown failed for %s", self.session_id, exc_info=True)
        if self.text_thread is not None:
            self.text_thread.join(timeout=3)

    def clear(self):
        with self.lock:
            self.generation += 1
            next_segment = self.segment_state.reset()
            self.timeline.reset()
            self.reject_current_recording = True
            self.recording_sample_count = 0
            self._wakeword_voice_window = False
            self._wakeword_followup_generation += 1
            self._clear_recorder_followup_gate_locked()
            self.status = self._waiting_state_locked()
        self.service.scheduler.cancel_session(self.session_id)
        self.service.cancel_pending_recorder_transcriptions(self.session_id)
        self.service.deactivate_speaker(self.session_id)
        try:
            self.recorder.abort()
        except Exception:
            LOGGER.debug("Recorder abort failed during clear for %s", self.session_id, exc_info=True)
        self.service.manager.publish_session(
            self.session_id,
            {
                "type": "clear",
                "sessionId": self.session_id,
                "nextSegmentId": next_segment,
            },
        )
        self.publish_status(self.status)

    def ingest_audio_packet(self, packet):
        samples = self.service.packet_to_server_samples(packet)
        if samples.size == 0:
            return True, None
        with self.lock:
            if not self.streaming:
                self.rejected_audio_chunks += 1
                return False, "Audio stream is stopped; send a start command before audio packets."
        try:
            self.recorder.feed_audio(samples, original_sample_rate=SERVER_SAMPLE_RATE)
        except Exception as exc:
            LOGGER.exception("Could not feed recorder audio")
            self.dropped_audio_chunks += 1
            return False, str(exc)
        warning = self._enforce_recording_duration(samples)
        if warning:
            self.service.manager.publish_session(
                self.session_id,
                {"type": "warning", "sessionId": self.session_id, "message": warning},
            )
        return True, None

    def handle_inference_result(self, result: InferenceResult):
        # Recorder-backed sessions consume scheduler results through
        # transcribe_for_recorder(); direct event routing is only used by the
        # older inline session tests.
        self.service.complete_pending_recorder_transcription(result)

    def on_job_dropped(self, job: InferenceJob, reason: str):
        if reason == "coalesced" and job.kind == "realtime":
            self.coalesced_realtime += 1
        elif reason == "stale" and job.kind == "realtime":
            self.stale_realtime_discarded += 1
        elif reason == "cancelled":
            self.cancelled_jobs += 1
        self.service.fail_pending_recorder_transcription(
            job.request_id,
            f"{job.kind} transcription was {reason}",
        )

    def on_submit_result(self, job: InferenceJob, result: QueueSubmitResult):
        if result.accepted:
            if job.kind == "realtime":
                self.realtime_submitted += 1
                if result.coalesced:
                    self.coalesced_realtime += 1
            else:
                self.final_submitted += 1
            return

        if job.kind == "realtime":
            self.realtime_rejected += 1
        else:
            self.final_rejected += 1
        self.service.fail_pending_recorder_transcription(job.request_id, result.reason)

    def record_executor_result(self, result: InferenceResult):
        self.queue_delay[result.kind].record(result.queue_delay)
        self.inference_duration[result.kind].record(result.inference_duration)
        self.total_latency[result.kind].record(result.total_latency)
        if result.kind == "realtime":
            self.realtime_completed += 1
        else:
            self.final_completed += 1

    def publish_status(self, state=None):
        with self.lock:
            state = state or self.status
            self.status = state
            message = {
                "type": "status",
                "sessionId": self.session_id,
                "state": state,
                "timestamp": time.time(),
                "activeClientId": self.session_id if self.streaming else None,
                "queueDepth": self._recorder_queue_depth(),
                "droppedChunks": self.dropped_audio_chunks,
                "coalescedRealtime": self.coalesced_realtime,
                "staleRealtimeDiscarded": self.stale_realtime_discarded,
                "activeSessions": self.service.session_count(),
                "activeSpeakers": self.service.active_speaker_count(),
                "wakeWordEnabled": self.settings.wake_word_enabled(),
                "wakeWord": {
                    "enabled": self.settings.wake_word_enabled(),
                    "backend": self.settings.wakeword_backend,
                    "wakeWords": self.settings.wake_words,
                    "state": state if str(state).startswith("wakeword") else None,
                },
            }
            message["timestampIso"] = timestamp_iso(message["timestamp"])
        self.service.manager.publish_session(self.session_id, message)

    def snapshot(self):
        with self.lock:
            state = self.status
            streaming = self.streaming
            recording = bool(getattr(self.recorder, "is_recording", False))
        return {
            "sessionId": self.session_id,
            "streaming": streaming,
            "recording": recording,
            "state": state,
            "wakeWordEnabled": self.settings.wake_word_enabled(),
            "currentSegmentId": self.segment_state.current(),
            "currentSegment": self.timeline.snapshot(self.segment_state.current()),
            "queueDepth": self._recorder_queue_depth(),
            "recordingSeconds": self.recording_sample_count / float(SERVER_SAMPLE_RATE),
            "droppedAudioChunks": self.dropped_audio_chunks,
            "rejectedAudioChunks": self.rejected_audio_chunks,
            "coalescedRealtime": self.coalesced_realtime,
            "staleRealtimeDiscarded": self.stale_realtime_discarded,
            "cancelledJobs": self.cancelled_jobs,
            "realtimeSubmitted": self.realtime_submitted,
            "finalSubmitted": self.final_submitted,
            "realtimeCompleted": self.realtime_completed,
            "finalCompleted": self.final_completed,
            "realtimeRejected": self.realtime_rejected,
            "finalRejected": self.final_rejected,
            "forcedFinalizations": self.forced_finalizations,
            "droppedRecordedSegments": self.dropped_recorded_segments,
            "queueDelay": {
                "realtime": self.queue_delay["realtime"].snapshot_ms(),
                "final": self.queue_delay["final"].snapshot_ms(),
            },
            "inferenceDuration": {
                "realtime": self.inference_duration["realtime"].snapshot_ms(),
                "final": self.inference_duration["final"].snapshot_ms(),
            },
            "totalLatency": {
                "realtime": self.total_latency["realtime"].snapshot_ms(),
                "final": self.total_latency["final"].snapshot_ms(),
            },
        }

    def _text_worker(self):
        while not self.service.stop_event.is_set():
            with self.lock:
                text_generation = self.generation
            try:
                text = self.recorder.text()
            except Exception as exc:
                if getattr(self.recorder, "is_shut_down", False):
                    break
                LOGGER.exception("Session recorder text loop failed")
                self.service.manager.publish_session(
                    self.session_id,
                    {
                        "type": "error",
                        "sessionId": self.session_id,
                        "message": str(exc),
                        "where": "recorder",
                    },
                )
                time.sleep(0.1)
                continue

            if getattr(self.recorder, "is_shut_down", False):
                break
            text = (text or "").strip()
            if not text:
                continue
            self._publish_final_text(text, text_generation)

    def _publish_final_text(self, text, text_generation):
        with self.lock:
            if text_generation != self.generation:
                return False
            segment_id = self.segment_state.final()
            streaming = self.streaming
            segment = self._timeline_snapshot(segment_id)
        timestamp = time.time()
        payload = {
            "type": "final",
            "sessionId": self.session_id,
            "segmentId": segment_id,
            "text": text,
            "timestamp": timestamp,
            "timestampIso": timestamp_iso(timestamp),
        }
        if segment is not None:
            payload["segment"] = segment
            payload.update(segment_text_fields(segment))
        self.service.manager.publish_session(
            self.session_id,
            payload,
        )
        self._publish_timeline_event(
            "final_transcript",
            timestamp=timestamp,
            segment_id=segment_id,
            segment=segment,
            text=text,
        )
        self.publish_status(self._waiting_state_locked(streaming))
        return True

    def _on_realtime_text(self, text):
        with self.lock:
            if self.reject_current_recording:
                return
            segment_id = self.segment_state.realtime()
            segment = self._timeline_snapshot(segment_id)
        text = (text or "").strip()
        if not text:
            return
        timestamp = time.time()
        payload = {
            "type": "realtime",
            "sessionId": self.session_id,
            "segmentId": segment_id,
            "text": text,
            "timestamp": timestamp,
            "timestampIso": timestamp_iso(timestamp),
        }
        if segment is not None:
            payload["segment"] = segment
            payload.update(segment_text_fields(segment))
        self.service.manager.publish_session(
            self.session_id,
            payload,
        )
        self._publish_timeline_event(
            "realtime_transcript",
            timestamp=timestamp,
            segment_id=segment_id,
            segment=segment,
            text=text,
        )

    def _on_realtime_stabilization_event(self, event):
        with self.lock:
            if self.reject_current_recording:
                return

        raw_text = (getattr(event, "raw_observation_text", "") or "").strip()
        committed_stable_text = getattr(event, "stable_text", "") or ""
        unstable_text = getattr(event, "unstable_text", "") or ""
        display_text = (getattr(event, "display_text", "") or "").strip()
        consensus_text = getattr(event, "consensus_text", "") or committed_stable_text
        consensus_unstable_text = getattr(event, "consensus_unstable_text", "") or ""
        consensus_display_text = (
            getattr(event, "consensus_display_text", "") or display_text
        ).strip()
        if (
            not raw_text
            and not display_text
            and not committed_stable_text
            and not unstable_text
        ):
            return

        segment_id = getattr(event, "segment_id", None)
        if segment_id is None:
            segment_id = self.segment_state.realtime()

        text = (
            display_text
            if self.settings.realtime_callback == "stabilized"
            else raw_text or display_text
        )
        timing = getattr(event, "timing", None)
        timestamp = time.time()
        segment = self._timeline_snapshot(segment_id)
        payload = {
            "type": "realtime",
            "sessionId": self.session_id,
            "segmentId": segment_id,
            "recordingId": getattr(event, "recording_id", None),
            "sequence": getattr(event, "sequence", None),
            "text": text,
            "rawText": raw_text,
            "displayText": display_text or raw_text,
            "stableText": committed_stable_text,
            "stableDelta": getattr(event, "stable_delta", "") or "",
            "unstableText": unstable_text,
            "committedStableText": committed_stable_text,
            "committedStableDelta": getattr(event, "stable_delta", "") or "",
            "visualStableText": committed_stable_text,
            "visualUnstableText": unstable_text,
            "consensusText": consensus_text,
            "consensusUnstableText": consensus_unstable_text,
            "consensusDisplayText": consensus_display_text,
            "publicConsensusAligned": bool(
                getattr(event, "public_consensus_aligned", True)
            ),
            "internalRevision": bool(getattr(event, "internal_revision", False)),
            "isOutlier": bool(getattr(event, "is_outlier", False)),
            "stablePrefixConflict": bool(
                getattr(event, "stable_prefix_conflict", False)
            ),
            "commitReason": getattr(event, "commit_reason", None),
            "stableNormalizedOffset": getattr(
                event,
                "stable_normalized_offset",
                None,
            ),
            "timestamp": timestamp,
            "timestampIso": timestamp_iso(timestamp),
        }
        if timing is not None:
            payload["timing"] = asdict(timing)
        if segment is not None:
            payload["segment"] = segment
            payload.update(segment_text_fields(segment))

        self.service.manager.publish_session(self.session_id, payload)
        self._publish_timeline_event(
            "realtime_transcript",
            timestamp=timestamp,
            segment_id=segment_id,
            segment=segment,
            text=text,
            sequence=payload.get("sequence"),
        )

    def _on_recording_start(self):
        segment = None
        segment_id = None
        with self.lock:
            self._wakeword_followup_generation += 1
            self._clear_recorder_followup_gate_locked()
            if not self.service.try_activate_speaker(self.session_id):
                self.reject_current_recording = True
                self.recording_sample_count = 0
                self._force_finalize_in_progress = False
                self._wakeword_voice_window = False
                self.rejected_audio_chunks += 1
                self.service.manager.publish_session(
                    self.session_id,
                    {
                        "type": "warning",
                        "sessionId": self.session_id,
                        "message": "Server active speaker limit reached; recording will be ignored.",
                    },
                )
            else:
                self.reject_current_recording = False
                self.recording_sample_count = 0
                self._force_finalize_in_progress = False
                self._wakeword_voice_window = False
                segment_id = self.segment_state.current()
                segment = self.timeline.mark_recording_started(segment_id)
        if segment is not None:
            self._publish_timeline_event(
                "recording_started",
                timestamp=segment.get("recordingStartedAt"),
                segment_id=segment_id,
                segment=segment,
                preRecordingBuffer=segment.get("preRecordingBuffer"),
            )
        self.publish_status("recording")

    def _on_recording_stop(self):
        self._trim_recorded_audio_queue()
        segment = None
        segment_id = None
        with self.lock:
            segment_id = self.segment_state.current()
            duration_seconds = (
                self.recording_sample_count / float(SERVER_SAMPLE_RATE)
                if self.recording_sample_count
                else None
            )
            segment = self.timeline.mark_recording_ended(
                "recording_stop",
                segment_id=segment_id,
                actual_duration_seconds=duration_seconds,
            )
            self.recording_sample_count = 0
            self._force_finalize_in_progress = False
            self._wakeword_voice_window = False
        self.service.deactivate_speaker(self.session_id)
        if segment is not None:
            self._publish_timeline_event(
                "recording_ended",
                timestamp=segment.get("recordingEndedAt"),
                segment_id=segment_id,
                segment=segment,
                durationSeconds=segment.get("durationSeconds"),
                reason=segment.get("endReason"),
            )
        self._start_wakeword_followup_window()
        self.publish_status(self._waiting_state_locked())

    def _on_transcription_start(self, *_):
        segment_id = self.segment_state.current()
        self._publish_timeline_event(
            "transcription_started",
            segment_id=segment_id,
            segment=self._timeline_snapshot(segment_id),
        )
        self.publish_status("transcribing")
        with self.lock:
            return True if self.reject_current_recording else False

    def _waiting_state_locked(self, streaming=None):
        if streaming is None:
            streaming = self.streaming
        if not streaming:
            return "idle"
        if self.settings.wake_word_enabled():
            if self._wakeword_voice_window:
                return "wakeword_detected"
            return "wakeword_wait"
        return "listening"

    def _start_wakeword_followup_window(self):
        try:
            window = max(0.0, float(self.settings.wake_word_followup_window))
        except (TypeError, ValueError):
            window = 0.0
        if not self.settings.wake_word_enabled() or window <= 0:
            return False

        with self.lock:
            if not self.streaming or self.reject_current_recording:
                return False
            self._wakeword_voice_window = True
            self._wakeword_followup_generation += 1
            generation = self._wakeword_followup_generation
            recorder = self.recorder
            try:
                if self._recorder_wake_word_timeout_before_followup is None:
                    self._recorder_wake_word_timeout_before_followup = getattr(
                        recorder,
                        "wake_word_timeout",
                        None,
                    )
                if self._recorder_start_recording_before_followup is None:
                    self._recorder_start_recording_before_followup = getattr(
                        recorder,
                        "start_recording_on_voice_activity",
                        None,
                    )
                if self._recorder_stop_recording_before_followup is None:
                    self._recorder_stop_recording_before_followup = getattr(
                        recorder,
                        "stop_recording_on_voice_deactivity",
                        None,
                    )
                recorder.wakeword_detected = True
                recorder.wake_word_detect_time = time.time()
                recorder.wake_word_timeout = window
                recorder.start_recording_on_voice_activity = True
                recorder.stop_recording_on_voice_deactivity = True
            except Exception:
                self._wakeword_voice_window = False
                self._clear_recorder_followup_gate_locked()
                LOGGER.debug(
                    "Could not arm wake-word follow-up window for %s",
                    self.session_id,
                    exc_info=True,
                )
                return False

        self._publish_timeline_event(
            "wakeword_followup_started",
            durationSeconds=window,
        )
        threading.Thread(
            target=self._wakeword_followup_timeout_worker,
            args=(generation, window),
            name=f"RealtimeSTTSessionWakeFollowup-{self.session_id}",
            daemon=True,
        ).start()
        return True

    def _wakeword_followup_timeout_worker(self, generation, window):
        time.sleep(window)
        self._finish_wakeword_followup(generation)

    def _finish_wakeword_followup(self, generation=None):
        with self.lock:
            if generation is not None and generation != self._wakeword_followup_generation:
                return False
            if not self._wakeword_voice_window:
                return False
            if bool(getattr(self.recorder, "is_recording", False)):
                return False
            self._wakeword_voice_window = False
            self._wakeword_followup_generation += 1
            self._clear_recorder_followup_gate_locked()
            streaming = self.streaming

        self._publish_timeline_event("wakeword_followup_timeout")
        self.publish_status("wakeword_wait" if streaming else "idle")
        return True

    def _clear_recorder_followup_gate_locked(self):
        recorder = self.recorder
        try:
            recorder.wakeword_detected = False
            recorder.wake_word_detect_time = 0
            if self._recorder_wake_word_timeout_before_followup is not None:
                recorder.wake_word_timeout = self._recorder_wake_word_timeout_before_followup
            if self._recorder_start_recording_before_followup is not None:
                recorder.start_recording_on_voice_activity = (
                    self._recorder_start_recording_before_followup
                )
            if self._recorder_stop_recording_before_followup is not None:
                recorder.stop_recording_on_voice_deactivity = (
                    self._recorder_stop_recording_before_followup
                )
        except Exception:
            LOGGER.debug(
                "Could not clear wake-word follow-up gate for %s",
                self.session_id,
                exc_info=True,
            )
        self._recorder_wake_word_timeout_before_followup = None
        self._recorder_start_recording_before_followup = None
        self._recorder_stop_recording_before_followup = None

    def _on_vad_start(self):
        self.publish_status(self._voice_or_waiting_state())

    def _on_vad_stop(self):
        self.publish_status(self._silence_or_waiting_state())

    def _on_vad_detect_start(self):
        self.publish_status(self._voice_or_waiting_state())

    def _on_vad_detect_stop(self):
        self.publish_status(self._silence_or_waiting_state())

    def _voice_or_waiting_state(self):
        with self.lock:
            if not self.settings.wake_word_enabled():
                return "voice"
            if self._wakeword_voice_window:
                return "voice"
            return self._waiting_state_locked()

    def _silence_or_waiting_state(self):
        with self.lock:
            if not self.settings.wake_word_enabled():
                return "silence"
            if self._wakeword_voice_window:
                return "silence"
            return self._waiting_state_locked()

    def _on_wakeword_detection_start(self):
        with self.lock:
            self._wakeword_voice_window = False
            self._wakeword_followup_generation += 1
            self._clear_recorder_followup_gate_locked()
        event = self.timeline.mark_wakeword_wait_started()
        self._publish_timeline_event(
            "wakeword_wait_started",
            wakeWord=event.get("wakeWord"),
        )
        self.publish_status("wakeword_wait")

    def _on_wakeword_detection_end(self):
        event = self.timeline.mark_wakeword_wait_ended()
        self._publish_timeline_event(
            "wakeword_wait_ended",
            wakeWord=event.get("wakeWord"),
        )

    def _on_wakeword_detected(self):
        with self.lock:
            self._wakeword_voice_window = True
            self._wakeword_followup_generation += 1
        event = self.timeline.mark_wakeword_detected()
        self._publish_timeline_event(
            "wakeword_detected",
            wakeWord=event.get("wakeWord"),
        )
        self.publish_status("wakeword_detected")

    def _on_wakeword_timeout(self):
        with self.lock:
            self._wakeword_voice_window = False
            self._wakeword_followup_generation += 1
            self._clear_recorder_followup_gate_locked()
        event = self.timeline.mark_wakeword_timeout()
        self._publish_timeline_event(
            "wakeword_timeout",
            wakeWord=event.get("wakeWord"),
        )
        self.publish_status("wakeword_timeout")

    def _publish_timeline_event(
        self,
        event,
        *,
        timestamp=None,
        segment_id=None,
        segment=None,
        **fields,
    ):
        if not hasattr(self, "timeline"):
            return
        timestamp = time.time() if timestamp is None else float(timestamp)
        payload = {
            "type": "timeline",
            "sessionId": self.session_id,
            "event": event,
            "timestamp": timestamp,
            "timestampIso": timestamp_iso(timestamp),
        }
        if segment_id is not None:
            payload["segmentId"] = segment_id
        if segment is not None:
            payload["segment"] = segment
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        self.service.manager.publish_session(self.session_id, payload)

    def _timeline_snapshot(self, segment_id=None):
        timeline = getattr(self, "timeline", None)
        if timeline is None:
            return None
        return timeline.snapshot(segment_id)

    def _recorder_queue_depth(self):
        depth = 0
        try:
            depth += int(self.recorder.audio_queue.qsize())
        except Exception:
            pass
        try:
            depth += int(self.recorder.recorded_audio_queue.qsize())
        except Exception:
            pass
        return depth

    def _enforce_recording_duration(self, samples):
        if self._recorded_chunk_callback_seen:
            return None

        max_samples = int(self.settings.max_audio_queue_seconds_per_session * SERVER_SAMPLE_RATE)
        if max_samples <= 0:
            return None

        should_finalize = False
        with self.lock:
            if bool(getattr(self.recorder, "is_recording", False)):
                self.recording_sample_count += int(samples.size)
                should_finalize = self.recording_sample_count >= max_samples

        if not should_finalize:
            return None

        self._force_finalize_after_limit()
        return None

    def _on_recorded_chunk(self, data):
        self._recorded_chunk_callback_seen = True
        max_samples = int(self.settings.max_audio_queue_seconds_per_session * SERVER_SAMPLE_RATE)
        if max_samples <= 0:
            return
        if not bool(getattr(self.recorder, "is_recording", False)):
            return

        try:
            sample_count = len(data) // 2
        except Exception:
            return

        should_finalize = False
        with self.lock:
            self.recording_sample_count += int(sample_count)
            if (
                self.recording_sample_count >= max_samples
                and not self._force_finalize_in_progress
            ):
                self._force_finalize_in_progress = True
                should_finalize = True

        if should_finalize:
            threading.Thread(
                target=self._force_finalize_after_limit,
                name=f"RealtimeSTTSessionForceFinalize-{self.session_id}",
                daemon=True,
            ).start()

    def _force_finalize_after_limit(self):
        finalized = False
        try:
            finalized = bool(self.recorder.flush_buffered_audio())
            self._trim_recorded_audio_queue()
        except Exception:
            LOGGER.debug("Could not force-finalize long recording for %s", self.session_id, exc_info=True)
        finally:
            with self.lock:
                self.recording_sample_count = 0
                self._force_finalize_in_progress = False
                if finalized:
                    self.forced_finalizations += 1

        if not finalized:
            return
        self.service.deactivate_speaker(self.session_id)
        self.service.manager.publish_session(
            self.session_id,
            {
                "type": "warning",
                "sessionId": self.session_id,
                "message": "Maximum per-session audio buffer reached; finalized the current segment.",
            },
        )

    def _trim_recorded_audio_queue(self):
        queue_obj = getattr(self.recorder, "recorded_audio_queue", None)
        if queue_obj is None:
            return 0

        max_pending = max(0, int(self.settings.max_final_queue_depth_per_session))
        dropped = 0
        while True:
            try:
                if queue_obj.qsize() <= max_pending:
                    break
                queue_obj.get_nowait()
                dropped += 1
            except Exception:
                break

        if dropped:
            with self.lock:
                self.dropped_recorded_segments += dropped
                self.final_rejected += dropped
            self.service.manager.publish_session(
                self.session_id,
                {
                    "type": "warning",
                    "sessionId": self.session_id,
                    "message": (
                        "Final transcription backlog exceeded the per-session limit; "
                        f"dropped {dropped} queued recorded segment(s)."
                    ),
                },
            )
        return dropped


class SessionStore:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._sessions: Dict[str, RealtimeSession] = {}
        self._reserved_session_ids = set()
        self._active_speakers = set()
        self.rejected_sessions = 0

    def reserve(self, session_id):
        with self._lock:
            if session_id in self._sessions or session_id in self._reserved_session_ids:
                self.rejected_sessions += 1
                return False
            if self._session_slots_used_locked() >= self.settings.max_sessions:
                self.rejected_sessions += 1
                return False
            self._reserved_session_ids.add(session_id)
            return True

    def add(self, session):
        with self._lock:
            reserved = session.session_id in self._reserved_session_ids
            if session.session_id in self._sessions:
                self._reserved_session_ids.discard(session.session_id)
                self.rejected_sessions += 1
                return False
            if not reserved and self._session_slots_used_locked() >= self.settings.max_sessions:
                self.rejected_sessions += 1
                return False
            self._reserved_session_ids.discard(session.session_id)
            self._sessions[session.session_id] = session
            return True

    def can_accept(self):
        with self._lock:
            if self._session_slots_used_locked() >= self.settings.max_sessions:
                self.rejected_sessions += 1
                return False
            return True

    def release_reservation(self, session_id):
        with self._lock:
            self._reserved_session_ids.discard(session_id)

    def remove(self, session_id):
        with self._lock:
            session = self._sessions.pop(session_id, None)
            self._reserved_session_ids.discard(session_id)
            self._active_speakers.discard(session_id)
            return session

    def remove_all(self):
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._reserved_session_ids.clear()
            self._active_speakers.clear()
            return sessions

    def get(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)

    def try_activate_speaker(self, session_id):
        with self._lock:
            if session_id in self._active_speakers:
                return True
            if len(self._active_speakers) >= self.settings.max_active_speakers:
                return False
            self._active_speakers.add(session_id)
            return True

    def deactivate_speaker(self, session_id):
        with self._lock:
            self._active_speakers.discard(session_id)

    def count(self):
        with self._lock:
            return len(self._sessions)

    def active_speaker_count(self):
        with self._lock:
            return len(self._active_speakers)

    def snapshots(self):
        with self._lock:
            sessions = list(self._sessions.values())
            rejected = self.rejected_sessions
            active_speakers = len(self._active_speakers)
            reserved_sessions = len(self._reserved_session_ids)
        return {
            "activeSessions": len(sessions),
            "activeSpeakers": active_speakers,
            "pendingSessionAdmissions": reserved_sessions,
            "rejectedSessions": rejected,
            "sessions": {session.session_id: session.snapshot() for session in sessions},
        }

    def _session_slots_used_locked(self):
        return len(self._sessions) + len(self._reserved_session_ids)


class RealtimeSTTService:
    def __init__(
        self,
        settings: ServerSettings,
        manager: ConnectionManager,
        scheduler_factory: Optional[Callable[..., Any]] = None,
        recorder_factory: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings
        self.manager = manager
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.sessions = SessionStore(settings)
        self.startup_errors = []
        self._pending_recorder_results = {}
        self._pending_recorder_lock = threading.Lock()
        self.recorder_factory = recorder_factory
        factory = scheduler_factory or InferenceScheduler
        self.scheduler = factory(
            settings,
            self._on_inference_result,
            self._on_scheduler_drop,
            self._on_scheduler_error,
        )
        self.ready_thread = None

    def start(self, loop):
        self.manager.bind_loop(loop)
        self.scheduler.start()
        self.ready_thread = threading.Thread(
            target=self._ready_worker,
            name="RealtimeSTTServerReady",
            daemon=True,
        )
        self.ready_thread.start()

    def stop(self):
        self.stop_event.set()
        for session in self.sessions.remove_all():
            session.close()
        self.scheduler.stop()
        if self.ready_thread is not None:
            self.ready_thread.join(timeout=5)

    def admit_session(self, session_id):
        if not self.sessions.reserve(session_id):
            return None
        session = None
        try:
            session = RecorderBackedRealtimeSession(self, session_id)
            if not self.sessions.add(session):
                session.close()
                return None
            return session
        except Exception:
            self.sessions.release_reservation(session_id)
            if session is not None:
                session.close()
            raise

    def remove_session(self, session_id):
        session = self.sessions.remove(session_id)
        if session is not None:
            session.close()

    def submit_inference_job(self, job: InferenceJob):
        result = self.scheduler.submit(job)
        session = self.sessions.get(job.session_id)
        if session is not None:
            session.on_submit_result(job, result)
        return result

    def try_activate_speaker(self, session_id):
        return self.sessions.try_activate_speaker(session_id)

    def deactivate_speaker(self, session_id):
        self.sessions.deactivate_speaker(session_id)

    def session_count(self):
        return self.sessions.count()

    def active_speaker_count(self):
        return self.sessions.active_speaker_count()

    def packet_to_server_samples(self, packet):
        if len(packet.audio) > self.settings.max_audio_packet_bytes:
            raise AudioPacketError("audio packet is too large")

        sample_rate = require_positive_int(packet.metadata, "sampleRate")
        channels = packet.metadata.get("channels", 1)
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise AudioPacketError("audio packet metadata field 'channels' must be a positive integer")
        if channels > 8:
            raise AudioPacketError("audio packet metadata field 'channels' must be at most 8")
        audio_format = packet.metadata.get("format", "pcm_s16le")
        if audio_format != "pcm_s16le":
            raise AudioPacketError("only pcm_s16le audio packets are supported")
        frame_width = channels * 2
        if len(packet.audio) % frame_width:
            raise AudioPacketError("pcm_s16le audio packet is not aligned to whole frames")
        if "frames" in packet.metadata:
            expected_frames = require_positive_int(packet.metadata, "frames")
            expected_bytes = expected_frames * frame_width
            if len(packet.audio) != expected_bytes:
                raise AudioPacketError(
                    "audio packet metadata field 'frames' does not match payload length"
                )

        samples = np.frombuffer(packet.audio, dtype=np.int16)
        if channels > 1:
            usable = len(samples) - (len(samples) % channels)
            if usable <= 0:
                return np.array([], dtype=np.int16)
            samples = samples[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)
        return resample_int16(samples, sample_rate, SERVER_SAMPLE_RATE)

    def metrics(self):
        data = self.sessions.snapshots()
        data["ready"] = self.ready.is_set()
        data["ok"] = self.ready.is_set() and self.scheduler.healthy()
        data["scheduler"] = self.scheduler.snapshot()
        data["limits"] = self.limits_dict()
        data["startupErrors"] = list(self.startup_errors)
        return data

    def limits_dict(self):
        return {
            "maxSessions": self.settings.max_sessions,
            "maxActiveSpeakers": self.settings.max_active_speakers,
            "maxAudioQueueSecondsPerSession": self.settings.max_audio_queue_seconds_per_session,
            "maxRealtimeQueueAgeMs": self.settings.max_realtime_queue_age_ms,
            "maxFinalQueueDepthPerSession": self.settings.max_final_queue_depth_per_session,
            "maxGlobalInferenceQueueDepth": self.settings.max_global_inference_queue_depth,
            "realtimeDegradationThresholdMs": self.settings.realtime_degradation_threshold_ms,
        }

    def runtime_settings_contract(self):
        return runtime_settings_contract()

    def update_settings(self, updates):
        applied = {}
        rejected = {}
        if not isinstance(updates, dict):
            raise ValueError("settings update must be a JSON object")

        for name, value in updates.items():
            if name in STARTUP_ONLY_SETTINGS:
                rejected[name] = {
                    "reason": "startup_only",
                    "message": "This setting requires a server restart because shared resources are already initialized.",
                }
                continue
            if name not in ACTIVE_RUNTIME_SETTINGS and name not in NEW_SESSION_RUNTIME_SETTINGS:
                rejected[name] = {
                    "reason": "unknown",
                    "message": "Unknown or unsupported server setting.",
                }
                continue
            try:
                coerced = coerce_setting_value(name, value)
            except ValueError as exc:
                rejected[name] = {
                    "reason": "invalid_value",
                    "message": str(exc),
                }
                continue
            setattr(self.settings, name, coerced)
            applied[name] = {
                "value": coerced,
                "appliesTo": (
                    "active_sessions"
                    if name in ACTIVE_RUNTIME_SETTINGS
                    else "new_sessions"
                ),
            }

        return {
            "applied": applied,
            "rejected": rejected,
            "settings": self.settings.public_dict(),
            "runtimeSettings": self.runtime_settings_contract(),
        }

    def transcribe_for_recorder(self, session_id, kind, audio, language, use_prompt):
        from RealtimeSTT.transcription_engines import TranscriptionResult

        session = self.sessions.get(session_id)
        if session is None:
            return TranscriptionResult(text="")

        generation = getattr(session, "generation", 0)
        request_id = uuid.uuid4().hex
        holder = {
            "event": threading.Event(),
            "result": None,
            "error": None,
            "sessionId": session_id,
            "generation": generation,
        }
        with self._pending_recorder_lock:
            self._pending_recorder_results[request_id] = holder

        job = InferenceJob(
            request_id=request_id,
            session_id=session_id,
            kind=kind,
            audio=audio,
            language=language,
            use_prompt=use_prompt,
            segment_id=session.segment_state.current(),
            sequence=0,
            generation=generation,
            created_at=time.monotonic(),
            deadline_at=(
                time.monotonic() + (self.settings.max_realtime_queue_age_ms / 1000.0)
                if kind == "realtime"
                else None
            ),
        )

        submit_result = self.submit_inference_job(job)
        if not submit_result.accepted:
            self._pop_pending_recorder_result(request_id)
            raise RuntimeError(submit_result.reason)

        while not holder["event"].wait(timeout=0.1):
            current_session = self.sessions.get(session_id)
            if (
                self.stop_event.is_set()
                or current_session is None
                or getattr(current_session, "generation", generation) != generation
            ):
                self._pop_pending_recorder_result(request_id)
                return TranscriptionResult(text="")

        self._pop_pending_recorder_result(request_id)
        current_session = self.sessions.get(session_id)
        if current_session is None or getattr(current_session, "generation", generation) != generation:
            return TranscriptionResult(text="")

        if holder["error"]:
            raise RuntimeError(holder["error"])

        result = holder["result"]
        if result is None:
            return TranscriptionResult(text="")
        if result.error:
            raise RuntimeError(result.error)

        current_session.record_executor_result(result)
        return TranscriptionResult(text=result.text)

    def complete_pending_recorder_transcription(self, result: InferenceResult):
        with self._pending_recorder_lock:
            holder = self._pending_recorder_results.get(result.request_id)
        if holder is None:
            return False
        holder["result"] = result
        holder["event"].set()
        return True

    def fail_pending_recorder_transcription(self, request_id, error):
        with self._pending_recorder_lock:
            holder = self._pending_recorder_results.get(request_id)
        if holder is None:
            return False
        holder["error"] = error
        holder["event"].set()
        return True

    def cancel_pending_recorder_transcriptions(self, session_id):
        with self._pending_recorder_lock:
            pending = [
                (request_id, holder)
                for request_id, holder in self._pending_recorder_results.items()
                if holder["sessionId"] == session_id
            ]
        for request_id, holder in pending:
            holder["error"] = "session was cancelled"
            holder["event"].set()
            self._pop_pending_recorder_result(request_id)

    def _pop_pending_recorder_result(self, request_id):
        with self._pending_recorder_lock:
            return self._pending_recorder_results.pop(request_id, None)

    def _ready_worker(self):
        self.scheduler.wait_ready()
        self.ready.set()
        ready_message = {
            "type": "ready",
            "settings": self.settings.public_dict(),
            "limits": self.limits_dict(),
            "runtimeSettings": self.runtime_settings_contract(),
            "ok": self.scheduler.healthy(),
        }
        self.manager.publish_all(ready_message)
        if self.startup_errors:
            for error in self.startup_errors:
                self.manager.publish_all(error)

    def _on_inference_result(self, result: InferenceResult):
        if self.complete_pending_recorder_transcription(result):
            return
        session = self.sessions.get(result.session_id)
        if session is not None:
            session.handle_inference_result(result)

    def _on_scheduler_drop(self, job: InferenceJob, reason: str, lane: str):
        session = self.sessions.get(job.session_id)
        if session is not None:
            session.on_job_dropped(job, reason)
        else:
            self.fail_pending_recorder_transcription(
                job.request_id,
                f"{job.kind} transcription was {reason}",
            )

    def _on_scheduler_error(self, lane, exc):
        message = {
            "type": "error",
            "message": str(exc),
            "where": f"{lane}_engine",
        }
        self.startup_errors.append(message)
        self.manager.publish_all(message)


@dataclass(frozen=True)
class AudioData:
    samples: Any
    sample_rate: int


def read_wav_float32(path: Path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        usable = len(samples) - (len(samples) % channels)
        samples = samples[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)
    samples = resample_int16(samples, sample_rate, SERVER_SAMPLE_RATE)
    return AudioData(samples=samples.astype(np.float32) / INT16_MAX_ABS_VALUE, sample_rate=SERVER_SAMPLE_RATE)


def resample_int16(samples, source_rate, target_rate):
    samples = np.asarray(samples, dtype=np.int16)
    if source_rate == target_rate or samples.size == 0:
        return samples.copy()

    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), int(target_rate))
        up = int(target_rate // divisor)
        down = int(source_rate // divisor)
        resampled = resample_poly(samples.astype(np.float32), up, down)
    except Exception:
        duration = samples.size / float(source_rate)
        target_size = max(1, int(round(duration * target_rate)))
        source_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        resampled = np.interp(target_positions, source_positions, samples.astype(np.float32))

    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def effective_device(device):
    if str(device).lower() != "cuda":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_fastapi():
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI server dependencies are missing. Install them with "
            "'python -m pip install -r example_fastapi_server/requirements.txt'."
        ) from exc
    return FastAPI, WebSocket, WebSocketDisconnect, HTMLResponse, JSONResponse


def create_app(settings: Optional[ServerSettings] = None, scheduler_factory=None, recorder_factory=None):
    FastAPI, WebSocket, WebSocketDisconnect, HTMLResponse, JSONResponse = load_fastapi()
    from contextlib import asynccontextmanager
    from RealtimeSTT.transcription_engines import get_supported_transcription_engines

    settings = settings or ServerSettings()
    manager = ConnectionManager()
    service = RealtimeSTTService(
        settings,
        manager,
        scheduler_factory=scheduler_factory,
        recorder_factory=recorder_factory,
    )

    @asynccontextmanager
    async def lifespan(app):
        service.start(asyncio.get_running_loop())
        yield
        service.stop()

    app = FastAPI(
        title="RealtimeSTT FastAPI Server",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def index():
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        metrics = service.metrics()
        return JSONResponse({
            "ok": metrics["ok"],
            "ready": metrics["ready"],
            "activeSessions": metrics["activeSessions"],
            "activeSpeakers": metrics["activeSpeakers"],
            "rejectedSessions": metrics["rejectedSessions"],
            "scheduler": metrics["scheduler"],
            "startupErrors": metrics["startupErrors"],
        })

    @app.get("/api/config")
    async def config():
        return JSONResponse({
            "settings": settings.public_dict(),
            "limits": service.limits_dict(),
            "supportedEngines": get_supported_transcription_engines(),
            "runtimeSettings": service.runtime_settings_contract(),
        })

    @app.patch("/api/config")
    async def update_config(payload: dict):
        updates = payload.get("settings", payload)
        try:
            result = service.update_settings(updates)
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=400,
            )
        return JSONResponse(
            result,
            status_code=400 if result["rejected"] else 200,
        )

    @app.get("/api/metrics")
    async def metrics():
        return JSONResponse(service.metrics())

    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        session_id = uuid.uuid4().hex
        session = service.admit_session(session_id)
        if session is None:
            await websocket.accept()
            await websocket.send_text(json.dumps({
                "type": "error",
                "where": "admission",
                "message": "Server is at the configured session limit.",
                "limits": service.limits_dict(),
            }))
            await websocket.close(code=1013)
            return

        await manager.connect(session_id, websocket)
        await websocket.send_text(json.dumps({
            "type": "hello",
            "clientId": session_id,
            "sessionId": session_id,
            "settings": settings.public_dict(),
            "limits": service.limits_dict(),
            "supportedEngines": get_supported_transcription_engines(),
            "runtimeSettings": service.runtime_settings_contract(),
        }))
        if service.ready.is_set():
            await websocket.send_text(json.dumps({
                "type": "ready",
                "sessionId": session_id,
                "settings": settings.public_dict(),
                "limits": service.limits_dict(),
                "runtimeSettings": service.runtime_settings_contract(),
                "ok": service.scheduler.healthy(),
            }))
            for error in service.startup_errors:
                await websocket.send_text(json.dumps(error))

        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"] is not None:
                    try:
                        accepted, warning = session.ingest_audio_packet(
                            decode_audio_packet(message["bytes"])
                        )
                    except AudioPacketError as exc:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "sessionId": session_id,
                            "message": str(exc),
                            "where": "audio_packet",
                        }))
                        continue
                    except Exception as exc:
                        LOGGER.exception("Could not ingest audio packet")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "sessionId": session_id,
                            "message": str(exc),
                            "where": "audio",
                        }))
                        continue
                    if not accepted:
                        await websocket.send_text(json.dumps({
                            "type": "warning",
                            "sessionId": session_id,
                            "message": warning or "Audio chunk was rejected.",
                        }))
                elif "text" in message and message["text"] is not None:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError as exc:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "sessionId": session_id,
                            "message": f"Invalid command JSON: {exc.msg}",
                            "where": "command",
                        }))
                        continue

                    if not isinstance(data, dict):
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "sessionId": session_id,
                            "message": "WebSocket commands must be JSON objects.",
                            "where": "command",
                        }))
                        continue

                    command = data.get("type")
                    if command == "start":
                        session.start_streaming()
                    elif command == "stop":
                        session.stop_streaming()
                    elif command == "clear":
                        session.clear()
                    elif command == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "sessionId": session_id,
                            "serverTime": time.time(),
                        }))
                    elif command == "metrics":
                        await websocket.send_text(json.dumps({
                            "type": "metrics",
                            "sessionId": session_id,
                            "metrics": session.snapshot(),
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "sessionId": session_id,
                            "message": f"Unknown command: {command}",
                            "where": "command",
                        }))
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:
            if "disconnect" not in str(exc).lower():
                raise
        finally:
            service.remove_session(session_id)
            await manager.disconnect(session_id)

    app.state.realtimestt_service = service
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RealtimeSTT FastAPI browser streaming server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--profile",
        "--tuning-profile",
        dest="tuning_profile",
        choices=sorted(TUNING_PROFILES),
        default="custom",
        help="Named tuning profile. Parakeet profiles tune cadence/batching/VAD timing, not Whisper beam search.",
    )
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--realtime-model", default="tiny.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--engine", "--transcription-engine", dest="transcription_engine", default="faster_whisper")
    parser.add_argument("--realtime-engine", "--realtime-transcription-engine", dest="realtime_transcription_engine")
    parser.add_argument("--engine-options", dest="transcription_engine_options")
    parser.add_argument("--realtime-engine-options", dest="realtime_transcription_engine_options")
    parser.add_argument("--download-root")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-device-index", type=int, default=0)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--beam-size-realtime", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--realtime-batch-size", type=int)
    parser.add_argument("--no-vad-filter", action="store_true")
    parser.add_argument("--normalize-audio", action="store_true")
    parser.add_argument("--realtime-callback", choices=("update", "stabilized"), default="update")
    parser.add_argument("--min-length-of-recording", type=float)
    parser.add_argument("--min-gap-between-recordings", type=float, default=0.0)
    parser.add_argument("--post-speech-silence-duration", type=float)
    parser.add_argument("--silero-sensitivity", type=float, default=0.05)
    parser.add_argument("--webrtc-sensitivity", type=int, default=3)
    parser.add_argument("--realtime-processing-pause", type=float)
    parser.add_argument("--realtime-use-syllable-boundaries", action="store_true")
    parser.add_argument("--realtime-boundary-detector-sensitivity", type=float, default=0.6)
    parser.add_argument("--realtime-boundary-followup-delays", default="0.05,0.2")
    parser.add_argument("--early-transcription-on-silence", type=float)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--initial-prompt-realtime")
    parser.add_argument("--wakeword-backend", default="")
    parser.add_argument("--openwakeword-model-paths")
    parser.add_argument("--openwakeword-inference-framework", default="onnx")
    parser.add_argument("--wake-words", default="")
    parser.add_argument("--wake-words-sensitivity", type=float, default=0.5)
    parser.add_argument("--wake-word-activation-delay", type=float, default=0.0)
    parser.add_argument("--wake-word-timeout", type=float, default=5.0)
    parser.add_argument("--wake-word-buffer-duration", type=float, default=0.1)
    parser.add_argument("--wake-word-followup-window", type=float, default=0.0)
    parser.add_argument("--use-main-model-for-realtime", action="store_true")
    parser.add_argument("--audio-queue-size", type=int, default=128)
    parser.add_argument("--max-audio-packet-bytes", type=int, default=512 * 1024)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--max-active-speakers", type=int, default=4)
    parser.add_argument("--max-audio-queue-seconds-per-session", type=float, default=30.0)
    parser.add_argument("--pre-recording-buffer-duration", type=float, default=0.75)
    parser.add_argument("--max-realtime-queue-age-ms", type=int, default=1500)
    parser.add_argument("--max-final-queue-depth-per-session", type=int, default=8)
    parser.add_argument("--max-global-inference-queue-depth", type=int, default=64)
    parser.add_argument("--realtime-degradation-threshold-ms", type=int, default=1500)
    parser.add_argument("--realtime-min-audio-seconds", type=float, default=0.25)
    parser.add_argument("--realtime-max-audio-seconds", type=float, default=20.0)
    parser.add_argument("--vad-energy-threshold", type=float, default=250.0)
    parser.add_argument("--no-model-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _tuning_defaults(profile):
    defaults = dict(BASE_TUNING_DEFAULTS)
    defaults.update(TUNING_PROFILES[profile]["settings"])
    return defaults


def _value_or_default(args, defaults, name):
    value = getattr(args, name)
    return defaults[name] if value is None else value


def parse_float_tuple(value, flag_name):
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)

    parts = [part.strip() for part in str(value).split(",")]
    try:
        return tuple(float(part) for part in parts if part)
    except ValueError as exc:
        raise SystemExit(f"{flag_name} must be a comma-separated list of numbers") from exc


def settings_from_args(args):
    tuning_profile = args.tuning_profile
    defaults = _tuning_defaults(tuning_profile)
    return ServerSettings(
        host=args.host,
        port=args.port,
        tuning_profile=tuning_profile,
        tuning_description=TUNING_PROFILES[tuning_profile]["description"],
        model=args.model,
        realtime_model=args.realtime_model,
        language=args.language,
        transcription_engine=normalize_engine_name(args.transcription_engine),
        realtime_transcription_engine=normalize_engine_name(args.realtime_transcription_engine),
        transcription_engine_options=parse_json_object(args.transcription_engine_options, "--engine-options"),
        realtime_transcription_engine_options=parse_json_object(
            args.realtime_transcription_engine_options,
            "--realtime-engine-options",
        ),
        download_root=args.download_root,
        compute_type=args.compute_type,
        device=args.device,
        gpu_device_index=args.gpu_device_index,
        beam_size=_value_or_default(args, defaults, "beam_size"),
        beam_size_realtime=_value_or_default(args, defaults, "beam_size_realtime"),
        batch_size=_value_or_default(args, defaults, "batch_size"),
        realtime_batch_size=_value_or_default(args, defaults, "realtime_batch_size"),
        vad_filter=not args.no_vad_filter,
        normalize_audio=args.normalize_audio,
        realtime_callback=args.realtime_callback,
        min_length_of_recording=_value_or_default(args, defaults, "min_length_of_recording"),
        min_gap_between_recordings=args.min_gap_between_recordings,
        post_speech_silence_duration=_value_or_default(args, defaults, "post_speech_silence_duration"),
        silero_sensitivity=args.silero_sensitivity,
        webrtc_sensitivity=args.webrtc_sensitivity,
        realtime_processing_pause=_value_or_default(args, defaults, "realtime_processing_pause"),
        realtime_transcription_use_syllable_boundaries=args.realtime_use_syllable_boundaries,
        realtime_boundary_detector_sensitivity=args.realtime_boundary_detector_sensitivity,
        realtime_boundary_followup_delays=parse_float_tuple(
            args.realtime_boundary_followup_delays,
            "--realtime-boundary-followup-delays",
        ),
        early_transcription_on_silence=_value_or_default(args, defaults, "early_transcription_on_silence"),
        initial_prompt=args.initial_prompt,
        initial_prompt_realtime=args.initial_prompt_realtime,
        wakeword_backend=(
            args.wakeword_backend
            or ("pvporcupine" if args.wake_words else "")
        ),
        openwakeword_model_paths=args.openwakeword_model_paths,
        openwakeword_inference_framework=args.openwakeword_inference_framework,
        wake_words=args.wake_words,
        wake_words_sensitivity=args.wake_words_sensitivity,
        wake_word_activation_delay=args.wake_word_activation_delay,
        wake_word_timeout=args.wake_word_timeout,
        wake_word_buffer_duration=args.wake_word_buffer_duration,
        wake_word_followup_window=args.wake_word_followup_window,
        use_main_model_for_realtime=args.use_main_model_for_realtime,
        audio_queue_size=args.audio_queue_size,
        max_audio_packet_bytes=args.max_audio_packet_bytes,
        max_sessions=args.max_sessions,
        max_active_speakers=args.max_active_speakers,
        max_audio_queue_seconds_per_session=args.max_audio_queue_seconds_per_session,
        pre_recording_buffer_duration=args.pre_recording_buffer_duration,
        max_realtime_queue_age_ms=args.max_realtime_queue_age_ms,
        max_final_queue_depth_per_session=args.max_final_queue_depth_per_session,
        max_global_inference_queue_depth=args.max_global_inference_queue_depth,
        realtime_degradation_threshold_ms=args.realtime_degradation_threshold_ms,
        realtime_min_audio_seconds=args.realtime_min_audio_seconds,
        realtime_max_audio_seconds=args.realtime_max_audio_seconds,
        vad_energy_threshold=args.vad_energy_threshold,
        model_warmup=not args.no_model_warmup,
        log_level=args.log_level.upper(),
    )


def main(argv=None):
    args = parse_args(argv)
    settings = settings_from_args(args)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "uvicorn is missing. Install server dependencies with "
            "'python -m pip install -r example_fastapi_server/requirements.txt'."
        ) from exc

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
