"""Supported production HTTP/WebSocket server for RealtimeSTT.

The original :mod:`RealtimeSTT_server.stt_server` module is a compatibility
implementation for the recorder client's two WebSocket protocol.  This module
provides a separate, versioned remote API.  Its ASR scheduler and recorder
sessions are deliberately borrowed from the tested FastAPI implementation so
that model sharing, fair queues, and per-session cleanup have one owner.

The production WebSocket contract is intentionally small:

* JSON ``start`` creates a turn and supplies a ``turnId`` (or receives one).
* Binary packets carry ``audioSequence`` in their metadata and must be
  contiguous, beginning at zero.
* JSON ``resume`` records the exact accepted-PCM boundary for a new candidate
  without discarding the cumulative turn buffer.
* JSON ``finalize`` drains final inference and produces one ``completion``.
* JSON ``reset`` clears the current turn; ``cancel`` cancels it.

Every server event includes ``apiVersion``, ``protocolVersion``,
``sessionId``, and a monotonically increasing ``eventSequence``.  The legacy
server entry point and the source-only browser example remain separate.
"""

import argparse
import asyncio
import collections
import concurrent.futures
import ipaddress
import importlib.metadata
import json
import logging
import math
import os
import queue
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from RealtimeSTT.core.realtime_merge import StickyRealtimeTranscriptionMerger


LOGGER = logging.getLogger("realtimestt.production_server")

API_VERSION = "v1"
PROTOCOL_VERSION = "realtimestt.remote.v1"
SERVER_NAME = "RealtimeSTT production server"
_PACKAGE_VERSION_FALLBACK = "1.1.0"


def _package_version() -> str:
    """Read the installed distribution version without importing RealtimeSTT."""

    try:
        version = importlib.metadata.version("realtimestt")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return _PACKAGE_VERSION_FALLBACK
    return str(version).strip() or _PACKAGE_VERSION_FALLBACK


SERVER_VERSION = _package_version()
SERVER_SAMPLE_RATE = 16000
PCM_FORMAT = "pcm_s16le"
PREVIEW_TAIL_SECONDS = 5.0
PREVIEW_EMPTY_RETRY_SILENCE_SECONDS = 0.5
LATE_FINAL_OPERATION = "late_full_turn_correction"
RESUME_ACK_TYPE = "resume_ack"
_LIVE_CANCEL = object()
_LIVE_QUEUE_PACKET_FLOOR_SAMPLES = SERVER_SAMPLE_RATE // 10
_MAX_LIVE_CANCEL_THREADS = 4
_LIVE_CANCEL_SLOTS = threading.BoundedSemaphore(_MAX_LIVE_CANCEL_THREADS)
_MAX_LIVE_STREAM_OPERATIONS = 4
_LIVE_STREAM_OPERATION_SLOTS = threading.BoundedSemaphore(
    _MAX_LIVE_STREAM_OPERATIONS
)
_MAX_RESUME_PROVENANCE = 256
REMOTE_LANGUAGES = ("en", "de", "fr", "es", "it", "pt", "ru")
# ``auto`` asks the realtime/final provider to detect the language. Keep the
# seven explicit supported languages alongside it in the public contract.
REMOTE_LANGUAGE_CHOICES = ("auto", *REMOTE_LANGUAGES)
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?$")


class _PreviewAdmissionCancelled(RuntimeError):
    """Internal sentinel for Preview work invalidated before ASR admission."""


class _SampleBoundedQueue:
    """FIFO whose audio capacity is measured in samples, not packet count."""

    def __init__(self, max_samples: int):
        self.max_samples = max(1, int(max_samples))
        self._items = collections.deque()
        self._queued_samples = 0
        self._condition = threading.Condition()
        self.unfinished_tasks = 0

    @staticmethod
    def _weight(item: Any) -> int:
        if item is None or item is _LIVE_CANCEL:
            return 0
        return max(0, int(getattr(item, "size", len(item))))

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        weight = self._weight(item)
        if weight > self.max_samples:
            raise queue.Full
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._queued_samples + weight > self.max_samples:
                if not block:
                    raise queue.Full
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Full
                self._condition.wait(remaining)
            self._items.append((item, weight))
            self._queued_samples += weight
            self.unfinished_tasks += 1
            self._condition.notify_all()

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._items:
                if not block:
                    raise queue.Empty
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            item, weight = self._items.popleft()
            self._queued_samples -= weight
            self._condition.notify_all()
            return item

    def get_nowait(self) -> Any:
        return self.get(block=False)

    def task_done(self) -> None:
        with self._condition:
            if self.unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self.unfinished_tasks -= 1
            self._condition.notify_all()

    def can_put(self, item: Any) -> bool:
        weight = self._weight(item)
        with self._condition:
            return (
                weight <= self.max_samples
                and self._queued_samples + weight <= self.max_samples
            )

    def full(self) -> bool:
        with self._condition:
            return self._queued_samples >= self.max_samples


try:
    from example_fastapi_server.protocol import (
        AudioPacket,
        AudioPacketError,
        decode_audio_packet,
        encode_audio_packet,
        normalize_engine_name,
        parse_json_object,
        require_positive_int,
    )
    from example_fastapi_server.server import (
        ConnectionManager,
        InferenceJob,
        InferenceResult,
        QueueSubmitResult,
        RealtimeSTTService,
        ServerSettings as _ReferenceServerSettings,
        effective_device,
        resample_int16,
        set_current_thread_cpu_affinity,
    )
except (ModuleNotFoundError, RuntimeError) as exc:  # pragma: no cover - wheel smoke tests
    _BACKEND_IMPORT_ERROR = exc

    class AudioPacketError(ValueError):
        """Fallback import-time error type when optional server files are absent."""

    AudioPacket = Any  # type: ignore[assignment,misc]
    ConnectionManager = Any  # type: ignore[assignment,misc]
    InferenceJob = Any  # type: ignore[assignment,misc]
    InferenceResult = Any  # type: ignore[assignment,misc]
    QueueSubmitResult = Any  # type: ignore[assignment,misc]
    RealtimeSTTService = Any  # type: ignore[assignment,misc]

    @dataclass
    class _ReferenceServerSettings:  # type: ignore[no-redef]
        """Minimal fallback allowing settings/help inspection without FastAPI."""

        host: str = "127.0.0.1"
        port: int = 8010
        model: str = "small.en"
        realtime_model: str = "tiny.en"
        language: str = "en"
        transcription_engine: str = "faster_whisper"
        realtime_transcription_engine: Optional[str] = None
        main_cpu_affinity: Optional[Tuple[int, ...]] = None
        realtime_cpu_affinity: Optional[Tuple[int, ...]] = None
        ultrafast_realtime_model_type: Optional[str] = None
        ultrafast_realtime_transcription_engine: Optional[str] = None
        ultrafast_realtime_cpu_affinity: Optional[Tuple[int, ...]] = None
        ultrafast_realtime_transcription_engine_options: Optional[Dict[str, Any]] = None
        ultrafast_realtime_max_tail_words: int = 5
        max_audio_packet_bytes: int = 512 * 1024
        max_sessions: int = 4
        max_active_speakers: int = 4
        model_warmup: bool = True

    def normalize_engine_name(name):
        return None if name is None else str(name).strip().lower().replace("-", "_")

    def parse_json_object(value, name):
        if value in (None, ""):
            return None
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must decode to a JSON object")
        return parsed

    def require_positive_int(metadata, key):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AudioPacketError(f"audio packet metadata field '{key}' must be a positive integer")
        return value

    def decode_audio_packet(message):
        raise RuntimeError("The production server backend is unavailable") from _BACKEND_IMPORT_ERROR

    def encode_audio_packet(metadata, audio):
        raise RuntimeError("The production server backend is unavailable") from _BACKEND_IMPORT_ERROR

    def resample_int16(samples, source_rate, target_rate):
        return samples

    def effective_device(device):
        return device


@dataclass
class ProductionServerSettings(_ReferenceServerSettings):
    """Runtime settings for the supported versioned server.

    All inherited recorder/scheduler settings remain accepted.  The extra
    fields are server-contract controls and are deliberately not sent to
    inference engines.
    """

    host: str = "127.0.0.1"
    port: int = 8010
    bearer_token: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    idle_timeout_seconds: float = 300.0
    max_turn_audio_seconds: float = 120.0
    finalize_timeout_seconds: float = 60.0
    preview_only_transcription: bool = False
    allow_late_final_transcription: bool = False
    late_final_max_audio_seconds: float = 30.0
    preview_tail_seconds: float = PREVIEW_TAIL_SECONDS
    preview_min_live_words_for_fuzzy_repair: int = 3
    allowed_sample_rates: Tuple[int, ...] = (8000, 16000, 24000, 32000, 44100, 48000)
    supported_languages: Tuple[str, ...] = REMOTE_LANGUAGE_CHOICES
    max_http_audio_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        self.host = str(self.host).strip() or "127.0.0.1"
        if self.bearer_token is None:
            self.bearer_token = os.environ.get("REALTIMESTT_SERVER_BEARER_TOKEN")
        self.bearer_token = str(self.bearer_token).strip() if self.bearer_token else None
        self.ssl_certfile = str(self.ssl_certfile).strip() if self.ssl_certfile else None
        self.ssl_keyfile = str(self.ssl_keyfile).strip() if self.ssl_keyfile else None
        self.allowed_sample_rates = tuple(int(rate) for rate in self.allowed_sample_rates)
        self.supported_languages = tuple(
            str(language).strip().lower()
            for language in self.supported_languages
            if str(language).strip()
        )
        configured_language = str(getattr(self, "language", "") or "").strip().lower()
        if configured_language and configured_language not in self.supported_languages:
            self.supported_languages = (configured_language, *self.supported_languages)
        if not self.allowed_sample_rates or any(rate <= 0 for rate in self.allowed_sample_rates):
            raise ValueError("allowed_sample_rates must contain positive sample rates")
        if self.idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be greater than zero")
        if self.max_turn_audio_seconds <= 0:
            raise ValueError("max_turn_audio_seconds must be greater than zero")
        if self.finalize_timeout_seconds <= 0:
            raise ValueError("finalize_timeout_seconds must be greater than zero")
        if (
            not math.isfinite(self.late_final_max_audio_seconds)
            or not 0 < self.late_final_max_audio_seconds <= 30
        ):
            raise ValueError(
                "late_final_max_audio_seconds must be between 0 and 30 seconds"
            )
        if self.preview_tail_seconds <= 0:
            raise ValueError("preview_tail_seconds must be greater than zero")
        if (
            isinstance(self.ultrafast_realtime_max_tail_words, bool)
            or not isinstance(self.ultrafast_realtime_max_tail_words, int)
            or self.ultrafast_realtime_max_tail_words < 1
        ):
            raise ValueError(
                "ultrafast_realtime_max_tail_words must be a positive integer"
            )
        if isinstance(self.preview_min_live_words_for_fuzzy_repair, bool) or self.preview_min_live_words_for_fuzzy_repair < 1:
            raise ValueError("preview_min_live_words_for_fuzzy_repair must be a positive integer")
        if self.max_http_audio_bytes <= 0:
            raise ValueError("max_http_audio_bytes must be greater than zero")
        if bool(self.ssl_certfile) != bool(self.ssl_keyfile):
            raise ValueError(
                "ssl_certfile and ssl_keyfile must be provided together"
            )
        if not is_loopback_host(self.host) and not self.bearer_token:
            raise ValueError(
                "A bearer token is required when host is not loopback; "
                "bind to 127.0.0.1/::1 or set bearer_token."
            )
        if not is_loopback_host(self.host) and not (
            self.ssl_certfile and self.ssl_keyfile
        ):
            raise ValueError(
                "TLS is required when host is not loopback; provide "
                "ssl_certfile and ssl_keyfile, or bind to 127.0.0.1/::1 "
                "behind a TLS-terminating reverse proxy."
            )

    def public_dict(self):
        """Return settings safe to expose through capabilities/configuration."""

        try:
            data = super().public_dict()
        except AttributeError:
            data = asdict(self)
        data.pop("bearer_token", None)
        data.pop("ssl_certfile", None)
        data.pop("ssl_keyfile", None)
        data["allowed_sample_rates"] = list(self.allowed_sample_rates)
        data["supported_languages"] = list(self.supported_languages)
        data["auth_enabled"] = bool(self.bearer_token)
        data["tls_enabled"] = bool(self.ssl_certfile and self.ssl_keyfile)
        return data


# The shorter name is convenient for callers that used ``ServerSettings`` in
# the source-only implementation.  It does not alter that module's class.
ServerSettings = ProductionServerSettings


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is unambiguously local."""

    value = str(host or "").strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _backend_available() -> None:
    if "_BACKEND_IMPORT_ERROR" in globals():
        raise RuntimeError(
            "The production server backend is unavailable. Install the FastAPI "
            "server dependencies and include example_fastapi_server."
        ) from _BACKEND_IMPORT_ERROR


def _language_error(language: Any, settings: ProductionServerSettings) -> Optional[Dict[str, Any]]:
    if not isinstance(language, str) or not language.strip():
        return {"code": "invalid_language", "message": "language must be a non-empty string"}
    normalized = language.strip().lower()
    if normalized not in REMOTE_LANGUAGE_CHOICES and not _LANGUAGE_RE.fullmatch(language.strip()):
        return {
            "code": "unsupported_language",
            "message": f"Unsupported language: {language}",
            "details": {"supportedLanguages": list(settings.supported_languages)},
        }
    if settings.supported_languages and normalized not in settings.supported_languages:
        return {
            "code": "unsupported_language",
            "message": f"Unsupported language: {language}",
            "details": {"supportedLanguages": list(settings.supported_languages)},
        }
    return None


def capabilities_for(settings: ProductionServerSettings) -> Dict[str, Any]:
    """Build the stable capabilities document consumed by remote clients."""

    final_provider = normalize_engine_name(getattr(settings, "transcription_engine", None))
    live_provider = normalize_engine_name(
        getattr(settings, "realtime_transcription_engine", None) or final_provider
    )
    ultrafast_provider = normalize_engine_name(
        getattr(settings, "ultrafast_realtime_transcription_engine", None)
        or live_provider
    )
    final_model = getattr(settings, "model", None)
    live_model = getattr(settings, "realtime_model", None) or final_model
    ultrafast_model = getattr(settings, "ultrafast_realtime_model_type", None)
    ultrafast_enabled = bool(str(ultrafast_model or "").strip())
    max_tail_words = int(
        getattr(settings, "ultrafast_realtime_max_tail_words", 5)
    )
    late_final_enabled = bool(
        not settings.preview_only_transcription
        or settings.allow_late_final_transcription
    )
    late_final_max_audio_seconds = min(
        float(settings.max_turn_audio_seconds),
        float(settings.late_final_max_audio_seconds),
    )
    languages = list(settings.supported_languages or (getattr(settings, "language", "en"),))
    audio = {
        "encoding": "pcm16",
        "format": PCM_FORMAT,
        "sampleRate": SERVER_SAMPLE_RATE,
        "serverSampleRate": SERVER_SAMPLE_RATE,
        "channels": 1,
        "sampleRates": [SERVER_SAMPLE_RATE],
        "httpSampleRates": list(settings.allowed_sample_rates),
        "maxPacketBytes": getattr(settings, "max_audio_packet_bytes", 512 * 1024),
        "maxHttpAudioBytes": settings.max_http_audio_bytes,
    }
    accurate_live = {
        "model": live_model,
        "provider": live_provider,
        "engine": live_provider,
        "languages": languages,
        "language": getattr(settings, "language", None),
        "authoritative": True,
    }
    ultrafast_live = {
        "model": ultrafast_model,
        "provider": ultrafast_provider,
        "engine": ultrafast_provider,
        "languages": languages,
        "language": getattr(settings, "language", None),
        "enabled": ultrafast_enabled,
        "authoritative": False,
        "rawEventType": "ultrafast",
        "rawTextField": "ultrafastText",
    }
    live_contract = {
        "model": live_model,
        "provider": live_provider,
        "engine": live_provider,
        "languages": languages,
        "language": getattr(settings, "language", None),
        "accurate": accurate_live,
        "ultrafast": ultrafast_live,
        "merged": {
            "enabled": ultrafast_enabled,
            "maxTailWords": max_tail_words,
            "accurateTextField": "accurateText",
            "mergedTextField": "mergedText",
            "ultrafastTextField": "ultrafastText",
            "ultrafastSuffixField": "ultrafastSuffix",
            "ultrafastEventType": "ultrafast",
            "rawUltrafastEventType": "ultrafast",
        },
    }
    return {
        "apiVersion": API_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "models": {
            "final": {
                "model": final_model,
                "provider": final_provider,
                "engine": final_provider,
                "languages": languages,
                "language": getattr(settings, "language", None),
                "enabled": not settings.preview_only_transcription,
            },
            "lateFinal": {
                "model": final_model,
                "provider": final_provider,
                "engine": final_provider,
                "languages": languages,
                "language": getattr(settings, "language", None),
                "enabled": late_final_enabled,
                "operation": LATE_FINAL_OPERATION,
                "maxAudioSeconds": late_final_max_audio_seconds,
            },
            "preview": {
                "model": final_model,
                "provider": final_provider,
                "engine": final_provider,
                "languages": languages,
                "language": getattr(settings, "language", None),
                "inputCoverage": "full_turn",
            },
            "live": live_contract,
            # realtime is retained as a descriptive alias for live/partial.
            "realtime": dict(live_contract),
        },
        "resume": {
            "command": "resume",
            "ackType": RESUME_ACK_TYPE,
            "resumeIdField": "resumeId",
            "requestIdField": "requestId",
            "correlationFields": ["resumeId", "requestId"],
            "turnIdField": "turnId",
            "candidateIdField": "candidateId",
            "audioSequenceField": "audioSequence",
            "sampleOffsetField": "sampleOffset",
            "byteOffsetField": "byteOffset",
            "inputSampleRangeField": "inputSampleRange",
            "inputByteRangeField": "inputByteRange",
            "liveProvenance": {
                "resumeEpochField": "resumeEpoch",
                "candidateIdField": "candidateId",
                "resumeIdField": "resumeId",
                "candidateStartSampleField": "candidateStartSample",
                "audioEndSampleExclusiveField": "audioEndSampleExclusive",
                "endExclusive": True,
            },
            "preview": {
                "cumulativeTextField": "cumulativeText",
                "candidateTextField": "candidateText",
                "candidateOnlyTextField": "candidateOnlyText",
                "inputModeField": "candidateInputScope",
                "fullTurnInputMode": "full_turn",
                "inputCoverageField": "previewInputCoverage",
                "fullTurnInputCoverage": "full_turn",
            },
        },
        "mergedRealtime": {
            "enabled": ultrafast_enabled,
            "accurateModel": live_model,
            "ultrafastModel": ultrafast_model,
            "maxTailWords": max_tail_words,
            "accurateTextField": "accurateText",
            "mergedTextField": "mergedText",
            "ultrafastTextField": "ultrafastText",
            "ultrafastSuffixField": "ultrafastSuffix",
            "mergeStatusField": "mergeStatus",
            "ultrafastEventType": "ultrafast",
            "rawUltrafastEventType": "ultrafast",
        },
        "finalModel": final_model,
        "finalProvider": final_provider,
        "liveModel": live_model,
        "liveProvider": live_provider,
        "ultrafastLiveModel": ultrafast_model,
        "ultrafastLiveProvider": ultrafast_provider if ultrafast_enabled else None,
        "previewModel": final_model,
        "previewProvider": final_provider,
        "languages": languages,
        "audioFormat": PCM_FORMAT,
        "previewOnlyTranscription": bool(settings.preview_only_transcription),
        "finalAsrEnabled": not bool(settings.preview_only_transcription),
        "lateFinalAsrEnabled": late_final_enabled,
        "previewInputCoverage": "full_turn",
        "audio": audio,
        "limits": {
            "maxSessions": getattr(settings, "max_sessions", None),
            "maxActiveSpeakers": getattr(settings, "max_active_speakers", None),
            "maxTurnAudioSeconds": settings.max_turn_audio_seconds,
            "idleTimeoutSeconds": settings.idle_timeout_seconds,
            "finalizeTimeoutSeconds": settings.finalize_timeout_seconds,
            "lateFinalMaxAudioSeconds": late_final_max_audio_seconds,
            "maxAudioPacketBytes": audio["maxPacketBytes"],
        },
        "operations": {
            "websocket": [
                "start",
                "audio",
                "preview",
                "resume",
                "finalize",
                "reset",
                "cancel",
            ],
            "resume": {
                "command": "resume",
                "ackType": RESUME_ACK_TYPE,
                "correlationField": "resumeId",
                "legacyCorrelationField": "requestId",
                "boundary": {
                    "sampleRate": SERVER_SAMPLE_RATE,
                    "bytesPerSample": 2,
                    "endExclusive": True,
                },
                "liveProvenance": {
                    "resumeEpochField": "resumeEpoch",
                    "candidateIdField": "candidateId",
                    "resumeIdField": "resumeId",
                    "candidateStartSampleField": "candidateStartSample",
                    "audioEndSampleExclusiveField": "audioEndSampleExclusive",
                    "endExclusive": True,
                },
            },
            "events": (
                ["partial"]
                + (["ultrafast"] if ultrafast_enabled else [])
                + ["preview", RESUME_ACK_TYPE, "completion", "status", "error"]
                if settings.preview_only_transcription
                else ["partial"]
                + (["ultrafast"] if ultrafast_enabled else [])
                + [
                    "final",
                    "preview",
                    RESUME_ACK_TYPE,
                    "completion",
                    "status",
                    "error",
                ]
            ),
            "http": (
                ["transcribe-pcm16"]
                if not settings.preview_only_transcription
                else (
                    [f"transcribe-pcm16?operation={LATE_FINAL_OPERATION}"]
                    if settings.allow_late_final_transcription
                    else []
                )
            ),
        },
        "authentication": {
            "scheme": "bearer",
            "required": bool(settings.bearer_token) or not is_loopback_host(settings.host),
        },
    }

def _structured_error(
    code: str,
    message: str,
    *,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "error",
        "apiVersion": API_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
        "error": {"code": code, "message": message},
        # Top-level fields are intentionally duplicated for simple clients and
        # compatibility with the reference server's error shape.
        "code": code,
        "message": message,
    }
    if details:
        payload["error"]["details"] = details
        payload["details"] = details
    if session_id is not None:
        payload["sessionId"] = session_id
    if turn_id is not None:
        payload["turnId"] = turn_id
    return payload


@dataclass
class _PendingDelivery:
    event: Dict[str, Any]
    completion: concurrent.futures.Future


@dataclass
class _SessionDelivery:
    loop: asyncio.AbstractEventLoop
    wakeup: asyncio.Event
    queue: collections.deque
    sender_task: Optional[asyncio.Task] = None
    closed: bool = False
    resume_ack_pending: int = 0
    # A Resume can briefly require two slots beyond ordinary interim
    # backpressure: its ordered acknowledgement plus the first candidate
    # partial behind that acknowledgement. Both reservations are per active
    # ACK barrier, never a general control-message escape hatch.
    resume_candidate_reserve_key: Optional[Tuple[Any, Any, Any]] = None
    resume_candidate_partial_reserved: bool = False


class FinalEventBarrier:
    """One-shot notification for a final outcome already queued for delivery.

    A counter says that inference finished; it does not say that the final
    event has been handed to the session's ordered outbound lane.  This
    barrier is resolved by :class:`OrderedConnectionManager` only after that
    enqueue has happened, so a completion producer cannot overtake the final
    (or its structured failure).
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._resolved = False
        self.outcome: Optional[Dict[str, Any]] = None

    def resolve(self, outcome: Optional[Dict[str, Any]]) -> bool:
        with self._lock:
            if self._resolved:
                return False
            self._resolved = True
            self.outcome = dict(outcome) if isinstance(outcome, dict) else outcome
            self._event.set()
            return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout=timeout)


class OrderedConnectionManager(ConnectionManager):
    """Decorate and deliver backend events in per-session FIFO order.

    Recorder callbacks can run on worker threads while command handlers run on
    the WebSocket event loop.  A sequence number alone is not sufficient to
    preserve their order: two independent send coroutines can still complete
    in the opposite order.  Every session therefore owns a small FIFO and one
    sender task.  Numbering and appending to that FIFO happen while holding
    ``_event_lock``, giving concurrent producers one linearizable order while
    keeping slow clients isolated from other sessions.
    """

    _TERMINAL_TYPES = frozenset({"preview", "final", "completion"})

    def __init__(self, max_pending_events: int = 256):
        if isinstance(max_pending_events, bool) or int(max_pending_events) <= 0:
            raise ValueError("max_pending_events must be a positive integer")
        super().__init__()
        self.max_pending_events = int(max_pending_events)
        self._event_lock = threading.RLock()
        self._event_sequences = collections.defaultdict(int)
        self._connection_epochs: Dict[str, int] = {}
        self._next_connection_epoch = 0
        self._turn_ids: Dict[str, Optional[str]] = {}
        self._audio_sequences: Dict[str, Optional[int]] = {}
        self._suppressed_types = collections.defaultdict(set)
        self._delivery_states: Dict[str, _SessionDelivery] = {}
        self._final_barriers: Dict[Tuple[str, str], FinalEventBarrier] = {}

    async def connect(self, session_id: str, websocket) -> None:
        with self._event_lock:
            existing = self._delivery_states.get(session_id)
        if existing is not None:
            await self.disconnect(session_id)
        await super().connect(session_id, websocket)
        loop = asyncio.get_running_loop()
        state = _SessionDelivery(loop, asyncio.Event(), collections.deque())
        with self._event_lock:
            self._next_connection_epoch += 1
            self._connection_epochs[session_id] = self._next_connection_epoch
            self._delivery_states[session_id] = state
        state.sender_task = loop.create_task(
            self._deliver_session(session_id, state),
            name=f"RealtimeSTT-send-{session_id}",
        )

    @staticmethod
    def _resolve_delivery(
        completion: concurrent.futures.Future, delivered: bool
    ) -> None:
        if not completion.done():
            completion.set_result(bool(delivered))

    def _close_delivery_state(self, state: _SessionDelivery) -> None:
        """Stop a sender and fail all events still waiting in its queue."""

        with self._event_lock:
            if state.closed:
                pending = []
            else:
                state.closed = True
                pending = list(state.queue)
                state.queue.clear()
                state.resume_ack_pending = 0
                state.resume_candidate_reserve_key = None
                state.resume_candidate_partial_reserved = False
        for item in pending:
            self._resolve_delivery(item.completion, False)

        task = state.sender_task
        if task is None or task.done():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is state.loop:
            if task is not asyncio.current_task():
                task.cancel()
        else:
            try:
                state.loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    async def disconnect(self, session_id: str) -> None:
        with self._event_lock:
            state = self._delivery_states.pop(session_id, None)
        if state is not None:
            self._close_delivery_state(state)
            task = state.sender_task
            if task is not None and not task.done() and task is not asyncio.current_task():
                await asyncio.gather(task, return_exceptions=True)
        await super().disconnect(session_id)

    def _wake_delivery(self, state: _SessionDelivery) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        try:
            if running_loop is state.loop:
                state.wakeup.set()
            else:
                state.loop.call_soon_threadsafe(state.wakeup.set)
        except RuntimeError:
            self._close_delivery_state(state)

    def _disconnect_overflowed_session(
        self,
        session_id: str,
        state: _SessionDelivery,
    ) -> None:
        """Close one stalled transport without blocking a producer thread."""

        self._close_delivery_state(state)

        async def disconnect() -> None:
            async with self._lock:
                websocket = self._connections.get(session_id)
            close = getattr(websocket, "close", None)
            if callable(close):
                try:
                    await close(code=1013, reason="outbound backpressure")
                except TypeError:
                    await close()
                except Exception:
                    LOGGER.debug(
                        "Could not close overflowed WebSocket session %s",
                        session_id,
                        exc_info=True,
                    )
            await self.disconnect(session_id)

        try:
            state.loop.call_soon_threadsafe(
                lambda: state.loop.create_task(disconnect())
            )
        except RuntimeError:
            pass

    async def _deliver_session(
        self, session_id: str, state: _SessionDelivery
    ) -> None:
        current: Optional[_PendingDelivery] = None
        try:
            while True:
                await state.wakeup.wait()
                while True:
                    stale_pending = None
                    with self._event_lock:
                        if state.closed or self._delivery_states.get(session_id) is not state:
                            stale_pending = list(state.queue)
                            state.queue.clear()
                            state.resume_ack_pending = 0
                            state.resume_candidate_reserve_key = None
                            state.resume_candidate_partial_reserved = False
                            state.closed = True
                            state.wakeup.clear()
                        elif not state.queue:
                            state.wakeup.clear()
                            break
                        else:
                            current = state.queue.popleft()

                    if stale_pending is not None:
                        for item in stale_pending:
                            self._resolve_delivery(item.completion, False)
                        return

                    if current.event.get("type") == RESUME_ACK_TYPE:
                        delivered = await self.send(session_id, current.event)
                        # Keep the barrier (and its one candidate reserve)
                        # active until the ACK has actually passed the
                        # transport. While a blocked sender owns the ACK,
                        # producers can safely append one matching candidate
                        # partial after it without coalescing into pre-ACK
                        # text or consuming unbounded queue space.
                        with self._event_lock:
                            if self._delivery_states.get(session_id) is state:
                                state.resume_ack_pending = 0
                                state.resume_candidate_reserve_key = None
                                state.resume_candidate_partial_reserved = False
                    else:
                        delivered = await self.send(session_id, current.event)
                    self._resolve_delivery(current.completion, delivered)
                    current = None
                    if not delivered:
                        self._close_delivery_state(state)
                        return
        except asyncio.CancelledError:
            if current is not None:
                self._resolve_delivery(current.completion, False)
            with self._event_lock:
                pending = list(state.queue)
                state.queue.clear()
                state.closed = True
                state.resume_ack_pending = 0
                state.resume_candidate_reserve_key = None
                state.resume_candidate_partial_reserved = False
            for item in pending:
                self._resolve_delivery(item.completion, False)
            raise

    def clear_session(self, session_id: str) -> None:
        with self._event_lock:
            state = self._delivery_states.pop(session_id, None)
            self._turn_ids.pop(session_id, None)
            self._audio_sequences.pop(session_id, None)
            self._event_sequences.pop(session_id, None)
            self._connection_epochs.pop(session_id, None)
            self._suppressed_types.pop(session_id, None)
            barriers = [
                barrier
                for (barrier_session_id, _), barrier in self._final_barriers.items()
                if barrier_session_id == session_id
            ]
            for key in [
                key for key in self._final_barriers if key[0] == session_id
            ]:
                self._final_barriers.pop(key, None)
        if state is not None:
            self._close_delivery_state(state)
        for barrier in barriers:
            barrier.resolve(None)

    def register_final_barrier(self, session_id: str, turn_id: str) -> FinalEventBarrier:
        """Register the final outcome expected for one active turn.

        Registration occurs before recorder flushing starts.  That makes a
        final event emitted from a worker thread observable even when it wins
        the race with the command handler's completion waiter.
        """

        barrier = FinalEventBarrier()
        with self._event_lock:
            previous = self._final_barriers.get((session_id, turn_id))
            self._final_barriers[(session_id, turn_id)] = barrier
        if previous is not None:
            previous.resolve(None)
        return barrier

    def unregister_final_barrier(
        self, session_id: str, turn_id: str, barrier: FinalEventBarrier
    ) -> None:
        with self._event_lock:
            if self._final_barriers.get((session_id, turn_id)) is barrier:
                self._final_barriers.pop((session_id, turn_id), None)

    @staticmethod
    def _is_final_outcome(message: Dict[str, Any]) -> bool:
        source_type = message.get("type")
        if source_type in {"final", "preview"}:
            return True
        if source_type != "error":
            return False
        code = str(message.get("code") or "").lower()
        where = str(message.get("where") or "").lower()
        return where in {"final", "recorder"} or code.startswith("final")

    @staticmethod
    def _resume_candidate_key(message: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Return the immutable provenance that owns one Resume candidate."""

        resume_id = message.get("resumeId")
        if resume_id is None:
            resume_id = message.get("resumeRequestId")
        return (
            message.get("resumeEpoch"),
            message.get("candidateId"),
            resume_id,
        )

    def set_turn(self, session_id: str, turn_id: Optional[str]) -> None:
        with self._event_lock:
            self._turn_ids[session_id] = turn_id
            self._audio_sequences[session_id] = None

    def connection_epoch(self, session_id: str) -> int:
        """Return the monotonic transport generation for one session id."""

        with self._event_lock:
            return int(self._connection_epochs.get(session_id, 0))

    def set_audio_sequence(self, session_id: str, sequence: Optional[int]) -> None:
        with self._event_lock:
            self._audio_sequences[session_id] = sequence

    def suppress_type(self, session_id: str, message_type: str, enabled: bool = True) -> None:
        with self._event_lock:
            if enabled:
                self._suppressed_types[session_id].add(message_type)
            else:
                self._suppressed_types[session_id].discard(message_type)

    def _decorate(self, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        with self._event_lock:
            return self._decorate_locked(session_id, message)

    def _decorate_locked(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        sequence: Optional[int] = None,
    ) -> Dict[str, Any]:
        if sequence is None:
            self._event_sequences[session_id] += 1
            sequence = self._event_sequences[session_id]
        turn_id = self._turn_ids.get(session_id)
        audio_sequence = self._audio_sequences.get(session_id)
        event = dict(message)
        event.pop("_connectionEpoch", None)
        source_type = event.get("type")
        if source_type == "realtime":
            event["type"] = "partial"
            event["kind"] = "realtime"
            event.setdefault("partialText", event.get("text", ""))
        elif source_type == "clear":
            event["type"] = "reset"
            event["kind"] = "clear"
        event["apiVersion"] = API_VERSION
        event["protocolVersion"] = PROTOCOL_VERSION
        event["sessionId"] = session_id
        event["eventSequence"] = sequence
        if turn_id is not None:
            event.setdefault("turnId", turn_id)
        if audio_sequence is not None:
            event.setdefault("audioSequence", audio_sequence)
        if event.get("type") == "error":
            code = str(event.get("code") or event.get("where") or "server_error")
            message_text = str(event.get("message") or "Server error")
            details = {
                key: event[key]
                for key in ("where", "requestId")
                if key in event
            }
            event["code"] = code
            event["error"] = {"code": code, "message": message_text}
            if details:
                event["error"]["details"] = details
        return event

    def _enqueue(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        respect_suppression: bool = False,
    ) -> Optional[concurrent.futures.Future]:
        with self._event_lock:
            state = self._delivery_states.get(session_id)
            if state is None or state.closed:
                return None
            expected_epoch = message.get("_connectionEpoch")
            if (
                expected_epoch is not None
                and int(expected_epoch) != self._connection_epochs.get(session_id, 0)
            ):
                return None
            if (
                respect_suppression
                and (
                    message.get("type") in self._suppressed_types.get(session_id, set())
                    or (
                        "final_outcome" in self._suppressed_types.get(session_id, set())
                        and self._is_final_outcome(message)
                    )
                )
            ):
                return None

            source_type = message.get("type")
            is_partial = source_type in {"partial", "realtime"}
            reserve_candidate_partial = False
            if is_partial:
                # Keep the oldest pending sequence slot but replace its
                # hypothesis with the newest one.  No new sequence is
                # allocated for coalesced updates, so clients see a contiguous
                # stream even when a slow client skips stale partials.
                current_turn_id = self._turn_ids.get(session_id)
                message_turn_id = message.get("turnId", current_turn_id)
                if message_turn_id != current_turn_id:
                    return None
                if message_turn_id == current_turn_id:
                    if state.resume_ack_pending:
                        candidate_resume_key = self._resume_candidate_key(message)
                        # A Resume acknowledgement is an ordered client-side
                        # ownership barrier. The client deliberately ignores
                        # candidate events until this ack arrives, so a new
                        # candidate partial must never replace a pre-ack queue
                        # slot and inherit its older event sequence.
                        for pending in reversed(state.queue):
                            if pending.event.get("type") == RESUME_ACK_TYPE:
                                break
                            if (
                                pending.event.get("type") != "partial"
                                or pending.event.get("turnId") != current_turn_id
                            ):
                                continue
                            # A delayed partial from the pre-Resume stream
                            # may have the same turn id, but it belongs to a
                            # different candidate.  Replacing a reserved
                            # candidate slot with it would make the client
                            # discard the only valid update after its ACK.
                            if (
                                self._resume_candidate_key(pending.event)
                                != candidate_resume_key
                            ):
                                continue
                            sequence = int(pending.event["eventSequence"])
                            pending.event = self._decorate_locked(
                                session_id, message, sequence=sequence
                            )
                            self._wake_delivery(state)
                            return pending.completion
                        if (
                            state.resume_candidate_reserve_key is not None
                            and not state.resume_candidate_partial_reserved
                            and len(state.queue) >= self.max_pending_events
                            and candidate_resume_key
                            == state.resume_candidate_reserve_key
                        ):
                            # The normal interim budget is already occupied
                            # by pre-ACK work. Reserve exactly one matching
                            # post-ACK candidate slot so the client cannot
                            # miss the first valid candidate update.
                            reserve_candidate_partial = True
                    else:
                        for pending in reversed(state.queue):
                            if (
                                pending.event.get("type") != "partial"
                                or pending.event.get("turnId") != current_turn_id
                            ):
                                continue
                            sequence = int(pending.event["eventSequence"])
                            pending.event = self._decorate_locked(
                                session_id, message, sequence=sequence
                            )
                            self._wake_delivery(state)
                            return pending.completion

            is_resume_ack = source_type == RESUME_ACK_TYPE
            is_final_outcome = self._is_final_outcome(message)
            resume_reserved_slots = (
                int(bool(state.resume_ack_pending))
                + int(state.resume_candidate_partial_reserved)
            )
            terminal_hard_limit = self.max_pending_events + 2 + resume_reserved_slots
            terminal_overflow = (
                is_final_outcome
                and len(state.queue)
                > self.max_pending_events + resume_reserved_slots
            ) or (
                source_type == "completion"
                and len(state.queue) >= terminal_hard_limit
            )
            if terminal_overflow:
                LOGGER.warning(
                    "Outbound terminal reserve exhausted for session %s; disconnecting",
                    session_id,
                )
                self._disconnect_overflowed_session(session_id, state)
                return None

            if is_resume_ack:
                if (
                    state.resume_ack_pending
                    or len(state.queue) >= self.max_pending_events + 1
                ):
                    # A Resume ACK alone may use one bounded FIFO reserve
                    # when ordinary interim traffic has saturated the queue.
                    # A second ACK (or terminal-reserve saturation) cannot
                    # bypass that limit, so arbitrary controls remain bounded.
                    LOGGER.warning(
                        "Outbound Resume ACK reserve unavailable for session %s",
                        session_id,
                    )
                    return None
            if (
                len(state.queue) >= self.max_pending_events
                and source_type not in self._TERMINAL_TYPES
                and not is_final_outcome
                and not is_resume_ack
                and not reserve_candidate_partial
            ):
                # Reject before decorating so a producer-side backpressure
                # decision never creates an eventSequence gap.  Terminal
                # events are allowed through the small limit so final and
                # completion cannot be displaced by stale interim traffic.
                LOGGER.warning(
                    "Outbound event queue full for session %s; rejecting %s",
                    session_id,
                    source_type,
                )
                return None
            event = self._decorate_locked(session_id, message)
            completion: concurrent.futures.Future = concurrent.futures.Future()
            state.queue.append(_PendingDelivery(event, completion))
            if is_resume_ack:
                state.resume_ack_pending += 1
                state.resume_candidate_reserve_key = self._resume_candidate_key(event)
                state.resume_candidate_partial_reserved = False
            elif reserve_candidate_partial:
                state.resume_candidate_reserve_key = None
                state.resume_candidate_partial_reserved = True
            if is_final_outcome:
                turn_id = event.get("turnId")
                if turn_id is not None:
                    barrier = self._final_barriers.get((session_id, str(turn_id)))
                    if barrier is not None:
                        # Append first, then resolve.  A completion producer
                        # released by this notification will therefore append
                        # after the final outcome in the same FIFO.
                        barrier.resolve(event)
        self._wake_delivery(state)
        return completion

    async def emit(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Queue a decorated event and wait for this session's sender."""

        completion = self._enqueue(session_id, message)
        if completion is None:
            return False
        return bool(await asyncio.wrap_future(completion))

    def publish_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        authoritative: bool = False,
    ) -> Optional[concurrent.futures.Future]:
        if self._loop is None:
            return None
        return self._enqueue(
            session_id,
            message,
            respect_suppression=not authoritative,
        )

    def queue_resume_ack(
        self,
        session_id: str,
        message: Dict[str, Any],
    ) -> Optional[concurrent.futures.Future]:
        """Append one accepted Resume ACK before candidate state is unlocked."""

        if message.get("type") != RESUME_ACK_TYPE:
            raise ValueError("queue_resume_ack accepts only resume_ack events")
        return self._enqueue(session_id, message)

    def publish_all(self, message: Dict[str, Any]) -> None:
        # The backend emits a process-wide ``ready`` event.  Per-session
        # connect handlers send the versioned ready event, so dropping this
        # broadcast avoids an unscoped/unversioned event.
        LOGGER.debug("Dropping unscoped backend broadcast: %s", message.get("type"))


@dataclass
class TurnState:
    turn_id: str
    language: str
    phase: str = "receiving"
    expected_audio_sequence: int = 0
    first_audio_sequence: Optional[int] = None
    last_audio_sequence: Optional[int] = None
    packet_count: int = 0
    audio_frames: int = 0
    audio_seconds: float = 0.0
    audio_revision: int = 0
    partial_count: int = 0
    final_count: int = 0
    preview_count: int = 0
    preview_requested: bool = False
    latest_preview_request_id: Optional[str] = None
    latest_preview_audio_revision: Optional[int] = None
    preview_completed_request_id: Optional[str] = None
    preview_completed_audio_revision: Optional[int] = None
    preview_status: str = ""
    preview_failure: Optional[Dict[str, Any]] = None
    resume_count: int = 0
    # Each accepted Resume is an immutable PCM boundary.  Retain a bounded
    # suffix because a cumulative live worker can publish an older endpoint
    # after a newer candidate has already been acknowledged, while an
    # authenticated client must not be able to grow this list forever.
    resume_epoch: int = 0
    resume_provenance: list[tuple[int, int, str, str]] = field(
        default_factory=list
    )
    last_resume_request_id: Optional[str] = None
    last_resume_ack: Optional[Dict[str, Any]] = None
    last_resume_ack_delivery: Optional[concurrent.futures.Future] = None
    candidate_id: Optional[str] = None
    candidate_start_sample: int = 0
    candidate_start_byte: int = 0
    candidate_base_text: str = ""
    finalize_requested: bool = False
    completion_sent: bool = False
    generation: int = 0
    connection_epoch: Optional[int] = None
    pcm_buffer: bytearray = field(default_factory=bytearray)
    terminal_sent: bool = False
    live_queue: Any = None
    live_stream: Any = None
    live_thread: Any = None
    live_done: threading.Event = field(default_factory=threading.Event)
    live_cancelled: threading.Event = field(default_factory=threading.Event)
    live_cancel_attempted: bool = False
    ultrafast_live_queue: Any = None
    ultrafast_live_stream: Any = None
    ultrafast_live_thread: Any = None
    ultrafast_live_done: threading.Event = field(default_factory=threading.Event)
    ultrafast_live_cancel_attempted: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    # Preview inference is independently cancellable.  A Resume invalidates
    # only snapshot work; it must not tear down the shared live stream.
    preview_epoch: int = 0
    preview_cancelled: threading.Event = field(default_factory=threading.Event)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)


class ProductionSessionHandle:
    """Lightweight SessionStore entry for the explicit production protocol."""

    def __init__(self, service: Any, session_id: str):
        self.service = service
        self.session_id = session_id
        self.generation = 0
        self.closed = False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "streaming": not self.closed,
            "recording": False,
            "finalSubmitted": 0,
            "finalCompleted": 0,
            "realtimeCompleted": 0,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.generation += 1
        self.service.scheduler.cancel_session(self.session_id)
        self.service.cancel_pending_recorder_transcriptions(self.session_id)

    def clear(self) -> None:
        self.generation += 1
        self.service.scheduler.cancel_session(self.session_id)
        self.service.cancel_pending_recorder_transcriptions(self.session_id)

    def handle_inference_result(self, result: Any) -> None:
        del result

    def on_job_dropped(self, job: Any, reason: str) -> None:
        self.service.fail_pending_recorder_transcription(
            job.request_id,
            f"{job.kind} transcription was {reason}",
        )

    def on_submit_result(self, job: Any, result: Any) -> None:
        del job, result


def _admit_production_session(service: Any, session_id: str) -> Optional[ProductionSessionHandle]:
    if not service.sessions.reserve(session_id):
        return None
    session = ProductionSessionHandle(service, session_id)
    if not service.sessions.add(session):
        session.close()
        return None
    return session


def _turn_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 128:
        return None
    return value


def _audio_sequence(metadata: Dict[str, Any]) -> Optional[int]:
    for key in ("audioSequence", "audio_sequence", "sequence"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AudioPacketError("audio packet metadata audioSequence must be a non-negative integer")
            return value
    return None


def _validate_production_packet(
    packet: AudioPacket,
    settings: ProductionServerSettings,
) -> Tuple[int, int, int]:
    """Validate packet metadata and return ``(sequence, frames, rate)``."""

    if len(packet.audio) == 0:
        raise AudioPacketError("audio packet payload must not be empty")
    if len(packet.audio) > getattr(settings, "max_audio_packet_bytes", 512 * 1024):
        raise AudioPacketError("audio packet is too large")
    sample_rate = require_positive_int(packet.metadata, "sampleRate")
    if sample_rate not in settings.allowed_sample_rates:
        raise AudioPacketError(
            f"sample rate {sample_rate} is not supported; "
            f"choose one of {list(settings.allowed_sample_rates)}"
        )
    channels = packet.metadata.get("channels")
    if channels != 1:
        raise AudioPacketError("production remote audio must be mono (channels=1)")
    audio_format = packet.metadata.get("format")
    if audio_format != PCM_FORMAT:
        raise AudioPacketError(f"only {PCM_FORMAT} audio packets are supported")
    if len(packet.audio) % 2:
        raise AudioPacketError("pcm_s16le audio packet is not aligned to whole samples")
    frames = require_positive_int(packet.metadata, "frames")
    if len(packet.audio) != frames * 2:
        raise AudioPacketError("audio packet metadata field 'frames' does not match payload length")
    sequence = _audio_sequence(packet.metadata)
    if sequence is None:
        raise AudioPacketError("audio packet metadata field 'audioSequence' is required")
    return sequence, frames, sample_rate


def _release_engine(engine: Any) -> None:
    if engine is None:
        return
    for method_name in ("shutdown", "close", "release"):
        method = getattr(engine, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                LOGGER.debug("Engine %s() failed during shutdown", method_name, exc_info=True)
            return


def release_service_resources(service: Any) -> None:
    """Release backend engines after workers have stopped, exactly once."""

    if getattr(service, "_production_resources_released", False):
        return
    scheduler = getattr(service, "scheduler", None)
    workers = []
    for name in ("main_worker", "realtime_worker", "ultrafast_worker"):
        worker = getattr(scheduler, name, None)
        if worker is not None:
            workers.append(worker)
    seen = set()
    for worker in workers:
        engine = getattr(worker, "engine", None)
        if engine is not None and id(engine) not in seen:
            seen.add(id(engine))
            _release_engine(engine)
            try:
                worker.engine = None
            except Exception:
                pass
    try:
        service._production_resources_released = True
    except Exception:
        pass


class ProductionSessionProtocol:
    """State machine for one versioned remote WebSocket connection."""

    def __init__(self, service: Any, manager: OrderedConnectionManager, session_id: str, settings):
        self.service = service
        self.manager = manager
        self.settings = settings
        self.session_id = session_id
        self.session = None
        self.turn: Optional[TurnState] = None
        self.closed = False
        self._lock = threading.RLock()
        self._publish_lock = threading.Lock()
        self._completion_threads = set()
        # Preview requests are latest-only snapshots.  Keep at most one
        # dispatcher thread and one pending snapshot per protocol, so a
        # client cannot turn request frequency into unbounded native workers.
        self._preview_dispatch_thread = None
        self._preview_pending = None
        self._live_cancel_threads = set()
        self._live_stream_operation_threads = set()
        self._generation = 0
        self._last_partial = ""
        self._last_partial_sent_at = 0.0
        self._last_ultrafast_text = ""
        self._last_ultrafast_recording_id = None

        self._last_merged_text = ""
        self._ultrafast_enabled = bool(
            str(
                getattr(settings, "ultrafast_realtime_model_type", "") or ""
            ).strip()
        )
        self._realtime_merger = StickyRealtimeTranscriptionMerger(
            max_ultrafast_tail_words=int(
                getattr(settings, "ultrafast_realtime_max_tail_words", 5)
            )
        )

    def attach(self, session: Any) -> None:
        self.session = session

    @staticmethod
    def _close_live_stream(stream: Any) -> None:
        close = getattr(stream, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            LOGGER.debug("Could not close production live stream", exc_info=True)

    def _live_queue_capacity_samples(self) -> int:
        configured_samples = int(
            max(1, int(getattr(self.settings, "audio_queue_size", 100)))
            * SERVER_SAMPLE_RATE
            * 0.04
        )
        duration_samples = int(
            max(
                0.04,
                float(
                    getattr(
                        self.settings,
                        "max_audio_queue_seconds_per_session",
                        30.0,
                    )
                ),
            )
            * SERVER_SAMPLE_RATE
        )
        # A single supported transport packet must always fit into an empty
        # sample-bounded queue.  Otherwise ``audio_queue_size=1`` spuriously
        # rejects the documented 64/100-ms packet cadences before any backlog.
        return max(
            _LIVE_QUEUE_PACKET_FLOOR_SAMPLES,
            min(configured_samples, duration_samples),
        )

    def _install_live_stream(
        self,
        turn_id: str,
        generation: int,
        stream: Any,
        ultrafast_stream: Any = None,
    ) -> bool:
        """Install both created streams only while their starting turn owns them."""

        threads = []
        turn = None
        with self._lock:
            turn = self.turn
            current = (
                not self.closed
                and turn is not None
                and turn.turn_id == turn_id
                and turn.generation == generation
                and turn.phase == "starting"
            )
            if current:
                turn.live_stream = stream
                turn.live_queue = _SampleBoundedQueue(
                    self._live_queue_capacity_samples()
                )
                turn.live_done.clear()
                turn.ultrafast_live_stream = ultrafast_stream
                if ultrafast_stream is not None:
                    turn.ultrafast_live_queue = _SampleBoundedQueue(
                        self._live_queue_capacity_samples()
                    )
                    turn.ultrafast_live_done.clear()
                else:
                    turn.ultrafast_live_queue = None
                    turn.ultrafast_live_done.set()
                self._last_partial = ""
                self._last_ultrafast_text = ""
                self._last_ultrafast_recording_id = (
                    turn.turn_id,
                    turn.generation,
                )
                self._last_merged_text = ""
                self._last_partial_sent_at = 0.0
                self._realtime_merger.reset((turn.turn_id, turn.generation))
                turn.live_thread = threading.Thread(
                    target=self._live_worker,
                    args=(
                        turn.turn_id,
                        turn.generation,
                        "realtime",
                        turn.live_queue,
                        turn.live_stream,
                        turn.live_done,
                        turn.live_cancelled,
                    ),
                    name=(
                        f"RealtimeSTTProductionLive-{self.session_id}-"
                        f"{turn.turn_id}"
                    ),
                    daemon=True,
                )
                threads.append(turn.live_thread)
                if ultrafast_stream is not None:
                    turn.ultrafast_live_thread = threading.Thread(
                        target=self._live_worker,
                        args=(
                            turn.turn_id,
                            turn.generation,
                            "ultrafast",
                            turn.ultrafast_live_queue,
                            turn.ultrafast_live_stream,
                            turn.ultrafast_live_done,
                            turn.live_cancelled,
                        ),
                        name=(
                            f"RealtimeSTTProductionUltrafast-{self.session_id}-"
                            f"{turn.turn_id}"
                        ),
                        daemon=True,
                    )
                    threads.append(turn.ultrafast_live_thread)
                turn.phase = "receiving"

        if not current:
            return False
        try:
            for live_thread in threads:
                live_thread.start()
        except Exception:
            turn.live_cancelled.set()
            self._stop_live_stream(turn)
            raise
        return True

    def _run_live_stream_start(
        self,
        result: concurrent.futures.Future,
        streaming_worker: Any,
        turn_id: str,
        generation: int,
        language: str,
    ) -> None:
        """Create, transfer, or reap both streams without an ASGI executor."""

        stream = None
        ultrafast_stream = None
        installed = False
        try:
            worker = streaming_worker("realtime")
            stream = worker.create_streaming_session(
                language=language,
                use_prompt=False,
            )
            if self._ultrafast_enabled:
                ultrafast_worker = streaming_worker("ultrafast")
                ultrafast_stream = ultrafast_worker.create_streaming_session(
                    language=language,
                    use_prompt=False,
                )
            installed = self._install_live_stream(
                turn_id,
                generation,
                stream,
                ultrafast_stream,
            )
            if not installed:
                self._close_live_stream(stream)
                self._close_live_stream(ultrafast_stream)
            with self._lock:
                owned = (
                    installed
                    and self.turn is not None
                    and self.turn.turn_id == turn_id
                    and self.turn.generation == generation
                    and self.turn.live_stream is stream
                    and self.turn.ultrafast_live_stream is ultrafast_stream
                )
            if not result.done():
                result.set_result(owned)
        except Exception as exc:
            if not installed:
                if stream is not None:
                    self._close_live_stream(stream)
                if ultrafast_stream is not None:
                    self._close_live_stream(ultrafast_stream)
            if not result.done():
                result.set_exception(exc)
        finally:
            _LIVE_STREAM_OPERATION_SLOTS.release()
            with self._lock:
                self._live_stream_operation_threads.discard(threading.current_thread())
    def _begin_live_stream_start(
        self,
        streaming_worker: Any,
        turn_id: str,
        generation: int,
        language: str,
    ) -> concurrent.futures.Future:
        if not _LIVE_STREAM_OPERATION_SLOTS.acquire(blocking=False):
            LOGGER.warning("Production live stream startup capacity is exhausted")
            raise RuntimeError("Production live stream startup capacity is exhausted")
        result: concurrent.futures.Future = concurrent.futures.Future()
        thread = threading.Thread(
            target=self._run_live_stream_start,
            args=(result, streaming_worker, turn_id, generation, language),
            name=f"RealtimeSTTProductionLiveCreate-{self.session_id}-{turn_id}",
            daemon=True,
        )
        with self._lock:
            self._live_stream_operation_threads.add(thread)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._live_stream_operation_threads.discard(thread)
            _LIVE_STREAM_OPERATION_SLOTS.release()
            raise
        return result

    def _retire_starting_turn(self, turn_id: str, generation: int) -> None:
        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.turn_id != turn_id
                or turn.generation != generation
            ):
                return
            turn.cancelled.set()
            self.turn = None
            self.manager.set_turn(self.session_id, None)
        self._stop_live_stream(turn)

    async def _start_live_stream(
        self,
        turn_id: str,
        generation: int,
        language: str,
    ) -> bool:
        streaming_worker = getattr(
            getattr(self.service, "scheduler", None),
            "streaming_worker",
            None,
        )
        if not callable(streaming_worker):
            with self._lock:
                turn = self.turn
                if (
                    self.closed
                    or turn is None
                    or turn.turn_id != turn_id
                    or turn.generation != generation
                    or turn.phase != "starting"
                ):
                    return False
                turn.live_done.set()
                turn.ultrafast_live_done.set()
                turn.phase = "receiving"
            return True

        # Native creation and stale-stream reaping have one bounded worker
        # slot.  There is deliberately no executor queue behind this limit.
        result = self._begin_live_stream_start(
            streaming_worker,
            turn_id,
            generation,
            language,
        )
        completion = asyncio.wrap_future(result)
        try:
            return bool(await asyncio.shield(completion))
        except asyncio.CancelledError:
            def consume_late_exception(future: asyncio.Future) -> None:
                if not future.cancelled():
                    future.exception()

            completion.add_done_callback(consume_late_exception)
            # The operation keeps ownership of a late native result.  Retire
            # the turn now so it either reaps the result itself or the normal
            # stop path owns an already-installed stream.
            self._retire_starting_turn(turn_id, generation)
            raise

    def _live_worker(
        self,
        turn_id: str,
        generation: int,
        lane: str,
        live_queue: Any,
        stream: Any,
        live_done: threading.Event,
        live_cancelled: threading.Event,
    ) -> None:
        # The streaming engines return text without the input endpoint that
        # produced it.  Keep that endpoint at the protocol boundary instead
        # of guessing from the turn's concurrently accepted PCM: a decoder
        # can be behind the websocket receiver when Resume arrives.
        audio_end_sample_exclusive = 0
        try:
            affinity_setting = (
                "ultrafast_realtime_cpu_affinity"
                if lane == "ultrafast"
                else "realtime_cpu_affinity"
            )
            set_current_thread_cpu_affinity(
                f"{lane} live",
                getattr(self.settings, affinity_setting, None),
            )
            while True:
                samples = live_queue.get()
                try:
                    if samples is _LIVE_CANCEL or live_cancelled.is_set():
                        return
                    if samples is None:
                        stream.input_finished()
                        result = stream.finish()
                        if live_cancelled.is_set():
                            return
                        self._observe_realtime_result(
                            turn_id,
                            generation,
                            lane,
                            result,
                            audio_end_sample_exclusive=audio_end_sample_exclusive,
                        )
                        return
                    decode_started = time.monotonic()
                    stream.accept_audio(samples, sample_rate=SERVER_SAMPLE_RATE)
                    try:
                        sample_count = int(getattr(samples, "size"))
                    except (AttributeError, TypeError, ValueError):
                        sample_count = len(samples)
                    audio_end_sample_exclusive += max(0, sample_count)
                    stream.decode()
                    if live_cancelled.is_set():
                        return
                    decode_done = time.monotonic()
                    with self._lock:
                        turn = self.turn
                        if (
                            turn is not None
                            and turn.turn_id == turn_id
                            and turn.generation == generation
                        ):
                            if lane == "ultrafast":
                                first_key = "firstUltrafastDecodeStartedAt"
                                last_key = "lastUltrafastDecodeDoneAt"
                                calls_key = "ultrafastDecodeCalls"
                                seconds_key = "ultrafastDecodeSeconds"
                            else:
                                first_key = "firstDecodeStartedAt"
                                last_key = "lastDecodeDoneAt"
                                calls_key = "decodeCalls"
                                seconds_key = "decodeSeconds"
                            turn.telemetry.setdefault(first_key, decode_started)
                            turn.telemetry[last_key] = decode_done
                            turn.telemetry[calls_key] = int(
                                turn.telemetry.get(calls_key, 0)
                            ) + 1
                            turn.telemetry[seconds_key] = float(
                                turn.telemetry.get(seconds_key, 0.0)
                            ) + (decode_done - decode_started)
                    self._observe_realtime_result(
                        turn_id,
                        generation,
                        lane,
                        stream.get_result(),
                        audio_end_sample_exclusive=audio_end_sample_exclusive,
                    )
                finally:
                    live_queue.task_done()
        except Exception as exc:
            LOGGER.exception(
                "Production %s live stream failed for turn %s",
                lane,
                turn_id,
            )
            self._publish_live_lane_failure(
                turn_id,
                generation,
                lane,
                live_cancelled,
                exc,
            )
        finally:
            while True:
                try:
                    live_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    live_queue.task_done()
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    LOGGER.debug(
                        "Could not close production %s live stream",
                        lane,
                        exc_info=True,
                    )
            live_done.set()

    def _publish_live_lane_failure(
        self,
        turn_id: str,
        generation: int,
        lane: str,
        live_cancelled: threading.Event,
        exc: Exception,
    ) -> None:
        """Surface an unexpected live-lane exit while keeping final fallback."""

        if live_cancelled.is_set():
            return
        message = str(exc) or exc.__class__.__name__
        failure = _structured_error(
            "live_lane_failed",
            f"Production {lane} live transcription degraded: {message}",
            session_id=self.session_id,
            turn_id=turn_id,
            details={"lane": lane, "degraded": True},
        )
        # The connection manager's generic error decoration preserves
        # ``where`` in nested error details.  Keep the lane/degraded markers
        # top-level too, because lightweight test and non-manager consumers
        # receive the structured event directly.
        failure["where"] = f"live_{lane}"
        failure["lane"] = lane
        failure["degraded"] = True
        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.turn_id != turn_id
                or turn.generation != generation
                or turn.cancelled.is_set()
                or live_cancelled.is_set()
            ):
                return
            lane_status = dict(turn.telemetry.get("liveLaneStatus") or {})
            lane_status[lane] = "failed"
            turn.telemetry["liveLaneStatus"] = lane_status
            failures = list(turn.telemetry.get("liveLaneFailures") or [])
            failures.append(
                {
                    "lane": lane,
                    "code": "live_lane_failed",
                    "message": message,
                }
            )
            turn.telemetry["liveLaneFailures"] = failures[-8:]
            if turn.connection_epoch is not None:
                failure["_connectionEpoch"] = turn.connection_epoch
        publish = getattr(self.manager, "publish_session", None)
        if not callable(publish):
            return
        try:
            publish(self.session_id, failure, authoritative=True)
        except TypeError:
            publish(self.session_id, failure)
        except Exception:
            LOGGER.exception(
                "Could not publish live %s lane failure for turn %s",
                lane,
                turn_id,
            )

    def _observe_realtime_result(
        self,
        turn_id: str,
        generation: int,
        lane: str,
        result: Any,
        *,
        audio_end_sample_exclusive: Optional[int] = None,
    ) -> None:
        """Publish one observation batch without cross-lane overtaking."""

        with self._publish_lock:
            self._observe_realtime_result_ordered(
                turn_id,
                generation,
                lane,
                result,
                audio_end_sample_exclusive=audio_end_sample_exclusive,
            )

    @staticmethod
    def _live_resume_provenance(
        turn: TurnState,
        audio_end_sample_exclusive: Any,
    ) -> Dict[str, Any]:
        """Return the immutable Resume boundary that owned one live result."""

        audio_end = (
            int(audio_end_sample_exclusive)
            if type(audio_end_sample_exclusive) is int
            and audio_end_sample_exclusive >= 0
            else None
        )
        provenance: Dict[str, Any] = {
            "resumeEpoch": 0,
            "candidateId": None,
            "resumeId": None,
            "resumeRequestId": None,
            "candidateStartSample": 0,
            "audioEndSampleExclusive": audio_end,
        }
        if audio_end is None:
            return provenance
        for start_sample, epoch, request_id, candidate_id in reversed(
            turn.resume_provenance
        ):
            if audio_end > start_sample:
                provenance.update(
                    {
                        "resumeEpoch": int(epoch),
                        "candidateId": candidate_id,
                        "resumeId": request_id,
                        "resumeRequestId": request_id,
                        "candidateStartSample": int(start_sample),
                    }
                )
                break
        return provenance

    @staticmethod
    def _replace_preview_token_locked(
        turn: TurnState,
    ) -> tuple[int, threading.Event]:
        """Cancel an older Preview snapshot and mint a non-global token."""

        turn.preview_cancelled.set()
        turn.preview_epoch += 1
        turn.preview_cancelled = threading.Event()
        return turn.preview_epoch, turn.preview_cancelled

    def _preview_worker_can_admit(
        self,
        turn_id: str,
        generation: int,
        request_id: str,
        preview_epoch: int,
        cancelled: threading.Event,
    ) -> bool:
        """Check cancellation immediately before a Preview occupies ASR work."""

        if cancelled.is_set():
            return False
        with self._lock:
            turn = self.turn
            return bool(
                turn is not None
                and turn.turn_id == turn_id
                and turn.generation == generation
                and not turn.cancelled.is_set()
                and not cancelled.is_set()
                and turn.preview_epoch == preview_epoch
                and turn.latest_preview_request_id == request_id
            )

    def _observe_realtime_result_ordered(
        self,
        turn_id: str,
        generation: int,
        lane: str,
        result: Any,
        *,
        audio_end_sample_exclusive: Optional[int] = None,
    ) -> None:
        text = " ".join(str(getattr(result, "text", result) or "").split())
        if not text:
            return
        now = time.monotonic()
        raw_payload = None
        partial_payload = None
        with self._lock:
            turn = self.turn
            if turn is None or turn.turn_id != turn_id or turn.generation != generation:
                return
            recording_id = (turn_id, generation)
            if self._last_ultrafast_recording_id != recording_id:
                self._last_ultrafast_recording_id = recording_id
                self._last_ultrafast_text = ""
            accurate_changed = False
            observation_kwargs: Dict[str, Any] = {"recording_id": recording_id}
            if (
                type(audio_end_sample_exclusive) is int
                and audio_end_sample_exclusive >= 0
            ):
                observation_kwargs["audio_end_sample_exclusive"] = (
                    audio_end_sample_exclusive
                )
            if lane == "ultrafast":
                merge = self._realtime_merger.observe_ultrafast(
                    text,
                    **observation_kwargs,
                )
                # The merger returns its current accepted raw hypothesis.  A
                # stale sequenced result must not leak as a new wire event.
                if merge.ultrafast_text == text and text != self._last_ultrafast_text:
                    self._last_ultrafast_text = text
                    raw_payload = {
                        "type": "ultrafast",
                        "turnId": turn_id,
                        "text": merge.ultrafast_text,
                        "ultrafastText": merge.ultrafast_text,
                        "slowText": merge.slow_text,
                        "accurateText": merge.slow_text,
                        "mergedText": merge.text,
                        "ultrafastSuffix": merge.ultrafast_suffix,
                        "mergeStatus": merge.status,
                        "mergeMatched": merge.matched,
                        "mergeHeld": merge.held,
                        "mergeUsedFuzzyMatch": merge.used_fuzzy_match,
                        "mergeAnchorLength": merge.anchor_length,
                        "mergeDistance": merge.distance,
                        "mergeSlowGeneration": merge.slow_generation,
                        "mergeSlowSequence": merge.slow_sequence,
                        "mergeUltrafastSequence": merge.ultrafast_sequence,
                        "mergeSlowAudioEndSampleExclusive": (
                            merge.slow_audio_end_sample_exclusive
                        ),
                        "mergeUltrafastAudioEndSampleExclusive": (
                            merge.ultrafast_audio_end_sample_exclusive
                        ),
                    }
                    raw_payload.update(
                        self._live_resume_provenance(
                            turn,
                            merge.ultrafast_audio_end_sample_exclusive,
                        )
                    )
                    if turn.connection_epoch is not None:
                        raw_payload["_connectionEpoch"] = turn.connection_epoch
                should_publish_partial = bool(
                    merge.slow_text and merge.should_publish
                )
            else:
                accurate_changed = text != self._last_partial
                self._last_partial = text
                merge = self._realtime_merger.observe_slow(
                    text,
                    **observation_kwargs,
                )
                should_publish_partial = bool(
                    merge.slow_text
                    and (accurate_changed or merge.should_publish)
                )

            if (
                should_publish_partial
                and merge.text == self._last_merged_text
                and not accurate_changed
            ):
                should_publish_partial = False

            if should_publish_partial:
                self._last_merged_text = merge.text
                self._last_partial_sent_at = now
                turn.partial_count += 1
                turn.telemetry.setdefault("firstPartialSentAt", now)
                turn.telemetry["lastPartialSentAt"] = now
                partial_payload = {
                    "type": "realtime",
                    "turnId": turn_id,
                    "text": merge.slow_text,
                    "accurateText": merge.slow_text,
                    "slowText": merge.slow_text,
                    "ultrafastText": merge.ultrafast_text,
                    "mergedText": merge.text,
                    "ultrafastSuffix": merge.ultrafast_suffix,
                    "mergeStatus": merge.status,
                    "mergeMatched": merge.matched,
                    "mergeHeld": merge.held,
                    "mergeUsedFuzzyMatch": merge.used_fuzzy_match,
                    "mergeAnchorLength": merge.anchor_length,
                    "mergeDistance": merge.distance,
                    "mergeSlowGeneration": merge.slow_generation,
                    "mergeSlowSequence": merge.slow_sequence,
                    "mergeUltrafastSequence": merge.ultrafast_sequence,
                    "mergeSlowAudioEndSampleExclusive": (
                        merge.slow_audio_end_sample_exclusive
                    ),
                    "mergeUltrafastAudioEndSampleExclusive": (
                        merge.ultrafast_audio_end_sample_exclusive
                    ),
                }
                partial_payload.update(
                    self._live_resume_provenance(
                        turn,
                        merge.slow_audio_end_sample_exclusive,
                    )
                )
                if turn.connection_epoch is not None:
                    partial_payload["_connectionEpoch"] = turn.connection_epoch

        # Preserve callback/event order: a raw fast observation is visible
        # before the corresponding merged partial, while both are produced
        # outside the protocol lock so transport backpressure cannot block the
        # ASR worker's state machine.
        if raw_payload is not None:
            self.manager.publish_session(
                self.session_id,
                raw_payload,
            )
        if partial_payload is not None:
            self.manager.publish_session(
                self.session_id,
                partial_payload,
            )

    def _publish_changed_partial(
        self,
        turn_id: str,
        generation: int,
        result: Any,
    ) -> None:
        """Backward-compatible internal entry point for the accurate lane."""

        self._observe_realtime_result(
            turn_id,
            generation,
            "realtime",
            result,
        )
    def touch(self) -> None:
        with self._lock:
            if self.turn is not None:
                self.turn.last_activity = time.monotonic()

    def _error(self, code: str, message: str, details=None) -> Dict[str, Any]:
        turn_id = self.turn.turn_id if self.turn else None
        return _structured_error(
            code,
            message,
            session_id=self.session_id,
            turn_id=turn_id,
            details=details,
        )

    async def send_error(self, code: str, message: str, details=None) -> None:
        await self.manager.emit(self.session_id, self._error(code, message, details))

    async def start(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.turn is not None and self.turn.phase in (
                "starting",
                "receiving",
                "draining",
                "final_submitted",
                "terminal_result",
            ):
                return self._error("turn_in_progress", "A turn is already active", {
                    "turnId": self.turn.turn_id,
                    "phase": self.turn.phase,
                })
            turn_id = _turn_id(payload.get("turnId")) or uuid.uuid4().hex
            language = payload.get("language", getattr(self.settings, "language", "en"))
            language_error = _language_error(language, self.settings)
            if language_error:
                return self._error(
                    language_error["code"],
                    language_error["message"],
                    language_error.get("details"),
                )
            self._generation += 1
            generation = self._generation
            connection_epoch = getattr(self.manager, "connection_epoch", None)
            self.turn = TurnState(
                turn_id=turn_id,
                language=language.strip().lower(),
                phase="starting",
                generation=generation,
                connection_epoch=(
                    int(connection_epoch(self.session_id))
                    if callable(connection_epoch)
                    else None
                ),
            )
            if hasattr(self.session, "generation"):
                self.session.generation = generation
            self.manager.set_turn(self.session_id, turn_id)
            suppress_type = getattr(self.manager, "suppress_type", None)
            if callable(suppress_type):
                suppress_type(self.session_id, "final_outcome")
            self.touch()

        try:
            if hasattr(self.session, "settings"):
                self.session.settings.language = language.strip().lower()
            recorder = getattr(self.session, "recorder", None)
            if recorder is not None and hasattr(recorder, "language"):
                recorder.language = language.strip().lower()
            realtime_executor = getattr(
                recorder,
                "realtime_transcription_executor",
                None,
            )
            if realtime_executor is not None:
                set_language = getattr(realtime_executor, "set_streaming_language", None)
                if callable(set_language):
                    set_language(language.strip().lower())
            if not await self._start_live_stream(
                turn_id,
                generation,
                language.strip().lower(),
            ):
                return _structured_error(
                    "start_cancelled",
                    "Turn was retired while the live stream was starting",
                    session_id=self.session_id,
                    turn_id=turn_id,
                )
        except Exception as exc:
            with self._lock:
                current = self.turn
                active = (
                    current is not None
                    and current.turn_id == turn_id
                    and current.generation == generation
                )
                if active:
                    self.turn = None
                    self.manager.set_turn(self.session_id, None)
            if not active:
                return _structured_error(
                    "start_cancelled",
                    "Turn was retired while the live stream was starting",
                    session_id=self.session_id,
                    turn_id=turn_id,
                )
            return _structured_error(
                "start_failed",
                str(exc),
                session_id=self.session_id,
                turn_id=turn_id,
            )
        return {
            "type": "started",
            "sessionId": self.session_id,
            "turnId": turn_id,
            "language": language.strip().lower(),
            "expectedAudioSequence": 0,
        }

    async def audio(self, message: bytes) -> Optional[Dict[str, Any]]:
        with self._lock:
            turn = self.turn
            if turn is None:
                return self._error("turn_not_started", "Send start before audio")
            if turn.phase != "receiving":
                return self._error("turn_not_active", f"Turn is {turn.phase}")
        try:
            packet = decode_audio_packet(message)
            sequence, frames, sample_rate = _validate_production_packet(packet, self.settings)
        except AudioPacketError as exc:
            return self._error("invalid_audio", str(exc))

        if sample_rate != SERVER_SAMPLE_RATE:
            return self._error(
                "invalid_sample_rate",
                f"Production WebSocket audio must use {SERVER_SAMPLE_RATE} Hz canonical PCM",
                {"expected": SERVER_SAMPLE_RATE, "received": sample_rate},
            )

        with self._lock:
            turn = self.turn
            if turn is None or turn.phase != "receiving":
                return self._error("turn_not_active", "Turn is no longer active")
            if sequence != turn.expected_audio_sequence:
                return self._error(
                    "audio_sequence_out_of_order",
                    "Audio packets must use contiguous audioSequence values starting at zero",
                    {
                        "expected": turn.expected_audio_sequence,
                        "received": sequence,
                    },
                )
            duration = frames / float(sample_rate)
            if turn.audio_seconds + duration > self.settings.max_turn_audio_seconds:
                return self._error(
                    "audio_duration_limit",
                    "The turn audio duration limit was exceeded",
                    {
                        "maxSeconds": self.settings.max_turn_audio_seconds,
                        "receivedSeconds": turn.audio_seconds + duration,
                    },
                )
            if turn.live_queue is not None:
                import numpy as np

                samples = (
                    np.frombuffer(packet.audio, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                live_queues = [turn.live_queue]
                if turn.ultrafast_live_queue is not None:
                    live_queues.append(turn.ultrafast_live_queue)
                if any(
                    not live_queue.can_put(samples)
                    for live_queue in live_queues
                ):
                    return self._error(
                        "backpressure",
                        "Production live audio queue is full",
                    )
                try:
                    queued_at = time.monotonic()
                    for live_queue in live_queues:
                        # Every lane owns its own array. A streaming backend
                        # may mutate the received buffer during decoding.
                        live_queue.put_nowait(samples.copy())
                except queue.Full:
                    return self._error(
                        "backpressure",
                        "Production live audio queue is full",
                    )
                turn.telemetry.setdefault("firstQueuedAt", queued_at)
                turn.telemetry["lastQueuedAt"] = queued_at
            turn.pcm_buffer.extend(packet.audio)
            received_at = time.monotonic()
            turn.telemetry.setdefault("firstReceivedAt", received_at)
            turn.telemetry["lastReceivedAt"] = received_at
            if turn.first_audio_sequence is None:
                turn.first_audio_sequence = sequence
            turn.last_audio_sequence = sequence
            self.manager.set_audio_sequence(self.session_id, sequence)
            turn.expected_audio_sequence += 1
            turn.packet_count += 1
            turn.audio_revision += 1
            turn.audio_frames += frames
            turn.audio_seconds += duration
            turn.last_activity = time.monotonic()
        return None

    @staticmethod
    def _resume_int(payload: Dict[str, Any], *keys: str) -> tuple[Optional[int], Optional[str]]:
        """Read one optional non-negative integer from a resume command."""

        for key in keys:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None, key
            return int(value), None
        return None, None

    def _resume_request_id(self, payload: Dict[str, Any], turn: Optional[TurnState]) -> tuple[str, bool]:
        """Return the canonical correlation id, accepting transition aliases."""

        raw = payload.get("resumeId")
        if raw is None:
            raw = payload.get("requestId")
        if raw is None:
            # This was accepted by an early Preview implementation before the
            # resume command acquired its canonical ``resumeId`` field.
            raw = payload.get("resumeRequestId")
        if raw is None:
            turn_id = "none" if turn is None else turn.turn_id
            count = 1 if turn is None else turn.resume_count + 1
            return f"resume-{turn_id}-{count}", True
        request_id = _turn_id(raw)
        return (request_id or "resume-invalid", False)

    def _resume_error(
        self,
        request_id: str,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error = self._error(code, message, details)
        # ``resumeId`` is canonical; retain ``requestId`` for transitional
        # clients and correlate both names to the same boundary.
        error["requestId"] = request_id
        error["resumeId"] = request_id
        return error

    def _queue_resume_ack_locked(
        self,
        ack: Dict[str, Any],
    ) -> Optional[concurrent.futures.Future]:
        """Reserve the ordered ACK slot before live workers see the candidate."""

        queue_resume_ack = getattr(self.manager, "queue_resume_ack", None)
        if not callable(queue_resume_ack):
            return None
        return queue_resume_ack(self.session_id, ack)

    def take_queued_resume_ack_delivery(
        self,
        response: Dict[str, Any],
    ) -> Optional[concurrent.futures.Future]:
        """Return the ACK delivery already reserved by :meth:`resume` once."""

        if response.get("type") != RESUME_ACK_TYPE:
            return None
        response_resume_id = response.get("resumeId")
        if not isinstance(response_resume_id, str):
            return None
        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.last_resume_request_id != response_resume_id
            ):
                return None
            delivery = turn.last_resume_ack_delivery
            turn.last_resume_ack_delivery = None
            return delivery

    async def resume(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mark the exact accepted-PCM endpoint for a newly resumed candidate.

        Resume never clears or rewinds the logical turn.  The next Preview can
        therefore transcribe only the bytes accepted after this boundary while
        the cumulative buffer remains available for the eventual Final/late
        correction path.
        """

        with self._lock:
            current_turn = self.turn
            request_id, _generated_request_id = self._resume_request_id(
                payload,
                current_turn,
            )
            raw_resume_id = payload.get("resumeId")
            raw_request_id = payload.get("requestId")
            normalized_resume_id = (
                _turn_id(raw_resume_id) if raw_resume_id is not None else None
            )
            normalized_request_id = (
                _turn_id(raw_request_id) if raw_request_id is not None else None
            )
            if raw_resume_id is not None and normalized_resume_id is None:
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    "resumeId must be a non-empty string of at most 128 characters",
                )
            if raw_request_id is not None and normalized_request_id is None:
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    "requestId must be a non-empty string of at most 128 characters",
                )
            if (
                normalized_resume_id is not None
                and normalized_request_id is not None
                and normalized_resume_id != normalized_request_id
            ):
                return self._resume_error(
                    normalized_resume_id,
                    "resume_correlation_mismatch",
                    "resumeId and requestId must match when both are supplied",
                    {
                        "resumeId": normalized_resume_id,
                        "requestId": normalized_request_id,
                    },
                )
            raw_legacy_request_id = payload.get("resumeRequestId")
            if (
                raw_resume_id is None
                and raw_request_id is None
                and raw_legacy_request_id is not None
                and not _turn_id(raw_legacy_request_id)
            ):
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    "resumeId must be a non-empty string of at most 128 characters",
                )
            if current_turn is None:
                return self._resume_error(
                    request_id,
                    "turn_not_started",
                    "Send start before resume",
                )
            if current_turn.phase != "receiving":
                return self._resume_error(
                    request_id,
                    "turn_not_active",
                    f"Turn is {current_turn.phase}",
                )

            requested_turn_id = payload.get("turnId")
            if requested_turn_id is not None and requested_turn_id != current_turn.turn_id:
                return self._resume_error(
                    request_id,
                    "stale_turn",
                    "Resume request does not belong to the active turn",
                    {"activeTurnId": current_turn.turn_id},
                )

            if (
                current_turn.last_resume_request_id == request_id
                and current_turn.last_resume_ack is not None
            ):
                ack = dict(current_turn.last_resume_ack)
                delivery = self._queue_resume_ack_locked(ack)
                if delivery is None:
                    return self._resume_error(
                        request_id,
                        "resume_ack_unavailable",
                        "The outbound queue cannot reserve a Resume acknowledgement",
                        {"reason": "outbound_backpressure"},
                    )
                current_turn.last_resume_ack_delivery = delivery
                return ack

            expected_sequence = current_turn.expected_audio_sequence
            audio_sequence, invalid_sequence_key = self._resume_int(
                payload,
                "audioSequence",
                "audio_sequence",
            )
            if invalid_sequence_key is not None:
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    f"{invalid_sequence_key} must be a non-negative integer",
                )
            if audio_sequence is not None and audio_sequence != expected_sequence:
                return self._resume_error(
                    request_id,
                    "audio_sequence_out_of_order",
                    "Resume audioSequence must name the next accepted packet",
                    {"expected": expected_sequence, "received": audio_sequence},
                )

            sample_offset, invalid_sample_key = self._resume_int(
                payload,
                "sampleOffset",
                "sample_offset",
                "startSample",
            )
            if invalid_sample_key is not None:
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    f"{invalid_sample_key} must be a non-negative integer",
                )

            byte_offset, invalid_byte_key = self._resume_int(
                payload,
                "byteOffset",
                "byte_offset",
                "startByte",
            )
            if invalid_byte_key is not None:
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    f"{invalid_byte_key} must be a non-negative integer",
                )

            buffered_samples = int(current_turn.audio_frames)
            buffered_bytes = len(current_turn.pcm_buffer)
            if buffered_bytes != buffered_samples * 2 or buffered_bytes % 2:
                return self._resume_error(
                    request_id,
                    "buffer_boundary_invalid",
                    "The server PCM buffer is not aligned to canonical PCM16 samples",
                    {
                        "bufferedSamples": buffered_samples,
                        "bufferedBytes": buffered_bytes,
                    },
                )
            if sample_offset is not None and sample_offset != buffered_samples:
                return self._resume_error(
                    request_id,
                    "sample_offset_mismatch",
                    "sampleOffset must equal the accepted PCM sample count",
                    {"expected": buffered_samples, "received": sample_offset},
                )
            if byte_offset is not None and byte_offset != buffered_bytes:
                return self._resume_error(
                    request_id,
                    "byte_offset_mismatch",
                    "byteOffset must equal the accepted PCM byte count",
                    {"expected": buffered_bytes, "received": byte_offset},
                )
            if sample_offset is not None and sample_offset * 2 != buffered_bytes:
                return self._resume_error(
                    request_id,
                    "buffer_boundary_invalid",
                    "sampleOffset and byteOffset do not describe the same PCM boundary",
                    {"sampleOffset": sample_offset, "bufferedBytes": buffered_bytes},
                )

            raw_candidate_id = payload.get("candidateId")
            if raw_candidate_id is None:
                raw_candidate_id = payload.get("candidateSerial")
            if raw_candidate_id is None:
                candidate_id = f"candidate-{current_turn.resume_count + 1}"
            elif isinstance(raw_candidate_id, bool):
                return self._resume_error(
                    request_id,
                    "invalid_resume_request",
                    "candidateId must be a non-empty string or integer",
                )
            else:
                candidate_id = _turn_id(str(raw_candidate_id))
                if candidate_id is None:
                    return self._resume_error(
                        request_id,
                        "invalid_resume_request",
                        "candidateId must be a non-empty string of at most 128 characters",
                    )

            # Freeze the authoritative Live snapshot for diagnostics only;
            # speculative ultrafast suffixes are never copied into it, and
            # Preview never uses it to construct the model result.
            candidate_base_text = " ".join(self._last_partial.split())
            resume_epoch = current_turn.resume_count + 1

            # Keep the previous authoritative hypotheses as Live-event dedupe
            # anchors. The native stream remains cumulative, so clearing them
            # would re-deliver an unchanged old sentence to the new candidate.

            ack = {
                "type": RESUME_ACK_TYPE,
                "sessionId": self.session_id,
                "turnId": current_turn.turn_id,
                "requestId": request_id,
                "resumeId": request_id,
                "candidateId": candidate_id,
                "resumeEpoch": resume_epoch,
                "accepted": True,
                "audioSequence": expected_sequence,
                "expectedAudioSequence": expected_sequence,
                "sampleOffset": buffered_samples,
                "byteOffset": buffered_bytes,
                "candidateStartSample": buffered_samples,
                "candidateStartByte": buffered_bytes,
                "bufferedSamples": buffered_samples,
                "bufferedBytes": buffered_bytes,
                "inputScope": "candidate",
                "inputSampleRange": {
                    "start": buffered_samples,
                    "end": buffered_samples,
                },
                "inputByteRange": {
                    "start": buffered_bytes,
                    "end": buffered_bytes,
                },
            }
            # Queue the ACK while holding the protocol lock. A live worker
            # cannot form a post-Resume candidate payload until this lock is
            # released, so its first candidate event is necessarily ordered
            # after the ACK reservation. Do not publish a new candidate
            # boundary unless that reservation actually exists.
            delivery = self._queue_resume_ack_locked(ack)
            if delivery is None:
                return self._resume_error(
                    request_id,
                    "resume_ack_unavailable",
                    "The outbound queue cannot reserve a Resume acknowledgement",
                    {"reason": "outbound_backpressure"},
                )

            current_turn.candidate_base_text = candidate_base_text
            current_turn.resume_count = resume_epoch
            current_turn.resume_epoch = resume_epoch
            current_turn.candidate_id = candidate_id
            current_turn.candidate_start_sample = buffered_samples
            current_turn.candidate_start_byte = buffered_bytes
            current_turn.last_resume_request_id = request_id
            current_turn.resume_provenance.append(
                (
                    buffered_samples,
                    resume_epoch,
                    request_id,
                    candidate_id,
                )
            )
            if len(current_turn.resume_provenance) > _MAX_RESUME_PROVENANCE:
                del current_turn.resume_provenance[:-_MAX_RESUME_PROVENANCE]
            # A Preview captured before this boundary is not a valid result
            # for the resumed candidate, even when no new PCM has arrived yet.
            # Its worker is fenced both at admission and publication.  This
            # token is intentionally separate from ``cancelled`` so Resume
            # does not interrupt the shared realtime stream.
            self._replace_preview_token_locked(current_turn)
            current_turn.preview_requested = False
            current_turn.latest_preview_request_id = None
            current_turn.latest_preview_audio_revision = None
            current_turn.preview_completed_request_id = None
            current_turn.preview_completed_audio_revision = None
            current_turn.preview_status = ""
            current_turn.preview_failure = None
            current_turn.last_activity = time.monotonic()
            current_turn.last_resume_ack = dict(ack)
            current_turn.last_resume_ack_delivery = delivery
            return ack

    def _preview_selection(
        self,
        turn: TurnState,
        live_text: str,
    ) -> tuple[
        bytes,
        str,
        int,
        int,
        int,
        int,
        Optional[str],
        str,
        Optional[str],
        int,
    ]:
        """Freeze one Preview input and its absolute PCM ranges.

        The returned sample/byte endpoints are exclusive.  Preview always
        transcribes the complete retained logical-turn buffer.  A Resume
        boundary still fences request ownership and is reported separately;
        it never replaces the Preview model's full-turn input with Live text.
        """

        full_pcm = bytes(turn.pcm_buffer)
        candidate_active = turn.candidate_id is not None
        candidate_start_byte = (
            int(turn.candidate_start_byte) if candidate_active else 0
        )
        candidate_start_byte = max(0, min(candidate_start_byte, len(full_pcm)))
        candidate_start_byte -= candidate_start_byte % 2
        candidate_start_sample = candidate_start_byte // 2
        # Preview ASR is the transcript source.  Do not shorten its input just
        # because Live ASR happened to emit a partial hypothesis; doing so
        # turns Live/Preview alignment into a correctness dependency.
        input_pcm = full_pcm
        base_scope = "full_buffer"
        input_start_sample = 0
        input_end_sample = len(full_pcm) // 2
        if candidate_active:
            # ``candidate`` remains the stable correlation domain.  The event
            # explicitly marks that its ASR text covers the full logical turn.
            input_scope = "candidate"
        else:
            input_scope = base_scope
            input_start_sample = max(0, input_start_sample)
        return (
            input_pcm,
            input_scope,
            input_start_sample,
            input_end_sample,
            candidate_start_sample,
            candidate_start_byte,
            turn.candidate_id if candidate_active else None,
            turn.candidate_base_text if candidate_active else "",
            turn.last_resume_request_id if candidate_active else None,
            int(turn.resume_epoch) if candidate_active else 0,
        )

    def _start_preview_worker(
        self,
        turn_id: str,
        generation: int,
        language: str,
        input_pcm: bytes,
        input_scope: str,
        live_text: str,
        request_id: str,
        requested_at: float,
        audio_revision: int,
        audio_frames: int,
        audio_packets: int,
        input_sample_start: int,
        input_sample_end: int,
        candidate_start_sample: int,
        candidate_start_byte: int,
        candidate_id: Optional[str],
        candidate_base_text: str,
        resume_request_id: Optional[str],
        resume_epoch: int,
        preview_epoch: int,
        cancelled: threading.Event,
    ) -> None:
        work = (
            turn_id,
            generation,
            language,
            input_pcm,
            input_scope,
            live_text,
            request_id,
            requested_at,
            audio_revision,
            audio_frames,
            audio_packets,
            input_sample_start,
            input_sample_end,
            candidate_start_sample,
            candidate_start_byte,
            candidate_id,
            candidate_base_text,
            resume_request_id,
            resume_epoch,
            preview_epoch,
            cancelled,
        )
        thread_to_start = None
        with self._lock:
            # Preview is a latest-only snapshot operation.  Replacing this
            # pending tuple coalesces bursts while an older native inference
            # is still in flight; the preview epoch fences its publication.
            self._preview_pending = work
            thread = self._preview_dispatch_thread
            # The ownership marker is installed before ``start()`` runs and
            # cleared by the dispatcher's ``finally`` block.  Testing
            # ``is_alive()`` here reopens a start-window race: a second request
            # can observe the newly allocated but not-yet-started thread as
            # inactive and create a second dispatcher.
            if thread is None:
                thread = threading.Thread(
                    target=self._run_preview_dispatcher,
                    name=f"RealtimeSTTProductionPreview-{self.session_id}-{turn_id}",
                    daemon=True,
                )
                self._preview_dispatch_thread = thread
                self._completion_threads.add(thread)
                thread_to_start = thread
        if thread_to_start is None:
            return
        try:
            thread_to_start.start()
        except Exception:
            with self._lock:
                if self._preview_dispatch_thread is thread_to_start:
                    self._preview_dispatch_thread = None
                    self._preview_pending = None
                self._completion_threads.discard(thread_to_start)
            raise

    def _run_preview_dispatcher(self) -> None:
        """Run one latest-only Preview snapshot at a time."""

        current_thread = threading.current_thread()
        try:
            while True:
                with self._lock:
                    work = self._preview_pending
                    self._preview_pending = None
                    if work is None:
                        if self._preview_dispatch_thread is current_thread:
                            # Clear the ownership marker while holding the
                            # same lock as _start_preview_worker.  A request
                            # arriving immediately after this branch can then
                            # install a replacement dispatcher safely.
                            self._preview_dispatch_thread = None
                            self._completion_threads.discard(current_thread)
                        return
                try:
                    self._run_preview_worker(*work, _release_thread=False)
                except Exception:
                    # Keep the bounded dispatcher alive for a newer request;
                    # the worker normally converts inference failures into a
                    # Preview error event itself.
                    LOGGER.exception("Preview dispatcher worker failed")
        finally:
            with self._lock:
                if self._preview_dispatch_thread is current_thread:
                    self._preview_dispatch_thread = None
                self._completion_threads.discard(current_thread)

    def _run_preview_worker(
        self,
        turn_id: str,
        generation: int,
        language: str,
        input_pcm: bytes,
        input_scope: str,
        live_text: str,
        request_id: str,
        requested_at: float,
        audio_revision: int,
        audio_frames: int,
        audio_packets: int,
        input_sample_start: int,
        input_sample_end: int,
        candidate_start_sample: int,
        candidate_start_byte: int,
        candidate_id: Optional[str],
        candidate_base_text: str,
        resume_request_id: Optional[str],
        resume_epoch: int,
        preview_epoch: int,
        cancelled: threading.Event,
        _release_thread: bool = True,
    ) -> None:
        worker_started_at = time.monotonic()
        try:
            status = "empty"
            preview_text = ""
            tail_text = ""
            asr_queue_ms = None
            asr_inference_ms = 0.0
            asr_total_ms = 0.0
            native_timings_ms = None
            native_phase_total_ms = None
            asr_unattributed_ms = None
            asr_attempts = []
            empty_retry_attempted = False
            empty_retry_recovered = False
            empty_retry_error = None
            matched = False
            used_fuzzy_match = False
            anchor_length = 0
            distance = 0
            failure = None
            if not self._preview_worker_can_admit(
                turn_id,
                generation,
                request_id,
                preview_epoch,
                cancelled,
            ):
                return
            try:
                if input_pcm:
                    import numpy as np

                    audio = (
                        np.frombuffer(input_pcm, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    transcribe = getattr(self.service, "transcribe_turn", None)

                    def run_asr_attempt(
                        attempt_audio,
                        *,
                        attempt_index: int,
                        added_silence_ms: float,
                    ):
                        attempt_started_at = time.monotonic()
                        try:
                            if callable(transcribe):
                                if not self._preview_worker_can_admit(
                                    turn_id,
                                    generation,
                                    request_id,
                                    preview_epoch,
                                    cancelled,
                                ):
                                    raise _PreviewAdmissionCancelled()
                                attempt_result = transcribe(
                                    attempt_audio,
                                    language,
                                    False,
                                )
                            else:
                                attempt_result = _service_turn_transcription(
                                    self.service,
                                    attempt_audio,
                                    language,
                                    self.settings.finalize_timeout_seconds,
                                    self.session_id,
                                    generation,
                                    admission_check=lambda: self._preview_worker_can_admit(
                                        turn_id,
                                        generation,
                                        request_id,
                                        preview_epoch,
                                        cancelled,
                                    ),
                                    admission_lock=self._lock,
                                )
                        finally:
                            attempt_completed_at = time.monotonic()

                        measured_ms = round(
                            max(0.0, attempt_completed_at - attempt_started_at)
                            * 1000.0,
                            3,
                        )
                        queue_delay = getattr(attempt_result, "queue_delay", None)
                        inference_duration = getattr(
                            attempt_result,
                            "inference_duration",
                            None,
                        )
                        total_latency = getattr(
                            attempt_result,
                            "total_latency",
                            None,
                        )
                        queue_ms = (
                            None
                            if queue_delay is None
                            else round(max(0.0, float(queue_delay)) * 1000.0, 3)
                        )
                        inference_ms = (
                            measured_ms
                            if inference_duration is None
                            else round(
                                max(0.0, float(inference_duration)) * 1000.0,
                                3,
                            )
                        )
                        total_ms = (
                            measured_ms
                            if total_latency is None
                            else round(
                                max(0.0, float(total_latency)) * 1000.0,
                                3,
                            )
                        )
                        result_metadata = getattr(attempt_result, "metadata", None)
                        raw_native_timings = (
                            result_metadata.get("timings_ms")
                            if isinstance(result_metadata, dict)
                            else None
                        )
                        attempt_native_timings = None
                        attempt_native_phase_total_ms = None
                        attempt_unattributed_ms = None
                        if isinstance(raw_native_timings, dict):
                            attempt_native_timings = {}
                            for name in (
                                "load_ms",
                                "mel_ms",
                                "encode_ms",
                                "decode_ms",
                            ):
                                value = raw_native_timings.get(name)
                                if isinstance(value, (int, float)):
                                    attempt_native_timings[name] = round(
                                        max(0.0, float(value)),
                                        3,
                                    )
                            phase_values = [
                                attempt_native_timings[name]
                                for name in ("mel_ms", "encode_ms", "decode_ms")
                                if name in attempt_native_timings
                            ]
                            if phase_values:
                                attempt_native_phase_total_ms = round(
                                    sum(phase_values),
                                    3,
                                )
                                attempt_unattributed_ms = round(
                                    max(
                                        0.0,
                                        inference_ms
                                        - attempt_native_phase_total_ms,
                                    ),
                                    3,
                                )
                        raw_tokens = (
                            result_metadata.get("tokens")
                            if isinstance(result_metadata, dict)
                            else None
                        )
                        token_count = (
                            len(raw_tokens)
                            if isinstance(raw_tokens, (list, tuple))
                            else None
                        )
                        attempt_text = " ".join(
                            str(
                                getattr(
                                    attempt_result,
                                    "text",
                                    attempt_result,
                                )
                                or ""
                            ).split()
                        )
                        attempt_error = getattr(attempt_result, "error", None)
                        timing = {
                            "attempt": attempt_index,
                            "inputSeconds": (
                                len(attempt_audio) / float(SERVER_SAMPLE_RATE)
                            ),
                            "addedSilenceMs": float(added_silence_ms),
                            "queueMs": queue_ms,
                            "inferenceMs": inference_ms,
                            "totalMs": total_ms,
                            "nativeTimingsMs": attempt_native_timings,
                            "nativePhaseTotalMs": attempt_native_phase_total_ms,
                            "unattributedMs": attempt_unattributed_ms,
                            "tokenCount": token_count,
                            "textEmpty": not bool(attempt_text),
                        }
                        if attempt_error:
                            timing["error"] = str(attempt_error)
                        return attempt_text, timing, attempt_error

                    # A Resume can arrive while this worker was waiting for a
                    # thread slot. Recheck at the exact ASR admission point;
                    # do not put stale Preview audio onto the GPU queue.
                    if not self._preview_worker_can_admit(
                        turn_id,
                        generation,
                        request_id,
                        preview_epoch,
                        cancelled,
                    ):
                        return
                    tail_text, first_attempt, first_error = run_asr_attempt(
                        audio,
                        attempt_index=1,
                        added_silence_ms=0.0,
                    )
                    asr_attempts.append(first_attempt)
                    if first_error:
                        raise RuntimeError(str(first_error))

                    if not tail_text:
                        # The native transducer can occasionally return zero
                        # tokens when a snapshot ends directly on its final
                        # acoustic frame. Give every empty result exactly one
                        # bounded flush-only retry; the original PCM prefix
                        # remains byte-for-byte identical and no retry loop is
                        # possible.
                        if not self._preview_worker_can_admit(
                            turn_id,
                            generation,
                            request_id,
                            preview_epoch,
                            cancelled,
                        ):
                            return
                        silence_samples = int(
                            round(
                                PREVIEW_EMPTY_RETRY_SILENCE_SECONDS
                                * SERVER_SAMPLE_RATE
                            )
                        )
                        retry_audio = np.concatenate(
                            (
                                audio,
                                np.zeros(silence_samples, dtype=np.float32),
                            )
                        )
                        empty_retry_attempted = True
                        try:
                            retry_text, retry_attempt, retry_error = run_asr_attempt(
                                retry_audio,
                                attempt_index=2,
                                added_silence_ms=(
                                    PREVIEW_EMPTY_RETRY_SILENCE_SECONDS * 1000.0
                                ),
                            )
                        except Exception as exc:
                            empty_retry_error = str(exc)
                            LOGGER.warning(
                                "Preview empty-result silence retry failed: %s",
                                exc,
                            )
                        else:
                            asr_attempts.append(retry_attempt)
                            if retry_error:
                                empty_retry_error = str(retry_error)
                                LOGGER.warning(
                                    "Preview empty-result silence retry failed: %s",
                                    retry_error,
                                )
                            else:
                                tail_text = retry_text
                                empty_retry_recovered = bool(tail_text)

                    queue_values = [
                        float(attempt["queueMs"])
                        for attempt in asr_attempts
                        if isinstance(attempt.get("queueMs"), (int, float))
                    ]
                    asr_queue_ms = (
                        round(sum(queue_values), 3) if queue_values else None
                    )
                    asr_inference_ms = round(
                        sum(
                            float(attempt["inferenceMs"])
                            for attempt in asr_attempts
                            if isinstance(
                                attempt.get("inferenceMs"),
                                (int, float),
                            )
                        ),
                        3,
                    )
                    asr_total_ms = round(
                        sum(
                            float(attempt["totalMs"])
                            for attempt in asr_attempts
                            if isinstance(attempt.get("totalMs"), (int, float))
                        ),
                        3,
                    )
                    native_totals = {}
                    for attempt in asr_attempts:
                        attempt_native = attempt.get("nativeTimingsMs")
                        if not isinstance(attempt_native, dict):
                            continue
                        for name, value in attempt_native.items():
                            if isinstance(value, (int, float)):
                                native_totals[name] = (
                                    native_totals.get(name, 0.0) + float(value)
                                )
                    if native_totals:
                        native_timings_ms = {
                            name: round(value, 3)
                            for name, value in native_totals.items()
                        }
                    native_phase_values = [
                        float(attempt["nativePhaseTotalMs"])
                        for attempt in asr_attempts
                        if isinstance(
                            attempt.get("nativePhaseTotalMs"),
                            (int, float),
                        )
                    ]
                    if native_phase_values:
                        native_phase_total_ms = round(
                            sum(native_phase_values),
                            3,
                        )
                    unattributed_values = [
                        float(attempt["unattributedMs"])
                        for attempt in asr_attempts
                        if isinstance(
                            attempt.get("unattributedMs"),
                            (int, float),
                        )
                    ]
                    if unattributed_values:
                        asr_unattributed_ms = round(
                            sum(unattributed_values),
                            3,
                        )

                if tail_text:
                    preview_text = tail_text
                    status = "full_buffer"
                else:
                    status = "empty"
            except _PreviewAdmissionCancelled:
                return
            except Exception as exc:
                LOGGER.exception("Preview production transcription failed")
                preview_text = ""
                status = "error"
                failure = _structured_error(
                    "preview_transcription_failed",
                    str(exc),
                    session_id=self.session_id,
                    turn_id=turn_id,
                )

            # ``preview_text`` is the one full-turn inference result.  A
            # resumed event keeps candidate correlation metadata but does not
            # invent a candidate-only suffix from Live ASR.  New clients use
            # candidateCumulativeText plus candidateInputScope=full_turn;
            # legacy candidate-only fields remain present as empty strings.
            candidate_active = candidate_id is not None
            candidate_text = "" if candidate_active else None
            cumulative_text = preview_text
            cumulative_live_text = live_text

            publish_ready_at = time.monotonic()
            payload = {
                "type": "preview",
                "turnId": turn_id,
                "previewRequestId": request_id,
                "text": cumulative_text,
                "cumulativeText": cumulative_text,
                "liveText": cumulative_live_text,
                "tailText": tail_text,
                "previewModelText": tail_text,
                "previewText": preview_text,
                "status": status,
                "matched": matched,
                "usedFuzzyMatch": used_fuzzy_match,
                "anchorLength": anchor_length,
                "distance": distance,
                "audioRevision": audio_revision,
                "audioPackets": audio_packets,
                "audioFrames": audio_frames,
                "inputScope": input_scope,
                "previewInputCoverage": "full_turn",
                "inputSeconds": len(input_pcm) / float(SERVER_SAMPLE_RATE * 2),
                "inputSampleRange": {
                    "start": int(input_sample_start),
                    "end": int(input_sample_end),
                },
                "inputByteRange": {
                    "start": int(input_sample_start * 2),
                    "end": int(input_sample_end * 2),
                },
                "candidateId": candidate_id,
                "candidateBaseText": candidate_base_text if candidate_active else None,
                "candidateCumulativeText": (
                    cumulative_text if candidate_active else None
                ),
                "resumeId": resume_request_id,
                "resumeRequestId": resume_request_id,
                "resumeEpoch": resume_epoch if candidate_active else None,
                "candidateInputScope": (
                    "full_turn" if candidate_active else None
                ),
                "candidateStartSample": int(candidate_start_sample),
                "candidateStartByte": int(candidate_start_byte),
                "candidateSampleRange": (
                    {
                        "start": int(candidate_start_sample),
                        "end": int(audio_frames),
                    }
                    if candidate_active
                    else None
                ),
                "candidateByteRange": (
                    {
                        "start": int(candidate_start_byte),
                        "end": int(audio_frames * 2),
                    }
                    if candidate_active
                    else None
                ),
                "previewTiming": {
                    "requestToWorkerStartMs": round(
                        max(0.0, worker_started_at - requested_at) * 1000.0,
                        3,
                    ),
                    "asrQueueMs": asr_queue_ms,
                    "asrInferenceMs": asr_inference_ms,
                    "asrTotalMs": asr_total_ms,
                    "nativeTimingsMs": native_timings_ms,
                    "nativePhaseTotalMs": native_phase_total_ms,
                    "asrUnattributedMs": asr_unattributed_ms,
                    "asrAttemptCount": len(asr_attempts),
                    "asrAttempts": asr_attempts,
                    "emptyRetryAttempted": empty_retry_attempted,
                    "emptyRetryRecovered": empty_retry_recovered,
                    "emptyRetryReason": (
                        "empty_transcript"
                        if empty_retry_attempted
                        else None
                    ),
                    "emptyRetrySilenceMs": (
                        PREVIEW_EMPTY_RETRY_SILENCE_SECONDS * 1000.0
                        if empty_retry_attempted
                        else 0.0
                    ),
                    "emptyRetryError": empty_retry_error,
                    "requestToPublishMs": round(
                        max(0.0, publish_ready_at - requested_at) * 1000.0,
                        3,
                    ),
                },
            }
            if candidate_active:
                payload["candidateText"] = candidate_text
                payload["candidateOnlyText"] = candidate_text
            if failure is not None:
                payload["error"] = dict(
                    failure.get("error")
                    or {
                        "code": failure.get("code", "preview_transcription_failed"),
                        "message": failure.get(
                            "message", "Preview transcription failed"
                        ),
                    }
                )
            self._publish_preview_result(
                turn_id,
                generation,
                request_id,
                audio_revision,
                payload,
                status,
                failure,
                preview_epoch=preview_epoch,
                cancelled=cancelled,
            )
        finally:
            if _release_thread:
                with self._lock:
                    self._completion_threads.discard(threading.current_thread())

    def _complete_preview_only_turn(
        self,
        turn_id: str,
        generation: int,
    ) -> None:
        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.turn_id != turn_id
                or turn.generation != generation
                or turn.completion_sent
            ):
                return
            turn.terminal_sent = True
            turn.phase = "terminal_result"
            if turn.preview_failure is not None:
                status = "failed"
                failure = turn.preview_failure
            elif turn.preview_status == "empty":
                status = "no_speech"
                failure = None
            else:
                status = "completed"
                failure = None
            try:
                snapshot = self.session.snapshot()
            except Exception:
                snapshot = {}
            self._publish_completion(
                turn_id,
                generation,
                status,
                snapshot,
                failure=failure,
            )

    def _publish_preview_result(
        self,
        turn_id: str,
        generation: int,
        request_id: str,
        audio_revision: int,
        payload: Dict[str, Any],
        status: str,
        failure: Optional[Dict[str, Any]],
        *,
        preview_epoch: Optional[int] = None,
        cancelled: Optional[threading.Event] = None,
    ) -> None:
        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.turn_id != turn_id
                or turn.generation != generation
                or turn.cancelled.is_set()
                or (cancelled is not None and cancelled.is_set())
                or (
                    preview_epoch is not None
                    and turn.preview_epoch != preview_epoch
                )
            ):
                return
            # A resume changes candidate and correlation ownership without
            # changing the logical turn generation. Reject a worker snapshot
            # from an earlier boundary even if a caller reuses candidateId.
            payload_resume_id = payload.get("resumeId")
            if payload_resume_id is None:
                payload_resume_id = payload.get("resumeRequestId")
            if payload_resume_id is None:
                payload_resume_id = payload.get("requestId")
            if (
                payload.get("candidateId") != turn.candidate_id
                or payload_resume_id != turn.last_resume_request_id
            ):
                return
            turn.preview_count += 1
            is_latest = request_id == turn.latest_preview_request_id
            if turn.finalize_requested and not is_latest:
                return
            if is_latest:
                turn.preview_completed_request_id = request_id
                turn.preview_completed_audio_revision = audio_revision
                turn.preview_status = status
                turn.preview_failure = failure
            event = dict(payload)
            if turn.connection_epoch is not None:
                event["_connectionEpoch"] = turn.connection_epoch
            publish = getattr(self.manager, "publish_session")
            try:
                publish(self.session_id, event, authoritative=True)
            except TypeError:
                publish(self.session_id, event)
            should_complete = bool(
                turn.finalize_requested
                and is_latest
                and audio_revision == turn.audio_revision
            )
            if should_complete:
                self._complete_preview_only_turn(turn_id, generation)

    async def preview(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run one independent Preview snapshot without finalizing the turn."""

        with self._lock:
            turn = self.turn
            if turn is None:
                return self._error("turn_not_started", "Send start before preview")
            if turn.phase != "receiving":
                return self._error("turn_not_active", f"Turn is {turn.phase}")
            requested_turn_id = payload.get("turnId")
            if requested_turn_id is not None and requested_turn_id != turn.turn_id:
                return self._error(
                    "stale_turn",
                    "Preview request does not belong to the active turn",
                )
            requested_resume_id = payload.get("resumeId")
            if requested_resume_id is None:
                requested_resume_id = payload.get("resumeRequestId")
            if requested_resume_id is not None:
                requested_resume_id = _turn_id(requested_resume_id)
                if requested_resume_id is None:
                    return self._error(
                        "invalid_preview_request",
                        "resumeId must be a non-empty string of at most 128 characters",
                    )
                if requested_resume_id != turn.last_resume_request_id:
                    return self._error(
                        "stale_resume",
                        "Preview request does not belong to the active resume boundary",
                        {
                            "activeResumeId": turn.last_resume_request_id,
                            "activeResumeRequestId": turn.last_resume_request_id,
                        },
                    )
            requested_candidate_id = payload.get("candidateId")
            if requested_candidate_id is not None:
                if isinstance(requested_candidate_id, bool):
                    return self._error(
                        "invalid_preview_request",
                        "candidateId must be a non-empty string or integer",
                    )
                requested_candidate_id = _turn_id(str(requested_candidate_id))
                if requested_candidate_id is None:
                    return self._error(
                        "invalid_preview_request",
                        "candidateId must be a non-empty string of at most 128 characters",
                    )
                if requested_candidate_id != turn.candidate_id:
                    return self._error(
                        "stale_resume",
                        "Preview request does not belong to the active candidate",
                        {"activeCandidateId": turn.candidate_id},
                    )
            raw_request_id = payload.get("previewRequestId")
            if raw_request_id is None:
                raw_request_id = payload.get("requestId")
            if raw_request_id is None:
                request_id = f"preview-{turn.turn_id}-{turn.preview_count + 1}"
            else:
                request_id = _turn_id(raw_request_id)
                if request_id is None:
                    return self._error(
                        "invalid_preview_request",
                        "previewRequestId must be a non-empty string of at most 128 characters",
                    )
            audio_revision = turn.audio_revision
            audio_frames = turn.audio_frames
            turn.preview_requested = True
            turn.latest_preview_request_id = request_id
            turn.latest_preview_audio_revision = audio_revision
            turn.preview_completed_request_id = None
            turn.preview_completed_audio_revision = None
            turn.preview_status = ""
            turn.preview_failure = None
            turn.last_activity = time.monotonic()
            turn.telemetry["lastPreviewRequestedAt"] = turn.last_activity
            turn_id = turn.turn_id
            generation = turn.generation
            language = turn.language
            live_text = " ".join(self._last_partial.split())
            (
                input_pcm,
                input_scope,
                input_sample_start,
                input_sample_end,
                candidate_start_sample,
                candidate_start_byte,
                candidate_id,
                candidate_base_text,
                resume_request_id,
                resume_epoch,
            ) = self._preview_selection(turn, live_text)
            preview_epoch, cancelled = self._replace_preview_token_locked(turn)
            requested_at = turn.last_activity
            packet_count = turn.packet_count
            audio_seconds = turn.audio_seconds
        try:
            self._start_preview_worker(
                turn_id,
                generation,
                language,
                input_pcm,
                input_scope,
                live_text,
                request_id,
                requested_at,
                audio_revision,
                audio_frames,
                packet_count,
                input_sample_start,
                input_sample_end,
                candidate_start_sample,
                candidate_start_byte,
                candidate_id,
                candidate_base_text,
                resume_request_id,
                resume_epoch,
                preview_epoch,
                cancelled,
            )
        except Exception as exc:
            return self._error("preview_failed", str(exc))
        return {
            "type": "previewing",
            "sessionId": self.session_id,
            "turnId": turn_id,
            "previewRequestId": request_id,
            "audioPackets": packet_count,
            "audioDurationSeconds": round(audio_seconds, 6),
        }

    async def _finalize_preview_only(self) -> Optional[Dict[str, Any]]:
        """Close a turn without submitting complete-utterance Final ASR."""

        start_args = None
        complete_now = False
        with self._lock:
            turn = self.turn
            if turn is None:
                return self._error("turn_not_started", "There is no active turn to finalize")
            if turn.phase != "receiving":
                return self._error("turn_not_active", f"Turn is {turn.phase}")
            turn.phase = "draining"
            turn.finalize_requested = True
            turn.last_activity = time.monotonic()
            turn_id = turn.turn_id
            generation = turn.generation
            request_id = turn.latest_preview_request_id
            preview_is_fresh = (
                request_id is not None
                and turn.preview_requested
                and turn.latest_preview_audio_revision == turn.audio_revision
            )
            if not preview_is_fresh:
                request_id = (
                    f"preview-{turn_id}-finalize-{turn.audio_revision}-"
                    f"{uuid.uuid4().hex[:8]}"
                )
                audio_revision = turn.audio_revision
                audio_frames = turn.audio_frames
                packet_count = turn.packet_count
                turn.preview_requested = True
                turn.latest_preview_request_id = request_id
                turn.latest_preview_audio_revision = audio_revision
                turn.preview_completed_request_id = None
                turn.preview_completed_audio_revision = None
                turn.preview_status = ""
                turn.preview_failure = None
                turn.telemetry["lastPreviewRequestedAt"] = turn.last_activity
                live_text = " ".join(self._last_partial.split())
                (
                    input_pcm,
                    input_scope,
                    input_sample_start,
                    input_sample_end,
                    candidate_start_sample,
                    candidate_start_byte,
                    candidate_id,
                    candidate_base_text,
                    resume_request_id,
                    resume_epoch,
                ) = self._preview_selection(turn, live_text)
                preview_epoch, preview_cancelled = self._replace_preview_token_locked(
                    turn
                )
                start_args = (
                    turn_id,
                    generation,
                    turn.language,
                    input_pcm,
                    input_scope,
                    live_text,
                    request_id,
                    turn.last_activity,
                    audio_revision,
                    audio_frames,
                    packet_count,
                    input_sample_start,
                    input_sample_end,
                    candidate_start_sample,
                    candidate_start_byte,
                    candidate_id,
                    candidate_base_text,
                    resume_request_id,
                    resume_epoch,
                    preview_epoch,
                    preview_cancelled,
                )
            elif (
                request_id == turn.preview_completed_request_id
                and turn.preview_completed_audio_revision == turn.audio_revision
            ):
                complete_now = True
            turn.pcm_buffer.clear()
            packet_count = turn.packet_count
            audio_seconds = turn.audio_seconds
            current_turn = turn
        self._stop_live_stream(current_turn)
        if start_args is not None:
            try:
                self._start_preview_worker(*start_args)
            except Exception as exc:
                with self._lock:
                    current = self.turn
                    if (
                        current is not None
                        and current.turn_id == turn_id
                        and current.generation == generation
                    ):
                        current.preview_status = "error"
                        current.preview_failure = _structured_error(
                            "preview_transcription_failed",
                            str(exc),
                            session_id=self.session_id,
                            turn_id=turn_id,
                        )
                        current.preview_completed_request_id = request_id
                        current.preview_completed_audio_revision = current.audio_revision
                self._complete_preview_only_turn(turn_id, generation)
        elif complete_now:
            self._complete_preview_only_turn(turn_id, generation)
        return {
            "type": "finalizing",
            "sessionId": self.session_id,
            "turnId": turn_id,
            "previewOnly": True,
            "previewRequestId": request_id,
            "audioPackets": packet_count,
            "audioDurationSeconds": round(audio_seconds, 6),
        }
    async def finalize(self) -> Optional[Dict[str, Any]]:
        if getattr(self.settings, "preview_only_transcription", False):
            return await self._finalize_preview_only()
        with self._lock:
            turn = self.turn
            if turn is None:
                return self._error("turn_not_started", "There is no active turn to finalize")
            if turn.phase != "receiving":
                return self._error("turn_not_active", f"Turn is {turn.phase}")
            turn.phase = "draining"
            turn.last_activity = time.monotonic()
            turn_id = turn.turn_id
            generation = turn.generation
            language = turn.language
            pcm = bytes(turn.pcm_buffer)
            turn.pcm_buffer.clear()
        drain_failure = None
        try:
            live_queues = [
                live_queue
                for live_queue in (
                    turn.live_queue,
                    turn.ultrafast_live_queue,
                )
                if live_queue is not None
            ]
            drain_deadline = (
                time.monotonic() + self.settings.finalize_timeout_seconds
            )
            for live_queue in live_queues:
                remaining = max(0.0, drain_deadline - time.monotonic())
                await asyncio.to_thread(
                    live_queue.put,
                    None,
                    True,
                    remaining,
                )
        except Exception as exc:
            drain_failure = _structured_error(
                "finalize_failed",
                f"Could not seal the live audio queues: {exc}",
                session_id=self.session_id,
                turn_id=turn_id,
            )
        finalizing_admitted = threading.Event()
        thread = threading.Thread(
            target=self._run_authoritative_final_worker_after_finalizing,
            args=(
                finalizing_admitted,
                turn_id,
                generation,
                language,
                pcm,
                turn.cancelled,
                drain_failure,
            ),
            name=f"RealtimeSTTProductionFinal-{self.session_id}-{turn_id}",
            daemon=True,
        )
        with self._lock:
            self._completion_threads.add(thread)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._completion_threads.discard(thread)
            raise
        # The WebSocket command handler enqueues the returned ``finalizing``
        # response synchronously before its next await.  Release the worker on
        # the following event-loop turn so an immediately completed Final can
        # never overtake that acknowledgement in the per-session FIFO.
        asyncio.get_running_loop().call_soon(finalizing_admitted.set)
        return {
            "type": "finalizing",
            "sessionId": self.session_id,
            "turnId": turn_id,
            "audioPackets": turn.packet_count,
            "audioDurationSeconds": round(turn.audio_seconds, 6),
        }

    def _run_authoritative_final_worker_after_finalizing(
        self,
        finalizing_admitted: threading.Event,
        turn_id: str,
        generation: int,
        language: str,
        pcm: bytes,
        cancelled: threading.Event,
        drain_failure: Optional[Dict[str, Any]] = None,
    ) -> None:
        finalizing_admitted.wait()
        self._run_authoritative_final_worker(
            turn_id,
            generation,
            language,
            pcm,
            cancelled,
            drain_failure,
        )

    def _run_authoritative_final_worker(
        self,
        turn_id: str,
        generation: int,
        language: str,
        pcm: bytes,
        cancelled: threading.Event,
        drain_failure: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self._authoritative_final_worker(
                turn_id,
                generation,
                language,
                pcm,
                cancelled,
                drain_failure,
            )
        finally:
            with self._lock:
                self._completion_threads.discard(threading.current_thread())

    def _authoritative_final_worker(
        self,
        turn_id: str,
        generation: int,
        language: str,
        pcm: bytes,
        cancelled: threading.Event,
        drain_failure: Optional[Dict[str, Any]] = None,
    ) -> None:
        if cancelled.is_set():
            return
        failure = drain_failure
        status = "completed"
        text = ""
        live_done_events = []
        with self._lock:
            current = self.turn
            if (
                current is not None
                and current.turn_id == turn_id
                and current.generation == generation
            ):
                if current.live_queue is not None:
                    live_done_events.append(current.live_done)
                if current.ultrafast_live_queue is not None:
                    live_done_events.append(current.ultrafast_live_done)
                current.phase = "final_submitted"
        if failure is not None:
            status = "failed"
        elif live_done_events:
            deadline = time.monotonic() + self.settings.finalize_timeout_seconds
            for live_done in live_done_events:
                remaining = max(0.0, deadline - time.monotonic())
                if not live_done.wait(timeout=remaining):
                    status = "failed"
                    failure = _structured_error(
                        "final_transcription_failed",
                        "Live audio drain timed out before final transcription",
                        session_id=self.session_id,
                        turn_id=turn_id,
                    )
                    break
            if failure is not None:
                with self._lock:
                    current = self.turn
                    if (
                        current is not None
                        and current.turn_id == turn_id
                        and current.generation == generation
                    ):
                        live_turn = current
                    else:
                        live_turn = None
                self._stop_live_stream(live_turn)
        if cancelled.is_set():
            return
        if pcm and failure is None:
            try:
                import numpy as np

                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                transcribe = getattr(self.service, "transcribe_turn", None)
                if callable(transcribe):
                    result = transcribe(audio, language, False)
                else:
                    result = _service_turn_transcription(
                        self.service,
                        audio,
                        language,
                        self.settings.finalize_timeout_seconds,
                        self.session_id,
                        generation,
                    )
                text = str(getattr(result, "text", result) or "").strip()
            except Exception as exc:
                LOGGER.exception("Authoritative production turn finalization failed")
                status = "failed"
                failure = _structured_error(
                    "final_transcription_failed",
                    str(exc),
                    session_id=self.session_id,
                    turn_id=turn_id,
                )

        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.turn_id != turn_id
                or turn.generation != generation
                or turn.terminal_sent
            ):
                return
            turn.terminal_sent = True
            turn.phase = "terminal_result"
            if failure is not None:
                terminal = failure
            else:
                terminal = {
                    "type": "final",
                    "sessionId": self.session_id,
                    "turnId": turn_id,
                    "text": text,
                    "status": "completed" if text else "no_speech",
                }
            if turn.connection_epoch is not None:
                terminal["_connectionEpoch"] = turn.connection_epoch
            publish = getattr(self.manager, "publish_session")
            try:
                publish(self.session_id, terminal, authoritative=True)
            except TypeError:
                publish(self.session_id, terminal)
            try:
                snapshot = self.session.snapshot()
            except Exception:
                snapshot = {}
            self._publish_completion(
                turn_id,
                generation,
                status,
                snapshot,
                failure=failure,
            )

    async def reset(self) -> Dict[str, Any]:
        with self._lock:
            current_turn = self.turn
            old_turn = current_turn.turn_id if current_turn else None
            if current_turn is not None:
                current_turn.cancelled.set()
            self.turn = None
            self.manager.set_turn(self.session_id, None)
        self._stop_live_stream(current_turn)
        self.manager.suppress_type(self.session_id, "clear")
        try:
            self.session.clear()
        except Exception as exc:
            return self._error("reset_failed", str(exc))
        finally:
            self.manager.suppress_type(self.session_id, "clear", False)
        return {
            "type": "reset",
            "sessionId": self.session_id,
            "previousTurnId": old_turn,
        }

    async def cancel(self) -> Dict[str, Any]:
        with self._lock:
            turn = self.turn
            if turn is None:
                return self._error("turn_not_started", "There is no active turn to cancel")
            turn_id = turn.turn_id
            turn.phase = "cancelled"
            turn.cancelled.set()
            self.turn = None
            self.manager.set_turn(self.session_id, None)
        self._stop_live_stream(turn)
        self.manager.suppress_type(self.session_id, "clear")
        try:
            self.session.clear()
        except Exception as exc:
            return self._error("cancel_failed", str(exc), {"turnId": turn_id})
        finally:
            self.manager.suppress_type(self.session_id, "clear", False)
        return {
            "type": "cancelled",
            "sessionId": self.session_id,
            "turnId": turn_id,
        }

    def _publish_completion(
        self,
        turn_id: str,
        generation: int,
        status: str,
        snapshot: Dict[str, Any],
        *,
        failure: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            turn = self.turn
            if (
                turn is None
                or turn.turn_id != turn_id
                or turn.generation != generation
                or turn.completion_sent
            ):
                return
            turn.completion_sent = True
            preview_only = bool(getattr(self.settings, "preview_only_transcription", False))
            turn.final_count = 0 if preview_only else 1
            payload = {
                "type": "completion",
                "sessionId": self.session_id,
                "turnId": turn_id,
                "status": status,
                "audioPackets": turn.packet_count,
                "audioFrames": turn.audio_frames,
                "audioDurationSeconds": round(turn.audio_seconds, 6),
                "firstAudioSequence": turn.first_audio_sequence,
                "lastAudioSequence": turn.last_audio_sequence,
                "finalCount": turn.final_count,
                "previewCount": turn.preview_count,
                "partialCount": turn.partial_count,
                "stageTelemetry": dict(turn.telemetry),
            }
            if turn.connection_epoch is not None:
                payload["_connectionEpoch"] = turn.connection_epoch
            if status == "timeout":
                payload["error"] = {
                    "code": "completion_timeout",
                    "message": (
                        "Preview inference did not complete before the configured timeout"
                        if preview_only
                        else "Final inference did not complete before the configured timeout"
                    ),
                }
            elif status == "failed" and isinstance(failure, dict):
                payload["error"] = dict(
                    failure.get("error")
                    or {
                        "code": failure.get("code", "preview_transcription_failed" if preview_only else "final_transcription_failed"),
                        "message": failure.get("message", "Preview transcription failed" if preview_only else "Final transcription failed"),
                    }
                )
        # Both members of the terminal pair are admitted synchronously while
        # the turn lock is held. Reset/cancel may retire the turn afterwards,
        # but can no longer split an already-promised final/completion pair.
        if self.manager._loop is not None:
            delivery = self.manager.publish_session(
                self.session_id,
                payload,
                authoritative=True,
            )
            if delivery is None:
                return

            def mark_delivered(future: concurrent.futures.Future) -> None:
                try:
                    delivered = bool(future.result())
                except Exception:
                    delivered = False
                if not delivered:
                    return
                with self._lock:
                    current = self.turn
                    if (
                        current is not None
                        and current.turn_id == turn_id
                        and current.generation == generation
                        and current.completion_sent
                    ):
                        current.phase = "completed" if status == "completed" else status

            delivery.add_done_callback(mark_delivered)

    def close(self) -> None:
        with self._lock:
            self.closed = True
            current_turn = self.turn
            if current_turn is not None:
                current_turn.cancelled.set()
            self.turn = None
            self.manager.set_turn(self.session_id, None)
        self._stop_live_stream(current_turn)

    def _stop_live_stream(self, turn: Optional[TurnState] = None) -> None:
        if turn is None:
            return
        turn.live_cancelled.set()
        lanes = (
            (
                "realtime",
                turn.live_stream,
                turn.live_queue,
                "live_cancel_attempted",
            ),
            (
                "ultrafast",
                turn.ultrafast_live_stream,
                turn.ultrafast_live_queue,
                "ultrafast_live_cancel_attempted",
            ),
        )
        for lane, stream, live_queue, attempted_attribute in lanes:
            with self._lock:
                cancel_already_attempted = bool(
                    getattr(turn, attempted_attribute)
                )
                setattr(turn, attempted_attribute, True)

            cancel = getattr(stream, "cancel", None)
            if not cancel_already_attempted and callable(cancel):
                if not _LIVE_CANCEL_SLOTS.acquire(blocking=False):
                    LOGGER.warning(
                        "Production %s live cancellation capacity is exhausted; "
                        "closing the stream from its live worker instead",
                        lane,
                    )
                else:

                    def cancel_in_background(
                        cancel_callable=cancel,
                        lane_name=lane,
                    ) -> None:
                        try:
                            cancel_callable()
                        except Exception:
                            LOGGER.debug(
                                "Could not cancel production %s live stream",
                                lane_name,
                                exc_info=True,
                            )
                        finally:
                            _LIVE_CANCEL_SLOTS.release()
                            with self._lock:
                                self._live_cancel_threads.discard(
                                    threading.current_thread()
                                )

                    cancel_thread = threading.Thread(
                        target=cancel_in_background,
                        name=(
                            f"RealtimeSTTProductionLiveCancel-"
                            f"{self.session_id}-{lane}"
                        ),
                        daemon=True,
                    )
                    with self._lock:
                        self._live_cancel_threads.add(cancel_thread)
                    try:
                        cancel_thread.start()
                    except Exception:
                        with self._lock:
                            self._live_cancel_threads.discard(cancel_thread)
                        _LIVE_CANCEL_SLOTS.release()
                        raise

            if live_queue is None:
                continue
            while True:
                try:
                    live_queue.get_nowait()
                except (AttributeError, queue.Empty):
                    break
                else:
                    try:
                        live_queue.task_done()
                    except (AttributeError, ValueError):
                        pass
            try:
                live_queue.put_nowait(_LIVE_CANCEL)
            except queue.Full:
                LOGGER.warning(
                    "Could not wake cancelled production %s live stream",
                    lane,
                )

def _auth_ok(headers: Any, token: Optional[str]) -> bool:
    if not token:
        return True
    value = headers.get("authorization") if headers is not None else None
    if not value or not value.lower().startswith("bearer "):
        return False
    supplied = value[7:].strip()
    return bool(supplied) and secrets.compare_digest(supplied, token)


def _http_error(JSONResponse: Any, error: Dict[str, Any], status_code: int):
    return JSONResponse(error, status_code=status_code)


async def _read_limited_body(request: Any, max_bytes: int) -> bytes:
    """Read a request without allowing an unbounded body allocation."""

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise AudioPacketError("raw PCM body is too large")
        body.extend(chunk)
    return bytes(body)


def _make_settings_from_base(base_settings: Any, overrides: Dict[str, Any]) -> ProductionServerSettings:
    data = asdict(base_settings)
    data.update(overrides)
    # ``asdict`` can include fields from a future reference settings version;
    # use the production dataclass's declared fields as the compatibility
    # boundary rather than passing unknown values through.
    field_names = set(ProductionServerSettings.__dataclass_fields__)
    return ProductionServerSettings(**{key: value for key, value in data.items() if key in field_names})


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse production-only flags plus all reference server flags."""

    _backend_available()
    production = argparse.ArgumentParser(
        description="RealtimeSTT versioned production FastAPI server",
        add_help=False,
    )
    production.add_argument("--host", default="127.0.0.1")
    production.add_argument("--port", type=int, default=8010)
    production.add_argument(
        "--ssl-certfile",
        help="Uvicorn TLS certificate chain file for direct HTTPS/WSS binds",
    )
    production.add_argument(
        "--ssl-keyfile",
        help="Uvicorn TLS private key file for direct HTTPS/WSS binds",
    )
    production.add_argument("--idle-timeout-seconds", type=float, default=300.0)
    production.add_argument("--max-turn-audio-seconds", type=float, default=120.0)
    production.add_argument("--finalize-timeout-seconds", type=float, default=60.0)
    production.add_argument("--preview-only-transcription", "--preview-only", dest="preview_only_transcription", action="store_true", help="Disable complete-utterance Final ASR; serve only Preview results")
    production.add_argument(
        "--allow-late-final-transcription",
        action="store_true",
        help=(
            "In Preview-only mode, allow authenticated HTTP requests explicitly "
            f"marked operation={LATE_FINAL_OPERATION}"
        ),
    )
    production.add_argument(
        "--late-final-max-audio-seconds",
        type=float,
        default=30.0,
        help="Maximum audio duration accepted by the explicit late-Final operation",
    )
    production.add_argument(
        "--preview-tail-seconds",
        type=float,
        default=PREVIEW_TAIL_SECONDS,
        help=(
            "Deprecated compatibility option; production Preview now transcribes "
            "the complete selected buffer"
        ),
    )
    production.add_argument("--preview-min-live-words-for-fuzzy-repair", type=int, default=3)
    production.add_argument("--allowed-sample-rates", default="8000,16000,24000,32000,44100,48000")
    production.add_argument("--supported-languages", default=",".join(REMOTE_LANGUAGE_CHOICES))
    production.add_argument("--max-http-audio-bytes", type=int, default=8 * 1024 * 1024)
    production.add_argument("--help", action="store_true")
    known, remaining = production.parse_known_args(argv)
    if known.help:
        production.print_help()
        # The production parser owns the security/contract flags, while the
        # reference parser owns the complete engine/model/VAD option surface.
        # Print both so ``stt-server-production --help`` is a useful complete
        # CLI document instead of hiding the inherited options.
        from example_fastapi_server.server import parse_args as parse_reference_args

        parse_reference_args(["--help"])
        raise SystemExit(0)  # pragma: no cover - argparse exits above
    from example_fastapi_server.server import parse_args as parse_reference_args

    base_args = parse_reference_args(remaining)
    return argparse.Namespace(
        _base_args=base_args,
        host=known.host,
        port=known.port,
        bearer_token=None,
        ssl_certfile=known.ssl_certfile,
        ssl_keyfile=known.ssl_keyfile,
        idle_timeout_seconds=known.idle_timeout_seconds,
        max_turn_audio_seconds=known.max_turn_audio_seconds,
        finalize_timeout_seconds=known.finalize_timeout_seconds,
        preview_only_transcription=known.preview_only_transcription,
        allow_late_final_transcription=known.allow_late_final_transcription,
        late_final_max_audio_seconds=known.late_final_max_audio_seconds,
        preview_tail_seconds=known.preview_tail_seconds,
        preview_min_live_words_for_fuzzy_repair=known.preview_min_live_words_for_fuzzy_repair,
        allowed_sample_rates=tuple(
            int(part.strip()) for part in known.allowed_sample_rates.split(",") if part.strip()
        ),
        supported_languages=tuple(
            part.strip().lower() for part in known.supported_languages.split(",") if part.strip()
        ),
        max_http_audio_bytes=known.max_http_audio_bytes,
    )


def settings_from_args(args: Any) -> ProductionServerSettings:
    _backend_available()
    if hasattr(args, "_base_args"):
        from example_fastapi_server.server import settings_from_args as settings_from_reference_args

        base = settings_from_reference_args(args._base_args)
        overrides = {
            "host": args.host,
            "port": args.port,
            "bearer_token": args.bearer_token,
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
            "idle_timeout_seconds": args.idle_timeout_seconds,
            "max_turn_audio_seconds": args.max_turn_audio_seconds,
            "finalize_timeout_seconds": args.finalize_timeout_seconds,
            "preview_only_transcription": args.preview_only_transcription,
            "allow_late_final_transcription": args.allow_late_final_transcription,
            "late_final_max_audio_seconds": args.late_final_max_audio_seconds,
            "preview_tail_seconds": args.preview_tail_seconds,
            "preview_min_live_words_for_fuzzy_repair": args.preview_min_live_words_for_fuzzy_repair,
            "allowed_sample_rates": args.allowed_sample_rates,
            "supported_languages": args.supported_languages,
            "max_http_audio_bytes": args.max_http_audio_bytes,
        }
        return _make_settings_from_base(base, overrides)
    return ProductionServerSettings(**vars(args))


def _service_raw_transcription(service: Any, audio: Any, language: str, timeout: float):
    """Submit one HTTP transcription through the shared final model lane."""

    request_id = uuid.uuid4().hex
    session_id = f"http-{request_id}"
    generation = 0
    holder = {
        "event": threading.Event(),
        "result": None,
        "error": None,
        "sessionId": session_id,
        "generation": generation,
    }
    with service._pending_recorder_lock:
        service._pending_recorder_results[request_id] = holder
    created_at = time.monotonic()
    job = InferenceJob(
        request_id=request_id,
        session_id=session_id,
        kind="final",
        audio=audio,
        language=language,
        use_prompt=False,
        segment_id=1,
        sequence=0,
        generation=generation,
        created_at=created_at,
        deadline_at=created_at + timeout,
    )
    try:
        result = service.scheduler.submit(job)
    except Exception:
        service._pop_pending_recorder_result(request_id)
        raise
    if not result.accepted:
        service._pop_pending_recorder_result(request_id)
        raise RuntimeError(result.reason or "final transcription queue rejected the request")
    if not holder["event"].wait(timeout=timeout):
        service._pop_pending_recorder_result(request_id)
        raise TimeoutError("final transcription timed out")
    service._pop_pending_recorder_result(request_id)
    if holder["error"]:
        raise RuntimeError(holder["error"])
    inference_result = holder["result"]
    if inference_result is None:
        raise RuntimeError("final transcription returned no result")
    if inference_result.error:
        raise RuntimeError(inference_result.error)
    return inference_result


def _service_turn_transcription(
    service: Any,
    audio: Any,
    language: str,
    timeout: float,
    session_id: str,
    generation: int,
    *,
    admission_check=None,
    admission_lock=None,
):
    """Submit one cancellable authoritative WebSocket turn final."""

    request_id = uuid.uuid4().hex
    holder = {
        "event": threading.Event(),
        "result": None,
        "error": None,
        "sessionId": session_id,
        "generation": generation,
    }
    with service._pending_recorder_lock:
        service._pending_recorder_results[request_id] = holder
    created_at = time.monotonic()
    job = InferenceJob(
        request_id=request_id,
        session_id=session_id,
        kind="final",
        audio=audio,
        language=language,
        use_prompt=False,
        segment_id=1,
        sequence=0,
        generation=generation,
        created_at=created_at,
        deadline_at=created_at + timeout,
    )

    def submit_if_admitted():
        if admission_check is not None and not admission_check():
            raise _PreviewAdmissionCancelled()
        return service.scheduler.submit(job)

    try:
        if admission_lock is None:
            submitted = submit_if_admitted()
        else:
            # The production protocol's RLock is also held by Resume while it
            # replaces the Preview token.  This makes the admission check and
            # scheduler submission one linearizable critical section without
            # holding the lock while the worker waits for inference.
            with admission_lock:
                submitted = submit_if_admitted()
        if not submitted.accepted:
            raise RuntimeError(
                submitted.reason or "final transcription queue rejected the request"
            )
        if not holder["event"].wait(timeout=timeout):
            service.scheduler.cancel_request(request_id)
            raise TimeoutError("final transcription timed out")
        if holder["error"]:
            raise RuntimeError(holder["error"])
        result = holder["result"]
        if result is None:
            raise RuntimeError("final transcription returned no result")
        if result.error:
            raise RuntimeError(result.error)
        return result
    finally:
        service._pop_pending_recorder_result(request_id)


def _reported_detected_language(result: Any, requested_language: str) -> Optional[str]:
    """Return a provider detection for ``auto`` without inventing one.

    Fixed-language requests historically report their normalized requested
    language. Auto detection is different: returning ``"auto"`` claims a
    detection that never happened, so only an explicit provider result is
    reported and otherwise the JSON field is ``null``.
    """

    requested = str(requested_language or "").strip().lower()
    if requested != "auto":
        return requested or None

    candidates = [
        getattr(result, "detected_language", None),
        getattr(getattr(result, "info", None), "language", None),
    ]
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        candidates.extend(
            (metadata.get("detected_language"), metadata.get("language"))
        )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip().lower()
        if normalized and normalized != "auto":
            return normalized
    return None


def _reported_language_probability(result: Any) -> Optional[float]:
    """Return a finite provider language probability when one is available."""

    info = getattr(result, "info", None)
    metadata = getattr(result, "metadata", None)
    language_candidates = [
        getattr(result, "detected_language", None),
        getattr(result, "language", None),
        getattr(info, "language", None),
        getattr(info, "detected_language", None),
    ]
    if isinstance(info, dict):
        language_candidates.extend(
            (info.get("language"), info.get("detected_language"))
        )
    if isinstance(metadata, dict):
        language_candidates.extend(
            (metadata.get("language"), metadata.get("detected_language"))
        )
    has_explicit_language = any(
        isinstance(candidate, str)
        and candidate.strip()
        and candidate.strip().lower() != "auto"
        for candidate in language_candidates
    )
    candidates = [
        getattr(result, "language_probability", None),
        getattr(info, "language_probability", None),
        getattr(info, "languageProbability", None),
    ]
    if isinstance(info, dict):
        candidates.extend(
            (info.get("language_probability"), info.get("languageProbability"))
        )
    if isinstance(metadata, dict):
        candidates.extend(
            (
                metadata.get("language_probability"),
                metadata.get("languageProbability"),
            )
        )
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        probability = float(candidate)
        if math.isfinite(probability) and 0.0 <= probability <= 1.0:
            if probability == 0.0 and not has_explicit_language:
                continue
            return probability
    return None


def create_app(
    settings: Optional[ProductionServerSettings] = None,
    scheduler_factory=None,
    recorder_factory=None,
):
    """Create the versioned FastAPI application.

    ``scheduler_factory`` and ``recorder_factory`` are intentionally exposed
    for deterministic unit tests and are the same injection points as the
    source-only FastAPI implementation.
    """

    _backend_available()
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI server dependencies are missing. Install the server extras."
        ) from exc

    settings = settings or ProductionServerSettings()
    if not isinstance(settings, ProductionServerSettings):
        settings = _make_settings_from_base(settings, {})
    manager = OrderedConnectionManager()
    service = RealtimeSTTService(
        settings,
        manager,
        scheduler_factory=scheduler_factory,
        recorder_factory=recorder_factory,
    )

    @asynccontextmanager
    async def lifespan(app):
        service.start(asyncio.get_running_loop())
        try:
            yield
        finally:
            service.stop()
            release_service_resources(service)

    app = FastAPI(title=SERVER_NAME, version=SERVER_VERSION, lifespan=lifespan)

    def unauthorized():
        return _http_error(
            JSONResponse,
            _structured_error("unauthorized", "A valid bearer token is required"),
            401,
        )

    def health_payload(kind: str):
        try:
            metrics = service.metrics()
        except Exception as exc:
            LOGGER.exception("Could not collect health metrics")
            metrics = {"ready": False, "ok": False, "startupErrors": [str(exc)]}
        ready = bool(metrics.get("ready")) and bool(metrics.get("ok"))
        device = getattr(settings, "device", "cpu")
        try:
            reported_device = effective_device(device)
        except Exception:
            reported_device = device
        provider = "cuda" if str(reported_device).lower().startswith("cuda") else "cpu"
        return {
            "apiVersion": API_VERSION,
            "protocolVersion": PROTOCOL_VERSION,
            "status": "ready" if kind == "ready" and ready else "live" if kind == "live" else "not_ready",
            "ok": True if kind == "live" else ready,
            "ready": ready,
            "activeSessions": metrics.get("activeSessions", 0),
            "activeSpeakers": metrics.get("activeSpeakers", 0),
            "scheduler": metrics.get("scheduler", {}),
            "startupErrors": metrics.get("startupErrors", []),
            # Keep the compact fields used by existing HTTP ASR health
            # probes, in addition to the versioned service counters above.
            "engine": getattr(settings, "transcription_engine", None),
            "model": getattr(settings, "model", None),
            "realtime_engine": getattr(settings, "realtime_transcription_engine", None)
            or getattr(settings, "transcription_engine", None),
            "realtime_model": getattr(settings, "realtime_model", None),
            "device": reported_device,
            "provider": provider,
            "compute_type": getattr(settings, "compute_type", None),
            "max_inflight": getattr(settings, "max_global_inference_queue_depth", None),
            "model_load_seconds": None,
            "warmup_seconds": None,
        }

    async def health_route(request: Request, kind: str):
        if not _auth_ok(request.headers, settings.bearer_token):
            return unauthorized()
        payload = health_payload(kind)
        return JSONResponse(payload, status_code=200 if kind == "live" or payload["ok"] else 503)

    @app.get("/api/v1/live")
    async def api_live(request: Request):
        return await health_route(request, "live")

    @app.get("/api/v1/ready")
    async def api_ready(request: Request):
        return await health_route(request, "ready")

    @app.get("/api/v1/capabilities")
    async def api_capabilities(request: Request):
        if not _auth_ok(request.headers, settings.bearer_token):
            return unauthorized()
        payload = capabilities_for(settings)
        payload["ready"] = bool(service.ready.is_set() and service.scheduler.healthy())
        return JSONResponse(payload)

    # Short aliases make the version visible without requiring clients to
    # know whether the deployment prefixes APIs with ``/api``.
    @app.get("/v1/live")
    async def short_live(request: Request):
        return await health_route(request, "live")

    @app.get("/v1/ready")
    async def short_ready(request: Request):
        return await health_route(request, "ready")

    @app.get("/v1/capabilities")
    async def short_capabilities(request: Request):
        return await api_capabilities(request)

    @app.get("/api/v1/health/live")
    async def api_health_live(request: Request):
        return await health_route(request, "live")

    @app.get("/api/v1/health/ready")
    async def api_health_ready(request: Request):
        return await health_route(request, "ready")

    @app.get("/v1/health/live")
    async def short_health_live(request: Request):
        return await health_route(request, "live")

    @app.get("/v1/health/ready")
    async def short_health_ready(request: Request):
        return await health_route(request, "ready")

    @app.get("/health")
    async def compatibility_health(request: Request):
        """Compatibility health shape for existing HTTP ASR probes."""

        if not _auth_ok(request.headers, settings.bearer_token):
            return unauthorized()
        payload = health_payload("ready")
        payload["status"] = "ok" if payload["ok"] else "loading"
        return JSONResponse(payload, status_code=200 if payload["ok"] else 503)

    @app.post("/transcribe-pcm16")
    async def transcribe_pcm16(
        request: Request,
        sample_rate: int = SERVER_SAMPLE_RATE,
        encoding: str = "pcm16",
        language: str = "en",
        beam_size: int = 3,
        best_of: int = 1,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
        without_timestamps: bool = True,
        operation: Optional[str] = None,
    ):
        del word_timestamps, vad_filter, condition_on_previous_text, without_timestamps
        if not _auth_ok(request.headers, settings.bearer_token):
            return unauthorized()
        operation_value = str(operation or "").strip()
        if operation_value not in {"", LATE_FINAL_OPERATION}:
            return _http_error(
                JSONResponse,
                _structured_error(
                    "invalid_operation",
                    f"Unsupported transcription operation: {operation_value}",
                ),
                400,
            )
        late_final_requested = operation_value == LATE_FINAL_OPERATION
        if settings.preview_only_transcription:
            if late_final_requested and settings.allow_late_final_transcription:
                pass
            elif late_final_requested:
                return _http_error(
                    JSONResponse,
                    _structured_error(
                        "late_final_asr_disabled",
                        "The explicit late-Final ASR operation is disabled",
                    ),
                    409,
                )
            else:
                return _http_error(
                    JSONResponse,
                    _structured_error(
                        "final_asr_disabled",
                        "Final ASR is disabled; use the WebSocket Preview operation",
                    ),
                    409,
                )
        if encoding.lower() != "pcm16":
            return _http_error(
                JSONResponse,
                _structured_error("invalid_encoding", "encoding must be pcm16"),
                400,
            )
        if sample_rate not in settings.allowed_sample_rates:
            return _http_error(
                JSONResponse,
                _structured_error(
                    "invalid_sample_rate",
                    f"sample_rate must be one of {list(settings.allowed_sample_rates)}",
                ),
                400,
            )
        language_error = _language_error(language, settings)
        if language_error:
            return _http_error(
                JSONResponse,
                _structured_error(
                    language_error["code"], language_error["message"], details=language_error.get("details")
                ),
                400,
            )
        if beam_size <= 0 or best_of <= 0 or temperature < 0:
            return _http_error(
                JSONResponse,
                _structured_error("invalid_decode_options", "beam_size/best_of must be positive and temperature non-negative"),
                400,
            )
        max_audio_seconds = float(settings.max_turn_audio_seconds)
        if late_final_requested:
            max_audio_seconds = min(
                max_audio_seconds,
                float(settings.late_final_max_audio_seconds),
            )
        max_audio_bytes = min(
            settings.max_http_audio_bytes,
            int(max_audio_seconds * sample_rate) * 2,
        )
        content_length = request.headers.get("content-length")
        try:
            declared_length = int(content_length) if content_length else None
        except ValueError:
            return _http_error(JSONResponse, _structured_error("invalid_content_length", "invalid Content-Length"), 400)
        if declared_length is not None and declared_length > max_audio_bytes:
            return _http_error(
                JSONResponse,
                _structured_error("audio_size_limit", "raw PCM body is too large"),
                413,
            )
        try:
            body = await _read_limited_body(request, max_audio_bytes)
        except AudioPacketError as exc:
            return _http_error(JSONResponse, _structured_error("audio_size_limit", str(exc)), 413)
        if not body:
            return _http_error(JSONResponse, _structured_error("empty_audio", "raw PCM body must not be empty"), 400)
        if len(body) % 2:
            return _http_error(
                JSONResponse,
                _structured_error("invalid_audio", "raw PCM16 body must contain whole samples"),
                400,
            )
        if not service.ready.is_set() or not service.scheduler.healthy():
            return _http_error(JSONResponse, _structured_error("not_ready", "ASR model is not ready"), 503)

        import numpy as np

        samples = np.frombuffer(body, dtype=np.int16)
        if sample_rate != SERVER_SAMPLE_RATE:
            samples = resample_int16(samples, sample_rate, SERVER_SAMPLE_RATE)
        audio = samples.astype(np.float32) / 32768.0
        audio_duration = len(samples) / float(SERVER_SAMPLE_RATE)
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(
                _service_raw_transcription,
                service,
                audio,
                language.strip().lower(),
                settings.finalize_timeout_seconds,
            )
        except TimeoutError as exc:
            return _http_error(JSONResponse, _structured_error("transcription_timeout", str(exc)), 504)
        except Exception as exc:
            LOGGER.exception("Raw PCM transcription failed")
            return _http_error(JSONResponse, _structured_error("transcription_failed", str(exc)), 500)
        elapsed = time.monotonic() - started
        requested_language = language.strip().lower()
        return JSONResponse({
            "text": getattr(result, "text", "") or "",
            "detected_language": _reported_detected_language(result, requested_language),
            "language_probability": _reported_language_probability(result),
            "elapsed_seconds": elapsed,
            "queue_seconds": getattr(result, "queue_delay", 0.0),
            "audio_duration_seconds": audio_duration,
            "decode_seconds": getattr(result, "inference_duration", elapsed),
            "rtf": getattr(result, "inference_duration", elapsed) / max(audio_duration, 1 / SERVER_SAMPLE_RATE),
            "engine": getattr(settings, "transcription_engine", None),
            "model": getattr(settings, "model", None),
            "provider": "cuda" if str(effective_device(getattr(settings, "device", "cpu"))).lower().startswith("cuda") else "cpu",
            "num_threads": None,
            "segments": [],
        })

    @app.websocket("/api/v1/ws/transcribe")
    @app.websocket("/api/v1/ws")
    @app.websocket("/v1/ws/transcribe")
    @app.websocket("/v1/audio/transcriptions/stream")
    async def websocket_transcribe(websocket: WebSocket):
        session_id = uuid.uuid4().hex
        if not _auth_ok(websocket.headers, settings.bearer_token):
            await websocket.accept()
            await websocket.send_text(json.dumps(_structured_error("unauthorized", "A valid bearer token is required")))
            await websocket.close(code=1008)
            return
        try:
            session = _admit_production_session(service, session_id)
        except Exception as exc:
            LOGGER.exception("Could not construct production session")
            session = None
            admission_error = _structured_error("session_init_failed", str(exc))
        else:
            admission_error = None
        if session is None:
            await websocket.accept()
            payload = admission_error or _structured_error(
                "session_limit",
                "Server is at the configured session limit",
                details=service.limits_dict(),
            )
            await websocket.send_text(json.dumps(payload))
            await websocket.close(code=1013)
            return

        protocol = None
        try:
            protocol = ProductionSessionProtocol(service, manager, session_id, settings)
            protocol.attach(session)
            await manager.connect(session_id, websocket)
            await manager.emit(session_id, {
                "type": "hello",
                "clientId": session_id,
                "capabilities": capabilities_for(settings),
            })
            if service.ready.is_set():
                await manager.emit(session_id, {
                    "type": "ready",
                    "ready": service.scheduler.healthy(),
                    "capabilities": capabilities_for(settings),
                })
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(), timeout=settings.idle_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    await protocol.send_error(
                        "idle_timeout",
                        "WebSocket closed after exceeding the idle timeout",
                        {"idleTimeoutSeconds": settings.idle_timeout_seconds},
                    )
                    await websocket.close(code=1000)
                    break
                if message.get("type") == "websocket.disconnect":
                    break
                protocol.touch()
                if message.get("bytes") is not None:
                    error = await protocol.audio(message["bytes"])
                    if error:
                        await manager.emit(session_id, error)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    await protocol.send_error("invalid_command_json", f"Invalid command JSON: {exc.msg}")
                    continue
                if not isinstance(payload, dict):
                    await protocol.send_error("invalid_command", "WebSocket commands must be JSON objects")
                    continue
                command = payload.get("type") or payload.get("command")
                if command == "start":
                    response = await protocol.start(payload)
                elif command == "preview":
                    response = await protocol.preview(payload)
                elif command == "resume":
                    response = await protocol.resume(payload)
                    if response:
                        queued_delivery = protocol.take_queued_resume_ack_delivery(
                            response
                        )
                        if queued_delivery is not None:
                            # The ACK already owns its FIFO slot. Waiting for
                            # a stalled sender here would stop receive(),
                            # including fresh PCM and peer disconnects; the
                            # dedicated sender task preserves wire order.
                            continue
                        if response.get("type") == "error":
                            # Resume validation failures are just as capable
                            # of occurring behind a stalled sender as an ACK
                            # reserve rejection.  They already carry their
                            # request/turn correlation, so reserve their FIFO
                            # place without waiting for transport delivery.
                            # Otherwise receive() would stop accepting PCM or
                            # a peer disconnect while the client is stalled.
                            queued_error = manager.publish_session(
                                session_id,
                                response,
                            )
                            if queued_error is None:
                                # No unbounded control-message bypass: if the
                                # bounded lane cannot carry the correlated
                                # error, terminate rather than leave the
                                # client waiting silently.
                                try:
                                    await websocket.close(
                                        code=1013,
                                        reason="outbound backpressure",
                                    )
                                except TypeError:
                                    await websocket.close(code=1013)
                                break
                            continue
                elif command in ("finalize", "finish", "stop"):
                    response = await protocol.finalize()
                elif command == "reset":
                    response = await protocol.reset()
                elif command == "cancel":
                    response = await protocol.cancel()
                elif command == "ping":
                    # A pong is advisory. Queue it in the same bounded FIFO,
                    # but never await a stalled transport here: an earlier
                    # Resume error may already own the sender while PCM and a
                    # peer disconnect still need to reach receive(). A full
                    # queue simply drops this pong; it does not bypass the
                    # ordinary backpressure limit.
                    manager.publish_session(
                        session_id,
                        {"type": "pong", "serverTime": time.time()},
                    )
                    continue
                elif command == "capabilities":
                    response = {"type": "capabilities", **capabilities_for(settings)}
                else:
                    response = protocol._error("unknown_command", f"Unknown command: {command}")
                if response:
                    await manager.emit(session_id, response)
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            # ASGI servers may cancel the receive task as part of a normal
            # peer disconnect.  Cleanup below owns the session shutdown, so
            # do not leak that transport-level cancellation to the client.
            pass
        except RuntimeError as exc:
            if "disconnect" not in str(exc).lower():
                LOGGER.debug("Production websocket runtime error", exc_info=True)
        finally:
            if protocol is not None:
                protocol.close()
            try:
                service.remove_session(session_id)
            finally:
                try:
                    try:
                        await manager.disconnect(session_id)
                    except asyncio.CancelledError:
                        # A peer-close can cancel the ASGI receive scope while
                        # its per-session sender is being drained.  Session
                        # state still has to be cleared, but this normal close
                        # must not escape as a failed WebSocket context.
                        pass
                finally:
                    manager.clear_session(session_id)

    app.state.realtimestt_service = service
    app.state.production_settings = settings
    app.state.capabilities = capabilities_for(settings)
    return app


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not hasattr(args, "_base_args"):
        return
    settings = settings_from_args(args)
    logging.basicConfig(
        level=getattr(settings, "log_level", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError("uvicorn is required for the production server") from exc
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=getattr(settings, "log_level", "INFO").lower(),
        ssl_certfile=settings.ssl_certfile,
        ssl_keyfile=settings.ssl_keyfile,
    )


__all__ = [
    "API_VERSION",
    "PROTOCOL_VERSION",
    "SERVER_VERSION",
    "PCM_FORMAT",
    "REMOTE_LANGUAGES",
    "REMOTE_LANGUAGE_CHOICES",
    "ProductionServerSettings",
    "ServerSettings",
    "OrderedConnectionManager",
    "ProductionSessionProtocol",
    "capabilities_for",
    "create_app",
    "is_loopback_host",
    "main",
    "parse_args",
    "release_service_resources",
    "settings_from_args",
]


if __name__ == "__main__":
    main()
