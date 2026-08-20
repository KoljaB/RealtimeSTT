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
import os
import queue
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


LOGGER = logging.getLogger("realtimestt.production_server")

API_VERSION = "v1"
PROTOCOL_VERSION = "realtimestt.remote.v1"
SERVER_NAME = "RealtimeSTT production server"
_PACKAGE_VERSION_FALLBACK = "1.0.4rc1"


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
_LIVE_CANCEL = object()
_LIVE_QUEUE_PACKET_FLOOR_SAMPLES = SERVER_SAMPLE_RATE // 10
_MAX_LIVE_CANCEL_THREADS = 4
_LIVE_CANCEL_SLOTS = threading.BoundedSemaphore(_MAX_LIVE_CANCEL_THREADS)
_MAX_LIVE_STREAM_OPERATIONS = 4
_LIVE_STREAM_OPERATION_SLOTS = threading.BoundedSemaphore(
    _MAX_LIVE_STREAM_OPERATIONS
)
REMOTE_LANGUAGES = ("en", "de", "fr", "es", "it", "pt", "ru")
# ``auto`` asks the realtime/final provider to detect the language. Keep the
# seven explicit AgentTalk languages alongside it in the public contract.
REMOTE_LANGUAGE_CHOICES = ("auto", *REMOTE_LANGUAGES)
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?$")


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
    allowed_sample_rates: Tuple[int, ...] = (8000, 16000, 24000, 32000, 44100, 48000)
    supported_languages: Tuple[str, ...] = REMOTE_LANGUAGE_CHOICES
    max_http_audio_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
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
    final_model = getattr(settings, "model", None)
    live_model = getattr(settings, "realtime_model", None) or final_model
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
            },
            "live": {
                "model": live_model,
                "provider": live_provider,
                "engine": live_provider,
                "languages": languages,
                "language": getattr(settings, "language", None),
            },
            # ``realtime`` is retained as a descriptive alias for clients that
            # use that term instead of live/partial transcription.
            "realtime": {
                "model": live_model,
                "provider": live_provider,
                "engine": live_provider,
                "languages": languages,
                "language": getattr(settings, "language", None),
            },
        },
        "finalModel": final_model,
        "finalProvider": final_provider,
        "liveModel": live_model,
        "liveProvider": live_provider,
        "languages": languages,
        "audioFormat": PCM_FORMAT,
        "audio": audio,
        "limits": {
            "maxSessions": getattr(settings, "max_sessions", None),
            "maxActiveSpeakers": getattr(settings, "max_active_speakers", None),
            "maxTurnAudioSeconds": settings.max_turn_audio_seconds,
            "idleTimeoutSeconds": settings.idle_timeout_seconds,
            "finalizeTimeoutSeconds": settings.finalize_timeout_seconds,
            "maxAudioPacketBytes": audio["maxPacketBytes"],
        },
        "operations": {
            "websocket": ["start", "audio", "finalize", "reset", "cancel"],
            "events": ["partial", "final", "completion", "status", "error"],
            "http": ["transcribe-pcm16"],
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

    _TERMINAL_TYPES = frozenset({"final", "completion"})

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
        if source_type == "final":
            return True
        if source_type != "error":
            return False
        code = str(message.get("code") or "").lower()
        where = str(message.get("where") or "").lower()
        return where in {"final", "recorder"} or code.startswith("final")

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
            if is_partial:
                # Keep the oldest pending sequence slot but replace its
                # hypothesis with the newest one.  No new sequence is
                # allocated for coalesced updates, so clients see a contiguous
                # stream even when a slow client skips stale partials.
                current_turn_id = self._turn_ids.get(session_id)
                message_turn_id = message.get("turnId", current_turn_id)
                if message_turn_id == current_turn_id:
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

            is_final_outcome = self._is_final_outcome(message)
            terminal_hard_limit = self.max_pending_events + 2
            terminal_overflow = (
                is_final_outcome and len(state.queue) > self.max_pending_events
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

            if (
                len(state.queue) >= self.max_pending_events
                and source_type not in self._TERMINAL_TYPES
                and not is_final_outcome
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
    partial_count: int = 0
    final_count: int = 0
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
    cancelled: threading.Event = field(default_factory=threading.Event)
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
    for name in ("main_worker", "realtime_worker"):
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
        self._completion_threads = set()
        self._live_cancel_threads = set()
        self._live_stream_operation_threads = set()
        self._generation = 0
        self._last_partial = ""
        self._last_partial_sent_at = 0.0

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
    ) -> bool:
        """Install a created stream only while its starting turn still owns it."""

        live_thread = None
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
                self._last_partial = ""
                self._last_partial_sent_at = 0.0
                live_thread = threading.Thread(
                    target=self._live_worker,
                    args=(
                        turn.turn_id,
                        turn.generation,
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
                turn.live_thread = live_thread
                turn.phase = "receiving"

        if not current:
            return False
        try:
            live_thread.start()
        except Exception:
            # ``start()`` owns the matching-turn cleanup and this bounded
            # worker owns closing the just-created stream.
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
        """Create, transfer, or reap one stream without involving an ASGI loop."""

        stream = None
        try:
            worker = streaming_worker("realtime")
            stream = worker.create_streaming_session(
                language=language,
                use_prompt=False,
            )
            installed = self._install_live_stream(turn_id, generation, stream)
            if not installed:
                # The stream was completed after its owner retired.  Reap it
                # in this already-bounded native worker; never depend on the
                # cancelled ASGI task receiving its result.
                self._close_live_stream(stream)
            with self._lock:
                owned = (
                    installed
                    and self.turn is not None
                    and self.turn.turn_id == turn_id
                    and self.turn.generation == generation
                    and self.turn.live_stream is stream
                )
            if not result.done():
                result.set_result(owned)
        except Exception as exc:
            if stream is not None:
                self._close_live_stream(stream)
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
        live_queue: Any,
        stream: Any,
        live_done: threading.Event,
        live_cancelled: threading.Event,
    ) -> None:
        try:
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
                        self._publish_changed_partial(turn_id, generation, result)
                        return
                    decode_started = time.monotonic()
                    stream.accept_audio(samples, sample_rate=SERVER_SAMPLE_RATE)
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
                            turn.telemetry.setdefault("firstDecodeStartedAt", decode_started)
                            turn.telemetry["lastDecodeDoneAt"] = decode_done
                            turn.telemetry["decodeCalls"] = int(
                                turn.telemetry.get("decodeCalls", 0)
                            ) + 1
                            turn.telemetry["decodeSeconds"] = float(
                                turn.telemetry.get("decodeSeconds", 0.0)
                            ) + (decode_done - decode_started)
                    self._publish_changed_partial(turn_id, generation, stream.get_result())
                finally:
                    live_queue.task_done()
        except Exception:
            LOGGER.exception("Production live stream failed for turn %s", turn_id)
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
                    LOGGER.debug("Could not close production live stream", exc_info=True)
            live_done.set()

    def _publish_changed_partial(self, turn_id: str, generation: int, result: Any) -> None:
        text = " ".join(str(getattr(result, "text", result) or "").split())
        if not text:
            return
        now = time.monotonic()
        with self._lock:
            turn = self.turn
            if turn is None or turn.turn_id != turn_id or turn.generation != generation:
                return
            if text == self._last_partial:
                return
            self._last_partial = text
            if now - self._last_partial_sent_at < (1.0 / 15.0):
                return
            self._last_partial_sent_at = now
            turn.partial_count += 1
            turn.telemetry.setdefault("firstPartialSentAt", now)
            turn.telemetry["lastPartialSentAt"] = now
            connection_epoch = turn.connection_epoch
        payload = {"type": "realtime", "turnId": turn_id, "text": text}
        if connection_epoch is not None:
            payload["_connectionEpoch"] = connection_epoch
        self.manager.publish_session(
            self.session_id,
            payload,
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

                try:
                    queued_at = time.monotonic()
                    turn.live_queue.put_nowait(
                        np.frombuffer(packet.audio, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                except queue.Full:
                    return self._error("backpressure", "Production live audio queue is full")
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
            turn.audio_frames += frames
            turn.audio_seconds += duration
            turn.last_activity = time.monotonic()
        return None

    async def finalize(self) -> Optional[Dict[str, Any]]:
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
            if turn.live_queue is not None:
                await asyncio.to_thread(
                    turn.live_queue.put,
                    None,
                    True,
                    self.settings.finalize_timeout_seconds,
                )
        except Exception as exc:
            drain_failure = _structured_error(
                "finalize_failed",
                f"Could not seal the live audio queue: {exc}",
                session_id=self.session_id,
                turn_id=turn_id,
            )
        thread = threading.Thread(
            target=self._run_authoritative_final_worker,
            args=(turn_id, generation, language, pcm, turn.cancelled, drain_failure),
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
        return {
            "type": "finalizing",
            "sessionId": self.session_id,
            "turnId": turn_id,
            "audioPackets": turn.packet_count,
            "audioDurationSeconds": round(turn.audio_seconds, 6),
        }

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
        with self._lock:
            current = self.turn
            live_done = (
                current.live_done
                if current is not None
                and current.turn_id == turn_id
                and current.generation == generation
                else None
            )
            if current is not None and live_done is not None:
                current.phase = "final_submitted"
        if failure is not None:
            status = "failed"
        elif live_done is not None and not live_done.wait(
            timeout=self.settings.finalize_timeout_seconds
        ):
            status = "failed"
            failure = _structured_error(
                "final_transcription_failed",
                "Live audio drain timed out before final transcription",
                session_id=self.session_id,
                turn_id=turn_id,
            )
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
            turn.final_count = 1
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
                "partialCount": turn.partial_count,
                "stageTelemetry": dict(turn.telemetry),
            }
            if turn.connection_epoch is not None:
                payload["_connectionEpoch"] = turn.connection_epoch
            if status == "timeout":
                payload["error"] = {
                    "code": "completion_timeout",
                    "message": "Final inference did not complete before the configured timeout",
                }
            elif status == "failed" and isinstance(failure, dict):
                payload["error"] = dict(
                    failure.get("error")
                    or {
                        "code": failure.get("code", "final_transcription_failed"),
                        "message": failure.get("message", "Final transcription failed"),
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
        with self._lock:
            cancel_already_attempted = turn.live_cancel_attempted
            turn.live_cancel_attempted = True

        cancel = getattr(turn.live_stream, "cancel", None)
        if not cancel_already_attempted and callable(cancel):
            if not _LIVE_CANCEL_SLOTS.acquire(blocking=False):
                LOGGER.warning(
                    "Production live cancellation capacity is exhausted; "
                    "closing the stream from its live worker instead"
                )
            else:

                def cancel_in_background() -> None:
                    try:
                        cancel()
                    except Exception:
                        LOGGER.debug(
                            "Could not cancel production live stream",
                            exc_info=True,
                        )
                    finally:
                        _LIVE_CANCEL_SLOTS.release()
                        with self._lock:
                            self._live_cancel_threads.discard(threading.current_thread())

                cancel_thread = threading.Thread(
                    target=cancel_in_background,
                    name=f"RealtimeSTTProductionLiveCancel-{self.session_id}",
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

        live_queue = turn.live_queue
        if live_queue is None:
            return
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
            LOGGER.warning("Could not wake cancelled production live stream")


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
    try:
        submitted = service.scheduler.submit(job)
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
    ):
        del word_timestamps, vad_filter, condition_on_previous_text, without_timestamps
        if not _auth_ok(request.headers, settings.bearer_token):
            return unauthorized()
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
        max_audio_bytes = min(
            settings.max_http_audio_bytes,
            int(settings.max_turn_audio_seconds * sample_rate) * 2,
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
            "language_probability": None,
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
                elif command in ("finalize", "finish", "stop"):
                    response = await protocol.finalize()
                elif command == "reset":
                    response = await protocol.reset()
                elif command == "cancel":
                    response = await protocol.cancel()
                elif command == "ping":
                    response = {"type": "pong", "serverTime": time.time()}
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
