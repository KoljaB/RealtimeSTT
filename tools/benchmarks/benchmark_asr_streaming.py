#!/usr/bin/env python3
"""Benchmark the versioned RealtimeSTT WebSocket streaming contract.

The benchmark deliberately exercises the public wire protocol instead of a
private server object.  Each turn sends length-prefixed ``pcm_s16le`` packets
while a receiver task consumes events concurrently.  The resulting JSON file
contains protocol, latency, partial-hypothesis, and final-quality metrics; a
human-readable Markdown rendering is written next to it.

The only authentication input is ``REALTIMESTT_SERVER_BEARER_TOKEN``.  It is
used as a request header and is never copied into reports or error text.
"""

from __future__ import annotations

import argparse
import asyncio
from array import array
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import wave
from typing import Any, Iterable, Mapping, Sequence


SERVER_SAMPLE_RATE = 16_000
SUPPORTED_SOURCE_RATES = (8_000, 16_000, 24_000, 32_000, 44_100, 48_000)
PCM_FORMAT = "pcm_s16le"
MAX_METADATA_BYTES = 64 * 1024

_LANGUAGE_ALIASES = {
    "auto": "auto",
    "english": "en",
    "en": "en",
    "en-us": "en",
    "en_us": "en",
    "en-gb": "en",
    "en_gb": "en",
    "german": "de",
    "de": "de",
    "de-de": "de",
    "de_de": "de",
    "french": "fr",
    "fr": "fr",
    "fr-fr": "fr",
    "fr_fr": "fr",
    "spanish": "es",
    "es": "es",
    "es-es": "es",
    "es_es": "es",
    "italian": "it",
    "it": "it",
    "it-it": "it",
    "it_it": "it",
    "portuguese": "pt",
    "pt": "pt",
    "pt-pt": "pt",
    "pt_pt": "pt",
    "pt-br": "pt",
    "pt_br": "pt",
    "russian": "ru",
    "ru": "ru",
    "ru-ru": "ru",
    "ru_ru": "ru",
}


@dataclass(frozen=True)
class StreamClip:
    """One manifest record after WAV decoding and 16 kHz conversion."""

    clip_id: str
    expected_language: str
    reference: str
    reference_kind: str
    wav_path: Path
    pcm16: bytes
    audio_duration_s: float
    source_sample_rate: int


def map_language(value: Any) -> str:
    """Map language names/codes used by AgentTalk manifests to ISO-like codes.

    ``expected_detected_language`` is often a display name such as
    ``"German"`` while ``requested_language`` is ``"Auto"``.  Unknown values
    are retained in normalized form so a manifest never silently changes its
    quality bucket.
    """

    normalized = str(value or "").strip().casefold().replace("–", "-")
    if normalized in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[normalized]
    return normalized or "unknown"


def normalize_text(value: Any) -> str:
    """Normalize text consistently for WER/CER/exact-match measurements."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(
        "".join(char if char.isalnum() else " " for char in normalized).split()
    )


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _read_manifest_records(manifest: Path) -> list[dict[str, Any]]:
    if manifest.suffix.casefold() == ".jsonl":
        payload: Any = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("turns", "clips", "samples", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("manifest must contain a list or turns/clips/samples list")
    return [record for record in payload if isinstance(record, dict)]


def _resolve_wav(manifest: Path, value: str) -> Path:
    supplied = Path(value.replace("\\", "/"))
    candidates = (supplied, manifest.parent / supplied, manifest.parent / supplied.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {value!r} relative to {manifest}")


def _to_little_endian_pcm16(raw: bytes) -> bytes:
    if len(raw) % 2:
        raise ValueError("PCM16 WAV payload is not aligned to whole samples")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def resample_pcm16(raw: bytes, source_rate: int) -> bytes:
    """Convert little-endian PCM16 to 16 kHz with an anti-aliased boundary.

    ``scipy.signal.resample_poly`` is intentionally required for non-native
    rates.  Linear interpolation can alias microphone content and would make
    the benchmark measure a different audio signal than the production path.
    """

    source_rate = int(source_rate)
    if source_rate not in SUPPORTED_SOURCE_RATES:
        raise ValueError(
            f"source sample rate {source_rate} is unsupported; "
            f"choose one of {list(SUPPORTED_SOURCE_RATES)}"
        )
    raw = _to_little_endian_pcm16(bytes(raw))
    if source_rate == SERVER_SAMPLE_RATE or not raw:
        return raw
    try:
        import numpy as np
        from scipy.signal import resample_poly
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "scipy and numpy are required for anti-aliased WAV resampling; "
            "install the benchmark/server dependencies or provide 16 kHz WAV"
        ) from exc

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    divisor = math.gcd(source_rate, SERVER_SAMPLE_RATE)
    up = SERVER_SAMPLE_RATE // divisor
    down = source_rate // divisor
    converted = resample_poly(samples, up, down)
    expected_count = round(len(samples) * SERVER_SAMPLE_RATE / source_rate)
    if len(converted) > expected_count:
        converted = converted[:expected_count]
    elif len(converted) < expected_count:
        converted = np.pad(converted, (0, expected_count - len(converted)))
    converted = np.clip(np.rint(converted), -32768, 32767).astype("<i2")
    return converted.tobytes()


def _read_pcm16_wav(path: Path) -> tuple[bytes, int, float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        source_rate = handle.getframerate()
        compression = handle.getcomptype()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if channels != 1 or sample_width != 2 or compression != "NONE":
        raise ValueError(f"{path} must be an uncompressed mono PCM16 WAV")
    if source_rate not in SUPPORTED_SOURCE_RATES:
        raise ValueError(
            f"{path} uses {source_rate} Hz; supported rates are "
            f"{list(SUPPORTED_SOURCE_RATES)}"
        )
    pcm16 = resample_pcm16(raw, source_rate)
    return pcm16, source_rate, frame_count / float(source_rate)


def load_manifest(manifest: Path | str, limit: int | None = None) -> list[StreamClip]:
    """Load the flexible manifest shapes accepted by the HTTP A/B tool."""

    manifest = Path(manifest).resolve()
    clips: list[StreamClip] = []
    for index, item in enumerate(_read_manifest_records(manifest), start=1):
        wav_value = item.get("wav") or item.get("wav_path") or item.get("file")
        if not wav_value:
            continue
        wav_path = _resolve_wav(manifest, str(wav_value))
        pcm16, source_rate, duration_s = _read_pcm16_wav(wav_path)
        reference = str(
            item.get("reference_text")
            or item.get("text")
            or item.get("asr_text")
            or ""
        )
        expected_language = map_language(
            item.get("expected_detected_language")
            or item.get("expected_language")
            or item.get("language")
            or item.get("requested_language")
            or "unknown"
        )
        clip = StreamClip(
            clip_id=str(
                item.get("turn_id")
                or item.get("clip_id")
                or item.get("generation")
                or item.get("id")
                or index
            ),
            expected_language=expected_language,
            reference=reference,
            reference_kind=str(
                item.get("reference_kind")
                or ("ground_truth" if reference else "unknown")
            ),
            wav_path=wav_path,
            pcm16=pcm16,
            audio_duration_s=duration_s,
            source_sample_rate=source_rate,
        )
        clips.append(clip)
        if limit is not None and len(clips) >= limit:
            break
    if not clips:
        raise ValueError(f"No WAV records found in {manifest}")
    return clips


def encode_audio_packet(metadata: Mapping[str, Any], audio: bytes) -> bytes:
    """Encode the production server's length-prefixed binary audio packet."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a JSON object")
    if not isinstance(audio, (bytes, bytearray, memoryview)):
        raise TypeError("audio must be bytes-like")
    metadata_bytes = json.dumps(dict(metadata), separators=(",", ":")).encode("utf-8")
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ValueError("audio packet metadata is too large")
    return struct.pack("<I", len(metadata_bytes)) + metadata_bytes + bytes(audio)


def decode_audio_packet(message: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode a packet for local shape tests and diagnostics."""

    data = bytes(message)
    if len(data) < 4:
        raise ValueError("audio packet is missing metadata length")
    metadata_length = struct.unpack("<I", data[:4])[0]
    if metadata_length > MAX_METADATA_BYTES or len(data) < 4 + metadata_length:
        raise ValueError("audio packet metadata is incomplete or too large")
    try:
        metadata = json.loads(data[4 : 4 + metadata_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("audio packet metadata is invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("audio packet metadata must be an object")
    return metadata, data[4 + metadata_length :]


def chunk_pcm16(pcm16: bytes, chunk_ms: float) -> list[tuple[int, bytes]]:
    """Split PCM16 into packets whose sequence starts at zero and is contiguous."""

    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be greater than zero")
    samples_per_chunk = max(1, round(SERVER_SAMPLE_RATE * float(chunk_ms) / 1000.0))
    bytes_per_chunk = samples_per_chunk * 2
    if len(pcm16) % 2:
        raise ValueError("PCM16 stream is not aligned to whole samples")
    return [
        (sequence, pcm16[offset : offset + bytes_per_chunk])
        for sequence, offset in enumerate(range(0, len(pcm16), bytes_per_chunk))
    ]


def validate_audio_sequences(sequences: Iterable[int]) -> dict[str, Any]:
    values = list(sequences)
    expected = list(range(len(values)))
    violations = [
        {"index": index, "expected": expected_value, "received": received}
        for index, (expected_value, received) in enumerate(zip(expected, values))
        if received != expected_value
    ]
    return {
        "valid": not violations,
        "count": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "violations": violations,
    }


def validate_event_sequences(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [event.get("eventSequence") for event in events]
    violations: list[dict[str, Any]] = []
    previous: int | None = None
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            violations.append({"index": index, "expected": "integer", "received": value})
            previous = None
            continue
        if previous is not None and value != previous + 1:
            violations.append({"index": index, "expected": previous + 1, "received": value})
        previous = value
    return {
        "valid": not violations and bool(values),
        "count": len(values),
        "first": values[0] if values and isinstance(values[0], int) else None,
        "last": values[-1] if values and isinstance(values[-1], int) else None,
        "violations": violations,
    }


def partial_prefix_monotonicity(partials: Sequence[str]) -> dict[str, Any]:
    normalized = [normalize_text(value) for value in partials]
    comparisons = max(0, len(normalized) - 1)
    monotonic = sum(
        current.startswith(previous)
        for previous, current in zip(normalized, normalized[1:])
    )
    return {
        "updates": len(normalized),
        "comparisons": comparisons,
        "prefix_monotonic_comparisons": monotonic,
        "prefix_monotonic_rate": monotonic / comparisons if comparisons else None,
        "revisions_observed": max(0, comparisons - monotonic),
    }


def hypothesis_to_final_semantics(partials: Sequence[str], final_text: str) -> dict[str, Any]:
    latest = str(partials[-1] if partials else "").strip()
    final = str(final_text or "").strip()
    return {
        "partial_count": len(partials),
        "latest_partial": latest,
        "final_text": final,
        "replacement_required": bool(latest and final and normalize_text(latest) != normalize_text(final)),
        "latest_partial_matches_final": bool(latest and final and normalize_text(latest) == normalize_text(final)),
    }


def _accuracy_for(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [record for record in records if normalize_text(record.get("reference"))]
    word_errors = reference_words = char_errors = reference_chars = exact = 0
    for record in selected:
        reference = normalize_text(record.get("reference"))
        hypothesis = normalize_text(record.get("final_text"))
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        word_errors += edit_distance(reference_tokens, hypothesis_tokens)
        reference_words += len(reference_tokens)
        reference_characters = list(reference.replace(" ", ""))
        hypothesis_characters = list(hypothesis.replace(" ", ""))
        char_errors += edit_distance(reference_characters, hypothesis_characters)
        reference_chars += len(reference_characters)
        exact += reference == hypothesis
    return {
        "count": len(selected),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "wer": word_errors / max(1, reference_words),
        "cer": char_errors / max(1, reference_chars),
        "exact_match_rate": exact / max(1, len(selected)),
        "empty_final_hypotheses": sum(not normalize_text(record.get("final_text")) for record in selected),
    }


def accuracy_by_language(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        language = map_language(record.get("expected_language") or "unknown")
        buckets.setdefault(language, []).append(record)
    return {
        "overall": _accuracy_for(records),
        "by_language": {
            language: _accuracy_for(bucket)
            for language, bucket in sorted(buckets.items())
        },
    }


def _websocket_url(url: str) -> str:
    value = str(url).strip().rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise ValueError("url must start with http://, https://, ws://, or wss://")
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path or ""
    # Preserve an explicitly selected endpoint.  In particular, the
    # production server's OpenAI-shaped alias is already a WebSocket route;
    # appending ``/api/v1/ws/transcribe`` would silently target another API.
    explicit_paths = {
        "/api/v1/ws",
        "/api/v1/ws/transcribe",
        "/v1/ws/transcribe",
        "/v1/audio/transcriptions/stream",
    }
    if (
        path in explicit_paths
        or path.endswith("/api/v1/ws")
        or path.endswith("/api/v1/ws/transcribe")
        or path.endswith("/audio/transcriptions/stream")
    ):
        websocket_path = path
    else:
        websocket_path = path.rstrip("/") + "/api/v1/ws/transcribe"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, websocket_path, parsed.query, ""))


def _event_text(event: Mapping[str, Any]) -> str:
    value = event.get("partialText") if event.get("type") == "partial" else event.get("text")
    if value is None:
        value = event.get("text")
    return str(value or "").strip()


async def _connect_websocket(url: str, token: str | None, timeout_s: float):
    try:
        import websockets
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "websockets is required for the streaming benchmark; install the server extras"
        ) from exc
    headers = {"Authorization": f"Bearer {token}"} if token else None
    options: dict[str, Any] = {
        "open_timeout": timeout_s,
        "close_timeout": timeout_s,
    }
    if headers:
        # websockets 12 calls this extra_headers; newer releases renamed it.
        options["extra_headers"] = headers
    try:
        return await websockets.connect(url, **options)
    except TypeError as exc:
        if "extra_headers" not in options:
            raise
        options.pop("extra_headers", None)
        options["additional_headers"] = headers
        try:
            return await websockets.connect(url, **options)
        except TypeError:
            raise exc


async def _stream_clip(
    clip: StreamClip,
    *,
    url: str,
    language_mode: str,
    fixed_language: str,
    chunk_ms: float,
    pace: float,
    timeout_s: float,
    token: str | None,
) -> dict[str, Any]:
    packet_chunks = chunk_pcm16(clip.pcm16, chunk_ms)
    packet_sequences = [sequence for sequence, _ in packet_chunks]
    packet_integrity = validate_audio_sequences(packet_sequences)
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    receive_done = asyncio.Event()
    completion_seen = asyncio.Event()
    stream_started = time.perf_counter()
    finalize_sent_at: float | None = None

    async def receive_events(websocket) -> None:
        try:
            while not completion_seen.is_set():
                raw = await websocket.recv()
                received_at = time.perf_counter()
                if isinstance(raw, bytes):
                    errors.append("server sent an unexpected binary message")
                    continue
                try:
                    event = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid server event JSON: {exc}")
                    continue
                if not isinstance(event, dict):
                    errors.append("server event must be a JSON object")
                    continue
                event["_received_at"] = received_at
                events.append(event)
                if event.get("type") == "error":
                    error = event.get("error")
                    code = error.get("code") if isinstance(error, dict) else event.get("code")
                    errors.append(f"server error: {code or 'unknown'}")
                if event.get("type") == "completion":
                    completion_seen.set()
        except Exception as exc:
            # A normal close after completion is not an error.  The caller
            # reports a missing completion if the close happened too soon.
            if not completion_seen.is_set():
                errors.append(f"WebSocket receive failed: {type(exc).__name__}: {exc}")
        finally:
            receive_done.set()

    request_language = "auto" if language_mode == "auto" else map_language(fixed_language)
    if request_language == "auto" and language_mode != "auto":
        raise ValueError("fixed language mode requires a concrete --language")

    try:
        websocket = await _connect_websocket(_websocket_url(url), token, timeout_s)
        try:
            # Exclude the TCP/TLS/WebSocket handshake from stream latency.  A
            # deployment can report connection setup separately; these values
            # should describe the ASR turn after the protocol is established.
            stream_started = time.perf_counter()
            receiver = asyncio.create_task(receive_events(websocket))
            try:
                await asyncio.wait_for(
                    websocket.send(
                        json.dumps(
                            {
                                "type": "start",
                                "turnId": f"benchmark-{clip.clip_id}",
                                "language": request_language,
                            },
                            separators=(",", ":"),
                        )
                    ),
                    timeout=timeout_s,
                )
                for index, (sequence, audio) in enumerate(packet_chunks):
                    packet = encode_audio_packet(
                        {
                            "sampleRate": SERVER_SAMPLE_RATE,
                            "channels": 1,
                            "format": PCM_FORMAT,
                            "frames": len(audio) // 2,
                            "audioSequence": sequence,
                        },
                        audio,
                    )
                    await asyncio.wait_for(websocket.send(packet), timeout=timeout_s)
                    if pace > 0 and index + 1 < len(packet_chunks):
                        await asyncio.sleep((len(audio) / 2 / SERVER_SAMPLE_RATE) * pace)
                finalize_sent_at = time.perf_counter()
                await asyncio.wait_for(
                    websocket.send(json.dumps({"type": "finalize"}, separators=(",", ":"))),
                    timeout=timeout_s,
                )
                try:
                    await asyncio.wait_for(completion_seen.wait(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    errors.append(f"completion was not received within {timeout_s:g} seconds")
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        finally:
            # The protocol object returned by websockets 12 is not itself an
            # asynchronous context manager.  Explicit close works on both the
            # legacy protocol and newer client-connection implementations.
            close = getattr(websocket, "close", None)
            if callable(close):
                await close()
    except Exception as exc:
        errors.append(f"WebSocket stream failed: {type(exc).__name__}: {exc}")

    event_sequence = validate_event_sequences(events)
    partial_events = [event for event in events if event.get("type") == "partial"]
    partials = [_event_text(event) for event in partial_events if _event_text(event)]
    final_events = [event for event in events if event.get("type") == "final"]
    completion_events = [event for event in events if event.get("type") == "completion"]
    final_event = final_events[0] if final_events else None
    completion_event = completion_events[0] if completion_events else None
    final_text = _event_text(final_event or {})
    final_at = final_event.get("_received_at") if final_event else None
    completion_at = completion_event.get("_received_at") if completion_event else None
    first_partial_at = next(
        (event.get("_received_at") for event in partial_events), None
    )
    first_nonempty_partial_at = next(
        (event.get("_received_at") for event in partial_events if _event_text(event)),
        None,
    )
    if not final_events:
        errors.append("missing final event")
    if not completion_events:
        errors.append("missing completion event")
    if completion_event and completion_event.get("status") != "completed":
        errors.append(f"completion status was {completion_event.get('status')!r}")
    if not event_sequence["valid"]:
        errors.append("eventSequence was not contiguous and strictly increasing")
    if errors and any("server error:" in error for error in errors):
        # Keep the original machine-readable error code in the report but do
        # not include raw server payloads that could accidentally contain a
        # credential or deployment detail.
        errors = list(dict.fromkeys(errors))

    def elapsed(timestamp: float | None, origin: float) -> float | None:
        return timestamp - origin if timestamp is not None else None

    replacement = hypothesis_to_final_semantics(partials, final_text)
    replacement["final_latency_from_finalize_s"] = (
        completion_at - finalize_sent_at if completion_at is not None and finalize_sent_at else None
    )
    return {
        "clip_id": clip.clip_id,
        "expected_language": clip.expected_language,
        "reference": clip.reference,
        "reference_kind": clip.reference_kind,
        "wav_path": str(clip.wav_path),
        "source_sample_rate": clip.source_sample_rate,
        "audio_duration_s": clip.audio_duration_s,
        "packet_count": len(packet_chunks),
        "audio_sequence": packet_integrity,
        "event_sequence": event_sequence,
        "event_count": len(events),
        "event_types": [event.get("type") for event in events],
        "errors": list(dict.fromkeys(errors)),
        "ok": not errors,
        "partial_count": len(partials),
        "partial_texts": partials,
        "partial_latency_s": {
            "first_partial": elapsed(first_partial_at, stream_started),
            "first_nonempty_partial": elapsed(first_nonempty_partial_at, stream_started),
        },
        "final_latency_s": elapsed(final_at, stream_started),
        "completion_latency_s": elapsed(completion_at, stream_started),
        "completion_latency_from_finalize_s": (
            completion_at - finalize_sent_at
            if completion_at is not None and finalize_sent_at is not None
            else None
        ),
        "final_text": final_text,
        "partial_prefix_monotonicity": partial_prefix_monotonicity(partials),
        "hypothesis_to_final": replacement,
    }


def _summary(values: Iterable[float | None]) -> dict[str, float | None]:
    selected = sorted(float(value) for value in values if value is not None)
    if not selected:
        return {"median": None, "p90": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        position = (len(selected) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return selected[lower] if lower == upper else selected[lower] * (upper - position) + selected[upper] * (position - lower)

    return {
        "median": selected[len(selected) // 2] if len(selected) % 2 else (selected[len(selected) // 2 - 1] + selected[len(selected) // 2]) / 2,
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": selected[-1],
    }


async def _run_async(
    clips: Sequence[StreamClip],
    *,
    url: str,
    language_mode: str,
    fixed_language: str,
    chunk_ms: float,
    pace: float,
    timeout_s: float,
    token: str | None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for clip in clips:
        records.append(
            await _stream_clip(
                clip,
                url=url,
                language_mode=language_mode,
                fixed_language=fixed_language,
                chunk_ms=chunk_ms,
                pace=pace,
                timeout_s=timeout_s,
                token=token,
            )
        )
    return {
        "kind": "realtimestt_asr_websocket_streaming_benchmark",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "websocket": "/api/v1/ws/transcribe",
            "protocol_version": "realtimestt.remote.v1",
            "audio": "length-prefixed JSON metadata + mono little-endian PCM16",
            "event_order": "strictly increasing contiguous eventSequence",
        },
        "config": {
            "url": url.rstrip("/"),
            "chunk_ms": chunk_ms,
            "pace": pace,
            "timeout_s": timeout_s,
            "language_mode": language_mode,
            "fixed_language": map_language(fixed_language),
        },
        "corpus": {
            "clips": len(clips),
            "audio_seconds": sum(clip.audio_duration_s for clip in clips),
        },
        "ok": all(record["ok"] for record in records),
        "metrics": {
            "accuracy": accuracy_by_language(records),
            "latency": {
                "first_partial_s": _summary(record["partial_latency_s"]["first_partial"] for record in records),
                "first_nonempty_partial_s": _summary(record["partial_latency_s"]["first_nonempty_partial"] for record in records),
                "final_s": _summary(record["final_latency_s"] for record in records),
                "completion_s": _summary(record["completion_latency_s"] for record in records),
                "completion_from_finalize_s": _summary(record["completion_latency_from_finalize_s"] for record in records),
            },
            "partial_prefix_monotonicity": {
                "clips": len(records),
                "updates": sum(record["partial_prefix_monotonicity"]["updates"] for record in records),
                "comparisons": sum(record["partial_prefix_monotonicity"]["comparisons"] for record in records),
                "monotonic_comparisons": sum(record["partial_prefix_monotonicity"]["prefix_monotonic_comparisons"] for record in records),
                "revision_clips": sum(bool(record["partial_prefix_monotonicity"]["revisions_observed"]) for record in records),
            },
            "protocol": {
                "event_sequence_failures": sum(not record["event_sequence"]["valid"] for record in records),
                "audio_sequence_failures": sum(not record["audio_sequence"]["valid"] for record in records),
                "missing_final": sum("missing final event" in record["errors"] for record in records),
                "missing_completion": sum("missing completion event" in record["errors"] for record in records),
            },
        },
        "records": records,
    }


def run_benchmark(
    clips: Sequence[StreamClip],
    *,
    url: str,
    language_mode: str = "auto",
    fixed_language: str = "en",
    chunk_ms: float = 100.0,
    pace: float = 1.0,
    timeout_s: float = 60.0,
    token: str | None = None,
) -> dict[str, Any]:
    """Synchronous API used by the CLI and integration callers."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero")
    if pace < 0:
        raise ValueError("pace must be zero or greater")
    return asyncio.run(
        _run_async(
            clips,
            url=url,
            language_mode=language_mode,
            fixed_language=fixed_language,
            chunk_ms=chunk_ms,
            pace=pace,
            timeout_s=timeout_s,
            token=token,
        )
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown_report(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    accuracy = metrics["accuracy"]
    lines = [
        "# RealtimeSTT WebSocket streaming benchmark",
        "",
        f"- URL: `{report['config']['url']}`",
        f"- Corpus: {report['corpus']['clips']} clips / {report['corpus']['audio_seconds']:.3f} s",
        f"- Chunk: {report['config']['chunk_ms']:.1f} ms; pace factor: {report['config']['pace']:.3f}",
        f"- Overall status: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "## Final quality",
        "",
        "| Language | Clips | WER | CER | Exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for language, values in [("overall", accuracy["overall"]), *sorted(accuracy["by_language"].items())]:
        lines.append(
            f"| {language} | {values['count']} | {_format_metric(values['wer'])} | "
            f"{_format_metric(values['cer'])} | {_format_metric(values['exact_match_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Latency (seconds)",
            "",
            "| Event | Median | p90 | p95 | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, field in (
        ("First partial", "first_partial_s"),
        ("First non-empty partial", "first_nonempty_partial_s"),
        ("Final", "final_s"),
        ("Completion", "completion_s"),
        ("Completion after finalize", "completion_from_finalize_s"),
    ):
        values = metrics["latency"][field]
        lines.append(
            f"| {label} | {_format_metric(values['median'])} | {_format_metric(values['p90'])} | "
            f"{_format_metric(values['p95'])} | {_format_metric(values['max'])} |"
        )
    partials = metrics["partial_prefix_monotonicity"]
    protocol = metrics["protocol"]
    lines.extend(
        [
            "",
            "## Protocol and hypothesis observations",
            "",
            f"- Partial updates: {partials['updates']}; comparisons: {partials['comparisons']}; "
            f"revision clips: {partials['revision_clips']}",
            f"- Event-sequence failures: {protocol['event_sequence_failures']}; "
            f"audio-sequence failures: {protocol['audio_sequence_failures']}",
            f"- Missing final events: {protocol['missing_final']}; "
            f"missing completions: {protocol['missing_completion']}",
            "",
            "## Clips",
            "",
            "| Clip | Language | Final text | Replacement | Status |",
            "|---|---|---|---|---|",
        ]
    )
    for record in report["records"]:
        final_text = record["final_text"].replace("|", "\\|").replace("\n", " ")
        replacement = "yes" if record["hypothesis_to_final"]["replacement_required"] else "no"
        status = "PASS" if record["ok"] else "FAIL: " + "; ".join(record["errors"])
        lines.append(f"| {record['clip_id']} | {record['expected_language']} | {final_text} | {replacement} | {status} |")
    return "\n".join(lines) + "\n"


def write_reports(report: Mapping[str, Any], output: Path | str) -> tuple[Path, Path]:
    output = Path(output)
    markdown_path = output.with_suffix(".md")
    _atomic_write(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(markdown_path, markdown_report(report))
    return output, markdown_path


def _print_utf8(value: str) -> None:
    """Print reports containing the seven-language corpus on Windows too."""

    try:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
        print(value)
    except UnicodeEncodeError:
        # A redirected/custom stream may not implement ``reconfigure``.
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            raise
        buffer.write(value.encode("utf-8"))
        buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", required=True, help="HTTP(S) or WS(S) server base URL")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path; Markdown is written beside it")
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--pace", type=float, default=1.0, help="Realtime pacing multiplier; zero sends as fast as possible")
    parser.add_argument("--timeout", "--timeout-s", dest="timeout_s", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--language-mode", choices=("fixed", "auto"), default="auto")
    parser.add_argument("--language", default="en", help="Fixed-mode language code/name (default: en)")
    args = parser.parse_args(argv)
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be greater than zero")
    if args.pace < 0:
        parser.error("--pace must be zero or greater")
    if args.timeout_s <= 0:
        parser.error("--timeout must be greater than zero")
    clips = load_manifest(args.manifest, args.limit)
    report = run_benchmark(
        clips,
        url=args.url,
        language_mode=args.language_mode,
        fixed_language=args.language,
        chunk_ms=args.chunk_ms,
        pace=args.pace,
        timeout_s=args.timeout_s,
        token=os.environ.get("REALTIMESTT_SERVER_BEARER_TOKEN") or None,
    )
    write_reports(report, args.output)
    _print_utf8(markdown_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
