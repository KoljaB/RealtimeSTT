import json
import struct
from dataclasses import dataclass
from typing import Any, Dict


MAX_METADATA_BYTES = 64 * 1024


class AudioPacketError(ValueError):
    pass


@dataclass(frozen=True)
class AudioPacket:
    metadata: Dict[str, Any]
    audio: bytes


def normalize_engine_name(name):
    if name is None:
        return None
    return str(name).strip().lower().replace("-", "_")


def parse_json_object(value, name):
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return parsed


def encode_audio_packet(metadata, audio):
    if not isinstance(metadata, dict):
        raise AudioPacketError("metadata must be a JSON object")
    if not isinstance(audio, (bytes, bytearray, memoryview)):
        raise AudioPacketError("audio must be bytes-like")

    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise AudioPacketError("metadata is too large")
    return struct.pack("<I", len(metadata_bytes)) + metadata_bytes + bytes(audio)


def decode_audio_packet(message):
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise AudioPacketError("audio packet must be binary")

    data = bytes(message)
    if len(data) < 4:
        raise AudioPacketError("audio packet is missing metadata length")

    metadata_length = struct.unpack("<I", data[:4])[0]
    if metadata_length > MAX_METADATA_BYTES:
        raise AudioPacketError("audio packet metadata is too large")
    if len(data) < 4 + metadata_length:
        raise AudioPacketError("audio packet metadata is incomplete")

    metadata_bytes = data[4:4 + metadata_length]
    audio = data[4 + metadata_length:]
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioPacketError("audio packet metadata is invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise AudioPacketError("audio packet metadata must be a JSON object")

    return AudioPacket(metadata=metadata, audio=audio)


def require_positive_int(metadata, key):
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AudioPacketError(f"audio packet metadata field '{key}' must be a positive integer")
    return value
