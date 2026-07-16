"""
Generate and compare final utterance JSON for a slow-CPU gap investigation.

This script streams a WAV file through the normal AudioToTextRecorder.text()
loop while audio keeps arriving in a separate feeder thread. That timing is
important: it can expose cases where final transcription is slow and the next
utterance begins before the application loop arms the recorder again.

Typical workflow:

1. Generate the expected utterance JSON with a fast/GPU configuration.
2. Run the same file with a CPU/int8 configuration and compare against it.

Example:
    python tests/final_transcription_gap_regression.py --mode generate
    python tests/final_transcription_gap_regression.py --mode compare
"""

import argparse
import difflib
import json
import os
import re
import sys
import threading
import time
import wave

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RealtimeSTT import AudioToTextRecorder


AUDIO_DIR = ROOT_DIR / "tests" / "unit" / "audio"
DEFAULT_AUDIO_FILE = AUDIO_DIR / "asr-reference.wav"
DEFAULT_EXPECTED_JSON = AUDIO_DIR / "asr-reference.expected_sentences.json"
INT16_MAX_ABS_VALUE = 32768.0


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GPU expected utterances and compare CPU output."
    )
    parser.add_argument("--mode", choices=("generate", "compare", "both"), default="both")
    parser.add_argument("--audio-file", type=Path, default=DEFAULT_AUDIO_FILE)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--chunk-ms", type=float, default=32.0)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 feeds audio in realtime.")
    parser.add_argument("--lead-silence", type=float, default=0.4)
    parser.add_argument("--tail-silence", type=float, default=1.2)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=2.5,
        help="Stop the text loop after feed completion once recorder is idle.",
    )
    parser.add_argument("--model", type=str, default="large-v2")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--gpu-device", type=str, default="cuda")
    parser.add_argument("--gpu-compute-type", type=str, default="default")
    parser.add_argument("--cpu-device", type=str, default="cpu")
    parser.add_argument("--cpu-compute-type", type=str, default="int8")
    parser.add_argument("--max-combined-wer", type=float, default=0.08)
    parser.add_argument("--max-segment-wer", type=float, default=0.35)
    parser.add_argument("--pre-recording-buffer-duration", type=float, default=2.0)
    parser.add_argument("--post-speech-silence-duration", type=float, default=0.6)
    parser.add_argument("--min-length-of-recording", type=float, default=0.5)
    parser.add_argument("--silero-sensitivity", type=float, default=0.05)
    parser.add_argument("--webrtc-sensitivity", type=int, default=3)
    parser.add_argument("--print-transcription-time", action="store_true")
    parser.add_argument(
        "--snippet-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-utterance WAV snippets. Defaults to a .snippets "
            "folder next to --expected-json."
        ),
    )
    parser.add_argument(
        "--no-save-snippets",
        action="store_true",
        help="Disable writing per-utterance WAV snippets.",
    )
    return parser.parse_args()


def read_wav_samples(path: Path) -> AudioData:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1).astype(np.int16)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=float(samples.size) / float(sample_rate),
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: List[str], right: List[str]) -> int:
    previous = list(range(len(right) + 1))

    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1]


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = normalize_text(expected).split()
    actual_words = normalize_text(actual).split()
    return edit_distance(expected_words, actual_words) / max(1, len(expected_words))


def diff_windows(expected: str, actual: str, context: int = 8, limit: int = 8) -> List[Dict[str, str]]:
    expected_words = normalize_text(expected).split()
    actual_words = normalize_text(actual).split()
    matcher = difflib.SequenceMatcher(a=expected_words, b=actual_words, autojunk=False)

    windows = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        exp_start = max(0, i1 - context)
        exp_end = min(len(expected_words), i2 + context)
        act_start = max(0, j1 - context)
        act_end = min(len(actual_words), j2 + context)

        windows.append({
            "type": tag,
            "expected": " ".join(expected_words[exp_start:exp_end]),
            "actual": " ".join(actual_words[act_start:act_end]),
        })

        if len(windows) >= limit:
            break

    return windows


def feed_samples(recorder, samples, sample_rate, chunk_size, speed):
    for start in range(0, samples.size, chunk_size):
        chunk = samples[start:start + chunk_size]
        recorder.feed_audio(chunk, original_sample_rate=sample_rate)
        time.sleep((chunk.size / float(sample_rate)) / speed)


def feed_silence(recorder, seconds, sample_rate, chunk_size, speed):
    if seconds <= 0:
        return

    samples = np.zeros(int(round(seconds * sample_rate)), dtype=np.int16)
    feed_samples(recorder, samples, sample_rate, chunk_size, speed)


def safe_label(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "pass"


def default_snippet_dir(args) -> Path:
    return args.expected_json.parent / f"{args.expected_json.stem}.snippets"


def write_wav_snippet(path: Path, audio: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * INT16_MAX_ABS_VALUE).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())


def save_transcription_snippet(args, label: str, index: int, recorder, audio: AudioData):
    if args.no_save_snippets:
        return None

    snippet_audio = getattr(recorder, "last_transcription_bytes", None)
    if snippet_audio is None or len(snippet_audio) == 0:
        return None

    pass_dir = args.snippet_dir / safe_label(label)
    filename = f"{index:02d}_{safe_label(label)}.wav"
    path = pass_dir / filename

    try:
        write_wav_snippet(path, snippet_audio, audio.sample_rate)
    except Exception as exc:
        return {
            "wav": str(path),
            "error": repr(exc),
            "duration_seconds": float(len(snippet_audio)) / float(audio.sample_rate),
            "samples": int(len(snippet_audio)),
        }

    return {
        "wav": str(path),
        "duration_seconds": float(len(snippet_audio)) / float(audio.sample_rate),
        "samples": int(len(snippet_audio)),
    }


def snippet_pass_dir(args, label: str) -> Path:
    return args.snippet_dir / safe_label(label)


def prepare_snippet_pass_dir(args, label: str):
    if args.no_save_snippets:
        return

    pass_dir = snippet_pass_dir(args, label)
    pass_dir.mkdir(parents=True, exist_ok=True)

    for path in pass_dir.glob("*.wav"):
        path.unlink()

    manifest_path = pass_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def write_run_manifest(args, result: Dict, audio: AudioData):
    if args.no_save_snippets:
        return None

    pass_dir = snippet_pass_dir(args, result["label"])
    manifest_path = pass_dir / "manifest.json"
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": result["label"],
        "device": result["device"],
        "compute_type": result["compute_type"],
        "model": result["model"],
        "audio_file": str(Path(args.audio_file).resolve()),
        "audio_duration_seconds": audio.duration,
        "sample_rate": audio.sample_rate,
        "settings": {
            "chunk_ms": args.chunk_ms,
            "speed": args.speed,
            "pre_recording_buffer_duration": args.pre_recording_buffer_duration,
            "post_speech_silence_duration": args.post_speech_silence_duration,
            "min_length_of_recording": args.min_length_of_recording,
        },
        "utterances": result["utterances"],
        "combined_text": result["combined_text"],
        "combined_normalized": result["combined_normalized"],
        "errors": result["errors"],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(manifest_path)


def make_recorder_config(args, device: str, compute_type: str):
    return {
        "use_microphone": False,
        "spinner": False,
        "model": args.model,
        "language": args.language,
        "device": device,
        "compute_type": compute_type,
        "enable_realtime_transcription": False,
        "min_gap_between_recordings": 0,
        "min_length_of_recording": args.min_length_of_recording,
        "post_speech_silence_duration": args.post_speech_silence_duration,
        "pre_recording_buffer_duration": args.pre_recording_buffer_duration,
        "silero_sensitivity": args.silero_sensitivity,
        "silero_deactivity_detection": True,
        "silero_use_onnx": True,
        "webrtc_sensitivity": args.webrtc_sensitivity,
        "faster_whisper_vad_filter": False,
        "no_log_file": True,
        "print_transcription_time": args.print_transcription_time,
    }


def run_streaming_text_pass(label: str, args, audio: AudioData, device: str, compute_type: str):
    print()
    print("=" * 80)
    print(label)
    print("=" * 80)
    print(f"device={device} compute_type={compute_type} model={args.model}")

    prepare_snippet_pass_dir(args, label)
    recorder = AudioToTextRecorder(**make_recorder_config(args, device, compute_type))
    utterances = []
    feed_done = threading.Event()
    text_done = threading.Event()
    feed_done_at = {"value": None}
    text_errors = []

    chunk_size = max(1, int(round(audio.sample_rate * args.chunk_ms / 1000.0)))
    speed = max(0.001, float(args.speed))

    def recorder_idle():
        has_pending_recordings = getattr(recorder, "has_pending_recordings", None)
        pending_recordings = (
            has_pending_recordings()
            if callable(has_pending_recordings)
            else False
        )

        return (
            not recorder.is_recording
            and not recorder.frames
            and not pending_recordings
        )

    def feeder():
        try:
            feed_silence(recorder, args.lead_silence, audio.sample_rate, chunk_size, speed)
            feed_samples(recorder, audio.samples, audio.sample_rate, chunk_size, speed)
            feed_silence(recorder, args.tail_silence, audio.sample_rate, chunk_size, speed)
            flush_buffered_audio = getattr(recorder, "flush_buffered_audio", None)
            if callable(flush_buffered_audio):
                flush_buffered_audio()
        finally:
            feed_done_at["value"] = time.time()
            feed_done.set()

    def text_loop():
        try:
            while True:
                text = recorder.text()
                if text:
                    index = len(utterances)
                    snippet = save_transcription_snippet(args, label, index, recorder, audio)
                    utterance = {
                        "index": index,
                        "text": text,
                        "normalized": normalize_text(text),
                        "received_at_seconds": time.time() - started_at,
                    }
                    if snippet:
                        utterance["snippet"] = snippet
                    utterances.append(utterance)
                    print(f"[{utterance['index']:02d}] {text}")

                if feed_done.is_set() and recorder_idle():
                    break
        except Exception as exc:
            text_errors.append(repr(exc))
        finally:
            text_done.set()

    try:
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)

        started_at = time.time()
        feeder_thread = threading.Thread(target=feeder, daemon=True)
        text_thread = threading.Thread(target=text_loop, daemon=True)
        feeder_thread.start()
        text_thread.start()

        while not text_done.is_set():
            if feed_done.is_set() and feed_done_at["value"] is not None:
                idle_for = time.time() - feed_done_at["value"]
                if (
                        idle_for >= args.idle_timeout
                        and recorder_idle()
                        and getattr(recorder, "transcribe_count", 0) == 0):
                    recorder.interrupt_stop_event.set()
                    break

            time.sleep(0.05)

        text_thread.join(timeout=30)
        feeder_thread.join(timeout=5)

        if text_thread.is_alive():
            text_errors.append("text loop did not exit within timeout")

        elapsed = time.time() - started_at
        combined_text = " ".join(utterance["text"] for utterance in utterances)

        result = {
            "label": label,
            "device": device,
            "compute_type": compute_type,
            "model": args.model,
            "elapsed_seconds": elapsed,
            "audio_duration_seconds": audio.duration,
            "utterances": utterances,
            "combined_text": combined_text,
            "combined_normalized": normalize_text(combined_text),
            "errors": text_errors,
        }
        manifest_path = write_run_manifest(args, result, audio)
        if manifest_path:
            result["snippet_manifest"] = manifest_path
            print(f"Wrote snippet manifest: {manifest_path}")

        return result
    finally:
        recorder.shutdown()


def write_expected_json(path: Path, args, audio: AudioData, result: Dict):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio_file": str(Path(args.audio_file).resolve()),
        "audio_duration_seconds": audio.duration,
        "sample_rate": audio.sample_rate,
        "settings": {
            "model": args.model,
            "language": args.language,
            "device": result["device"],
            "compute_type": result["compute_type"],
            "chunk_ms": args.chunk_ms,
            "speed": args.speed,
            "pre_recording_buffer_duration": args.pre_recording_buffer_duration,
            "post_speech_silence_duration": args.post_speech_silence_duration,
            "min_length_of_recording": args.min_length_of_recording,
            "snippet_dir": None if args.no_save_snippets else str(args.snippet_dir),
        },
        "utterances": result["utterances"],
        "combined_text": result["combined_text"],
        "combined_normalized": result["combined_normalized"],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote expected JSON: {path}")


def load_expected_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_against_expected(expected, actual, args) -> bool:
    expected_combined = expected.get("combined_text", "")
    actual_combined = actual.get("combined_text", "")
    combined_wer = word_error_rate(expected_combined, actual_combined)

    print()
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"expected utterances: {len(expected.get('utterances', []))}")
    print(f"actual utterances:   {len(actual.get('utterances', []))}")
    print(f"combined WER:        {combined_wer:.3f}")
    print(f"allowed WER:         {args.max_combined_wer:.3f}")

    ok = combined_wer <= args.max_combined_wer and not actual.get("errors")

    expected_utterances = expected.get("utterances", [])
    actual_utterances = actual.get("utterances", [])
    max_count = max(len(expected_utterances), len(actual_utterances))

    print()
    print("Utterance comparison:")
    for index in range(max_count):
        expected_text = expected_utterances[index]["text"] if index < len(expected_utterances) else ""
        actual_text = actual_utterances[index]["text"] if index < len(actual_utterances) else ""
        segment_wer = word_error_rate(expected_text, actual_text) if expected_text else 1.0
        segment_ok = bool(expected_text and actual_text and segment_wer <= args.max_segment_wer)
        ok = ok and segment_ok

        status = "OK" if segment_ok else "DIFF"
        print(f"[{index:02d}] {status} WER={segment_wer:.3f}")
        print(f"  expected: {expected_text}")
        print(f"  actual:   {actual_text}")

    diff = diff_windows(expected_combined, actual_combined)
    if diff:
        print()
        print("First word-level diff windows:")
        for item in diff:
            print(f"- {item['type']}")
            print(f"  expected: {item['expected']}")
            print(f"  actual:   {item['actual']}")

    if actual.get("errors"):
        print()
        print("Actual run errors:")
        for error in actual["errors"]:
            print(f"- {error}")

    print()
    print(f"PASS: {ok}")
    return ok


def main():
    args = parse_args()
    args.audio_file = Path(args.audio_file)
    args.expected_json = Path(args.expected_json)
    if args.snippet_dir is None:
        args.snippet_dir = default_snippet_dir(args)

    audio = read_wav_samples(args.audio_file)
    print(f"Loaded {args.audio_file}")
    print(f"sample_rate={audio.sample_rate}Hz duration={audio.duration:.2f}s")
    if args.no_save_snippets:
        print("snippet dumping disabled")
    else:
        print(f"snippet_dir={args.snippet_dir}")

    expected = None

    if args.mode in ("generate", "both"):
        gpu_result = run_streaming_text_pass(
            "GPU expected pass",
            args,
            audio,
            args.gpu_device,
            args.gpu_compute_type,
        )
        write_expected_json(args.expected_json, args, audio, gpu_result)
        expected = load_expected_json(args.expected_json)

    if args.mode in ("compare", "both"):
        if expected is None:
            expected = load_expected_json(args.expected_json)

        cpu_result = run_streaming_text_pass(
            "CPU comparison pass",
            args,
            audio,
            args.cpu_device,
            args.cpu_compute_type,
        )

        if not compare_against_expected(expected, cpu_result, args):
            sys.exit(1)


if __name__ == "__main__":
    main()
