"""
Compare realtime transcription counts using a deterministic WAV input file.

The script feeds the same audio into AudioToTextRecorder twice:
1. timer-based realtime transcription
2. syllable-boundary realtime transcription

It reports realtime model-call counters and validates the final transcription
against the expected text. This is a manual integration benchmark, not a
unittest.
"""

import argparse
import os
import re
import sys
import time
import wave

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from RealtimeSTT import AudioToTextRecorder


DEFAULT_AUDIO_FILE = os.path.join(ROOT_DIR, "tests", "unit", "audio", "asr-reference-short.wav")
DEFAULT_EXPECTED_TEXT = (
    "Hey guys! Welcome to the new demo of my real-time transcription library, "
    "designed to showcase its lightning-fast capabilities. As you'll see, "
    "speech is transcribed almost instantly into text"
)


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration: float


def parse_delays(value):
    if value is None:
        return (0.05, 0.2)

    value = str(value).strip()
    if not value:
        return ()

    delays = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        delays.append(float(part))
    return tuple(delays)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare realtime transcription counts for a WAV file."
    )
    parser.add_argument("--audio-file", type=str, default=DEFAULT_AUDIO_FILE)
    parser.add_argument("--expected-text", type=str, default=DEFAULT_EXPECTED_TEXT)
    parser.add_argument("--mode", choices=("both", "timer", "syllable"), default="both")
    parser.add_argument("--chunk-ms", type=float, default=32.0, help="Audio chunk size fed to the recorder.")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 feeds audio in realtime.")
    parser.add_argument(
        "--lead-silence",
        type=float,
        default=0.2,
        help="Seconds of silence to feed before the file.",
    )
    parser.add_argument(
        "--tail-silence",
        type=float,
        default=0.8,
        help="Seconds of silence to feed after the file so followups can fire.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Extra pause after recorder init. Warmup is already done before this.",
    )
    parser.add_argument(
        "--post-stop-wait",
        type=float,
        default=0.5,
        help="Wait after stop before reading counters/final transcription.",
    )
    parser.add_argument("--timer-pause", type=float, default=0.02)
    parser.add_argument("--syllable-pause", type=float, default=1.0)
    parser.add_argument("--sensitivity", type=float, default=0.6)
    parser.add_argument(
        "--followup-delays",
        type=parse_delays,
        default=(0.5,),
        help="Comma-separated syllable follow-up delays in seconds. Empty string disables.",
    )
    parser.add_argument("--model", type=str, default="large-v2", help="Main/final transcription model.")
    parser.add_argument("--rt-model", type=str, default="tiny.en", help="Realtime transcription model.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="default")
    parser.add_argument(
        "--max-wer",
        type=float,
        default=0.25,
        help="Maximum word error rate allowed for final transcription validation.",
    )
    parser.add_argument(
        "--no-realtime-sleep",
        action="store_true",
        help="Feed the file as fast as possible. Counts will not represent realtime behavior.",
    )
    return parser.parse_args()


def read_wav_samples(path: str) -> AudioData:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1).astype(np.int16)

    duration = float(samples.size) / float(sample_rate)
    return AudioData(samples=samples, sample_rate=sample_rate, duration=duration)


def normalize_transcript(text: str) -> str:
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


def transcription_metrics(expected: str, actual: str) -> Dict[str, object]:
    expected_words = normalize_transcript(expected).split()
    actual_words = normalize_transcript(actual).split()

    distance = edit_distance(expected_words, actual_words)
    wer = distance / max(1, len(expected_words))

    return {
        "expected_normalized": " ".join(expected_words),
        "actual_normalized": " ".join(actual_words),
        "expected_words": len(expected_words),
        "actual_words": len(actual_words),
        "edit_distance": distance,
        "wer": wer,
    }


def reset_realtime_counters(recorder):
    recorder.realtime_transcription_count = 0
    recorder.realtime_transcription_success_count = 0
    recorder.realtime_transcription_empty_count = 0
    recorder.realtime_transcription_trigger_counts = {}


def snapshot_realtime_counters(recorder, update_callback_count: int, elapsed: float):
    return {
        "elapsed": elapsed,
        "attempts": int(getattr(recorder, "realtime_transcription_count", 0)),
        "successes": int(getattr(recorder, "realtime_transcription_success_count", 0)),
        "empty": int(getattr(recorder, "realtime_transcription_empty_count", 0)),
        "updates": update_callback_count,
        "triggers": dict(getattr(recorder, "realtime_transcription_trigger_counts", {})),
    }


def make_recorder_config(args, use_syllable_boundaries: bool):
    return {
        "use_microphone": False,
        "spinner": False,
        "model": args.model,
        "realtime_model_type": args.rt_model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
        "enable_realtime_transcription": True,
        "realtime_processing_pause": (
            args.syllable_pause if use_syllable_boundaries else args.timer_pause
        ),
        "realtime_transcription_use_syllable_boundaries": use_syllable_boundaries,
        "realtime_boundary_detector_sensitivity": args.sensitivity,
        "realtime_boundary_followup_delays": args.followup_delays,
        "use_main_model_for_realtime": False,
        "min_length_of_recording": 0,
        "min_gap_between_recordings": 0,
        "no_log_file": True,
        "faster_whisper_vad_filter": False,
    }


def feed_samples(
        recorder,
        samples: np.ndarray,
        sample_rate: int,
        chunk_size: int,
        speed: float,
        sleep_enabled: bool):
    for start in range(0, samples.size, chunk_size):
        chunk = samples[start:start + chunk_size]
        recorder.feed_audio(chunk, original_sample_rate=sample_rate)

        if sleep_enabled:
            time.sleep((chunk.size / float(sample_rate)) / speed)


def feed_silence(
        recorder,
        seconds: float,
        sample_rate: int,
        chunk_size: int,
        speed: float,
        sleep_enabled: bool):
    if seconds <= 0:
        return

    total_samples = int(round(seconds * sample_rate))
    silence = np.zeros(total_samples, dtype=np.int16)
    feed_samples(recorder, silence, sample_rate, chunk_size, speed, sleep_enabled)


def run_mode(label: str, args, audio: AudioData, use_syllable_boundaries: bool):
    update_callback_count = 0
    realtime_updates = []

    def on_realtime_update(text):
        nonlocal update_callback_count
        update_callback_count += 1
        realtime_updates.append(text)

    config = make_recorder_config(args, use_syllable_boundaries)
    config["on_realtime_transcription_update"] = on_realtime_update

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)
    print("Initializing recorder. Main and realtime engine warmups happen during initialization.")

    recorder = AudioToTextRecorder(**config)

    try:
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)

        reset_realtime_counters(recorder)
        update_callback_count = 0
        realtime_updates.clear()

        chunk_size = max(1, int(round(audio.sample_rate * args.chunk_ms / 1000.0)))
        sleep_enabled = not args.no_realtime_sleep
        speed = max(0.001, float(args.speed))

        print(
            "Feeding file: {duration:.2f}s audio, chunk={chunk_ms:.1f}ms, "
            "speed={speed:.2f}x".format(
                duration=audio.duration,
                chunk_ms=args.chunk_ms,
                speed=speed if sleep_enabled else 0.0,
            )
        )

        recorder.start()
        started = time.time()

        feed_silence(
            recorder,
            args.lead_silence,
            audio.sample_rate,
            chunk_size,
            speed,
            sleep_enabled,
        )
        feed_samples(
            recorder,
            audio.samples,
            audio.sample_rate,
            chunk_size,
            speed,
            sleep_enabled,
        )
        feed_silence(
            recorder,
            args.tail_silence,
            audio.sample_rate,
            chunk_size,
            speed,
            sleep_enabled,
        )

        elapsed_feed = time.time() - started
        recorder.stop()

        if args.post_stop_wait > 0:
            time.sleep(args.post_stop_wait)

        counters = snapshot_realtime_counters(recorder, update_callback_count, elapsed_feed)
        final_text = recorder.text()
        metrics = transcription_metrics(args.expected_text, final_text)
        final_ok = metrics["wer"] <= args.max_wer

        return {
            "label": label,
            "use_syllable_boundaries": use_syllable_boundaries,
            "counters": counters,
            "final_text": final_text,
            "metrics": metrics,
            "final_ok": final_ok,
            "last_realtime_update": realtime_updates[-1] if realtime_updates else "",
        }
    finally:
        recorder.shutdown()


def print_mode_result(result):
    counters = result["counters"]
    metrics = result["metrics"]
    per_second = counters["attempts"] / counters["elapsed"] if counters["elapsed"] else 0.0

    print()
    print(f"{result['label']} result:")
    print(f"  realtime attempts:  {counters['attempts']} ({per_second:.2f}/s)")
    print(f"  realtime successes: {counters['successes']}")
    print(f"  realtime empty:     {counters['empty']}")
    print(f"  realtime updates:   {counters['updates']}")
    print(f"  trigger counts:     {counters['triggers']}")
    print(f"  final WER:          {metrics['wer']:.3f}")
    print(f"  final OK:           {result['final_ok']}")
    print(f"  final text:         {result['final_text']}")
    if result["last_realtime_update"]:
        print(f"  last realtime text: {result['last_realtime_update']}")


def print_comparison(results):
    print()
    print("=" * 80)
    print("Evaluation")
    print("=" * 80)

    for result in results:
        print_mode_result(result)

    by_mode = {result["use_syllable_boundaries"]: result for result in results}
    if False in by_mode and True in by_mode:
        timer_attempts = by_mode[False]["counters"]["attempts"]
        syllable_attempts = by_mode[True]["counters"]["attempts"]

        if timer_attempts:
            reduction = 100.0 * (1.0 - syllable_attempts / float(timer_attempts))
            print()
            print(
                "Realtime attempt delta: timer={timer}, syllable={syllable}, "
                "reduction={reduction:.1f}%".format(
                    timer=timer_attempts,
                    syllable=syllable_attempts,
                    reduction=reduction,
                )
            )

    all_ok = all(result["final_ok"] for result in results)
    print(f"Final transcription validation for all modes: {all_ok}")
    return all_ok


def main():
    args = parse_args()

    audio = read_wav_samples(args.audio_file)
    print(f"Loaded {args.audio_file}")
    print(f"  sample_rate={audio.sample_rate}Hz duration={audio.duration:.2f}s samples={audio.samples.size}")

    modes: List[Tuple[str, bool]] = []
    if args.mode in ("both", "timer"):
        modes.append(("timer mode", False))
    if args.mode in ("both", "syllable"):
        modes.append(("syllable-boundary mode", True))

    results = [
        run_mode(label, args, audio, use_syllable_boundaries)
        for label, use_syllable_boundaries in modes
    ]

    all_ok = print_comparison(results)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
