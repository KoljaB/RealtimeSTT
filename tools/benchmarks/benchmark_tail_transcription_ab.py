"""
Measure tail finalization with the real Nemotron streaming Live ASR path.

The Live transcript is produced by the repository's
``sherpa_onnx_nemotron`` streaming session. Audio is fed in small incremental
chunks and the current result is captured after each decode; the session is
not finalized before the Live text is recorded.

The before path runs the complete active-speech audio through the Parakeet
Final ASR engine. The after path runs only the configured tail retained from
active speech through the same Final ASR engine and merges it into the actual
Nemotron Live transcript. An alignment failure measures the production
full-utterance fallback.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Running this file directly puts ``tools/benchmarks`` on sys.path rather than
# the repository root, so make the local package import explicit.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    import soundfile as sf
    from scipy.signal import resample_poly
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The real streaming benchmark requires soundfile and scipy for WAV "
        "loading/resampling."
    ) from exc

from RealtimeSTT.core.tail_transcription import (
    FINAL_TRANSCRIPTION_TAIL_SECONDS,
    merge_live_and_tail_transcription,
)
from RealtimeSTT.transcription_engines import (
    NemotronEngine,
    TranscriptionEngineConfig,
)
from RealtimeSTT.transcription_engines.sherpa_onnx_engine import (
    SherpaOnnxParakeetEngine,
)


SAMPLE_RATE = 16000
DEFAULT_LIVE_MODEL = (
    "test-model-cache/sherpa-onnx/"
    "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
)
DEFAULT_FINAL_MODEL = (
    "test-model-cache/sherpa-onnx/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
)


def _load_audio(path: Path) -> np.ndarray:
    """Load a mono WAV and resample it to Nemotron's required 16 kHz."""

    audio, sample_rate = sf.read(str(path), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = audio.reshape(-1)
    if int(sample_rate) != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, int(sample_rate))
    return np.ascontiguousarray(audio, dtype=np.float32)


def _trim_trailing_silence(audio: np.ndarray) -> np.ndarray:
    """
    Approximate the active-speech buffer's end for unannotated WAV fixtures.

    The production rolling buffer receives active speech frames, not the
    recording's trailing file silence. This frame-energy trim keeps that
    benchmark input equivalent without using an ASR model to find the end.
    """

    if audio.size == 0:
        return audio

    frame_samples = int(round(SAMPLE_RATE * 0.02))
    rms_values = []
    for start in range(0, audio.size, frame_samples):
        frame = audio[start : start + frame_samples]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame), dtype=np.float64))))

    maximum_rms = max(rms_values, default=0.0)
    threshold = max(0.0015, maximum_rms * 0.03)
    active_frames = np.flatnonzero(np.asarray(rms_values) > threshold)
    if active_frames.size == 0:
        return audio

    end_sample = min(audio.size, int((int(active_frames[-1]) + 1) * frame_samples))
    return np.ascontiguousarray(audio[:end_sample], dtype=np.float32)


def _transcribe(final_engine, audio: np.ndarray) -> tuple[str, float]:
    """Run one Final ASR request and return text plus wall-clock latency."""

    started = time.perf_counter()
    result = final_engine.transcribe(audio, language="en")
    elapsed = time.perf_counter() - started
    return str(result.text or "").strip(), elapsed


def _stream_live(
    live_engine,
    audio: np.ndarray,
    chunk_samples: int,
    silence_samples: int,
) -> tuple[str, str, float, int, int]:
    """
    Feed one utterance through the actual Nemotron streaming session.

    ``live_at_vad`` is captured immediately after the final active-speech
    chunk, before any silence or final drain. That is the hypothesis used by
    the tail merge. Additional silence is fed only to expose how many words
    the streaming model emits after the boundary; ``finish()`` is never used.
    """

    session = live_engine.create_streaming_session(language="en")
    latest_text = ""
    accepted_chunks = 0
    nonempty_updates = 0
    started = time.perf_counter()

    def observe() -> None:
        nonlocal latest_text, nonempty_updates
        session.decode()
        current_text = str(session.get_result().text or "").strip()
        if current_text:
            latest_text = current_text
            nonempty_updates += 1

    try:
        for start in range(0, audio.size, chunk_samples):
            session.accept_audio(
                audio[start : start + chunk_samples],
                sample_rate=SAMPLE_RATE,
            )
            accepted_chunks += 1
            observe()

        live_at_vad = latest_text
        live_after_silence = latest_text
        if silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.float32)
            for start in range(0, silence.size, chunk_samples):
                session.accept_audio(
                    silence[start : start + chunk_samples],
                    sample_rate=SAMPLE_RATE,
                )
                accepted_chunks += 1
                observe()
            live_after_silence = latest_text
    finally:
        session.close()

    return (
        live_at_vad,
        live_after_silence,
        time.perf_counter() - started,
        accepted_chunks,
        nonempty_updates,
    )


def _percent(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _word_count(text: str) -> int:
    """Count whitespace-delimited words for boundary diagnostics."""

    return len(str(text or "").split())


def _latency_summary(records: list[dict], key: str) -> dict:
    values = [record[key] for record in records if not record.get("error")]
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
    }


def _summary(records: list[dict]) -> dict:
    completed = [record for record in records if not record.get("error")]
    eligible = [record for record in completed if record["tail_eligible"]]
    exact_matches = [record for record in eligible if record["alignment"] == "exact"]
    fuzzy_matches = [record for record in eligible if record["alignment"] == "fuzzy"]
    alignment_failures = [
        record for record in eligible if record["alignment"] == "failure"
    ]
    tail_only = [record for record in eligible if not record["fallback"]]
    tail_only_exact = [record for record in tail_only if record["exact_parity"]]
    exact_parity = [record for record in completed if record["exact_parity"]]
    eligible_before = [record["before_latency_seconds"] for record in eligible]
    eligible_after = [record["after_latency_seconds"] for record in eligible]
    boundary_changed = [
        record
        for record in eligible
        if record["live_text_at_vad"] != record["live_text_after_silence"]
    ]
    boundary_word_recovery = [
        record
        for record in eligible
        if record["live_words_after_silence"] > record["live_words_at_vad"]
    ]

    return {
        "recordings": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "tail_eligible": len(eligible),
        "exact_anchor_matches": len(exact_matches),
        "fuzzy_anchor_matches": len(fuzzy_matches),
        "alignment_failures": len(alignment_failures),
        "alignment_success_rate_percent": _percent(
            len(exact_matches) + len(fuzzy_matches), len(eligible)
        ),
        "alignment_failure_rate_percent": _percent(
            len(alignment_failures), len(eligible)
        ),
        "exact_final_text_matches": len(exact_parity),
        "exact_final_text_rate_percent": _percent(
            len(exact_parity), len(completed)
        ),
        "tail_only_exact_final_text_matches": len(tail_only_exact),
        "tail_only_exact_final_text_rate_percent": _percent(
            len(tail_only_exact), len(tail_only)
        ),
        "tail_only_cases": len(tail_only),
        "fallbacks": sum(record["fallback"] for record in completed),
        "live_hypothesis_changed_after_vad_count": len(boundary_changed),
        "live_boundary_word_recovery_count": len(boundary_word_recovery),
        "live_boundary_added_words": sum(
            max(
                0,
                record["live_words_after_silence"]
                - record["live_words_at_vad"],
            )
            for record in eligible
        ),
        "before_latency_seconds": _latency_summary(
            completed, "before_latency_seconds"
        ),
        "after_latency_seconds": _latency_summary(
            completed, "after_latency_seconds"
        ),
        "eligible_before_latency_seconds": {
            "mean": statistics.mean(eligible_before) if eligible_before else 0.0,
            "median": statistics.median(eligible_before) if eligible_before else 0.0,
        },
        "eligible_after_latency_seconds": {
            "mean": statistics.mean(eligible_after) if eligible_after else 0.0,
            "median": statistics.median(eligible_after) if eligible_after else 0.0,
        },
        "eligible_latency_reduction_percent": _percent(
            sum(eligible_before) - sum(eligible_after), sum(eligible_before)
        ),
        "active_speech_seconds": sum(
            record["active_speech_seconds"] for record in completed
        ),
        "after_input_seconds": sum(
            record["after_input_seconds"] for record in completed
        ),
        "live_processing_seconds": sum(
            record["live_processing_seconds"] for record in eligible
        ),
    }


def run_benchmark(
    live_engine,
    final_engine,
    audio_paths: list[Path],
    chunk_ms: float,
    live_silence_ms: float,
    final_engine_name: str = "sherpa_onnx_parakeet",
) -> dict:
    records = []
    tail_samples = int(round(SAMPLE_RATE * FINAL_TRANSCRIPTION_TAIL_SECONDS))
    chunk_samples = max(1, int(round(SAMPLE_RATE * chunk_ms / 1000.0)))
    silence_samples = max(0, int(round(SAMPLE_RATE * live_silence_ms / 1000.0)))

    for audio_path in audio_paths:
        record = {
            "audio": str(audio_path),
            "tail_eligible": False,
            "alignment": "not_eligible",
            "fallback": False,
        }
        try:
            raw_audio = _load_audio(audio_path)
            active_audio = _trim_trailing_silence(raw_audio)
            active_seconds = float(active_audio.size) / SAMPLE_RATE
            record["raw_audio_seconds"] = float(raw_audio.size) / SAMPLE_RATE
            record["active_speech_seconds"] = active_seconds

            before_text, before_latency = _transcribe(final_engine, active_audio)
            record["before_text"] = before_text
            record["before_latency_seconds"] = before_latency
            record["before_input_seconds"] = active_seconds

            if active_audio.size <= tail_samples:
                after_text, after_latency = _transcribe(final_engine, active_audio)
                record.update(
                    {
                        "after_text": after_text,
                        "after_latency_seconds": after_latency,
                        "after_input_seconds": active_seconds,
                        "exact_parity": after_text == before_text,
                    }
                )
                records.append(record)
                continue

            record["tail_eligible"] = True
            (
                live_text,
                live_after_silence,
                live_seconds,
                accepted_chunks,
                nonempty_updates,
            ) = _stream_live(
                live_engine,
                active_audio,
                chunk_samples,
                silence_samples,
            )
            tail_audio = np.array(active_audio[-tail_samples:], copy=True)
            tail_text, tail_latency = _transcribe(final_engine, tail_audio)
            merge_result = merge_live_and_tail_transcription(live_text, tail_text)

            if merge_result.matched:
                after_text = merge_result.text
                after_latency = tail_latency
                record["alignment"] = (
                    "fuzzy" if merge_result.used_fuzzy_match else "exact"
                )
            else:
                fallback_text, fallback_latency = _transcribe(final_engine, active_audio)
                after_text = fallback_text
                after_latency = tail_latency + fallback_latency
                record["alignment"] = "failure"
                record["fallback"] = True

            record.update(
                {
                    "live_text": live_text,
                    "live_text_at_vad": live_text,
                    "live_text_after_silence": live_after_silence,
                    "live_words_at_vad": _word_count(live_text),
                    "live_words_after_silence": _word_count(live_after_silence),
                    "tail_text": tail_text,
                    "after_text": after_text,
                    "after_latency_seconds": after_latency,
                    "after_input_seconds": FINAL_TRANSCRIPTION_TAIL_SECONDS
                    + (active_seconds if record["fallback"] else 0.0),
                    "live_processing_seconds": live_seconds,
                    "live_accepted_chunks": accepted_chunks,
                    "live_nonempty_updates": nonempty_updates,
                    "tail_latency_seconds": tail_latency,
                    "exact_parity": after_text == before_text,
                    "anchor_length": merge_result.anchor_length,
                    "anchor_distance": merge_result.distance,
                }
            )
            records.append(record)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)

    return {
        "benchmark": "tail_transcription_before_after_real_streaming",
        "live_engine": "sherpa_onnx_nemotron",
        "final_engine": final_engine_name,
        "sample_rate": SAMPLE_RATE,
        "tail_seconds": FINAL_TRANSCRIPTION_TAIL_SECONDS,
        "chunk_ms": chunk_ms,
        "live_silence_ms": live_silence_ms,
        "records": records,
        "summary": _summary(records),
    }


def _print_report(report: dict) -> None:
    summary = report["summary"]
    print("Tail transcription before/after benchmark")
    print("Live: Nemotron sherpa-onnx streaming session")
    print(f"Final: {report.get('final_engine', 'offline Final ASR engine')}")
    print(f"recordings: {summary['completed']}/{summary['recordings']}")
    print(
        "alignment: "
        f"{summary['exact_anchor_matches']} exact, "
        f"{summary['fuzzy_anchor_matches']} fuzzy, "
        f"{summary['alignment_failures']} failures "
        f"({summary['alignment_success_rate_percent']:.1f}% success)"
    )
    print(
        "final text exact parity (including fallback): "
        f"{summary['exact_final_text_matches']}/{summary['completed']} "
        f"({summary['exact_final_text_rate_percent']:.1f}%)"
    )
    print(
        "tail-only final text exact parity: "
        f"{summary['tail_only_exact_final_text_matches']}/"
        f"{summary['tail_only_cases']} "
        f"({summary['tail_only_exact_final_text_rate_percent']:.1f}%)"
    )
    print(f"fallbacks: {summary['fallbacks']}")
    print(
        "Live boundary recovery: "
        f"{summary['live_boundary_word_recovery_count']} utterances gained "
        f"{summary['live_boundary_added_words']} word(s) after VAD silence"
    )
    print(
        "eligible finalization latency: "
        f"{summary['eligible_before_latency_seconds']['mean']:.3f}s -> "
        f"{summary['eligible_after_latency_seconds']['mean']:.3f}s "
        f"({summary['eligible_latency_reduction_percent']:.1f}% reduction)"
    )
    print(
        "Final ASR input audio: "
        f"{summary['active_speech_seconds']:.2f}s active speech -> "
        f"{summary['after_input_seconds']:.2f}s"
    )
    print("")
    for record in report["records"]:
        name = Path(record["audio"]).name
        if record.get("error"):
            print(f"{name}: ERROR {record['error']}")
            continue
        print(
            f"{name}: {record['alignment']}, "
            f"exact={'yes' if record['exact_parity'] else 'no'}, "
            f"{record['before_latency_seconds']:.3f}s -> "
            f"{record['after_latency_seconds']:.3f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-model", default=DEFAULT_LIVE_MODEL)
    parser.add_argument("--final-model", default=DEFAULT_FINAL_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=20.0,
        help="Incremental Live ASR input chunk size; default 20 ms",
    )
    parser.add_argument(
        "--live-silence-ms",
        type=float,
        default=560.0,
        help="Silence fed to Live ASR before capturing its current result",
    )
    parser.add_argument(
        "--corpus-dir",
        default="tests/unit/audio/voice_corpus/manual",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    live_model = Path(args.live_model)
    final_model = Path(args.final_model)
    if not live_model.is_dir():
        parser.error(f"Live Nemotron model directory does not exist: {live_model}")
    if not final_model.is_dir():
        parser.error(f"Final Parakeet model directory does not exist: {final_model}")

    engine_options = {
        "provider": "cpu" if args.device == "cpu" else args.device,
        "num_threads": args.threads,
    }
    live_engine = NemotronEngine(
        TranscriptionEngineConfig(
            model=str(live_model),
            device=args.device,
            engine_options=dict(engine_options),
        )
    )
    final_engine = SherpaOnnxParakeetEngine(
        TranscriptionEngineConfig(
            model=str(final_model),
            device=args.device,
            engine_options=dict(engine_options),
        )
    )

    audio_paths = sorted(Path(args.corpus_dir).glob("*.wav"))
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]
    if not audio_paths:
        parser.error(f"No WAV files found in {args.corpus_dir}")

    report = run_benchmark(
        live_engine,
        final_engine,
        audio_paths,
        args.chunk_ms,
        args.live_silence_ms,
    )
    report["live_model"] = str(live_model)
    report["final_model"] = str(final_model)
    report["corpus_dir"] = str(Path(args.corpus_dir))
    _print_report(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
