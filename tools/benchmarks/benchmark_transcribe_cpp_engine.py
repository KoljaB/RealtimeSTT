"""Benchmark the resident RealtimeSTT transcribe.cpp engine on one exact WAV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path


DEFAULT_MODEL_SHA256 = (
    "b68557be1e3c40207fd7c4bd9d63f1d3316b963f15325bfb0cc16a8bb0ffd181"
)
DEFAULT_PCM_SHA256 = (
    "c4056da582d0e6ede0ea02c7333e8b6dc45cdb9e3178bc8a01172d619ea2bbf2"
)
DEFAULT_NATIVE_COMMIT = "ea077b8"
DEFAULT_DEVICE_DESCRIPTION = "NVIDIA GeForce RTX 4090"
DEFAULT_CPU_AFFINITY = "0,2,6,8,10,12,14"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--flash", choices=("off", "on"), default="off")
    parser.add_argument("--warmups", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--expected-text")
    parser.add_argument("--expected-pcm-sha256", default=DEFAULT_PCM_SHA256)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--expected-native-commit", default=DEFAULT_NATIVE_COMMIT)
    parser.add_argument("--expected-device", default=DEFAULT_DEVICE_DESCRIPTION)
    parser.add_argument("--expected-affinity", default=DEFAULT_CPU_AFFINITY)
    parser.add_argument("--max-mean-ms", type=float, default=40.0)
    parser.add_argument("--max-p95-ms", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", text.lower())).strip()


def percentile_95(values):
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def parse_cpu_affinity(value):
    try:
        cpus = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("affinity must be comma-separated integers") from exc
    if not cpus:
        raise argparse.ArgumentTypeError("affinity must contain at least one CPU")
    return cpus


def main():
    args = parse_args()
    if args.warmups < 5:
        raise SystemExit("--warmups must be at least 5")
    if args.iterations < 25:
        raise SystemExit("--iterations must be at least 25")

    configured_library = os.environ.get("TRANSCRIBE_LIBRARY")
    if not configured_library:
        raise SystemExit("TRANSCRIBE_LIBRARY must select the tuned shared library")
    expected_library = str(Path(configured_library).expanduser().resolve())
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("the tuned benchmark requires Linux CPU affinity support")
    expected_affinity = parse_cpu_affinity(args.expected_affinity)
    actual_affinity = sorted(os.sched_getaffinity(0))

    # These must exist before the optional native binding is imported.
    if args.flash == "off":
        os.environ["TRANSCRIBE_NO_FLASH"] = "1"
    else:
        os.environ.pop("TRANSCRIBE_NO_FLASH", None)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np
    import soundfile as sf
    import transcribe_cpp
    transcribe_cpp.set_log_callback(None)

    from RealtimeSTT.transcription_engines import (
        TranscriptionEngineConfig,
        create_transcription_engine,
    )

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000:
        raise SystemExit(f"expected 16000 Hz audio, got {sample_rate}")
    if audio.ndim != 1:
        raise SystemExit(f"expected mono 1-D audio, got shape {audio.shape}")
    if audio.size != 80000:
        raise SystemExit(
            f"expected an exact 5.000 s / 80000-sample fixture, got {audio.size}"
        )
    if not audio.flags.c_contiguous:
        audio = np.ascontiguousarray(audio)

    pcm_sha256 = hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest()
    if args.expected_pcm_sha256 and pcm_sha256 != args.expected_pcm_sha256.lower():
        raise SystemExit(
            "decoded float32 PCM SHA-256 mismatch: "
            f"expected {args.expected_pcm_sha256.lower()}, got {pcm_sha256}"
        )

    engine = create_transcription_engine(
        "transcribe_cpp",
        TranscriptionEngineConfig(
            model=str(args.model),
            device="cuda",
            gpu_device_index=args.device_index,
            beam_size=1,
            batch_size=1,
            vad_filter=False,
            engine_options={
                "backend": "cuda",
                "model_sha256": args.model_sha256,
                "session": {
                    "n_threads": args.threads,
                    "kv_type": "auto",
                    "n_ctx": 0,
                },
                "transcribe": {
                    "timestamps": "none",
                },
            },
        ),
    )

    try:
        for _ in range(args.warmups):
            engine.transcribe(audio, language=args.language or None)

        samples_ms = []
        transcripts = []
        native_timings = []
        outer_start = time.perf_counter_ns()
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            result = engine.transcribe(audio, language=args.language or None)
            samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
            transcripts.append(result.text)
            native_timings.append(result.metadata.get("timings_ms", {}))
        outer_ms = (time.perf_counter_ns() - outer_start) / 1_000_000.0

        mean_ms = statistics.fmean(samples_ms)
        median_ms = statistics.median(samples_ms)
        p95_ms = percentile_95(samples_ms)
        stdev_ms = statistics.stdev(samples_ms)
        runtime = dict(result.metadata)
        backend = str(runtime.get("backend") or "")
        device = runtime.get("device") or {}
        native_library = runtime.get("native_library")
        resolved_native_library = (
            str(Path(native_library).expanduser().resolve())
            if native_library
            else None
        )
        runtime["process_cpu_affinity"] = actual_affinity

        expected_normalized = (
            normalize_text(args.expected_text) if args.expected_text else None
        )
        transcript_normalized = normalize_text(transcripts[-1])
        acceptance = {
            "sample_count_pass": len(samples_ms) == args.iterations,
            "cuda_backend_pass": (
                backend.lower().startswith("cuda")
                and str(device.get("kind") or "").lower() == "cuda"
            ),
            "device_description_pass": (
                str(device.get("description") or "") == args.expected_device
            ),
            "native_commit_pass": (
                str(runtime.get("native_commit") or "") == args.expected_native_commit
            ),
            "native_library_pass": resolved_native_library == expected_library,
            "cpu_affinity_pass": actual_affinity == expected_affinity,
            "stable_transcript_pass": len(set(transcripts)) == 1,
            "nonempty_transcript_pass": bool(transcript_normalized),
            "expected_text_pass": (
                expected_normalized is None
                or transcript_normalized == expected_normalized
            ),
            "outer_wall_pass": outer_ms >= sum(samples_ms),
            "mean_pass": mean_ms < args.max_mean_ms,
            "p95_pass": p95_ms < args.max_p95_ms,
        }
        acceptance["passed"] = all(acceptance.values())

        report = {
            "contract": {
                "audio": str(args.audio.resolve()),
                "sample_rate": sample_rate,
                "samples": int(audio.size),
                "duration_seconds": audio.size / sample_rate,
                "dtype": str(audio.dtype),
                "pcm_float32_sha256": pcm_sha256,
                "model": str(args.model.resolve()),
                "model_sha256": args.model_sha256,
                "expected_native_commit": args.expected_native_commit,
                "expected_native_library": expected_library,
                "expected_device_description": args.expected_device,
                "expected_cpu_affinity": expected_affinity,
                "max_mean_ms": args.max_mean_ms,
                "max_p95_ms": args.max_p95_ms,
                "warmups_excluded": args.warmups,
                "iterations": args.iterations,
                "timestamps": "none",
                "flash_attention": args.flash,
                "threads": args.threads,
                "language": args.language or None,
            },
            "latency": {
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "p95_ms": p95_ms,
                "stdev_ms": stdev_ms,
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
                "samples_ms": samples_ms,
                "outer_ms": outer_ms,
            },
            "runtime": runtime,
            "native_timings_ms": native_timings,
            "transcript": transcripts[-1],
            "acceptance": acceptance,
        }

        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        if not acceptance["passed"]:
            raise SystemExit(1)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
