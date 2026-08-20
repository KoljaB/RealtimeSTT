#!/usr/bin/env python3
"""Run a reproducible A/B benchmark against PCM16 ASR HTTP endpoints.

The wire contract is the endpoint used by AgentTalk and the WWZ CPU reference
server: POST raw mono 16 kHz little-endian PCM16 to ``/transcribe-pcm16`` and
read a JSON object containing at least ``text``.  The tool deliberately keeps
the target URL configurable so a candidate RealtimeSTT deployment can be
started on a separate port and compared without touching the reference.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
from dataclasses import dataclass
import hashlib
import json
import math
import random
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import wave
from array import array
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Clip:
    clip_id: str
    language: str
    reference: str
    reference_kind: str
    category: str | None
    scenario: str | None
    wav_path: Path
    pcm16: bytes
    audio_duration_s: float


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(
        "".join(char if char.isalnum() else " " for char in value).split()
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


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _load_pcm16(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        source_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(f"{path} must be mono PCM16 WAV")

    source = array("h")
    source.frombytes(frames)
    if source.itemsize != 2:
        raise RuntimeError("The host Python does not expose 16-bit signed samples")
    if sys.byteorder != "little":
        source.byteswap()
    if source_rate == 16_000:
        converted = source
    else:
        output_count = max(1, round(len(source) * 16_000 / source_rate))
        converted = array("h")
        for output_index in range(output_count):
            source_position = output_index * source_rate / 16_000
            left = min(len(source) - 1, int(source_position))
            right = min(len(source) - 1, left + 1)
            fraction = source_position - left
            value = round(source[left] * (1.0 - fraction) + source[right] * fraction)
            converted.append(max(-32768, min(32767, value)))
    if sys.byteorder != "little":
        converted.byteswap()
    return converted.tobytes(), len(converted) / 16_000.0


def _resolve_wav(manifest: Path, value: str) -> Path:
    supplied = Path(value.replace("\\", "/"))
    candidates = (supplied, manifest.parent / supplied, manifest.parent / supplied.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {value!r} relative to {manifest}")


def _manifest_records(manifest: Path) -> list[dict[str, Any]]:
    if manifest.suffix.lower() == ".jsonl":
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
    return [item for item in payload if isinstance(item, dict)]


def load_corpus(manifest: Path, limit: int | None = None) -> list[Clip]:
    corpus: list[Clip] = []
    for index, item in enumerate(_manifest_records(manifest), start=1):
        wav_value = item.get("wav") or item.get("wav_path") or item.get("file")
        if not wav_value:
            continue
        wav_path = _resolve_wav(manifest, str(wav_value))
        pcm16, duration_s = _load_pcm16(wav_path)
        reference = str(
            item.get("reference_text")
            or item.get("text")
            or item.get("asr_text")
            or ""
        )
        clip = Clip(
            clip_id=str(
                item.get("turn_id")
                or item.get("clip_id")
                or item.get("generation")
                or item.get("id")
                or index
            ),
            language=str(
                item.get("language")
                or item.get("requested_language")
                or "en"
            ).lower(),
            reference=reference,
            reference_kind=str(
                item.get("reference_kind")
                or ("ground_truth" if reference else "unknown")
            ),
            category=item.get("category"),
            scenario=item.get("scenario"),
            wav_path=wav_path,
            pcm16=pcm16,
            audio_duration_s=duration_s,
        )
        corpus.append(clip)
        if limit is not None and len(corpus) >= limit:
            break
    if not corpus:
        raise ValueError(f"No WAV records found in {manifest}")
    return corpus


@dataclass(frozen=True)
class Target:
    label: str
    base_url: str


def parse_target(value: str) -> Target:
    label, separator, base_url = value.partition("=")
    if not separator or not label.strip() or not base_url.strip():
        raise argparse.ArgumentTypeError("target must use LABEL=URL")
    return Target(label.strip(), base_url.strip().rstrip("/"))


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return value


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return f"HTTP {exc.code}: {detail}"
    return f"{type(exc).__name__}: {exc}"


def transcribe_http(target: Target, clip: Clip, timeout_s: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "sample_rate": "16000",
            "encoding": "pcm16",
            "language": clip.language,
            "beam_size": "3",
            "best_of": "1",
            "temperature": "0",
            "word_timestamps": "false",
            "vad_filter": "false",
            "condition_on_previous_text": "false",
            "without_timestamps": "true",
        }
    )
    request = urllib.request.Request(
        f"{target.base_url}/transcribe-pcm16?{query}",
        data=clip.pcm16,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("ASR endpoint returned a non-object response")
        client_seconds = time.perf_counter() - started
        return {
            "clip_id": clip.clip_id,
            "audio_duration_s": clip.audio_duration_s,
            "client_seconds": client_seconds,
            "server_elapsed_seconds": payload.get("elapsed_seconds"),
            "server_queue_seconds": payload.get("queue_seconds"),
            "server_decode_seconds": payload.get("decode_seconds"),
            "rtf": payload.get("rtf"),
            "text": str(payload.get("text") or "").strip(),
            "error": None,
        }
    except Exception as exc:
        return {
            "clip_id": clip.clip_id,
            "audio_duration_s": clip.audio_duration_s,
            "client_seconds": None,
            "server_elapsed_seconds": None,
            "server_queue_seconds": None,
            "server_decode_seconds": None,
            "rtf": None,
            "text": "",
            "error": _error_text(exc),
        }


def _summary(records: Iterable[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [
        float(item[field])
        for item in records
        if item.get(field) is not None
    ]
    return {
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [item for item in records if item["reference_normalized"]]
    word_errors = 0
    reference_words = 0
    char_errors = 0
    reference_chars = 0
    utterance_wers: list[float] = []
    exact = 0
    for item in selected:
        reference_words_value = item["reference_normalized"].split()
        hypothesis_words_value = item["hypothesis_normalized"].split()
        errors = edit_distance(reference_words_value, hypothesis_words_value)
        word_errors += errors
        reference_words += len(reference_words_value)
        utterance_wers.append(errors / max(1, len(reference_words_value)))
        reference_chars_value = list(item["reference_normalized"].replace(" ", ""))
        hypothesis_chars_value = list(item["hypothesis_normalized"].replace(" ", ""))
        char_errors += edit_distance(reference_chars_value, hypothesis_chars_value)
        reference_chars += len(reference_chars_value)
        exact += item["reference_normalized"] == item["hypothesis_normalized"]
    return {
        "count": len(selected),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "wer": word_errors / max(1, reference_words),
        "macro_utterance_wer": statistics.fmean(utterance_wers)
        if utterance_wers
        else None,
        "cer": char_errors / max(1, reference_chars),
        "exact_match_rate": exact / max(1, len(selected)),
        "empty_hypotheses": sum(not item["hypothesis_normalized"] for item in selected),
    }


def run_target(
    target: Target,
    corpus: list[Clip],
    *,
    repetitions: int,
    seed: int,
    timeout_s: float,
    concurrency: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    health_before: dict[str, Any] | None = None
    health_error: str | None = None
    try:
        health_before = _get_json(f"{target.base_url}/health", timeout_s)
    except Exception as exc:
        health_error = _error_text(exc)

    jobs: list[tuple[int, Clip]] = []
    rng = random.Random(seed)
    for repetition in range(max(1, repetitions)):
        order = list(corpus)
        rng.shuffle(order)
        jobs.extend((repetition, clip) for clip in order)

    def one(job: tuple[int, Clip]) -> dict[str, Any]:
        repetition, clip = job
        result = transcribe_http(target, clip, timeout_s)
        result["repetition"] = repetition
        result["target"] = target.label
        return result

    if concurrency == 1:
        requests = [one(job) for job in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, job) for job in jobs]
            requests = [future.result() for future in futures]

    successful = [item for item in requests if item["error"] is None]
    hypotheses: dict[str, list[str]] = {clip.clip_id: [] for clip in corpus}
    for item in requests:
        hypotheses[item["clip_id"]].append(item["text"])
    accuracy_records = []
    for clip in corpus:
        variants = hypotheses[clip.clip_id]
        counts: dict[str, int] = {}
        for variant in variants:
            counts[variant] = counts.get(variant, 0) + 1
        hypothesis = min(counts, key=lambda value: (-counts[value], value)) if counts else ""
        accuracy_records.append(
            {
                "clip_id": clip.clip_id,
                "language": clip.language,
                "reference_kind": clip.reference_kind,
                "category": clip.category,
                "scenario": clip.scenario,
                "audio_duration_s": clip.audio_duration_s,
                "reference": clip.reference,
                "hypothesis": hypothesis,
                "reference_normalized": normalize_text(clip.reference),
                "hypothesis_normalized": normalize_text(hypothesis),
                "deterministic_across_repetitions": len(set(variants)) <= 1,
                "hypothesis_variants": dict(sorted(counts.items())),
            }
        )

    total_wall_seconds = time.perf_counter() - started
    health_after = None
    with_error = None
    try:
        health_after = _get_json(f"{target.base_url}/health", timeout_s)
    except Exception as exc:
        with_error = _error_text(exc)
    return {
        "label": target.label,
        "base_url": target.base_url,
        "health_before": health_before,
        "health_before_error": health_error,
        "health_after": health_after,
        "health_after_error": with_error,
        "repetitions": max(1, repetitions),
        "seed": seed,
        "concurrency": {
            "workers": concurrency,
            "total_wall_seconds": total_wall_seconds,
            "successful_requests": len(successful),
            "failed_requests": len(requests) - len(successful),
            "throughput_requests_per_second": len(successful) / max(total_wall_seconds, 1e-9),
        },
        "corpus": {
            "clips": len(corpus),
            "audio_seconds": sum(clip.audio_duration_s for clip in corpus),
        },
        "accuracy": accuracy(accuracy_records),
        "latency": {
            "final_client_seconds": _summary(successful, "client_seconds"),
            "server_elapsed_seconds": _summary(successful, "server_elapsed_seconds"),
            "server_queue_seconds": _summary(successful, "server_queue_seconds"),
            "server_decode_seconds": _summary(successful, "server_decode_seconds"),
            "rtf": _summary(successful, "rtf"),
        },
        "partial_latency": {
            "available": False,
            "reason": "The PCM16 HTTP contract returns one final JSON response; it has no partial-event channel.",
        },
        "final_latency_definition": "client_seconds from POST start until the final JSON response is decoded",
        "accuracy_records": accuracy_records,
        "requests": requests,
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"## {report['label']}",
        "",
        f"- URL: `{report['base_url']}`",
        f"- Corpus: {report['corpus']['clips']} clips / {report['corpus']['audio_seconds']:.3f} s",
        f"- Repetitions: {report['repetitions']}; workers: {report['concurrency']['workers']}",
        f"- Successful requests: {report['concurrency']['successful_requests']}; failed: {report['concurrency']['failed_requests']}",
        f"- Throughput: {report['concurrency']['throughput_requests_per_second']:.3f} requests/s",
        "",
        "| Quality scope | Clips | WER | CER | Exact |",
        "|---|---:|---:|---:|---:|",
        f"| all | {report['accuracy']['count']} | {_format_metric(report['accuracy']['wer'])} | {_format_metric(report['accuracy']['cer'])} | {_format_metric(report['accuracy']['exact_match_rate'])} |",
        "",
        "| Latency | Median | p90 | p95 | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, field in (
        ("Final client s", "final_client_seconds"),
        ("Server elapsed s", "server_elapsed_seconds"),
        ("Server queue s", "server_queue_seconds"),
        ("Server decode s", "server_decode_seconds"),
        ("RTF", "rtf"),
    ):
        values = report["latency"][field]
        lines.append(
            f"| {label} | {_format_metric(values['median'])} | {_format_metric(values['p90'])} | {_format_metric(values['p95'])} | {_format_metric(values['max'])} |"
        )
    lines.extend(
        [
            "",
            "Partial latency: unavailable for this final-only HTTP contract.",
            "",
        ]
    )
    return "\n".join(lines)


def redact_report(result: dict[str, Any]) -> dict[str, Any]:
    """Remove private corpus paths and reconstructable ASR content."""

    safe = copy.deepcopy(result)
    safe["manifest"] = "<redacted-manifest>"
    for report in safe.get("targets", []):
        report["base_url"] = "<redacted-endpoint>"
        for key in (
            "health_before",
            "health_before_error",
            "health_after",
            "health_after_error",
            "requests",
        ):
            report.pop(key, None)
        for record in report.get("accuracy_records", []):
            clip_id = str(record.get("clip_id", ""))
            record["clip_id"] = "clip-" + hashlib.sha256(
                clip_id.encode("utf-8", errors="replace")
            ).hexdigest()[:12]
            for key in (
                "reference",
                "hypothesis",
                "reference_normalized",
                "hypothesis_normalized",
                "hypothesis_variants",
            ):
                record.pop(key, None)
    safe["sensitive_details_included"] = False
    return safe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
        metavar="LABEL=URL",
        help="Repeat for each A/B target, for example reference=http://host:8766.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-sensitive-details",
        action="store_true",
        help=(
            "Include manifest paths, endpoints, transcripts, and per-request "
            "details in the protected local JSON; default output redacts them"
        ),
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    corpus = load_corpus(args.manifest, args.limit)
    reports = [
        run_target(
            target,
            corpus,
            repetitions=args.repetitions,
            seed=args.seed,
            timeout_s=args.timeout_s,
            concurrency=args.concurrency,
        )
        for target in args.target
    ]
    result = {
        "kind": "asr_http_ab_benchmark",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(args.manifest.resolve()),
        "targets": reports,
        "contract": {
            "request": "POST /transcribe-pcm16 with query sample_rate=16000, encoding=pcm16 and raw mono little-endian PCM16 body",
            "response": "JSON object with text; optional elapsed_seconds, queue_seconds, decode_seconds, and rtf",
            "partial_events": False,
        },
    }
    output_result = result if args.include_sensitive_details else redact_report(result)
    if args.include_sensitive_details:
        output_result["sensitive_details_included"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = "# ASR HTTP A/B benchmark\n\n" + "\n".join(
        markdown_report(report) for report in output_result["targets"]
    )
    args.output.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
