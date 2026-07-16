import json
import os
import re
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from example_fastapi_server.protocol import decode_audio_packet, encode_audio_packet
from example_fastapi_server.server import (
    ConnectionManager,
    RealtimeSTTService,
    ServerSettings,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT_DIR / "tests" / "unit" / "audio"
REFERENCE_WAV = AUDIO_DIR / "asr-reference.wav"
EXPECTED_JSON = AUDIO_DIR / "asr-reference.expected_sentences.json"


class CollectingManager(ConnectionManager):
    def __init__(self):
        super().__init__()
        self.messages = {}
        self._sync_lock = threading.Lock()

    def publish_session(self, session_id, message):
        message = dict(message)
        message["_receivedAt"] = time.monotonic()
        message["_receivedWallTime"] = time.time()
        with self._sync_lock:
            self.messages.setdefault(session_id, []).append(message)

    def publish_all(self, message):
        pass


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def env_json(name, default=None):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        normalized = value.replace('\\"', '"').replace('""', '"')
        if normalized != value:
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                pass
        parsed = {}
        for item in value.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise
            key, raw = item.split("=", 1)
            parsed[key.strip()] = parse_option_value(raw.strip())
        return parsed


def parse_option_value(value):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def env_float_tuple(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def run_real_asr_test_enabled():
    return (
        env_flag("REALTIMESTT_RUN_FASTAPI_MULTI_USER_ASR")
        or env_flag("REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF")
    )


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left, right):
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


def word_error_rate(expected, actual):
    expected_words = normalize_text(expected).split()
    actual_words = normalize_text(actual).split()
    return edit_distance(expected_words, actual_words) / max(1, len(expected_words))


def percentile(values, fraction):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
    return values[index]


def ms_between(start, end):
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def first_at_or_after(values, start):
    for value in sorted(values):
        if start is None or value >= start:
            return value
    return None


def round_or_none(value, digits=2):
    if value is None:
        return None
    return round(float(value), digits)


def event_times(messages, event_type):
    return [
        message["_receivedAt"]
        for message in messages
        if message.get("type") == event_type and "_receivedAt" in message
    ]


def status_times(messages, state):
    return [
        message["_receivedAt"]
        for message in messages
        if (
            message.get("type") == "status"
            and message.get("state") == state
            and "_receivedAt" in message
        )
    ]


def build_performance_report(
    *,
    settings,
    messages_by_session,
    session_ids,
    stream_started_at,
    audio_finished_at,
    stop_finished_at,
    service_started_at,
    ready_at,
    report_finished_at,
    audio_seconds,
    chunk_ms,
    speed,
    wers,
    metrics,
):
    session_reports = {}
    first_realtime_latencies = []
    first_final_latencies = []
    final_after_audio_end_latencies = []
    final_scheduler_p95 = []

    for session_id in session_ids:
        messages = messages_by_session.get(session_id, [])
        realtime_messages = [message for message in messages if message.get("type") == "realtime"]
        final_messages = [message for message in messages if message.get("type") == "final"]
        realtime_received = event_times(messages, "realtime")
        final_received = event_times(messages, "final")
        voice_received = status_times(messages, "voice")
        recording_received = status_times(messages, "recording")
        realtime_intervals = [
            (right - left) * 1000.0
            for left, right in zip(realtime_received, realtime_received[1:])
        ]
        final_intervals = [
            (right - left) * 1000.0
            for left, right in zip(final_received, final_received[1:])
        ]

        stream_start = stream_started_at.get(session_id)
        first_realtime_at = first_at_or_after(realtime_received, stream_start)
        first_final_at = first_at_or_after(final_received, stream_start)
        first_recording_at = first_at_or_after(recording_received, stream_start)
        first_voice_at = first_at_or_after(voice_received, stream_start)

        first_realtime_ms = ms_between(stream_start, first_realtime_at)
        first_recording_ms = ms_between(
            stream_start,
            first_recording_at,
        )
        first_voice_ms = ms_between(
            stream_start,
            first_voice_at,
        )
        first_final_ms = ms_between(
            stream_start,
            first_final_at,
        )
        final_after_audio_end_ms = ms_between(
            audio_finished_at.get(session_id),
            final_received[-1] if final_received else None,
        )

        if first_realtime_ms is not None:
            first_realtime_latencies.append(first_realtime_ms)
        if first_final_ms is not None:
            first_final_latencies.append(first_final_ms)
        if final_after_audio_end_ms is not None:
            final_after_audio_end_latencies.append(final_after_audio_end_ms)

        session_metrics = metrics.get("sessions", {}).get(session_id, {})
        final_total_latency = session_metrics.get("totalLatency", {}).get("final", {})
        realtime_total_latency = session_metrics.get("totalLatency", {}).get("realtime", {})
        if final_total_latency.get("p95Ms") is not None:
            final_scheduler_p95.append(final_total_latency.get("p95Ms"))

        session_reports[session_id] = {
            "events": {
                "realtime": len(realtime_messages),
                "final": len(final_messages),
            },
            "latencyMs": {
                "firstVoiceFromStreamStart": round_or_none(first_voice_ms),
                "firstRecordingFromStreamStart": round_or_none(first_recording_ms),
                "firstRealtimeFromStreamStart": round_or_none(first_realtime_ms),
                "firstRealtimeAfterRecordingStart": round_or_none(
                    ms_between(
                        first_recording_at,
                        first_realtime_at,
                    )
                ),
                "firstFinalFromStreamStart": round_or_none(first_final_ms),
                "lastFinalAfterAudioEnd": round_or_none(final_after_audio_end_ms),
                "streamSendDuration": round_or_none(
                    ms_between(stream_started_at.get(session_id), audio_finished_at.get(session_id))
                ),
                "stopCallDuration": round_or_none(
                    ms_between(audio_finished_at.get(session_id), stop_finished_at.get(session_id))
                ),
            },
            "realtimeCadenceMs": {
                "p50": round_or_none(percentile(realtime_intervals, 0.50)),
                "p95": round_or_none(percentile(realtime_intervals, 0.95)),
            },
            "finalCadenceMs": {
                "p50": round_or_none(percentile(final_intervals, 0.50)),
                "p95": round_or_none(percentile(final_intervals, 0.95)),
            },
            "schedulerLatencyMs": {
                "realtimeP50": round_or_none(realtime_total_latency.get("p50Ms")),
                "realtimeP95": round_or_none(realtime_total_latency.get("p95Ms")),
                "finalP50": round_or_none(final_total_latency.get("p50Ms")),
                "finalP95": round_or_none(final_total_latency.get("p95Ms")),
            },
            "quality": {
                "wer": round_or_none(wers.get(session_id), 4),
            },
            "counters": {
                "realtimeSubmitted": session_metrics.get("realtimeSubmitted", 0),
                "realtimeCompleted": session_metrics.get("realtimeCompleted", 0),
                "finalSubmitted": session_metrics.get("finalSubmitted", 0),
                "finalCompleted": session_metrics.get("finalCompleted", 0),
                "coalescedRealtime": session_metrics.get("coalescedRealtime", 0),
                "staleRealtimeDiscarded": session_metrics.get("staleRealtimeDiscarded", 0),
                "realtimeRejected": session_metrics.get("realtimeRejected", 0),
                "finalRejected": session_metrics.get("finalRejected", 0),
            },
        }

    return {
        "kind": "fastapi_multi_user_asr_performance",
        "audio": {
            "path": str(REFERENCE_WAV),
            "durationMs": round_or_none(audio_seconds * 1000.0),
            "chunkMs": chunk_ms,
            "streamSpeed": speed,
        },
        "engine": {
            "final": settings.transcription_engine,
            "finalModel": settings.model,
            "realtime": settings.realtime_transcription_engine,
            "realtimeModel": settings.realtime_model,
            "useMainModelForRealtime": settings.use_main_model_for_realtime,
            "device": settings.device,
            "computeType": settings.compute_type,
        },
        "summary": {
            "clients": len(session_ids),
            "modelReadyMs": round_or_none(ms_between(service_started_at, ready_at)),
            "reportWallMs": round_or_none(ms_between(service_started_at, report_finished_at)),
            "firstRealtimeFromStreamStartP50Ms": round_or_none(percentile(first_realtime_latencies, 0.50)),
            "firstRealtimeFromStreamStartP95Ms": round_or_none(percentile(first_realtime_latencies, 0.95)),
            "firstFinalFromStreamStartP50Ms": round_or_none(percentile(first_final_latencies, 0.50)),
            "firstFinalFromStreamStartP95Ms": round_or_none(percentile(first_final_latencies, 0.95)),
            "lastFinalAfterAudioEndP50Ms": round_or_none(percentile(final_after_audio_end_latencies, 0.50)),
            "lastFinalAfterAudioEndP95Ms": round_or_none(percentile(final_after_audio_end_latencies, 0.95)),
            "firstFinalLatencySkewMs": round_or_none(
                max(first_final_latencies) - min(first_final_latencies)
                if len(first_final_latencies) > 1
                else 0.0
            ),
            "schedulerFinalP95SkewMs": round_or_none(
                max(final_scheduler_p95) - min(final_scheduler_p95)
                if len(final_scheduler_p95) > 1
                else 0.0
            ),
            "werMax": round_or_none(max(wers.values()) if wers else None, 4),
        },
        "sessions": session_reports,
        "scheduler": metrics.get("scheduler", {}),
        "limits": metrics.get("limits", {}),
    }


def emit_performance_report(report):
    if env_flag("REALTIMESTT_FASTAPI_ASR_PRINT_METRICS", default=True):
        print("\nFASTAPI_MULTI_USER_ASR_PERFORMANCE")
        print(json.dumps(report, indent=2, sort_keys=True))
    output_path = os.environ.get("REALTIMESTT_FASTAPI_ASR_METRICS_JSON")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


class PerformanceReportHelperTests(unittest.TestCase):
    def test_env_json_accepts_cmd_friendly_key_value_options(self):
        with mock.patch.dict(
            os.environ,
            {"REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS": "num_threads=4,provider=cpu,vad_filter=false"},
        ):
            self.assertEqual(
                env_json("REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS"),
                {"num_threads": 4, "provider": "cpu", "vad_filter": False},
            )

    def test_env_json_recovers_common_cmd_quote_forms(self):
        with mock.patch.dict(
            os.environ,
            {"REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS": '{""num_threads"":4,""provider"":""cpu""}'},
        ):
            self.assertEqual(
                env_json("REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS"),
                {"num_threads": 4, "provider": "cpu"},
            )

    def test_report_summarizes_event_latencies_and_scheduler_metrics(self):
        settings = ServerSettings(
            transcription_engine="fake-final",
            realtime_transcription_engine="fake-realtime",
            model="final-model",
            realtime_model="realtime-model",
            device="cpu",
        )
        report = build_performance_report(
            settings=settings,
            messages_by_session={
                "client-0": [
                    {"type": "status", "state": "voice", "_receivedAt": 10.2},
                    {"type": "status", "state": "recording", "_receivedAt": 10.3},
                    {"type": "realtime", "_receivedAt": 11.0},
                    {"type": "realtime", "_receivedAt": 11.5},
                    {"type": "final", "_receivedAt": 13.0},
                ],
                "client-1": [
                    {"type": "status", "state": "recording", "_receivedAt": 20.4},
                    {"type": "realtime", "_receivedAt": 21.2},
                    {"type": "final", "_receivedAt": 23.4},
                ],
            },
            session_ids=["client-0", "client-1"],
            stream_started_at={"client-0": 10.0, "client-1": 20.0},
            audio_finished_at={"client-0": 12.0, "client-1": 22.0},
            stop_finished_at={"client-0": 12.1, "client-1": 22.2},
            service_started_at=9.0,
            ready_at=9.5,
            report_finished_at=24.0,
            audio_seconds=2.0,
            chunk_ms=32.0,
            speed=1.0,
            wers={"client-0": 0.1, "client-1": 0.2},
            metrics={
                "sessions": {
                    "client-0": {
                        "totalLatency": {
                            "final": {"p50Ms": 100.0, "p95Ms": 120.0},
                            "realtime": {"p50Ms": 30.0, "p95Ms": 50.0},
                        },
                        "realtimeSubmitted": 2,
                        "realtimeCompleted": 2,
                        "finalSubmitted": 1,
                        "finalCompleted": 1,
                    },
                    "client-1": {
                        "totalLatency": {
                            "final": {"p50Ms": 130.0, "p95Ms": 180.0},
                            "realtime": {"p50Ms": 35.0, "p95Ms": 60.0},
                        },
                        "realtimeSubmitted": 1,
                        "realtimeCompleted": 1,
                        "finalSubmitted": 1,
                        "finalCompleted": 1,
                    },
                },
                "scheduler": {"workers": {}},
                "limits": {},
            },
        )

        self.assertEqual(report["summary"]["modelReadyMs"], 500.0)
        self.assertEqual(report["summary"]["firstFinalLatencySkewMs"], 400.0)
        self.assertEqual(report["summary"]["schedulerFinalP95SkewMs"], 60.0)
        self.assertEqual(report["sessions"]["client-0"]["latencyMs"]["firstVoiceFromStreamStart"], 200.0)
        self.assertEqual(report["sessions"]["client-0"]["latencyMs"]["firstRecordingFromStreamStart"], 300.0)
        self.assertEqual(report["sessions"]["client-0"]["latencyMs"]["firstRealtimeFromStreamStart"], 1000.0)
        self.assertEqual(report["sessions"]["client-0"]["latencyMs"]["firstRealtimeAfterRecordingStart"], 700.0)
        self.assertEqual(report["sessions"]["client-0"]["realtimeCadenceMs"]["p50"], 500.0)
        self.assertEqual(report["summary"]["werMax"], 0.2)


def read_reference_audio():
    with wave.open(str(REFERENCE_WAV), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("reference WAV must be 16-bit PCM")

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate


def make_packet(samples, sample_rate):
    return decode_audio_packet(encode_audio_packet(
        {
            "sampleRate": sample_rate,
            "channels": 1,
            "format": "pcm_s16le",
            "frames": int(samples.size),
        },
        samples.astype(np.int16).tobytes(),
    ))


@unittest.skipUnless(
    run_real_asr_test_enabled(),
    "set REALTIMESTT_RUN_FASTAPI_MULTI_USER_ASR=1 or REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF=1 to run real-engine multi-user ASR load test",
)
class FastAPIMultiUserRealEngineASRTests(unittest.TestCase):
    def test_parallel_reference_wav_sessions_meet_quality_and_fairness_bounds(self):
        with EXPECTED_JSON.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)["combined_normalized"]

        samples, sample_rate = read_reference_audio()
        client_count = env_int("REALTIMESTT_FASTAPI_ASR_CLIENTS", 2)
        chunk_ms = env_float("REALTIMESTT_FASTAPI_ASR_CHUNK_MS", 32.0)
        speed = env_float("REALTIMESTT_FASTAPI_ASR_SPEED", 1.0)
        timeout_seconds = env_float("REALTIMESTT_FASTAPI_ASR_TIMEOUT", 240.0)
        max_wer = env_float("REALTIMESTT_FASTAPI_ASR_MAX_WER", 0.30)
        max_latency_skew_ms = env_float("REALTIMESTT_FASTAPI_ASR_MAX_P95_SKEW_MS", 30000.0)
        audio_seconds = samples.size / float(sample_rate)

        settings = ServerSettings(
            transcription_engine=os.environ.get("REALTIMESTT_FASTAPI_ASR_ENGINE", "faster_whisper"),
            realtime_transcription_engine=os.environ.get("REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE") or None,
            model=os.environ.get("REALTIMESTT_FASTAPI_ASR_MODEL", "small.en"),
            realtime_model=os.environ.get("REALTIMESTT_FASTAPI_ASR_REALTIME_MODEL", "tiny.en"),
            language=os.environ.get("REALTIMESTT_FASTAPI_ASR_LANGUAGE", "en"),
            device=os.environ.get("REALTIMESTT_FASTAPI_ASR_DEVICE", "cuda"),
            compute_type=os.environ.get("REALTIMESTT_FASTAPI_ASR_COMPUTE_TYPE", "default"),
            download_root=os.environ.get("REALTIMESTT_FASTAPI_ASR_DOWNLOAD_ROOT") or None,
            transcription_engine_options=env_json("REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS"),
            realtime_transcription_engine_options=env_json("REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE_OPTIONS"),
            beam_size=env_int("REALTIMESTT_FASTAPI_ASR_BEAM_SIZE", 5),
            beam_size_realtime=env_int("REALTIMESTT_FASTAPI_ASR_BEAM_SIZE_REALTIME", 3),
            batch_size=env_int("REALTIMESTT_FASTAPI_ASR_BATCH_SIZE", 16),
            realtime_batch_size=env_int("REALTIMESTT_FASTAPI_ASR_REALTIME_BATCH_SIZE", 16),
            use_main_model_for_realtime=env_flag("REALTIMESTT_FASTAPI_ASR_USE_MAIN_MODEL_FOR_REALTIME"),
            max_sessions=max(client_count, 2),
            max_active_speakers=max(client_count, 2),
            max_global_inference_queue_depth=env_int(
                "REALTIMESTT_FASTAPI_ASR_MAX_GLOBAL_INFERENCE_QUEUE_DEPTH",
                max(64, client_count * 16),
            ),
            max_final_queue_depth_per_session=env_int(
                "REALTIMESTT_FASTAPI_ASR_MAX_FINAL_QUEUE_DEPTH_PER_SESSION",
                8,
            ),
            max_realtime_queue_age_ms=env_int("REALTIMESTT_FASTAPI_ASR_MAX_REALTIME_QUEUE_AGE_MS", 1500),
            realtime_processing_pause=env_float("REALTIMESTT_FASTAPI_ASR_REALTIME_PROCESSING_PAUSE", 0.25),
            realtime_min_audio_seconds=env_float("REALTIMESTT_FASTAPI_ASR_REALTIME_MIN_AUDIO_SECONDS", 0.25),
            post_speech_silence_duration=env_float("REALTIMESTT_FASTAPI_ASR_POST_SPEECH_SILENCE_DURATION", 0.6),
            min_length_of_recording=env_float("REALTIMESTT_FASTAPI_ASR_MIN_LENGTH_OF_RECORDING", 0.2),
            early_transcription_on_silence=env_float("REALTIMESTT_FASTAPI_ASR_EARLY_TRANSCRIPTION_ON_SILENCE", 0.2),
            realtime_transcription_use_syllable_boundaries=env_flag(
                "REALTIMESTT_FASTAPI_ASR_REALTIME_USE_SYLLABLE_BOUNDARIES"
            ),
            realtime_boundary_detector_sensitivity=env_float(
                "REALTIMESTT_FASTAPI_ASR_REALTIME_BOUNDARY_DETECTOR_SENSITIVITY",
                0.6,
            ),
            realtime_boundary_followup_delays=env_float_tuple(
                "REALTIMESTT_FASTAPI_ASR_REALTIME_BOUNDARY_FOLLOWUP_DELAYS",
                (0.05, 0.2),
            ),
        )
        manager = CollectingManager()
        service = RealtimeSTTService(settings, manager)
        service_started_at = time.monotonic()
        ready_at = None
        stream_started_at = {}
        audio_finished_at = {}
        stop_finished_at = {}
        timing_lock = threading.Lock()
        try:
            service.start(loop=None)
            self.assertTrue(service.scheduler.wait_ready(timeout=timeout_seconds))
            self.assertTrue(service.scheduler.healthy(), service.startup_errors)
            ready_at = time.monotonic()

            sessions = []
            for index in range(client_count):
                session = service.admit_session(f"client-{index}")
                self.assertIsNotNone(session)
                session.start_streaming()
                sessions.append(session)

            chunk_size = max(1, int(round(sample_rate * chunk_ms / 1000.0)))
            errors = []
            barrier = threading.Barrier(client_count)

            def feed_session(session):
                try:
                    barrier.wait(timeout=30)
                    with timing_lock:
                        stream_started_at[session.session_id] = time.monotonic()
                    for start in range(0, samples.size, chunk_size):
                        chunk = samples[start:start + chunk_size]
                        session.ingest_audio_packet(make_packet(chunk, sample_rate))
                        if speed > 0:
                            time.sleep((chunk.size / float(sample_rate)) / speed)
                    with timing_lock:
                        audio_finished_at[session.session_id] = time.monotonic()
                    session.stop_streaming()
                    with timing_lock:
                        stop_finished_at[session.session_id] = time.monotonic()
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=feed_session, args=(session,), daemon=True)
                for session in sessions
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=timeout_seconds)

            self.assertFalse(errors)
            self.assertTrue(all(not thread.is_alive() for thread in threads))

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                finals_ready = True
                with manager._sync_lock:
                    for session in sessions:
                        finals = [
                            msg for msg in manager.messages.get(session.session_id, [])
                            if msg.get("type") == "final"
                        ]
                        if not finals:
                            finals_ready = False
                            break
                if finals_ready:
                    break
                time.sleep(0.25)

            wers = {}
            transcripts = {}
            for session in sessions:
                with manager._sync_lock:
                    finals = [
                        msg for msg in manager.messages.get(session.session_id, [])
                        if msg.get("type") == "final"
                    ]
                transcript = " ".join(msg["text"] for msg in finals)
                transcripts[session.session_id] = transcript
                wers[session.session_id] = word_error_rate(expected, transcript)

            metrics = service.metrics()
            session_ids = [session.session_id for session in sessions]
            with manager._sync_lock:
                messages_by_session = {
                    session_id: list(manager.messages.get(session_id, []))
                    for session_id in session_ids
                }
            report = build_performance_report(
                settings=settings,
                messages_by_session=messages_by_session,
                session_ids=session_ids,
                stream_started_at=dict(stream_started_at),
                audio_finished_at=dict(audio_finished_at),
                stop_finished_at=dict(stop_finished_at),
                service_started_at=service_started_at,
                ready_at=ready_at,
                report_finished_at=time.monotonic(),
                audio_seconds=audio_seconds,
                chunk_ms=chunk_ms,
                speed=speed,
                wers=wers,
                metrics=metrics,
            )
            emit_performance_report(report)

            final_p95 = []
            for session in sessions:
                self.assertLessEqual(
                    wers[session.session_id],
                    max_wer,
                    f"{session.session_id} transcript WER too high: {transcripts[session.session_id]}",
                )
                session_metrics = metrics["sessions"][session.session_id]
                self.assertGreater(session_metrics["finalCompleted"], 0)
                final_p95.append(session_metrics["totalLatency"]["final"]["p95Ms"])
            if len(final_p95) > 1:
                self.assertLessEqual(max(final_p95) - min(final_p95), max_latency_skew_ms)
        finally:
            service.stop()


if __name__ == "__main__":
    unittest.main()
