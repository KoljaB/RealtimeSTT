import collections
import queue
import threading
import time
import unittest

import numpy as np

from example_fastapi_server.protocol import decode_audio_packet, encode_audio_packet
from example_fastapi_server.server import (
    create_app,
    InferenceResult,
    QueueSubmitResult,
    RecorderBackedRealtimeSession,
    RealtimeSTTService,
    SegmentState,
    ServerSettings,
)
from RealtimeSTT.core.realtime_text_stabilizer import (
    RealtimeTextEvidenceDiagnostics,
    RealtimeTextObservationTiming,
    RealtimeTextStabilizationEvent,
)


class CollectingManager:
    def __init__(self):
        self.messages = collections.defaultdict(list)
        self.global_messages = []

    def bind_loop(self, loop):
        pass

    def publish_session(self, session_id, message):
        self.messages[session_id].append(message)

    def publish_all(self, message):
        self.global_messages.append(message)


class RecorderRealtimeStabilizationPayloadTests(unittest.TestCase):
    def test_structured_stabilization_event_publishes_split_realtime_fields(self):
        manager = CollectingManager()
        service = type("Service", (), {})()
        service.manager = manager
        service.settings = ServerSettings(realtime_callback="update")

        session = RecorderBackedRealtimeSession.__new__(RecorderBackedRealtimeSession)
        session.service = service
        session.settings = service.settings
        session.session_id = "session-a"
        session.segment_state = SegmentState()
        session.lock = threading.RLock()
        session.reject_current_recording = False

        event = RealtimeTextStabilizationEvent(
            recording_id=3,
            segment_id=7,
            sequence=11,
            accepted=True,
            ignored_reason=None,
            publish_allowed=True,
            should_publish=True,
            raw_observation_text="Hello world",
            stable_text="Hello",
            stable_delta="Hello",
            unstable_text=" world",
            display_text="Hello world",
            stable_normalized_offset=5,
            stable_raw_end_offset=5,
            stable_audio_end_sample_exclusive=None,
            has_new_stable_text=True,
            is_outlier=False,
            stable_prefix_conflict=False,
            commit_reason="evidence-threshold",
            evidence=RealtimeTextEvidenceDiagnostics(),
            timing=RealtimeTextObservationTiming(
                created_at_monotonic=1.0,
                completed_at_monotonic=1.2,
            ),
            trigger_reason="timer",
        )

        session._on_realtime_stabilization_event(event)

        [message] = manager.messages["session-a"]
        self.assertEqual(message["type"], "realtime")
        self.assertEqual(message["segmentId"], 7)
        self.assertEqual(message["recordingId"], 3)
        self.assertEqual(message["sequence"], 11)
        self.assertEqual(message["text"], "Hello world")
        self.assertEqual(message["rawText"], "Hello world")
        self.assertEqual(message["displayText"], "Hello world")
        self.assertEqual(message["stableText"], "Hello")
        self.assertEqual(message["stableDelta"], "Hello")
        self.assertEqual(message["unstableText"], " world")
        self.assertEqual(message["committedStableText"], "Hello")
        self.assertEqual(message["visualStableText"], "Hello")
        self.assertEqual(message["visualUnstableText"], " world")
        self.assertTrue(message["publicConsensusAligned"])
        self.assertFalse(message["isOutlier"])
        self.assertEqual(message["timing"]["completed_at_monotonic"], 1.2)

    def test_structured_stabilization_event_keeps_committed_stable_visible(self):
        manager = CollectingManager()
        service = type("Service", (), {})()
        service.manager = manager
        service.settings = ServerSettings(realtime_callback="update")

        session = RecorderBackedRealtimeSession.__new__(RecorderBackedRealtimeSession)
        session.service = service
        session.settings = service.settings
        session.session_id = "session-a"
        session.segment_state = SegmentState()
        session.lock = threading.RLock()
        session.reject_current_recording = False

        event = RealtimeTextStabilizationEvent(
            recording_id=3,
            segment_id=7,
            sequence=12,
            accepted=True,
            ignored_reason=None,
            publish_allowed=True,
            should_publish=True,
            raw_observation_text="I would think that the current approach",
            stable_text="I would think that the card",
            stable_delta="",
            unstable_text=" ... current approach",
            display_text="I would think that the card ... current approach",
            stable_normalized_offset=27,
            stable_raw_end_offset=27,
            stable_audio_end_sample_exclusive=None,
            has_new_stable_text=False,
            is_outlier=False,
            stable_prefix_conflict=True,
            commit_reason="none",
            evidence=RealtimeTextEvidenceDiagnostics(),
            timing=RealtimeTextObservationTiming(),
            trigger_reason="timer",
            consensus_text="I would think that the current approach",
            consensus_unstable_text="",
            consensus_display_text="I would think that the current approach",
            consensus_normalized_offset=39,
            public_consensus_aligned=False,
            internal_revision=True,
        )

        session._on_realtime_stabilization_event(event)

        [message] = manager.messages["session-a"]
        self.assertEqual(message["stableText"], "I would think that the card")
        self.assertEqual(
            message["visualStableText"],
            "I would think that the card",
        )
        self.assertEqual(message["committedStableText"], "I would think that the card")
        self.assertEqual(message["stableDelta"], "")
        self.assertEqual(
            message["unstableText"],
            " ... current approach",
        )
        self.assertEqual(
            message["displayText"],
            "I would think that the card ... current approach",
        )
        self.assertEqual(
            message["consensusDisplayText"],
            "I would think that the current approach",
        )
        self.assertFalse(message["publicConsensusAligned"])
        self.assertTrue(message["internalRevision"])


class ManualScheduler:
    def __init__(self, settings, result_callback, drop_callback=None, error_callback=None):
        self.settings = settings
        self.result_callback = result_callback
        self.drop_callback = drop_callback
        self.error_callback = error_callback
        self.jobs = []
        self.cancelled_sessions = []

    def start(self):
        pass

    def stop(self):
        pass

    def wait_ready(self, timeout=None):
        return True

    def healthy(self):
        return True

    def submit(self, job):
        self.jobs.append(job)
        return QueueSubmitResult(True)

    def cancel_session(self, session_id):
        self.cancelled_sessions.append(session_id)

    def snapshot(self):
        return {"jobs": len(self.jobs)}

    def complete(self, job, text=None, error=None, delay=0.001):
        started_at = job.created_at + delay
        completed_at = started_at + delay
        self.result_callback(
            InferenceResult(
                request_id=job.request_id,
                session_id=job.session_id,
                kind=job.kind,
                segment_id=job.segment_id,
                sequence=job.sequence,
                generation=job.generation,
                text=text if text is not None else f"{job.kind}-{job.session_id}",
                error=error,
                created_at=job.created_at,
                started_at=started_at,
                completed_at=completed_at,
                queue_delay=max(0.0, started_at - job.created_at),
                inference_duration=max(0.0, completed_at - started_at),
                total_latency=max(0.0, completed_at - job.created_at),
            )
        )


class AutoScheduler(ManualScheduler):
    def submit(self, job):
        result = super().submit(job)
        threading.Thread(
            target=self.complete,
            args=(job, f"{job.kind}-{job.session_id}"),
            daemon=True,
        ).start()
        return result


class FakeRecorder:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeRecorder.instances.append(self)
        self.on_recording_start = kwargs.get("on_recording_start")
        self.on_recording_stop = kwargs.get("on_recording_stop")
        self.on_transcription_start = kwargs.get("on_transcription_start")
        self.on_wakeword_detected = kwargs.get("on_wakeword_detected")
        self.on_wakeword_timeout = kwargs.get("on_wakeword_timeout")
        self.on_wakeword_detection_start = kwargs.get("on_wakeword_detection_start")
        self.on_wakeword_detection_end = kwargs.get("on_wakeword_detection_end")
        self.on_vad_start = kwargs.get("on_vad_start")
        self.on_vad_stop = kwargs.get("on_vad_stop")
        self.on_vad_detect_start = kwargs.get("on_vad_detect_start")
        self.on_vad_detect_stop = kwargs.get("on_vad_detect_stop")
        self.realtime_callback = (
            kwargs.get("on_realtime_transcription_update")
            or kwargs.get("on_realtime_transcription_stabilized")
        )
        self.transcription_executor = kwargs["transcription_executor"]
        self.realtime_transcription_executor = kwargs["realtime_transcription_executor"]
        self.audio_queue = queue.Queue()
        self.recorded_audio_queue = queue.Queue()
        self.final_text = queue.Queue()
        self.wake_word_timeout = kwargs.get("wake_word_timeout", 5.0)
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.is_recording = False
        self.is_shut_down = False
        self.has_audio = False

    def feed_audio(self, samples, original_sample_rate=16000):
        self.audio_queue.put(samples)
        self.has_audio = True
        if not self.is_recording:
            self.is_recording = True
            if self.on_recording_start:
                self.on_recording_start()

        def run_realtime():
            try:
                result = self.realtime_transcription_executor.transcribe(samples, language="en", use_prompt=True)
            except RuntimeError:
                return
            if self.realtime_callback and result.text:
                self.realtime_callback(result.text)

        threading.Thread(target=run_realtime, daemon=True).start()

    def flush_buffered_audio(self):
        if not self.has_audio:
            return False
        self.has_audio = False
        self.is_recording = False
        if self.on_recording_stop:
            self.on_recording_stop()

        def run_final():
            abort = self.on_transcription_start(None) if self.on_transcription_start else False
            if abort:
                self.final_text.put("")
                return
            try:
                result = self.transcription_executor.transcribe(np.ones(32, dtype=np.float32), language="en", use_prompt=True)
            except RuntimeError:
                self.final_text.put("")
                return
            self.final_text.put(result.text)

        threading.Thread(target=run_final, daemon=True).start()
        return True

    def text(self):
        item = self.final_text.get()
        if item is None:
            return ""
        return item

    def abort(self):
        self.final_text.put("")

    def stop(self):
        return self.flush_buffered_audio()

    def shutdown(self):
        self.is_shut_down = True
        self.final_text.put(None)


def make_service(**overrides):
    model_warmup = overrides.pop("model_warmup", False)
    settings = ServerSettings(
        model_warmup=model_warmup,
        realtime_processing_pause=0.0,
        realtime_min_audio_seconds=0.01,
        min_length_of_recording=0.0,
        post_speech_silence_duration=60.0,
        vad_energy_threshold=1.0,
        webrtc_sensitivity=99,
        **overrides,
    )
    manager = CollectingManager()
    service = RealtimeSTTService(
        settings,
        manager,
        scheduler_factory=ManualScheduler,
        recorder_factory=FakeRecorder,
    )
    return service, manager


def audio_packet(level=2000, frames=640, sample_rate=16000):
    samples = np.full(frames, level, dtype=np.int16)
    return decode_audio_packet(encode_audio_packet(
        {"sampleRate": sample_rate, "channels": 1, "format": "pcm_s16le", "frames": frames},
        samples.tobytes(),
    ))


class FastAPIMultiUserSessionTests(unittest.TestCase):
    def setUp(self):
        FakeRecorder.instances.clear()

    def test_sessions_receive_only_their_own_final_transcripts(self):
        service, manager = make_service()
        first = service.admit_session("first")
        second = service.admit_session("second")

        first.start_streaming()
        second.start_streaming()
        self.assertTrue(first.ingest_audio_packet(audio_packet(level=2000))[0])
        self.assertTrue(second.ingest_audio_packet(audio_packet(level=3000))[0])
        first.stop_streaming()
        second.stop_streaming()

        self._wait_for(lambda: len([job for job in service.scheduler.jobs if job.kind == "final"]) >= 2)
        final_jobs = [job for job in service.scheduler.jobs if job.kind == "final"]
        self.assertEqual({job.session_id for job in final_jobs}, {"first", "second"})

        for job in final_jobs:
            service.scheduler.complete(job, text=f"private-{job.session_id}")

        self._wait_for(lambda: any(msg.get("type") == "final" for msg in manager.messages["first"]))
        self._wait_for(lambda: any(msg.get("type") == "final" for msg in manager.messages["second"]))

        first_finals = [msg for msg in manager.messages["first"] if msg.get("type") == "final"]
        second_finals = [msg for msg in manager.messages["second"] if msg.get("type") == "final"]

        self.assertEqual([msg["text"] for msg in first_finals], ["private-first"])
        self.assertEqual([msg["text"] for msg in second_finals], ["private-second"])
        self.assertNotIn("private-second", [msg.get("text") for msg in manager.messages["first"]])
        self.assertNotIn("private-first", [msg.get("text") for msg in manager.messages["second"]])

    def test_clear_resets_only_one_session_and_discards_old_results(self):
        service, manager = make_service()
        first = service.admit_session("first")
        second = service.admit_session("second")

        for session in (first, second):
            session.start_streaming()
            session.ingest_audio_packet(audio_packet())
            session.stop_streaming()

        self._wait_for(lambda: len([job for job in service.scheduler.jobs if job.kind == "final"]) >= 2)
        first_final = next(job for job in service.scheduler.jobs if job.kind == "final" and job.session_id == "first")
        second_final = next(job for job in service.scheduler.jobs if job.kind == "final" and job.session_id == "second")

        first.clear()
        service.scheduler.complete(first_final, text="old-first")
        service.scheduler.complete(second_final, text="still-second")
        self._wait_for(lambda: any(msg.get("text") == "still-second" for msg in manager.messages["second"]))

        self.assertTrue(any(msg.get("type") == "clear" for msg in manager.messages["first"]))
        self.assertFalse(any(msg.get("type") == "clear" for msg in manager.messages["second"]))
        self.assertFalse(any(msg.get("text") == "old-first" for msg in manager.messages["first"]))
        self.assertTrue(any(msg.get("text") == "still-second" for msg in manager.messages["second"]))

    def test_stale_final_text_from_previous_generation_is_not_published(self):
        service, manager = make_service()
        session = service.admit_session("first")
        stale_generation = session.generation

        session.clear()
        published = session._publish_final_text("old-final", stale_generation)

        self.assertFalse(published)
        self.assertFalse(any(msg.get("text") == "old-final" for msg in manager.messages["first"]))

    def test_active_speaker_limit_rejects_only_new_speaker(self):
        service, manager = make_service(max_active_speakers=1)
        first = service.admit_session("first")
        second = service.admit_session("second")

        first.start_streaming()
        second.start_streaming()

        self.assertTrue(first.ingest_audio_packet(audio_packet())[0])
        accepted, warning = second.ingest_audio_packet(audio_packet())

        self.assertTrue(accepted)
        self.assertIsNone(warning)
        self.assertTrue(any("active speaker limit" in msg.get("message", "") for msg in manager.messages["second"]))
        self.assertEqual(service.active_speaker_count(), 1)

    def test_session_admission_limit_is_explicit(self):
        service, _ = make_service(max_sessions=1)

        self.assertIsNotNone(service.admit_session("first"))
        self.assertIsNone(service.admit_session("second"))

        metrics = service.metrics()
        self.assertEqual(metrics["activeSessions"], 1)
        self.assertEqual(metrics["rejectedSessions"], 1)

    def test_recorder_sessions_receive_realtime_boundary_configuration(self):
        service, _ = make_service(
            realtime_transcription_use_syllable_boundaries=True,
            realtime_boundary_detector_sensitivity=0.7,
            realtime_boundary_followup_delays=(0.1, 0.2, 0.4),
        )

        self.assertIsNotNone(service.admit_session("first"))

        config = FakeRecorder.instances[-1].kwargs
        self.assertTrue(config["realtime_transcription_use_syllable_boundaries"])
        self.assertEqual(config["realtime_boundary_detector_sensitivity"], 0.7)
        self.assertEqual(config["realtime_boundary_followup_delays"], (0.1, 0.2, 0.4))
        self.assertFalse(config["warmup_vad"])

    def test_recorder_sessions_enable_vad_warmup_with_model_warmup(self):
        service, _ = make_service(model_warmup=True)

        self.assertIsNotNone(service.admit_session("first"))

        config = FakeRecorder.instances[-1].kwargs
        self.assertTrue(config["warmup_vad"])

    def test_recorder_sessions_receive_wake_word_configuration(self):
        service, _ = make_service(
            wakeword_backend="pvporcupine",
            wake_words="jarvis",
            wake_words_sensitivity=0.72,
            wake_word_timeout=3.5,
            wake_word_buffer_duration=0.25,
        )

        self.assertIsNotNone(service.admit_session("first"))

        config = FakeRecorder.instances[-1].kwargs
        self.assertEqual(config["wakeword_backend"], "pvporcupine")
        self.assertEqual(config["wake_words"], "jarvis")
        self.assertEqual(config["wake_words_sensitivity"], 0.72)
        self.assertEqual(config["wake_word_timeout"], 3.5)
        self.assertEqual(config["wake_word_buffer_duration"], 0.25)
        self.assertTrue(callable(config["on_wakeword_detected"]))
        self.assertTrue(callable(config["on_wakeword_timeout"]))

    def test_wake_word_callbacks_publish_status_and_timeline_events(self):
        service, manager = make_service(
            wakeword_backend="pvporcupine",
            wake_words="jarvis",
        )
        session = service.admit_session("first")
        self.assertIsNotNone(session)
        recorder = FakeRecorder.instances[-1]

        recorder.on_wakeword_detection_start()
        recorder.on_wakeword_detected()
        recorder.on_wakeword_timeout()

        timeline_events = [
            message.get("event")
            for message in manager.messages["first"]
            if message.get("type") == "timeline"
        ]
        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]

        self.assertIn("wakeword_wait_started", timeline_events)
        self.assertIn("wakeword_detected", timeline_events)
        self.assertIn("wakeword_timeout", timeline_events)
        self.assertIn("wakeword_wait", statuses)
        self.assertIn("wakeword_detected", statuses)
        self.assertIn("wakeword_timeout", statuses)

    def test_wake_word_session_returns_to_wake_wait_after_recording_and_final(self):
        service, manager = make_service(
            wakeword_backend="pvporcupine",
            wake_words="jarvis",
        )
        session = service.admit_session("first")
        self.assertIsNotNone(session)
        session.start_streaming()

        session._on_wakeword_detected()
        session._on_recording_start()
        session._on_recording_stop()

        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]
        self.assertEqual(statuses[-1], "wakeword_wait")

        self.assertTrue(session._publish_final_text("done", session.generation))
        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]
        self.assertEqual(statuses[-1], "wakeword_wait")

    def test_wake_word_followup_window_stays_in_voice_mode_after_recording(self):
        service, manager = make_service(
            wakeword_backend="pvporcupine",
            wake_words="jarvis",
            wake_word_timeout=3.0,
            wake_word_followup_window=5.0,
        )
        session = service.admit_session("first")
        self.assertIsNotNone(session)
        session.start_streaming()
        recorder = FakeRecorder.instances[-1]

        session._on_wakeword_detected()
        session._on_recording_start()
        session._on_recording_stop()

        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]
        self.assertEqual(statuses[-1], "wakeword_detected")
        self.assertTrue(recorder.wakeword_detected)
        self.assertEqual(recorder.wake_word_timeout, 5.0)
        self.assertTrue(recorder.start_recording_on_voice_activity)
        self.assertTrue(recorder.stop_recording_on_voice_deactivity)
        self.assertTrue(any(
            message.get("event") == "wakeword_followup_started"
            for message in manager.messages["first"]
            if message.get("type") == "timeline"
        ))

        self.assertTrue(session._publish_final_text("done", session.generation))
        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]
        self.assertEqual(statuses[-1], "wakeword_detected")

        generation = session._wakeword_followup_generation
        self.assertTrue(session._finish_wakeword_followup(generation))
        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]
        self.assertEqual(statuses[-1], "wakeword_wait")
        self.assertFalse(recorder.wakeword_detected)
        self.assertEqual(recorder.wake_word_timeout, 3.0)
        self.assertFalse(recorder.start_recording_on_voice_activity)
        self.assertFalse(recorder.stop_recording_on_voice_deactivity)
        self.assertTrue(any(
            message.get("event") == "wakeword_followup_timeout"
            for message in manager.messages["first"]
            if message.get("type") == "timeline"
        ))

    def test_wake_word_session_ignores_late_vad_detect_start_after_reset(self):
        service, manager = make_service(
            wakeword_backend="pvporcupine",
            wake_words="jarvis",
        )
        session = service.admit_session("first")
        self.assertIsNotNone(session)
        session.start_streaming()
        recorder = FakeRecorder.instances[-1]

        recorder.on_wakeword_detected()
        recorder.on_vad_detect_start()
        recorder.on_recording_start()
        recorder.on_recording_stop()
        recorder.on_vad_detect_start()

        statuses = [
            message.get("state")
            for message in manager.messages["first"]
            if message.get("type") == "status"
        ]
        self.assertIn("voice", statuses)
        self.assertEqual(statuses[-1], "wakeword_wait")

    def test_recording_timeline_metadata_is_attached_to_final_segment(self):
        service, manager = make_service(pre_recording_buffer_duration=0.4)
        session = service.admit_session("first")
        session.start_streaming()

        self.assertTrue(session.ingest_audio_packet(audio_packet())[0])
        session.stop_streaming()

        self._wait_for(lambda: any(job.kind == "final" for job in service.scheduler.jobs))
        final_job = next(job for job in service.scheduler.jobs if job.kind == "final")
        service.scheduler.complete(final_job, text="private-final")
        self._wait_for(lambda: any(msg.get("type") == "final" for msg in manager.messages["first"]))

        final = next(msg for msg in manager.messages["first"] if msg.get("type") == "final")
        segment = final["segment"]
        self.assertEqual(final["segmentId"], segment["segmentId"])
        self.assertIn("recordingStartedAt", segment)
        self.assertIn("recordingEndedAt", segment)
        self.assertIn("durationSeconds", segment)
        self.assertIn("preRecordingBuffer", segment)
        self.assertEqual(segment["preRecordingBuffer"]["configuredSeconds"], 0.4)
        self.assertTrue(any(
            msg.get("type") == "timeline" and msg.get("event") == "recording_started"
            for msg in manager.messages["first"]
        ))
        self.assertTrue(any(
            msg.get("type") == "timeline" and msg.get("event") == "final_transcript"
            for msg in manager.messages["first"]
        ))

    def test_sessions_use_separate_recorder_vad_state_with_shared_executors(self):
        service, _ = make_service(audio_queue_size=7)

        first = service.admit_session("first")
        second = service.admit_session("second")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(FakeRecorder.instances), 2)
        self.assertIsNot(FakeRecorder.instances[0], FakeRecorder.instances[1])
        for recorder in FakeRecorder.instances:
            self.assertIs(recorder.transcription_executor.service, service)
            self.assertIs(recorder.realtime_transcription_executor.service, service)
            self.assertEqual(recorder.kwargs["allowed_latency_limit"], 7)
            self.assertTrue(recorder.kwargs["handle_buffer_overflow"])
            self.assertTrue(callable(recorder.kwargs["on_recorded_chunk"]))

    def test_audio_packets_are_rejected_until_session_is_started(self):
        service, _ = make_service()
        session = service.admit_session("first")

        accepted, warning = session.ingest_audio_packet(audio_packet())

        self.assertFalse(accepted)
        self.assertIn("start command", warning)
        self.assertEqual(service.scheduler.jobs, [])
        self.assertEqual(session.snapshot()["rejectedAudioChunks"], 1)

    def test_max_recording_duration_forces_recorder_finalization(self):
        service, manager = make_service(max_audio_queue_seconds_per_session=0.01)
        session = service.admit_session("first")
        session.start_streaming()

        accepted, warning = session.ingest_audio_packet(audio_packet(frames=640))

        self.assertTrue(accepted)
        self.assertIsNone(warning)
        self._wait_for(lambda: any(job.kind == "final" for job in service.scheduler.jobs))
        self.assertEqual(session.snapshot()["forcedFinalizations"], 1)
        self.assertTrue(any(
            "Maximum per-session audio buffer" in msg.get("message", "")
            for msg in manager.messages["first"]
        ))

    def test_processed_recorded_chunks_enforce_max_recording_duration(self):
        service, manager = make_service(max_audio_queue_seconds_per_session=0.01)
        session = service.admit_session("first")
        session.start_streaming()
        session.recorder.is_recording = True
        session.recorder.has_audio = True

        session._on_recorded_chunk(np.zeros(640, dtype=np.int16).tobytes())

        self._wait_for(lambda: session.snapshot()["forcedFinalizations"] == 1)
        self.assertTrue(any(
            "Maximum per-session audio buffer" in msg.get("message", "")
            for msg in manager.messages["first"]
        ))

    def test_recorded_audio_backlog_is_trimmed_to_final_queue_limit(self):
        service, manager = make_service(max_final_queue_depth_per_session=1)
        session = service.admit_session("first")
        queue_obj = session.recorder.recorded_audio_queue
        queue_obj.put({"frames": [b"a"]})
        queue_obj.put({"frames": [b"b"]})
        queue_obj.put({"frames": [b"c"]})

        dropped = session._trim_recorded_audio_queue()

        self.assertEqual(dropped, 2)
        self.assertEqual(queue_obj.qsize(), 1)
        snapshot = session.snapshot()
        self.assertEqual(snapshot["droppedRecordedSegments"], 2)
        self.assertEqual(snapshot["finalRejected"], 2)
        self.assertTrue(any(
            "Final transcription backlog" in msg.get("message", "")
            for msg in manager.messages["first"]
        ))

    def test_service_stop_closes_active_sessions(self):
        service, _ = make_service()
        session = service.admit_session("first")

        service.stop()

        self.assertEqual(service.session_count(), 0)
        self.assertTrue(session.recorder.is_shut_down)

    def test_session_reservation_prevents_recorder_construction_over_capacity(self):
        class SlowRecorder(FakeRecorder):
            created = 0
            entered = threading.Event()
            release = threading.Event()

            def __init__(self, **kwargs):
                SlowRecorder.created += 1
                SlowRecorder.entered.set()
                SlowRecorder.release.wait(timeout=2.0)
                super().__init__(**kwargs)

        settings = ServerSettings(
            model_warmup=False,
            realtime_processing_pause=0.0,
            realtime_min_audio_seconds=0.01,
            min_length_of_recording=0.0,
            post_speech_silence_duration=60.0,
            vad_energy_threshold=1.0,
            webrtc_sensitivity=99,
            max_sessions=1,
        )
        manager = CollectingManager()
        service = RealtimeSTTService(
            settings,
            manager,
            scheduler_factory=ManualScheduler,
            recorder_factory=SlowRecorder,
        )
        admitted = []

        first_thread = threading.Thread(
            target=lambda: admitted.append(service.admit_session("first")),
            daemon=True,
        )
        first_thread.start()
        self.assertTrue(SlowRecorder.entered.wait(timeout=1.0))

        second = service.admit_session("second")
        self.assertIsNone(second)
        self.assertEqual(SlowRecorder.created, 1)

        SlowRecorder.release.set()
        first_thread.join(timeout=2.0)
        self.assertEqual(len(admitted), 1)
        self.assertIsNotNone(admitted[0])
        self.assertEqual(service.session_count(), 1)

    def test_realtime_results_are_routed_to_owner(self):
        service, manager = make_service()
        first = service.admit_session("first")
        second = service.admit_session("second")
        first.start_streaming()
        second.start_streaming()

        first.ingest_audio_packet(audio_packet(level=2000))
        second.ingest_audio_packet(audio_packet(level=2200))

        self._wait_for(lambda: len([job for job in service.scheduler.jobs if job.kind == "realtime"]) >= 2)
        realtime_jobs = [job for job in service.scheduler.jobs if job.kind == "realtime"]
        self.assertEqual({job.session_id for job in realtime_jobs}, {"first", "second"})
        for job in realtime_jobs:
            service.scheduler.complete(job, text=f"rt-{job.session_id}")

        self._wait_for(lambda: any(msg.get("type") == "realtime" for msg in manager.messages["first"]))
        self._wait_for(lambda: any(msg.get("type") == "realtime" for msg in manager.messages["second"]))

        self.assertTrue(any(msg.get("text") == "rt-first" for msg in manager.messages["first"]))
        self.assertTrue(any(msg.get("text") == "rt-second" for msg in manager.messages["second"]))
        self.assertFalse(any(msg.get("text") == "rt-second" for msg in manager.messages["first"]))
        self.assertFalse(any(msg.get("text") == "rt-first" for msg in manager.messages["second"]))

    def _wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Timed out waiting for condition")


try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover - optional dependency
    TestClient = None
    FASTAPI_IMPORT_ERROR = exc
else:
    FASTAPI_IMPORT_ERROR = None


@unittest.skipIf(TestClient is None, "FastAPI test client is not installed")
class FastAPIMultiUserWebSocketTests(unittest.TestCase):
    def test_config_endpoint_exposes_and_updates_runtime_settings(self):
        settings = ServerSettings(model_warmup=False, max_sessions=1)
        app = create_app(settings, scheduler_factory=AutoScheduler, recorder_factory=FakeRecorder)

        with TestClient(app) as client:
            config = client.get("/api/config")
            self.assertEqual(config.status_code, 200)
            config_body = config.json()
            self.assertIn("runtimeSettings", config_body)
            self.assertIn("kroko_onnx", config_body["supportedEngines"])

            update = client.patch(
                "/api/config",
                json={"settings": {"max_sessions": 3, "wake_words": "jarvis"}},
            )

            self.assertEqual(update.status_code, 200)
            body = update.json()
            self.assertEqual(body["applied"]["max_sessions"]["appliesTo"], "active_sessions")
            self.assertEqual(body["applied"]["wake_words"]["appliesTo"], "new_sessions")
            self.assertEqual(body["settings"]["max_sessions"], 3)

    def test_two_websocket_clients_get_isolated_transcripts(self):
        settings = ServerSettings(
            model_warmup=False,
            realtime_processing_pause=0.0,
            realtime_min_audio_seconds=0.01,
            min_length_of_recording=0.0,
            post_speech_silence_duration=60.0,
            vad_energy_threshold=1.0,
            webrtc_sensitivity=99,
            max_sessions=2,
        )
        app = create_app(settings, scheduler_factory=AutoScheduler, recorder_factory=FakeRecorder)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe") as first:
                with client.websocket_connect("/ws/transcribe") as second:
                    first_hello = first.receive_json()
                    second_hello = second.receive_json()
                    self.assertEqual(first_hello["type"], "hello")
                    self.assertEqual(second_hello["type"], "hello")
                    first_session = first_hello["sessionId"]
                    second_session = second_hello["sessionId"]
                    self.assertNotEqual(first_session, second_session)

                    first.send_text('{"type":"start"}')
                    second.send_text('{"type":"start"}')
                    first.send_bytes(encode_audio_packet(
                        {"sampleRate": 16000, "channels": 1, "format": "pcm_s16le", "frames": 640},
                        np.full(640, 2000, dtype=np.int16).tobytes(),
                    ))
                    second.send_bytes(encode_audio_packet(
                        {"sampleRate": 16000, "channels": 1, "format": "pcm_s16le", "frames": 640},
                        np.full(640, 3000, dtype=np.int16).tobytes(),
                    ))
                    first.send_text('{"type":"stop"}')
                    second.send_text('{"type":"stop"}')

                    first_final = self._receive_type(first, "final")
                    second_final = self._receive_type(second, "final")

                    self.assertEqual(first_final["sessionId"], first_session)
                    self.assertEqual(second_final["sessionId"], second_session)
                    self.assertIn(first_session, first_final["text"])
                    self.assertIn(second_session, second_final["text"])
                    self.assertNotIn(second_session, first_final["text"])
                    self.assertNotIn(first_session, second_final["text"])

    def test_admission_limit_rejects_extra_websocket(self):
        settings = ServerSettings(model_warmup=False, max_sessions=1)
        app = create_app(settings, scheduler_factory=AutoScheduler, recorder_factory=FakeRecorder)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe") as first:
                self.assertEqual(first.receive_json()["type"], "hello")
                with client.websocket_connect("/ws/transcribe") as second:
                    error = second.receive_json()
                    self.assertEqual(error["type"], "error")
                    self.assertEqual(error["where"], "admission")

    def _receive_type(self, websocket, event_type, limit=20):
        for _ in range(limit):
            message = websocket.receive_json()
            if message.get("type") == event_type:
                return message
        self.fail(f"Did not receive {event_type!r} event")


if __name__ == "__main__":
    unittest.main()
