"""Small, deterministic acceptance checks for the versioned production API.

These tests deliberately stop below the socket layer.  They exercise the
shared packet conversion used by the production service and the production
WebSocket state machine with lightweight fakes, so they do not need a model,
network listener, or generated audio fixture.
"""

import asyncio
import types
import unittest
from unittest import mock

import numpy as np

from RealtimeSTT_server import production_server as production
from example_fastapi_server import server as reference_server


try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - exercised by dependency-minimal installs
    resample_poly = None


class _Scheduler:
    """Minimal scheduler surface needed by ``RealtimeSTTService`` in unit tests."""

    def __init__(self, settings, result_callback, drop_callback=None, error_callback=None):
        self.settings = settings
        self.cancelled_sessions = []

    def cancel_session(self, session_id):
        self.cancelled_sessions.append(session_id)

    def snapshot(self):
        return {"workers": {}, "queues": {}}

    def healthy(self):
        return True


class _Session:
    def __init__(self):
        self.settings = types.SimpleNamespace(language="en")
        self.recorder = types.SimpleNamespace(realtime_transcription_executor=None)
        self.started = 0
        self.stopped = 0
        self.cleared = 0
        self.ingested = []

    def start_streaming(self):
        self.started += 1

    def stop_streaming(self):
        self.stopped += 1

    def ingest_audio_packet(self, packet):
        self.ingested.append(packet)
        return True, None

    def clear(self):
        self.cleared += 1

    def snapshot(self):
        return {
            "finalSubmitted": 0,
            "finalCompleted": 0,
            "finalRejected": 0,
            "recording": False,
        }


class ProductionServerAudioBoundaryTests(unittest.TestCase):
    @unittest.skipIf(resample_poly is None, "scipy is required for the anti-aliased boundary assertion")
    def test_48khz_pcm_is_resampled_once_with_scipy_polyphase_filter(self):
        sample_rate = 48_000
        samples = np.arange(sample_rate, dtype=np.float32)
        # Keep the expected result independent from the implementation under
        # test.  The high-frequency component also makes linear interpolation
        # observably different from an anti-aliased decimator.
        samples = (
            7_000.0 * np.sin(2 * np.pi * 4_000 * samples / sample_rate)
            + 7_000.0 * np.sin(2 * np.pi * 12_000 * samples / sample_rate)
        ).round().astype(np.int16)
        packet = production.decode_audio_packet(
            production.encode_audio_packet(
                {
                    "sampleRate": sample_rate,
                    "channels": 1,
                    "format": production.PCM_FORMAT,
                    "frames": int(samples.size),
                },
                samples.tobytes(),
            )
        )

        service = reference_server.RealtimeSTTService(
            reference_server.ServerSettings(),
            reference_server.ConnectionManager(),
            scheduler_factory=_Scheduler,
        )
        expected = resample_poly(samples.astype(np.float32), 1, 3)
        expected = np.clip(np.rint(expected), -32768, 32767).astype(np.int16)

        with mock.patch.object(
            reference_server,
            "resample_int16",
            wraps=reference_server.resample_int16,
        ) as boundary:
            actual = service.packet_to_server_samples(packet)

        self.assertEqual(boundary.call_count, 1)
        self.assertEqual(boundary.call_args.args[1:], (sample_rate, 16_000))
        self.assertEqual(actual.size, 16_000)
        np.testing.assert_array_equal(actual, expected)

    def test_canonical_turn_pcm_is_invariant_across_packet_boundaries(self):
        sample_rate = production.SERVER_SAMPLE_RATE
        timeline = np.arange(sample_rate // 2, dtype=np.float32)
        source = (
            7_000.0 * np.sin(2 * np.pi * 1_000 * timeline / sample_rate)
            + 4_000.0 * np.sin(2 * np.pi * 3_700 * timeline / sample_rate)
        ).round().astype(np.int16)

        for chunk_ms in (10, 20, 40, 64, 100):
            with self.subTest(chunk_ms=chunk_ms):
                protocol = production.ProductionSessionProtocol(
                    service=types.SimpleNamespace(),
                    manager=production.OrderedConnectionManager(),
                    session_id="session",
                    settings=production.ProductionServerSettings(model_warmup=False),
                )
                session = _Session()
                protocol.attach(session)
                asyncio.run(protocol.start({"turnId": f"turn-{chunk_ms}", "language": "en"}))
                chunk_frames = sample_rate * chunk_ms // 1_000
                for sequence, start in enumerate(range(0, source.size, chunk_frames)):
                    chunk = source[start : start + chunk_frames]
                    error = asyncio.run(
                        protocol.audio(
                            production.encode_audio_packet(
                                {
                                    "sampleRate": sample_rate,
                                    "channels": 1,
                                    "format": production.PCM_FORMAT,
                                    "frames": int(chunk.size),
                                    "audioSequence": sequence,
                                },
                                chunk.tobytes(),
                            )
                        )
                    )
                    self.assertIsNone(error)

                actual = np.frombuffer(bytes(protocol.turn.pcm_buffer), dtype=np.int16)
                np.testing.assert_array_equal(actual, source)

    def test_silence_and_noise_have_distinct_vad_decisions(self):
        settings = reference_server.ServerSettings(vad_energy_threshold=500.0)
        detector = reference_server.VoiceActivityDetector(settings)
        # Force the deterministic energy fallback even when webrtcvad is
        # installed in the test environment.
        detector.vad = None

        silence = np.zeros(320, dtype=np.int16)
        noise = np.random.default_rng(1234).integers(-2_000, 2_001, size=320, dtype=np.int16)

        self.assertFalse(detector.is_speech(silence))
        self.assertTrue(detector.is_speech(noise))


class ProductionServerProtocolAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.settings = production.ProductionServerSettings(
            max_turn_audio_seconds=1.0,
            max_audio_packet_bytes=64 * 1024,
            model_warmup=False,
        )
        self.manager = production.OrderedConnectionManager()
        self.session = _Session()
        self.protocol = production.ProductionSessionProtocol(
            service=types.SimpleNamespace(),
            manager=self.manager,
            session_id="session-1",
            settings=self.settings,
        )
        self.protocol.attach(self.session)

    @staticmethod
    def _run(awaitable):
        return asyncio.run(awaitable)

    @staticmethod
    def _packet(audio, *, sample_rate=16_000, frames=None, sequence=0):
        if frames is None:
            frames = len(audio) // 2
        return production.encode_audio_packet(
            {
                "sampleRate": sample_rate,
                "channels": 1,
                "format": production.PCM_FORMAT,
                "frames": frames,
                "audioSequence": sequence,
            },
            audio,
        )

    def _start(self, turn_id="turn-1"):
        result = self._run(self.protocol.start({"turnId": turn_id, "language": "en"}))
        self.assertEqual(result["type"], "started")

    def test_empty_and_odd_pcm_are_rejected_without_ingestion(self):
        self._start()

        empty = self._run(self.protocol.audio(self._packet(b"")))
        self.assertEqual(empty["error"]["code"], "invalid_audio")
        self.assertIn("must not be empty", empty["error"]["message"])

        odd = self._run(self.protocol.audio(self._packet(b"\x01", frames=1)))
        self.assertEqual(odd["error"]["code"], "invalid_audio")
        self.assertIn("whole samples", odd["error"]["message"])
        self.assertEqual(self.session.ingested, [])

    def test_invalid_sample_rate_and_duration_limit_are_rejected(self):
        self._start()

        invalid_rate = self._run(
            self.protocol.audio(self._packet(b"\x00\x00", sample_rate=22_050))
        )
        self.assertEqual(invalid_rate["error"]["code"], "invalid_audio")
        self.assertIn("not supported", invalid_rate["error"]["message"])

        oversized = self._run(
            self.protocol.audio(
                self._packet(
                    np.zeros(16_001, dtype=np.int16).tobytes(),
                    sample_rate=16_000,
                    sequence=0,
                )
            )
        )
        self.assertEqual(oversized["error"]["code"], "audio_duration_limit")
        self.assertEqual(self.session.ingested, [])

    def test_reset_and_cancel_clear_turn_sequence_and_backend_state(self):
        self._start("turn-reset")
        accepted = self._run(self.protocol.audio(self._packet(b"\x01\x00")))
        self.assertIsNone(accepted)
        self.assertEqual(self.manager._audio_sequences["session-1"], 0)

        reset = self._run(self.protocol.reset())
        self.assertEqual(reset["previousTurnId"], "turn-reset")
        self.assertIsNone(self.protocol.turn)
        self.assertIsNone(self.manager._turn_ids["session-1"])
        self.assertIsNone(self.manager._audio_sequences["session-1"])
        self.assertEqual(self.session.cleared, 1)

        self._start("turn-cancel")
        cancelled = self._run(self.protocol.cancel())
        self.assertEqual(cancelled["type"], "cancelled")
        self.assertEqual(cancelled["turnId"], "turn-cancel")
        self.assertIsNone(self.protocol.turn)
        self.assertIsNone(self.manager._turn_ids["session-1"])
        self.assertIsNone(self.manager._audio_sequences["session-1"])
        self.assertEqual(self.session.cleared, 2)

        after_cancel = self._run(self.protocol.audio(self._packet(b"\x00\x00")))
        self.assertEqual(after_cancel["error"]["code"], "turn_not_started")


if __name__ == "__main__":
    unittest.main()
