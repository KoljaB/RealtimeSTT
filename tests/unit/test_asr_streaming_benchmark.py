import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from tools.benchmarks import benchmark_asr_ab as ab_benchmark
from tools.benchmarks import benchmark_asr_streaming as benchmark


class StreamingBenchmarkUtilityTests(unittest.TestCase):
    def test_websocket_url_preserves_explicit_production_alias(self):
        self.assertEqual(
            benchmark._websocket_url(
                "http://127.0.0.1:8010/v1/audio/transcriptions/stream"
            ),
            "ws://127.0.0.1:8010/v1/audio/transcriptions/stream",
        )

    def test_agenttalk_display_language_names_map_to_supported_codes(self):
        expected = {
            "Auto": "auto",
            "English": "en",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
        }
        for value, language in expected.items():
            self.assertEqual(benchmark.map_language(value), language)

    def test_manifest_supports_samples_and_expected_detected_language(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "sample.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\x00\x00" * 160)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "file": "sample.wav",
                                "text": "hello",
                                "requested_language": "Auto",
                                "expected_detected_language": "English",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            clips = benchmark.load_manifest(manifest)
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0].expected_language, "en")
            self.assertEqual(clips[0].reference, "hello")

    def test_packet_shape_is_length_prefixed_and_metadata_is_exact(self):
        audio = b"\x01\x00\x02\x00"
        metadata = {
            "sampleRate": 16_000,
            "channels": 1,
            "format": "pcm_s16le",
            "frames": 2,
            "audioSequence": 0,
        }
        encoded = benchmark.encode_audio_packet(metadata, audio)
        length = struct.unpack("<I", encoded[:4])[0]
        self.assertEqual(length, len(encoded[4 : 4 + length]))
        decoded_metadata, decoded_audio = benchmark.decode_audio_packet(encoded)
        self.assertEqual(decoded_metadata, metadata)
        self.assertEqual(decoded_audio, audio)

    def test_sequence_validation_detects_gaps(self):
        self.assertTrue(benchmark.validate_audio_sequences([0, 1, 2])["valid"])
        invalid = benchmark.validate_audio_sequences([0, 2])
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["violations"][0]["expected"], 1)
        events = [{"eventSequence": 1}, {"eventSequence": 2}, {"eventSequence": 4}]
        self.assertFalse(benchmark.validate_event_sequences(events)["valid"])

    def test_terminal_contract_requires_exactly_one_ordered_pair_and_quiet_tail(self):
        self.assertEqual(
            benchmark.terminal_contract_errors(
                [{"type": "partial"}, {"type": "final"}, {"type": "completion"}]
            ),
            [],
        )
        duplicate = benchmark.terminal_contract_errors(
            [
                {"type": "final"},
                {"type": "final"},
                {"type": "completion"},
                {"type": "partial"},
            ]
        )
        self.assertIn("expected exactly one final event, received 2", duplicate)
        self.assertIn("server emitted events after completion", duplicate)
        reversed_pair = benchmark.terminal_contract_errors(
            [{"type": "completion"}, {"type": "final"}]
        )
        self.assertIn("completion did not follow the final event", reversed_pair)

    def test_pacing_uses_absolute_audio_clock_without_cumulative_drift(self):
        self.assertAlmostEqual(
            benchmark.pacing_delay(10.0, 640, 1.0, 10.015),
            0.025,
        )
        self.assertEqual(
            benchmark.pacing_delay(10.0, 1_280, 1.0, 10.090),
            0.0,
        )

    def test_long_run_repetitions_preserve_audio_and_make_ids_unique(self):
        clip = benchmark.StreamClip(
            clip_id="turn",
            expected_language="en",
            reference="",
            reference_kind="none",
            wav_path=Path("turn.wav"),
            pcm16=b"\x01\x00",
            audio_duration_s=1 / 16_000,
            source_sample_rate=16_000,
        )
        repeated = benchmark.repeat_clips([clip], 3)
        self.assertEqual(
            [item.clip_id for item in repeated],
            ["turn__repeat_001", "turn__repeat_002", "turn__repeat_003"],
        )
        self.assertTrue(all(item.pcm16 is clip.pcm16 for item in repeated))
        with self.assertRaisesRegex(ValueError, "at least one"):
            benchmark.repeat_clips([clip], 0)
        with self.assertRaisesRegex(ValueError, "concurrency must be at least one"):
            benchmark.run_benchmark(
                [clip],
                url="ws://127.0.0.1:1",
                concurrency=0,
            )

    def test_metrics_report_revisions_and_final_replacement(self):
        partials = ["hello", "hello wor", "hello world"]
        monotonic = benchmark.partial_prefix_monotonicity(partials)
        self.assertEqual(monotonic["prefix_monotonic_rate"], 1.0)
        semantics = benchmark.hypothesis_to_final_semantics(partials, "hello world today")
        self.assertTrue(semantics["replacement_required"])
        records = [
            {
                "expected_language": "English",
                "reference": "Hello world",
                "final_text": "hello world",
            },
            {
                "expected_language": "German",
                "reference": "Guten Tag",
                "final_text": "Guten",
            },
        ]
        metrics = benchmark.accuracy_by_language(records)
        self.assertEqual(metrics["overall"]["count"], 2)
        self.assertEqual(metrics["by_language"]["en"]["exact_match_rate"], 1.0)
        self.assertIn("de", metrics["by_language"])

    def test_48khz_resampling_uses_anti_aliasing(self):
        try:
            import numpy as np
            from scipy.signal import resample_poly
        except ImportError:
            self.skipTest("numpy/scipy are required for this resampling test")
        del resample_poly
        rate = 48_000
        seconds = 0.1
        sample_count = int(rate * seconds)
        t = np.arange(sample_count) / rate
        # 12 kHz is above the 8 kHz Nyquist limit of the 16 kHz output.  A
        # proper low-pass resampler must attenuate it materially.
        samples = (20_000 * np.sin(2 * np.pi * 12_000 * t)).round().astype("<i2")
        converted = benchmark.resample_pcm16(samples.tobytes(), rate)
        output = np.frombuffer(converted, dtype="<i2").astype(np.float64)
        self.assertEqual(len(output), round(sample_count * 16_000 / rate))
        self.assertLess(float(np.sqrt(np.mean(output * output))), 2_000.0)

    def test_default_report_redacts_paths_and_reconstructable_text(self):
        report = {
            "config": {"url": "wss://private.example/ws"},
            "records": [
                {
                    "clip_id": "speaker/private.wav",
                    "expected_language": "de",
                    "reference": "private reference words",
                    "reference_kind": "manifest",
                    "wav_path": "D:/private/speaker.wav",
                    "partial_texts": ["private partial words"],
                    "final_text": "private final words",
                    "errors": [],
                    "ok": True,
                    "hypothesis_to_final": {
                        "partial_count": 1,
                        "latest_partial": "private partial words",
                        "final_text": "private final words",
                        "replacement_required": True,
                        "latest_partial_matches_final": False,
                    },
                }
            ],
        }

        safe = benchmark.redact_report(report)
        serialized = json.dumps(safe)
        for secret in (
            "private.example",
            "speaker/private.wav",
            "D:/private/speaker.wav",
            "private reference words",
            "private partial words",
            "private final words",
        ):
            self.assertNotIn(secret, serialized)
        self.assertRegex(safe["records"][0]["clip_id"], r"^clip-[0-9a-f]{12}$")
        self.assertFalse(safe["sensitive_details_included"])

    def test_http_ab_default_report_is_publish_safe(self):
        report = {
            "manifest": "D:/private/manifest.json",
            "targets": [
                {
                    "base_url": "https://private.example",
                    "health_before": {"model": "private-model-path"},
                    "health_after": {},
                    "health_before_error": None,
                    "health_after_error": None,
                    "requests": [{"text": "private request words"}],
                    "accuracy_records": [
                        {
                            "clip_id": "private-speaker.wav",
                            "reference": "private reference words",
                            "hypothesis": "private hypothesis words",
                            "reference_normalized": "private reference words",
                            "hypothesis_normalized": "private hypothesis words",
                            "hypothesis_variants": {"private hypothesis words": 1},
                        }
                    ],
                }
            ],
        }

        safe = ab_benchmark.redact_report(report)
        serialized = json.dumps(safe)
        for secret in (
            "D:/private/manifest.json",
            "private.example",
            "private-model-path",
            "private request words",
            "private-speaker.wav",
            "private reference words",
            "private hypothesis words",
        ):
            self.assertNotIn(secret, serialized)
        self.assertFalse(safe["sensitive_details_included"])


if __name__ == "__main__":
    unittest.main()
