import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

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


if __name__ == "__main__":
    unittest.main()
