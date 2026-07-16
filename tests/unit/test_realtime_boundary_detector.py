import unittest

import numpy as np

from RealtimeSTT.core.realtime_boundary_detector import RealtimeSpeechBoundaryDetector


class RealtimeSpeechBoundaryDetectorTest(unittest.TestCase):
    def test_detects_clear_energy_valley_between_voiced_regions(self):
        sample_rate = 16000
        detector = RealtimeSpeechBoundaryDetector(
            sample_rate=sample_rate,
            sensitivity=0.7,
            min_boundary_interval_ms=90,
        )

        first_voiced = self._tone(sample_rate, 0.18, amplitude=0.28)
        valley = self._tone(sample_rate, 0.04, amplitude=0.025)
        second_voiced = self._tone(sample_rate, 0.18, amplitude=0.28)
        audio = np.concatenate((first_voiced, valley, second_voiced))

        events = []
        chunk_size = 512
        for offset in range(0, audio.size, chunk_size):
            result = detector.process_samples(audio[offset:offset + chunk_size])
            events.extend(result.events)

        self.assertTrue(events)

        first_event_time = events[0].boundary_time_seconds
        self.assertGreater(first_event_time, 0.15)
        self.assertLess(first_event_time, 0.25)

    def test_silence_does_not_emit_boundaries(self):
        detector = RealtimeSpeechBoundaryDetector(sample_rate=16000)
        silence = np.zeros(16000, dtype=np.float32)

        events = []
        chunk_size = 512
        for offset in range(0, silence.size, chunk_size):
            result = detector.process_samples(silence[offset:offset + chunk_size])
            events.extend(result.events)

        self.assertEqual(events, [])

    def test_low_level_noise_does_not_emit_boundaries(self):
        detector = RealtimeSpeechBoundaryDetector(sample_rate=16000)
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.006, 16000).astype(np.float32)

        events = []
        chunk_size = 512
        for offset in range(0, noise.size, chunk_size):
            result = detector.process_samples(noise[offset:offset + chunk_size])
            events.extend(result.events)

        self.assertEqual(events, [])

    def _tone(self, sample_rate, duration_seconds, amplitude):
        t = np.arange(int(sample_rate * duration_seconds), dtype=np.float32) / sample_rate
        return (amplitude * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
