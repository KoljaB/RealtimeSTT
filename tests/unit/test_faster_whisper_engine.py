import unittest
from unittest.mock import patch

import numpy as np

from RealtimeSTT.transcription_engines.base import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
)
from RealtimeSTT.transcription_engines.faster_whisper_engine import FasterWhisperEngine


class FasterWhisperEngineDependencyTests(unittest.TestCase):
    def test_missing_dependency_mentions_extra(self):
        config = TranscriptionEngineConfig(model="tiny")
        with patch(
            "RealtimeSTT.transcription_engines.faster_whisper_engine.import_module",
            side_effect=ModuleNotFoundError("No module named 'faster_whisper'"),
        ):
            with self.assertRaisesRegex(
                TranscriptionEngineError,
                r"RealtimeSTT\[faster-whisper\]",
            ):
                FasterWhisperEngine(config)

    def test_batched_vad_disabled_supplies_full_audio_clip_timestamps(self):
        class FeatureExtractor:
            sampling_rate = 16000
            chunk_length = 30

        class FakeWhisperModel:
            feature_extractor = FeatureExtractor()

            def __init__(self, **kwargs):
                pass

        class Segment:
            text = "ok"

        class Info:
            language = "en"
            language_probability = 1.0

        class FakeBatchedPipeline:
            last_kwargs = None

            def __init__(self, model):
                self.model = model

            def transcribe(self, audio, **kwargs):
                type(self).last_kwargs = kwargs
                return [Segment()], Info()

        class FakeFasterWhisper:
            WhisperModel = FakeWhisperModel
            BatchedInferencePipeline = FakeBatchedPipeline

        config = TranscriptionEngineConfig(
            model="tiny",
            batch_size=16,
            vad_filter=False,
        )
        audio = np.zeros(65 * 16000, dtype=np.float32)

        with patch(
            "RealtimeSTT.transcription_engines.faster_whisper_engine.import_module",
            return_value=FakeFasterWhisper,
        ):
            engine = FasterWhisperEngine(config)
            engine.transcribe(audio)

        self.assertEqual(
            FakeBatchedPipeline.last_kwargs["clip_timestamps"],
            [
                {"start": 0.0, "end": 30.0},
                {"start": 30.0, "end": 60.0},
                {"start": 60.0, "end": 65.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
