import unittest
from unittest.mock import patch

import numpy as np

from RealtimeSTT.transcription_engines.base import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
)
from RealtimeSTT.transcription_engines.faster_whisper_engine import FasterWhisperEngine


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeInfo:
    language = "en"
    language_probability = 0.9


class FakeWhisperModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return [FakeSegment(" hello"), FakeSegment("world ")], FakeInfo()


class FakeAudio:
    size = 1


class FakeWhisperModule:
    loaded = []

    @classmethod
    def WhisperModel(cls, **kwargs):
        model = FakeWhisperModel(**kwargs)
        cls.loaded.append(model)
        return model


def make_engine(config):
    with patch(
        "RealtimeSTT.transcription_engines.faster_whisper_engine._load_faster_whisper",
        return_value=(FakeWhisperModule, None),
    ):
        return FasterWhisperEngine(config)


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


class FasterWhisperEngineOptionsTests(unittest.TestCase):
    def tearDown(self):
        FakeWhisperModule.loaded.clear()

    def test_defaults_unchanged_without_engine_options(self):
        engine = make_engine(
            TranscriptionEngineConfig(model="tiny", initial_prompt="domain words")
        )

        result = engine.transcribe(FakeAudio(), language="en")

        model = FakeWhisperModule.loaded[0]
        self.assertEqual(
            model.kwargs,
            {
                "model_size_or_path": "tiny",
                "device": "cpu",
                "compute_type": "default",
                "device_index": 0,
                "download_root": None,
            },
        )
        self.assertEqual(
            model.calls[0][1],
            {
                "language": "en",
                "beam_size": 5,
                "initial_prompt": "domain words",
                "suppress_tokens": None,
                "vad_filter": True,
            },
        )
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.info.language, "en")

    def test_model_options_merge_into_model_init(self):
        make_engine(
            TranscriptionEngineConfig(
                model="tiny",
                engine_options={"model": {"cpu_threads": 4, "compute_type": "int8"}},
            )
        )

        self.assertEqual(
            FakeWhisperModule.loaded[0].kwargs,
            {
                "model_size_or_path": "tiny",
                "device": "cpu",
                "compute_type": "int8",
                "device_index": 0,
                "download_root": None,
                "cpu_threads": 4,
            },
        )

    def test_transcribe_options_merge_and_override(self):
        engine = make_engine(
            TranscriptionEngineConfig(
                model="tiny",
                engine_options={
                    "transcribe": {"task": "translate", "beam_size": 3},
                },
            )
        )

        engine.transcribe(FakeAudio(), language="es")

        params = FakeWhisperModule.loaded[0].calls[0][1]
        self.assertEqual(params["task"], "translate")
        self.assertEqual(params["beam_size"], 3)
        self.assertEqual(params["language"], "es")
        self.assertTrue(params["vad_filter"])


if __name__ == "__main__":
    unittest.main()
