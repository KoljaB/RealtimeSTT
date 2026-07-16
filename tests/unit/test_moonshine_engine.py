import unittest
from unittest.mock import patch

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
)
from RealtimeSTT.transcription_engines.moonshine_engine import (
    MoonshineBackend,
    MoonshineEngine,
)
from tests.unit.test_additional_transcription_engines import (
    AudioVector,
    FakeBackend,
    FakeMoonshineModel,
    FakeMoonshineProcessor,
    FakeTorch,
)


class MoonshineModuleTests(unittest.TestCase):
    def tearDown(self):
        FakeMoonshineProcessor.instances.clear()
        FakeMoonshineModel.instances.clear()

    def test_factory_accepts_moonshine_streaming_hyphen_alias(self):
        config = TranscriptionEngineConfig(model="UsefulSensors/moonshine-streaming-medium")

        with patch(
            "RealtimeSTT.transcription_engines.moonshine_engine.MoonshineBackend",
            FakeBackend,
        ):
            engine = create_transcription_engine("moonshine-streaming", config)

        self.assertIsInstance(engine, MoonshineEngine)
        self.assertIsInstance(engine.backend, FakeBackend)

    def test_backend_initializes_from_direct_engine_module(self):
        config = TranscriptionEngineConfig(
            model="UsefulSensors/moonshine-streaming-medium",
            download_root="D:/hf",
            device="cuda:0",
            compute_type="float16",
            engine_options={"generate": {"max_new_tokens": 32}},
        )

        backend = MoonshineBackend(
            config,
            processor_cls=FakeMoonshineProcessor,
            model_cls=FakeMoonshineModel,
            torch_module=FakeTorch,
        )

        decoded = backend.transcribe(AudioVector([0.3]))

        self.assertEqual(decoded, "moonshine text")
        self.assertEqual(FakeMoonshineModel.instances[0].kwargs, {"cache_dir": "D:/hf"})
        self.assertEqual(
            FakeMoonshineModel.instances[0].to_calls,
            [(("cuda:0",), {}), ((), {"dtype": "torch.float16"})],
        )
        self.assertEqual(
            FakeMoonshineModel.instances[0].generate_calls[0],
            {"input_values": "features", "max_new_tokens": 32},
        )

    def test_engine_is_english_only_and_returns_result(self):
        backend = FakeBackend(output=[" moonshine transcript "])
        engine = MoonshineEngine(
            TranscriptionEngineConfig(model="UsefulSensors/moonshine-streaming-medium"),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]), language="en")

        self.assertEqual(result.text, "moonshine transcript")
        self.assertEqual(result.info.language, "en")

        with self.assertRaisesRegex(TranscriptionEngineError, "English"):
            engine.transcribe(AudioVector([0.0]), language="de")

    def test_missing_transformers_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="UsefulSensors/moonshine-streaming-medium")

        with patch(
            "RealtimeSTT.transcription_engines.hf_transformers_engines.import_module",
            side_effect=ModuleNotFoundError("transformers"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "transformers"):
                MoonshineBackend(config)


if __name__ == "__main__":
    unittest.main()
