import unittest
from unittest.mock import patch

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
)
from RealtimeSTT.transcription_engines.cohere_transcribe_engine import (
    CohereTranscribeBackend,
    CohereTranscribeEngine,
)
from tests.unit.test_additional_transcription_engines import (
    AudioVector,
    FakeBackend,
    FakeCohereModel,
    FakeCohereProcessor,
    FakeTorch,
)


class CohereTranscribeModuleTests(unittest.TestCase):
    def tearDown(self):
        FakeCohereProcessor.instances.clear()
        FakeCohereModel.instances.clear()

    def test_factory_accepts_cohere_transcribe_hyphen_alias(self):
        config = TranscriptionEngineConfig(model="CohereLabs/cohere-transcribe-03-2026")

        with patch(
            "RealtimeSTT.transcription_engines.cohere_transcribe_engine.CohereTranscribeBackend",
            FakeBackend,
        ):
            engine = create_transcription_engine("cohere-transcribe", config)

        self.assertIsInstance(engine, CohereTranscribeEngine)
        self.assertIsInstance(engine.backend, FakeBackend)

    def test_backend_initializes_from_direct_engine_module(self):
        config = TranscriptionEngineConfig(
            model="CohereLabs/cohere-transcribe-03-2026",
            download_root="D:/hf",
            device="cuda",
            compute_type="float16",
            engine_options={
                "processor": {"trust_remote_code": True},
                "model": {"low_cpu_mem_usage": True},
            },
        )

        backend = CohereTranscribeBackend(
            config,
            processor_cls=FakeCohereProcessor,
            model_cls=FakeCohereModel,
            torch_module=FakeTorch,
        )

        decoded = backend.transcribe(AudioVector([0.1]), language="de")

        self.assertEqual(decoded, ["cohere text"])
        self.assertEqual(
            FakeCohereProcessor.instances[0].kwargs,
            {"trust_remote_code": True, "cache_dir": "D:/hf"},
        )
        self.assertEqual(
            FakeCohereModel.instances[0].kwargs,
            {
                "low_cpu_mem_usage": True,
                "cache_dir": "D:/hf",
                "device_map": "auto",
                "torch_dtype": "torch.float16",
            },
        )
        self.assertEqual(
            FakeCohereProcessor.instances[0].calls[0][1]["language"],
            "de",
        )

    def test_engine_requires_language_and_decodes(self):
        backend = FakeBackend(output=[" cohere transcript "])
        engine = CohereTranscribeEngine(
            TranscriptionEngineConfig(
                model="CohereLabs/cohere-transcribe-03-2026",
                engine_options={"language": "en"},
            ),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]))

        self.assertEqual(result.text, "cohere transcript")
        self.assertEqual(result.info.language, "en")

        no_language_engine = CohereTranscribeEngine(
            TranscriptionEngineConfig(model="CohereLabs/cohere-transcribe-03-2026"),
            backend=FakeBackend(output=["unused"]),
        )
        with self.assertRaisesRegex(TranscriptionEngineError, "requires a language code"):
            no_language_engine.transcribe(AudioVector([0.0]))

    def test_missing_transformers_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="CohereLabs/cohere-transcribe-03-2026")

        with patch(
            "RealtimeSTT.transcription_engines.hf_transformers_engines.import_module",
            side_effect=ModuleNotFoundError("transformers"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "transformers"):
                CohereTranscribeBackend(config)


if __name__ == "__main__":
    unittest.main()
