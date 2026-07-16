import unittest
from unittest.mock import patch

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
)
from RealtimeSTT.transcription_engines.granite_speech_engine import (
    GraniteSpeechBackend,
    GraniteSpeechEngine,
)
from tests.unit.test_additional_transcription_engines import (
    AudioVector,
    FakeBackend,
    FakeGraniteModel,
    FakeGraniteProcessor,
    FakeInputIds,
    FakeTensor,
    FakeTorch,
)


class GraniteSpeechModuleTests(unittest.TestCase):
    def tearDown(self):
        FakeGraniteProcessor.instances.clear()
        FakeGraniteModel.instances.clear()

    def test_factory_accepts_granite_speech_hyphen_alias(self):
        config = TranscriptionEngineConfig(model="ibm-granite/granite-speech-4.1-2b")

        with patch(
            "RealtimeSTT.transcription_engines.granite_speech_engine.GraniteSpeechBackend",
            FakeBackend,
        ):
            engine = create_transcription_engine("granite-speech", config)

        self.assertIsInstance(engine, GraniteSpeechEngine)
        self.assertIsInstance(engine.backend, FakeBackend)

    def test_backend_initializes_from_direct_engine_module(self):
        config = TranscriptionEngineConfig(
            model="ibm-granite/granite-speech-4.1-2b",
            download_root="D:/hf",
            device="cuda:0",
            compute_type="bfloat16",
            beam_size=4,
        )

        backend = GraniteSpeechBackend(
            config,
            processor_cls=FakeGraniteProcessor,
            model_cls=FakeGraniteModel,
            torch_module=FakeTorch,
        )

        decoded = backend.transcribe(AudioVector([0.2]), prompt="<|audio|>transcribe")

        self.assertEqual(decoded, ["granite text"])
        self.assertEqual(FakeGraniteProcessor.instances[0].kwargs, {"cache_dir": "D:/hf"})
        self.assertEqual(FakeGraniteModel.instances[0].kwargs["device_map"], "cuda:0")
        self.assertEqual(FakeGraniteModel.instances[0].kwargs["torch_dtype"], "torch.bfloat16")
        generate_call = FakeGraniteModel.instances[0].generate_calls[0]
        self.assertEqual(generate_call["num_beams"], 4)
        self.assertIsInstance(generate_call["input_ids"], FakeInputIds)
        self.assertIsInstance(generate_call["audio_values"], FakeTensor)

    def test_engine_builds_prompt_and_rejects_token_prompt(self):
        backend = FakeBackend(output=[" granite transcript "])
        engine = GraniteSpeechEngine(
            TranscriptionEngineConfig(
                model="ibm-granite/granite-speech-4.1-2b",
                initial_prompt="domain words",
                engine_options={"include_language_in_prompt": True},
            ),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]), language="de")

        self.assertEqual(result.text, "granite transcript")
        self.assertIn("Language: de.", backend.calls[0][1]["prompt"])
        self.assertIn("Context: domain words", backend.calls[0][1]["prompt"])

        bad_engine = GraniteSpeechEngine(
            TranscriptionEngineConfig(
                model="ibm-granite/granite-speech-4.1-2b",
                initial_prompt=[1, 2],
            ),
            backend=FakeBackend(),
        )
        with self.assertRaisesRegex(TranscriptionEngineError, "string initial_prompt"):
            bad_engine.transcribe(AudioVector([0.0]))

    def test_missing_transformers_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="ibm-granite/granite-speech-4.1-2b")

        with patch(
            "RealtimeSTT.transcription_engines.hf_transformers_engines.import_module",
            side_effect=ModuleNotFoundError("transformers"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "transformers"):
                GraniteSpeechBackend(config)


if __name__ == "__main__":
    unittest.main()
