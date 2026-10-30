import os
import unittest
from unittest.mock import patch

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.funasr_engine import (
    DEFAULT_FUNASR_MODEL,
    FunASRBackend,
    FunASREngine,
    decode_funasr_result,
)
from tests.unit import test_additional_transcription_engines as audio_fixtures
from tests.unit.test_additional_transcription_engines import AudioVector


class FakeAutoModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeAutoModel.instances.append(self)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|> hello world ",
            }
        ]


class FakeFunASRModule:
    AutoModel = FakeAutoModel


class FakeBackend:
    def __init__(self, config=None, output=None):
        self.config = config
        self.output = output if output is not None else [
            {
                "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|> mocked funasr ",
            }
        ]
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return self.output


class FunASRFactoryTests(unittest.TestCase):
    def test_supported_engines_include_funasr_aliases(self):
        engines = get_supported_transcription_engines()

        self.assertIn("funasr", engines)
        self.assertIn("fun_asr", engines)

    def test_factory_creates_funasr_with_mocked_backend(self):
        config = TranscriptionEngineConfig(model="iic/SenseVoiceSmall")

        with patch(
            "RealtimeSTT.transcription_engines.funasr_engine.FunASRBackend",
            FakeBackend,
        ):
            engine = create_transcription_engine("fun-asr", config)

        self.assertIsInstance(engine, FunASREngine)
        self.assertIsInstance(engine.backend, FakeBackend)
        self.assertIs(engine.backend.config, config)


class FunASRBackendTests(unittest.TestCase):
    def tearDown(self):
        FakeAutoModel.instances.clear()

    def test_initializes_auto_model_with_realtimestt_config(self):
        config = TranscriptionEngineConfig(
            model="tiny",
            download_root="D:/models/funasr",
            device="cuda",
            gpu_device_index=1,
            beam_size=3,
            batch_size=4,
            engine_options={
                "hub": "ms",
                "vad_filter": True,
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "generate": {"use_itn": True},
                "batch_size_s": 60,
            },
        )

        with patch.dict(os.environ, {}, clear=True):
            backend = FunASRBackend(config, funasr_module=FakeFunASRModule)

            self.assertEqual(
                os.environ["MODELSCOPE_CACHE"],
                os.path.normpath("D:/models/funasr"),
            )
            self.assertEqual(
                os.environ["HF_HOME"],
                os.path.normpath("D:/models/funasr"),
            )

        self.assertEqual(len(FakeAutoModel.instances), 1)
        model = FakeAutoModel.instances[0]
        self.assertEqual(
            model.kwargs,
            {
                "model": DEFAULT_FUNASR_MODEL,
                "device": "cuda:1",
                "batch_size": 4,
                "beam_size": 3,
                "hub": "ms",
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "disable_update": True,
                "disable_pbar": True,
            },
        )

        audio = object()
        backend.transcribe(audio, language="en")

        self.assertEqual(
            model.calls[0],
            {
                "input": audio,
                "use_itn": True,
                "batch_size_s": 60,
                "language": "en",
            },
        )

    def test_explicit_model_name_and_auto_model_options_are_preserved(self):
        config = TranscriptionEngineConfig(
            model="custom",
            device="cpu",
            engine_options={
                "model_name": "FunAudioLLM/Fun-ASR-Nano-2512",
                "auto_model": {
                    "trust_remote_code": True,
                    "remote_code": "./model.py",
                },
                "hub": "hf",
            },
        )

        FunASRBackend(config, funasr_module=FakeFunASRModule)

        self.assertEqual(
            FakeAutoModel.instances[0].kwargs["model"],
            "FunAudioLLM/Fun-ASR-Nano-2512",
        )
        self.assertTrue(FakeAutoModel.instances[0].kwargs["trust_remote_code"])
        self.assertEqual(FakeAutoModel.instances[0].kwargs["remote_code"], "./model.py")
        self.assertEqual(FakeAutoModel.instances[0].kwargs["hub"], "hf")

    def test_vad_filter_false_disables_vad_model(self):
        config = TranscriptionEngineConfig(
            model="iic/SenseVoiceSmall",
            engine_options={
                "vad_filter": False,
                "vad_model": "fsmn-vad",
            },
        )

        FunASRBackend(config, funasr_module=FakeFunASRModule)

        self.assertNotIn("vad_model", FakeAutoModel.instances[0].kwargs)

    def test_missing_dependency_mentions_extra(self):
        config = TranscriptionEngineConfig(model="iic/SenseVoiceSmall")

        with patch(
            "RealtimeSTT.transcription_engines.funasr_engine.import_module",
            side_effect=ModuleNotFoundError("No module named 'funasr'"),
        ):
            with self.assertRaisesRegex(
                TranscriptionEngineError,
                r"RealtimeSTT\[funasr\]",
            ):
                FunASRBackend(config)

    def test_invalid_structured_options_raise_clear_error(self):
        config = TranscriptionEngineConfig(
            model="iic/SenseVoiceSmall",
            engine_options={"model": "bad"},
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "option 'model'.*dict"):
            FunASRBackend(config, funasr_module=FakeFunASRModule)


class FunASREngineContractTests(unittest.TestCase):
    def test_transcribe_normalizes_audio_and_maps_result(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(
            model="iic/SenseVoiceSmall",
            initial_prompt="product names",
            normalize_audio=True,
            engine_options={"language": "auto"},
        )
        engine = FunASREngine(config, backend=backend)

        result = engine.transcribe(AudioVector([0.0, 2.0, -1.0]))

        backend_audio, params = backend.calls[0]
        self.assertEqual(backend_audio.values, [0.0, 0.95, -0.475])
        self.assertEqual(params, {"language": "auto", "hotword": "product names"})
        self.assertEqual(
            result.text,
            "<|en|><|NEUTRAL|><|Speech|><|woitn|> mocked funasr",
        )
        self.assertIsNone(result.info.language)
        self.assertEqual(result.info.language_probability, 0.0)

    def test_explicit_language_overrides_detected_language(self):
        backend = FakeBackend()
        engine = FunASREngine(
            TranscriptionEngineConfig(model="iic/SenseVoiceSmall"),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]), language="de")

        self.assertEqual(backend.calls[0][1], {"language": "de"})
        self.assertEqual(result.info.language, "de")

    def test_returns_raw_model_text(self):
        backend = FakeBackend()
        engine = FunASREngine(
            TranscriptionEngineConfig(model="iic/SenseVoiceSmall"),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]))

        self.assertEqual(
            result.text,
            "<|en|><|NEUTRAL|><|Speech|><|woitn|> mocked funasr",
        )
        self.assertIsNone(result.info.language)

    def test_rejects_token_prompt(self):
        engine = FunASREngine(
            TranscriptionEngineConfig(
                model="iic/SenseVoiceSmall",
                initial_prompt=[1, 2, 3],
            ),
            backend=FakeBackend(),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "string initial_prompt"):
            engine.transcribe(AudioVector([0.0]))


class FunASRResultDecodingTests(unittest.TestCase):
    def test_decodes_multiple_results_and_language_fields(self):
        text, language = decode_funasr_result(
            [
                {"text": "<|zh|><|Speech|> ni hao", "language": "Chinese"},
                {"text": "<|zh|> shi jie"},
            ]
        )

        self.assertEqual(text, "<|zh|><|Speech|> ni hao <|zh|> shi jie")
        self.assertEqual(language, "Chinese")

    def test_does_not_infer_language_from_text_tags(self):
        text, language = decode_funasr_result(
            [{"text": "<|SAD|><|Speech|><|woitn|> no language here"}]
        )

        self.assertEqual(text, "<|SAD|><|Speech|><|woitn|> no language here")
        self.assertIsNone(language)

        text, language = decode_funasr_result(
            [{"text": "<|yue|><|Speech|><|woitn|> nei hou"}]
        )

        self.assertEqual(text, "<|yue|><|Speech|><|woitn|> nei hou")
        self.assertIsNone(language)


class FunASRGoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("REALTIMESTT_RUN_FUNASR") != "1":
            self.skipTest("Set REALTIMESTT_RUN_FUNASR=1 to run the FunASR smoke test")
        if audio_fixtures.np is None:
            self.skipTest("NumPy is required for the FunASR smoke test")

    def test_transcribes_fixture_with_real_funasr_backend(self):
        audio, expected = audio_fixtures.read_fixture_audio()
        language = os.environ.get("REALTIMESTT_FUNASR_LANGUAGE", "auto")
        engine = FunASREngine(
            TranscriptionEngineConfig(
                model=os.environ.get("REALTIMESTT_FUNASR_MODEL", DEFAULT_FUNASR_MODEL),
                device=os.environ.get("REALTIMESTT_FUNASR_DEVICE", "cpu"),
                download_root=os.environ.get("REALTIMESTT_FUNASR_MODEL_DIR"),
                engine_options={
                    "language": language,
                    "use_itn": True,
                },
            )
        )

        result = engine.transcribe(audio, language=language)
        actual = audio_fixtures.normalize_transcript(result.text)

        print("\n[RealtimeSTT test] funasr expected: %s" % expected)
        print("[RealtimeSTT test] funasr actual:   %s" % actual)

        self.assertTrue(actual)
        self.assertIn(" ".join(expected.split()[:2]), actual)


if __name__ == "__main__":
    unittest.main()
