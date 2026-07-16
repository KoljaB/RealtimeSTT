import unittest
from unittest.mock import patch

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.omnilingual_asr_engine import (
    DEFAULT_OMNILINGUAL_ASR_MODEL,
    OmnilingualASRBackend,
    OmnilingualASREngine,
)


class AudioVector:
    def __init__(self, values):
        self.values = [float(value) for value in values]
        self.size = len(self.values)

    def __abs__(self):
        return AudioVector(abs(value) for value in self.values)

    def __truediv__(self, value):
        return AudioVector(item / value for item in self.values)

    def __mul__(self, value):
        return AudioVector(item * value for item in self.values)

    def max(self):
        return max(self.values)


class FakeTorch:
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"


class FakeArray:
    ndim = 1

    def __init__(self, values):
        self.values = [float(value) for value in values]

    def __len__(self):
        return len(self.values)

    def astype(self, dtype):
        return self


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(values, dtype=None):
        return FakeArray(values)


class FakePipeline:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakePipeline.instances.append(self)

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return [" omnilingual transcript "]


class FakeBackend:
    def __init__(self, config=None, output=None):
        self.config = config
        self.output = output if output is not None else [" fake transcript "]
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return self.output

    def language_code(self, language=None):
        return "eng_Latn" if language == "en" else language


class ModelNotKnownError(Exception):
    pass


class UnknownModelPipeline:
    def __init__(self, **kwargs):
        raise ModelNotKnownError(kwargs.get("model_card"))


class OmnilingualASREngineTests(unittest.TestCase):
    def tearDown(self):
        FakePipeline.instances.clear()

    def test_supported_engines_include_omnilingual_aliases(self):
        engines = get_supported_transcription_engines()

        for name in (
            "omnilingual_asr",
            "omnilingual",
            "meta_omnilingual_asr",
            "omni_asr",
        ):
            self.assertIn(name, engines)

    def test_default_model_prefers_validated_public_v2_card(self):
        self.assertEqual(DEFAULT_OMNILINGUAL_ASR_MODEL, "omniASR_CTC_1B_v2")

    def test_factory_creates_omnilingual_engine_with_mocked_backend(self):
        for engine_name in (
            "omnilingual_asr",
            "omnilingual-asr",
            "omnilingual",
            "meta-omnilingual-asr",
            "omni-asr",
        ):
            with self.subTest(engine=engine_name):
                with patch(
                    "RealtimeSTT.transcription_engines.omnilingual_asr_engine."
                    "OmnilingualASRBackend",
                    FakeBackend,
                ):
                    engine = create_transcription_engine(
                        engine_name,
                        TranscriptionEngineConfig(model="omniASR_CTC_300M_v2"),
                    )
                self.assertIsInstance(engine, OmnilingualASREngine)
                self.assertIsInstance(engine.backend, FakeBackend)

    def test_backend_lazily_initializes_default_ctc_pipeline(self):
        config = TranscriptionEngineConfig(
            model="tiny",
            device="cuda",
            gpu_device_index=0,
            batch_size=0,
        )
        backend = OmnilingualASRBackend(
            config,
            pipeline_cls=FakePipeline,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
        )

        self.assertIsNone(backend.pipeline)
        output = backend.transcribe(AudioVector([0.1, -0.2]), language="en")

        self.assertEqual(output, [" omnilingual transcript "])
        self.assertEqual(
            FakePipeline.instances[0].kwargs,
            {
                "model_card": DEFAULT_OMNILINGUAL_ASR_MODEL,
                "device": "cuda:0",
                "dtype": "torch.float16",
            },
        )
        audio, params = FakePipeline.instances[0].calls[0]
        self.assertEqual(params, {"batch_size": 1})
        self.assertIsInstance(audio[0], dict)
        self.assertEqual(audio[0]["sample_rate"], 16000)
        self.assertEqual(audio[0]["waveform"].values, [0.1, -0.2])

    def test_backend_passes_language_for_llm_models(self):
        config = TranscriptionEngineConfig(
            model="omniASR_LLM_1B_v2",
            device="cuda",
            compute_type="bfloat16",
            engine_options={"batch_size": 2},
        )
        backend = OmnilingualASRBackend(
            config,
            pipeline_cls=FakePipeline,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
        )

        backend.transcribe(AudioVector([0.0]), language="en")

        self.assertEqual(FakePipeline.instances[0].kwargs["dtype"], "torch.bfloat16")
        self.assertEqual(
            FakePipeline.instances[0].calls[0][1],
            {"batch_size": 2, "lang": ["eng_Latn"]},
        )

    def test_backend_omits_language_for_ctc_models(self):
        config = TranscriptionEngineConfig(
            model="omniASR_CTC_1B_v2",
            engine_options={"transcribe": {"lang": ["eng_Latn"]}},
        )
        backend = OmnilingualASRBackend(
            config,
            pipeline_cls=FakePipeline,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
        )

        backend.transcribe(AudioVector([0.0]), language="en")

        self.assertEqual(FakePipeline.instances[0].calls[0][1], {"batch_size": 1})

    def test_engine_normalizes_audio_and_returns_text(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(
            model="omniASR_CTC_300M_v2",
            normalize_audio=True,
        )
        engine = OmnilingualASREngine(config, backend=backend)

        result = engine.transcribe(AudioVector([0.0, -2.0, 1.0]), language="en")

        self.assertEqual(backend.calls[0][0].values, [0.0, -0.95, 0.475])
        self.assertEqual(result.text, "fake transcript")
        self.assertEqual(result.info.language, "eng_Latn")
        self.assertEqual(result.info.language_probability, 1.0)

    def test_backend_uses_predecoded_dict_without_raw_numpy(self):
        config = TranscriptionEngineConfig(model="omniASR_CTC_300M_v2")
        backend = OmnilingualASRBackend(
            config,
            pipeline_cls=FakePipeline,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
        )
        decoded = {"waveform": FakeArray([0.0]), "sample_rate": 16000}

        backend.transcribe(decoded)

        self.assertIs(FakePipeline.instances[0].calls[0][0][0], decoded)

    def test_backend_rejects_overlong_in_memory_audio(self):
        config = TranscriptionEngineConfig(
            model="omniASR_CTC_300M_v2",
            engine_options={"sample_rate": 2, "max_audio_seconds": 1},
        )
        backend = OmnilingualASRBackend(
            config,
            pipeline_cls=FakePipeline,
            torch_module=FakeTorch,
            numpy_module=FakeNumpy,
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "shorter than 40 seconds"):
            backend.transcribe(AudioVector([0.0, 0.1]))

    def test_missing_dependency_reports_wsl_linux_requirement(self):
        config = TranscriptionEngineConfig(model="omniASR_CTC_300M_v2")

        with patch(
            "RealtimeSTT.transcription_engines.omnilingual_asr_engine.import_module",
            side_effect=ModuleNotFoundError("omnilingual_asr"),
        ):
            backend = OmnilingualASRBackend(config)
            with self.assertRaisesRegex(TranscriptionEngineError, "WSL2"):
                backend.transcribe(AudioVector([0.0]))

    def test_pipeline_import_oserror_reports_torch_torchaudio_guidance(self):
        config = TranscriptionEngineConfig(model="omniASR_CTC_300M_v2")

        with patch(
            "RealtimeSTT.transcription_engines.omnilingual_asr_engine.import_module",
            side_effect=OSError("libcudart.so.13"),
        ):
            backend = OmnilingualASRBackend(config, torch_module=FakeTorch)
            with self.assertRaisesRegex(
                TranscriptionEngineError,
                "matching torch and torchaudio",
            ):
                backend.transcribe(AudioVector([0.0]))

    def test_model_not_known_reports_dependency_blocker_without_fallback(self):
        config = TranscriptionEngineConfig(model="omniASR_CTC_1B_v2")
        backend = OmnilingualASRBackend(
            config,
            pipeline_cls=UnknownModelPipeline,
            torch_module=FakeTorch,
        )

        with self.assertRaises(TranscriptionEngineError) as caught:
            backend.transcribe({"waveform": FakeArray([0.0]), "sample_rate": 16000})

        message = str(caught.exception)
        self.assertIn("omniASR_CTC_1B_v2", message)
        self.assertIn("instead of silently falling back", message)
        self.assertIn("omnilingual-asr>=0.2.0", message)


if __name__ == "__main__":
    unittest.main()
