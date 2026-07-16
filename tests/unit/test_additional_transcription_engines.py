import os
import re
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.cohere_transcribe_engine import (
    CohereTranscribeBackend,
    CohereTranscribeEngine,
)
from RealtimeSTT.transcription_engines.granite_speech_engine import (
    GraniteSpeechBackend,
    GraniteSpeechEngine,
)
from RealtimeSTT.transcription_engines.moonshine_engine import (
    MoonshineBackend,
    MoonshineEngine,
)
from RealtimeSTT.transcription_engines.parakeet_engine import (
    ParakeetEngine,
    ParakeetNeMoBackend,
)
from RealtimeSTT.transcription_engines.qwen3_asr_engine import (
    Qwen3ASRBackend,
    Qwen3ASREngine,
)


AUDIO_DIR = Path(__file__).with_name("audio")


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


class FakeOutput:
    def __init__(self, text="transcript", language=None):
        self.text = text
        self.language = language


class FakeBackend:
    def __init__(self, config=None, output=None):
        self.config = config
        self.output = output if output is not None else [FakeOutput("mocked", "en")]
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return self.output


class FakeASRModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.to_calls = []
        self.eval_called = False
        FakeASRModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls(**kwargs)

    def transcribe(self, paths, **params):
        self.calls.append((paths, params))
        return [FakeOutput("parakeet text", "de")]

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeTorch:
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"

    @staticmethod
    def as_tensor(value):
        return FakeTensor(value)


class FakeTensor:
    def __init__(self, value, ndim=1):
        self.value = value
        self.ndim = ndim

    def unsqueeze(self, dim):
        return FakeTensor(self.value, ndim=2)


class FakeInputs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class FakeInputIds:
    shape = (1, 5)


class FakeTokenizer:
    def __init__(self):
        self.template_calls = []
        self.decode_calls = []

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
        self.template_calls.append((chat, tokenize, add_generation_prompt))
        return "templated prompt"

    def batch_decode(self, tokens, **kwargs):
        self.decode_calls.append((tokens, kwargs))
        return ["granite text"]


class FakeCohereProcessor:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.calls = []
        self.decode_calls = []
        FakeCohereProcessor.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return FakeInputs({"input_features": "features", "audio_chunk_index": [0]})

    def decode(self, outputs, **kwargs):
        self.decode_calls.append((outputs, kwargs))
        return ["cohere text"]


class FakeCohereModel:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.device = "cuda:0"
        self.dtype = "torch.float16"
        self.generate_calls = []
        FakeCohereModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return ["tokens"]


class FakeGraniteProcessor:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.calls = []
        FakeGraniteProcessor.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    def __call__(self, prompt, audio, **kwargs):
        self.calls.append((prompt, audio, kwargs))
        return FakeInputs({"input_ids": FakeInputIds(), "audio_values": audio})


class FakeGraniteModel:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.generate_calls = []
        FakeGraniteModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return ["granite tokens"]


class FakeFeatureExtractor:
    sampling_rate = 16000


class FakeMoonshineProcessor:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.feature_extractor = FakeFeatureExtractor()
        self.calls = []
        self.decode_calls = []
        FakeMoonshineProcessor.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return FakeInputs({"input_values": "features"})

    def decode(self, generated_ids, **kwargs):
        self.decode_calls.append((generated_ids, kwargs))
        return "moonshine text"


class FakeMoonshineModel:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.to_calls = []
        self.generate_calls = []
        FakeMoonshineModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return ["moonshine ids"]


class FakeQwenModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeOutput("qwen text", kwargs.get("language"))]


class FakeQwenFactory:
    pretrained_calls = []
    llm_calls = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.pretrained_calls.append((model_name, kwargs))
        return FakeQwenModel(model_name=model_name, **kwargs)

    @classmethod
    def LLM(cls, **kwargs):
        cls.llm_calls.append(kwargs)
        return FakeQwenModel(**kwargs)


def read_fixture_audio():
    wav_path = AUDIO_DIR / "LJ001-0002.wav"
    transcript_path = AUDIO_DIR / "LJ001-0002.txt"
    with wave.open(str(wav_path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    audio = resample_audio(audio, sample_rate, 16000)
    expected = normalize_transcript(transcript_path.read_text(encoding="utf-8"))
    return audio, expected


def resample_audio(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return audio
    target_length = int(len(audio) * target_rate / source_rate)
    source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def normalize_transcript(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class AdditionalEngineFactoryTests(unittest.TestCase):
    def test_supported_engines_include_new_model_families(self):
        engines = get_supported_transcription_engines()

        for name in (
            "parakeet",
            "nvidia_parakeet",
            "cohere_transcribe",
            "cohere",
            "granite_speech",
            "granite",
            "qwen3_asr",
            "qwen_asr",
            "moonshine",
            "moonshine_streaming",
        ):
            self.assertIn(name, engines)

    def test_factory_creates_new_engines_with_mocked_backends(self):
        cases = [
            ("parakeet", "ParakeetNeMoBackend", ParakeetEngine),
            ("nvidia-parakeet", "ParakeetNeMoBackend", ParakeetEngine),
            ("cohere_transcribe", "CohereTranscribeBackend", CohereTranscribeEngine),
            ("cohere-transcribe", "CohereTranscribeBackend", CohereTranscribeEngine),
            ("cohere", "CohereTranscribeBackend", CohereTranscribeEngine),
            ("granite_speech", "GraniteSpeechBackend", GraniteSpeechEngine),
            ("granite-speech", "GraniteSpeechBackend", GraniteSpeechEngine),
            ("granite", "GraniteSpeechBackend", GraniteSpeechEngine),
            ("qwen3_asr", "Qwen3ASRBackend", Qwen3ASREngine),
            ("qwen3-asr", "Qwen3ASRBackend", Qwen3ASREngine),
            ("qwen-asr", "Qwen3ASRBackend", Qwen3ASREngine),
            ("moonshine", "MoonshineBackend", MoonshineEngine),
            ("moonshine-streaming", "MoonshineBackend", MoonshineEngine),
        ]

        for engine_name, backend_name, engine_cls in cases:
            with self.subTest(engine=engine_name):
                patch_target = self._backend_patch_target(engine_name, backend_name)
                with patch(patch_target, FakeBackend):
                    engine = create_transcription_engine(
                        engine_name,
                        TranscriptionEngineConfig(model="model-id"),
                    )
                self.assertIsInstance(engine, engine_cls)
                self.assertIsInstance(engine.backend, FakeBackend)

    @staticmethod
    def _backend_patch_target(engine_name, backend_name):
        module = {
            "parakeet": "parakeet_engine",
            "nvidia-parakeet": "parakeet_engine",
            "cohere_transcribe": "cohere_transcribe_engine",
            "cohere-transcribe": "cohere_transcribe_engine",
            "cohere": "cohere_transcribe_engine",
            "granite_speech": "granite_speech_engine",
            "granite-speech": "granite_speech_engine",
            "granite": "granite_speech_engine",
            "qwen3_asr": "qwen3_asr_engine",
            "qwen3-asr": "qwen3_asr_engine",
            "qwen-asr": "qwen3_asr_engine",
            "moonshine": "moonshine_engine",
            "moonshine-streaming": "moonshine_engine",
        }[engine_name]
        return "RealtimeSTT.transcription_engines.%s.%s" % (module, backend_name)


class ParakeetEngineTests(unittest.TestCase):
    def tearDown(self):
        FakeASRModel.instances.clear()

    def test_backend_initializes_nemo_model_and_transcribes_file_path(self):
        config = TranscriptionEngineConfig(
            model="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda",
            engine_options={"model": {"map_location": "cpu"}},
        )
        backend = ParakeetNeMoBackend(config, asr_model_cls=FakeASRModel)

        output = backend.transcribe("sample.wav", batch_size=2)

        self.assertEqual(output[0].text, "parakeet text")
        self.assertEqual(
            FakeASRModel.instances[0].kwargs,
            {"model_name": "nvidia/parakeet-tdt-0.6b-v3", "map_location": "cpu"},
        )
        self.assertEqual(FakeASRModel.instances[0].to_calls, ["cuda"])
        self.assertTrue(FakeASRModel.instances[0].eval_called)
        self.assertEqual(
            FakeASRModel.instances[0].calls[0],
            (["sample.wav"], {"batch_size": 2}),
        )

    def test_missing_nemo_reports_wsl2_hint(self):
        config = TranscriptionEngineConfig(model="nvidia/parakeet-tdt-0.6b-v3")

        with patch(
            "RealtimeSTT.transcription_engines.parakeet_engine.import_module",
            side_effect=ModuleNotFoundError("nemo"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "WSL2"):
                ParakeetNeMoBackend(config)

    def test_engine_normalizes_audio_and_returns_language(self):
        backend = FakeBackend(output=[FakeOutput(" Parakeet transcript ", "de")])
        config = TranscriptionEngineConfig(
            model="nvidia/parakeet-tdt-0.6b-v3",
            batch_size=3,
            normalize_audio=True,
            engine_options={"timestamps": True},
        )
        engine = ParakeetEngine(config, backend=backend)

        result = engine.transcribe(AudioVector([0.0, -2.0, 1.0]))

        backend_audio, params = backend.calls[0]
        self.assertEqual(backend_audio.values, [0.0, -0.95, 0.475])
        self.assertEqual(params, {"batch_size": 3, "timestamps": True})
        self.assertEqual(result.text, "Parakeet transcript")
        self.assertEqual(result.info.language, "de")


class CohereTranscribeEngineTests(unittest.TestCase):
    def tearDown(self):
        FakeCohereProcessor.instances.clear()
        FakeCohereModel.instances.clear()

    def test_backend_initializes_transformers_model_and_decodes(self):
        config = TranscriptionEngineConfig(
            model="CohereLabs/cohere-transcribe-03-2026",
            download_root="D:/hf",
            device="cuda",
            compute_type="float16",
        )
        backend = CohereTranscribeBackend(
            config,
            processor_cls=FakeCohereProcessor,
            model_cls=FakeCohereModel,
            torch_module=FakeTorch,
        )

        decoded = backend.transcribe(AudioVector([0.1]), language="en")

        self.assertEqual(decoded, ["cohere text"])
        self.assertEqual(FakeCohereProcessor.instances[0].kwargs, {"cache_dir": "D:/hf"})
        self.assertEqual(
            FakeCohereModel.instances[0].kwargs,
            {
                "cache_dir": "D:/hf",
                "device_map": "auto",
                "torch_dtype": "torch.float16",
            },
        )
        self.assertEqual(
            FakeCohereProcessor.instances[0].calls[0][1],
            {"sampling_rate": 16000, "return_tensors": "pt", "language": "en"},
        )
        self.assertEqual(
            FakeCohereModel.instances[0].generate_calls[0],
            {
                "input_features": "features",
                "audio_chunk_index": [0],
                "max_new_tokens": 256,
            },
        )
        self.assertEqual(
            FakeCohereProcessor.instances[0].decode_calls[0][1],
            {
                "skip_special_tokens": True,
                "audio_chunk_index": [0],
                "language": "en",
            },
        )

    def test_missing_transformers_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="CohereLabs/cohere-transcribe-03-2026")

        with patch(
            "RealtimeSTT.transcription_engines.hf_transformers_engines.import_module",
            side_effect=ModuleNotFoundError("transformers"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "transformers"):
                CohereTranscribeBackend(config)

    def test_engine_requires_language(self):
        engine = CohereTranscribeEngine(
            TranscriptionEngineConfig(model="CohereLabs/cohere-transcribe-03-2026"),
            backend=FakeBackend(output=["unused"]),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "requires a language code"):
            engine.transcribe(AudioVector([0.0]))

    def test_engine_uses_language_from_options(self):
        backend = FakeBackend(output=[" cohere transcript "])
        engine = CohereTranscribeEngine(
            TranscriptionEngineConfig(
                model="CohereLabs/cohere-transcribe-03-2026",
                engine_options={"language": "de"},
            ),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]))

        self.assertEqual(backend.calls[0][1], {"language": "de"})
        self.assertEqual(result.text, "cohere transcript")
        self.assertEqual(result.info.language, "de")


class GraniteSpeechEngineTests(unittest.TestCase):
    def tearDown(self):
        FakeGraniteProcessor.instances.clear()
        FakeGraniteModel.instances.clear()

    def test_backend_initializes_and_decodes_generated_tokens(self):
        config = TranscriptionEngineConfig(
            model="ibm-granite/granite-speech-4.1-2b",
            download_root="D:/hf",
            device="cuda:0",
            compute_type="bfloat16",
            beam_size=2,
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
        self.assertEqual(
            FakeGraniteModel.instances[0].kwargs,
            {
                "cache_dir": "D:/hf",
                "device_map": "cuda:0",
                "torch_dtype": "torch.bfloat16",
            },
        )
        self.assertEqual(
            FakeGraniteProcessor.instances[0].calls[0][0],
            "templated prompt",
        )
        self.assertEqual(
            FakeGraniteModel.instances[0].generate_calls[0]["max_new_tokens"],
            200,
        )
        self.assertFalse(FakeGraniteModel.instances[0].generate_calls[0]["do_sample"])
        self.assertEqual(FakeGraniteModel.instances[0].generate_calls[0]["num_beams"], 2)
        self.assertIsInstance(
            FakeGraniteModel.instances[0].generate_calls[0]["input_ids"],
            FakeInputIds,
        )
        self.assertIsInstance(
            FakeGraniteModel.instances[0].generate_calls[0]["audio_values"],
            FakeTensor,
        )

    def test_engine_builds_prompt_with_initial_prompt(self):
        backend = FakeBackend(output=[" granite transcript "])
        config = TranscriptionEngineConfig(
            model="ibm-granite/granite-speech-4.1-2b",
            initial_prompt="product names",
            engine_options={"include_language_in_prompt": True},
        )
        engine = GraniteSpeechEngine(config, backend=backend)

        result = engine.transcribe(AudioVector([0.0]), language="de")

        prompt = backend.calls[0][1]["prompt"]
        self.assertIn("Language: de.", prompt)
        self.assertIn("Context: product names", prompt)
        self.assertEqual(result.text, "granite transcript")

    def test_engine_rejects_token_prompt(self):
        engine = GraniteSpeechEngine(
            TranscriptionEngineConfig(
                model="ibm-granite/granite-speech-4.1-2b",
                initial_prompt=[1, 2],
            ),
            backend=FakeBackend(),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "string initial_prompt"):
            engine.transcribe(AudioVector([0.0]))


class Qwen3ASREngineTests(unittest.TestCase):
    def tearDown(self):
        FakeQwenFactory.pretrained_calls.clear()
        FakeQwenFactory.llm_calls.clear()

    def test_backend_initializes_transformers_model(self):
        config = TranscriptionEngineConfig(
            model="Qwen/Qwen3-ASR-1.7B",
            download_root="D:/hf",
            device="cuda:0",
            compute_type="bfloat16",
        )

        backend = Qwen3ASRBackend(
            config,
            model_factory=FakeQwenFactory,
            torch_module=FakeTorch,
        )

        self.assertIsInstance(backend.model, FakeQwenModel)
        self.assertEqual(
            FakeQwenFactory.pretrained_calls[0],
            (
                "Qwen/Qwen3-ASR-1.7B",
                {
                    "cache_dir": "D:/hf",
                    "dtype": "torch.bfloat16",
                    "device_map": "cuda:0",
                },
            ),
        )

    def test_backend_wraps_in_memory_audio_with_sample_rate(self):
        audio = AudioVector([0.1])
        config = TranscriptionEngineConfig(
            model="Qwen/Qwen3-ASR-1.7B",
            engine_options={"sample_rate": 8000},
        )
        backend = Qwen3ASRBackend(
            config,
            model_factory=FakeQwenFactory,
            torch_module=FakeTorch,
        )

        backend.transcribe(audio, language="English")

        self.assertEqual(backend.model.calls[0]["audio"], (audio, 8000))
        self.assertEqual(backend.model.calls[0]["language"], "English")

    def test_backend_initializes_vllm_model(self):
        config = TranscriptionEngineConfig(
            model="Qwen/Qwen3-ASR-1.7B",
            engine_options={"backend": "vllm", "model": {"gpu_memory_utilization": 0.7}},
        )

        Qwen3ASRBackend(config, model_factory=FakeQwenFactory, torch_module=FakeTorch)

        self.assertEqual(
            FakeQwenFactory.llm_calls[0],
            {"gpu_memory_utilization": 0.7, "model": "Qwen/Qwen3-ASR-1.7B"},
        )

    def test_missing_qwen_asr_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="Qwen/Qwen3-ASR-1.7B")

        with patch(
            "RealtimeSTT.transcription_engines.qwen3_asr_engine.import_module",
            side_effect=ModuleNotFoundError("qwen_asr"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "qwen-asr"):
                Qwen3ASRBackend(config)

    def test_engine_maps_language_code_and_timestamps(self):
        backend = FakeBackend(output=[FakeOutput(" qwen transcript ", "German")])
        config = TranscriptionEngineConfig(
            model="Qwen/Qwen3-ASR-1.7B",
            engine_options={"return_time_stamps": True},
        )
        engine = Qwen3ASREngine(config, backend=backend)

        result = engine.transcribe(AudioVector([0.0]), language="de")

        self.assertEqual(
            backend.calls[0][1],
            {"language": "German", "return_time_stamps": True},
        )
        self.assertEqual(result.text, "qwen transcript")
        self.assertEqual(result.info.language, "German")


class MoonshineEngineTests(unittest.TestCase):
    def tearDown(self):
        FakeMoonshineProcessor.instances.clear()
        FakeMoonshineModel.instances.clear()

    def test_backend_initializes_streaming_model_and_decodes(self):
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
            FakeMoonshineProcessor.instances[0].calls[0][1],
            {"return_tensors": "pt", "sampling_rate": 16000},
        )
        self.assertEqual(
            FakeMoonshineModel.instances[0].generate_calls[0],
            {"input_values": "features", "max_new_tokens": 32},
        )

    def test_engine_rejects_non_english_language(self):
        engine = MoonshineEngine(
            TranscriptionEngineConfig(model="UsefulSensors/moonshine-streaming-medium"),
            backend=FakeBackend(),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "English"):
            engine.transcribe(AudioVector([0.0]), language="de")

    def test_engine_returns_english_result(self):
        backend = FakeBackend(output=[" moonshine transcript "])
        engine = MoonshineEngine(
            TranscriptionEngineConfig(model="UsefulSensors/moonshine-streaming-medium"),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0]), language="en")

        self.assertEqual(result.text, "moonshine transcript")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.info.language_probability, 1.0)


class AdditionalEngineGoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest("NumPy is required for opt-in real-model ASR smoke tests")

    def assert_engine_transcribes_fixture(self, engine, language="en"):
        audio, expected = read_fixture_audio()
        result = engine.transcribe(audio, language=language)
        actual = normalize_transcript(result.text)

        print("\n[RealtimeSTT test] %s expected: %s" % (engine.engine_name, expected))
        print("[RealtimeSTT test] %s actual:   %s" % (engine.engine_name, actual))

        self.assertTrue(actual)
        self.assertIn(" ".join(expected.split()[:2]), actual)

    def test_transcribes_fixture_with_real_parakeet_backend(self):
        if os.environ.get("REALTIMESTT_RUN_PARAKEET") != "1":
            self.skipTest("Set REALTIMESTT_RUN_PARAKEET=1 to run the Parakeet smoke test")
        if os.name == "nt":
            self.skipTest("Run the Parakeet/NeMo smoke test from Linux or WSL2")

        engine = ParakeetEngine(
            TranscriptionEngineConfig(
                model=os.environ.get(
                    "REALTIMESTT_PARAKEET_MODEL",
                    "nvidia/parakeet-tdt-0.6b-v3",
                ),
                device=os.environ.get("REALTIMESTT_PARAKEET_DEVICE", "cuda"),
                engine_options={"sample_rate": 16000},
            )
        )
        self.assert_engine_transcribes_fixture(engine, language=None)

    def test_transcribes_fixture_with_real_cohere_backend(self):
        if os.environ.get("REALTIMESTT_RUN_COHERE_TRANSCRIBE") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_COHERE_TRANSCRIBE=1 to run the Cohere smoke test"
            )

        engine = CohereTranscribeEngine(
            TranscriptionEngineConfig(
                model=os.environ.get(
                    "REALTIMESTT_COHERE_TRANSCRIBE_MODEL",
                    "CohereLabs/cohere-transcribe-03-2026",
                ),
                device=os.environ.get("REALTIMESTT_COHERE_TRANSCRIBE_DEVICE", "cpu"),
                compute_type=os.environ.get("REALTIMESTT_COHERE_TRANSCRIBE_COMPUTE_TYPE", "default"),
                download_root=os.environ.get("REALTIMESTT_HF_MODEL_DIR"),
                engine_options={"sample_rate": 16000, "language": "en"},
            )
        )
        self.assert_engine_transcribes_fixture(engine, language="en")

    def test_transcribes_fixture_with_real_granite_backend(self):
        if os.environ.get("REALTIMESTT_RUN_GRANITE_SPEECH") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_GRANITE_SPEECH=1 to run the Granite smoke test"
            )

        engine = GraniteSpeechEngine(
            TranscriptionEngineConfig(
                model=os.environ.get(
                    "REALTIMESTT_GRANITE_SPEECH_MODEL",
                    "ibm-granite/granite-speech-4.1-2b",
                ),
                device=os.environ.get("REALTIMESTT_GRANITE_SPEECH_DEVICE", "cpu"),
                compute_type=os.environ.get("REALTIMESTT_GRANITE_SPEECH_COMPUTE_TYPE", "float32"),
                download_root=os.environ.get("REALTIMESTT_HF_MODEL_DIR"),
            )
        )
        self.assert_engine_transcribes_fixture(engine, language="en")

    def test_transcribes_fixture_with_real_qwen3_asr_backend(self):
        if os.environ.get("REALTIMESTT_RUN_QWEN3_ASR") != "1":
            self.skipTest("Set REALTIMESTT_RUN_QWEN3_ASR=1 to run the Qwen3-ASR smoke test")
        backend = os.environ.get("REALTIMESTT_QWEN3_ASR_BACKEND", "transformers")
        if backend == "vllm" and os.name == "nt":
            self.skipTest("Run the Qwen3-ASR vLLM smoke test from Linux or WSL2")

        engine = Qwen3ASREngine(
            TranscriptionEngineConfig(
                model=os.environ.get("REALTIMESTT_QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
                device=os.environ.get("REALTIMESTT_QWEN3_ASR_DEVICE", "cpu"),
                compute_type=os.environ.get("REALTIMESTT_QWEN3_ASR_COMPUTE_TYPE", "float32"),
                download_root=os.environ.get("REALTIMESTT_HF_MODEL_DIR"),
                engine_options={"backend": backend, "sample_rate": 16000},
            )
        )
        self.assert_engine_transcribes_fixture(engine, language="en")

    def test_transcribes_fixture_with_real_moonshine_backend(self):
        if os.environ.get("REALTIMESTT_RUN_MOONSHINE") != "1":
            self.skipTest("Set REALTIMESTT_RUN_MOONSHINE=1 to run the Moonshine smoke test")

        engine = MoonshineEngine(
            TranscriptionEngineConfig(
                model=os.environ.get(
                    "REALTIMESTT_MOONSHINE_MODEL",
                    "UsefulSensors/moonshine-streaming-medium",
                ),
                device=os.environ.get("REALTIMESTT_MOONSHINE_DEVICE", "cpu"),
                compute_type=os.environ.get("REALTIMESTT_MOONSHINE_COMPUTE_TYPE", "float32"),
                download_root=os.environ.get("REALTIMESTT_HF_MODEL_DIR"),
            )
        )
        self.assert_engine_transcribes_fixture(engine, language="en")


if __name__ == "__main__":
    unittest.main()
