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
from RealtimeSTT.transcription_engines.openai_whisper_engine import (
    OpenAIWhisperBackend,
    OpenAIWhisperEngine,
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


class FakeWhisperModel:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return {"text": " hello world ", "language": params.get("language") or "en"}


class FakeWhisperModule:
    loaded = []

    @classmethod
    def load_model(cls, model, **kwargs):
        model_instance = FakeWhisperModel(model, **kwargs)
        cls.loaded.append(model_instance)
        return model_instance


class FakeBackend:
    def __init__(self, config=None):
        self.config = config
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return {"text": " mocked transcript ", "language": params.get("language")}


def read_wav_samples(path):
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        return np.frombuffer(frames, dtype=np.int16), wav.getframerate()


def normalize_transcript(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class OpenAIWhisperFactoryTests(unittest.TestCase):
    def test_supported_engines_include_openai_whisper(self):
        self.assertIn("openai_whisper", get_supported_transcription_engines())

    def test_factory_creates_openai_whisper_with_optional_backend(self):
        config = TranscriptionEngineConfig(model="tiny.en")

        with patch(
            "RealtimeSTT.transcription_engines.openai_whisper_engine.OpenAIWhisperBackend",
            FakeBackend,
        ):
            engine = create_transcription_engine("openai_whisper", config)

        self.assertIsInstance(engine, OpenAIWhisperEngine)
        self.assertIsInstance(engine.backend, FakeBackend)
        self.assertIs(engine.backend.config, config)


class OpenAIWhisperBackendTests(unittest.TestCase):
    def tearDown(self):
        FakeWhisperModule.loaded.clear()

    def test_initializes_openai_whisper_model_with_config(self):
        config = TranscriptionEngineConfig(
            model="base",
            download_root="D:/models/whisper",
            device="cuda",
            engine_options={"model": {"in_memory": True}},
        )

        OpenAIWhisperBackend(config, whisper_module=FakeWhisperModule)

        self.assertEqual(len(FakeWhisperModule.loaded), 1)
        model = FakeWhisperModule.loaded[0]
        self.assertEqual(model.model, "base")
        self.assertEqual(
            model.kwargs,
            {
                "device": "cuda",
                "download_root": "D:/models/whisper",
                "in_memory": True,
            },
        )

    def test_load_model_options_alias_is_supported(self):
        config = TranscriptionEngineConfig(
            model="tiny",
            engine_options={"load_model": {"device": "cpu"}},
        )

        OpenAIWhisperBackend(config, whisper_module=FakeWhisperModule)

        self.assertEqual(FakeWhisperModule.loaded[0].kwargs, {"device": "cpu"})

    def test_missing_openai_whisper_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="tiny")

        with patch(
            "RealtimeSTT.transcription_engines.openai_whisper_engine.import_module",
            side_effect=ModuleNotFoundError("whisper"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "pip install openai-whisper"):
                OpenAIWhisperBackend(config)

    def test_transcribe_merges_backend_options(self):
        config = TranscriptionEngineConfig(
            model="tiny",
            engine_options={"transcribe": {"temperature": 0.0, "verbose": None}},
        )
        backend = OpenAIWhisperBackend(config, whisper_module=FakeWhisperModule)
        audio = object()

        result = backend.transcribe(audio, verbose=False, language="en")

        self.assertEqual(result["text"], " hello world ")
        self.assertEqual(
            FakeWhisperModule.loaded[0].calls[0],
            (audio, {"verbose": None, "language": "en", "temperature": 0.0}),
        )


class OpenAIWhisperEngineContractTests(unittest.TestCase):
    def test_transcribe_normalizes_audio_and_maps_result(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(
            model="tiny.en",
            beam_size=4,
            compute_type="float32",
            initial_prompt="domain words",
            normalize_audio=True,
            suppress_tokens=[-1, 50256],
        )
        engine = OpenAIWhisperEngine(config, backend=backend)
        audio = AudioVector([0.0, 2.0, -1.0])

        result = engine.transcribe(audio, language="en")

        backend_audio, params = backend.calls[0]
        self.assertEqual(backend_audio.values, [0.0, 0.95, -0.475])
        self.assertEqual(
            params,
            {
                "language": "en",
                "verbose": False,
                "beam_size": 4,
                "initial_prompt": "domain words",
                "suppress_tokens": [-1, 50256],
                "fp16": False,
            },
        )
        self.assertEqual(result.text, "mocked transcript")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.info.language_probability, 1.0)

    def test_transcribe_omits_prompt_when_disabled(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(model="tiny", initial_prompt="domain words")
        engine = OpenAIWhisperEngine(config, backend=backend)

        engine.transcribe(AudioVector([0.0]), use_prompt=False)

        self.assertNotIn("initial_prompt", backend.calls[0][1])

    def test_transcribe_rejects_token_prompt(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(model="tiny", initial_prompt=[1, 2, 3])
        engine = OpenAIWhisperEngine(config, backend=backend)

        with self.assertRaisesRegex(TranscriptionEngineError, "string initial_prompt"):
            engine.transcribe(AudioVector([0.0]))

    def test_transcribe_rejects_empty_audio(self):
        engine = OpenAIWhisperEngine(
            TranscriptionEngineConfig(model="tiny"),
            backend=FakeBackend(),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "Received None audio"):
            engine.transcribe(AudioVector([]))


class OpenAIWhisperGoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("REALTIMESTT_RUN_OPENAI_WHISPER") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_OPENAI_WHISPER=1 to run the slow "
                "openai-whisper golden transcription test"
            )
        if np is None:
            self.skipTest("NumPy is required for the openai-whisper golden test")

    def test_transcribes_ljspeech_sample_with_real_openai_whisper_backend(self):
        transcript_path = AUDIO_DIR / "LJ001-0002.txt"
        wav_path = AUDIO_DIR / "LJ001-0002.wav"
        samples, _ = read_wav_samples(wav_path)
        audio = samples.astype(np.float32) / 32768.0
        expected = normalize_transcript(transcript_path.read_text(encoding="utf-8"))

        engine = OpenAIWhisperEngine(
            TranscriptionEngineConfig(
                model=os.environ.get("REALTIMESTT_OPENAI_WHISPER_MODEL", "tiny.en"),
                download_root=os.environ.get("REALTIMESTT_OPENAI_WHISPER_MODEL_DIR"),
                device=os.environ.get("REALTIMESTT_OPENAI_WHISPER_DEVICE", "cpu"),
                compute_type=os.environ.get("REALTIMESTT_OPENAI_WHISPER_COMPUTE_TYPE", "float32"),
            )
        )
        result = engine.transcribe(audio, language="en")
        actual = normalize_transcript(result.text)

        print("\n[RealtimeSTT test] openai_whisper expected: %s" % expected)
        print("[RealtimeSTT test] openai_whisper actual:   %s" % actual)

        self.assertTrue(actual)
        self.assertIn(" ".join(expected.split()[:2]), actual)


if __name__ == "__main__":
    unittest.main()
