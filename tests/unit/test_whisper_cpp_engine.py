import json
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
from RealtimeSTT.transcription_engines.whisper_cpp_engine import (
    PyWhisperCppBackend,
    WhisperCppEngine,
)


AUDIO_DIR = Path(__file__).with_name("audio")
MANIFEST_PATH = AUDIO_DIR / "manifest.json"


class FakeSegment:
    def __init__(self, text):
        self.text = text


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


class FakeBackend:
    def __init__(self, config=None):
        self.config = config
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return [FakeSegment(" hello"), FakeSegment("world ")]


class FakeModel:
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.calls = []
        FakeModel.instances.append(self)

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return [FakeSegment("ok")]


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_wav_samples(path):
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        return np.frombuffer(frames, dtype=np.int16), wav.getframerate()


def normalize_transcript(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class WhisperCppFactoryTests(unittest.TestCase):
    def test_supported_engines_include_whisper_cpp(self):
        self.assertIn("whisper_cpp", get_supported_transcription_engines())

    def test_factory_creates_whisper_cpp_with_optional_backend(self):
        config = TranscriptionEngineConfig(model="tiny.en")

        with patch(
            "RealtimeSTT.transcription_engines.whisper_cpp_engine.PyWhisperCppBackend",
            FakeBackend,
        ):
            engine = create_transcription_engine("whisper_cpp", config)

        self.assertIsInstance(engine, WhisperCppEngine)
        self.assertIsInstance(engine.backend, FakeBackend)
        self.assertIs(engine.backend.config, config)


class PyWhisperCppBackendTests(unittest.TestCase):
    def tearDown(self):
        FakeModel.instances.clear()

    def test_initializes_pywhispercpp_model_quietly(self):
        config = TranscriptionEngineConfig(
            model="tiny.en",
            download_root="D:/models",
        )

        PyWhisperCppBackend(config, model_cls=FakeModel)

        self.assertEqual(len(FakeModel.instances), 1)
        model = FakeModel.instances[0]
        self.assertEqual(model.model, "tiny.en")
        self.assertEqual(
            model.kwargs,
            {
                "models_dir": "D:/models",
                "params_sampling_strategy": 1,
                "print_progress": False,
                "print_realtime": False,
                "print_timestamps": False,
            },
        )

    def test_initializes_pywhispercpp_model_with_greedy_strategy_for_single_beam(self):
        config = TranscriptionEngineConfig(model="tiny.en", beam_size=1)

        PyWhisperCppBackend(config, model_cls=FakeModel)

        self.assertEqual(FakeModel.instances[0].kwargs["params_sampling_strategy"], 0)

    def test_initializes_pywhispercpp_model_with_engine_options(self):
        config = TranscriptionEngineConfig(
            model="tiny.en",
            engine_options={
                "model": {
                    "n_threads": 8,
                    "single_segment": True,
                },
            },
        )

        PyWhisperCppBackend(config, model_cls=FakeModel)

        self.assertEqual(FakeModel.instances[0].kwargs["n_threads"], 8)
        self.assertTrue(FakeModel.instances[0].kwargs["single_segment"])

    def test_missing_pywhispercpp_reports_optional_dependency(self):
        config = TranscriptionEngineConfig(model="tiny.en")

        with patch(
            "RealtimeSTT.transcription_engines.whisper_cpp_engine.import_module",
            side_effect=ModuleNotFoundError("pywhispercpp"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "pip install pywhispercpp"):
                PyWhisperCppBackend(config)

    def test_transcribe_delegates_to_model(self):
        config = TranscriptionEngineConfig(
            model="tiny.en",
            engine_options={"transcribe": {"single_segment": True}},
        )
        backend = PyWhisperCppBackend(config, model_cls=FakeModel)
        audio = object()

        segments = backend.transcribe(audio, language="en")

        self.assertEqual([segment.text for segment in segments], ["ok"])
        self.assertIs(FakeModel.instances[0].calls[0][0], audio)
        self.assertEqual(
            FakeModel.instances[0].calls[0][1],
            {"n_processors": None, "language": "en", "single_segment": True},
        )


class WhisperCppEngineContractTests(unittest.TestCase):
    def test_transcribe_normalizes_audio_and_maps_segments_to_result(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(
            model="tiny.en",
            beam_size=3,
            initial_prompt="domain words",
            normalize_audio=True,
        )
        engine = WhisperCppEngine(config, backend=backend)
        audio = AudioVector([0.0, 2.0, -1.0])

        result = engine.transcribe(audio, language="en")

        backend_audio, params = backend.calls[0]
        self.assertEqual(backend_audio.values, [0.0, 0.95, -0.475])
        self.assertEqual(
            params,
            {
                "language": "en",
                "beam_search": {"beam_size": 3, "patience": -1.0},
                "print_progress": False,
                "print_realtime": False,
                "print_timestamps": False,
                "initial_prompt": "domain words",
            },
        )
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.info.language_probability, 1.0)

    def test_transcribe_omits_prompt_when_disabled(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(
            model="tiny.en",
            initial_prompt="domain words",
        )
        engine = WhisperCppEngine(config, backend=backend)

        engine.transcribe(AudioVector([0.0]), use_prompt=False)

        self.assertNotIn("initial_prompt", backend.calls[0][1])

    def test_transcribe_passes_token_prompts(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(
            model="tiny.en",
            initial_prompt=[1, 2, 3],
        )
        engine = WhisperCppEngine(config, backend=backend)

        engine.transcribe(AudioVector([0.0]))

        self.assertEqual(backend.calls[0][1]["prompt_tokens"], (1, 2, 3))
        self.assertEqual(backend.calls[0][1]["prompt_n_tokens"], 3)

    def test_transcribe_uses_greedy_decoding_for_single_beam(self):
        backend = FakeBackend()
        config = TranscriptionEngineConfig(model="tiny.en", beam_size=1)
        engine = WhisperCppEngine(config, backend=backend)

        engine.transcribe(AudioVector([0.0]))

        self.assertEqual(backend.calls[0][1]["greedy"], {"best_of": 1})
        self.assertNotIn("beam_search", backend.calls[0][1])

    def test_transcribe_rejects_empty_audio(self):
        engine = WhisperCppEngine(
            TranscriptionEngineConfig(model="tiny.en"),
            backend=FakeBackend(),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "Received None audio"):
            engine.transcribe(AudioVector([]))


class WhisperCppGoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("REALTIMESTT_RUN_WHISPER_CPP") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_WHISPER_CPP=1 to run the slow whisper.cpp "
                "golden transcription test"
            )
        if np is None:
            self.skipTest("NumPy is required for the whisper.cpp golden transcription test")

    def test_transcribes_ljspeech_sample_with_real_pywhispercpp_backend(self):
        manifest = load_manifest()
        sample = manifest["samples"][0]
        samples, _ = read_wav_samples(AUDIO_DIR / sample["file"])
        audio = samples.astype(np.float32) / 32768.0
        expected = normalize_transcript(sample["normalized_transcript"])

        engine = WhisperCppEngine(
            TranscriptionEngineConfig(
                model=os.environ.get("REALTIMESTT_WHISPER_CPP_MODEL", "tiny.en"),
                download_root=os.environ.get("REALTIMESTT_WHISPER_CPP_MODEL_DIR"),
                beam_size=int(os.environ.get("REALTIMESTT_WHISPER_CPP_BEAM_SIZE", "5")),
            )
        )
        result = engine.transcribe(audio, language="en")
        actual = normalize_transcript(result.text)

        print(f"\n[RealtimeSTT test] whisper_cpp {sample['id']} expected: {expected}")
        print(f"[RealtimeSTT test] whisper_cpp {sample['id']} actual:   {actual}")

        self.assertTrue(actual)
        self.assertIn(" ".join(expected.split()[:2]), actual)


if __name__ == "__main__":
    unittest.main()
