import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.parakeet_engine import ParakeetEngine
from RealtimeSTT.transcription_engines.moonshine_engine import MoonshineEngine
from RealtimeSTT.transcription_engines.sherpa_onnx_engine import (
    DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL,
    DEFAULT_SHERPA_ONNX_PARAKEET_MODEL,
    SherpaOnnxDecodedOutput,
    SherpaOnnxMoonshineBackend,
    SherpaOnnxMoonshineEngine,
    SherpaOnnxParakeetBackend,
    SherpaOnnxParakeetEngine,
)
from tests.unit import test_additional_transcription_engines as audio_fixtures
from tests.unit.test_additional_transcription_engines import AudioVector


PARAKEET_FILES = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
]
MOONSHINE_FILES = [
    "preprocess.onnx",
    "encode.int8.onnx",
    "uncached_decode.int8.onnx",
    "cached_decode.int8.onnx",
    "tokens.txt",
]


class FakeSherpaResult:
    def __init__(self, text="sherpa text", lang="de"):
        self.text = text
        self.lang = lang


class FakeSherpaStream:
    def __init__(self):
        self.accepted = []
        self.result = FakeSherpaResult()

    def accept_waveform(self, sample_rate, audio):
        self.accepted.append((sample_rate, audio))


class FakeSherpaRecognizer:
    transducer_calls = []
    moonshine_calls = []

    def __init__(self, family, kwargs):
        self.family = family
        self.kwargs = kwargs
        self.streams = []

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.transducer_calls.append(kwargs)
        return cls("transducer", kwargs)

    @classmethod
    def from_moonshine(cls, **kwargs):
        cls.moonshine_calls.append(kwargs)
        return cls("moonshine", kwargs)

    def create_stream(self):
        stream = FakeSherpaStream()
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        stream.result = FakeSherpaResult()


class FakeSherpaBackend:
    def __init__(self, config=None):
        self.config = config
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return SherpaOnnxDecodedOutput("mocked sherpa text", "en")


class SherpaOnnxEngineTests(unittest.TestCase):
    def tearDown(self):
        FakeSherpaRecognizer.transducer_calls.clear()
        FakeSherpaRecognizer.moonshine_calls.clear()

    def make_model_dir(self, filenames, directory_name=None):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        model_dir = root / directory_name if directory_name else root
        model_dir.mkdir(exist_ok=True)
        for filename in filenames:
            (model_dir / filename).write_text("placeholder", encoding="utf-8")
        return temp_dir, model_dir

    def test_supported_engines_include_sherpa_aliases(self):
        engines = get_supported_transcription_engines()

        for name in (
            "sherpa_onnx_parakeet",
            "sherpa_parakeet",
            "parakeet_sherpa_onnx",
            "sherpa_onnx_moonshine",
            "sherpa_moonshine",
            "moonshine_sherpa_onnx",
        ):
            self.assertIn(name, engines)

    def test_factory_creates_sherpa_engines_with_mocked_backends(self):
        cases = [
            (
                "sherpa-onnx-parakeet",
                "SherpaOnnxParakeetBackend",
                SherpaOnnxParakeetEngine,
            ),
            (
                "sherpa-onnx-moonshine",
                "SherpaOnnxMoonshineBackend",
                SherpaOnnxMoonshineEngine,
            ),
        ]

        for engine_name, backend_name, engine_cls in cases:
            with self.subTest(engine=engine_name):
                with patch(
                    "RealtimeSTT.transcription_engines.sherpa_onnx_engine.%s"
                    % backend_name,
                    FakeSherpaBackend,
                ):
                    engine = create_transcription_engine(
                        engine_name,
                        TranscriptionEngineConfig(model="model-dir"),
                    )

                self.assertIsInstance(engine, engine_cls)
                self.assertIsInstance(engine.backend, FakeSherpaBackend)

    def test_parakeet_backend_initializes_nemo_transducer_int8_model(self):
        temp_dir, model_dir = self.make_model_dir(PARAKEET_FILES)
        self.addCleanup(temp_dir.cleanup)
        config = TranscriptionEngineConfig(
            model=str(model_dir),
            engine_options={
                "num_threads": 3,
                "decoding_method": "greedy_search",
                "hotwords_score": 2.0,
            },
        )

        backend = SherpaOnnxParakeetBackend(
            config,
            recognizer_cls=FakeSherpaRecognizer,
        )

        call = FakeSherpaRecognizer.transducer_calls[0]
        self.assertEqual(call["encoder"], str(model_dir / "encoder.int8.onnx"))
        self.assertEqual(call["decoder"], str(model_dir / "decoder.int8.onnx"))
        self.assertEqual(call["joiner"], str(model_dir / "joiner.int8.onnx"))
        self.assertEqual(call["tokens"], str(model_dir / "tokens.txt"))
        self.assertEqual(call["model_type"], "nemo_transducer")
        self.assertEqual(call["provider"], "cpu")
        self.assertEqual(call["num_threads"], 3)
        self.assertEqual(call["hotwords_score"], 2.0)
        self.assertEqual(backend.recognizer.family, "transducer")

    def test_parakeet_backend_resolves_known_model_under_download_root(self):
        temp_dir, model_dir = self.make_model_dir(
            PARAKEET_FILES,
            DEFAULT_SHERPA_ONNX_PARAKEET_MODEL,
        )
        self.addCleanup(temp_dir.cleanup)
        config = TranscriptionEngineConfig(
            model="nvidia/parakeet-tdt-0.6b-v3",
            download_root=str(Path(temp_dir.name)),
        )

        SherpaOnnxParakeetBackend(config, recognizer_cls=FakeSherpaRecognizer)

        call = FakeSherpaRecognizer.transducer_calls[0]
        self.assertEqual(call["encoder"], str(model_dir / "encoder.int8.onnx"))

    def test_parakeet_engine_transcribes_with_sherpa_stream(self):
        temp_dir, model_dir = self.make_model_dir(PARAKEET_FILES)
        self.addCleanup(temp_dir.cleanup)
        backend = SherpaOnnxParakeetBackend(
            TranscriptionEngineConfig(model=str(model_dir)),
            recognizer_cls=FakeSherpaRecognizer,
        )
        engine = SherpaOnnxParakeetEngine(
            TranscriptionEngineConfig(model=str(model_dir)),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0, 0.5]), language=None)

        self.assertEqual(result.text, "sherpa text")
        self.assertEqual(result.info.language, "de")
        self.assertEqual(backend.recognizer.streams[0].accepted[0][0], 16000)

    def test_moonshine_backend_initializes_tiny_int8_model(self):
        temp_dir, model_dir = self.make_model_dir(MOONSHINE_FILES)
        self.addCleanup(temp_dir.cleanup)
        config = TranscriptionEngineConfig(
            model=str(model_dir),
            engine_options={"num_threads": 2},
        )

        SherpaOnnxMoonshineBackend(config, recognizer_cls=FakeSherpaRecognizer)

        call = FakeSherpaRecognizer.moonshine_calls[0]
        self.assertEqual(call["preprocessor"], str(model_dir / "preprocess.onnx"))
        self.assertEqual(call["encoder"], str(model_dir / "encode.int8.onnx"))
        self.assertEqual(
            call["uncached_decoder"],
            str(model_dir / "uncached_decode.int8.onnx"),
        )
        self.assertEqual(
            call["cached_decoder"],
            str(model_dir / "cached_decode.int8.onnx"),
        )
        self.assertEqual(call["tokens"], str(model_dir / "tokens.txt"))
        self.assertEqual(call["num_threads"], 2)

    def test_moonshine_backend_resolves_default_directory_under_download_root(self):
        temp_dir, model_dir = self.make_model_dir(
            MOONSHINE_FILES,
            DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL,
        )
        self.addCleanup(temp_dir.cleanup)
        config = TranscriptionEngineConfig(
            model="UsefulSensors/moonshine-streaming-medium",
            download_root=str(Path(temp_dir.name)),
        )

        SherpaOnnxMoonshineBackend(config, recognizer_cls=FakeSherpaRecognizer)

        call = FakeSherpaRecognizer.moonshine_calls[0]
        self.assertEqual(call["encoder"], str(model_dir / "encode.int8.onnx"))

    def test_moonshine_engine_rejects_non_english_language(self):
        engine = SherpaOnnxMoonshineEngine(
            TranscriptionEngineConfig(model="model-dir"),
            backend=FakeSherpaBackend(),
        )

        with self.assertRaisesRegex(TranscriptionEngineError, "English"):
            engine.transcribe(AudioVector([0.0]), language="de")

    def test_existing_engine_names_can_select_sherpa_backend(self):
        config = TranscriptionEngineConfig(
            model="model-dir",
            engine_options={"backend": "sherpa_onnx"},
        )

        with patch(
            "RealtimeSTT.transcription_engines.sherpa_onnx_engine."
            "SherpaOnnxParakeetBackend",
            FakeSherpaBackend,
        ):
            parakeet = create_transcription_engine("parakeet", config)
        with patch(
            "RealtimeSTT.transcription_engines.sherpa_onnx_engine."
            "SherpaOnnxMoonshineBackend",
            FakeSherpaBackend,
        ):
            moonshine = create_transcription_engine("moonshine", config)

        self.assertIsInstance(parakeet, ParakeetEngine)
        self.assertIsInstance(parakeet.backend, FakeSherpaBackend)
        self.assertIsInstance(moonshine, MoonshineEngine)
        self.assertIsInstance(moonshine.backend, FakeSherpaBackend)

    def test_missing_dependency_reports_install_hint(self):
        temp_dir, model_dir = self.make_model_dir(PARAKEET_FILES)
        self.addCleanup(temp_dir.cleanup)
        config = TranscriptionEngineConfig(model=str(model_dir))

        with patch(
            "RealtimeSTT.transcription_engines.sherpa_onnx_engine.import_module",
            side_effect=ModuleNotFoundError("sherpa_onnx"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "sherpa-onnx"):
                SherpaOnnxParakeetBackend(config)

    def test_missing_model_file_reports_download_hint(self):
        temp_dir, model_dir = self.make_model_dir(PARAKEET_FILES[:-1])
        self.addCleanup(temp_dir.cleanup)
        config = TranscriptionEngineConfig(model=str(model_dir))

        with self.assertRaisesRegex(TranscriptionEngineError, "tokens"):
            SherpaOnnxParakeetBackend(config, recognizer_cls=FakeSherpaRecognizer)


class SherpaOnnxGoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if audio_fixtures.np is None:
            self.skipTest("NumPy is required for sherpa-onnx golden tests")

    def assert_engine_transcribes_fixture(self, engine):
        audio, expected = audio_fixtures.read_fixture_audio()
        start = time.time()
        result = engine.transcribe(audio, language="en")
        elapsed = time.time() - start
        duration = len(audio) / 16000.0
        actual = audio_fixtures.normalize_transcript(result.text)

        print("\n[RealtimeSTT test] %s expected: %s" % (engine.engine_name, expected))
        print("[RealtimeSTT test] %s actual:   %s" % (engine.engine_name, actual))
        print(
            "[RealtimeSTT test] %s RTF: %.3f / %.3f = %.3f"
            % (engine.engine_name, elapsed, duration, elapsed / duration)
        )

        self.assertTrue(actual)
        self.assertIn(" ".join(expected.split()[:2]), actual)

    def test_transcribes_fixture_with_real_sherpa_parakeet_backend(self):
        if os.environ.get("REALTIMESTT_RUN_SHERPA_ONNX_PARAKEET") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_SHERPA_ONNX_PARAKEET=1 to run the "
                "sherpa-onnx Parakeet smoke test"
            )
        model_dir = Path(
            os.environ.get(
                "REALTIMESTT_SHERPA_ONNX_PARAKEET_MODEL",
                "test-model-cache/sherpa-onnx/%s"
                % DEFAULT_SHERPA_ONNX_PARAKEET_MODEL,
            )
        )
        if not model_dir.is_dir():
            self.skipTest("sherpa-onnx Parakeet model directory not found: %s" % model_dir)

        engine = SherpaOnnxParakeetEngine(
            TranscriptionEngineConfig(
                model=str(model_dir),
                device="cpu",
                engine_options={
                    "num_threads": int(
                        os.environ.get("REALTIMESTT_SHERPA_ONNX_NUM_THREADS", "1")
                    ),
                },
            )
        )
        self.assert_engine_transcribes_fixture(engine)

    def test_transcribes_fixture_with_real_sherpa_moonshine_backend(self):
        if os.environ.get("REALTIMESTT_RUN_SHERPA_ONNX_MOONSHINE") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_SHERPA_ONNX_MOONSHINE=1 to run the "
                "sherpa-onnx Moonshine smoke test"
            )
        model_dir = Path(
            os.environ.get(
                "REALTIMESTT_SHERPA_ONNX_MOONSHINE_MODEL",
                "test-model-cache/sherpa-onnx/%s"
                % DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL,
            )
        )
        if not model_dir.is_dir():
            self.skipTest("sherpa-onnx Moonshine model directory not found: %s" % model_dir)

        engine = SherpaOnnxMoonshineEngine(
            TranscriptionEngineConfig(
                model=str(model_dir),
                device="cpu",
                engine_options={
                    "num_threads": int(
                        os.environ.get("REALTIMESTT_SHERPA_ONNX_NUM_THREADS", "1")
                    ),
                },
            )
        )
        self.assert_engine_transcribes_fixture(engine)


if __name__ == "__main__":
    unittest.main()
