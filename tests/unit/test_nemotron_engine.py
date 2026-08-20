import os
import hashlib
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from RealtimeSTT.model_manifests import (
    ModelFileManifest,
    ModelManifest,
    SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST,
    SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST,
)
from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.nemotron_engine import (
    DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL,
    SherpaOnnxNemotronBackend,
    SherpaOnnxNemotronEngine,
    SherpaOnnxNemotronStreamingSession,
)
from RealtimeSTT.transcription_engines.parakeet_engine import ParakeetEngine
from RealtimeSTT.transcription_engines.sherpa_onnx_engine import (
    SherpaOnnxDecodedOutput,
    SherpaOnnxParakeetEngine,
)


class _AudioVector:
    def __init__(self, values):
        self.values = list(values)
        self.size = len(self.values)


class _Result:
    def __init__(self, text=""):
        self.text = text


class _Stream:
    def __init__(self):
        self.accepted = []
        self.options = []
        self.finished = False
        self.closed = False
        self.pending_decodes = 0
        self.result = _Result()

    def set_option(self, key, value):
        self.options.append((key, value))

    def accept_waveform(self, sample_rate, audio):
        self.accepted.append((sample_rate, audio))
        self.pending_decodes += 1
        self.result.text = "partial"

    def input_finished(self):
        self.finished = True
        self.pending_decodes += 1
        self.result.text = "final"

    def close(self):
        self.closed = True


class _Recognizer:
    calls = []

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.streams = []

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls(kwargs)

    def create_stream(self):
        stream = _Stream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return stream.pending_decodes > 0

    def decode_stream(self, stream):
        stream.pending_decodes -= 1

    def get_result_all(self, stream):
        return stream.result


class _Backend:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.output


class _ParakeetOutput:
    def __init__(self, text, language=None):
        self.text = text
        self.language = language


class ModelManifestTests(unittest.TestCase):
    def test_manifests_are_pinned_to_exact_archives(self):
        nemotron = SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST
        self.assertEqual(nemotron.model_id, DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL)
        self.assertEqual(nemotron.archive_size_bytes, 475271763)
        self.assertEqual(
            nemotron.archive_sha256,
            "c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a",
        )
        self.assertEqual(
            nemotron.expected_files,
            ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
        )
        self.assertIn("Open Model Data Warehouse", nemotron.license_name)

        parakeet = SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST
        self.assertEqual(parakeet.archive_size_bytes, 487170055)
        self.assertEqual(
            parakeet.archive_sha256,
            "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
        )
        self.assertEqual(parakeet.license_name, "CC-BY-4.0")

    def test_extracted_file_metadata_is_structured_and_verifiable(self):
        payload = b"model"
        digest = hashlib.sha256(payload).hexdigest()
        manifest = ModelManifest(
            model_id="test-model",
            archive_url="https://example.invalid/test.tar.bz2",
            archive_filename="test.tar.bz2",
            archive_size_bytes=len(payload),
            archive_sha256=digest,
            expected_files=("encoder.onnx",),
            license_name="test",
            license_url="https://example.invalid/license",
            file_metadata=(ModelFileManifest("encoder.onnx", len(payload), digest),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.onnx"
            path.write_bytes(payload)
            self.assertEqual(manifest.expected_file_metadata[0].filename, "encoder.onnx")
            self.assertEqual(manifest.invalid_files(Path(directory)), ())
            path.write_bytes(b"changed")
            self.assertIn("encoder.onnx", manifest.describe_invalid_files(Path(directory)))


class NemotronBackendTests(unittest.TestCase):
    def setUp(self):
        _Recognizer.calls.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)
        for name in SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST.expected_files:
            (self.model_dir / name).write_text("placeholder", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_engine(self, **options):
        config = TranscriptionEngineConfig(
            model=str(self.model_dir),
            engine_options=options,
        )
        backend = SherpaOnnxNemotronBackend(config, recognizer_cls=_Recognizer)
        return SherpaOnnxNemotronEngine(config, backend=backend), backend

    def test_backend_uses_online_recognizer_and_nemotron_feature_shape(self):
        engine, backend = self.make_engine(num_threads=3, provider="cpu")

        call = _Recognizer.calls[0]
        self.assertIs(engine.backend, backend)
        self.assertEqual(call["feature_dim"], 128)
        self.assertEqual(call["sample_rate"], 16000)
        self.assertEqual(call["enable_endpoint_detection"], False)
        self.assertEqual(call["model_type"], "")
        self.assertEqual(call["num_threads"], 3)

    def test_opt_in_model_file_verification_rejects_placeholder_files(self):
        with self.assertRaisesRegex(TranscriptionEngineError, "verification failed"):
            self.make_engine(verify_model_files=True)

    def test_stream_sets_language_before_audio_and_decodes_partials(self):
        engine, backend = self.make_engine()
        session = engine.create_streaming_session(language="de")

        self.assertIsInstance(session, SherpaOnnxNemotronStreamingSession)
        stream = backend.recognizer.streams[0]
        self.assertEqual(stream.options, [("language", "de")])

        session.accept_audio([0.1, 0.2], sample_rate=16000)
        session.decode()
        partial = session.get_result()
        self.assertEqual(partial.text, "partial")
        self.assertEqual(partial.info.language, "de")
        self.assertEqual(partial.info.language_probability, 0.0)
        self.assertEqual(stream.accepted[0][0], 16000)

        final = session.finish()
        self.assertEqual(final.text, "final")
        self.assertTrue(stream.finished)
        self.assertEqual(session.input_finished().text, "final")
        session.close()
        self.assertTrue(stream.closed)

    def test_auto_language_is_stream_local_and_fixed_language_is_not_detection(self):
        engine, backend = self.make_engine()
        auto = engine.create_streaming_session(language=None)
        fixed = engine.create_streaming_session(language="fr-FR")

        self.assertEqual(backend.recognizer.streams[0].options, [("language", "auto")])
        self.assertEqual(backend.recognizer.streams[1].options, [("language", "fr-FR")])
        auto.accept_audio([0.1])
        auto.decode()
        self.assertIsNone(auto.get_result().info.language)
        fixed.accept_audio([0.1])
        fixed.decode()
        self.assertEqual(fixed.get_result().info.language, "fr-FR")
        self.assertEqual(fixed.get_result().info.language_probability, 0.0)

    def test_rejects_non_16khz_or_stereo_audio(self):
        engine, _ = self.make_engine()
        session = engine.create_streaming_session()

        with self.assertRaisesRegex(TranscriptionEngineError, "16 kHz"):
            session.accept_audio([0.1], sample_rate=8000)
        with self.assertRaisesRegex(TranscriptionEngineError, "mono"):
            session.accept_audio([[0.1, 0.2], [0.3, 0.4]], sample_rate=16000)

    def test_reset_releases_old_stream_and_close_is_idempotent(self):
        engine, backend = self.make_engine()
        session = engine.create_streaming_session()
        first = backend.recognizer.streams[0]
        session.accept_audio([0.1])
        session.reset()
        second = backend.recognizer.streams[1]

        self.assertTrue(first.closed)
        self.assertIsNot(first, second)
        session.close()
        session.close()
        self.assertTrue(second.closed)

    def test_transcribe_finishes_and_releases_its_stream(self):
        engine, backend = self.make_engine()

        result = engine.transcribe([0.1, 0.2], language="en")

        self.assertEqual(result.text, "final")
        stream = backend.recognizer.streams[0]
        self.assertTrue(stream.finished)
        self.assertTrue(stream.closed)


class ParakeetLanguageAndEmptyOutputTests(unittest.TestCase):
    def test_nemo_parakeet_fixed_language_is_not_reported_as_detected(self):
        engine = ParakeetEngine(
            TranscriptionEngineConfig(model="model"),
            backend=_Backend([_ParakeetOutput("hello", None)]),
        )

        result = engine.transcribe(_AudioVector([0.1]), language="en")

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.info.language_probability, 0.0)

    def test_nemo_parakeet_empty_output_has_no_language(self):
        engine = ParakeetEngine(
            TranscriptionEngineConfig(model="model"),
            backend=_Backend([_ParakeetOutput("", None)]),
        )

        result = engine.transcribe(_AudioVector([0.1]), language="en")

        self.assertEqual(result.text, "")
        self.assertIsNone(result.info.language)
        self.assertEqual(result.info.language_probability, 0.0)

    def test_sherpa_parakeet_empty_output_does_not_fallback_to_caller_language(self):
        engine = SherpaOnnxParakeetEngine(
            TranscriptionEngineConfig(model="model"),
            backend=_Backend(SherpaOnnxDecodedOutput("", "")),
        )

        result = engine.transcribe(_AudioVector([0.1]), language="de")

        self.assertEqual(result.text, "")
        self.assertIsNone(result.info.language)
        self.assertEqual(result.info.language_probability, 0.0)


class NemotronFactoryTests(unittest.TestCase):
    def test_aliases_are_registered(self):
        engines = get_supported_transcription_engines()
        for name in (
            "nemotron",
            "sherpa_nemotron",
            "sherpa_onnx_nemotron",
            "nemotron_sherpa_onnx",
        ):
            self.assertIn(name, engines)

    def test_factory_imports_the_new_engine_class_lazily(self):
        config = TranscriptionEngineConfig(model="model")
        original = SherpaOnnxNemotronEngine.__init__

        def fake_init(self, config):
            super(SherpaOnnxNemotronEngine, self).__init__(config)
            self.backend = object()

        SherpaOnnxNemotronEngine.__init__ = fake_init
        try:
            engine = create_transcription_engine("nemotron", config)
        finally:
            SherpaOnnxNemotronEngine.__init__ = original
        self.assertIsInstance(engine, SherpaOnnxNemotronEngine)


class NemotronGoldenTranscriptionTests(unittest.TestCase):
    def test_transcribes_fixture_with_real_nemotron_backend(self):
        if os.environ.get("REALTIMESTT_RUN_SHERPA_ONNX_NEMOTRON") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_SHERPA_ONNX_NEMOTRON=1 to run the "
                "Nemotron smoke test"
            )
        model_dir = Path(
            os.environ.get(
                "REALTIMESTT_SHERPA_ONNX_NEMOTRON_MODEL",
                "test-model-cache/sherpa-onnx/%s" % DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL,
            )
        )
        if not model_dir.is_dir():
            self.skipTest("Nemotron model directory not found: %s" % model_dir)

        try:
            engine = SherpaOnnxNemotronEngine(
                TranscriptionEngineConfig(
                    model=str(model_dir),
                    device="cpu",
                    engine_options={
                        "provider": "cpu",
                        "num_threads": int(
                            os.environ.get("REALTIMESTT_SHERPA_ONNX_NUM_THREADS", "1")
                        ),
                    },
                )
            )
        except TranscriptionEngineError as exc:
            self.skipTest(str(exc))

        fixture = Path(__file__).with_name("audio") / "asr-reference.wav"
        with wave.open(str(fixture), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            samples = [
                value / 32768.0
                for value in struct.unpack("<%dh" % (wav.getnframes(),), wav.readframes(wav.getnframes()))
            ]
        result = engine.transcribe(samples, language="en")
        self.assertTrue(result.text)


if __name__ == "__main__":
    unittest.main()
