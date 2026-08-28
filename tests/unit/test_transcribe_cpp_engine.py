import ctypes
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.transcribe_cpp_engine import (
    TranscribeCppBackend,
    TranscribeCppEngine,
)


class FakeDevice:
    def __init__(self, kind, name, device_id):
        self.kind = kind
        self.name = name
        self.description = name
        self.device_type = "gpu" if kind != "cpu" else "cpu"
        self.device_id = device_id


class FakeTimings:
    load_ms = 1.0
    mel_ms = 2.0
    encode_ms = 3.0
    decode_ms = 4.0


class FakeWord:
    def __init__(self, text, t0_ms, t1_ms):
        self.text = text
        self.t0_ms = t0_ms
        self.t1_ms = t1_ms


class FakeNativeResult:
    def __init__(self, text=" hello ", language="", timestamp_kind="none"):
        self.text = text
        self.language = language
        self.timestamp_kind = timestamp_kind
        self.timings = FakeTimings()
        self.words = (FakeWord("hello", 100, 500),)
        self.tokens = ()
        self.segments = ()
        self.speaker_segments = ()


class FakeSession:
    def __init__(self, model, options):
        self.model = model
        self.options = options
        self.calls = []
        self.closed = False
        self.failure = None
        self.delay = 0.0
        self.active = 0
        self.max_active = 0
        self.state_lock = threading.Lock()
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.before_clear = None
        self.release_clear = None

    def run(self, pcm, **options):
        # The real binding clears its per-session cancellation flag at the
        # beginning of each run, making a cancelled session reusable.
        if self.before_clear is not None:
            self.before_clear.set()
            if not self.release_clear.wait(0.5):
                raise AssertionError("native run clear barrier was not released")
        self.cancelled.clear()
        with self.state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            if self.delay and self.cancelled.wait(self.delay):
                raise RuntimeError("aborted")
            if self.failure is not None:
                raise self.failure
            self.calls.append((pcm, options))
            return FakeNativeResult()
        finally:
            with self.state_lock:
                self.active -= 1

    def cancel(self):
        self.model.cancel_calls += 1
        self.cancelled.set()

    def close(self):
        self.closed = True
        self.model.close_order.append("session")


class FakeModel:
    def __init__(self, module, path, backend=None, device=None):
        self.module = module
        self.path = path
        if backend == "cpu_accel" and device is None:
            self.device = FakeDevice("cpu", "CPU", "cpu")
        else:
            self.device = device or FakeDevice(backend or "cpu", "CPU", "cpu")
        if self.device.kind == "cuda":
            self.backend = "CUDA0"
        else:
            self.backend = "CPU" if self.device.kind == "cpu" else self.device.kind
        self.arch = module.model_arch
        self.variant = "0.6b-v3"
        self.close_order = []
        self.closed = False
        self.cancel_calls = 0
        self.session_instance = None

    def session(self, **options):
        self.session_instance = FakeSession(self, options)
        return self.session_instance

    def close(self):
        self.closed = True
        self.close_order.append("model")


class FakeTranscribeModule:
    def __init__(self, devices=None, model_arch="parakeet-tdt"):
        self.devices = devices or [
            FakeDevice("cpu", "CPU", "cpu"),
            FakeDevice("cuda", "CUDA0", "0000:01:00.0"),
        ]
        self.models = []

        self.model_arch = model_arch
    def backends(self):
        return list(self.devices)

    def Model(self, path, backend=None, device=None):
        model = FakeModel(self, path, backend=backend, device=device)
        self.models.append(model)
        return model

    @staticmethod
    def native_version():
        return "0.2.1"

    @staticmethod
    def native_commit():
        return "test-commit"

    @staticmethod
    def native_provider():
        return "test-provider"

    @staticmethod
    def library_path():
        return "/runtime/libtranscribe.so"


class FakeEngineBackend:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.runtime_metadata = {"backend": "CUDA0"}
        self.result = FakeNativeResult()

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        return self.result

    def close(self):
        self.closed = True


class TranscribeCppEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "model.gguf"
        self.model_path.write_bytes(b"gguf-test-model")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_config(self, **changes):
        values = {
            "model": str(self.model_path),
            "device": "cuda",
            "gpu_device_index": 0,
            "engine_options": {
                "backend": "cuda",
            },
        }
        values.update(changes)
        return TranscriptionEngineConfig(**values)

    def make_backend(self, module=None, **config_changes):
        return TranscribeCppBackend(
            self.make_config(**config_changes),
            transcribe_module=module or FakeTranscribeModule(),
            numpy_module=np,
        )

    def test_factory_registers_lazy_engine_aliases(self):
        supported = get_supported_transcription_engines()
        self.assertIn("transcribe_cpp", supported)
        self.assertIn("parakeet_transcribe_cpp", supported)

        fake_backend = FakeEngineBackend()
        with patch(
            "RealtimeSTT.transcription_engines.transcribe_cpp_engine."
            "TranscribeCppBackend",
            return_value=fake_backend,
        ):
            engine = create_transcription_engine(
                "parakeet-transcribe-cpp",
                self.make_config(),
            )

        self.assertIsInstance(engine, TranscribeCppEngine)
        self.assertIs(engine.backend, fake_backend)

    def test_missing_python_binding_has_actionable_error(self):
        missing = ModuleNotFoundError(
            "No module named 'transcribe_cpp'",
            name="transcribe_cpp",
        )
        with patch(
            "RealtimeSTT.transcription_engines.transcribe_cpp_engine.import_module",
            side_effect=missing,
        ):
            with self.assertRaisesRegex(
                TranscriptionEngineError,
                r"RealtimeSTT\[transcribe-cpp\]",
            ):
                TranscribeCppBackend._load_transcribe_cpp()

    def test_resolves_relative_model_under_download_root(self):
        nested = Path(self.temp_dir.name) / "models"
        nested.mkdir()
        nested_model = nested / "parakeet.gguf"
        nested_model.write_bytes(b"model")

        backend = self.make_backend(
            model="parakeet.gguf",
            download_root=str(nested),
        )

        self.assertEqual(backend.model_path, nested_model.resolve())
        backend.close()

    def test_selects_exact_cuda_device_and_reuses_model_session(self):
        module = FakeTranscribeModule(
            [
                FakeDevice("cuda", "CUDA0", "0000:01:00.0"),
                FakeDevice("cuda", "CUDA1", "0000:02:00.0"),
                FakeDevice("cpu", "CPU", "cpu"),
            ]
        )
        config = self.make_config(gpu_device_index=1)
        backend = TranscribeCppBackend(
            config,
            transcribe_module=module,
            numpy_module=np,
        )

        first = backend.transcribe(np.array([0.0, 0.5], dtype=np.float32))
        second = backend.transcribe(
            np.array([0.25, -0.25], dtype=np.float32),
            language="auto",
        )

        self.assertEqual(first.text.strip(), "hello")
        self.assertEqual(second.text.strip(), "hello")
        self.assertEqual(len(module.models), 1)
        self.assertEqual(module.models[0].device.name, "CUDA1")
        self.assertEqual(
            module.models[0].session_instance.options,
            {"n_threads": 0, "kv_type": "auto", "n_ctx": 0},
        )
        self.assertEqual(
            backend.runtime_metadata["native_library"],
            "/runtime/libtranscribe.so",
        )
        calls = module.models[0].session_instance.calls
        self.assertEqual(len(calls), 2)
        self.assertIsInstance(calls[0][0], ctypes.Array)
        self.assertEqual(list(calls[0][0]), [0.0, 0.5])
        self.assertEqual(
            calls[0][1],
            {"timestamps": "none", "language": None},
        )
        backend.close()

    def test_refuses_cuda_fallback(self):
        module = FakeTranscribeModule([FakeDevice("cpu", "CPU", "cpu")])

        with self.assertRaisesRegex(TranscriptionEngineError, "Refusing to fall back"):
            self.make_backend(module=module)

    def test_accepts_real_cpu_accel_backend_report(self):
        config = self.make_config(
            device="cpu",
            engine_options={"backend": "cpu_accel"},
        )
        backend = TranscribeCppBackend(
            config,
            transcribe_module=FakeTranscribeModule(),
            numpy_module=np,
        )

        self.assertEqual(backend.model.device.kind, "cpu")
        self.assertEqual(backend.model.backend, "CPU")
        backend.close()

    def test_rejects_non_parakeet_model_family(self):
        module = FakeTranscribeModule(model_arch="whisper")
        with self.assertRaisesRegex(TranscriptionEngineError, "Parakeet models only"):
            self.make_backend(module=module)
        self.assertTrue(module.models[0].closed)

    def test_word_timestamps_and_metadata_are_mapped(self):
        fake_backend = FakeEngineBackend()
        engine = TranscribeCppEngine(self.make_config(), backend=fake_backend)

        result = engine.transcribe(
            np.array([0.0], dtype=np.float32),
            language="auto",
            word_timestamps=True,
        )

        self.assertEqual(result.text, "hello")
        self.assertIsNone(result.info.language)
        self.assertEqual(result.info.language_probability, 0.0)
        self.assertEqual(
            fake_backend.calls[0][1],
            {"language": "auto", "word_timestamps": True},
        )
        self.assertEqual(
            result.metadata["words"],
            [{"word": "hello", "start": 0.1, "end": 0.5}],
        )
        self.assertEqual(
            result.metadata["timings_ms"],
            {"load_ms": 1.0, "mel_ms": 2.0, "encode_ms": 3.0, "decode_ms": 4.0},
        )
        self.assertEqual(result.metadata["timestamp_kind"], "none")

    def test_speaker_segments_and_speaker_tagged_segments_are_mapped(self):
        fake_backend = FakeEngineBackend()
        fake_backend.result.timestamp_kind = "none"
        fake_backend.result.segments = (
            SimpleNamespace(
                text="speaker text",
                t0_ms=250,
                t1_ms=900,
                speaker_id=2,
            ),
        )
        fake_backend.result.speaker_segments = (
            SimpleNamespace(
                t0_ms=200,
                t1_ms=950,
                speaker_id=2,
                p=0.875,
            ),
            SimpleNamespace(
                t0_ms=950,
                t1_ms=1200,
                speaker_id=3,
                p=float("nan"),
            ),
        )
        engine = TranscribeCppEngine(self.make_config(), backend=fake_backend)

        result = engine.transcribe(np.array([0.0], dtype=np.float32))

        self.assertEqual(
            result.metadata["segments"],
            [
                {
                    "text": "speaker text",
                    "start": 0.25,
                    "end": 0.9,
                    "speaker_id": 2,
                }
            ],
        )
        self.assertEqual(
            result.metadata["speaker_segments"],
            [
                {
                    "start": 0.2,
                    "end": 0.95,
                    "speaker_id": 2,
                    "probability": 0.875,
                },
                {
                    "start": 0.95,
                    "end": 1.2,
                    "speaker_id": 3,
                    "probability": None,
                }
            ],
        )

    def test_explicit_language_is_not_reported_as_detected(self):
        fake_backend = FakeEngineBackend()
        engine = TranscribeCppEngine(self.make_config(), backend=fake_backend)

        result = engine.transcribe(
            np.array([0.0], dtype=np.float32),
            language="en",
        )

        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.info.language_probability, 0.0)

    def test_noncontiguous_and_readonly_pcm_are_made_native_safe(self):
        backend = self.make_backend()
        source = np.arange(8, dtype=np.float32)[::2]
        source.flags.writeable = False

        backend.transcribe(source)

        passed = backend.session.calls[0][0]
        self.assertIsInstance(passed, ctypes.Array)
        self.assertEqual(list(passed), [0.0, 2.0, 4.0, 6.0])
        backend.close()

    def test_concurrent_calls_are_serialized(self):
        backend = self.make_backend()
        backend.session.delay = 0.03
        errors = []

        def run():
            try:
                backend.transcribe(np.array([0.0], dtype=np.float32))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(backend.session.max_active, 1)
        backend.close()

    def test_native_errors_are_wrapped(self):
        backend = self.make_backend()
        backend.session.failure = RuntimeError("native failed")

        with self.assertRaisesRegex(
            TranscriptionEngineError,
            "transcription failed: native failed",
        ):
            backend.transcribe(np.array([0.0], dtype=np.float32))
        backend.close()

    def test_close_releases_session_before_model_and_is_idempotent(self):
        backend = self.make_backend()
        model = backend.model

        backend.close()
        backend.close()

        self.assertEqual(model.cancel_calls, 1)
        self.assertEqual(model.close_order, ["session", "model"])
        with self.assertRaisesRegex(TranscriptionEngineError, "is closed"):
            backend.transcribe(np.array([0.0], dtype=np.float32))

    def test_close_cancels_active_run_before_releasing_resources(self):
        backend = self.make_backend()
        backend.session.delay = 1.0
        errors = []

        def run():
            try:
                backend.transcribe(np.array([0.0], dtype=np.float32))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(backend.session.entered.wait(0.5))

        started = time.perf_counter()
        backend.close()
        elapsed = time.perf_counter() - started
        thread.join(timeout=0.5)

        self.assertFalse(thread.is_alive())
        self.assertLess(elapsed, 0.5)
        self.assertEqual(len(errors), 1)
        self.assertIn("transcription failed: aborted", str(errors[0]))
        self.assertEqual(backend.model.close_order, ["session", "model"])

    def test_cancel_active_aborts_only_current_run_and_reuses_session(self):
        backend = self.make_backend()
        backend.session.delay = 1.0
        errors = []

        def run():
            try:
                backend.transcribe(np.array([0.0], dtype=np.float32))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(backend.session.entered.wait(0.5))

        self.assertTrue(backend.cancel_active())
        thread.join(timeout=0.5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("transcription failed: aborted", str(errors[0]))

        backend.session.delay = 0.0
        result = backend.transcribe(np.array([0.25], dtype=np.float32))
        self.assertEqual(result.text.strip(), "hello")
        backend.close()

    def test_cancel_active_bridges_session_start_clear_race(self):
        backend = self.make_backend()
        backend.session.delay = 1.0
        backend.session.before_clear = threading.Event()
        backend.session.release_clear = threading.Event()
        cancel_event = threading.Event()
        errors = []

        def run():
            try:
                backend.transcribe(
                    np.array([0.0], dtype=np.float32),
                    _cancel_event=cancel_event,
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(backend.session.before_clear.wait(0.5))

        cancel_event.set()
        self.assertTrue(backend.cancel_active())
        self.assertTrue(backend.session.cancelled.wait(0.2))
        backend.session.release_clear.set()
        thread.join(timeout=0.5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("transcription failed: aborted", str(errors[0]))
        self.assertGreaterEqual(backend.model.cancel_calls, 2)

        backend.session.before_clear = None
        backend.session.release_clear = None
        backend.session.delay = 0.0
        result = backend.transcribe(np.array([0.25], dtype=np.float32))
        self.assertEqual(result.text.strip(), "hello")
        backend.close()

    def test_engine_does_not_mutate_process_tuning_environment(self):
        options = {
            "backend": "cuda",
        }
        config = self.make_config(engine_options=options)
        with patch.dict(os.environ, {}, clear=True):
            backend = TranscribeCppBackend(
                config,
                transcribe_module=FakeTranscribeModule(),
                numpy_module=np,
            )
            self.assertNotIn("TRANSCRIBE_NO_FLASH", os.environ)
            self.assertNotIn("OPENBLAS_NUM_THREADS", os.environ)
            self.assertNotIn("OMP_NUM_THREADS", os.environ)
            backend.close()

    def test_optional_model_digest_is_verified(self):
        digest = __import__("hashlib").sha256(self.model_path.read_bytes()).hexdigest()
        config = self.make_config(
            engine_options={
                "backend": "cuda",
                "model_sha256": digest,
            }
        )
        backend = TranscribeCppBackend(
            config,
            transcribe_module=FakeTranscribeModule(),
            numpy_module=np,
        )

        self.assertEqual(backend.model_sha256, digest)
        backend.close()

        bad_config = self.make_config(
            engine_options={
                "backend": "cuda",
                "model_sha256": "0" * 64,
            }
        )
        with self.assertRaisesRegex(TranscriptionEngineError, "does not match"):
            TranscribeCppBackend(
                bad_config,
                transcribe_module=FakeTranscribeModule(),
                numpy_module=np,
            )


if __name__ == "__main__":
    unittest.main()
