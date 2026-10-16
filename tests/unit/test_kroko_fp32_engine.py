import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from RealtimeSTT.transcription_engines.base import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
)
from RealtimeSTT.transcription_engines.factory import get_supported_transcription_engines
from RealtimeSTT.transcription_engines.kroko_fp32_engine import KrokoFp32Engine


READY = {
    "type": "ready",
    "protocol": 1,
    "language": "en",
    "sample_rate": 16000,
    "sample_format": "float32le",
    "runtime": "C++ CPU only",
    "ort_version": "test",
    "model_and_frontend_load_ms": 700.0,
}


class FakeReadable:
    def __init__(self, initial=()):
        self.items = queue.Queue()
        for item in initial:
            self.items.put(item)

    def readline(self):
        item = self.items.get(timeout=2.0)
        return item

    def push(self, item):
        self.items.put(item)

    def close(self):
        self.items.put(b"")


class FakeStdin:
    def __init__(self, process):
        self.process = process
        self.buffer = bytearray()
        self.closed = False

    def write(self, data):
        if self.closed or self.process.returncode is not None:
            raise BrokenPipeError("closed")
        self.buffer.extend(data)
        return len(data)

    def flush(self):
        if self.buffer == b"shutdown\n":
            self.buffer.clear()
            self.process.finish(0)
            return
        newline = self.buffer.find(b"\n")
        if newline < 0:
            return
        header = bytes(self.buffer[:newline]).decode("ascii").split()
        if len(header) == 2 and header[0] == "cancel":
            del self.buffer[:newline + 1]
            self.process.stdout.push((json.dumps({"type": "cancelled", "id": int(header[1])}) + "\n").encode())
            return
        if len(header) != 3 or header[0] != "transcribe":
            return
        request_id, sample_count = int(header[1]), int(header[2])
        payload_size = sample_count * 4
        if len(self.buffer) < newline + 1 + payload_size:
            return
        payload = bytes(self.buffer[newline + 1 : newline + 1 + payload_size])
        del self.buffer[: newline + 1 + payload_size]
        self.process.request_audio = np.frombuffer(payload, dtype="<f4").copy()
        self.process.request_seen.set()
        if self.process.respond:
            result = {
                "type": "result",
                "id": request_id,
                "transcript": " Please schedule the meeting. ",
                "audio_seconds": sample_count / 16000.0,
                "model_input_seconds": sample_count / 16000.0,
                "tail_padding_ms": 0.0,
                "inference_ms": 50.1,
                "features_ms": 0.7,
                "encoder_ms": 47.0,
                "decoder_ms": 2.4,
                "tokens": [2, 3],
                "token_encoder_frames": [1, 2],
            }
            self.process.stdout.push((json.dumps(result) + "\n").encode("utf-8"))

    def close(self):
        self.closed = True


class FakeProcess:
    next_pid = 100

    def __init__(self, command, respond=True, cooperative=False):
        self.command = command
        self.respond = respond
        self.returncode = None
        self.request_audio = None
        self.request_seen = threading.Event()
        ready = {**READY, **({"cancellation": "request_id_run_options"} if cooperative else {})}
        self.stdout = FakeReadable(((json.dumps(ready) + "\n").encode("utf-8"),))
        self.stderr = FakeReadable((b"",))
        self.stdin = FakeStdin(self)
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise TimeoutError("still running")
        return self.returncode

    def finish(self, code):
        if self.returncode is None:
            self.returncode = code
            self.stdout.close()

    def terminate(self):
        self.finish(-15)

    def kill(self):
        self.finish(-9)


class FakePopenFactory:
    def __init__(self, responses=None, cooperative=False):
        self.cooperative = cooperative
        self.responses = list(responses or [True])
        self.processes = []

    def __call__(self, command, **kwargs):
        respond = self.responses.pop(0) if self.responses else True
        process = FakeProcess(command, respond=respond, cooperative=self.cooperative)
        self.processes.append(process)
        return process


class KrokoFp32EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "models"
        self.model.mkdir()
        for filename in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
            (self.model / filename).write_bytes(b"x")
        self.executable = self.root / "kroko-cli"
        self.executable.write_bytes(b"binary")

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **options):
        return TranscriptionEngineConfig(
            model=str(self.model),
            device="cpu",
            engine_options={"executable": str(self.executable), **options},
        )

    def test_resident_process_transcribes_and_reports_native_timing(self):
        factory = FakePopenFactory()
        engine = KrokoFp32Engine(
            self.config(encoder_threads=7, feature_workers=1),
            popen_factory=factory,
        )

        result = engine.transcribe(np.zeros(16000, dtype=np.float32), language="en")

        self.assertEqual(result.text, "Please schedule the meeting.")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.info.language_probability, 1.0)
        self.assertEqual(result.metadata["inference_ms"], 50.1)
        self.assertEqual(result.metadata["transport"], "stdio_float32le")
        self.assertEqual(factory.processes[0].request_audio.shape, (16000,))
        self.assertIn("--stdio-server", factory.processes[0].command)
        self.assertIn("7", factory.processes[0].command)
        self.assertIn("kroko_fp32", get_supported_transcription_engines())
        self.assertIn("kroko_native", get_supported_transcription_engines())
        engine.close()
        self.assertEqual(factory.processes[0].returncode, 0)

    def test_rejects_non_english_invalid_audio_and_unknown_options(self):
        engine = KrokoFp32Engine(self.config(), popen_factory=FakePopenFactory())
        for language in ("auto", "de", "fr"):
            with self.assertRaisesRegex(TranscriptionEngineError, "English-only"):
                engine.transcribe(np.zeros(160, dtype=np.float32), language=language)
        for audio in ([], [[0.0]], [np.nan]):
            with self.assertRaises(TranscriptionEngineError):
                engine.transcribe(audio, language="en")
        engine.close()

        with self.assertRaisesRegex(TranscriptionEngineError, "Unknown kroko_fp32"):
            KrokoFp32Engine(
                self.config(beam=1),
                popen_factory=FakePopenFactory(),
            )

    def test_en_variants_are_normalized_and_duration_is_bounded(self):
        engine = KrokoFp32Engine(
            self.config(max_audio_seconds=0.01),
            popen_factory=FakePopenFactory(),
        )
        self.assertEqual(
            engine.transcribe(np.zeros(160), language="en-US").info.language,
            "en",
        )
        with self.assertRaisesRegex(TranscriptionEngineError, "exceeds"):
            engine.transcribe(np.zeros(161), language="en")
        engine.close()

    def test_active_cancel_terminates_process_and_successor_restarts(self):
        factory = FakePopenFactory(responses=[False, True])
        engine = KrokoFp32Engine(self.config(), popen_factory=factory)
        errors = []

        def transcribe():
            try:
                engine.transcribe(np.zeros(160), language="en")
            except TranscriptionEngineError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=transcribe)
        thread.start()
        self.assertTrue(factory.processes[0].request_seen.wait(1.0))
        self.assertTrue(engine.cancel_active())
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertEqual(
            engine.transcribe(np.zeros(160), language="en").text,
            "Please schedule the meeting.",
        )
        self.assertEqual(len(factory.processes), 2)
        engine.close()

    def test_cooperative_cancel_preserves_process_and_hot_successor(self):
        factory = FakePopenFactory(responses=[False], cooperative=True)
        engine = KrokoFp32Engine(self.config(), popen_factory=factory)
        errors = []
        def transcribe():
            try:
                engine.transcribe(np.zeros(160), language="en")
            except TranscriptionEngineError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=transcribe)
        thread.start()
        try:
            self.assertTrue(factory.processes[0].request_seen.wait(1))
            self.assertTrue(engine.cancel_active())
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["Kroko FP32 transcription cancelled"])
            self.assertIsNone(factory.processes[0].returncode)
            factory.processes[0].respond = True
            self.assertEqual(engine.transcribe(np.zeros(160), language="en").text, "Please schedule the meeting.")
            self.assertEqual(len(factory.processes), 1)
            self.assertFalse(engine.cancel_active())
        finally:
            engine.close()
            thread.join(1)

    def test_cancel_during_process_start_keeps_new_model_without_submitting(self):
        factory = FakePopenFactory(cooperative=True)
        engine = KrokoFp32Engine(self.config(), popen_factory=factory)
        cancel = threading.Event()
        original = engine._ensure_process
        def start_then_cancel():
            result = original()
            cancel.set()
            return result
        engine._ensure_process = start_then_cancel
        try:
            with self.assertRaisesRegex(TranscriptionEngineError, "cancelled"):
                engine.transcribe_cancellable(np.zeros(160), language="en", cancel_event=cancel)
            self.assertFalse(factory.processes[0].request_seen.is_set())
            self.assertIsNone(factory.processes[0].returncode)
        finally:
            engine.close()

    def test_protocol_failure_discards_process_and_successor_restarts(self):
        invalid_responses = (
            b"not json\n", b"\xff\n", b"[]\n",
            b'{"type":"result","id":999}\n',
            b'{"type":"unexpected","id":1}\n',
            b'{"type":[],"id":1}\n', b"",
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                factory = FakePopenFactory()
                engine = KrokoFp32Engine(self.config(), popen_factory=factory)
                failed_process = factory.processes[0]
                try:
                    failed_process.stdout.push(response)
                    with self.assertRaises(TranscriptionEngineError):
                        engine.transcribe(np.zeros(160), language="en")
                    self.assertIsNotNone(failed_process.returncode)
                    self.assertEqual(
                        engine.transcribe(np.zeros(160), language="en").text,
                        "Please schedule the meeting.",
                    )
                    self.assertEqual(len(factory.processes), 2)
                finally:
                    engine.close()

    def test_write_failure_discards_process_and_successor_restarts(self):
        factory = FakePopenFactory()
        engine = KrokoFp32Engine(self.config(), popen_factory=factory)
        failed_process = factory.processes[0]
        try:
            failed_process.stdin.closed = True
            with self.assertRaisesRegex(TranscriptionEngineError, "Could not communicate"):
                engine.transcribe(np.zeros(160), language="en")
            self.assertIsNotNone(failed_process.returncode)
            self.assertEqual(
                engine.transcribe(np.zeros(160), language="en").text,
                "Please schedule the meeting.",
            )
            self.assertEqual(len(factory.processes), 2)
        finally:
            engine.close()

    def test_native_inference_error_keeps_resident_process(self):
        factory = FakePopenFactory(responses=[False])
        engine = KrokoFp32Engine(self.config(), popen_factory=factory)
        process = factory.processes[0]
        try:
            process.stdout.push(b'{"type":"error","id":1,"error":"test failure"}\n')
            with self.assertRaisesRegex(TranscriptionEngineError, "inference failed: test failure"):
                engine.transcribe(np.zeros(160), language="en")
            self.assertIsNone(process.returncode)
            process.respond = True
            self.assertEqual(
                engine.transcribe(np.zeros(160), language="en").text,
                "Please schedule the meeting.",
            )
            self.assertEqual(len(factory.processes), 1)
        finally:
            engine.close()

    def test_requires_cpu_model_files_and_executable(self):
        config = self.config()
        config.device = "cuda"
        with self.assertRaisesRegex(TranscriptionEngineError, "device=cpu"):
            KrokoFp32Engine(config, popen_factory=FakePopenFactory())

        (self.model / "joiner.onnx").unlink()
        with self.assertRaisesRegex(TranscriptionEngineError, "joiner.onnx"):
            KrokoFp32Engine(self.config(), popen_factory=FakePopenFactory())


if __name__ == "__main__":
    unittest.main()
