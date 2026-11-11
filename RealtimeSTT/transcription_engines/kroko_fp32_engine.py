"""Resident adapter for the optimized English Kroko FP32 CPU executable."""

from collections import deque
from importlib import import_module
import itertools
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


_MODEL_FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")
_OPTIONS = {
    "executable",
    "encoder_threads",
    "feature_workers",
    "startup_timeout",
    "request_timeout",
    "max_audio_seconds",
    "strict_length",
}


def _positive_int(options, name, default, maximum=64):
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise TranscriptionEngineError(
            f"kroko_fp32 option {name} must be an integer between 1 and {maximum}"
        )
    return value


def _positive_float(options, name, default, maximum=None):
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptionEngineError(f"kroko_fp32 option {name} must be a positive number")
    value = float(value)
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise TranscriptionEngineError(
            f"kroko_fp32 option {name} must be positive and{suffix}"
        )
    return value


class KrokoFp32Engine(BaseTranscriptionEngine):
    """Use one warm native Kroko process for serialized English transcription."""

    engine_name = "kroko_fp32"

    def __init__(self, config, *, popen_factory=None, numpy_module=None):
        super().__init__(config)
        if str(config.device).strip().lower() != "cpu":
            raise TranscriptionEngineError("kroko_fp32 requires device=cpu")

        options = dict(config.engine_options or {})
        unknown = set(options) - _OPTIONS
        if unknown:
            raise TranscriptionEngineError(f"Unknown kroko_fp32 options: {sorted(unknown)}")

        strict_length = options.get("strict_length", False)
        if not isinstance(strict_length, bool):
            raise TranscriptionEngineError("kroko_fp32 option strict_length must be a boolean")
        self.encoder_threads = _positive_int(options, "encoder_threads", 4)
        self.feature_workers = _positive_int(options, "feature_workers", 1)
        self.startup_timeout = _positive_float(options, "startup_timeout", 15.0)
        self.request_timeout = _positive_float(options, "request_timeout", 60.0)
        self.max_audio_seconds = _positive_float(
            options, "max_audio_seconds", 120.0, maximum=300.0
        )
        self.strict_length = strict_length

        self.model_path = Path(config.model).expanduser().resolve()
        for filename in _MODEL_FILES:
            path = self.model_path / filename
            if not path.is_file() or not path.stat().st_size:
                raise TranscriptionEngineError(f"Missing Kroko FP32 model file: {path}")

        self.executable = self._resolve_executable(options.get("executable"))
        try:
            self.np = numpy_module or import_module("numpy")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError("kroko_fp32 requires numpy") from exc

        self._popen = popen_factory or subprocess.Popen
        self._request_ids = itertools.count(1)
        self._request_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._cancel_sent_request_id = None
        self._state_lock = threading.RLock()
        self._process = None
        self._responses = None
        self._stderr_tail = deque(maxlen=20)
        self._active_request_id = None
        self._active_cancel_event = None
        self._closed = False
        self.runtime_metadata = {}
        self.backend = SimpleNamespace(runtime_metadata=self.runtime_metadata)
        self._start_process()

    def _resolve_executable(self, configured):
        if configured is not None:
            if not isinstance(configured, (str, os.PathLike)) or not str(configured).strip():
                raise TranscriptionEngineError(
                    "kroko_fp32 option executable must be a nonempty path"
                )
            candidate = Path(configured).expanduser().resolve()
            if not candidate.is_file():
                raise TranscriptionEngineError(f"Missing Kroko FP32 executable: {candidate}")
            return candidate

        binary_name = "kroko-cli.exe" if os.name == "nt" else "kroko-cli"
        adjacent = self.model_path.parent / "bin" / binary_name
        if adjacent.is_file():
            return adjacent.resolve()
        discovered = shutil.which(binary_name)
        if discovered:
            return Path(discovered).resolve()
        raise TranscriptionEngineError(
            "Could not find kroko-cli; set kroko_fp32 engine option 'executable'"
        )

    def _command(self):
        command = [
            str(self.executable),
            "--stdio-server",
            "--model-dir",
            str(self.model_path),
            "--encoder-threads",
            str(self.encoder_threads),
            "--feature-workers",
            str(self.feature_workers),
        ]
        if self.strict_length:
            command.append("--strict-length")
        return command

    @staticmethod
    def _read_stdout(stream, responses):
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                responses.put(line)
        finally:
            responses.put(None)

    @staticmethod
    def _read_stderr(stream, stderr_tail):
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                stderr_tail.append(line.decode("utf-8", errors="replace").strip())
        except Exception:
            return

    def _stderr_detail(self):
        with self._state_lock:
            lines = [line for line in self._stderr_tail if line]
        return f" Native stderr: {' | '.join(lines)}" if lines else ""

    @staticmethod
    def _decode_response(raw):
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptionEngineError(
                f"Invalid JSON from kroko_fp32 process: {raw[:200]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise TranscriptionEngineError("Invalid non-object response from kroko_fp32 process")
        return payload

    def _next_response(self, responses, timeout, process):
        try:
            raw = responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TranscriptionEngineError(
                f"kroko_fp32 process timed out after {timeout:.3f}s{self._stderr_detail()}"
            ) from exc
        if raw is None:
            code = process.poll()
            raise TranscriptionEngineError(
                f"kroko_fp32 process exited unexpectedly (code {code}){self._stderr_detail()}"
            )
        return self._decode_response(raw)

    def _start_process(self):
        with self._state_lock:
            if self._closed:
                raise TranscriptionEngineError("kroko_fp32 is closed")

        responses = queue.Queue()
        stderr_tail = deque(maxlen=20)
        try:
            process = self._popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            raise TranscriptionEngineError(
                f"Could not start Kroko FP32 executable {self.executable}: {exc}"
            ) from exc
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._terminate_process(process, graceful=False)
            raise TranscriptionEngineError("kroko_fp32 process did not expose stdio pipes")

        with self._state_lock:
            if self._closed:
                self._terminate_process(process, graceful=False)
                raise TranscriptionEngineError("kroko_fp32 is closed")
            self._process = process
            self._responses = responses
            self._stderr_tail = stderr_tail

        threading.Thread(
            target=self._read_stdout,
            args=(process.stdout, responses),
            name="KrokoFp32Stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(process.stderr, stderr_tail),
            name="KrokoFp32Stderr",
            daemon=True,
        ).start()

        try:
            ready = self._next_response(responses, self.startup_timeout, process)
            if ready.get("type") != "ready" or ready.get("protocol") != 1:
                raise TranscriptionEngineError(
                    f"Unexpected kroko_fp32 startup response: {ready!r}"
                )
        except Exception:
            self._discard_process(process)
            raise

        self.runtime_metadata.clear()
        self.runtime_metadata.update(
            {
                **ready,
                "engine": self.engine_name,
                "provider": "cpu",
                "model_path": str(self.model_path),
                "executable": str(self.executable),
                "transport": "stdio_float32le",
            }
        )

    def _terminate_process(self, process, *, graceful):
        if process is None:
            return
        if graceful and process.poll() is None:
            try:
                process.stdin.write(b"shutdown\n")
                process.stdin.flush()
                process.wait(timeout=1.0)
            except Exception:
                pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except Exception:
                    pass
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _discard_process(self, process):
        with self._state_lock:
            if self._process is process:
                self._process = None
                self._responses = None
        self._terminate_process(process, graceful=False)

    def _ensure_process(self):
        with self._state_lock:
            if self._closed:
                raise TranscriptionEngineError("kroko_fp32 is closed")
            process = self._process
            if process is not None and process.poll() is None:
                return process, self._responses
        if process is not None:
            self._discard_process(process)
        self._start_process()
        with self._state_lock:
            return self._process, self._responses

    @staticmethod
    def _validate_language(language):
        normalized = str(language or "en").strip().lower().replace("_", "-")
        if normalized == "english":
            normalized = "en"
        if normalized != "en" and not normalized.startswith("en-"):
            raise TranscriptionEngineError(
                "kroko_fp32 is English-only; use language='en'"
            )
        return "en"

    def _wait_for_request(self, responses, process, request_id, cancel_event):
        deadline = time.monotonic() + self.request_timeout
        while True:
            if cancel_event.is_set():
                self.cancel_active()
                if self.runtime_metadata.get("cancellation") != "request_id_run_options":
                    raise TranscriptionEngineError("Kroko FP32 transcription cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._discard_process(process)
                raise TranscriptionEngineError(
                    f"kroko_fp32 request timed out after {self.request_timeout:.3f}s"
                )
            try:
                raw = responses.get(timeout=min(0.02, remaining))
            except queue.Empty:
                continue
            if raw is None:
                code = process.poll()
                self._discard_process(process)
                raise TranscriptionEngineError(
                    f"kroko_fp32 process exited unexpectedly (code {code}){self._stderr_detail()}"
                )
            try:
                response = self._decode_response(raw)
                if response.get("id") != request_id:
                    raise TranscriptionEngineError(
                        f"kroko_fp32 response id mismatch: expected {request_id}, got {response.get('id')}"
                    )
                if response.get("type") not in ("result", "error", "cancelled"):
                    raise TranscriptionEngineError(
                        f"Unexpected kroko_fp32 response: {response!r}"
                    )
            except TranscriptionEngineError:
                # A broken protocol cannot safely associate later replies with requests.
                self._discard_process(process)
                raise
            if cancel_event.is_set() or response.get("type") == "cancelled":
                raise TranscriptionEngineError("Kroko FP32 transcription cancelled")
            return response

    def transcribe_cancellable(
        self, audio, language=None, use_prompt=True, *, cancel_event, **kwargs
    ):
        return self.transcribe(
            audio,
            language=language,
            use_prompt=use_prompt,
            _cancel_event=cancel_event,
            **kwargs,
        )

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        if kwargs.get("word_timestamps"):
            raise TranscriptionEngineError("kroko_fp32 does not expose word timestamps")
        language = self._validate_language(language)
        cancel_event = kwargs.get("_cancel_event") or threading.Event()
        if cancel_event.is_set():
            raise TranscriptionEngineError("Kroko FP32 transcription cancelled")

        audio = self.np.asarray(audio, dtype=self.np.float32)
        if audio.ndim != 1 or not audio.size or not self.np.isfinite(audio).all():
            raise TranscriptionEngineError(
                "Expected nonempty finite mono float32 PCM at 16000 Hz"
            )
        if audio.size > int(self.max_audio_seconds * 16000):
            raise TranscriptionEngineError(
                f"kroko_fp32 audio exceeds {self.max_audio_seconds:g} seconds"
            )
        audio = self.np.ascontiguousarray(self._normalize_audio(audio), dtype="<f4")

        with self._request_lock:
            process, responses = self._ensure_process()
            request_id = next(self._request_ids)
            try:
                # Publish the active id and its complete audio packet before a
                # concurrent cancellation can write its request-scoped command.
                with self._write_lock:
                    if cancel_event.is_set():
                        raise TranscriptionEngineError("Kroko FP32 transcription cancelled")
                    with self._state_lock:
                        self._active_request_id = request_id
                        self._active_cancel_event = cancel_event
                        self._cancel_sent_request_id = None
                    process.stdin.write(
                        f"transcribe {request_id} {audio.size}\n".encode("ascii")
                    )
                    process.stdin.write(audio.tobytes(order="C"))
                    process.stdin.flush()
                response = self._wait_for_request(
                    responses, process, request_id, cancel_event
                )
            except (BrokenPipeError, OSError) as exc:
                self._discard_process(process)
                raise TranscriptionEngineError(
                    f"Could not communicate with kroko_fp32 process: {exc}{self._stderr_detail()}"
                ) from exc
            finally:
                with self._state_lock:
                    if self._active_request_id == request_id:
                        self._active_request_id = None
                        self._active_cancel_event = None

        if response.get("type") == "error":
            raise TranscriptionEngineError(
                f"Kroko FP32 inference failed: {response.get('error', 'unknown native error')}"
            )
        metadata = {
            **self.runtime_metadata,
            **{
                key: value
                for key, value in response.items()
                if key not in {"type", "id", "transcript", "tokens", "token_encoder_frames"}
            },
            "timestamp_kind": "none",
        }
        return TranscriptionResult(
            text=str(response.get("transcript") or "").strip(),
            info=TranscriptionInfo(language=language, language_probability=1.0),
            metadata=metadata,
        )

    def cancel_active(self):
        with self._write_lock:
            with self._state_lock:
                request_id = self._active_request_id
                if request_id is None:
                    return False
                if self._active_cancel_event is not None:
                    self._active_cancel_event.set()
                process = self._process
                cooperative = self.runtime_metadata.get("cancellation") == "request_id_run_options"
                if cooperative:
                    if self._cancel_sent_request_id == request_id:
                        return True
                    self._cancel_sent_request_id = request_id
                else:
                    # Older binaries must never be reused while SIGTERM is pending.
                    self._process = None
                    self._responses = None
            if process is not None and process.poll() is None:
                try:
                    if cooperative:
                        process.stdin.write(f"cancel {request_id}\n".encode("ascii"))
                        process.stdin.flush()
                    else:
                        process.terminate()
                except Exception:
                    self._discard_process(process)
        return True

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            active = self._active_request_id is not None
            if self._active_cancel_event is not None:
                self._active_cancel_event.set()
            process = self._process
            self._process = None
            self._responses = None
        self._terminate_process(process, graceful=not active)
