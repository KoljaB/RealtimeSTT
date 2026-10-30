"""Adapts transcribe.cpp models to the transcription engine interface."""

from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import threading
from importlib import import_module
from pathlib import Path

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_SESSION_THREADS = 0


def _normalize_language(language):
    """Returns a native language hint, using None for automatic detection."""

    if language is None:
        return None
    normalized = str(language).strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _finite_probability(value):
    """Returns a JSON-safe probability or None when it is unavailable."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return probability if math.isfinite(probability) else None


def _normalize_device_index(value):
    """Normalizes RealtimeSTT's scalar-or-list GPU index configuration."""

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise TranscriptionEngineError(
                "The 'transcribe_cpp' engine accepts exactly one GPU device index."
            )
        value = value[0]
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise TranscriptionEngineError(
            "The 'transcribe_cpp' GPU device index must be an integer."
        ) from exc
    if index < 0:
        raise TranscriptionEngineError(
            "The 'transcribe_cpp' GPU device index must be zero or greater."
        )
    return index


def _resolve_model_path(config):
    """Resolves an existing local GGUF without downloading model weights."""

    if not config.model:
        raise TranscriptionEngineError(
            "The 'transcribe_cpp' engine requires a local GGUF model path."
        )

    supplied = Path(config.model).expanduser()
    candidates = [supplied]
    if config.download_root and not supplied.is_absolute():
        candidates.insert(0, Path(config.download_root).expanduser() / supplied)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise TranscriptionEngineError(
        "The 'transcribe_cpp' model file was not found. Pass an existing local "
        f"GGUF path; checked: {checked}"
    )


def _verify_model_sha256(path, expected):
    """Verifies an optional model digest before native model loading."""

    if not expected:
        return None
    normalized = str(expected).strip().lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise TranscriptionEngineError(
            "transcription_engine_options['model_sha256'] must be 64 lowercase "
            "or uppercase hexadecimal characters."
        )

    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != normalized:
        raise TranscriptionEngineError(
            f"The 'transcribe_cpp' model SHA-256 does not match: expected "
            f"{normalized}, got {actual} for {path}"
        )
    return actual


class TranscribeCppBackend:
    """Owns one resident transcribe.cpp Model and Session."""

    def __init__(self, config, transcribe_module=None, numpy_module=None):
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._run_state_lock = threading.Lock()
        self._active_run_finished = None
        self._closed = False
        self._resources_closed = False

        self.numpy = numpy_module or import_module("numpy")
        self.transcribe_cpp = transcribe_module or self._load_transcribe_cpp()

        self.model_path = _resolve_model_path(config)
        self.model_sha256 = _verify_model_sha256(
            self.model_path,
            self.engine_options.get("model_sha256"),
        )
        self.backend_name, self.device_index = self._resolve_backend()
        try:
            self.model = self._load_model()
            self._verify_model_family()

            self._verify_loaded_backend()
            session_options = dict(self.engine_options.get("session", {}))
            session_options.setdefault(
                "n_threads",
                int(self.engine_options.get("n_threads", DEFAULT_SESSION_THREADS)),
            )
            session_options.setdefault("kv_type", "auto")
            session_options.setdefault("n_ctx", 0)
            self.session = self.model.session(**session_options)
        except Exception as exc:
            model = getattr(self, "model", None)
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass
            self._resources_closed = True
            if isinstance(exc, TranscriptionEngineError):
                raise
            raise TranscriptionEngineError(
                f"Failed to initialize the 'transcribe_cpp' model at "
                f"{self.model_path}: {exc}"
            ) from exc

        self.run_options = dict(self.engine_options.get("transcribe", {}))
        self.run_options.setdefault("timestamps", "none")
        self.runtime_metadata = self._build_runtime_metadata()

    @staticmethod
    def _load_transcribe_cpp():
        """Loads the optional binding while distinguishing provider failures."""

        try:
            return import_module("transcribe_cpp")
        except ModuleNotFoundError as exc:
            if exc.name == "transcribe_cpp":
                raise TranscriptionEngineError(
                    "The 'transcribe_cpp' engine requires transcribe-cpp 0.2.x. "
                    "Install the Python binding with "
                    "'pip install \"RealtimeSTT[transcribe-cpp]\"' and install "
                    "a matching CUDA native provider for GPU execution."
                ) from exc
            raise TranscriptionEngineError(
                "The transcribe-cpp Python package is installed, but one of its "
                f"native provider dependencies is missing: {exc}"
            ) from exc
        except Exception as exc:
            raise TranscriptionEngineError(
                "Failed to load the transcribe-cpp native provider. Verify that "
                "the binding and native provider have the same 0.2.x version and "
                f"that the CUDA runtime is available: {exc}"
            ) from exc

    def _resolve_backend(self):
        requested = str(
            self.engine_options.get("backend", self.config.device or "auto")
        ).strip().lower()
        device_index = self.engine_options.get(
            "device_index",
            self.config.gpu_device_index,
        )

        if requested.startswith("cuda:"):
            _, requested_index = requested.split(":", 1)
            requested = "cuda"
            device_index = requested_index
        requested = requested.replace("-", "_")
        if requested == "gpu":
            requested = "cuda"

        supported = {
            "auto",
            "cpu",
            "cpu_accel",
            "cuda",
            "metal",
            "rocm",
            "vulkan",
        }
        if requested not in supported:
            raise TranscriptionEngineError(
                f"Unsupported transcribe-cpp backend '{requested}'. Expected one "
                f"of: {', '.join(sorted(supported))}."
            )

        if requested in {"cuda", "rocm", "vulkan", "metal"}:
            device_index = _normalize_device_index(device_index)
        else:
            device_index = None
        return requested, device_index

    def _load_model(self):
        if self.backend_name not in {"cuda", "rocm", "vulkan", "metal"}:
            return self.transcribe_cpp.Model(
                str(self.model_path),
                backend=self.backend_name,
            )

        devices = [
            device
            for device in self.transcribe_cpp.backends()
            if str(getattr(device, "kind", "")).lower() == self.backend_name
        ]
        if self.device_index >= len(devices):
            raise TranscriptionEngineError(
                f"Requested transcribe-cpp {self.backend_name} device index "
                f"{self.device_index}, but only {len(devices)} matching device(s) "
                "were found. Refusing to fall back to CPU."
            )
        return self.transcribe_cpp.Model(
            str(self.model_path),
            device=devices[self.device_index],
        )

    def _verify_model_family(self):
        """Keeps the MVP honest: its run options are specific to Parakeet TDT."""

        architecture = str(getattr(self.model, "arch", "")).lower()
        if "parakeet" not in architecture:
            raise TranscriptionEngineError(
                "The RealtimeSTT 'transcribe_cpp' engine currently supports "
                f"Parakeet models only; loaded architecture={architecture or 'unknown'}."
            )

    def _verify_loaded_backend(self):
        if self.backend_name == "auto":
            return
        loaded_device = getattr(self.model, "device", None)
        loaded_kind = str(getattr(loaded_device, "kind", "")).lower()
        loaded_backend = str(getattr(self.model, "backend", "")).lower()
        if (
            self.backend_name == "cpu_accel"
            and loaded_kind == "cpu"
            and loaded_backend.startswith("cpu")
        ):
            return
        if (
            self.backend_name not in {loaded_kind, loaded_backend}
            and not loaded_backend.startswith(self.backend_name)
        ):
            raise TranscriptionEngineError(
                f"transcribe-cpp requested {self.backend_name}, but loaded "
                f"backend={loaded_backend or 'unknown'} and "
                f"device={loaded_kind or 'unknown'}. Refusing fallback."
            )

    def _build_runtime_metadata(self):
        model_device = getattr(self.model, "device", None)
        return {
            "engine": "transcribe_cpp",
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "model_arch": getattr(self.model, "arch", None),
            "model_variant": getattr(self.model, "variant", None),
            "backend": getattr(self.model, "backend", None),
            "device": {
                "name": getattr(model_device, "name", None),
                "description": getattr(model_device, "description", None),
                "kind": getattr(model_device, "kind", None),
                "device_type": getattr(model_device, "device_type", None),
                "device_id": getattr(model_device, "device_id", None),
            },
            "native_version": self._optional_native_value("native_version"),
            "native_commit": self._optional_native_value("native_commit"),
            "native_provider": self._optional_native_value("native_provider"),
            "native_library": self._optional_native_value("library_path"),
        }

    def _optional_native_value(self, name):
        value = getattr(self.transcribe_cpp, name, None)
        if callable(value):
            try:
                return value()
            except Exception:
                return None
        return value

    def _prepare_pcm(self, audio):
        """Returns owned float32 PCM and a zero-copy ctypes view for run()."""

        pcm = self.numpy.asarray(audio, dtype=self.numpy.float32)
        if pcm.ndim != 1:
            raise TranscriptionEngineError(
                f"transcribe-cpp requires mono 1-D PCM; received shape {pcm.shape}."
            )
        if pcm.size == 0:
            raise TranscriptionEngineError(
                "transcribe-cpp received an empty PCM buffer."
            )
        if not pcm.flags.c_contiguous or not pcm.flags.writeable:
            pcm = self.numpy.array(
                pcm,
                dtype=self.numpy.float32,
                order="C",
                copy=True,
            )
        array_type = ctypes.c_float * int(pcm.size)
        return pcm, array_type.from_buffer(pcm)

    def transcribe(self, audio, language=None, word_timestamps=False, **options):
        """Runs one serialized inference against the resident native session."""

        cancel_event = options.pop("_cancel_event", None)
        pcm, pcm_view = self._prepare_pcm(audio)
        run_options = dict(self.run_options)
        run_options.update(options)
        run_options["language"] = _normalize_language(language)
        if word_timestamps:
            run_options["timestamps"] = "word"

        try:
            with self._lock:
                if self._closed:
                    raise TranscriptionEngineError(
                        "The 'transcribe_cpp' engine is closed."
                    )
                if cancel_event is not None and cancel_event.is_set():
                    raise TranscriptionEngineError(
                        "transcribe-cpp transcription was cancelled before start"
                    )
                run_finished = threading.Event()
                with self._run_state_lock:
                    self._active_run_finished = run_finished
                try:
                    # Cancellation may have arrived after the first check but
                    # before this run became visible to cancel_active().
                    if cancel_event is not None and cancel_event.is_set():
                        raise TranscriptionEngineError(
                            "transcribe-cpp transcription was cancelled before start"
                        )
                    result = self.session.run(pcm_view, **run_options)
                finally:
                    with self._run_state_lock:
                        if self._active_run_finished is run_finished:
                            self._active_run_finished = None
                    run_finished.set()
            return result
        except TranscriptionEngineError:
            raise
        except Exception as exc:
            raise TranscriptionEngineError(
                f"transcribe-cpp transcription failed: {exc}"
            ) from exc
        finally:
            # Keep the NumPy owner alive until the native call has returned.
            del pcm

    def cancel_active(self):
        """Cooperatively abort the current native run without closing the session.

        Do not take ``_lock`` here: the run that needs interrupting owns it
        until native inference returns.
        """

        with self._close_lock:
            if self._closed:
                return False
            with self._run_state_lock:
                run_finished = self._active_run_finished
            if run_finished is None or run_finished.is_set():
                return False
            session = getattr(self, "session", None)
            cancel = getattr(session, "cancel", None)
            if not callable(cancel):
                return False
            cancel()

            def repeat_until_run_observes_cancel():
                # Session.run() clears its per-session flag on entry. A
                # second pulse after that clear closes the start race while
                # this exact run's Event prevents poisoning the next run.
                while not run_finished.wait(0.001):
                    try:
                        cancel()
                    except Exception:
                        logging.debug(
                            "Could not repeat transcribe-cpp cancellation.",
                            exc_info=True,
                        )
                        return

            threading.Thread(
                target=repeat_until_run_observes_cancel,
                name="RealtimeSTT-transcribe-cpp-cancel",
                daemon=True,
            ).start()
            return True

    def close(self):
        """Releases the native session before its owning model."""

        with self._close_lock:
            if self._resources_closed:
                return
            self._closed = True
            session = getattr(self, "session", None)
            cancel = getattr(session, "cancel", None)
            if cancel is not None:
                try:
                    cancel()
                except Exception:
                    logging.debug(
                        "Ignoring transcribe-cpp cancellation failure.",
                        exc_info=True,
                    )

            # cancel() is thread-safe in transcribe-cpp 0.2.1. It releases an
            # active native run so shutdown does not wait for a long utterance.
            with self._lock:
                self._resources_closed = True
                model = getattr(self, "model", None)
                try:
                    if session is not None:
                        session.close()
                finally:
                    if model is not None:
                        model.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logging.debug(
                "Ignoring transcribe-cpp cleanup failure during interpreter shutdown.",
                exc_info=True,
            )


class TranscribeCppEngine(BaseTranscriptionEngine):
    """Transcribes in-memory PCM with a resident transcribe.cpp backend."""

    engine_name = "transcribe_cpp"

    def __init__(self, config, backend=None, backend_cls=None):
        super().__init__(config)
        self.backend = backend or (backend_cls or TranscribeCppBackend)(config)

    def transcribe(
        self,
        audio,
        language=None,
        use_prompt=True,
        word_timestamps=False,
        **options,
    ):
        """Transcribes audio and converts the native result contract."""

        del use_prompt  # Parakeet TDT has no initial-prompt decode path.
        audio = self._normalize_audio(audio)
        native_result = self.backend.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            **options,
        )

        text = str(getattr(native_result, "text", "")).strip()
        requested_language = _normalize_language(language)
        native_language = _normalize_language(
            getattr(native_result, "language", None)
        )
        if not text:
            result_language = None
        elif native_language:
            result_language = native_language
        else:
            result_language = requested_language
        language_probability = 0.0

        metadata = dict(getattr(self.backend, "runtime_metadata", {}) or {})
        timings = getattr(native_result, "timings", None)
        if timings is not None:
            metadata["timings_ms"] = {
                name: float(getattr(timings, name, 0.0))
                for name in ("load_ms", "mel_ms", "encode_ms", "decode_ms")
            }
        timestamp_kind = str(
            getattr(native_result, "timestamp_kind", "none") or "none"
        ).lower()
        metadata["timestamp_kind"] = timestamp_kind
        native_words = tuple(getattr(native_result, "words", None) or ())
        if timestamp_kind == "word" or word_timestamps or native_words:
            metadata["words"] = [
                {
                    "word": getattr(word, "text", ""),
                    "start": float(getattr(word, "t0_ms", 0)) / 1000.0,
                    "end": float(getattr(word, "t1_ms", 0)) / 1000.0,
                }
                for word in native_words
            ]
        if timestamp_kind == "token":
            metadata["tokens"] = [
                {
                    "token": getattr(token, "text", ""),
                    "id": int(getattr(token, "id", 0)),
                    "probability": float(getattr(token, "p", 0.0)),
                    "start": float(getattr(token, "t0_ms", 0)) / 1000.0,
                    "end": float(getattr(token, "t1_ms", 0)) / 1000.0,
                }
                for token in (getattr(native_result, "tokens", None) or ())
            ]
        native_segments = tuple(
            getattr(native_result, "segments", None) or ()
        )
        if timestamp_kind in {"segment", "word", "token"} or native_segments:
            metadata["segments"] = [
                {
                    "text": getattr(segment, "text", ""),
                    "start": float(getattr(segment, "t0_ms", 0)) / 1000.0,
                    "end": float(getattr(segment, "t1_ms", 0)) / 1000.0,
                    "speaker_id": int(getattr(segment, "speaker_id", -1)),
                }
                for segment in native_segments
            ]
        native_speaker_segments = tuple(
            getattr(native_result, "speaker_segments", None) or ()
        )
        if native_speaker_segments:
            metadata["speaker_segments"] = [
                {
                    "start": float(getattr(segment, "t0_ms", 0)) / 1000.0,
                    "end": float(getattr(segment, "t1_ms", 0)) / 1000.0,
                    "speaker_id": int(getattr(segment, "speaker_id", -1)),
                    "probability": _finite_probability(getattr(segment, "p", None)),
                }
                for segment in native_speaker_segments
            ]

        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=result_language,
                language_probability=language_probability,
            ),
            metadata=metadata,
        )

    def transcribe_cancellable(
        self,
        audio,
        language=None,
        use_prompt=True,
        word_timestamps=False,
        *,
        cancel_event,
        **options,
    ):
        """Transcribe while forwarding one request's cancellation token."""

        options["_cancel_event"] = cancel_event
        return self.transcribe(
            audio,
            language=language,
            use_prompt=use_prompt,
            word_timestamps=word_timestamps,
            **options,
        )

    def cancel_active(self):
        """Abort only the backend run currently using this engine."""

        cancel = getattr(self.backend, "cancel_active", None)
        return bool(callable(cancel) and cancel())

    def close(self):
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()
