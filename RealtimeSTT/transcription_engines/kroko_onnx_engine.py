import os
import re
import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from ._model_utils import text_from_output
from .base import (
    BaseTranscriptionEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_KROKO_ONNX_MODEL = "Kroko-EN-Community-64-L-Streaming-001.data"
KROKO_ONNX_HF_REPO = "Banafo/Kroko-ASR"
KROKO_ONNX_MODEL_URL = "https://huggingface.co/Banafo/Kroko-ASR"
KROKO_ONNX_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "realtimestt" / "kroko-onnx"
KROKO_ONNX_PUBLIC_MODELS = {
    "Kroko-EN-Community-64-L-Streaming-001.data",
    "Kroko-EN-Community-128-L-Streaming-001.data",
}
KROKO_ONNX_LICENSE_QUIET_ENV = "KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT"
KROKO_ONNX_FALLBACK_TAIL_PADDING_SECONDS = 0.66
KROKO_ONNX_TAIL_PADDING_MARGIN_SECONDS = 0.1
_KROKO_OUTPUT_SUPPRESSION_LOCK = threading.RLock()


@dataclass
class KrokoOnnxDecodedOutput:
    text: str
    language: str = ""


def _load_numpy():
    try:
        return import_module("numpy")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'kroko_onnx' transcription engine requires numpy audio arrays."
        ) from exc


def _load_online_recognizer_class():
    try:
        kroko_onnx = import_module("kroko_onnx")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'kroko_onnx' transcription engine requires the optional "
            "'kroko-onnx' package. Install it from "
            "https://github.com/kroko-ai/kroko-onnx. On Windows, use the "
            "cross-platform-builds branch to build or install the generated "
            "Windows wheel in the same Python environment."
        ) from exc

    try:
        return kroko_onnx.OnlineRecognizer
    except AttributeError as exc:
        raise TranscriptionEngineError(
            "The installed 'kroko-onnx' package does not expose "
            "OnlineRecognizer. Install a current kroko-onnx checkout."
        ) from exc


def _bool_option(options, name, default=False):
    value = options.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _output_suppression_enabled(options):
    return _bool_option(
        options,
        "suppress_native_output",
        _bool_option(
            options,
            "suppress_output",
            _bool_option(options, "quiet", _bool_option(options, "silent", False)),
        ),
    )


def _enable_kroko_license_quiet(enabled):
    if not enabled:
        return

    os.environ[KROKO_ONNX_LICENSE_QUIET_ENV] = "1"
    try:
        os.putenv(KROKO_ONNX_LICENSE_QUIET_ENV, "1")
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetEnvironmentVariableW(
                KROKO_ONNX_LICENSE_QUIET_ENV,
                "1",
            )
        except Exception:
            pass


@contextmanager
def _suppress_native_output(enabled):
    if not enabled:
        yield
        return

    with _KROKO_OUTPUT_SUPPRESSION_LOCK:
        null_fd = None
        saved_fds = []
        try:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass

            null_fd = os.open(os.devnull, os.O_WRONLY)
            for fd in (1, 2):
                saved_fd = None
                try:
                    saved_fd = os.dup(fd)
                    os.dup2(null_fd, fd)
                except OSError:
                    if saved_fd is not None:
                        try:
                            os.close(saved_fd)
                        except OSError:
                            pass
                    continue
                saved_fds.append((fd, saved_fd))

            yield
        finally:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass

            for fd, saved_fd in reversed(saved_fds):
                try:
                    os.dup2(saved_fd, fd)
                finally:
                    os.close(saved_fd)

            if null_fd is not None:
                os.close(null_fd)


def _int_option(options, name, default):
    try:
        return int(options.get(name, default))
    except (TypeError, ValueError):
        raise TranscriptionEngineError("kroko-onnx option '%s' must be an integer." % name)


def _float_option(options, name, default):
    try:
        return float(options.get(name, default))
    except (TypeError, ValueError):
        raise TranscriptionEngineError("kroko-onnx option '%s' must be a number." % name)


def _provider_from_config(config, options):
    explicit = options.get("provider")
    if explicit:
        return str(explicit)
    device = str(config.device or "").lower()
    return "cuda" if device.startswith("cuda") else "cpu"


def _maybe_under_download_root(download_root, value):
    path = Path(str(value)).expanduser()
    if path.is_absolute() or not download_root:
        return path
    return Path(download_root).expanduser() / path


def _default_cache_path(filename):
    return KROKO_ONNX_DEFAULT_CACHE_DIR / filename


def _looks_like_kroko_data_file(filename):
    return filename.startswith("Kroko-") and filename.endswith(".data")


def _download_url(repo_id, filename, revision="main"):
    repo = quote(str(repo_id).strip("/"), safe="/")
    rev = quote(str(revision or "main"), safe="")
    name = quote(filename, safe="")
    return "https://huggingface.co/%s/resolve/%s/%s" % (repo, rev, name)


def _download_file(url, target_path, token=""):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name("%s.part" % target_path.name)
    headers = {}
    if token:
        headers["Authorization"] = "Bearer %s" % token
    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        tmp_path.replace(target_path)
    except Exception as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise TranscriptionEngineError(
            "Could not download Kroko model from %s to %s: %s"
            % (url, target_path, exc)
        ) from exc
    return target_path


def _first_existing_data_file(model_dir):
    try:
        data_files = sorted(model_dir.glob("*.data"))
    except OSError:
        return None
    return data_files[0] if len(data_files) == 1 else None


def _language_from_model_path(path):
    match = re.search(r"(?:^|[-_/\\])Kroko-([A-Za-z]{2})-", str(path))
    return match.group(1).lower() if match else ""


def _chunk_seconds_from_model_path(path):
    match = re.search(r"-(\d+)-[LMS](?:[-_/\\]|$)", str(path))
    if not match:
        return None
    return int(match.group(1)) * 0.02


def _default_tail_padding_seconds(path):
    chunk_seconds = _chunk_seconds_from_model_path(path)
    if chunk_seconds is None:
        return KROKO_ONNX_FALLBACK_TAIL_PADDING_SECONDS
    return max(
        KROKO_ONNX_FALLBACK_TAIL_PADDING_SECONDS,
        chunk_seconds + KROKO_ONNX_TAIL_PADDING_MARGIN_SECONDS,
    )


def _tail_padding_option(options, path):
    value = options.get("tail_padding_seconds", options.get("finalization_padding_seconds"))
    if value is None:
        return _default_tail_padding_seconds(path)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return _default_tail_padding_seconds(path)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TranscriptionEngineError(
            "kroko-onnx option 'tail_padding_seconds' must be a number or 'auto'."
        )


class KrokoOnnxBackend:
    def __init__(self, config, recognizer_cls=None, numpy_module=None):
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.np = numpy_module or _load_numpy()
        self.model_path = self._resolve_model_path()
        self.sample_rate = _int_option(self.engine_options, "sample_rate", 16000)
        self.tail_padding_seconds = _tail_padding_option(self.engine_options, self.model_path)
        self.suppress_native_output = _output_suppression_enabled(self.engine_options)
        _enable_kroko_license_quiet(self.suppress_native_output)
        recognizer_cls = recognizer_cls or _load_online_recognizer_class()
        self.recognizer = self._create_recognizer(recognizer_cls)

    def _resolve_model_path(self):
        model_path_value = self.engine_options.get("model_path") or self.engine_options.get("model_file")
        model_dir_value = self.engine_options.get("model_dir")
        filename = None

        if model_path_value:
            path = _maybe_under_download_root(self.config.download_root, model_path_value)
            filename = path.name
        elif model_dir_value:
            model_dir = _maybe_under_download_root(self.config.download_root, model_dir_value)
            if model_dir.suffix == ".data":
                path = model_dir
                filename = path.name
            else:
                filename = self.engine_options.get("model_filename") or DEFAULT_KROKO_ONNX_MODEL
                path = model_dir / filename
                if not path.is_file() and model_dir.is_dir():
                    path = _first_existing_data_file(model_dir) or path
                    filename = path.name
        else:
            model_value = self.config.model or DEFAULT_KROKO_ONNX_MODEL
            model_path = Path(str(model_value)).expanduser()
            filename = model_path.name
            if (
                not model_path.is_absolute()
                and not self.config.download_root
                and model_path.parent == Path(".")
                and _looks_like_kroko_data_file(filename)
            ):
                path = _default_cache_path(filename)
            else:
                path = _maybe_under_download_root(self.config.download_root, model_value)

        if not path.is_file():
            path = self._maybe_download_model(path, filename)

        if not path.is_file():
            raise TranscriptionEngineError(
                "Missing kroko-onnx model file: %s. Public Community models are "
                "downloaded automatically when auto_download_model is enabled. "
                "For Pro/private models, pass an existing .data file, a direct "
                "model_download_url, or a Hugging Face repo/token option. Source: "
                "%s (%s), for example %s, then pass it as model or "
                "engine_options['model_path']." % (
                    path,
                    KROKO_ONNX_MODEL_URL,
                    KROKO_ONNX_HF_REPO,
                    DEFAULT_KROKO_ONNX_MODEL,
                )
            )
        return path

    def _maybe_download_model(self, path, filename):
        if filename is None:
            return path

        auto_download = _bool_option(
            self.engine_options,
            "auto_download_model",
            _bool_option(self.engine_options, "download_model", True),
        )
        if not auto_download:
            return path

        download_url = self.engine_options.get("model_download_url")
        repo_id = (
            self.engine_options.get("model_repo_id")
            or self.engine_options.get("hf_repo_id")
            or KROKO_ONNX_HF_REPO
        )
        revision = self.engine_options.get("model_revision") or self.engine_options.get("hf_revision") or "main"
        token = self.engine_options.get("hf_token") or self.engine_options.get("token") or ""

        public_default_repo_model = repo_id == KROKO_ONNX_HF_REPO and filename in KROKO_ONNX_PUBLIC_MODELS
        if not download_url and not public_default_repo_model:
            if not _looks_like_kroko_data_file(filename):
                return path
            if repo_id == KROKO_ONNX_HF_REPO:
                return path

        if download_url:
            return _download_file(str(download_url), path, token=token)

        try:
            hf_hub_download = import_module("huggingface_hub").hf_hub_download
        except ModuleNotFoundError:
            return _download_file(_download_url(repo_id, filename, revision), path, token=token)

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token or None,
                local_dir=str(path.parent),
            )
        except TypeError:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token or None,
            )
        except Exception as exc:
            raise TranscriptionEngineError(
                "Could not download Kroko model %s from %s: %s"
                % (filename, repo_id, exc)
            ) from exc

        downloaded_path = Path(downloaded_path)
        if downloaded_path != path and downloaded_path.is_file() and not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, path)
        return path

    def _recognizer_kwargs(self):
        options = self.engine_options
        kwargs = {
            "model_path": str(self.model_path),
            "key": options.get("key", ""),
            "referralcode": options.get("referralcode", ""),
            "num_threads": _int_option(options, "num_threads", 1),
            "provider": _provider_from_config(self.config, options),
            "sample_rate": self.sample_rate,
            "feature_dim": _int_option(options, "feature_dim", 80),
            "decoding_method": options.get("decoding_method", "greedy_search"),
            "max_active_paths": _int_option(options, "max_active_paths", 4),
            "hotwords_file": options.get("hotwords_file", ""),
            "hotwords_score": _float_option(options, "hotwords_score", 1.5),
            "blank_penalty": _float_option(options, "blank_penalty", 0.0),
            "enable_endpoint_detection": _bool_option(
                options,
                "enable_endpoint_detection",
                True,
            ),
            "rule1_min_trailing_silence": _float_option(
                options,
                "rule1_min_trailing_silence",
                2.4,
            ),
            "rule2_min_trailing_silence": _float_option(
                options,
                "rule2_min_trailing_silence",
                1.2,
            ),
            "rule3_min_utterance_length": _float_option(
                options,
                "rule3_min_utterance_length",
                20.0,
            ),
        }

        nested = options.get("recognizer", {})
        if nested is None:
            nested = {}
        if not isinstance(nested, dict):
            raise TranscriptionEngineError(
                "kroko-onnx option 'recognizer' must be a JSON object."
            )
        kwargs.update(nested)
        return kwargs

    def _create_recognizer(self, recognizer_cls):
        try:
            with _suppress_native_output(self.suppress_native_output):
                return recognizer_cls.from_transducer(**self._recognizer_kwargs())
        except AttributeError as exc:
            raise TranscriptionEngineError(
                "The installed 'kroko-onnx' package does not expose "
                "OnlineRecognizer.from_transducer."
            ) from exc

    def _as_float32_audio(self, audio):
        if hasattr(audio, "values"):
            audio = audio.values
        array = self.np.asarray(audio, dtype=self.np.float32)
        if getattr(array, "ndim", 1) > 1:
            array = array.reshape(-1)
        return array

    def _accept_waveform(self, stream, sample_rate, audio):
        try:
            with _suppress_native_output(self.suppress_native_output):
                stream.accept_waveform(sample_rate, audio)
        except TypeError:
            with _suppress_native_output(self.suppress_native_output):
                stream.accept_waveform(sample_rate=sample_rate, waveform=audio)

    def _decode_ready_stream(self, stream):
        if not hasattr(self.recognizer, "is_ready"):
            if hasattr(self.recognizer, "decode_stream"):
                with _suppress_native_output(self.suppress_native_output):
                    self.recognizer.decode_stream(stream)
                return
            raise TranscriptionEngineError(
                "The installed 'kroko-onnx' recognizer exposes neither "
                "is_ready/decode_streams nor decode_stream."
            )

        while True:
            with _suppress_native_output(self.suppress_native_output):
                ready = self.recognizer.is_ready(stream)
            if not ready:
                break

            if hasattr(self.recognizer, "decode_streams"):
                with _suppress_native_output(self.suppress_native_output):
                    self.recognizer.decode_streams([stream])
            elif hasattr(self.recognizer, "decode_stream"):
                with _suppress_native_output(self.suppress_native_output):
                    self.recognizer.decode_stream(stream)
            else:
                raise TranscriptionEngineError(
                    "The installed 'kroko-onnx' recognizer cannot decode streams."
                )

    def _result_text(self, stream):
        if hasattr(self.recognizer, "get_result"):
            with _suppress_native_output(self.suppress_native_output):
                result = self.recognizer.get_result(stream)
        else:
            result = getattr(stream, "result", "")
        return text_from_output(result)

    def transcribe(self, audio, **params):
        sample_rate = int(params.get("sample_rate", self.sample_rate))
        with _suppress_native_output(self.suppress_native_output):
            stream = self.recognizer.create_stream()
        audio = self._as_float32_audio(audio)
        self._accept_waveform(stream, sample_rate, audio)

        if self.tail_padding_seconds > 0:
            tail_padding = self.np.zeros(
                int(self.tail_padding_seconds * sample_rate),
                dtype=self.np.float32,
            )
            self._accept_waveform(stream, sample_rate, tail_padding)

        if hasattr(stream, "input_finished"):
            with _suppress_native_output(self.suppress_native_output):
                stream.input_finished()

        self._decode_ready_stream(stream)
        return KrokoOnnxDecodedOutput(
            text=self._result_text(stream),
            language=str(params.get("language") or _language_from_model_path(self.model_path)),
        )


class KrokoOnnxEngine(BaseTranscriptionEngine):
    engine_name = "kroko_onnx"
    supports_streaming = True

    def __init__(self, config, backend=None, backend_cls=None):
        super().__init__(config)
        self.backend = backend or (backend_cls or KrokoOnnxBackend)(config)

    def create_streaming_session(self, language=None, use_prompt=True):
        return KrokoOnnxStreamingSession(self, language=language, use_prompt=use_prompt)

    def transcribe(self, audio, language=None, use_prompt=True):
        audio = self._normalize_audio(audio)
        options = self.config.engine_options or {}
        output = self.backend.transcribe(
            audio,
            language=options.get("language") or language,
        )
        detected_language = output.language or options.get("language") or language
        return TranscriptionResult(
            text=output.text.strip(),
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if detected_language else 0.0,
            ),
        )


class KrokoOnnxStreamingSession(StreamingTranscriptionSession):
    def __init__(self, engine, language=None, use_prompt=True):
        self.engine = engine
        self.backend = engine.backend
        self.use_prompt = use_prompt
        options = engine.config.engine_options or {}
        self.language = options.get("language") or language
        self.sample_rate = self.backend.sample_rate
        self.stream = None
        self.closed = False
        self.finished = False
        self.reset()

    def reset(self):
        with _suppress_native_output(self.backend.suppress_native_output):
            self.stream = self.backend.recognizer.create_stream()
        self.closed = False
        self.finished = False

    def accept_audio(self, audio, sample_rate=None):
        if self.closed:
            raise TranscriptionEngineError("Cannot feed a closed Kroko streaming session.")
        if self.finished:
            raise TranscriptionEngineError("Cannot feed a finished Kroko streaming session.")

        audio = self.backend._as_float32_audio(audio)
        if audio.size == 0:
            return

        if self.engine.config.normalize_audio:
            audio = self.engine._normalize_audio(audio)

        self.backend._accept_waveform(
            self.stream,
            int(sample_rate or self.sample_rate),
            audio,
        )

    def decode(self):
        if self.closed or self.stream is None:
            return
        self.backend._decode_ready_stream(self.stream)

    def get_result(self):
        if self.closed or self.stream is None:
            return TranscriptionResult(text="")

        detected_language = (
            self.language or _language_from_model_path(self.backend.model_path)
        )
        return TranscriptionResult(
            text=self.backend._result_text(self.stream).strip(),
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if detected_language else 0.0,
            ),
        )

    def finish(self):
        if self.closed:
            return self.get_result()

        if not self.finished:
            if self.backend.tail_padding_seconds > 0:
                tail_padding = self.backend.np.zeros(
                    int(self.backend.tail_padding_seconds * self.sample_rate),
                    dtype=self.backend.np.float32,
                )
                self.backend._accept_waveform(
                    self.stream,
                    self.sample_rate,
                    tail_padding,
                )

            if hasattr(self.stream, "input_finished"):
                with _suppress_native_output(self.backend.suppress_native_output):
                    self.stream.input_finished()

            self.finished = True

        self.decode()
        return self.get_result()

    def close(self):
        self.closed = True
        self.stream = None
