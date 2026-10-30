"""Streaming sherpa-onnx adapter for the multilingual Nemotron 3.5 model."""

from importlib import import_module
from pathlib import Path
import re

from ..model_manifests import SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST
from ._model_utils import attr_or_key, first_item, text_from_output
from .base import (
    BaseTranscriptionEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL = (
    SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST.model_id
)
NEMOTRON_DOWNLOAD_URL = SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST.archive_url
NEMOTRON_MODEL_MANIFEST = SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST
_MODEL_SAMPLE_RATE = 16000
_MODEL_FEATURE_DIM = 128
_LANGUAGE_TAG_RE = re.compile(
    r"^\s*<(?P<language>[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?)>\s*"
)


def _load_numpy():
    """Load NumPy when available for contiguous float32 model input."""

    try:
        return import_module("numpy")
    except ModuleNotFoundError:
        return None


def _load_online_recognizer_class():
    """Load the optional sherpa-onnx ``OnlineRecognizer`` class."""

    try:
        sherpa_onnx = import_module("sherpa_onnx")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The sherpa-onnx Nemotron engine requires the optional "
            "'sherpa-onnx' package. Install it with 'pip install sherpa-onnx'."
        ) from exc

    try:
        return sherpa_onnx.OnlineRecognizer
    except AttributeError as exc:
        raise TranscriptionEngineError(
            "The installed 'sherpa-onnx' package does not expose "
            "OnlineRecognizer. Install the release-pinned sherpa-onnx 1.13.4."
        ) from exc


def _bool_option(options, name, default=False):
    """Read a boolean engine option."""

    value = options.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _int_option(options, name, default):
    """Read an integer engine option with a stable setup error."""

    try:
        return int(options.get(name, default))
    except (TypeError, ValueError) as exc:
        raise TranscriptionEngineError(
            "sherpa-onnx Nemotron option '%s' must be an integer." % name
        ) from exc


def _float_option(options, name, default):
    """Read a numeric engine option with a stable setup error."""

    try:
        return float(options.get(name, default))
    except (TypeError, ValueError) as exc:
        raise TranscriptionEngineError(
            "sherpa-onnx Nemotron option '%s' must be a number." % name
        ) from exc


def _resolve_model_dir(config, options):
    """Resolve a persistent model directory without downloading anything."""

    model_value = (
        options.get("model_dir")
        or config.model
        or DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL
    )
    model_path = Path(str(model_value)).expanduser()
    candidates = []
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        if config.download_root:
            candidates.append(Path(config.download_root).expanduser() / model_path)
        candidates.append(model_path)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


class SherpaOnnxNemotronBackend:
    """Owns one sherpa ``OnlineRecognizer`` for independent stream sessions."""

    family = "sherpa_onnx_nemotron"
    default_model_dir = DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL
    download_url = NEMOTRON_DOWNLOAD_URL
    model_manifest = NEMOTRON_MODEL_MANIFEST
    sample_rate = _MODEL_SAMPLE_RATE

    def __init__(self, config, recognizer_cls=None, numpy_module=None):
        """Resolve the user-owned model directory and create the recognizer."""

        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_dir = _resolve_model_dir(config, self.engine_options)
        self.np = numpy_module if numpy_module is not None else _load_numpy()
        recognizer_cls = recognizer_cls or _load_online_recognizer_class()
        missing_message = self.model_manifest.describe_missing_files(self.model_dir)
        if missing_message:
            raise TranscriptionEngineError(missing_message)
        if _bool_option(self.engine_options, "verify_model_files", False):
            verification_error = self.model_manifest.describe_invalid_files(
                self.model_dir,
                include_optional=_bool_option(
                    self.engine_options,
                    "verify_optional_model_files",
                    False,
                ),
            )
            if verification_error:
                raise TranscriptionEngineError(verification_error)
        self.recognizer = self._create_recognizer(recognizer_cls)

    def _file(self, name):
        """Return one required file from the pinned model layout."""

        path = self.model_dir / name
        if path.is_file():
            return str(path)
        # The constructor normally catches this; keep the method defensive for
        # embedders that replace ``model_dir`` after setup in a test double.
        raise TranscriptionEngineError(
            self.model_manifest.describe_missing_files(self.model_dir)
            or "Missing sherpa-onnx Nemotron model file: %s" % path
        )

    def _recognizer_kwargs(self):
        """Build OnlineRecognizer.from_transducer arguments for Nemotron."""

        options = self.engine_options
        kwargs = {
            "tokens": self._file("tokens.txt"),
            "encoder": self._file("encoder.int8.onnx"),
            "decoder": self._file("decoder.int8.onnx"),
            "joiner": self._file("joiner.int8.onnx"),
            "num_threads": _int_option(options, "num_threads", 1),
            "sample_rate": _MODEL_SAMPLE_RATE,
            # The Nemotron export uses 128-dimensional NeMo features.
            "feature_dim": _int_option(options, "feature_dim", _MODEL_FEATURE_DIM),
            "low_freq": _float_option(options, "low_freq", 20.0),
            "high_freq": _float_option(options, "high_freq", -400.0),
            "dither": _float_option(options, "dither", 0.0),
            "normalize_samples": _bool_option(options, "normalize_samples", True),
            "snip_edges": _bool_option(options, "snip_edges", False),
            "enable_endpoint_detection": _bool_option(
                options,
                "enable_endpoint_detection",
                False,
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
            "decoding_method": options.get("decoding_method", "greedy_search"),
            "max_active_paths": _int_option(options, "max_active_paths", 4),
            "hotwords_score": _float_option(options, "hotwords_score", 1.5),
            "blank_penalty": _float_option(options, "blank_penalty", 0.0),
            "hotwords_file": options.get("hotwords_file", ""),
            # Keep this empty so sherpa can auto-detect the Nemotron model and
            # does not load the large encoder twice during setup.
            "model_type": options.get("model_type", ""),
            "modeling_unit": options.get("modeling_unit", "cjkchar"),
            "bpe_vocab": options.get("bpe_vocab", ""),
            "lm": options.get("lm", ""),
            "lm_scale": _float_option(options, "lm_scale", 0.1),
            "lm_shallow_fusion": _bool_option(options, "lm_shallow_fusion", True),
            "temperature_scale": _float_option(options, "temperature_scale", 2.0),
            "reset_encoder": _bool_option(options, "reset_encoder", False),
            "debug": _bool_option(options, "debug", False),
            "rule_fsts": options.get("rule_fsts", ""),
            "rule_fars": options.get("rule_fars", ""),
            "provider": options.get("provider", "cpu"),
            "device": _int_option(options, "device", 0),
            "hr_dict_dir": options.get("hr_dict_dir", ""),
            "hr_rule_fsts": options.get("hr_rule_fsts", ""),
            "hr_lexicon": options.get("hr_lexicon", ""),
            "lodr_fst": options.get("lodr_fst", ""),
            "lodr_scale": _float_option(options, "lodr_scale", 0.0),
        }
        nested = options.get("recognizer", {})
        if nested is None:
            nested = {}
        if not isinstance(nested, dict):
            raise TranscriptionEngineError(
                "sherpa-onnx Nemotron option 'recognizer' must be a JSON object."
            )
        kwargs.update(nested)
        return kwargs

    def _create_recognizer(self, recognizer_cls):
        """Create the OnlineRecognizer through its transducer factory."""

        try:
            return recognizer_cls.from_transducer(**self._recognizer_kwargs())
        except AttributeError as exc:
            raise TranscriptionEngineError(
                "The installed 'sherpa-onnx' package does not expose "
                "OnlineRecognizer.from_transducer."
            ) from exc

    def as_float32_audio(self, audio):
        """Validate and convert one new mono 16 kHz frame block."""

        if audio is None:
            raise TranscriptionEngineError("Received None audio for Nemotron streaming")
        if hasattr(audio, "values"):
            audio = audio.values

        np = self.np
        if np is not None:
            try:
                array = np.asarray(audio, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise TranscriptionEngineError(
                    "Nemotron audio must be a numeric mono frame array."
                ) from exc
            if getattr(array, "ndim", 1) == 2 and array.shape[1] == 1:
                array = array.reshape(-1)
            elif getattr(array, "ndim", 1) != 1:
                raise TranscriptionEngineError(
                    "The sherpa-onnx Nemotron engine accepts mono audio only."
                )
            return np.ascontiguousarray(array, dtype=np.float32)

        # NumPy is a sherpa-onnx dependency in normal installations, but the
        # adapter remains unit-testable with plain Python frame lists.
        try:
            values = list(audio)
        except TypeError as exc:
            raise TranscriptionEngineError(
                "Nemotron audio must be a numeric mono frame array."
            ) from exc
        if values and isinstance(values[0], (list, tuple)):
            if any(not isinstance(row, (list, tuple)) or len(row) != 1 for row in values):
                raise TranscriptionEngineError(
                    "The sherpa-onnx Nemotron engine accepts mono audio only."
                )
            values = [row[0] for row in values]
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise TranscriptionEngineError(
                "Nemotron audio must be a numeric mono frame array."
            ) from exc

    def accept_waveform(self, stream, audio):
        """Feed validated samples to a sherpa stream."""

        stream.accept_waveform(_MODEL_SAMPLE_RATE, audio)

    def decode_ready(self, stream):
        """Run the authoritative ready-frame decode loop."""

        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

    def result(self, stream):
        """Read the complete current OnlineRecognizer result object."""

        if hasattr(self.recognizer, "get_result_all"):
            return self.recognizer.get_result_all(stream)
        if hasattr(self.recognizer, "get_result"):
            return self.recognizer.get_result(stream)
        return getattr(stream, "result", "")


def _normalize_language(language):
    """Normalize a stream language request while preserving locale codes."""

    if language is None:
        return "auto"
    value = str(language).strip()
    return "auto" if not value or value.lower() == "auto" else value


def _set_stream_language(stream, language):
    """Set Nemotron's per-stream language before any waveform is accepted."""

    setter = getattr(stream, "set_option", None)
    if not callable(setter):
        setter = getattr(stream, "SetOption", None)
    if not callable(setter):
        raise TranscriptionEngineError(
            "The installed 'sherpa-onnx' OnlineStream does not support "
            "per-stream language options. Install the release-pinned "
            "sherpa-onnx 1.13.4."
        )
    setter("language", language)


def _result_language_and_text(result):
    """Extract text and a language tag when the runtime exposes one."""

    item = first_item(result)
    text = text_from_output(item)
    language = (
        attr_or_key(item, "language")
        or attr_or_key(item, "lang")
        or attr_or_key(item, "detected_language")
    )
    match = _LANGUAGE_TAG_RE.match(text)
    if match:
        language = language or match.group("language")
        text = text[match.end() :].strip()
    if language is not None:
        language = str(language).strip()
        if not language or language.lower() == "auto":
            language = None
    return text, language


def _transcription_result(result, requested_language):
    """Normalize partial/final output without inventing auto-detection."""

    text, detected_language = _result_language_and_text(result)
    if not text:
        # Empty output is a valid result for silence and for runtimes affected
        # by native decode failures.  Never attach a caller language to it.
        return TranscriptionResult(text="")

    if detected_language:
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=detected_language,
                # Nemotron does not expose a calibrated language probability.
                language_probability=0.0,
            ),
        )

    # A fixed language is a decoding hint, not a detection result.
    fixed_language = None
    if requested_language and requested_language.lower() != "auto":
        fixed_language = requested_language
    return TranscriptionResult(
        text=text,
        info=TranscriptionInfo(language=fixed_language, language_probability=0.0),
    )


class SherpaOnnxNemotronEngine(BaseTranscriptionEngine):
    """True streaming BaseTranscriptionEngine for Nemotron 3.5 ASR."""

    engine_name = "sherpa_onnx_nemotron"
    supports_streaming = True

    def __init__(self, config, backend=None, backend_cls=None):
        """Initialize the shared OnlineRecognizer backend."""

        super().__init__(config)
        self.backend = backend or (backend_cls or SherpaOnnxNemotronBackend)(config)

    def create_streaming_session(self, language=None, use_prompt=True):
        """Create a stream with its own language hint and decoder state."""

        options = self.config.engine_options or {}
        requested = options.get("language") if "language" in options else language
        return SherpaOnnxNemotronStreamingSession(
            self,
            language=requested,
            use_prompt=use_prompt,
        )

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        """Transcribe one complete 16 kHz mono buffer through a fresh stream."""

        session = self.create_streaming_session(language=language, use_prompt=use_prompt)
        try:
            session.accept_audio(audio, sample_rate=kwargs.pop("sample_rate", None))
            return session.finish()
        finally:
            session.close()


class SherpaOnnxNemotronStreamingSession(StreamingTranscriptionSession):
    """Owns one OnlineStream and exposes partial/final lifecycle methods."""

    def __init__(self, engine, language=None, use_prompt=True):
        """Create an independent stream and set its language before audio."""

        self.engine = engine
        self.backend = engine.backend
        self.use_prompt = use_prompt
        self.language = _normalize_language(language)
        self.stream = None
        self.closed = False
        self.finished = False
        self._last_result = TranscriptionResult(text="")
        self.reset()

    def reset(self):
        """Release the old stream and start a fresh utterance."""

        if self.closed:
            raise TranscriptionEngineError(
                "Cannot reset a closed Nemotron streaming session."
            )
        self._release_stream()
        stream = None
        try:
            stream = self.backend.recognizer.create_stream()
            _set_stream_language(stream, self.language)
        except Exception:
            if stream is not None:
                for method_name in ("close", "release"):
                    method = getattr(stream, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass
                        break
            self.stream = None
            raise
        self.stream = stream
        self.finished = False
        self._last_result = TranscriptionResult(text="")

    def accept_audio(self, audio, sample_rate=None):
        """Accept only new mono frames at the model's native 16 kHz rate."""

        if self.closed:
            raise TranscriptionEngineError(
                "Cannot feed a closed Nemotron streaming session."
            )
        if self.finished:
            raise TranscriptionEngineError(
                "Cannot feed a finished Nemotron streaming session."
            )
        if sample_rate is not None:
            try:
                valid_sample_rate = int(sample_rate) == _MODEL_SAMPLE_RATE
            except (TypeError, ValueError):
                valid_sample_rate = False
            if not valid_sample_rate:
                raise TranscriptionEngineError(
                    "The sherpa-onnx Nemotron engine accepts 16 kHz mono audio only."
                )
        audio = self.backend.as_float32_audio(audio)
        if len(audio) == 0 or getattr(audio, "size", len(audio)) == 0:
            return
        if self.engine.config.normalize_audio:
            audio = self._normalize_audio(audio)
        self.backend.accept_waveform(self.stream, audio)

    def _normalize_audio(self, audio):
        """Normalize list or NumPy audio without changing frame count."""

        np = self.backend.np
        if np is not None:
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 0.0:
                return (audio / peak) * 0.95
            return audio
        peak = max(abs(float(value)) for value in audio) if audio else 0.0
        if peak <= 0.0:
            return audio
        return [float(value) / peak * 0.95 for value in audio]

    def decode(self):
        """Decode every currently ready frame and retain partial output."""

        if self.closed or self.stream is None:
            return
        self.backend.decode_ready(self.stream)
        self._last_result = _transcription_result(
            self.backend.result(self.stream),
            self.language,
        )

    def get_result(self):
        """Return the current partial or final result."""

        if self.closed or self.stream is None:
            return self._last_result
        self._last_result = _transcription_result(
            self.backend.result(self.stream),
            self.language,
        )
        return self._last_result

    def finish(self):
        """Signal input completion, flush ready frames, and return final text."""

        if self.closed:
            return self._last_result
        if not self.finished:
            input_finished = getattr(self.stream, "input_finished", None)
            if not callable(input_finished):
                input_finished = getattr(self.stream, "InputFinished", None)
            if not callable(input_finished):
                raise TranscriptionEngineError(
                    "The installed 'sherpa-onnx' OnlineStream does not expose "
                    "input_finished()."
                )
            input_finished()
            self.finished = True
        self.decode()
        return self._last_result

    def input_finished(self):
        """Compatibility alias for :meth:`finish` at the session boundary."""

        return self.finish()

    def _release_stream(self):
        """Release native stream resources when the binding exposes a hook."""

        stream = self.stream
        self.stream = None
        if stream is None:
            return
        for method_name in ("close", "release"):
            method = getattr(stream, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    # Native stream destructors still run when the last Python
                    # reference is dropped; cleanup must remain idempotent.
                    pass
                break

    def close(self):
        """Release the native stream exactly once."""

        if self.closed:
            return
        self._release_stream()
        self.closed = True


# Concise aliases are additive and keep the descriptive names available.
NemotronEngine = SherpaOnnxNemotronEngine
NemotronStreamingSession = SherpaOnnxNemotronStreamingSession


__all__ = [
    "DEFAULT_SHERPA_ONNX_NEMOTRON_MODEL",
    "NEMOTRON_DOWNLOAD_URL",
    "NEMOTRON_MODEL_MANIFEST",
    "SherpaOnnxNemotronBackend",
    "SherpaOnnxNemotronEngine",
    "SherpaOnnxNemotronStreamingSession",
    "NemotronEngine",
    "NemotronStreamingSession",
]
