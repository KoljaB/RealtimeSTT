"""Adapts sherpa-onnx offline recognizers to the engine interface."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_SHERPA_ONNX_PARAKEET_MODEL = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL = "sherpa-onnx-moonshine-tiny-en-int8"

PARAKEET_DOWNLOAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
)
MOONSHINE_DOWNLOAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-moonshine-tiny-en-int8.tar.bz2"
)

KNOWN_MODEL_DIRS = {
    "nvidia/parakeet-tdt-0.6b-v3": DEFAULT_SHERPA_ONNX_PARAKEET_MODEL,
    DEFAULT_SHERPA_ONNX_PARAKEET_MODEL: DEFAULT_SHERPA_ONNX_PARAKEET_MODEL,
    "UsefulSensors/moonshine-streaming-medium": DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL,
    "UsefulSensors/moonshine-streaming-tiny": DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL,
    DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL: DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL,
}


@dataclass
class SherpaOnnxDecodedOutput:
    """
    Carries sherpa-onnx decoded text and language metadata.
    """

    text: str
    language: str = ""


def _load_offline_recognizer_class():
    try:
        sherpa_onnx = import_module("sherpa_onnx")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The sherpa-onnx transcription engines require the optional "
            "'sherpa-onnx' package. Install it with 'pip install sherpa-onnx'."
        ) from exc

    try:
        return sherpa_onnx.OfflineRecognizer
    except AttributeError as exc:
        raise TranscriptionEngineError(
            "The installed 'sherpa-onnx' package does not expose "
            "OfflineRecognizer. Install a current sherpa-onnx release."
        ) from exc


def _bool_option(options, name, default=False):
    value = options.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _int_option(options, name, default):
    try:
        return int(options.get(name, default))
    except (TypeError, ValueError):
        raise TranscriptionEngineError(
            "sherpa-onnx option '%s' must be an integer." % name
        )


def _float_option(options, name, default):
    try:
        return float(options.get(name, default))
    except (TypeError, ValueError):
        raise TranscriptionEngineError(
            "sherpa-onnx option '%s' must be a number." % name
        )


def _maybe_join(base_dir, value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


class SherpaOnnxOfflineBackend:
    """
    Provides shared setup for sherpa-onnx offline recognizers.
    """

    family = "sherpa_onnx"
    default_model_dir = ""
    download_url = ""

    def __init__(self, config, recognizer_cls=None):
        """
        Initializes shared sherpa-onnx recognizer state.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.file_options = dict(self.engine_options.get("files", {}))
        self.model_dir = self._resolve_model_dir()
        self.input_sample_rate = _int_option(
            self.engine_options,
            "input_sample_rate",
            self.engine_options.get("sample_rate", 16000),
        )
        recognizer_cls = recognizer_cls or _load_offline_recognizer_class()
        self.recognizer = self._create_recognizer(recognizer_cls)

    def _resolve_model_dir(self):
        model_value = (
            self.engine_options.get("model_dir")
            or self.config.model
            or self.default_model_dir
        )
        model_value = KNOWN_MODEL_DIRS.get(str(model_value), str(model_value))
        model_path = Path(str(model_value)).expanduser()

        candidates = []
        if model_path.is_absolute():
            candidates.append(model_path)
        else:
            if self.config.download_root:
                candidates.append(Path(self.config.download_root).expanduser() / model_path)
            candidates.append(model_path)

        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        return candidates[0]

    def _file(self, option_name, default_name):
        value = self.file_options.get(option_name, self.engine_options.get(option_name))
        path = _maybe_join(self.model_dir, value or default_name)
        if path.is_file():
            return str(path)
        raise TranscriptionEngineError(
            "Missing sherpa-onnx %s file for %s: %s. Download and extract %s, "
            "then pass the extracted directory as model or engine_options['model_dir']."
            % (
                option_name,
                self.family,
                path,
                self.download_url or "the required sherpa-onnx model bundle",
            )
        )

    def _common_recognizer_kwargs(self):
        model_options = self.engine_options.get("model", {})
        if not isinstance(model_options, dict):
            model_options = {}

        return {
            "num_threads": int(
                self.engine_options.get(
                    "num_threads",
                    model_options.get("num_threads", 1),
                )
            ),
            "decoding_method": self.engine_options.get(
                "decoding_method",
                "greedy_search",
            ),
            "debug": _bool_option(self.engine_options, "debug", False),
            "provider": self.engine_options.get("provider", "cpu"),
            "rule_fsts": self.engine_options.get("rule_fsts", ""),
            "rule_fars": self.engine_options.get("rule_fars", ""),
            "hr_dict_dir": self.engine_options.get("hr_dict_dir", ""),
            "hr_rule_fsts": self.engine_options.get("hr_rule_fsts", ""),
            "hr_lexicon": self.engine_options.get("hr_lexicon", ""),
        }

    def _create_recognizer(self, recognizer_cls):
        raise NotImplementedError

    def transcribe(self, audio, **params):
        """
        Runs sherpa-onnx offline transcription for one audio input.
        """
        stream = self.recognizer.create_stream()
        stream.accept_waveform(
            int(params.get("sample_rate", self.input_sample_rate)),
            audio,
        )
        self.recognizer.decode_stream(stream)
        result = stream.result
        return SherpaOnnxDecodedOutput(
            text=str(getattr(result, "text", result)).strip(),
            language=str(
                getattr(result, "language", getattr(result, "lang", ""))
                or ""
            ),
        )


class SherpaOnnxParakeetBackend(SherpaOnnxOfflineBackend):
    """
    Wraps the sherpa-onnx Parakeet recognizer.
    """

    family = "sherpa_onnx_parakeet"
    default_model_dir = DEFAULT_SHERPA_ONNX_PARAKEET_MODEL
    download_url = PARAKEET_DOWNLOAD_URL

    def _create_recognizer(self, recognizer_cls):
        kwargs = self._common_recognizer_kwargs()
        kwargs.update(
            {
                "encoder": self._file("encoder", "encoder.int8.onnx"),
                "decoder": self._file("decoder", "decoder.int8.onnx"),
                "joiner": self._file("joiner", "joiner.int8.onnx"),
                "tokens": self._file("tokens", "tokens.txt"),
                "sample_rate": _int_option(self.engine_options, "sample_rate", 16000),
                "feature_dim": _int_option(self.engine_options, "feature_dim", 80),
                "dither": _float_option(self.engine_options, "dither", 0.0),
                "max_active_paths": _int_option(
                    self.engine_options,
                    "max_active_paths",
                    4,
                ),
                "hotwords_file": self.engine_options.get("hotwords_file", ""),
                "hotwords_score": _float_option(
                    self.engine_options,
                    "hotwords_score",
                    1.5,
                ),
                "blank_penalty": _float_option(
                    self.engine_options,
                    "blank_penalty",
                    0.0,
                ),
                "modeling_unit": self.engine_options.get("modeling_unit", "cjkchar"),
                "bpe_vocab": self.engine_options.get("bpe_vocab", ""),
                "model_type": self.engine_options.get(
                    "model_type",
                    "nemo_transducer",
                ),
                "lm": self.engine_options.get("lm", ""),
                "lm_scale": _float_option(self.engine_options, "lm_scale", 0.1),
                "lodr_fst": self.engine_options.get("lodr_fst", ""),
                "lodr_scale": _float_option(self.engine_options, "lodr_scale", 0.0),
            }
        )
        return recognizer_cls.from_transducer(**kwargs)


class SherpaOnnxMoonshineBackend(SherpaOnnxOfflineBackend):
    """
    Wraps the sherpa-onnx Moonshine recognizer.
    """

    family = "sherpa_onnx_moonshine"
    default_model_dir = DEFAULT_SHERPA_ONNX_MOONSHINE_MODEL
    download_url = MOONSHINE_DOWNLOAD_URL

    def _create_recognizer(self, recognizer_cls):
        kwargs = self._common_recognizer_kwargs()
        kwargs.update(
            {
                "preprocessor": self._file("preprocessor", "preprocess.onnx"),
                "encoder": self._file("encoder", "encode.int8.onnx"),
                "uncached_decoder": self._file(
                    "uncached_decoder",
                    "uncached_decode.int8.onnx",
                ),
                "cached_decoder": self._file(
                    "cached_decoder",
                    "cached_decode.int8.onnx",
                ),
                "tokens": self._file("tokens", "tokens.txt"),
            }
        )
        return recognizer_cls.from_moonshine(**kwargs)


class SherpaOnnxParakeetEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with the sherpa-onnx Parakeet recognizer.
    """

    engine_name = "sherpa_onnx_parakeet"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the sherpa-onnx Parakeet engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or SherpaOnnxParakeetBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with sherpa-onnx Parakeet.
        """
        audio = self._normalize_audio(audio)
        output = self.backend.transcribe(audio)
        detected_language = output.language or language
        return TranscriptionResult(
            text=output.text,
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if detected_language else 0.0,
            ),
        )


class SherpaOnnxMoonshineEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with the sherpa-onnx Moonshine recognizer.
    """

    engine_name = "sherpa_onnx_moonshine"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the sherpa-onnx Moonshine engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or SherpaOnnxMoonshineBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with sherpa-onnx Moonshine.
        """
        if language and language.lower() not in ("en", "english"):
            raise TranscriptionEngineError(
                "The sherpa-onnx Moonshine tiny INT8 engine supports English "
                "transcription only."
            )
        audio = self._normalize_audio(audio)
        output = self.backend.transcribe(audio)
        return TranscriptionResult(
            text=output.text,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )
