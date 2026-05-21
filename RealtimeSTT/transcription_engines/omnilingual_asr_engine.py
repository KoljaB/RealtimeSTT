from importlib import import_module
from pathlib import Path

from ._model_utils import decode_to_text, torch_dtype_from_compute_type
from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_OMNILINGUAL_ASR_MODEL = "omniASR_CTC_300M_v2"

KNOWN_OMNILINGUAL_ASR_MODELS = (
    "omniASR_CTC_300M_v2",
    "omniASR_CTC_1B_v2",
    "omniASR_LLM_1B_v2",
    "omniASR_CTC_3B",
    "omniASR_CTC_7B",
    "omniASR_LLM_3B",
    "omniASR_LLM_7B",
    "omniASR_LLM_7B_ZS",
)

OPTIONAL_OMNILINGUAL_ASR_V2_MODEL_NAMES = (
    "omniASR_CTC_3B_v2",
    "omniASR_CTC_7B_v2",
    "omniASR_LLM_3B_v2",
    "omniASR_LLM_7B_v2",
    "omniASR_LLM_7B_ZS_v2",
)

_WHISPER_DEFAULT_MODEL_NAMES = {"tiny", "tiny.en"}

_LANGUAGE_ALIASES = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
}


def _dependency_error_message():
    return (
        "The 'omnilingual_asr' transcription engine requires Meta's "
        "'omnilingual-asr' package, PyTorch, fairseq2, and fairseq2n. "
        "Install and run it from Linux or WSL2; native Windows installs "
        "currently fail because fairseq2n has no Windows wheel."
    )


def _model_card_from_config(config, engine_options):
    model_card = engine_options.get("model_card")
    if model_card:
        return str(model_card)

    model = str(config.model or "").strip()
    if not model or model.lower() in _WHISPER_DEFAULT_MODEL_NAMES:
        return DEFAULT_OMNILINGUAL_ASR_MODEL
    return model


def _is_ctc_model(model_card):
    return "_CTC_" in model_card.upper()


def _is_llm_model(model_card):
    return "_LLM_" in model_card.upper()


def _boolish_false(value):
    return value is False or str(value).lower() in {"0", "false", "no", "off"}


def _int_option(options, name, default):
    value = options.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TranscriptionEngineError(
            "omnilingual_asr option '%s' must be an integer." % name
        ) from exc


def _float_option(options, name, default):
    value = options.get(name, default)
    if value is None or _boolish_false(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TranscriptionEngineError(
            "omnilingual_asr option '%s' must be a number." % name
        ) from exc


def _dtype_from_option(torch_module, value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "auto", "default", "none"}:
        return None
    if normalized in {"float16", "fp16", "half"}:
        return getattr(torch_module, "float16", None)
    if normalized in {"bfloat16", "bf16"}:
        return getattr(torch_module, "bfloat16", None)
    if normalized in {"float32", "fp32", "full"}:
        return getattr(torch_module, "float32", None)

    if value.startswith("torch."):
        return getattr(torch_module, value.split(".", 1)[1], None)
    return getattr(torch_module, value, None)


def _normalize_language_code(language, aliases):
    if not language:
        return None
    language = str(language)
    if "_" in language:
        return language
    return aliases.get(language.lower().replace("-", "_"), language)


class OmnilingualASRBackend:
    def __init__(
        self,
        config,
        pipeline_cls=None,
        torch_module=None,
        numpy_module=None,
    ):
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_card = _model_card_from_config(config, self.engine_options)
        self.sample_rate = _int_option(self.engine_options, "sample_rate", 16000)
        self.batch_size = _int_option(
            self.engine_options,
            "batch_size",
            config.batch_size if config.batch_size and config.batch_size > 0 else 1,
        )
        self.max_audio_seconds = _float_option(
            self.engine_options,
            "max_audio_seconds",
            39.9,
        )
        self.pipeline_options = dict(self.engine_options.get("pipeline", {}))
        self.transcribe_options = dict(self.engine_options.get("transcribe", {}))
        self.language_aliases = dict(_LANGUAGE_ALIASES)
        self.language_aliases.update(self.engine_options.get("language_aliases", {}))

        self._pipeline_cls = pipeline_cls
        self._torch_module = torch_module
        self._numpy_module = numpy_module
        self.pipeline = None

    @staticmethod
    def _load_pipeline_cls():
        try:
            module = import_module("omnilingual_asr.models.inference.pipeline")
        except ImportError as exc:
            raise TranscriptionEngineError(_dependency_error_message()) from exc
        return module.ASRInferencePipeline

    @staticmethod
    def _load_torch():
        try:
            return import_module("torch")
        except ImportError as exc:
            raise TranscriptionEngineError(_dependency_error_message()) from exc

    @staticmethod
    def _load_numpy():
        try:
            return import_module("numpy")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'omnilingual_asr' transcription engine requires numpy "
                "to pass in-memory audio to Meta's Omnilingual ASR pipeline."
            ) from exc

    def _resolve_device(self):
        device = self.engine_options.get("device", self.config.device)
        if not device:
            return None

        device = str(device)
        if device == "cuda" and isinstance(self.config.gpu_device_index, int):
            return "cuda:%s" % self.config.gpu_device_index
        if device == "cuda" and isinstance(self.config.gpu_device_index, (list, tuple)):
            if self.config.gpu_device_index:
                return "cuda:%s" % self.config.gpu_device_index[0]
        return device

    def _resolve_dtype(self, torch_module):
        dtype = _dtype_from_option(
            torch_module,
            self.engine_options.get("dtype", self.engine_options.get("torch_dtype")),
        )
        if dtype is not None:
            return dtype

        dtype = torch_dtype_from_compute_type(torch_module, self.config.compute_type)
        if dtype is not None:
            return dtype

        device = self._resolve_device() or ""
        if str(device).startswith("cuda"):
            return getattr(torch_module, "float16", None)
        return getattr(torch_module, "float32", None)

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        torch_module = self._torch_module or self._load_torch()
        pipeline_cls = self._pipeline_cls or self._load_pipeline_cls()
        options = dict(self.pipeline_options)
        options.pop("model_card", None)
        options.setdefault("device", self._resolve_device())

        dtype = self._resolve_dtype(torch_module)
        if dtype is not None:
            options.setdefault("dtype", dtype)

        try:
            self.pipeline = pipeline_cls(model_card=self.model_card, **options)
        except ImportError as exc:
            raise TranscriptionEngineError(_dependency_error_message()) from exc
        return self.pipeline

    def _numpy(self):
        if self._numpy_module is None:
            self._numpy_module = self._load_numpy()
        return self._numpy_module

    def _check_duration(self, waveform, sample_rate):
        if self.max_audio_seconds is None:
            return
        if sample_rate <= 0:
            return
        duration = float(len(waveform)) / float(sample_rate)
        if duration >= self.max_audio_seconds:
            raise TranscriptionEngineError(
                "Meta Omnilingual ASR requires audio shorter than 40 seconds; "
                "received %.2f seconds." % duration
            )

    def _audio_input(self, audio):
        if isinstance(audio, (str, Path)):
            return str(audio)

        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            sample_rate = int(audio.get("sample_rate", self.sample_rate))
            if waveform is not None:
                self._check_duration(waveform, sample_rate)
            return audio

        numpy_module = self._numpy()
        values = getattr(audio, "values", audio)
        waveform = numpy_module.asarray(values, dtype=numpy_module.float32)

        if getattr(waveform, "ndim", 1) == 2:
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            elif waveform.shape[0] == 1:
                waveform = waveform[0]
            else:
                waveform = waveform.mean(axis=1).astype(numpy_module.float32)
        elif getattr(waveform, "ndim", 1) > 2:
            waveform = waveform.reshape(-1)

        self._check_duration(waveform, self.sample_rate)
        return {"waveform": waveform, "sample_rate": self.sample_rate}

    def language_code(self, language=None):
        selected = (
            language
            or self.engine_options.get("language")
            or self.engine_options.get("lang")
        )
        return _normalize_language_code(selected, self.language_aliases)

    def transcribe(self, audio, language=None, **params):
        pipeline = self._ensure_pipeline()
        audio_input = self._audio_input(audio)

        merged_params = dict(self.transcribe_options)
        merged_params.update(params)
        merged_params.setdefault("batch_size", self.batch_size)

        language_code = self.language_code(language)
        if _is_llm_model(self.model_card) and language_code:
            merged_params.setdefault("lang", [language_code])
        elif _is_ctc_model(self.model_card):
            merged_params.pop("lang", None)

        return pipeline.transcribe([audio_input], **merged_params)


class OmnilingualASREngine(BaseTranscriptionEngine):
    engine_name = "omnilingual_asr"

    def __init__(self, config, backend=None, backend_cls=None):
        super().__init__(config)
        self.backend = backend or (backend_cls or OmnilingualASRBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        if not isinstance(audio, (str, Path, dict)):
            audio = self._normalize_audio(audio)
        output = self.backend.transcribe(audio, language=language)
        language_code = self.backend.language_code(language)
        return TranscriptionResult(
            text=decode_to_text(output),
            info=TranscriptionInfo(
                language=language_code,
                language_probability=1.0 if language_code else 0.0,
            ),
        )
