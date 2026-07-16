"""
Adapts Qwen3-ASR models to the transcription engine interface.
"""

from importlib import import_module

from ._model_utils import language_from_output, text_from_output, torch_dtype_from_compute_type
from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"


QWEN_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


class Qwen3ASRBackend:
    """
    Wraps Qwen3-ASR model backends.
    """

    def __init__(self, config, model_factory=None, torch_module=None):
        """
        Initializes the Qwen3-ASR backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_name = config.model or DEFAULT_QWEN3_ASR_MODEL
        self.backend = self.engine_options.get("backend", "transformers")
        self.sample_rate = self.engine_options.get("sample_rate", 16000)
        self.transcribe_options = dict(self.engine_options.get("transcribe", {}))

        if model_factory is None:
            model_factory = self._load_model_factory()
        torch_module = torch_module or self._load_torch()

        model_options = dict(self.engine_options.get("model", {}))
        if config.download_root and "cache_dir" not in model_options:
            model_options["cache_dir"] = config.download_root

        if self.backend == "vllm":
            model_options.setdefault("model", self.model_name)
            self.model = model_factory.LLM(**model_options)
        else:
            dtype = torch_dtype_from_compute_type(
                torch_module,
                config.compute_type,
                default=(
                    getattr(torch_module, "float32", None)
                    if config.device == "cpu"
                    else getattr(torch_module, "bfloat16", None)
                ),
            )
            if dtype is not None:
                model_options.setdefault("dtype", dtype)
            model_options.setdefault("device_map", config.device)
            self.model = model_factory.from_pretrained(self.model_name, **model_options)

    @staticmethod
    def _load_model_factory():
        """
        Loads the optional Qwen3-ASR model factory.
        """

        try:
            qwen_asr = import_module("qwen_asr")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'qwen3_asr' transcription engine requires the optional "
                "'qwen-asr' package. Install it with 'pip install -U qwen-asr' "
                "or use 'pip install -U qwen-asr[vllm]' for the vLLM backend."
            ) from exc
        return qwen_asr.Qwen3ASRModel

    @staticmethod
    def _load_torch():
        """
        Loads torch for an optional backend.
        """

        try:
            return import_module("torch")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'qwen3_asr' transcription engine requires the optional 'torch' package."
            ) from exc

    def transcribe(self, audio, language=None, **params):
        """
        Runs Qwen3-ASR transcription for one audio input.
        """
        merged_params = dict(self.transcribe_options)
        merged_params.update(params)
        if not isinstance(audio, (str, list, tuple)):
            audio = (audio, self.sample_rate)
        return self.model.transcribe(
            audio=audio,
            language=language,
            **merged_params,
        )


class Qwen3ASREngine(BaseTranscriptionEngine):
    """
    Transcribes audio with Qwen3-ASR.
    """

    engine_name = "qwen3_asr"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Qwen3-ASR engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or Qwen3ASRBackend)(config)

    def _language_name(self, language):
        """
        Resolves a Qwen3-ASR language display name.
        """

        if not language:
            return None
        if len(language) == 2:
            return QWEN_LANGUAGE_NAMES.get(language.lower(), language)
        return language

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with Qwen3-ASR.
        """
        audio = self._normalize_audio(audio)
        options = self.config.engine_options or {}
        language_name = self._language_name(language or options.get("language"))
        params = {}
        if options.get("return_time_stamps") is not None:
            params["return_time_stamps"] = bool(options["return_time_stamps"])

        output = self.backend.transcribe(audio, language=language_name, **params)
        detected_language = language_from_output(output, language_name)
        return TranscriptionResult(
            text=text_from_output(output),
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if detected_language else 0.0,
            ),
        )
