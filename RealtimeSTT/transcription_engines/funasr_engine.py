"""
Adapts FunASR models to the transcription engine interface.
"""

import os
from importlib import import_module
from pathlib import Path

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_FUNASR_MODEL = "iic/SenseVoiceSmall"
WHISPER_DEFAULT_MODEL_NAMES = {"tiny"}

AUTO_MODEL_OPTION_KEYS = {
    "batch_size",
    "beam_size",
    "check_latest",
    "disable_pbar",
    "disable_update",
    "hub",
    "log_level",
    "model_conf",
    "model_path",
    "model_revision",
    "ncpu",
    "ngpu",
    "punc_kwargs",
    "punc_model",
    "punc_model_revision",
    "remote_code",
    "spk_kwargs",
    "spk_mode",
    "spk_model",
    "spk_model_revision",
    "trust_remote_code",
    "vad_kwargs",
    "vad_model",
    "vad_model_revision",
}

GENERATE_OPTION_KEYS = {
    "batch_size_s",
    "batch_size_threshold_s",
    "cache",
    "data_type",
    "en_post_proc",
    "hotword",
    "is_final",
    "language",
    "merge_length_s",
    "merge_vad",
    "output_timestamp",
    "progress_callback",
    "return_raw_text",
    "return_spk_res",
    "return_time_stamps",
    "sentence_timestamp",
    "use_itn",
}


def _load_funasr_module():
    """
    Loads the optional FunASR package.
    """
    try:
        return import_module("funasr")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'funasr' transcription engine requires the optional 'funasr' "
            "package. Install it with 'pip install \"RealtimeSTT[funasr]\"' "
            "or 'pip install funasr', or select a different transcription engine."
        ) from exc


def _cache_root(download_root):
    """
    Returns a normalized cache root for FunASR's hub clients.
    """
    if not download_root:
        return None
    return str(Path(download_root).expanduser())


def _apply_cache_environment(download_root):
    """
    Sets hub cache environment defaults used by FunASR downloads.
    """
    root = _cache_root(download_root)
    if not root:
        return
    os.environ.setdefault("MODELSCOPE_CACHE", root)
    os.environ.setdefault("HF_HOME", root)


def _funasr_device(device, gpu_device_index):
    """
    Converts shared device settings to a FunASR device string.
    """
    if device != "cuda":
        return device
    if isinstance(gpu_device_index, (list, tuple)):
        gpu_device_index = gpu_device_index[0] if gpu_device_index else 0
    return "cuda:%s" % gpu_device_index


def _resolve_model_name(config, engine_options):
    """
    Resolves the model name that should be passed to FunASR.
    """
    if "model_name" in engine_options:
        return engine_options["model_name"]

    model_name = config.model
    hub = str(engine_options.get("hub", "ms")).lower()
    use_default = engine_options.get("use_default_model", True)
    if (
        use_default
        and hub not in ("openai",)
        and (not model_name or str(model_name).lower() in WHISPER_DEFAULT_MODEL_NAMES)
    ):
        return engine_options.get("default_model", DEFAULT_FUNASR_MODEL)

    return model_name


def _normalize_text(text):
    """
    Trims surrounding whitespace from one FunASR text item.
    """
    return text.strip()


def _result_items(result):
    """
    Returns a list of model result items from FunASR output.
    """
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def _result_text(item):
    """
    Extracts text from one FunASR result item.
    """
    if isinstance(item, dict):
        text = item.get("text", "")
    else:
        text = getattr(item, "text", item)
    return "" if text is None else str(text)


def _result_language(item, text):
    """
    Extracts language metadata from one FunASR result item.
    """
    if isinstance(item, dict):
        for key in ("language", "lang"):
            if item.get(key):
                return str(item[key])
    else:
        for key in ("language", "lang"):
            value = getattr(item, key, None)
            if value:
                return str(value)
    return None


def _dict_option(engine_options, key):
    """
    Returns a dictionary option or raises a clear configuration error.
    """
    value = engine_options.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TranscriptionEngineError(
            "The 'funasr' transcription engine option '%s' must be a dict." % key
        )
    return dict(value)


def decode_funasr_result(result):
    """
    Converts FunASR output into text and optional language metadata.
    """
    text_parts = []
    detected_language = None

    for item in _result_items(result):
        text = _result_text(item)
        if detected_language is None:
            detected_language = _result_language(item, text)
        text = _normalize_text(text)
        if text:
            text_parts.append(text)

    return " ".join(text_parts).strip(), detected_language


class FunASRBackend:
    """
    Wraps a FunASR AutoModel instance.
    """

    def __init__(self, config, funasr_module=None, model_cls=None):
        """
        Initializes the FunASR model backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_name = _resolve_model_name(config, self.engine_options)
        self.generate_options = self._generate_options()

        _apply_cache_environment(config.download_root)
        funasr_module = funasr_module or _load_funasr_module()
        model_cls = model_cls or funasr_module.AutoModel
        self.model = model_cls(**self._model_options())

    def _model_options(self):
        """
        Builds keyword arguments for FunASR AutoModel.
        """
        model_options = _dict_option(self.engine_options, "model")
        model_options.update(_dict_option(self.engine_options, "auto_model"))

        model_options.setdefault("model", self.model_name)
        if self.config.device and "device" not in model_options:
            model_options["device"] = _funasr_device(
                self.config.device,
                self.config.gpu_device_index,
            )
        if self.config.batch_size and "batch_size" not in model_options:
            model_options["batch_size"] = self.config.batch_size
        if self.config.beam_size and "beam_size" not in model_options:
            model_options["beam_size"] = self.config.beam_size

        for key in AUTO_MODEL_OPTION_KEYS:
            if key in self.engine_options and key not in model_options:
                model_options[key] = self.engine_options[key]

        model_options.setdefault("disable_update", True)
        model_options.setdefault("disable_pbar", True)

        vad_filter = self.engine_options.get("vad_filter", self.config.vad_filter)
        if not vad_filter and "vad_model" in model_options:
            model_options.pop("vad_model")

        return model_options

    def _generate_options(self):
        """
        Builds default runtime options for AutoModel.generate().
        """
        generate_options = _dict_option(self.engine_options, "generate")
        generate_options.update(_dict_option(self.engine_options, "transcribe"))

        for key in GENERATE_OPTION_KEYS:
            if key in self.engine_options and key not in generate_options:
                generate_options[key] = self.engine_options[key]

        return generate_options

    def transcribe(self, audio, **params):
        """
        Runs FunASR generation with merged runtime options.
        """
        generate_options = dict(self.generate_options)
        generate_options.update(params)
        return self.model.generate(input=audio, **generate_options)


class FunASREngine(BaseTranscriptionEngine):
    """
    Transcribes audio with FunASR models.
    """

    engine_name = "funasr"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the FunASR engine backend.
        """
        super().__init__(config)
        self.engine_options = dict(config.engine_options or {})
        self.backend = backend or (backend_cls or FunASRBackend)(config)

    def _runtime_language(self, language):
        """
        Returns the request language that should be sent to FunASR.
        """
        return language or self.engine_options.get("language")

    def _prompt_hotword(self, use_prompt):
        """
        Maps string initial prompts to FunASR hotwords.
        """
        prompt = self._get_prompt(use_prompt)
        if isinstance(prompt, str):
            return prompt
        if prompt:
            raise TranscriptionEngineError(
                "The 'funasr' transcription engine only supports string "
                "initial_prompt values."
            )
        return None

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        """
        Transcribes audio and returns normalized FunASR output.
        """
        audio = self._normalize_audio(audio)
        params = dict(kwargs)

        runtime_language = self._runtime_language(language)
        if runtime_language:
            params["language"] = runtime_language

        hotword = self._prompt_hotword(use_prompt)
        if hotword and "hotword" not in params:
            params["hotword"] = hotword

        decoded = self.backend.transcribe(audio, **params)
        text, detected_language = decode_funasr_result(decoded)

        result_language = detected_language
        if runtime_language and str(runtime_language).lower() != "auto":
            result_language = runtime_language

        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=result_language,
                language_probability=1.0 if result_language else 0.0,
            ),
        )
