"""Adapts OpenAI Whisper Python models to the engine interface."""

from importlib import import_module

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


class OpenAIWhisperBackend:
    """
    Wraps the openai-whisper model object.
    """

    def __init__(self, config, whisper_module=None):
        """
        Loads an openai-whisper model backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.transcribe_options = dict(self.engine_options.get("transcribe", {}))
        whisper_module = whisper_module or self._load_whisper_module()

        model_options = dict(self.engine_options.get("model", {}))
        model_options.update(self.engine_options.get("load_model", {}))
        if config.download_root and "download_root" not in model_options:
            model_options["download_root"] = config.download_root
        if config.device and "device" not in model_options:
            model_options["device"] = config.device

        self.model = whisper_module.load_model(config.model, **model_options)

    @staticmethod
    def _load_whisper_module():
        """Loads the optional openai-whisper module."""
        try:
            return import_module("whisper")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'openai_whisper' transcription engine requires the optional "
                "'openai-whisper' package. Install it with 'pip install openai-whisper' "
                "or select a different transcription engine."
            ) from exc

    def transcribe(self, audio, **params):
        """
        Runs openai-whisper transcription with merged options.
        """
        merged_params = dict(params)
        merged_params.update(self.transcribe_options)
        return self.model.transcribe(audio, **merged_params)


class OpenAIWhisperEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with OpenAI Whisper Python models.
    """

    engine_name = "openai_whisper"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the OpenAI Whisper engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or OpenAIWhisperBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio and returns normalized Whisper output.
        """
        audio = self._normalize_audio(audio)
        params = {
            "language": language if language else None,
            "verbose": False,
        }

        if self.config.beam_size and self.config.beam_size > 1:
            params["beam_size"] = self.config.beam_size

        prompt = self._get_prompt(use_prompt)
        if isinstance(prompt, str):
            params["initial_prompt"] = prompt
        elif prompt:
            raise TranscriptionEngineError(
                "The 'openai_whisper' transcription engine only supports string "
                "initial_prompt values."
            )

        if self.config.suppress_tokens is not None:
            params["suppress_tokens"] = self.config.suppress_tokens

        compute_type = (self.config.compute_type or "").lower().replace("-", "_")
        if compute_type in ("float16", "fp16", "half"):
            params["fp16"] = True
        elif compute_type in ("float32", "fp32", "int8"):
            params["fp16"] = False

        result = self.backend.transcribe(audio, **params)
        if isinstance(result, dict):
            text = result.get("text", "")
            detected_language = result.get("language", language)
        else:
            text = getattr(result, "text", str(result))
            detected_language = getattr(result, "language", language)

        return TranscriptionResult(
            text=str(text).strip(),
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if language else 0.0,
            ),
        )
