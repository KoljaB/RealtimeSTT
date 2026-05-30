"""Adapts pywhispercpp models to the transcription engine interface."""

from importlib import import_module

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


class PyWhisperCppBackend:
    """
    Wraps a pywhispercpp model instance.
    """

    def __init__(self, config, model_cls=None):
        """
        Loads a pywhispercpp model backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.transcribe_options = dict(self.engine_options.get("transcribe", {}))
        model_cls = model_cls or self._load_model_class()

        model_kwargs = {
            "params_sampling_strategy": 1 if config.beam_size and config.beam_size > 1 else 0,
            "print_progress": False,
            "print_realtime": False,
            "print_timestamps": False,
        }
        model_kwargs.update(self.engine_options.get("model", {}))
        if config.download_root:
            model_kwargs["models_dir"] = config.download_root

        self.model = model_cls(config.model, **model_kwargs)

    @staticmethod
    def _load_model_class():
        """Loads the optional pywhispercpp model class."""
        try:
            module = import_module("pywhispercpp.model")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'whisper_cpp' transcription engine requires the optional "
                "'pywhispercpp' package. Install it with 'pip install pywhispercpp' "
                "or select a different transcription engine."
            ) from exc
        return module.Model

    def transcribe(self, audio, **params):
        """
        Runs pywhispercpp transcription with merged options.
        """
        merged_params = dict(params)
        merged_params.update(self.transcribe_options)
        n_processors = merged_params.pop("n_processors", None)
        return self.model.transcribe(audio, n_processors=n_processors, **merged_params)


class WhisperCppEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with pywhispercpp.
    """

    engine_name = "whisper_cpp"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the pywhispercpp engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or PyWhisperCppBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio and returns normalized pywhispercpp output.
        """
        audio = self._normalize_audio(audio)
        params = {
            "language": language if language else None,
            "print_progress": False,
            "print_realtime": False,
            "print_timestamps": False,
        }
        if self.config.beam_size and self.config.beam_size > 1:
            params["beam_search"] = {"beam_size": self.config.beam_size, "patience": -1.0}
        else:
            params["greedy"] = {"best_of": 1}

        prompt = self._get_prompt(use_prompt)
        if isinstance(prompt, str):
            params["initial_prompt"] = prompt
        elif prompt:
            prompt_tokens = tuple(prompt)
            params["prompt_tokens"] = prompt_tokens
            params["prompt_n_tokens"] = len(prompt_tokens)

        segments = self.backend.transcribe(audio, **params)
        text = " ".join(str(getattr(segment, "text", segment)) for segment in segments).strip()

        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=language if language else None,
                language_probability=1.0 if language else 0.0,
            ),
        )
