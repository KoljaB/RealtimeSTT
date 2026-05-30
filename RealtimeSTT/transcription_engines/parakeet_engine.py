"""
Adapts NVIDIA Parakeet backends to the transcription engine interface.
"""

import os
import tempfile
from importlib import import_module
from pathlib import Path

from ._model_utils import language_from_output, text_from_output
from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetNeMoBackend:
    """
    Wraps NVIDIA NeMo ASR models for Parakeet transcription.
    """

    def __init__(self, config, asr_model_cls=None, soundfile_module=None):
        """
        Initializes the NeMo Parakeet backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_name = config.model or DEFAULT_PARAKEET_MODEL
        self.sample_rate = self.engine_options.get("sample_rate", 16000)
        self.transcribe_options = dict(self.engine_options.get("transcribe", {}))
        self.soundfile_module = soundfile_module

        asr_model_cls = asr_model_cls or self._load_asr_model_class()
        model_options = dict(self.engine_options.get("model", {}))
        self.model = asr_model_cls.from_pretrained(
            model_name=self.model_name,
            **model_options,
        )
        if config.device and hasattr(self.model, "to"):
            self.model.to(config.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _load_asr_model_class():
        """
        Loads the optional NeMo ASR model class.
        """

        try:
            nemo_asr = import_module("nemo.collections.asr")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'parakeet' transcription engine requires NVIDIA NeMo. "
                "Install it with \"pip install -U nemo_toolkit['asr']\". "
                "NeMo ASR is primarily supported on Linux; use WSL2 on Windows "
                "for real-model testing."
            ) from exc
        return nemo_asr.models.ASRModel

    def _load_soundfile(self):
        """
        Loads the optional soundfile module.
        """

        if self.soundfile_module is not None:
            return self.soundfile_module
        try:
            self.soundfile_module = import_module("soundfile")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'parakeet' transcription engine needs 'soundfile' to pass "
                "in-memory audio to NeMo. Install it with 'pip install soundfile'."
            ) from exc
        return self.soundfile_module

    def _audio_paths(self, audio):
        """
        Writes audio inputs to temporary WAV files when needed.
        """

        if isinstance(audio, (str, Path)):
            return [str(audio)], None

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._load_soundfile().write(path, audio, self.sample_rate)
        return [path], path

    def transcribe(self, audio, **params):
        """
        Runs Parakeet transcription for one audio input.
        """
        merged_params = dict(params)
        merged_params.update(self.transcribe_options)
        paths, temporary_path = self._audio_paths(audio)
        try:
            return self.model.transcribe(paths, **merged_params)
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


class ParakeetEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with a Parakeet backend.
    """

    engine_name = "parakeet"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Parakeet engine backend.
        """
        super().__init__(config)
        engine_options = config.engine_options or {}
        backend_name = str(engine_options.get("backend", "")).lower().replace("-", "_")
        if backend is None and backend_cls is None and backend_name == "sherpa_onnx":
            from .sherpa_onnx_engine import SherpaOnnxParakeetBackend

            backend_cls = SherpaOnnxParakeetBackend
        self.backend = backend or (backend_cls or ParakeetNeMoBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with the configured Parakeet backend.
        """
        audio = self._normalize_audio(audio)
        params = {}
        if self.config.batch_size and self.config.batch_size > 0:
            params["batch_size"] = self.config.batch_size

        engine_options = self.config.engine_options or {}
        if engine_options.get("timestamps") is not None:
            params["timestamps"] = bool(engine_options["timestamps"])

        output = self.backend.transcribe(audio, **params)
        detected_language = language_from_output(output, language)
        return TranscriptionResult(
            text=text_from_output(output),
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if detected_language else 0.0,
            ),
        )
