"""Adapts faster-whisper models to the transcription engine interface."""

from importlib import import_module

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


def _load_faster_whisper():
    """Loads faster-whisper and its optional batched inference pipeline."""
    try:
        faster_whisper = import_module("faster_whisper")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'faster_whisper' transcription engine requires the optional "
            "'faster-whisper' package. Install it with "
            "'pip install \"RealtimeSTT[faster-whisper]\"' or select a "
            "different transcription engine."
        ) from exc

    return faster_whisper, faster_whisper.BatchedInferencePipeline


class FasterWhisperEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with faster-whisper.
    """

    engine_name = "faster_whisper"

    def __init__(self, config):
        """
        Initializes the faster-whisper model.
        """
        super().__init__(config)
        faster_whisper, batched_inference_pipeline = _load_faster_whisper()
        model = faster_whisper.WhisperModel(
            model_size_or_path=self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            device_index=self.config.gpu_device_index,
            download_root=self.config.download_root,
        )
        if self.config.batch_size > 0:
            model = batched_inference_pipeline(model=model)
        self.model = model

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio and returns normalized faster-whisper output.
        """
        audio = self._normalize_audio(audio)
        kwargs = {
            "language": language if language else None,
            "beam_size": self.config.beam_size,
            "initial_prompt": self._get_prompt(use_prompt),
            "suppress_tokens": self.config.suppress_tokens,
            "vad_filter": self.config.vad_filter,
        }
        if self.config.batch_size > 0:
            kwargs["batch_size"] = self.config.batch_size

        segments, info = self.model.transcribe(audio, **kwargs)
        text = " ".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", 0.0),
            ),
        )
