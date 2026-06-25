"""
Adapts faster-whisper models to the transcription engine interface.
"""

from importlib import import_module

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


def _load_faster_whisper():
    """
    Loads faster-whisper and its optional batched inference pipeline.
    """
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

    def _batched_clip_timestamps(self, audio):
        """
        Returns full-audio clip windows for batched decoding without VAD.
        """

        wrapped_model = getattr(self.model, "model", self.model)
        feature_extractor = getattr(wrapped_model, "feature_extractor", None)
        sample_rate = getattr(feature_extractor, "sampling_rate", 16000)
        chunk_length = getattr(feature_extractor, "chunk_length", 30)
        duration = audio.size / float(sample_rate)
        clips = []
        start = 0.0
        while start < duration:
            end = min(start + chunk_length, duration)
            clips.append({"start": start, "end": end})
            start = end
        return clips or [{"start": 0.0, "end": duration}]

    def transcribe(self, audio, language=None, use_prompt=True, word_timestamps=False):
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
        if word_timestamps:
            kwargs["word_timestamps"] = True
        if self.config.batch_size > 0:
            kwargs["batch_size"] = self.config.batch_size
            if not self.config.vad_filter:
                kwargs["clip_timestamps"] = self._batched_clip_timestamps(audio)

        segments, info = self.model.transcribe(audio, **kwargs)
        segments = list(segments)
        text = " ".join(segment.text for segment in segments).strip()
        metadata = {}
        if word_timestamps:
            metadata["words"] = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                }
                for segment in segments
                for word in (getattr(segment, "words", None) or [])
            ]
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", 0.0),
            ),
            metadata=metadata,
        )
