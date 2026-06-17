"""
Adapts FunASR to the transcription engine interface.
"""

import re
from importlib import import_module

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)

def _load_FunASR_():
    try:
        funasr = import_module("funasr")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'FunASR' transcription engine requires the optional "
            "FunASR' package. Install it with "
            "'pip install funasr' or select a "
            "different transcription engine."
        ) from exc

    return funasr

_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")

def _clean_text(text):
    """
        Strips SenseVoice special tags (e.g. ``<|zh|><|NEUTRAL|>``) so the
        engine returns clean text. Uses FunASR's official postprocessor
        when available and falls back to a tag regex otherwise. A no-op for
        models that already emit plain text (Paraformer, Fun-ASR).
    """
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        return rich_transcription_postprocess(text)
    except Exception:
        return _SENSEVOICE_TAG.sub("", text)

class FunASREngine(BaseTranscriptionEngine):

    engine_name = "funasr"

    def __init__(self, config):
        """
            Initializes the funasr model.
        """

        super().__init__(config)

        funasr = _load_FunASR_()

        kwargs = {
            "model": self.config.model,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "beam_size": self.config.beam_size,
        }

        if self.config.engine_options is not None:
            vad_model = self.config.engine_options.get("vad_model")

            if self.config.engine_options.get("vad_filter") is not None and self.config.engine_options.get("vad_filter") and vad_model is not None:
                kwargs["vad_model"] = vad_model

        self.model = funasr.AutoModel(**kwargs)


    def transcribe(self, audio, language=None, use_prompt=False, **kwargs):
        """
            Transcribes audio and returns funasr output.
        """
        audio = self._normalize_audio(audio)

        res = self.model.generate(input=audio)

        text = _clean_text(res[0]["text"]).strip() if res else ""

        return TranscriptionResult(
            text=text,
        )

