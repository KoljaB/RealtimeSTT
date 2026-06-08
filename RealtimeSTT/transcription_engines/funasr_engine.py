"""
Adapts FunASR to the transcription engine interface.
"""

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

class FunASREngine(BaseTranscriptionEngine):

    def __init__(self, config):
        super().__init__(config)

        funasr = _load_FunASR_()

        self.model = funasr.AutoModel(
            model=self.config.model,
            device=self.config.device,
        )

    def transcribe(self, audio, language=None, use_prompt=False, **kwargs):
        audio = self._normalize_audio(audio)

        res = self.model.generate(input=audio)

        text = res[0]["text"].strip() if res else ""
        language = res[0]["text"].split(">")[0].strip("<") if res else "None"

        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(language=language),
        )

