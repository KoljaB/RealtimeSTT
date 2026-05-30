"""Defines the placeholder OpenAI API transcription engine."""

from .base import BaseTranscriptionEngine, TranscriptionEngineError


class OpenAIAPIEngine(BaseTranscriptionEngine):
    """
    Reports that the OpenAI API engine is not wired yet.
    """

    engine_name = "openai_api"

    def __init__(self, config):
        """
        Rejects initialization until API request handling exists.
        """
        super().__init__(config)
        raise TranscriptionEngineError(
            "The 'openai_api' transcription engine has not been wired yet. "
            "Add API request handling before selecting it."
        )

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Reports that API transcription is unavailable.
        """
        raise TranscriptionEngineError("The 'openai_api' transcription engine is not available.")
