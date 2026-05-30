"""Exports the Cohere Transcribe engine adapter."""

from . import hf_transformers_engines as _hf
from .hf_transformers_engines import (
    DEFAULT_COHERE_MODEL,
)


class CohereTranscribeBackend(_hf.CohereTranscribeBackend):
    """
    Uses the shared Transformers Cohere backend implementation.
    """

    pass


class CohereTranscribeEngine(_hf.CohereTranscribeEngine):
    """
    Uses the shared Transformers Cohere engine implementation.
    """

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Cohere wrapper with the shared backend class.
        """
        super().__init__(
            config,
            backend=backend,
            backend_cls=backend_cls or CohereTranscribeBackend,
        )


__all__ = [
    "DEFAULT_COHERE_MODEL",
    "CohereTranscribeBackend",
    "CohereTranscribeEngine",
]
