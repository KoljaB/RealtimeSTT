"""
Exports the Granite Speech engine adapter.
"""

from . import hf_transformers_engines as _hf
from .hf_transformers_engines import (
    DEFAULT_GRANITE_MODEL,
)


class GraniteSpeechBackend(_hf.GraniteSpeechBackend):
    """
    Uses the shared Transformers Granite backend implementation.
    """

    pass


class GraniteSpeechEngine(_hf.GraniteSpeechEngine):
    """
    Uses the shared Transformers Granite engine implementation.
    """

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Granite wrapper with the shared backend class.
        """
        super().__init__(
            config,
            backend=backend,
            backend_cls=backend_cls or GraniteSpeechBackend,
        )


__all__ = [
    "DEFAULT_GRANITE_MODEL",
    "GraniteSpeechBackend",
    "GraniteSpeechEngine",
]
