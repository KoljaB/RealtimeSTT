"""Exports the Moonshine engine adapter."""

from . import hf_transformers_engines as _hf
from .hf_transformers_engines import (
    DEFAULT_MOONSHINE_MODEL,
)


class MoonshineBackend(_hf.MoonshineBackend):
    """
    Uses the shared Transformers Moonshine backend implementation.
    """

    pass


class MoonshineEngine(_hf.MoonshineEngine):
    """
    Uses the shared Transformers Moonshine engine implementation.
    """

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Moonshine wrapper and optional sherpa backend.
        """
        engine_options = config.engine_options or {}
        backend_name = str(engine_options.get("backend", "")).lower().replace("-", "_")
        if backend is None and backend_cls is None and backend_name == "sherpa_onnx":
            from .sherpa_onnx_engine import SherpaOnnxMoonshineBackend

            backend_cls = SherpaOnnxMoonshineBackend
        super().__init__(
            config,
            backend=backend,
            backend_cls=backend_cls or MoonshineBackend,
        )


__all__ = [
    "DEFAULT_MOONSHINE_MODEL",
    "MoonshineBackend",
    "MoonshineEngine",
]
