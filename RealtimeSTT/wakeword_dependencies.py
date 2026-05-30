"""
Wake word backend normalization and optional dependency loading.
"""

from importlib import import_module

from ._audio_recorder.wakeword import (
    OPENWAKEWORD_BACKENDS,
    PORCUPINE_WAKEWORD_BACKENDS,
    _load_openwakeword_modules as _internal_load_openwakeword_modules,
    _load_porcupine_module as _internal_load_porcupine_module,
    _normalize_wakeword_backend as _internal_normalize_wakeword_backend,
)


def _normalize_wakeword_backend(wakeword_backend, wake_words):
    return _internal_normalize_wakeword_backend(wakeword_backend, wake_words)


def _load_porcupine_module(importer=None):
    if importer is None:
        importer = import_module
    return _internal_load_porcupine_module(importer=importer)


def _load_openwakeword_modules(importer=None):
    if importer is None:
        importer = import_module
    return _internal_load_openwakeword_modules(importer=importer)
