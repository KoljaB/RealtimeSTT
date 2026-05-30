"""Creates transcription engine instances by configured backend name."""

from importlib import import_module

from .base import TranscriptionEngineConfig, UnsupportedTranscriptionEngineError


ENGINE_CLASS_PATHS = {
    "faster_whisper": (".faster_whisper_engine", "FasterWhisperEngine"),
    "whisper_cpp": (".whisper_cpp_engine", "WhisperCppEngine"),
    "openai_whisper": (".openai_whisper_engine", "OpenAIWhisperEngine"),
    "openai_api": (".openai_api_engine", "OpenAIAPIEngine"),
    "parakeet": (".parakeet_engine", "ParakeetEngine"),
    "nvidia_parakeet": (".parakeet_engine", "ParakeetEngine"),
    "cohere_transcribe": (".cohere_transcribe_engine", "CohereTranscribeEngine"),
    "cohere": (".cohere_transcribe_engine", "CohereTranscribeEngine"),
    "granite_speech": (".granite_speech_engine", "GraniteSpeechEngine"),
    "granite": (".granite_speech_engine", "GraniteSpeechEngine"),
    "qwen3_asr": (".qwen3_asr_engine", "Qwen3ASREngine"),
    "qwen_asr": (".qwen3_asr_engine", "Qwen3ASREngine"),
    "omnilingual_asr": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "omnilingual": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "meta_omnilingual_asr": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "omni_asr": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "moonshine": (".moonshine_engine", "MoonshineEngine"),
    "moonshine_streaming": (".moonshine_engine", "MoonshineEngine"),
    "sherpa_onnx_parakeet": (".sherpa_onnx_engine", "SherpaOnnxParakeetEngine"),
    "sherpa_parakeet": (".sherpa_onnx_engine", "SherpaOnnxParakeetEngine"),
    "parakeet_sherpa_onnx": (".sherpa_onnx_engine", "SherpaOnnxParakeetEngine"),
    "sherpa_onnx_moonshine": (".sherpa_onnx_engine", "SherpaOnnxMoonshineEngine"),
    "sherpa_moonshine": (".sherpa_onnx_engine", "SherpaOnnxMoonshineEngine"),
    "moonshine_sherpa_onnx": (".sherpa_onnx_engine", "SherpaOnnxMoonshineEngine"),
    "kroko_onnx": (".kroko_onnx_engine", "KrokoOnnxEngine"),
    "kroko": (".kroko_onnx_engine", "KrokoOnnxEngine"),
    "banafo_kroko": (".kroko_onnx_engine", "KrokoOnnxEngine"),
}


def _load_engine_class(name):
    """Loads the engine class registered for a normalized name."""
    module_name, class_name = ENGINE_CLASS_PATHS[name]
    module = import_module(module_name, package=__package__)
    return getattr(module, class_name)


def create_transcription_engine(name, config: TranscriptionEngineConfig):
    """
    Creates a transcription engine for a configured backend name.
    """
    normalized_name = (name or "faster_whisper").strip().lower().replace("-", "_")
    if normalized_name not in ENGINE_CLASS_PATHS:
        available_engines = ", ".join(sorted(ENGINE_CLASS_PATHS))
        raise UnsupportedTranscriptionEngineError(
            f"Unsupported transcription engine '{name}'. Available engines: {available_engines}"
        )
    engine_cls = _load_engine_class(normalized_name)
    return engine_cls(config)


def get_supported_transcription_engines():
    """
    Returns the sorted list of supported engine names.
    """
    return sorted(ENGINE_CLASS_PATHS)
