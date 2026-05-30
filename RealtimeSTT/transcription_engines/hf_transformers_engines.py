"""Adapts selected Hugging Face Transformers ASR models."""

from importlib import import_module

from ._model_utils import (
    decode_to_text,
    model_kwargs_from_inputs,
    move_to_device,
    torch_dtype_from_compute_type,
)
from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_GRANITE_MODEL = "ibm-granite/granite-speech-4.1-2b"
DEFAULT_MOONSHINE_MODEL = "UsefulSensors/moonshine-streaming-medium"


def _load_transformers_classes(engine_name, class_names):
    """Loads required Transformers classes for an engine."""
    try:
        transformers = import_module("transformers")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The '%s' transcription engine requires the optional 'transformers' "
            "package and its model-specific dependencies." % engine_name
        ) from exc

    classes = []
    for class_name in class_names:
        try:
            classes.append(getattr(transformers, class_name))
        except AttributeError as exc:
            raise TranscriptionEngineError(
                "The '%s' transcription engine requires a newer transformers "
                "version that provides '%s'." % (engine_name, class_name)
            ) from exc
    return classes


def _load_torch(engine_name):
    """Loads torch for a Transformers-backed engine."""
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The '%s' transcription engine requires the optional 'torch' package."
            % engine_name
        ) from exc


def _with_cache_dir(options, download_root):
    """Adds a cache directory to model options when needed."""
    options = dict(options)
    if download_root and "cache_dir" not in options:
        options["cache_dir"] = download_root
    return options


class CohereTranscribeBackend:
    """
    Wraps the Cohere Transcribe Transformers backend.
    """

    def __init__(self, config, processor_cls=None, model_cls=None, torch_module=None):
        """
        Initializes the Cohere Transcribe Transformers backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_name = config.model or DEFAULT_COHERE_MODEL
        self.sample_rate = self.engine_options.get("sample_rate", 16000)
        self.processor_options = dict(self.engine_options.get("processor", {}))
        self.generate_options = dict(
            self.engine_options.get("generate", {"max_new_tokens": 256})
        )

        if processor_cls is None or model_cls is None:
            processor_cls, model_cls = _load_transformers_classes(
                "cohere_transcribe",
                ["AutoProcessor", "CohereAsrForConditionalGeneration"],
            )
        torch_module = torch_module or _load_torch("cohere_transcribe")

        self.processor = processor_cls.from_pretrained(
            self.model_name,
            **_with_cache_dir(self.processor_options, config.download_root),
        )
        model_options = _with_cache_dir(
            self.engine_options.get("model", {}),
            config.download_root,
        )
        model_options.setdefault(
            "device_map",
            self.engine_options.get("device_map", "auto" if config.device != "cpu" else "cpu"),
        )
        dtype = torch_dtype_from_compute_type(
            torch_module,
            config.compute_type,
            default=None,
        )
        if dtype is not None:
            model_options.setdefault("torch_dtype", dtype)
        self.model = model_cls.from_pretrained(self.model_name, **model_options)

    def transcribe(self, audio, language, **params):
        """
        Runs Cohere Transcribe generation for one audio input.
        """
        processor_kwargs = {
            "sampling_rate": self.sample_rate,
            "return_tensors": "pt",
            "language": language,
        }
        if "punctuation" in self.engine_options:
            processor_kwargs["punctuation"] = self.engine_options["punctuation"]
        processor_kwargs.update(self.engine_options.get("processor_call", {}))

        inputs = self.processor(audio, **processor_kwargs)
        inputs = move_to_device(
            inputs,
            getattr(self.model, "device", None),
            getattr(self.model, "dtype", None),
        )

        generate_kwargs = dict(self.generate_options)
        generate_kwargs.update(params)
        outputs = self.model.generate(
            **model_kwargs_from_inputs(inputs),
            **generate_kwargs,
        )

        decode_kwargs = {"skip_special_tokens": True}
        audio_chunk_index = getattr(inputs, "get", lambda name, default=None: default)(
            "audio_chunk_index",
            None,
        )
        if audio_chunk_index is not None:
            decode_kwargs["audio_chunk_index"] = audio_chunk_index
            decode_kwargs["language"] = language
        decode_kwargs.update(self.engine_options.get("decode", {}))
        return self.processor.decode(outputs, **decode_kwargs)


class CohereTranscribeEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with Cohere Transcribe models.
    """

    engine_name = "cohere_transcribe"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Cohere Transcribe engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or CohereTranscribeBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with Cohere Transcribe.
        """
        language = language or (self.config.engine_options or {}).get("language")
        if not language:
            raise TranscriptionEngineError(
                "The 'cohere_transcribe' engine requires a language code, e.g. "
                "language='en'. Cohere Transcribe does not auto-detect language."
            )
        audio = self._normalize_audio(audio)
        decoded = self.backend.transcribe(audio, language=language)
        return TranscriptionResult(
            text=decode_to_text(decoded),
            info=TranscriptionInfo(language=language, language_probability=1.0),
        )


class GraniteSpeechBackend:
    """
    Wraps the Granite Speech Transformers backend.
    """

    def __init__(self, config, processor_cls=None, model_cls=None, torch_module=None):
        """
        Initializes the Granite Speech Transformers backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_name = config.model or DEFAULT_GRANITE_MODEL
        self.device = self.engine_options.get("device", config.device)
        self.generate_options = dict(
            self.engine_options.get(
                "generate",
                {
                    "max_new_tokens": 200,
                    "do_sample": False,
                    "num_beams": config.beam_size if config.beam_size else 1,
                },
            )
        )

        if processor_cls is None or model_cls is None:
            processor_cls, model_cls = _load_transformers_classes(
                "granite_speech",
                ["AutoProcessor", "AutoModelForSpeechSeq2Seq"],
            )
        self.torch = torch_module or _load_torch("granite_speech")
        self.processor = processor_cls.from_pretrained(
            self.model_name,
            **_with_cache_dir(self.engine_options.get("processor", {}), config.download_root),
        )
        self.tokenizer = self.processor.tokenizer

        default_dtype = (
            getattr(self.torch, "float32", None)
            if self.device == "cpu"
            else getattr(self.torch, "bfloat16", None)
        )
        dtype = torch_dtype_from_compute_type(
            self.torch,
            config.compute_type,
            default=default_dtype,
        )
        model_options = _with_cache_dir(
            self.engine_options.get("model", {}),
            config.download_root,
        )
        model_options.setdefault("device_map", self.device)
        if dtype is not None:
            model_options.setdefault("torch_dtype", dtype)
        self.model = model_cls.from_pretrained(self.model_name, **model_options)

    def _audio_tensor(self, audio):
        if isinstance(audio, str):
            return audio
        tensor = self.torch.as_tensor(audio)
        if getattr(tensor, "ndim", None) == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def transcribe(self, audio, prompt, **params):
        """
        Runs Granite Speech generation for one audio input.
        """
        chat = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.processor(
            prompt_text,
            self._audio_tensor(audio),
            device=self.device,
            return_tensors="pt",
        )
        model_inputs = move_to_device(model_inputs, self.device)

        generate_kwargs = dict(self.generate_options)
        generate_kwargs.update(params)
        model_outputs = self.model.generate(
            **model_kwargs_from_inputs(model_inputs),
            **generate_kwargs,
        )

        try:
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        except Exception:
            new_tokens = model_outputs

        return self.tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )


class GraniteSpeechEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with Granite Speech models.
    """

    engine_name = "granite_speech"

    DEFAULT_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Granite Speech engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or GraniteSpeechBackend)(config)

    def _prompt(self, language, use_prompt):
        engine_options = self.config.engine_options or {}
        prompt = engine_options.get("prompt", self.DEFAULT_PROMPT)
        if language and engine_options.get("include_language_in_prompt", False):
            prompt = prompt + " Language: %s." % language
        initial_prompt = self._get_prompt(use_prompt)
        if isinstance(initial_prompt, str):
            prompt = prompt + " Context: %s" % initial_prompt
        elif initial_prompt:
            raise TranscriptionEngineError(
                "The 'granite_speech' engine only supports string initial_prompt values."
            )
        return prompt

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with Granite Speech.
        """
        audio = self._normalize_audio(audio)
        decoded = self.backend.transcribe(
            audio,
            prompt=self._prompt(language, use_prompt),
        )
        return TranscriptionResult(
            text=decode_to_text(decoded),
            info=TranscriptionInfo(
                language=language,
                language_probability=1.0 if language else 0.0,
            ),
        )


class MoonshineBackend:
    """
    Wraps the Moonshine streaming Transformers backend.
    """

    def __init__(self, config, processor_cls=None, model_cls=None, torch_module=None):
        """
        Initializes the Moonshine Transformers backend.
        """
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.model_name = config.model or DEFAULT_MOONSHINE_MODEL
        self.device = self.engine_options.get("device", config.device)
        self.generate_options = dict(self.engine_options.get("generate", {}))

        if processor_cls is None or model_cls is None:
            model_cls, processor_cls = _load_transformers_classes(
                "moonshine",
                ["MoonshineStreamingForConditionalGeneration", "AutoProcessor"],
            )
        self.torch = torch_module or _load_torch("moonshine")

        self.model = model_cls.from_pretrained(
            self.model_name,
            **_with_cache_dir(self.engine_options.get("model", {}), config.download_root),
        )
        dtype = torch_dtype_from_compute_type(
            self.torch,
            config.compute_type,
            default=(
                getattr(self.torch, "float32", None)
                if self.device == "cpu"
                else getattr(self.torch, "float16", None)
            ),
        )
        self.model = move_to_device(self.model, self.device)
        self.model = move_to_device(self.model, dtype=dtype)

        self.processor = processor_cls.from_pretrained(
            self.model_name,
            **_with_cache_dir(self.engine_options.get("processor", {}), config.download_root),
        )
        self.sample_rate = self.engine_options.get(
            "sample_rate",
            getattr(self.processor.feature_extractor, "sampling_rate", 16000),
        )
        self.dtype = dtype

    def _default_max_length(self, inputs):
        try:
            token_limit_factor = 6.5 / self.sample_rate
            seq_lens = inputs.attention_mask.sum(dim=-1)
            return int((seq_lens * token_limit_factor).max().item())
        except Exception:
            return None

    def transcribe(self, audio, **params):
        """
        Runs Moonshine generation for one audio input.
        """
        inputs = self.processor(
            audio,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )
        inputs = move_to_device(inputs, self.device, self.dtype)

        generate_kwargs = dict(self.generate_options)
        generate_kwargs.update(params)
        if "max_length" not in generate_kwargs and "max_new_tokens" not in generate_kwargs:
            max_length = self._default_max_length(inputs)
            if max_length is not None:
                generate_kwargs["max_length"] = max_length
            else:
                generate_kwargs["max_new_tokens"] = 256

        generated_ids = self.model.generate(
            **model_kwargs_from_inputs(inputs),
            **generate_kwargs,
        )
        try:
            generated_ids = generated_ids[0]
        except Exception:
            pass
        return self.processor.decode(generated_ids, skip_special_tokens=True)


class MoonshineEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with Moonshine models.
    """

    engine_name = "moonshine"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Moonshine engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or MoonshineBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio with Moonshine.
        """
        if language and language.lower() not in ("en", "english"):
            raise TranscriptionEngineError(
                "The 'moonshine' engine currently supports English transcription only."
            )
        audio = self._normalize_audio(audio)
        decoded = self.backend.transcribe(audio)
        return TranscriptionResult(
            text=decode_to_text(decoded),
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )
