# Custom Transcription Engines

Custom transcription engines let you provide your own speech-to-text backend
while RealtimeSTT still handles recording, voice activity detection, buffering,
callbacks, realtime update scheduling, and text formatting.

This page covers two extension paths:

| Path | Best for | How it integrates |
| --- | --- | --- |
| User-owned engine | Application code, private engines, engines in another package | Pass an object or callable to `transcription_executor` and optionally `realtime_transcription_executor`. |
| Built-in style engine | Engines contributed to RealtimeSTT itself | Add an adapter under `RealtimeSTT.transcription_engines` and register it in the factory. |

Prefer the user-owned engine path unless you are contributing the engine to
RealtimeSTT.

## Public Engine Contract

Import the lightweight engine authoring API from `RealtimeSTT.engines`:

```python
from RealtimeSTT.engines import (
    BaseEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)
```

`BaseEngine` is the public short name for the same base class used by
RealtimeSTT's built-in transcription adapters. The older descriptive name,
`BaseTranscriptionEngine`, remains available for compatibility.

The engine contract is intentionally small:

| Method or attribute | Required | Purpose |
| --- | --- | --- |
| `engine_name` | No | Human-readable backend name used in logs and metadata. |
| `transcribe(audio, language=None, use_prompt=True, **kwargs)` | Yes | Runs one final or full-buffer transcription request. |
| `warmup(audio)` | No | Optional startup warmup. The base implementation calls `transcribe()` with English and no prompt. |
| `supports_streaming` | No | Set to `True` when `create_streaming_session()` returns a real streaming session. |
| `create_streaming_session(language=None, use_prompt=True)` | Only for streaming | Creates per-utterance incremental decoding state. |

`audio` is normally a mono NumPy float array at the recorder sample rate. The
default RealtimeSTT recorder sample rate is 16 kHz. A custom engine may resample
internally if its backend requires another rate.

`TranscriptionResult` is the normalized return object:

```python
TranscriptionResult(
    text="recognized text",
    info=TranscriptionInfo(
        language="en",
        language_probability=1.0,
    ),
    metadata={},
)
```

Use `metadata` for backend-specific structured data, such as word timings.
RealtimeSTT stores final-result metadata in `recorder.last_transcription_metadata`.

## Minimal Final Engine

Implement `transcribe()` and return `TranscriptionResult`:

```python
from RealtimeSTT import AudioToTextRecorder
from RealtimeSTT.engines import (
    BaseEngine,
    TranscriptionEngineConfig,
    TranscriptionInfo,
    TranscriptionResult,
)


class MyEngine(BaseEngine):
    engine_name = "my_engine"

    def __init__(self, config):
        super().__init__(config)
        options = config.engine_options or {}
        self.api_key = options.get("api_key")

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        audio = self._normalize_audio(audio)
        prompt = self._get_prompt(use_prompt)

        # Replace this block with a call into your local model or remote API.
        text = "recognized text"

        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=language,
                language_probability=1.0 if language else 0.0,
            ),
            metadata={
                "prompt_used": bool(prompt),
            },
        )


engine = MyEngine(
    TranscriptionEngineConfig(
        model="custom",
        normalize_audio=True,
        initial_prompt="domain vocabulary",
        engine_options={"api_key": "secret"},
    )
)

recorder = AudioToTextRecorder(
    transcription_executor=engine,
)
```

`transcription_executor` accepts either:

- an object with a `transcribe()` method, or
- a callable with the same arguments.

The executor receives:

| Argument | Meaning |
| --- | --- |
| `audio` | Audio samples to transcribe. |
| `language` | Language code, or `None` when automatic language handling is requested. |
| `use_prompt` | Whether the engine should use the configured prompt. |
| `**kwargs` | Per-request options such as `word_timestamps=True`. |

## Callable-Only Engines

If you do not need a class, pass a callable:

```python
from RealtimeSTT import AudioToTextRecorder
from RealtimeSTT.engines import TranscriptionResult


def transcribe_with_my_backend(audio, language=None, use_prompt=True, **kwargs):
    return TranscriptionResult(text="recognized text")


recorder = AudioToTextRecorder(
    transcription_executor=transcribe_with_my_backend,
)
```

Use a class when you need persistent model state, API clients, caches, or
configuration helpers.

## Word Timestamps

If your engine can return word timings, accept the `word_timestamps` request
flag and put the timings in result metadata:

```python
def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
    include_words = kwargs.get("word_timestamps", False)

    metadata = {}
    if include_words:
        metadata["words"] = [
            {"word": "hello", "start": 0.10, "end": 0.42},
            {"word": "world", "start": 0.50, "end": 0.86},
        ]

    return TranscriptionResult(
        text="hello world",
        metadata=metadata,
    )
```

Users can request these globally:

```python
recorder = AudioToTextRecorder(
    transcription_executor=engine,
    final_transcription_word_timestamps=True,
)
```

or per call:

```python
text = recorder.transcribe(word_timestamps=True)
```

## Realtime Transcription

For non-streaming realtime updates, pass a second executor:

```python
recorder = AudioToTextRecorder(
    enable_realtime_transcription=True,
    transcription_executor=final_engine,
    realtime_transcription_executor=realtime_engine,
)
```

The realtime executor can use the same `transcribe()` contract as the final
executor. Realtime calls are made repeatedly while speech is still active, so
use a small or fast model when latency matters.

## Streaming Realtime Engines

Use `StreamingTranscriptionSession` when your backend can keep incremental
decoder state instead of retranscribing the whole active buffer on every update.

Timeline for one utterance:

1. Speech starts.
2. RealtimeSTT calls `create_streaming_session()`.
3. New audio chunks arrive.
4. RealtimeSTT calls `accept_audio()` with only the new samples.
5. RealtimeSTT calls `decode()` and then `get_result()` for partial text.
6. Speech ends.
7. RealtimeSTT calls `finish()` for the final streaming result.
8. RealtimeSTT calls `close()` when available.

Example:

```python
from RealtimeSTT.engines import (
    BaseEngine,
    StreamingTranscriptionSession,
    TranscriptionInfo,
    TranscriptionResult,
)


class MyStreamingSession(StreamingTranscriptionSession):
    def __init__(self, recognizer, language=None):
        self.recognizer = recognizer
        self.language = language
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.current_text = ""

    def accept_audio(self, audio, sample_rate=None):
        self.total_samples += int(getattr(audio, "size", 0))
        self.recognizer.accept_audio(audio, sample_rate=sample_rate)

    def decode(self):
        self.current_text = self.recognizer.decode_partial()

    def get_result(self):
        return TranscriptionResult(
            text=self.current_text,
            info=TranscriptionInfo(language=self.language),
            metadata={"samples": self.total_samples},
        )

    def finish(self):
        self.current_text = self.recognizer.decode_final()
        return self.get_result()

    def close(self):
        self.recognizer.close()


class MyStreamingEngine(BaseEngine):
    engine_name = "my_streaming_engine"
    supports_streaming = True

    def __init__(self, config, recognizer_factory):
        super().__init__(config)
        self.recognizer_factory = recognizer_factory

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        recognizer = self.recognizer_factory()
        recognizer.accept_audio(audio, sample_rate=16000)
        return TranscriptionResult(text=recognizer.decode_final())

    def create_streaming_session(self, language=None, use_prompt=True):
        return MyStreamingSession(
            recognizer=self.recognizer_factory(),
            language=language,
        )
```

Then pass it as the realtime executor:

```python
recorder = AudioToTextRecorder(
    enable_realtime_transcription=True,
    transcription_executor=final_engine,
    realtime_transcription_executor=streaming_engine,
)
```

## Error Handling

Raise `TranscriptionEngineError` for setup or runtime failures that should be
reported as engine failures:

```python
from RealtimeSTT.engines import TranscriptionEngineError


def load_backend():
    try:
        import my_asr_package
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The custom engine requires 'my-asr-package'. "
            "Install it with 'pip install my-asr-package'."
        ) from exc
    return my_asr_package
```

For built-in RealtimeSTT engines, keep optional dependency imports lazy so
importing `RealtimeSTT` does not import every backend.

## Packaging A Third-Party Engine

A separate package can depend on RealtimeSTT and expose its own engine class:

```python
from RealtimeSTT.engines import BaseEngine, TranscriptionResult


class CompanyASREngine(BaseEngine):
    engine_name = "company_asr"

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        return TranscriptionResult(text="recognized text")
```

Application code wires it in:

```python
from RealtimeSTT import AudioToTextRecorder
from RealtimeSTT.engines import TranscriptionEngineConfig
from company_realtimestt_engine import CompanyASREngine


engine = CompanyASREngine(
    TranscriptionEngineConfig(
        model="company-large",
        engine_options={"region": "eu"},
    )
)

recorder = AudioToTextRecorder(
    transcription_executor=engine,
)
```

The `RealtimeSTT.engines` import path is lightweight, but the base RealtimeSTT
package still includes recorder and audio dependencies. If a project needs a
tiny interface-only package, that would require a separate packaging split.

## Built-In Style Engines

Only use this path for engines contributed to RealtimeSTT itself.

1. Add an adapter file under `RealtimeSTT/transcription_engines/`.
2. Derive the public adapter from `BaseTranscriptionEngine` or `BaseEngine`.
3. Keep optional backend imports inside helper functions or the constructor.
4. Return `TranscriptionResult` from every successful transcription.
5. Register engine names in `RealtimeSTT/transcription_engines/factory.py`.
6. Add docs under `docs/engines/`.
7. Add focused unit tests and keep real-model tests opt-in.

Factory registration example:

```python
ENGINE_CLASS_PATHS = {
    "my_engine": (".my_engine", "MyEngine"),
    "my-engine": (".my_engine", "MyEngine"),  # Hyphenated names are normalized.
}
```

The factory normalizes configured names by lowercasing and replacing `-` with
`_`, so aliases should usually use underscore form.

## Test Checklist

For user-owned engines, test your package or application code. For contributed
RealtimeSTT engines, add unit tests for:

- missing dependency error messages,
- constructor/config option mapping,
- audio normalization,
- prompt handling,
- language handling,
- result conversion to `TranscriptionResult`,
- word timestamp metadata when supported,
- factory name aliases,
- realtime full-buffer fallback,
- streaming session lifecycle when supported.

Real model downloads and microphone tests should stay opt-in. Keep default unit
tests fast and deterministic.

## Common Mistakes

- Returning a plain string instead of `TranscriptionResult`.
- Importing heavy optional backend packages at module import time.
- Ignoring `language` and `use_prompt` even when the backend supports them.
- Mutating the input audio array in place.
- Enabling `supports_streaming=True` without returning a real session.
- Returning backend-specific result objects directly instead of converting them.
- Committing API keys, downloaded models, generated logs, or local caches.
