# Custom Transcription Engines

Custom engines let you provide your own speech-to-text backend while keeping
RealtimeSTT's recorder, VAD, buffering, realtime callbacks, and text formatting.

## Public Base Class

Import the lightweight engine contract from `RealtimeSTT.engines`:

```python
from RealtimeSTT.engines import (
    BaseEngine,
    TranscriptionEngineConfig,
    TranscriptionInfo,
    TranscriptionResult,
)
```

`BaseEngine` is the public short name for the same base class used by
RealtimeSTT's built-in transcription adapters. The older descriptive name,
`BaseTranscriptionEngine`, remains available for compatibility.

## Minimal Final Transcription Engine

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
        self.api_key = (config.engine_options or {}).get("api_key")

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        # audio is a mono NumPy float array at the recorder sample rate.
        # Replace this with a call into your ASR runtime or API.
        text = "recognized text"
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=language,
                language_probability=1.0,
            ),
        )


engine = MyEngine(
    TranscriptionEngineConfig(
        model="custom",
        engine_options={"api_key": "secret"},
    )
)

recorder = AudioToTextRecorder(
    transcription_executor=engine,
)
```

`transcription_executor` accepts either an object with a `transcribe()` method
or a plain callable. The method receives:

| Argument | Meaning |
| --- | --- |
| `audio` | Mono audio samples as a NumPy array. |
| `language` | Language code, or `None` when auto-detection is requested. |
| `use_prompt` | Whether the current prompt should be used when your backend supports prompting. |
| `**kwargs` | Optional request flags, for example `word_timestamps=True` on compatible final-transcription calls. |

## Realtime Transcription

For non-streaming realtime updates, pass a second engine instance:

```python
recorder = AudioToTextRecorder(
    enable_realtime_transcription=True,
    transcription_executor=final_engine,
    realtime_transcription_executor=realtime_engine,
)
```

If your backend supports chunk streaming, set `supports_streaming = True` and
return a `StreamingTranscriptionSession` from `create_streaming_session()`.
A streaming session receives only newly recorded chunks during an active
utterance, then `finish()` is called when the utterance ends.

```python
from RealtimeSTT.engines import (
    BaseEngine,
    StreamingTranscriptionSession,
    TranscriptionResult,
)


class MyStreamingSession(StreamingTranscriptionSession):
    def reset(self):
        self.parts = []

    def accept_audio(self, audio, sample_rate=None):
        self.parts.append(audio)

    def decode(self):
        pass

    def get_result(self):
        return TranscriptionResult(text="partial or final text")


class MyStreamingEngine(BaseEngine):
    engine_name = "my_streaming_engine"
    supports_streaming = True

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        return TranscriptionResult(text="fallback full-buffer text")

    def create_streaming_session(self, language=None, use_prompt=True):
        session = MyStreamingSession()
        session.reset()
        return session
```

## Named Built-In Style Engines

Third-party code should prefer `transcription_executor` unless the engine is
being contributed to RealtimeSTT itself. Built-in-style engines live under
`RealtimeSTT.transcription_engines`, keep optional dependency imports lazy, and
are selected by adding an entry to
`RealtimeSTT/transcription_engines/factory.py`.

For contributed engines, add focused tests for:

- missing dependency error messages,
- config and option mapping,
- audio normalization,
- `TranscriptionResult` conversion,
- factory name aliases,
- realtime streaming behavior when supported.
