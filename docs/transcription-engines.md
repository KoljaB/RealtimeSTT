# Transcription Engines

RealtimeSTT routes speech recognition through a lazy-loaded engine factory.
`AudioToTextRecorder` selects the main final-transcription backend with
`transcription_engine`; realtime transcription can use the same backend or a
separate one with `realtime_transcription_engine`.

The compatibility default is `faster_whisper`.

## Choosing An Engine

| Use case | Start with | Why |
| --- | --- | --- |
| Default local GPU/CPU Whisper path | `faster_whisper` | Install with `RealtimeSTT[faster-whisper]`; mature, supports common Whisper model names and CTranslate2 models. |
| CPU-only experiments with small Whisper models | `whisper_cpp` | Uses whisper.cpp through `pywhispercpp`; good for low-dependency CPU testing. |
| Compatibility with OpenAI's local Whisper package | `openai_whisper` | Uses the original `openai-whisper` Python package. |
| English CPU server with manually downloaded ONNX models | `sherpa_onnx_moonshine` | Offline CPU INT8 path with predictable local model files. |
| CPU Parakeet without NeMo runtime | `sherpa_onnx_parakeet` | Offline CPU INT8 Parakeet through sherpa-onnx. |
| Kroko/Banafo `.data` streaming models | `kroko_onnx` | Optional Kroko-ONNX runtime with Community or licensed Pro models and realtime streaming previews. |
| NVIDIA Parakeet on Linux/WSL2 | `parakeet` | Uses NVIDIA NeMo ASR for the Parakeet checkpoint. |
| Hugging Face speech-language models | `granite_speech`, `qwen3_asr`, `moonshine`, `cohere_transcribe` | Thin adapters around model-family packages and Transformers. |

## Supported Engine Names

Engine names are normalized by replacing `-` with `_`, so both Python-style and
CLI-style names work where listed.

| Engine names | Status | Reference |
| --- | --- | --- |
| `faster_whisper` | Default production backend | [engines/faster-whisper.md](engines/faster-whisper.md) |
| `whisper_cpp` | Optional production backend | [engines/whisper-cpp.md](engines/whisper-cpp.md) |
| `openai_whisper` | Optional production backend | [engines/openai-whisper.md](engines/openai-whisper.md) |
| `moonshine`, `moonshine_streaming` | Experimental Transformers backend; English-only adapter | [engines/moonshine.md](engines/moonshine.md) |
| `sherpa_onnx_moonshine`, `sherpa_moonshine`, `moonshine_sherpa_onnx` | CPU INT8 sherpa-onnx backend | [engines/sherpa-onnx.md](engines/sherpa-onnx.md) |
| `kroko_onnx`, `kroko`, `banafo_kroko` | Optional Kroko-ONNX backend | [engines/kroko-onnx.md](engines/kroko-onnx.md) |
| `parakeet`, `nvidia_parakeet` | Experimental NVIDIA NeMo backend | [engines/parakeet-nemo.md](engines/parakeet-nemo.md) |
| `sherpa_onnx_parakeet`, `sherpa_parakeet`, `parakeet_sherpa_onnx` | CPU INT8 sherpa-onnx backend | [engines/sherpa-onnx.md](engines/sherpa-onnx.md) |
| `granite_speech`, `granite` | Experimental Transformers backend | [engines/hf-transformers.md](engines/hf-transformers.md) |
| `qwen3_asr`, `qwen_asr` | Experimental Qwen ASR backend | [engines/hf-transformers.md](engines/hf-transformers.md) |
| `cohere_transcribe`, `cohere` | Experimental Transformers backend, requires language | [engines/cohere.md](engines/cohere.md) |
| `openai_api` | Placeholder, not wired yet | Not available |

Unsupported names raise an error that lists the available engines.

## Selecting A Backend

Use the default:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    model="small.en",
    transcription_engine="faster_whisper",
)
```

Use different engines for final and realtime transcription:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="faster_whisper",
    model="small.en",
    enable_realtime_transcription=True,
    realtime_transcription_engine="whisper_cpp",
    realtime_model_type="tiny.en",
    realtime_transcription_engine_options={
        "model": {"n_threads": 8},
        "transcribe": {"single_segment": True, "no_context": True},
    },
)
```

If `realtime_transcription_engine` is `None`, realtime transcription uses the
same backend as `transcription_engine`.

## Engine-Specific Options

Use `transcription_engine_options` and
`realtime_transcription_engine_options` for backend-specific dictionaries:

```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_moonshine",
    model="models/sherpa-onnx-moonshine-tiny-en-int8",
    device="cpu",
    language="en",
    transcription_engine_options={
        "num_threads": 2,
        "provider": "cpu",
    },
)
```

These option dictionaries are intentionally backend-specific. A key that is
meaningful for one engine may be ignored or invalid for another.

## Model Download Behavior

| Engine family | Automatic download | Manual placement |
| --- | --- | --- |
| `faster_whisper` | Yes, for known Hugging Face/CTranslate2 model ids. | Local CTranslate2 model directories may be passed as `model`. |
| `whisper_cpp` | Usually yes for model names supported by `pywhispercpp`. | Local ggml model paths or `download_root`/`models_dir` may be used. |
| `openai_whisper` | Yes, through `openai-whisper`. | Local model names/paths supported by that package. |
| `moonshine`, `granite_speech`, `qwen3_asr`, `cohere_transcribe` | Yes, through Hugging Face or the engine package, subject to access. | `download_root` maps to cache options where supported. |
| `parakeet` NeMo | Yes, through NeMo model loading. | NeMo cache/model options may be passed in `transcription_engine_options`. |
| `sherpa_onnx_*` | No. | Download and extract the sherpa-onnx model bundle, then pass the extracted directory. |
| `kroko_onnx` | Yes, for known public Community `.data` files when enabled. | Pro/private models need an existing `.data` path, direct URL, or explicit repo/token options. |

Every optional engine page documents its install command, model behavior,
important options, and troubleshooting notes.

## Extending Engines

New engines should implement `BaseTranscriptionEngine`, return
`TranscriptionResult`, and be added to `RealtimeSTT/transcription_engines/factory.py`.
Keep imports lazy so optional dependencies are only imported when the engine is
selected.

Contract tests should cover missing dependency messages, parameter mapping,
audio normalization, result conversion, and factory selection. Real-model tests
should remain opt-in.
