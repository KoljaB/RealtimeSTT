# Installation

RealtimeSTT uses install extras so each environment can install only the
transcription engines and wake-word backends it needs.

Recommended default local Whisper install:

```bash
python -m pip install "RealtimeSTT[faster-whisper]"
```

Core package only, without a transcription engine or wake-word backend:

```bash
python -m pip install RealtimeSTT
```

The core install includes microphone/audio support, WebRTC VAD, Silero VAD,
websocket client/server dependencies, and shared audio utilities. It does not
install `faster-whisper`, Porcupine, or OpenWakeWord unless you request those
extras.

## Python Environment

Use Python 3.11 or newer. The current pinned core dependency set includes
packages whose published wheels require Python 3.11+.

The Meta Omnilingual ASR extra is narrower than the core package: use Linux or
WSL2 with Python 3.11.x. Native Windows cannot run the Omnilingual runtime, and
Python 3.12.x currently cannot resolve the upstream `omnilingual-asr>=0.2.0`
package from PyPI because its metadata excludes normal 3.12 patch releases.

Use a virtual environment when possible:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install "RealtimeSTT[faster-whisper]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install "RealtimeSTT[faster-whisper]"
```

## Install Extras

Extras can be combined with commas:

```bash
python -m pip install "RealtimeSTT[faster-whisper,porcupine]"
python -m pip install "RealtimeSTT[whisper-cpp,openwakeword]"
```

| Extra | Installs | Use when |
| --- | --- | --- |
| `faster-whisper` | `faster-whisper` | Default CTranslate2 Whisper backend. |
| `whisper-cpp` / `whispercpp` | `pywhispercpp` | whisper.cpp backend for CPU-focused setups. |
| `openai-whisper` | `openai-whisper` | Original OpenAI Whisper Python backend. |
| `sherpa-onnx` / `sherpa` | `sherpa-onnx` | CPU INT8 sherpa-onnx Parakeet or Moonshine engines. |
| `transformers` | `transformers` | Shared dependency for Transformers-based ASR engines. |
| `moonshine`, `granite`, `cohere` | `transformers` | Aliases for the corresponding Transformers engines. |
| `parakeet` / `nvidia-parakeet` | `nemo_toolkit[asr]` | NVIDIA NeMo Parakeet backend, best on Linux or WSL2. |
| `omnilingual` / `omnilingual-asr` / `meta-omnilingual-asr` | `omnilingual-asr>=0.2.0`, matching `torch==2.8.0` / `torchaudio==2.8.0` on Linux/WSL2 Python 3.11.x | Meta Omnilingual ASR backend. Native Windows installs skip the actual Omnilingual runtime by package marker, and Python 3.12.x is not a practical install target until upstream dependency metadata changes. |
| `qwen` / `qwen3-asr` | `qwen-asr` | Qwen ASR backend. |
| `qwen-vllm` | `qwen-asr[vllm]` | Qwen ASR with vLLM support. |
| `kroko-builder` | RealtimeSTT builder helper, `huggingface_hub` | Builds/installs Kroko-ONNX from upstream and downloads public Community models. |
| `porcupine` / `pvporcupine` / `pvp` | `pvporcupine` | Porcupine wake-word backend. |
| `openwakeword` / `oww` | `openwakeword` | OpenWakeWord wake-word backend. |
| `wakewords` / `wake-words` | `pvporcupine`, `openwakeword` | Both wake-word backends. |
| `recommended` / `default` | `faster-whisper`, `silero-vad[onnx-cpu]` | Default local transcription plus the faster raw CPU ONNX Silero VAD path. |
| `all` | All PyPI-installable optional backends | Broad development or experimentation environments. |

`kroko-builder` does not install Kroko-ONNX by itself; it exposes
`stt-install-kroko`, which builds and installs Kroko-ONNX from upstream:

```bash
python -m pip install "RealtimeSTT[kroko-builder]"
stt-install-kroko --build
```

On Windows, use Python 3.12 x64 and start Docker Desktop first. The Docker
Desktop Linux engine must be running before the builder starts:

```powershell
python --version
git --version
docker version
```

`docker version` should show both `Client` and `Server` sections.
`docker --version` only checks that the Docker CLI is installed; it does not
verify that Docker Desktop's engine is running. If the default builder cache is
not writable, use a project-local work directory:

```powershell
stt-install-kroko --build --work-dir .\kroko-builder-work
```

If the default builder cache is not writable and `--work-dir` is not set, the
helper falls back to `.\kroko-builder-work` automatically.

Download a public Community model after the builder finishes:

```powershell
New-Item -ItemType Directory -Path test-model-cache\kroko-onnx -Force
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Banafo/Kroko-ASR', filename='Kroko-EN-Community-64-L-Streaming-001.data', local_dir='test-model-cache/kroko-onnx')"
```

See [engines/kroko-onnx.md](engines/kroko-onnx.md).

## VAD Dependencies

WebRTC VAD and Silero VAD are still core dependencies. The recorder currently
initializes both `webrtcvad` and the Silero/PyTorch path, so they cannot be
split into independent extras without changing the recorder's VAD selection
logic.

Wake-word dependencies are optional. If you do not use wake words, install no
wake-word extra. If you set `wake_words` without a `wakeword_backend`,
RealtimeSTT uses Porcupine for backward compatibility, so install the
`porcupine` extra.

## Platform Notes

### Linux

Install PortAudio and Python headers before installing PyAudio:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
python -m pip install "RealtimeSTT[faster-whisper]"
```

Some examples and tests also use `ffmpeg` or `libsndfile`:

```bash
sudo apt-get install ffmpeg libsndfile1
```

### macOS

Install PortAudio through Homebrew:

```bash
brew install portaudio
python -m pip install "RealtimeSTT[faster-whisper]"
```

### Windows

Install from a normal terminal or PowerShell session:

```powershell
python -m pip install "RealtimeSTT[faster-whisper]"
```

If a dependency needs a compiler on your machine, install the relevant wheel
package first when one is available. `webrtcvad-wheels` is used by the project
to avoid the older source-only WebRTC VAD install path.

Meta Omnilingual ASR is not available for native Windows inference. The
`omnilingual` extra is guarded by a non-Windows package marker because
`fairseq2n`, one of Meta Omnilingual ASR's dependencies, does not provide a
Windows wheel. Use Linux or WSL2 with Python 3.11.x when selecting
`omnilingual_asr`.

## CUDA Notes

RealtimeSTT can run on CPU with small models, but CUDA is strongly preferred
for low-latency realtime transcription and larger Whisper-family models.

Install the NVIDIA driver, CUDA runtime/toolkit, and cuDNN version that matches
your PyTorch build. Then install a CUDA-enabled PyTorch and torchaudio wheel
before installing RealtimeSTT, for example:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install "RealtimeSTT[faster-whisper]"
```

Use the PyTorch install selector for the exact command for your driver and CUDA
version. Keep `device="cuda"` for GPU inference and use `device="cpu"` for CPU
or CPU-only engine stacks.

## Optional Engine Dependencies

Install only the engine stack you plan to use:

| Engine | Install command | Model handling |
| --- | --- | --- |
| `faster_whisper` | `python -m pip install "RealtimeSTT[faster-whisper]"` | Downloads CTranslate2 models automatically through faster-whisper. |
| `whisper_cpp` | `python -m pip install "RealtimeSTT[whisper-cpp]"` | `pywhispercpp` can download known ggml models; local paths are also supported. |
| `openai_whisper` | `python -m pip install "RealtimeSTT[openai-whisper]"` | Downloads OpenAI Whisper models automatically to its cache or `download_root`. |
| `moonshine` | `python -m pip install "RealtimeSTT[moonshine]"` | Downloads Hugging Face model files automatically. English-only in this adapter. |
| `sherpa_onnx_*` | `python -m pip install "RealtimeSTT[sherpa-onnx]"` | Model bundles must be downloaded and extracted manually. |
| `parakeet` | `python -m pip install -U "RealtimeSTT[parakeet]"` | NeMo downloads from the configured model id/cache. Best on Linux or WSL2. |
| [`omnilingual_asr`](engines/omnilingual-asr.md) | `python -m pip install "RealtimeSTT[omnilingual]"` | Meta Omnilingual ASR downloads through its Linux/WSL fairseq2 cache. Use Python 3.11.x. The extra requires `omnilingual-asr>=0.2.0` for v2 model cards and constrains matching `torch`/`torchaudio` builds. Native Windows installs are not supported because `fairseq2n` has no Windows wheel. |
| `granite_speech` | `python -m pip install "RealtimeSTT[granite]"` | Downloads Hugging Face model files automatically. |
| `qwen3_asr` | `python -m pip install -U "RealtimeSTT[qwen]"` | Downloads Qwen model files through the Qwen ASR package. |
| `cohere_transcribe` | `python -m pip install "RealtimeSTT[cohere]"` | Downloads Hugging Face model files; gated model access may be required. |
| `kroko_onnx` | `python -m pip install "RealtimeSTT[kroko-builder]"`, then `stt-install-kroko --build` | Public Community models can auto-download or be downloaded with `huggingface_hub`; Pro/private models need an existing `.data` path, direct URL, or explicit repo/token options. |

Per-engine setup lives in [transcription-engines.md](transcription-engines.md)
and the `docs/engines/` pages.

## Meta Omnilingual ASR Notes

Use a Linux or WSL2 virtual environment with Python 3.11.x:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install "RealtimeSTT[omnilingual]"
```

Do not use native Windows or Python 3.12.x for the Omnilingual runtime. Native
Windows lacks a `fairseq2n` wheel, and `omnilingual-asr==0.2.0` currently
declares `Requires-Python: <=3.12,>=3.10`, which excludes normal Python 3.12
patch releases such as 3.12.3 during dependency resolution.

If you install a CUDA-enabled PyTorch stack separately, make sure `torch` and
`torchaudio` come from matching builds. A mismatched pair can pass
`python -m pip check` but fail later while importing `omnilingual_asr`, for
example with a missing `libcudart.so` shared-library error.

The default Omnilingual model is `omniASR_CTC_1B_v2`. If the installed
`omnilingual-asr` package does not know that card, treat it as a dependency
version mismatch. Do not silently fall back to older non-v2 cards unless they
pass your own language and quality smoke tests.

## Wake Word Dependencies

Porcupine wake words:

```bash
python -m pip install "RealtimeSTT[porcupine]"
```

OpenWakeWord:

```bash
python -m pip install "RealtimeSTT[openwakeword]"
```

Both wake-word backends:

```bash
python -m pip install "RealtimeSTT[wakewords]"
```

Wake-word setup is documented in [wake-words.md](wake-words.md).

## FastAPI Server Dependencies

For the browser streaming server:

```bash
python -m pip install -r example_fastapi_server/requirements.txt
```

Then install the selected ASR engine stack, for example:

```bash
python -m pip install "RealtimeSTT[faster-whisper]"
```

See [fastapi-server.md](fastapi-server.md).

## Verifying The Install

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    with AudioToTextRecorder(model="tiny", device="cpu") as recorder:
        print("Speak now")
        print(recorder.text())
```

For the test workflow, see [testing.md](testing.md).
