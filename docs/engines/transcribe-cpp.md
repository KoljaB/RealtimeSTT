# transcribe.cpp CUDA

`transcribe_cpp` is the low-latency offline Parakeet engine using the
first-party [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
runtime.
The engine keeps one native model and session resident, accepts RealtimeSTT's
in-memory 16 kHz mono float32 PCM, and serializes calls because transcribe.cpp
0.x permits only one in-flight run per model.

The alias `parakeet_transcribe_cpp` selects the same adapter. The existing
`parakeet` name remains the NVIDIA NeMo adapter.

## Install

Install RealtimeSTT and the portable Python binding:

```bash
python -m pip install "RealtimeSTT[transcribe-cpp]"
```

As of transcribe.cpp 0.2.1, the CUDA provider wheel is attached to the upstream
GitHub release rather than published under the matching version on PyPI. On
Linux x86-64, install the official wheel explicitly:

```bash
python -m pip install \
  "https://github.com/handy-computer/transcribe.cpp/releases/download/v0.2.1/transcribe_cpp_native_cu12-0.2.1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

The official provider is the simple deployment path. The measured fastest path
used a source build specialized for RTX 4090 (SM89), LTO, CUDA, and system
OpenBLAS. Build that shared library when the generic provider is not fast
enough:

```bash
cmake -S transcribe.cpp -B build-realtimestt-sm89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTRANSCRIBE_CUDA=ON \
  -DTRANSCRIBE_BUILD_SHARED=ON \
  -DTRANSCRIBE_BUILD_TOOLS=OFF \
  -DTRANSCRIBE_BUILD_TESTS=OFF \
  -DTRANSCRIBE_BUILD_EXAMPLES=OFF \
  -DTRANSCRIBE_LTO=ON \
  -DTRANSCRIBE_USE_SYSTEM_BLAS=ON \
  -DTRANSCRIBE_USE_OPENMP=OFF \
  -DGGML_CUDA_FA=ON \
  -DBLA_VENDOR=OpenBLAS

cmake --build build-realtimestt-sm89 --target transcribe
```

Point the binding at the custom library before Python imports
`transcribe_cpp`:

```bash
export TRANSCRIBE_LIBRARY=/absolute/path/build-realtimestt-sm89/src/libtranscribe.so
```

The configure output must say
`transcribe: BLAS found — decoder will use cblas_sgemv`. Install the OpenBLAS
development headers or provide their include path through `CMAKE_C_FLAGS` and
`CMAKE_CXX_FLAGS`; a scalar-decoder message means this is not the measured fast
build.

Binding and native library must have the same 0.2.x version.

## Winning model

The validated model is
`parakeet-tdt-0.6b-v3-Q4_K_M.gguf` from
[handy-computer/parakeet-tdt-0.6b-v3-gguf](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf):

```text
SHA-256: b68557be1e3c40207fd7c4bd9d63f1d3316b963f15325bfb0cc16a8bb0ffd181
```

Download the model into a persistent model directory. RealtimeSTT deliberately
does not download or bundle the 502 MB weights during engine initialization.

## Fast final-transcription profile

Set the tuning variables before starting the process. FlashAttention was built
and measured, but disabling it won for five-second Parakeet inputs. Triton is
not part of this ggml runtime.

```bash
export TRANSCRIBE_NO_FLASH=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Then configure the recorder:

```python
import transcribe_cpp
# One global startup call; avoids per-run I/O.
transcribe_cpp.set_log_callback(None)

from RealtimeSTT import AudioToTextRecorder

MODEL = "/models/parakeet-tdt-0.6b-v3-Q4_K_M.gguf"

recorder = AudioToTextRecorder(
    transcription_engine="transcribe_cpp",
    model=MODEL,
    device="cuda",
    gpu_device_index=0,
    beam_size=1,
    batch_size=1,
    faster_whisper_vad_filter=False,
    transcription_engine_options={
        # Explicitly overrides RealtimeSTT's PyTorch-based device probe.
        "backend": "cuda",
        "model_sha256": (
            "b68557be1e3c40207fd7c4bd9d63f1d3316b963f15325bfb0cc16a8bb0ffd181"
        ),
        "session": {
            "n_threads": 7,
            "kv_type": "auto",
            "n_ctx": 0,
        },
        "transcribe": {
            "timestamps": "none",
        },
    },
)
```

The engine deliberately does not change process-wide environment variables.
Set the tuned profile in the service launcher before Python imports NumPy,
Torch, or `transcribe_cpp`.

`set_log_callback(None)` is also process-global. Call it once before loading
models or starting worker threads.

For the exact benchmark machine, launch the service with the measured physical
core affinity rather than changing process affinity inside the engine:

```bash
taskset -c 6,8,10,12,14,0,2 python your_stt_service.py
```

## Measured result

On the Linux RTX 4090 host, each run used 15 excluded warmups and 100 timed
transcriptions of the exact five-second fixture:

| Profile | Mean | p95 | Max |
| --- | ---: | ---: | ---: |
| Flash off, `n_threads=7` | **27.00 ms** | **27.17 ms** | 27.76 ms |
| Flash off, `n_threads=0` | 27.33 ms | 27.55 ms | 27.92 ms |
| Flash on, `n_threads=7` | 36.01 ms | 36.63 ms | 125.77 ms |

The strict final-source rerun measured 27.20 ms mean, 27.65 ms p95, and
27.85 ms max.

The winning run used model SHA-256
`b68557be1e3c40207fd7c4bd9d63f1d3316b963f15325bfb0cc16a8bb0ffd181`
and exact PCM SHA-256
`c4056da582d0e6ede0ea02c7333e8b6dc45cdb9e3178bc8a01172d619ea2bbf2`.
It exercises RealtimeSTT's factory and adapter and excludes one-time model
load, but it does not include an HTTP transport.

The included benchmark is a strict tuned-profile gate. It requires
`TRANSCRIBE_LIBRARY`, checks the native commit, exact shared-library path,
RTX 4090 description, measured CPU affinity, and the exact PCM hash above,
and defaults to a 40 ms mean and 50 ms p95 ceiling.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `backend` | recorder `device` | Native backend. Use `cuda` for the fast profile; explicit GPU selection refuses CPU fallback. |
| `device_index` | `gpu_device_index` | Index within matching native devices. Exactly one device is supported per engine instance. |
| `session.n_threads` | `0` | Native affinity-aware default. Set `7` with the measured taskset profile on the benchmark host. |
| `session.kv_type` | `auto` | Native session KV type. |
| `session.n_ctx` | `0` | Native default context/input limit. |
| `transcribe.timestamps` | `none` | Avoids timestamp materialization on the fast final path. A RealtimeSTT `word_timestamps=True` request overrides this with `word`. |
| `model_sha256` | unset | Optional 64-character digest verified once before model loading. |

`beam_size`, `initial_prompt`, RealtimeSTT's faster-whisper VAD option, and
batch size do not change Parakeet TDT decoding in this adapter.

## Contract and limitations

- Input is offline, mono, one-dimensional float32 PCM at 16 kHz.
- The model and session load once and are reused across warmup and requests.
- The MVP rejects non-Parakeet GGUF architectures instead of silently applying
  Parakeet-specific run options to another model family.
- Calls are protected by an engine lock. Load separate model instances for true
  concurrent inference.
- Recorder shutdown calls the engine cancellation path before waiting for worker
  threads, then releases the resident session and model.
- Multitalker speaker turns are returned in
  `TranscriptionResult.metadata["speaker_segments"]`.
- CUDA selection is exact and fails instead of silently using CPU.
- Automatic Parakeet language output has no confidence probability. Explicit
  language hints are returned as informational metadata with probability 0.0,
  not claimed as detected.
- Parakeet v3 supports 25 European languages, but the initial German controls
  from this project were poor. Run a representative per-language WER gate
  before multilingual production rollout.
- Normal final transcription requests explicitly disable timestamps. Word
  timestamps cost additional work and are returned in
  `TranscriptionResult.metadata["words"]`.
- Warm in-process inference and local HTTP round-trip latency are different
  measurements. Benchmark the deployed persistent server separately.

## Licenses

transcribe.cpp is MIT licensed. NVIDIA Parakeet-TDT 0.6B v3 and its converted
GGUF weights are CC-BY-4.0; preserve the required attribution when deploying or
redistributing the model.
