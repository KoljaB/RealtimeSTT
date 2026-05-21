# Omnilingual ASR WSL Smoke Notes

Date: 2026-05-21

Workspace:

```text
D:\Projekte\STT\RealtimeSTT\RealtimeSTT
```

This file is meant as the one handoff file for a future session. Read this first and do not repeat the Windows/WSL detours below.

## Scope

This work was intentionally limited to installation research, isolated WSL virtualenvs, and tiny smoke/latency tests.

Not done:

- No RealtimeSTT source implementation.
- No transcription engine integration.
- No broad benchmark suite.
- No model comparison beyond the targeted smoke timings below.
- No cleanup/deletion of downloaded model files or caches.

## Main Takeaway

For realtime transcription, start with `omniASR_CTC_300M_v2`.

It was the fastest tested option on the 1.62 s warmup clip, with a predecoded median around `20.7 ms`. `omniASR_CTC_1B_v2` was also comfortably realtime at around `39.2 ms` predecoded. The LLM model is much slower and jittery, even when warmed and forced to FP16.

## Known-Good WSL Setup

Use the existing Ubuntu distro:

```powershell
wsl.exe -d Ubuntu -- bash -lc "..."
```

Important:

- The Ubuntu distro already exists.
- It is stored on D:, not C:.
- It was registered in place from `D:\WSL\Ubuntu\ext4.vhdx`.
- Do not create a new WSL distro on C:.
- Do not use native Windows for `omnilingual-asr`; use WSL.

Observed storage:

```text
/dev/sdc       1007G   64G  893G   7% /
D:\             1.9T  1.7T  221G  89% /mnt/d
```

In the Codex sandbox, WSL commands may need escalation. Use the already-approved command pattern:

```text
wsl.exe -d Ubuntu
```

If WSL launch fails with `CreateVm/E_INVALIDARG` from inside the sandbox, rerun the same `wsl.exe -d Ubuntu ...` command escalated.

## Existing Venvs

Omnilingual ASR venv:

```text
/home/kolja/.venvs/omnilingual-llm1b-v2
```

Installed there:

```text
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
omnilingual-asr==0.2.0
fairseq2==0.6
fairseq2n==0.6+cu128
```

sherpa-onnx int8 venv:

```text
/home/kolja/.venvs/sherpa-onnx-omnilingual-int8
```

Installed there:

```text
sherpa-onnx==1.13.2
```

Approximate sizes after these tests:

```text
7.3G  /home/kolja/.venvs/omnilingual-llm1b-v2
64M   /home/kolja/.venvs/sherpa-onnx-omnilingual-int8
14G   /home/kolja/.cache/fairseq2/assets
629M  /home/kolja/.cache/sherpa-onnx-models
```

## Audio Used

Source audio:

```text
D:\Projekte\STT\RealtimeSTT\RealtimeSTT\RealtimeSTT\warmup_audio.wav
```

WSL path:

```text
/mnt/d/Projekte/STT/RealtimeSTT/RealtimeSTT/RealtimeSTT/warmup_audio.wav
```

Properties:

```text
duration: 1.6195625 s
sample rate: 16000 Hz
channels: mono
```

The Omnilingual reference pipeline accepts only audio shorter than 40 seconds for the tested non-streaming CTC and LLM suites.

## Official Facts Checked

Confirmed from official package/docs:

- PyPI package is `omnilingual-asr`.
- Declared Python requirement is `>=3.10, <=3.12`.
- Inference import is:

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
```

- LLM models accept language IDs such as `eng_Latn`.
- CTC models ignore `lang` if supplied.
- Model cache is under `~/.cache/fairseq2/assets/`.

Native Windows install was not viable:

- Windows venv creation worked.
- `pip install omnilingual-asr` failed because `fairseq2n` has no native Windows wheel.
- This was not a `libsndfile` failure.
- Do not retry native Windows unless upstream fairseq2/fairseq2n Windows support changes.

## Tested Model Files

Omnilingual CTC 300M v2:

```text
/home/kolja/.cache/fairseq2/assets/f33741966d852362f4d416d5/omniASR-CTC-300M-v2.pt
size: 1304065508 bytes
```

Omnilingual CTC 1B v2:

```text
/home/kolja/.cache/fairseq2/assets/fca49af82b51089226da7de9/omniASR-CTC-1B-v2.pt
size: 3902956068 bytes
```

Omnilingual LLM 1B v2:

```text
/home/kolja/.cache/fairseq2/assets/60787894e1cd3958ab8ac97c/omniASR-LLM-1B-v2.pt
size: 9118733852 bytes
```

sherpa-onnx Omnilingual 300M CTC int8:

```text
/home/kolja/.cache/sherpa-onnx-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx
/home/kolja/.cache/sherpa-onnx-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt
```

## Smoke Results

All rows use the same 1.62 s warmup clip.

| Model | Runtime | Median After Warmup | Range | Transcript | Notes |
|---|---:|---:|---:|---|---|
| `omniASR_CTC_300M_v2` | GPU FP16 | `~20.7 ms` predecoded, `~21.4 ms` file path | `17.9-22.9 ms` predecoded | `audio for model warm up` | Fastest and lowest VRAM |
| `omniASR_CTC_1B_v2` | GPU FP16 | `~39.2 ms` predecoded, `~42.1 ms` file path | `33.8-56.8 ms` predecoded | `audio for model warm up` | Still easily realtime |
| sherpa-onnx Omnilingual 300M CTC int8 | CPU int8, 12 threads | `~138 ms` | `135-141 ms` | `audio for model warm up` | Best sherpa CPU setting in this smoke |
| sherpa-onnx Omnilingual 300M CTC int8 | CPU int8, 8 threads | `~155 ms` | `109-160 ms` | `audio for model warm up` | Fast, but less stable than 12 threads here |
| sherpa-onnx Omnilingual 300M CTC int8 | CPU int8, 4 threads | `~161 ms` | `160-275 ms` | `audio for model warm up` | Good CPU fallback |
| `omniASR_LLM_1B_v2` | GPU FP16 | `~452 ms` predecoded, `~493 ms` local file path | `424-834 ms` predecoded | `audio for model warm-up` | Barely under 500 ms median, but jittery |
| sherpa-onnx Omnilingual 300M CTC int8 | CPU int8, 1 thread | `~537 ms` | `535-599 ms` | `audio for model warm up` | Just above latency target 500 ms line |
| `omniASR_LLM_1B_v2` | GPU BF16 | `~730-830 ms` warmed | roughly `732-833 ms` after first call | `audio for model warm-up` | Too slow for realtime targets |

## Detailed Timing Notes

`omniASR_CTC_300M_v2`:

```text
download size: 1.21 GiB progress, 1304065508 bytes on disk
cached init: 10.5456 s
warmup call: 3.7008 s
predecoded dict times:
  0.0228, 0.0179, 0.0185, 0.0190, 0.0224, 0.0179, 0.0225, 0.0229 s
predecoded dict median: 0.0207 s
file path median: 0.0214 s
GPU after init: about 2002 MiB used by nvidia-smi, 622 MiB torch allocated
GPU final: about 2060 MiB used by nvidia-smi, 646 MiB torch allocated
```

`omniASR_CTC_1B_v2`:

```text
download size: 3.63 GiB progress, 3902956068 bytes on disk
init including download: 137.638 s
warmup call: 3.906 s
predecoded 1D times:
  0.0344, 0.0418, 0.0338, 0.0436, 0.0425, 0.0354, 0.0366, 0.0568 s
predecoded median: 0.0392 s
file path median: 0.0421 s
GPU after init: about 3640 MiB used by nvidia-smi, 1863 MiB torch allocated
GPU peak/final: about 3654/3551 MiB used by nvidia-smi
```

`omniASR_LLM_1B_v2`:

```text
initial pipeline init including download: 356.012 s
first transcribe after initial init: 4.638 s
default dtype: torch.bfloat16
BF16 warmed calls: about 0.732-0.833 s after first call
FP16 warmed predecoded median: about 0.452 s
FP16 local ext4 file median: about 0.493 s
FP16 predecoded range: about 0.424-0.834 s
BF16 VRAM: about 5812 MiB after init, about 5993 MiB peak total used
```

The LLM model fit in VRAM on the RTX 2080 SUPER. The problem was latency and jitter, not obvious CPU offload.

sherpa-onnx Omnilingual 300M CTC int8:

```text
provider: cpu
model: model.int8.onnx
audio fed as float samples to OfflineRecognizer.from_omnilingual_asr_ctc()

1 thread:
  median 0.5373 s
  range 0.5350-0.5994 s

4 threads:
  median 0.1611 s
  range 0.1604-0.2749 s

8 threads:
  median 0.1551 s
  range 0.1093-0.1600 s

12 threads:
  median 0.1377 s
  range 0.1347-0.1411 s
```

## Input Format Pitfalls

For Omnilingual pipeline predecoded audio, use a dict:

```python
decoded = {"waveform": waveform_np, "sample_rate": sample_rate}
pipeline.transcribe([decoded], batch_size=1)
```

Do not pass a raw float NumPy waveform directly. The pipeline treats raw `np.ndarray` as encoded audio bytes and asserts the dtype must be `np.uint8` or `np.int8`.

For file path tests, copy the audio to ext4 first to avoid `/mnt/d` overhead/noise:

```text
/home/kolja/.cache/omnilingual-smoke/
```

For CTC models, do not pass `lang` unless testing behavior explicitly. The pipeline logs that `lang` is ignored for CTC.

For LLM models, pass language codes like:

```python
pipeline.transcribe([audio_path], lang=["eng_Latn"], batch_size=1)
```

## PowerShell And WSL Quoting Pitfalls

Avoid package specs like this in PowerShell:

```powershell
pip install sherpa-onnx>=1.12.17
```

PowerShell can interpret `>` as redirection and create a stray file named `=1.12.17`. Quote the spec:

```powershell
wsl.exe -d Ubuntu -- bash -lc "/home/kolja/.venvs/sherpa-onnx-omnilingual-int8/bin/python -m pip install 'sherpa-onnx>=1.12.17'"
```

For complex inline Python in WSL from PowerShell, prefer a base64 wrapper. This avoids PowerShell expanding `$HOME`, `$v`, `$model`, and similar shell variables before Bash sees them.

Pattern:

```powershell
$script = @'
print("python code here")
'@
$bytes = [System.Text.Encoding]::UTF8.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)
wsl.exe -d Ubuntu -- /path/to/venv/bin/python -c "import base64; exec(base64.b64decode('$b64').decode('utf-8'))"
```

## Minimal Reuse Commands

Check GPU from WSL:

```powershell
wsl.exe -d Ubuntu -- /usr/lib/wsl/lib/nvidia-smi
```

Check Omnilingual import/runtime:

```powershell
wsl.exe -d Ubuntu -- /home/kolja/.venvs/omnilingual-llm1b-v2/bin/python -c "import torch, omnilingual_asr; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('omnilingual_asr ok')"
```

Check sherpa-onnx:

```powershell
wsl.exe -d Ubuntu -- /home/kolja/.venvs/sherpa-onnx-omnilingual-int8/bin/python -c "import sherpa_onnx; print(sherpa_onnx.__version__)"
```

Minimal Omnilingual CTC smoke:

```python
import time
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

audio_path = "/mnt/d/Projekte/STT/RealtimeSTT/RealtimeSTT/RealtimeSTT/warmup_audio.wav"
pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M_v2", dtype=__import__("torch").float16)
t0 = time.perf_counter()
print(pipeline.transcribe([audio_path], batch_size=1))
print(time.perf_counter() - t0)
```

Minimal sherpa int8 smoke:

```python
import array
import os
import sys
import time
import wave
import sherpa_onnx

model_dir = "/home/kolja/.cache/sherpa-onnx-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
model = os.path.join(model_dir, "model.int8.onnx")
tokens = os.path.join(model_dir, "tokens.txt")
audio = "/mnt/d/Projekte/STT/RealtimeSTT/RealtimeSTT/RealtimeSTT/warmup_audio.wav"

with wave.open(audio, "rb") as f:
    rate = f.getframerate()
    raw = f.readframes(f.getnframes())

samples = array.array("h")
samples.frombytes(raw)
if sys.byteorder != "little":
    samples.byteswap()
samples = [x / 32768.0 for x in samples]

recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
    model=model,
    tokens=tokens,
    num_threads=12,
    decoding_method="greedy_search",
    provider="cpu",
)

stream = recognizer.create_stream()
stream.accept_waveform(rate, samples)
t0 = time.perf_counter()
recognizer.decode_stream(stream)
print(stream.result.text)
print(time.perf_counter() - t0)
```

## Recommended Next Step

Run a small realistic chunk test before integrating anything:

- Use `omniASR_CTC_300M_v2` first.
- Feed predecoded audio dictionaries.
- Test realistic microphone-like chunks, for example `0.5-5 s`.
- Measure end-to-end latency including audio capture/chunking, not only model inference.
- Compare accuracy against `omniASR_CTC_1B_v2` on the same chunks.

Only after that should a RealtimeSTT engine integration be considered.
