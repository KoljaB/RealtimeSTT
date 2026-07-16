# sherpa-onnx

RealtimeSTT includes CPU INT8 sherpa-onnx engines for Parakeet and Moonshine.
They are useful when you want offline CPU inference without loading NeMo or
Transformers at runtime.

## Install

```bash
python -m pip install sherpa-onnx
```

## Engines

| Engine | Model bundle | Language notes |
| --- | --- | --- |
| `sherpa_onnx_parakeet` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | Parakeet model behavior. |
| `sherpa_onnx_moonshine` | `sherpa-onnx-moonshine-tiny-en-int8` | English-only in the adapter. |

Aliases:

- `sherpa_parakeet`
- `parakeet_sherpa_onnx`
- `sherpa_moonshine`
- `moonshine_sherpa_onnx`

## Model Download Requirements

RealtimeSTT does not download sherpa-onnx model bundles automatically. Download
the `.tar.bz2` archives from the sherpa-onnx ASR model releases, extract them,
and pass the extracted directory as `model` or
`transcription_engine_options["model_dir"]`.

Known bundle names:

- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2`
- `sherpa-onnx-moonshine-tiny-en-int8.tar.bz2`

Windows PowerShell example:

```powershell
New-Item -ItemType Directory -Path test-model-cache\sherpa-onnx -Force
curl.exe -L -o test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
python -c "import tarfile; tarfile.open(r'test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8.tar.bz2', 'r:bz2').extractall(r'test-model-cache\sherpa-onnx')"
```

Linux/macOS example:

```bash
mkdir -p test-model-cache/sherpa-onnx
curl -L -o test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
python -c "import tarfile; tarfile.open('test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2', 'r:bz2').extractall('test-model-cache/sherpa-onnx')"
```

## Expected Model Files

Parakeet:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `joiner.int8.onnx`
- `tokens.txt`

Moonshine Tiny:

- `preprocess.onnx`
- `encode.int8.onnx`
- `uncached_decode.int8.onnx`
- `cached_decode.int8.onnx`
- `tokens.txt`

## Basic Use

Moonshine:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_moonshine",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8",
    device="cpu",
    language="en",
    transcription_engine_options={
        "num_threads": 2,
        "provider": "cpu",
    },
)
```

Parakeet:

```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_parakeet",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    device="cpu",
    transcription_engine_options={
        "num_threads": 4,
        "provider": "cpu",
    },
)
```

You can also select the public model-family engine and request the sherpa-onnx
backend:

```python
recorder = AudioToTextRecorder(
    transcription_engine="parakeet",
    model="nvidia/parakeet-tdt-0.6b-v3",
    download_root="test-model-cache/sherpa-onnx",
    device="cpu",
    transcription_engine_options={
        "backend": "sherpa_onnx",
        "num_threads": 4,
    },
)
```

When `download_root` is set, known model ids resolve to the expected extracted
directory names under that root.

## Common Options

| Option | Meaning |
| --- | --- |
| `model_dir` | Explicit extracted model directory. |
| `files` | Dictionary overriding individual file names/paths. |
| `num_threads` | CPU worker threads. |
| `provider` | ONNX Runtime provider, usually `"cpu"`. |
| `decoding_method` | sherpa-onnx decoding method, default `greedy_search`. |
| `debug` | Enables sherpa-onnx debug output. |
| `rule_fsts`, `rule_fars` | Optional text normalization resources. |
| `input_sample_rate`, `sample_rate` | Input/model sample-rate controls. |

Parakeet also supports transducer options such as `model_type`,
`max_active_paths`, `hotwords_file`, `hotwords_score`, `blank_penalty`,
`feature_dim`, `lm`, and `lm_scale`.

## Troubleshooting

- Missing file errors name the exact expected ONNX or `tokens.txt` path. Check
  that the archive was extracted, not just downloaded.
- Keep `model` pointed at the extracted directory, not the `.tar.bz2` archive.
- If latency is high, lower model size where possible, reduce realtime cadence,
  and tune `num_threads`.
- The Moonshine sherpa-onnx adapter is English-only.
