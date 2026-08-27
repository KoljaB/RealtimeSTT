# sherpa-onnx

RealtimeSTT includes CPU INT8 sherpa-onnx engines for Parakeet, Moonshine, and
the multilingual Nemotron 3.5 streaming model.
They are useful when you want offline CPU inference without loading NeMo or
Transformers at runtime.

## Install

```bash
python -m pip install "RealtimeSTT[sherpa-onnx]"
```

## Engines

| Engine | Model bundle | Language notes |
| --- | --- | --- |
| `sherpa_onnx_parakeet` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | Parakeet model behavior. |
| `sherpa_onnx_moonshine` | `sherpa-onnx-moonshine-tiny-en-int8` | English-only in the adapter. |
| `sherpa_onnx_nemotron` | `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11` | True streaming; per-stream language or automatic detection. |

Aliases:

- `sherpa_parakeet`
- `parakeet_sherpa_onnx`
- `sherpa_moonshine`
- `moonshine_sherpa_onnx`
- `nemotron`
- `sherpa_nemotron`
- `nemotron_sherpa_onnx`

## Model Download Requirements

Engine initialization never downloads or replaces model weights. Install the
pinned Nemotron and Parakeet bundles explicitly into a persistent root; the
installer resumes interrupted downloads, verifies the exact archive and
extracted files, rejects unsafe archives, commits extraction atomically, and
reuses verified offline caches:

```bash
stt-install-sherpa-models --root ./models/sherpa-onnx --model all
```

Use `--model nemotron` or `--model parakeet` for one bundle, and `--offline` to
forbid network access. Pass the resulting extracted directory as `model`, or
pass its root as `download_root` with the model ID. Moonshine bundles are not
part of this two-model manifest/installer and remain a manual upstream install.

Known bundle names:

- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2`
- `sherpa-onnx-moonshine-tiny-en-int8.tar.bz2`
- `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2`

The Nemotron and Parakeet archives are pinned in
`RealtimeSTT.model_manifests`:

| Bundle | Archive size | SHA-256 | Model license |
| --- | ---: | --- | --- |
| Nemotron 3.5 560 ms INT8 | 475271763 bytes | `c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a` | NVIDIA Open Model Data Warehouse License Agreement v1.1 |
| Parakeet v3 INT8 | 487170055 bytes | `5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf` | CC-BY-4.0 |

The archive size and SHA-256 are the manifest's primary integrity boundary:
verify them before extraction. The manifest's expected-file list is the fast
runtime boundary after extraction and detects an incomplete or incorrectly
rooted directory. For a full extracted-tree check, pass
`transcription_engine_options={"verify_model_files": True}`; this compares the
pinned per-file sizes and SHA-256 values and can take time for the large ONNX
encoder. `verify_optional_model_files=True` also checks Nemotron's optional
`README.md`.

## Expected Model Files

Parakeet:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `joiner.int8.onnx`
- `tokens.txt`

Nemotron 3.5:

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

Nemotron streaming:

```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_nemotron",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    device="cpu",
    language="auto",
    transcription_engine_options={
        "num_threads": 2,
        "provider": "cpu",
    },
)
```

Nemotron accepts only new mono frames at 16 kHz. `language="auto"` (or an
empty language) selects model-side detection; a code such as `de-DE` is set on
that stream only. Streaming sessions expose partials through
`get_result()`, and `finish()` calls sherpa `input_finished()` before draining
the ready-frame decode loop.

Parakeet:

```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_parakeet",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    device="cpu",
    language="auto",
    transcription_engine_options={
        "num_threads": 4,
        "provider": "cpu",
    },
)
```

Parakeet applies the same fixed/automatic language choice to its fresh
`OfflineStream` before accepting the authoritative final audio. It decodes that
complete audio exactly once; it is not used for growing-buffer partials.

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
| `verify_model_files` | Optional full size/SHA-256 validation of pinned Parakeet/Nemotron files at setup. |
| `verify_optional_model_files` | Include optional manifest files such as Nemotron `README.md` in verification. |

Parakeet also supports transducer options such as `model_type`,
`max_active_paths`, `hotwords_file`, `hotwords_score`, `blank_penalty`,
`feature_dim`, `lm`, and `lm_scale`.

## Platform support

The release-supported Nemotron-live/Parakeet-final production pair is Linux
x86-64 with `sherpa-onnx==1.13.4`. Native Windows can load and run both models,
and it is useful for development, but the Parakeet binding produced intermittent
empty authoritative results on voiced cumulative-recovery fixtures during
earlier validation. An empty decode is indistinguishable from legitimate
silence at the engine boundary, so native Windows Parakeet is not promoted for
authoritative production finals in this release. Deploy this pair on Linux, or
select another final engine (for example `faster_whisper`) while retaining
Nemotron for live hypotheses.

## Troubleshooting

- Missing file errors name the exact expected ONNX or `tokens.txt` path. Check
  that the archive was extracted, not just downloaded.
- Keep `model` pointed at the extracted directory, not the `.tar.bz2` archive.
- If latency is high, lower model size where possible, reduce realtime cadence,
  and tune `num_threads`.
- The Moonshine sherpa-onnx adapter is English-only.
