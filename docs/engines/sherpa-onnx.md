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
| `num_threads` | Threads per native model session; higher counts can increase contention. |
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

## Orukeet as the final recognizer

Orukeet is an optional Parakeet TDT derivative for completed utterances in 25
European languages. Install its pinned export explicitly (`all` retains the
Nemotron/Parakeet pair):

```sh
stt-install-sherpa-models --root ./models/sherpa-onnx --model orukeet
```

The installer downloads the real release manifest and 487 MB archive from
[Hugging Face](https://huggingface.co/oruk/orukeet), checks their SHA-256 hashes,
and preserves the weights' CC BY-SA 4.0 license and NVIDIA attribution. Fetching
the release manifest participates in Hugging Face's normal download accounting.
Verified cached installs and recognition make no model-download requests. Audio
is processed locally. The extracted files occupy about 672 MB.

Use the existing final-pass engine, with the Orukeet model identity so optional
file verification selects the correct checksums:

```python
recorder = AudioToTextRecorder(
    transcription_engine="sherpa_onnx_parakeet",
    model="oruk/orukeet",
    download_root="./models/sherpa-onnx",
    transcription_engine_options={"verify_model_files": True},
)
```

An absolute model directory works too. If you rename the extracted directory,
keep `model="oruk/orukeet"` and set `transcription_engine_options["model_dir"]`
to its new path. If both values identify known bundles, they must identify the
same model; conflicting Orukeet/Parakeet identities raise an error before file
verification or recognizer loading. Orukeet uses 128 input features and the
existing NeMo transducer loader. It is an offline recognizer; keep your
existing live recognizer and VAD for partial transcripts. Language coverage is listed on the model card; this
option does not change the default model or extend support to every Whisper
language.

### CPU performance and installation

**Recommended tested configuration, 2026-09-18:** official sherpa-ONNX 1.13.8,
seven threads and spinning disabled. Install RealtimeSTT and its selected extras
first, then explicitly select the tested runtime in that environment:

```bash
python -m pip install "sherpa-onnx==1.13.8" "sherpa-onnx-core==1.13.8"
```

Save these two lines in a local provider configuration file:

```ini
SessionConfig.session.intra_op.allow_spinning=0
SessionConfig.session.inter_op.allow_spinning=0
```

Pass `{"provider": "cpu:/absolute/path/cpu-session.conf", "num_threads": 7}`
through `transcription_engine_options` or the production server's
`--engine-options`, preserving any existing model-directory and verification
options. Stock 1.13.8 uses a common thread count for the three NeMo sessions;
the former custom `EncoderNumThreads` and `AllowSpinning` keys are not the
official configuration syntax. This is opt-in; reinstalling the pinned extra
can select an older runtime, so verify the installed version after dependency
changes. Its bundled native ONNX Runtime is 1.28.2; the separately installed
Python `onnxruntime` package does not identify sherpa's native runtime.

On the same Ubuntu 24.04 x86-64 / CPython 3.12 host, 20 alternating-order rounds
per duration compared identical audio and fixed affinity. Inactive benchmark
processes were stopped to prevent competing native pools:

| Native median | Tuned 1.13.4, 7/1/1 | Official 1.13.8, 7/7/7 |
| --- | ---: | ---: |
| 3-second audio | 74.72 ms | 70.82 ms |
| 5-second audio | 113.32 ms | 107.20 ms |
| 5-second audio, synthetic CPU contention | 124.32 ms | 116.61 ms |

Both configurations disabled spinning. Transcripts matched across all 240 timed
native calls, including the 2- and 4-thread candidate sweep. The authenticated
API median fell from 118.89 to 114.23 ms in sequential before/after checks.
Ten Miep Preview events, including continuation, passed with Live ASR output and
zero errors; eight short Preview requests took 34.44-38.39 ms. These are bounded
ASR measurements, not broad accuracy or audible-response measurements. Choose
the thread count for the available cores and concurrent workload.

#### Earlier custom build, 2026-09-17

The original experimental wheel, patch, build sources, notices,
and checksums remain available in the separate
[sherpa-onnx-cpu-tuning release](https://github.com/KoljaB/sherpa-onnx-cpu-tuning/releases/tag/v1.13.4-cpu.1).
This is an independently maintained build, not an official sherpa release or a
standard RealtimeSTT dependency. No upstream PR has been submitted.

CPU performance depends on thread-pool settings as well as the model. In the
pinned stock `sherpa-onnx==1.13.4`, the NeMo encoder, decoder, and joiner share
one thread-count setting, and their ONNX Runtime worker pools can spin while
waiting for work. Those pools may compete for the same cores. See the upstream
[session options](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.4/sherpa-onnx/csrc/session.cc)
and [NeMo sessions](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.4/sherpa-onnx/csrc/offline-transducer-nemo-model.cc).

A controlled Linux x86-64 test on 2026-09-17 used the same Orukeet INT8 weights,
bundled ONNX Runtime 1.27.0, and fixed affinity throughout. Each audio duration
had two warmups and 12 alternating-order measured rounds. Inactive benchmark
processes were stopped to prevent their thread pools influencing another model.

| Native inference median | 3-second audio | 5-second audio |
| --- | ---: | ---: |
| Stock Orukeet, 2 threads per session | 154.47 ms | 245.28 ms |
| Experimental patched Orukeet runtime, 7/1/1 threads, spinning disabled | 74.18 ms | 112.22 ms |
| Tuned Parakeet, `parakeet_onnx` / onnx-asr | 100.20 ms | 152.17 ms |

Orukeet transcripts matched across the two configurations. A separate control
held all three pools at seven threads: disabling spinning reduced the
five-second median from 662.35 to 107.17 ms. The 7/1/1 configuration was selected
for the paired comparison and server deployment. The authenticated API median
fell from 252.27 to 120.09 ms; ten Preview events, including continuation, passed.
These measurements cover one host and bounded fixtures, not all hardware,
languages, recognition accuracy, or audible assistant-response latency.

#### Legacy build availability

The experimental build is `sherpa-onnx==1.13.4+orukeet.cpu1`, distributed through
GitHub Releases rather than PyPI. Follow the
[versioned install command and CPU profile](https://github.com/KoljaB/sherpa-onnx-cpu-tuning/tree/v1.13.4-cpu.1#install).
The wheel was tested on Ubuntu 24.04 x86-64 with CPython 3.12 and requires glibc
2.38 or newer, GLIBCXX_3.4.32, and `libasound.so.2`. It is not a manylinux wheel;
other platforms and Python versions are not supported by this binary release.

The standard `RealtimeSTT[sherpa-onnx]` extra installs upstream 1.13.4, and
`stt-install-sherpa-models --model orukeet` installs model files only. It does not
build or replace the native runtime or configure the application.

The [public build instructions](https://github.com/KoljaB/sherpa-onnx-cpu-tuning/blob/main/BUILDING.md),
[original deployment notes](../../deploy/sherpa-onnx/README.md), and
[source preparation script](../../deploy/sherpa-onnx/prepare_cpu_session_build.py)
record how the native code was built. Installing the released wheel does not
require compiling C++; enable its provider configuration explicitly after
installation. Stock 1.13.4 does not implement the added provider keys and can
silently ignore them. The wheel retains upstream dependencies with their own
licenses, including eSpeak NG; the release includes their notices and build sources.

Seven encoder threads is a measured setting for this host, not a portable
default. Future CPU profiles need to account for available cores, process or
container affinity, and concurrent Live ASR, TTS, or other workloads.

#### Making this automatic

Upstream now supplies the required session-config forwarding. Future upgrades
should first compare a configured official release against the last working
runtime, using separate environments, pinned wheel hashes and the same paired
benchmark. A custom build is needed only if a remaining patch adds a measured
benefit. Build/import checks can run in CI; latency acceptance belongs on a fixed
machine, followed by an authenticated API and Preview/Live smoke before promotion.

An installer could verify runtime support and expose an explicit no-spinning
profile. Thread selection must respect available cores, process affinity and
other workloads; a fixed seven-thread default would not ensure high performance.
No automatic performance installer, scheduled upgrade workflow or autotuner is
currently provided.
