# Orukeet CPU session optimization: original deployment record

**Superseded for the tested host on 2026-09-18:** official sherpa-ONNX 1.13.8
with seven threads and `SessionConfig.session.{intra_op,inter_op}.allow_spinning=0`
outperformed this custom build. See the
[current configuration and paired results](../../docs/engines/sherpa-onnx.md#cpu-performance-and-installation).
The recipe below preserves the original 1.13.4 experiment and rollback artifact.

**Public experimental release:** the exact wheel, patch, build sources, notices,
and checksums are available in
[sherpa-onnx-cpu-tuning v1.13.4-cpu.1](https://github.com/KoljaB/sherpa-onnx-cpu-tuning/releases/tag/v1.13.4-cpu.1).
Use that repository's installation instructions and compatibility requirements.
This directory records the original local build and deployment. The standard
RealtimeSTT install does not provide this optional runtime, and the patch has
not been submitted upstream.

This opt-in native patch keeps Orukeet INT8, sherpa-ONNX 1.13.4, and its bundled
ONNX Runtime 1.27.0. It exposes CPU session spinning and separate NeMo encoder,
decoder, and joiner thread counts through sherpa's existing provider config file.
RealtimeSTT's engine and model files do not change.

The deployed configuration uses seven encoder threads, one decoder thread, one
joiner thread, inter-op threads of one, and disables spinning. The ordinary
`num_threads` option remains a fallback for stages without overrides. Plain
`provider=cpu` retains the upstream execution settings.

Use `provider=cpu:/absolute/path/cpu-sessions.conf` in the existing engine options.
**Require distribution version `1.13.4+orukeet.cpu1` first.** Stock sherpa 1.13.4
does not implement these additional config keys. A normal package upgrade must
carry this patch or explicitly revalidate performance.

## Rebuild

1. Obtain the upstream `v1.13.4` source archive from
   <https://github.com/k2-fsa/sherpa-onnx/archive/refs/tags/v1.13.4.tar.gz>.
   Verify SHA256 `3243cb386d3a4ac87596adf7d2c89fddf23e2948b154942b987b4d91c1fee295`.
2. Extract regular files and directories into a fresh directory with Python's
   tarfile `filter="data"`; omit example symlinks. Some upstream example symlinks
   are absolute. Retain the archive and build artifacts.
3. Run `python prepare_cpu_session_build.py SOURCE_DIR ARTIFACT_DIR`. It checks
   expected source fragments before writing, and records the exact four-file
   patch plus original and patched source hashes.
4. Use a fresh Python 3.12 build venv with `build`, `setuptools`, and `wheel`.
   Set `SHERPA_ONNX_CMAKE_ARGS` to:

   ```text
   -DCMAKE_BUILD_TYPE=Release
   -DPython_EXECUTABLE=/absolute/path/build-venv/bin/python
   -DBUILD_SHARED_LIBS=ON
   -DSHERPA_ONNX_ENABLE_BINARY=OFF
   -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF
   -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF
   -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF
   ```

   Set `SHERPA_ONNX_MAKE_ARGS=-j8`; leave `SHERPA_ONNX_SPLIT_PYTHON_PACKAGE`
   unset. Run `python -m build --no-isolation --wheel --sdist --outdir
   ARTIFACT_DIR SOURCE_DIR`. On the measured host the build ran with
   `taskset -c 16-23 nice -n 10` to leave the ASR affinity available.
5. Verify the wheel contains exactly one `_sherpa_onnx.cpython-312-*.so`, install
   into a fresh runtime, and verify imports and every installed package file
   against the wheel. Setting only uppercase `PYTHON_EXECUTABLE` is insufficient:
   upstream pybind11 uses CMake's mixed-case `Python_EXECUTABLE`.

The experimental wheel is distributed through the separate GitHub release;
there is no PyPI publication of this custom version.

## Evidence, 2026-09-17

On the user's Linux host, 12 alternating-order rounds used identical 3-second and
5-second audio, two warmups per duration, and affinity `1,3,7,9,11,13,15`.
Inactive benchmark processes were stopped to prevent idle pool contention.

| Native inference median | 3 seconds | 5 seconds |
| --- | ---: | ---: |
| Stock Orukeet, two threads | 154.47 ms | 245.28 ms |
| Optimized Orukeet, 7/1/1, no spinning | 74.18 ms | 112.22 ms |
| Tuned Parakeet, onnx-asr, 7/1, no spinning | 100.20 ms | 152.17 ms |

The Orukeet transcripts matched exactly across both configurations. A separate
five-round control using seven threads in all three stages measured 662.35 ms
with spinning versus 107.17 ms without it on the five-second clip. This identifies
thread-pool contention as the main cause of the earlier seven-thread regression.
The 7/1/1 configuration was the one verified in the paired comparison and deployed.

The real authenticated five-second API median fell from 252.27 to 120.09 ms;
server decode fell from 245.62 to 113.39 ms. Ten Miep Preview events, including
continuation, succeeded with zero errors. Eight short Preview request-to-callback
measurements were 35.29–38.99 ms, with identical text to the preceding deployment.
These are ASR measurements, not audible assistant-response measurements or a broad
recognition-quality evaluation.

Runtime: `/home/lon/Dev/sherpa-onnx-orukeet-opt-20260917/candidate-py312`.
Evidence and immutable artifacts: its parent directory, especially
`artifacts-py312`, `paired-native.json`, `contention-control.json`, and `state`.
Windows harnesses: `D:/Temp/Codex/orukeet-opt-20260917`.

Wheel SHA256: `1c781606aa4e4e261e48b5e4972d6f5a85c89e097cac52ec803cbd6fe9aaa99f`.
Sdist SHA256: `f5468bc186a9487c87f734470a83aa519bb96e5d8c0fb8bd2647ec92bff03a66`.
All 76 RealtimeSTT files and 14 sherpa files matched their installed wheels.
Live process mappings confirmed the candidate extension and bundled ORT library.

Rollback, while idle: move only the `zzzzzzzzzzzzzzzzzz-orukeet-cpu-sessions.conf`
drop-in out of `realtimestt-104rc1-final-20260820.service.d`, reload systemd user
units, restart that ASR unit, and verify authenticated readiness. This restores
the prior two-thread Orukeet environment. Retain the dependency environments
referenced by `baseline-dependencies.pth`.
