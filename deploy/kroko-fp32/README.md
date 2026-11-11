# Resident Kroko request cancellation

The patch applies to `src/main.cc` of the deployed 13 September 2026 Kroko
FP32 CLI MVP. Exact base/source/binary hashes are in the adjacent JSON file.
Build with the original model/runtime dependencies and CMake configuration;
the verified build used Release and KROKO_NATIVE_ARCH=ON.

Ready protocol 1 advertises `cancellation: request_id_run_options`.
`cancel ID` terminates only that job's ONNX RunOptions. A `cancelled` response
with the same ID acknowledges native completion; the resident model remains
loaded. Old IDs cannot cancel successors. Normal request/result fields and
non-stdio CLI inference remain compatible. The Python adapter waits for the
matching completion/cancellation receipt before submitting its successor.

The Windows source adapter, shared worker fence and regression tests live in
this repository. Deployed source, wheel/sdist, binary, rollback and signed
attestation are retained on the user's LAN server under
`/home/lon/codex_benchmarks/miep-three-latency-fixes-20260914-01/`.
