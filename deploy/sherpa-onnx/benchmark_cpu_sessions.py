"""Measure one resident CPU recognizer; invoke separately for each environment.

The JSON-lines worker mode lets a controller alternate warmed recognizers while
stopping inactive workers to prevent their native thread pools competing.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.sched_setaffinity(0, set(config["affinity"]))
    import numpy as np
    from RealtimeSTT.transcription_engines import (
        TranscriptionEngineConfig, create_transcription_engine,
    )

    with wave.open(str(args.audio), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    engine = create_transcription_engine(config["engine"], TranscriptionEngineConfig(
        model=config["model"], device="cpu", engine_options=config["options"]))

    def run(seconds):
        sample = audio[:round(seconds * 16000)]
        started = time.perf_counter()
        result = engine.transcribe(sample, language="en")
        return {"seconds": len(sample) / 16000, "ms": (time.perf_counter() - started) * 1000,
                "text": result.text}

    durations = [3, 5]
    for duration in durations:
        for _ in range(2):
            run(duration)
    metadata = {"label": config["label"], "config": config,
                "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
                "python": sys.executable,
                "packages": {name: importlib.metadata.version(name) for name in
                             ["sherpa-onnx", "onnxruntime", "onnx-asr"]}}
    if args.worker:
        print(json.dumps({"ready": True, **metadata}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("stop"):
                break
            print(json.dumps({**run(request["seconds"]), "label": config["label"]}), flush=True)
    else:
        rows = [run(duration) for _ in range(args.rounds) for duration in durations]
        medians = {str(duration): statistics.median(r["ms"] for r in rows if r["seconds"] == duration)
                   for duration in durations}
        print(json.dumps({**metadata, "rows": rows, "median_ms": medians}), flush=True)


if __name__ == "__main__":
    main()
