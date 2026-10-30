"""Standalone FunASR smoke and microphone test."""

from __future__ import print_function

import argparse
import importlib
import os
import re
import sys
import time
import wave
from pathlib import Path


DEFAULT_MODEL = "iic/SenseVoiceSmall"
DEFAULT_AUDIO = Path(__file__).resolve().parent / "unit" / "audio" / "LJ001-0002.wav"
DEFAULT_EXPECTED_TEXT = "in being"


def install_repo_on_path():
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "RealtimeSTT").is_dir():
        repo_root_text = str(repo_root)
        if repo_root_text not in sys.path:
            sys.path.insert(0, repo_root_text)


def check_import(module_name, package_name):
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print("Missing or broken runtime dependency: {0}".format(module_name))
        print("{0}: {1}".format(exc.__class__.__name__, exc))
        print("\nInstall it with:\n")
        print('  python -m pip install "RealtimeSTT[funasr]"')
        print("\nFor microphone tests, also install a local VAD backend, for example:\n")
        print('  python -m pip install "RealtimeSTT[funasr,silero-onnx-cpu]"')
        raise SystemExit(1)


def check_funasr_runtime():
    check_import("funasr", "funasr")


def check_numpy_runtime():
    check_import("numpy", "numpy")


def normalize_text(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return " ".join(text.split())


def read_wav_float32(path, target_rate=16000):
    check_numpy_runtime()
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise SystemExit("Only 16-bit PCM WAV files are supported by this smoke test.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate == target_rate:
        return audio

    target_length = int(len(audio) * float(target_rate) / sample_rate)
    source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def build_engine_options(args):
    auto_model = {
        "disable_update": not args.allow_update_check,
        "disable_pbar": not args.show_progress,
    }
    generate = {}

    if args.hub:
        auto_model["hub"] = args.hub
    if args.vad_model:
        auto_model["vad_model"] = args.vad_model
        auto_model["vad_kwargs"] = {
            "max_single_segment_time": args.vad_max_single_segment_time
        }
    if args.batch_size_s is not None:
        generate["batch_size_s"] = args.batch_size_s
    if args.use_itn:
        generate["use_itn"] = True
    if args.language:
        generate["language"] = args.language

    return {
        "auto_model": auto_model,
        "generate": generate,
    }


def create_engine(args):
    install_repo_on_path()
    from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig
    from RealtimeSTT.transcription_engines.factory import create_transcription_engine

    config = TranscriptionEngineConfig(
        model=args.model,
        download_root=args.download_root,
        gpu_device_index=args.gpu_device_index,
        device=args.device,
        batch_size=args.batch_size,
        normalize_audio=True,
        engine_options=build_engine_options(args),
    )
    return create_transcription_engine("funasr", config)


def run_init_only(args):
    engine = create_engine(args)
    backend = getattr(engine, "backend", None)

    print("FunASR engine initialized successfully.")
    print("Model: {0}".format(getattr(backend, "model_name", args.model)))
    print("Device: {0}".format(args.device))
    if args.download_root:
        print("Download root: {0}".format(args.download_root))


def run_file_smoke(args):
    audio_path = Path(args.audio).expanduser()
    if not audio_path.exists():
        raise SystemExit("Audio file does not exist: {0}".format(audio_path))

    engine = create_engine(args)
    audio = read_wav_float32(audio_path)

    print("Running FunASR file smoke.")
    print("Audio: {0}".format(audio_path))
    print("Model: {0}".format(args.model))
    print("Device: {0}".format(args.device))
    if args.expected_text:
        print("Expected text fragment: {0}".format(args.expected_text))

    start = time.perf_counter()
    result = engine.transcribe(audio, language=args.language)
    elapsed = time.perf_counter() - start
    transcript = result.text or ""
    normalized = normalize_text(transcript)
    expected = normalize_text(args.expected_text)
    duration = len(audio) / 16000.0 if len(audio) else 0.0

    print("\nTranscript:")
    print(transcript)
    print("\nNormalized transcript:")
    print(normalized)
    print("\nElapsed: {0:.3f}s, audio: {1:.3f}s, speed: {2:.2f}x".format(
        elapsed,
        duration,
        duration / elapsed if elapsed else 0.0,
    ))

    if expected and expected not in normalized:
        message = (
            "Smoke failed: expected normalized transcript to contain "
            "'{0}'.".format(expected)
        )
        if args.allow_mismatch:
            print("WARNING: " + message)
            return
        raise SystemExit(message)

    print("FunASR file smoke passed.")


def run_microphone(args):
    install_repo_on_path()
    from RealtimeSTT import AudioToTextRecorder

    full_sentences = []
    last_partial = [""]

    def on_realtime(text):
        text = (text or "").strip()
        if text and text != last_partial[0]:
            last_partial[0] = text
            print("Realtime: {0}".format(text))

    def on_final(text):
        text = (text or "").strip()
        if not text:
            return
        full_sentences.append(text)
        print("Final: {0}".format(text))

    realtime_model = args.realtime_model or args.model
    engine_options = build_engine_options(args)

    recorder_config = {
        "spinner": False,
        "model": args.model,
        "transcription_engine": "funasr",
        "transcription_engine_options": engine_options,
        "realtime_model_type": realtime_model,
        "realtime_transcription_engine": "funasr",
        "realtime_transcription_engine_options": dict(engine_options),
        "use_main_model_for_realtime": not args.separate_realtime_model,
        "download_root": args.download_root,
        "device": args.device,
        "gpu_device_index": args.gpu_device_index,
        "language": args.language,
        "input_device_index": args.input_device_index,
        "silero_sensitivity": 0.05,
        "webrtc_sensitivity": 3,
        "post_speech_silence_duration": args.post_speech_silence_duration,
        "min_length_of_recording": 1.1,
        "min_gap_between_recordings": 0,
        "enable_realtime_transcription": args.realtime,
        "realtime_processing_pause": args.realtime_processing_pause,
        "on_realtime_transcription_update": on_realtime,
        "silero_deactivity_detection": True,
        "early_transcription_on_silence": 0,
        "batch_size": args.batch_size,
        "realtime_batch_size": args.batch_size,
        "no_log_file": True,
        "faster_whisper_vad_filter": False,
        "normalize_audio": True,
    }

    recorder = None
    try:
        print("Initializing FunASR microphone test.")
        print("Model: {0}".format(args.model))
        print("Realtime model: {0}".format(realtime_model))
        print("Device: {0}".format(args.device))
        print("Realtime: {0}".format("on" if args.realtime else "off"))
        print("Press Ctrl+C to stop.\n")

        recorder = AudioToTextRecorder(**recorder_config)
        while True:
            recorder.text(on_final)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if recorder is not None:
            recorder.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone RealtimeSTT FunASR smoke and microphone test."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize the FunASR engine wrapper and exit.",
    )
    mode.add_argument(
        "--file-smoke",
        action="store_true",
        help="Transcribe the bundled LJ speech WAV and exit.",
    )
    mode.add_argument(
        "--microphone",
        action="store_true",
        help="Start the microphone loop. This is the default.",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--realtime-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-device-index", type=int, default=0)
    parser.add_argument("--download-root", "--root", default=None)
    parser.add_argument("--language", "--lang", default="auto")
    parser.add_argument("--hub", default="ms", help="FunASR model hub: ms or hf.")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--use-itn", action="store_true")
    parser.add_argument("--vad-model", default=None)
    parser.add_argument("--vad-max-single-segment-time", type=int, default=30000)
    parser.add_argument("--allow-update-check", action="store_true")
    parser.add_argument("--show-progress", action="store_true")

    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--expected-text", default=DEFAULT_EXPECTED_TEXT)
    parser.add_argument("--allow-mismatch", action="store_true")

    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--separate-realtime-model", action="store_true")
    parser.add_argument("--realtime-processing-pause", type=float, default=1.0)
    parser.add_argument("--post-speech-silence-duration", type=float, default=0.7)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_funasr_runtime()

    if args.init_only:
        run_init_only(args)
    elif args.file_smoke:
        run_file_smoke(args)
    else:
        run_microphone(args)


if __name__ == "__main__":
    main()
