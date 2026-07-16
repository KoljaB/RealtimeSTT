"""Standalone Omnilingual ASR smoke and microphone test."""

from __future__ import print_function

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_MODEL = "omniASR_CTC_1B_v2"
DEFAULT_AUDIO_FILE = "omnilingual-smoke-LJ001-0002.wav"
DEFAULT_AUDIO_URL = (
    "https://raw.githubusercontent.com/KoljaB/RealtimeSTT/release/v1.0.1/"
    "tests/unit/audio/LJ001-0002.wav"
)
DEFAULT_EXPECTED_TEXT = "in being"


def install_repo_on_path():
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "RealtimeSTT").is_dir():
        repo_root_text = str(repo_root)
        if repo_root_text not in sys.path:
            sys.path.insert(0, repo_root_text)


def normalize_text(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(text.split())


def check_supported_runtime(allow_unsupported=False):
    if sys.platform.startswith("linux"):
        return

    message = (
        "Meta Omnilingual ASR is supported by RealtimeSTT on Linux or WSL2 "
        "with Python 3.11.x. Native Windows installs intentionally skip the "
        "Omnilingual runtime because fairseq2n has no Windows wheel."
    )
    if allow_unsupported:
        print("WARNING: " + message)
        return
    raise SystemExit(message)


def check_runtime_imports():
    required_modules = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("fairseq2", "fairseq2"),
        ("fairseq2n", "fairseq2n"),
        ("omnilingual_asr", "omnilingual-asr"),
    ]
    failures = []

    for module_name, package_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # ImportError, OSError, CUDA shared-library errors.
            failures.append((module_name, package_name, exc))

    if not failures:
        return

    print("The Omnilingual runtime is not importable in this environment.\n")
    for module_name, package_name, exc in failures:
        print("- {0} ({1}): {2}: {3}".format(
            module_name,
            package_name,
            exc.__class__.__name__,
            exc,
        ))
    print(
        "\nInstall and run from Linux or WSL2 with Python 3.11.x, for example:\n\n"
        '  python -m pip install "RealtimeSTT[omnilingual]"\n'
    )
    raise SystemExit(1)


def apply_cache_home(cache_home):
    if not cache_home:
        return

    cache_home = Path(cache_home).expanduser().resolve()
    cache_home.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_home / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    print("Using isolated HOME/XDG cache under: {0}".format(cache_home))


def resolve_compute_type(device, compute_type):
    if compute_type != "auto":
        return compute_type
    return "float16" if str(device).startswith("cuda") else "float32"


def build_engine_options(args):
    return {
        "batch_size": args.batch_size,
        "sample_rate": args.sample_rate,
    }


def create_engine(args):
    install_repo_on_path()
    from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig
    from RealtimeSTT.transcription_engines.factory import create_transcription_engine

    config = TranscriptionEngineConfig(
        model=args.model,
        compute_type=resolve_compute_type(args.device, args.compute_type),
        gpu_device_index=args.gpu_device_index,
        device=args.device,
        batch_size=args.batch_size,
        normalize_audio=True,
        engine_options=build_engine_options(args),
    )
    return create_transcription_engine("omnilingual_asr", config)


def ensure_audio_file(args):
    audio_path = Path(args.audio).expanduser()
    if audio_path.exists():
        return audio_path

    if args.no_download_audio:
        raise SystemExit("Audio file does not exist: {0}".format(audio_path))

    print("Downloading smoke WAV:")
    print("  {0}".format(args.audio_url))
    print("  -> {0}".format(audio_path))
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(args.audio_url, str(audio_path))
    return audio_path


def run_init_only(args):
    engine = create_engine(args)
    backend = getattr(engine, "backend", None)
    model_card = getattr(backend, "model_card", args.model)
    sample_rate = getattr(backend, "sample_rate", args.sample_rate)

    print("Omnilingual engine wrapper initialized successfully.")
    print("Model card: {0}".format(model_card))
    print("Device: {0}".format(args.device))
    print("Compute type: {0}".format(resolve_compute_type(args.device, args.compute_type)))
    print("Sample rate: {0}".format(sample_rate))


def run_file_smoke(args):
    audio_path = ensure_audio_file(args)
    engine = create_engine(args)

    print("Running Omnilingual file smoke.")
    print("Audio: {0}".format(audio_path))
    print("Expected text fragment: {0}".format(args.expected_text))
    print("The first run may download several GiB of model/cache files.")

    result = engine.transcribe(str(audio_path), language=args.language)
    transcript = result.text or ""
    normalized = normalize_text(transcript)
    expected = normalize_text(args.expected_text)

    print("\nTranscript:")
    print(transcript)
    print("\nNormalized transcript:")
    print(normalized)

    if expected and expected not in normalized:
        message = (
            "Smoke failed: expected normalized transcript to contain "
            "'{0}'.".format(expected)
        )
        if args.allow_mismatch:
            print("WARNING: " + message)
            return
        raise SystemExit(message)

    print("Omnilingual file smoke passed.")


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
    compute_type = resolve_compute_type(args.device, args.compute_type)

    recorder_config = {
        "spinner": False,
        "model": args.model,
        "transcription_engine": "omnilingual_asr",
        "transcription_engine_options": engine_options,
        "realtime_model_type": realtime_model,
        "realtime_transcription_engine": "omnilingual_asr",
        "realtime_transcription_engine_options": dict(engine_options),
        "use_main_model_for_realtime": not args.separate_realtime_model,
        "device": args.device,
        "compute_type": compute_type,
        "gpu_device_index": args.gpu_device_index,
        "language": args.language,
        "input_device_index": args.input_device_index,
        "silero_sensitivity": 0.05,
        "webrtc_sensitivity": 3,
        "post_speech_silence_duration": args.post_speech_silence_duration,
        "min_length_of_recording": 1.1,
        "min_gap_between_recordings": 0,
        "enable_realtime_transcription": not args.no_realtime,
        "realtime_processing_pause": args.realtime_processing_pause,
        "on_realtime_transcription_update": on_realtime,
        "silero_deactivity_detection": True,
        "early_transcription_on_silence": 0,
        "batch_size": 0,
        "realtime_batch_size": 0,
        "no_log_file": True,
        "faster_whisper_vad_filter": False,
        "normalize_audio": True,
    }

    recorder = None
    try:
        print("Initializing Omnilingual microphone test.")
        print("Model: {0}".format(args.model))
        print("Realtime model: {0}".format(realtime_model))
        print("Device: {0}, compute type: {1}".format(args.device, compute_type))
        print("Realtime: {0}".format("off" if args.no_realtime else "on"))
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
        description=(
            "Standalone RealtimeSTT Omnilingual ASR smoke and microphone test. "
            "Run from Linux or WSL2 with Python 3.11.x."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init-only",
        action="store_true",
        help="Check platform/imports and initialize the Omnilingual engine wrapper.",
    )
    mode.add_argument(
        "--file-smoke",
        action="store_true",
        help="Download/use a tiny public WAV and run a real model transcription.",
    )
    mode.add_argument(
        "--microphone",
        action="store_true",
        help="Start an interactive microphone transcription loop. This is the default.",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL, help="Omnilingual model card.")
    parser.add_argument(
        "--realtime-model",
        default=None,
        help="Realtime Omnilingual model card. Defaults to --model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to Omnilingual. Default is cuda; use cpu for CPU-only tests.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="auto, float16, float32, bfloat16, or another torch dtype string.",
    )
    parser.add_argument("--gpu-device-index", type=int, default=0)
    parser.add_argument("--language", default="en", help="Language code. CTC models ignore it.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--cache-home",
        default=None,
        help="Optional isolated HOME/XDG cache directory for model downloads.",
    )
    parser.add_argument(
        "--allow-unsupported-platform",
        action="store_true",
        help="Warn instead of exiting on non-Linux platforms.",
    )

    parser.add_argument("--audio", default=DEFAULT_AUDIO_FILE)
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
    parser.add_argument(
        "--expected-text",
        default=DEFAULT_EXPECTED_TEXT,
        help="Normalized text fragment expected during --file-smoke.",
    )
    parser.add_argument(
        "--no-download-audio",
        action="store_true",
        help="Do not download the default smoke WAV if --audio is missing.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Print a warning instead of failing when --file-smoke text differs.",
    )

    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--separate-realtime-model", action="store_true")
    parser.add_argument("--realtime-processing-pause", type=float, default=1.0)
    parser.add_argument("--post-speech-silence-duration", type=float, default=0.7)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    apply_cache_home(args.cache_home)
    check_supported_runtime(args.allow_unsupported_platform)
    check_runtime_imports()

    if args.init_only:
        run_init_only(args)
    elif args.file_smoke:
        run_file_smoke(args)
    else:
        run_microphone(args)


if __name__ == "__main__":
    main()
