EXTENDED_LOGGING = False

# Set to 0 to deactivate writing to keyboard.
# Try lower values like 0.002 first, use higher values like 0.05 in case it fails.
WRITE_TO_KEYBOARD_INTERVAL = 0.002

DEFAULT_KROKO_FINAL_MODEL = "Kroko-EN-Community-128-L-Streaming-001.data"
DEFAULT_KROKO_REALTIME_MODEL = "Kroko-EN-Community-64-L-Streaming-001.data"
KROKO_KEY_ENV_NAMES = (
    "REALTIMESTT_KROKO_ONNX_KEY",
    "KROKO_ONNX_KEY",
    "KROKO_KEY",
)
KROKO_REFERRAL_ENV_NAMES = (
    "REALTIMESTT_KROKO_ONNX_REFERRALCODE",
    "KROKO_ONNX_REFERRALCODE",
    "KROKO_REFERRALCODE",
)


def first_env_value(names):
    import os

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def default_model_path(filename):
    import os

    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "test-model-cache",
            "kroko-onnx",
            filename,
        )
    )


def install_repo_on_path():
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def check_kroko_runtime():
    try:
        import kroko_onnx  # noqa: F401
    except ImportError:
        print(
            "This test requires kroko_onnx in the active Python environment.\n"
            "Install it with:\n\n"
            '  python -m pip install "RealtimeSTT[kroko-builder]"\n'
            "  stt-install-kroko --build\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    import logging
    import os
    import sys
    import threading

    default_threads = max(1, min(4, os.cpu_count() or 2))
    default_model = default_model_path(DEFAULT_KROKO_FINAL_MODEL)
    default_realtime_model = default_model_path(DEFAULT_KROKO_REALTIME_MODEL)

    parser = argparse.ArgumentParser(
        description="Start the realtime Speech-to-Text test using Kroko-ONNX."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=default_model,
        help=(
            "Path to the final-transcript Kroko .data model. Default is "
            "test-model-cache/kroko-onnx/%s."
        )
        % DEFAULT_KROKO_FINAL_MODEL,
    )
    parser.add_argument(
        "--realtime-model",
        type=str,
        default=default_realtime_model,
        help=(
            "Path to the realtime-preview Kroko .data model. Default is "
            "test-model-cache/kroko-onnx/%s."
        )
        % DEFAULT_KROKO_REALTIME_MODEL,
    )
    parser.add_argument(
        "-l",
        "--lang",
        "--language",
        type=str,
        default="en",
        help="Language code for transcription. Default is en.",
    )
    parser.add_argument(
        "--provider",
        default="cpu",
        help="Kroko ONNX Runtime provider. Default is cpu.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="RealtimeSTT device hint. Defaults to cuda for cuda providers, otherwise cpu.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=default_threads,
        help="Kroko thread count. Default is %s." % default_threads,
    )
    parser.add_argument(
        "--tail-padding",
        type=float,
        default=None,
        help=(
            "Seconds of synthetic trailing silence added before Kroko decoding. "
            "Default is auto based on the Kroko streaming chunk size."
        ),
    )
    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="Kroko blank penalty. Default is 0.0.",
    )
    parser.add_argument(
        "--decoding-method",
        default="greedy_search",
        choices=("greedy_search", "modified_beam_search"),
        help="Kroko decoding method. Default is greedy_search.",
    )
    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="Kroko max_active_paths for modified_beam_search. Default is 4.",
    )
    parser.add_argument(
        "--key",
        default=first_env_value(KROKO_KEY_ENV_NAMES),
        help=(
            "License key for Pro Kroko models. Defaults from "
            "REALTIMESTT_KROKO_ONNX_KEY/KROKO_ONNX_KEY/KROKO_KEY. "
            "Do not commit keys."
        ),
    )
    parser.add_argument(
        "--referralcode",
        default=first_env_value(KROKO_REFERRAL_ENV_NAMES),
        help=(
            "Optional Kroko referral code. Defaults from "
            "REALTIMESTT_KROKO_ONNX_REFERRALCODE/KROKO_ONNX_REFERRALCODE/"
            "KROKO_REFERRALCODE."
        ),
    )
    parser.add_argument(
        "--rt-pause",
        type=float,
        default=1.0,
        help="Seconds between realtime transcription attempts. Default is 1.0.",
    )
    parser.add_argument(
        "--no-syllable-boundaries",
        action="store_true",
        help=(
            "Use timer-based realtime checks instead of syllable-boundary "
            "scheduling. This is usually snappier for Kroko streaming."
        ),
    )
    parser.add_argument(
        "--separate-realtime-model",
        action="store_true",
        help=(
            "Force a second Kroko recognizer for realtime updates even when "
            "--model and --realtime-model are the same path."
        ),
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        default=None,
        help="Optional PyAudio input device index.",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Do not type finalized transcriptions into the active window.",
    )
    parser.add_argument(
        "--show-kroko-output",
        action="store_true",
        help="Allow native Kroko stdout/stderr messages such as license status updates.",
    )
    parser.add_argument(
        "--keyboard-interval",
        type=float,
        default=WRITE_TO_KEYBOARD_INTERVAL,
        help="Interval used by pyautogui.write. Set to 0 to disable keyboard output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable maximum RealtimeSTT test diagnostics.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize the recorder, print success, then shut down without opening the microphone loop.",
    )
    args = parser.parse_args()
    install_repo_on_path()

    keyboard_interval = 0 if args.no_keyboard else args.keyboard_interval

    from install_packages import check_and_install_packages

    packages = [{"import_name": "rich"}]
    if keyboard_interval:
        packages.append({"import_name": "pyautogui"})
    check_and_install_packages(packages)
    check_kroko_runtime()

    if EXTENDED_LOGGING or args.debug:
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.text import Text

    from RealtimeSTT import AudioToTextRecorder
    import RealtimeSTT.audio_recorder as audio_recorder_module

    if args.debug:
        print("Using RealtimeSTT audio_recorder from %s" % audio_recorder_module.__file__)

    if keyboard_interval:
        import pyautogui
    else:
        pyautogui = None

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path

        _init_dll_path()

    use_main_model_for_realtime = (
        os.path.abspath(args.model) == os.path.abspath(args.realtime_model)
        and not args.separate_realtime_model
    )

    console = Console()
    console.print("System initializing with Kroko-ONNX, please wait")
    console.print("Final model: %s" % args.model)
    console.print("Realtime model: %s" % args.realtime_model)
    console.print(
        "Realtime recognizer: %s"
        % ("shared with final" if use_main_model_for_realtime else "separate")
    )
    tail_padding_label = "auto" if args.tail_padding is None else "%.2fs" % args.tail_padding
    console.print(
        "Provider: %s, threads: %s, tail padding: %s"
        % (args.provider, args.threads, tail_padding_label)
    )
    console.print(
        "Realtime scheduler: %s"
        % ("timer" if args.no_syllable_boundaries else "syllable-boundary")
    )
    console.print(
        "Kroko native output: %s"
        % ("visible" if args.show_kroko_output else "quiet")
    )

    ui_lock = threading.RLock()

    def live_panel(renderable, title, border_style):
        # Leave the last terminal cell unused. Full-width borders can hard-wrap
        # on Windows terminals and make Rich Live's cursor math drift downward.
        panel_width = max(4, console.size.width - 1)
        return Panel(
            renderable,
            title=title,
            border_style=border_style,
            width=panel_width,
        )

    def update_live(renderable, title, border_style):
        with ui_lock:
            live.update(live_panel(renderable, title, border_style), refresh=True)

    def configure_realtimestt_live_logging(level):
        realtime_logger = logging.getLogger("realtimestt")
        for handler in list(realtime_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                realtime_logger.removeHandler(handler)
                handler.close()

        handler = RichHandler(
            console=console,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("RealTimeSTT: %(name)s - %(levelname)s - %(message)s")
        )
        realtime_logger.addHandler(handler)

    live = Live(console=console, refresh_per_second=10, screen=False, auto_refresh=False)
    live_started = False

    full_sentences = []
    displayed_text = ""
    prev_text = ""
    recorder = None

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    def preprocess_text(text):
        text = text.lstrip()

        if text.startswith("..."):
            text = text[3:]

        text = text.lstrip()

        if text:
            text = text[0].upper() + text[1:]

        return text

    def text_detected(text):
        global displayed_text, prev_text, recorder

        text = preprocess_text(text)

        with ui_lock:
            sentence_end_marks = [".", "!", "?", "\u3002"]
            if text.endswith("..."):
                recorder.post_speech_silence_duration = mid_sentence_detection_pause
            elif (
                text
                and text[-1] in sentence_end_marks
                and prev_text
                and prev_text[-1] in sentence_end_marks
            ):
                recorder.post_speech_silence_duration = end_of_sentence_detection_pause
            else:
                recorder.post_speech_silence_duration = unknown_sentence_detection_pause

            prev_text = text

            rich_text = Text()
            for i, sentence in enumerate(full_sentences):
                style = "yellow" if i % 2 == 0 else "cyan"
                rich_text += Text(sentence, style=style) + Text(" ")

            if text:
                rich_text += Text(text, style="bold yellow")

            new_displayed_text = rich_text.plain
            if new_displayed_text != displayed_text:
                displayed_text = new_displayed_text
                update_live(
                    rich_text,
                    title="[bold green]Live Transcription - Kroko[/bold green]",
                    border_style="bold green",
                )

    def process_text(text):
        global prev_text, recorder

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        if not text:
            return

        with ui_lock:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause
            full_sentences.append(text)
            prev_text = ""
            text_detected("")

        if keyboard_interval and pyautogui:
            pyautogui.write("%s " % text, interval=keyboard_interval)

    engine_options = {
        "provider": args.provider,
        "num_threads": args.threads,
        "blank_penalty": args.blank_penalty,
        "decoding_method": args.decoding_method,
        "max_active_paths": args.max_active_paths,
        "suppress_native_output": not args.show_kroko_output,
    }
    if args.tail_padding is not None:
        engine_options["tail_padding_seconds"] = args.tail_padding
    if args.key:
        engine_options["key"] = args.key
    if args.referralcode:
        engine_options["referralcode"] = args.referralcode

    device = args.device
    if device is None:
        device = "cuda" if args.provider.lower().startswith("cuda") else "cpu"

    recorder_config = {
        "spinner": False,
        "model": args.model,
        "transcription_engine": "kroko_onnx",
        "transcription_engine_options": engine_options,
        "realtime_model_type": args.realtime_model,
        "realtime_transcription_engine": "kroko_onnx",
        "realtime_transcription_engine_options": dict(engine_options),
        "use_main_model_for_realtime": use_main_model_for_realtime,
        "device": device,
        "language": args.lang,
        "input_device_index": args.input_device_index,
        "silero_sensitivity": 0.05,
        "webrtc_sensitivity": 3,
        "post_speech_silence_duration": unknown_sentence_detection_pause,
        "min_length_of_recording": 1.1,
        "min_gap_between_recordings": 0,
        "enable_realtime_transcription": True,
        "realtime_processing_pause": args.rt_pause,
        "on_realtime_transcription_update": text_detected,
        "silero_deactivity_detection": True,
        "early_transcription_on_silence": 0,
        "batch_size": 0,
        "realtime_batch_size": 0,
        "realtime_transcription_use_syllable_boundaries": not args.no_syllable_boundaries,
        "realtime_boundary_detector_sensitivity": 0.6,
        "realtime_boundary_followup_delays": (0.5,),
        "no_log_file": True,
        "faster_whisper_vad_filter": False,
        "normalize_audio": True,
    }

    if args.debug:
        recorder_config["level"] = logging.DEBUG
        recorder_config["debug_mode"] = True
        recorder_config["use_extended_logging"] = True
        recorder_config["no_log_file"] = False
    if args.init_only:
        recorder_config["use_microphone"] = False

    try:
        recorder = AudioToTextRecorder(**recorder_config)
        configure_realtimestt_live_logging(recorder_config.get("level", logging.WARNING))
        if args.init_only:
            console.print("[bold green]Kroko recorder initialized successfully.[/bold green]")
            raise SystemExit(0)

        live.start()
        live_started = True
        update_live(
            Text("Say something...", style="cyan bold"),
            title="[bold yellow]Waiting for Input - Kroko[/bold yellow]",
            border_style="bold yellow",
        )

        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
    finally:
        if live_started:
            live.stop()
        if recorder is not None:
            recorder.shutdown()
